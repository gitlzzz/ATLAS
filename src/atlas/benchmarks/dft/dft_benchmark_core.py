"""Core logic for DFT parameter benchmarking.

Selects representative structures, submits VASP calculations varying one
parameter at a time, collects results, and analyses convergence to find
the cheapest settings within an energy threshold.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import TYPE_CHECKING, Any

from ase import Atoms

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Parameters where smaller value = tighter/more expensive
_DIRECTION_MIN = {'kspacing', 'ediff', 'sigma'}
# Parameters where larger value = tighter/more expensive
_DIRECTION_MAX = {'encut', 'ecut'}


def select_representative_structures(
    database: list[Atoms],
) -> dict[str, Atoms]:
    """Pick one representative bulk structure per phase.

    Prefers the ``base=True`` structure for each phase.  Falls back to
    the smallest bulk structure when no base is tagged.
    """
    by_phase: dict[str, list[Atoms]] = {}
    for atoms in database:
        stype = atoms.info.get('atl_struct_type', '')
        if stype != 'bulk':
            continue
        phase = str(atoms.info.get('phase', 'unknown'))
        by_phase.setdefault(phase, []).append(atoms)

    representatives: dict[str, Atoms] = {}
    for phase, structs in by_phase.items():
        base = [s for s in structs if s.info.get('base')]
        if base:
            representatives[phase] = base[0]
        else:
            representatives[phase] = min(structs, key=len)
    return representatives


def determine_reference_value(
    param_name: str,
    values: list,
    direction: str | None = None,
    explicit_reference: float | int | None = None,
) -> float | int:
    """Return the tightest (most expensive) value from a sweep list.

    Heuristics by parameter name:
    - ``kspacing``, ``ediff``, ``sigma``: smallest is tightest
    - ``encut``, ``ispin``: largest is tightest
    - Otherwise uses *direction* (``"min"`` or ``"max"``), defaulting to ``"max"``
    """
    if explicit_reference is not None:
        return explicit_reference
    key = param_name.lower()
    if direction is not None:
        use_min = direction.lower() == 'min'
    elif key in _DIRECTION_MIN:
        use_min = True
    else:
        use_min = False
    return min(values) if use_min else max(values)


def _sort_values_cheap_first(
    param_name: str, values: list, direction: str | None = None
) -> list:
    """Sort values from cheapest to most expensive."""
    key = param_name.lower()
    if direction is not None:
        ascending = direction.lower() == 'min'
    elif key in _DIRECTION_MIN:
        ascending = False
    else:
        ascending = True
    return sorted(values, reverse=not ascending)


def build_benchmark_calculations(
    representative_structures: dict[str, Atoms],
    benchmark_config: dict,
    base_incar: dict,
    base_kspacing_dict: dict,
) -> list[dict]:
    """Build calculation descriptors for every benchmark sweep.

    Each descriptor contains everything needed to submit one VASP
    calculation via ``submit_aiida_vasp_calculation``.

    Returns a list of dicts with keys: ``phase``, ``param_name``,
    ``param_value``, ``structure``, ``incar``, ``kspacing_dict``,
    ``is_reference``, ``calc_label``.
    """
    threshold = benchmark_config.get('threshold_meV', 1.0)  # noqa: F841
    descriptors: list[dict] = []

    for param_name, param_cfg in benchmark_config.items():
        if param_name in ('threshold_meV', 'output_dir'):
            continue
        if not isinstance(param_cfg, dict) or 'values' not in param_cfg:
            continue

        values = param_cfg['values']
        direction = param_cfg.get('direction')
        ref_val = determine_reference_value(
            param_name,
            values,
            direction=direction,
            explicit_reference=param_cfg.get('reference_value'),
        )

        for phase, structure in representative_structures.items():
            for val in values:
                is_ref = val == ref_val

                if param_name.lower() == 'kspacing':
                    incar = copy.deepcopy(base_incar)
                    ksp = copy.deepcopy(base_kspacing_dict)
                    ksp[phase] = val
                else:
                    incar = copy.deepcopy(base_incar)
                    incar[param_name.lower()] = val
                    ksp = copy.deepcopy(base_kspacing_dict)

                label = f'bench_{param_name}_{val}_{phase}'
                descriptors.append(
                    {
                        'phase': phase,
                        'param_name': param_name,
                        'param_value': val,
                        'structure': structure,
                        'incar': incar,
                        'kspacing_dict': ksp,
                        'is_reference': is_ref,
                        'calc_label': label,
                    }
                )
    return descriptors


def submit_benchmark_calculations(
    calc_descriptors: list[dict],
    config_dict: dict,
    dry_run: bool = False,
) -> list[dict]:
    """Submit all benchmark calculations via AiiDA.

    Each descriptor gets submitted using the existing
    ``submit_aiida_vasp_calculation`` function.  Benchmark metadata is
    stored in AiiDA node extras for later retrieval.

    Returns the descriptors list with an added ``node`` key (the AiiDA
    node, or ``None`` for dry runs).
    """
    from aiida.orm import Group

    from atlas.core import code_utils as atl_cut
    from atlas.workflows.aiida_utils import CalcType, submit_aiida_vasp_calculation

    potential_family = config_dict.get('calculation', {}).get(
        'aiida_potential_family', ''
    )
    potential_mapping = config_dict.get('calculation', {}).get('potential_mapping', {})
    queue_dict = config_dict.get('queue', {})
    aiida_vasp_settings = config_dict.get('aiida_vasp', {})
    max_batch = config_dict.get('general', {}).get('max_batch', 50)
    queue_interval = config_dict.get('general', {}).get(
        'queue_check_interval_seconds', 240
    )
    group_name = config_dict.get('general', {}).get('aiida_group_name', 'dft_benchmark')

    if dry_run:
        atl_cut.custom_print(
            'Dry run: calculations will not be submitted.',
            'warning',
        )
        group = None
    else:
        ctime = time.strftime('%Y%m%dT%H%M%S')
        group_label = f'{group_name}_{ctime}'
        group = Group(label=group_label)
        group.store()
        atl_cut.custom_print(f'Benchmark group: {group_label}', 'info')

    submitted: list[dict] = []
    pending_nodes: list = []

    for i, desc in enumerate(calc_descriptors):
        # Wait if we've hit the batch limit
        if not dry_run and len(pending_nodes) >= max_batch:
            atl_cut.custom_print(
                f'Batch limit ({max_batch}) reached, waiting for slots...',
                'info',
            )
            pending_nodes = _wait_for_slots(pending_nodes, max_batch, queue_interval)

        node = submit_aiida_vasp_calculation(
            index=i,
            target_structure=desc['structure'],
            phase=desc['phase'],
            material_name='benchmark',
            unique_id=desc['calc_label'],
            kspacing_dict=desc['kspacing_dict'],
            incar_settings_dict=desc['incar'],
            calc_type=CalcType.SP_BULK,
            queue_dict=queue_dict,
            potential_family=potential_family,
            potential_mapping=potential_mapping,
            dry_run=dry_run,
            return_builder=dry_run,
            group=group,
            aiida_vasp_settings=aiida_vasp_settings,
        )

        if not dry_run:
            node.base.extras.set('atl_benchmark_param', desc['param_name'])
            node.base.extras.set('atl_benchmark_value', desc['param_value'])
            node.base.extras.set('atl_benchmark_phase', desc['phase'])
            node.base.extras.set('atl_benchmark_is_ref', desc['is_reference'])
            pending_nodes.append(node)

        desc['node'] = node if not dry_run else None
        submitted.append(desc)

    return submitted


def _wait_for_slots(nodes: list, max_batch: int, interval: int) -> list:
    active = [n for n in nodes if not n.is_finished]
    while len(active) >= max_batch:
        time.sleep(interval)
        active = [n for n in active if not n.is_finished]
    return active


def monitor_and_collect(
    submitted: list[dict],
    check_interval: int = 240,
) -> list[dict]:
    """Wait for all benchmark calculations to finish and collect energies.

    Returns the submitted list with added ``energy_per_atom`` and
    ``n_atoms`` keys.
    """
    from atlas.active_learning.conversion import gather_calc_data_from_node
    from atlas.core import code_utils as atl_cut

    nodes = [d for d in submitted if d.get('node') is not None]
    if not nodes:
        return submitted

    total = len(nodes)
    atl_cut.custom_print(
        f'Monitoring {total} benchmark calculations...',
        'info',
    )

    remaining = list(nodes)
    while remaining:
        still_running = []
        for desc in remaining:
            node = desc['node']
            if node.is_finished:
                try:
                    last_calc = _get_last_calcjob(node)
                    data = gather_calc_data_from_node(last_calc, units='mace')
                    energy = data['energy']
                    n_atoms = len(data['symbols'])
                    desc['energy_per_atom'] = energy / n_atoms
                    desc['n_atoms'] = n_atoms
                    atl_cut.custom_print(
                        f'  {desc["calc_label"]}: '
                        f'E/atom = {desc["energy_per_atom"]:.6f}'
                        f'  eV (exit={node.exit_status})',
                        'debug',
                    )
                except Exception as exc:
                    atl_cut.custom_print(
                        f'  {desc["calc_label"]}: FAILED — {exc}',
                        'warning',
                    )
                    desc['energy_per_atom'] = None
                    desc['n_atoms'] = None
            else:
                still_running.append(desc)

        done = total - len(still_running)
        if still_running:
            atl_cut.custom_print(
                f'{done}/{total} done, waiting {check_interval}s...',
                'info',
            )
            time.sleep(check_interval)
        remaining = still_running

    atl_cut.custom_print(f'All {total} benchmark calculations finished.', 'info')
    return submitted


def _get_last_calcjob(workchain_node):
    """Get the last CalcJobNode descendant of a workchain."""
    from aiida.orm import CalcJobNode

    descendants = workchain_node.called_descendants
    calcjobs = [d for d in descendants if isinstance(d, CalcJobNode)]
    if not calcjobs:
        raise ValueError(f'No CalcJobNode found for workchain {workchain_node.pk}')
    return sorted(calcjobs, key=lambda n: n.ctime)[-1]


def analyze_convergence(
    submitted: list[dict],
    benchmark_config: dict,
) -> dict[str, dict[str, Any]]:
    """Analyse convergence for each benchmark parameter and phase.

    Returns a nested dict::

        {
            'kspacing': {
                'alpha': {
                    'reference_value': 0.05,
                    'reference_energy': -5.123,
                    'converged_value': 0.125,
                    'values': [0.05, 0.075, ...],
                    'energies': [-5.123, -5.123, ...],
                    'diffs_meV': [0.0, 0.3, ...],
                },
                ...
            },
            ...
        }
    """
    threshold = benchmark_config.get('threshold_meV', 1.0)
    results: dict[str, dict[str, Any]] = {}

    param_names = set()
    for desc in submitted:
        param_names.add(desc['param_name'])

    for param_name in sorted(param_names):
        param_cfg = benchmark_config.get(param_name, {})
        direction = param_cfg.get('direction')
        values_list = param_cfg.get('values', [])
        ref_val = determine_reference_value(
            param_name,
            values_list,
            direction=direction,
            explicit_reference=param_cfg.get('reference_value'),
        )

        phases: dict[str, dict] = {}
        for desc in submitted:
            if desc['param_name'] != param_name:
                continue
            phase = desc['phase']
            phases.setdefault(phase, {})
            if desc['energy_per_atom'] is not None:
                phases[phase][desc['param_value']] = desc['energy_per_atom']

        results[param_name] = {}
        for phase, val_energies in phases.items():
            ref_energy = val_energies.get(ref_val)
            if ref_energy is None:
                continue

            sorted_vals = sorted(values_list)
            energies = []
            diffs = []
            for v in sorted_vals:
                e = val_energies.get(v)
                if e is not None:
                    energies.append(e)
                    diffs.append(abs(e - ref_energy) * 1000)
                else:
                    energies.append(None)
                    diffs.append(None)

            # Find cheapest converged value
            cheap_order = _sort_values_cheap_first(param_name, sorted_vals, direction)
            converged = ref_val
            for v in cheap_order:
                idx = sorted_vals.index(v)
                if diffs[idx] is not None and diffs[idx] <= threshold:
                    converged = v
                    break

            results[param_name][phase] = {
                'reference_value': ref_val,
                'reference_energy': ref_energy,
                'converged_value': converged,
                'values': sorted_vals,
                'energies': energies,
                'diffs_meV': diffs,
                'threshold_meV': threshold,
            }

    return results


def generate_toml_snippet(convergence_results: dict) -> str:
    """Generate a TOML snippet with recommended settings."""
    try:
        import tomli_w
    except ImportError:
        tomli_w = None

    rec: dict[str, Any] = {}

    for param_name, phases in convergence_results.items():
        if param_name.lower() == 'kspacing':
            ksp = {}
            for phase, data in phases.items():
                ksp[phase] = data['converged_value']
            rec.setdefault('kpoints', {})['kspacing'] = ksp
        else:
            # For INCAR params, pick the most conservative across phases
            vals = [d['converged_value'] for d in phases.values()]
            chosen = min(vals) if param_name.lower() in _DIRECTION_MIN else max(vals)
            rec.setdefault('incar', {}).setdefault('bulk', {})[param_name.lower()] = (
                chosen
            )

    if tomli_w is not None:
        return tomli_w.dumps(rec)
    return _fallback_toml_format(rec)


def _fallback_toml_format(d: dict, prefix: str = '') -> str:
    """Minimal TOML formatter when tomli_w is unavailable."""
    lines: list[str] = []
    for key, val in d.items():
        if isinstance(val, dict):
            section = f'{prefix}.{key}' if prefix else key
            lines.append(f'[{section}]')
            lines.append(_fallback_toml_format(val, section))
        else:
            lines.append(f'{key} = {_toml_val(val)}')
    return '\n'.join(lines)


def _toml_val(v: Any) -> str:
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)
