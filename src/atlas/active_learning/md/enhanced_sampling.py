"""CV-free enhanced sampling for the active-learning MD step.

Selectable methods that explore configuration space more aggressively than plain
MD, to generate more diverse / off-equilibrium training structures:

- ``gamd`` : Gaussian Accelerated MD. Harmonic boost ``dV = 1/2 k (E - V)^2``
  applied when the potential energy ``V`` is below a threshold ``E``; forces are
  scaled by ``(1 - k (E - V))``. ``E`` and ``k`` can be auto-tuned from a short
  equilibration phase (Miao et al., 2015).
- ``amd``  : Accelerated MD. Boost ``dV = (E - V)^2 / (alpha + (E - V))``
  (Hamelberg et al., 2004).
- ``minima_hopping`` : ASE ``MinimaHopping`` driver (MD + relaxation hops).
- ``replica_exchange`` : in-process parallel tempering across a temperature
  ladder with Metropolis configuration swaps.

GaMD/aMD are implemented as a :class:`BoostCalculator` that wraps any ASE
calculator, so they slot in alongside ``ATLSafeCalculatorWrapper``. Minima
hopping and replica exchange are standalone trajectory drivers that return a
list of ASE ``Atoms`` frames (each carrying ``REF_energy``/``REF_forces``).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

ENHANCED_SAMPLING_METHODS = (
    'none',
    'gamd',
    'amd',
    'minima_hopping',
    'replica_exchange',
)


class BoostCalculator(Calculator):
    """Wrap an ASE calculator with a GaMD or aMD boost potential.

    The boost only acts on basins (``V < E``); barriers (``V >= E``) are left
    untouched, so the boosted force factor is always ``>= 0``.
    """

    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(
        self,
        calculator,
        method: str = 'gamd',
        boost_energy: float | None = None,
        boost_k: float | None = None,
        amd_alpha: float | None = None,
        sigma0: float = 6.0,
        equilibration_steps: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if method not in ('gamd', 'amd'):
            raise ValueError(f"BoostCalculator method must be 'gamd' or 'amd', got {method}")
        self.calculator = calculator
        self.method = method
        self.boost_energy = boost_energy
        self.boost_k = boost_k
        self.amd_alpha = amd_alpha
        self.sigma0 = sigma0
        self.equilibration_steps = equilibration_steps
        self.last_boost = 0.0

        # Running statistics for auto-tuning.
        self._n_calls = 0
        self._v_min = np.inf
        self._v_max = -np.inf
        self._v_sum = 0.0
        self._v_sumsq = 0.0
        # Parameters are ready if supplied manually.
        if method == 'gamd':
            self._params_ready = boost_energy is not None and boost_k is not None
        else:
            self._params_ready = boost_energy is not None and amd_alpha is not None

    def _update_stats(self, v: float) -> None:
        self._n_calls += 1
        self._v_min = min(self._v_min, v)
        self._v_max = max(self._v_max, v)
        self._v_sum += v
        self._v_sumsq += v * v

    def _finalize_auto_params(self) -> None:
        n = self._n_calls
        v_avg = self._v_sum / n
        v_std = np.sqrt(max(self._v_sumsq / n - v_avg * v_avg, 0.0))
        vmax, vmin = self._v_max, self._v_min
        spread = vmax - vmin
        self.boost_energy = vmax
        if self.method == 'gamd':
            if spread <= 0:
                self.boost_k = 0.0
            else:
                denom = max(vmax - v_avg, 1e-12)
                k0 = min(1.0, (self.sigma0 / (v_std + 1e-12)) * (spread / denom))
                self.boost_k = k0 / spread
        elif self.amd_alpha is None:
            self.amd_alpha = (spread / 5.0) if spread > 0 else 1.0
        self._params_ready = True

    def _boost(self, v: float) -> tuple[float, float]:
        """Return (delta_V, force_scale) for potential energy ``v``."""
        e = self.boost_energy
        if v >= e:
            return 0.0, 1.0
        u = e - v  # > 0
        if self.method == 'gamd':
            k = self.boost_k
            delta_v = 0.5 * k * u * u
            scale = max(1.0 - k * u, 0.0)
        else:  # amd
            alpha = self.amd_alpha
            delta_v = (u * u) / (alpha + u)
            scale = max(1.0 - u * (2.0 * alpha + u) / (alpha + u) ** 2, 0.0)
        return delta_v, scale

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        probe = self.atoms.copy()
        probe.calc = self.calculator
        v = probe.get_potential_energy()
        forces = probe.get_forces()

        # Equilibration phase: collect statistics, apply no boost.
        if not self._params_ready:
            self._update_stats(v)
            if self._n_calls >= self.equilibration_steps:
                self._finalize_auto_params()
            self.last_boost = 0.0
            self.results = {'energy': v, 'free_energy': v, 'forces': forces}
            return

        delta_v, scale = self._boost(v)
        self.last_boost = delta_v
        boosted_e = v + delta_v
        self.results = {
            'energy': boosted_e,
            'free_energy': boosted_e,
            'forces': scale * forces,
        }


def make_enhanced_calculator(base_calc, settings: dict | None):
    """Return ``base_calc`` wrapped for GaMD/aMD, or unchanged otherwise."""
    settings = settings or {}
    method = settings.get('method', 'none')
    if method not in ('gamd', 'amd'):
        return base_calc
    return BoostCalculator(
        calculator=base_calc,
        method=method,
        boost_energy=settings.get('boost_energy'),
        boost_k=settings.get('boost_k'),
        amd_alpha=settings.get('amd_alpha'),
        sigma0=settings.get('gamd_sigma0', 6.0),
        equilibration_steps=settings.get('equilibration_steps', 0),
    )


def replica_temperature_ladder(
    t_start: float, n_replicas: int, max_multiplier: float = 2.0
) -> list[float]:
    """Geometric temperature ladder from ``t_start`` to ``t_start*max_multiplier``."""
    n_replicas = max(int(n_replicas), 1)
    if n_replicas == 1:
        return [float(t_start)]
    ratio = max_multiplier ** (1.0 / (n_replicas - 1))
    return [float(t_start * ratio**i) for i in range(n_replicas)]


def run_replica_exchange_md(
    atoms: Atoms,
    calculator_factory: Callable[[], Calculator],
    temperatures: list[float],
    n_steps: int,
    swap_interval: int = 100,
    timestep_fs: float = 1.0,
    friction_ps: float = 100.0,
    rng: np.random.Generator | None = None,
) -> list[Atoms]:
    """Parallel-tempering MD run in a single process.

    Runs one Langevin replica per temperature, harvesting a frame from each
    replica after every ``swap_interval`` steps and attempting Metropolis
    configuration swaps between adjacent temperatures.
    """
    rng = rng or np.random.default_rng()
    temps = [float(t) for t in temperatures]

    replicas: list[Atoms] = []
    dyns: list[Langevin] = []
    for temp in temps:
        rep = atoms.copy()
        rep.calc = calculator_factory()
        MaxwellBoltzmannDistribution(rep, temperature_K=temp)
        Stationary(rep)
        ZeroRotation(rep)
        dyns.append(
            Langevin(
                rep,
                timestep=timestep_fs * units.fs,
                temperature_K=temp,
                friction=(friction_ps / 1000) / units.fs,
            )
        )
        replicas.append(rep)

    frames: list[Atoms] = []
    n_blocks = max(1, int(n_steps) // max(int(swap_interval), 1))

    for _ in range(n_blocks):
        for dyn in dyns:
            dyn.run(swap_interval)

        for rep in replicas:
            frame = rep.copy()
            frame.info['REF_energy'] = rep.get_potential_energy()
            frame.arrays['REF_forces'] = rep.get_forces()
            frames.append(frame)

        # Attempt swaps between adjacent temperature replicas.
        for i in range(len(replicas) - 1):
            e_i = replicas[i].get_potential_energy()
            e_j = replicas[i + 1].get_potential_energy()
            beta_i = 1.0 / (units.kB * temps[i])
            beta_j = 1.0 / (units.kB * temps[i + 1])
            arg = (beta_i - beta_j) * (e_i - e_j)
            if arg >= 0 or rng.random() < np.exp(arg):
                pos_i = replicas[i].get_positions().copy()
                pos_j = replicas[i + 1].get_positions().copy()
                mom_i = replicas[i].get_momenta().copy()
                mom_j = replicas[i + 1].get_momenta().copy()
                replicas[i].set_positions(pos_j)
                replicas[i + 1].set_positions(pos_i)
                # Rescale momenta to each replica's own temperature.
                replicas[i].set_momenta(mom_j * np.sqrt(temps[i] / temps[i + 1]))
                replicas[i + 1].set_momenta(mom_i * np.sqrt(temps[i + 1] / temps[i]))

    return frames


def run_minima_hopping(
    atoms: Atoms,
    calculator: Calculator,
    total_steps: int = 10,
    temperature: float = 1000.0,
    ediff0: float = 0.5,
) -> list[Atoms]:
    """Explore distinct PES minima with ASE ``MinimaHopping``.

    Returns the minima discovered (read back from the ``minima.traj`` the
    optimizer writes). Runs in a temporary directory to avoid clobbering files.
    """
    import os

    from ase.io import read as ase_read
    from ase.optimize.minimahopping import MinimaHopping

    work = atoms.copy()
    work.calc = calculator

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            hop = MinimaHopping(work, T0=temperature, Ediff0=ediff0)
            hop(totalsteps=total_steps)
            minima = ase_read('minima.traj', index=':')
        finally:
            os.chdir(cwd)

    for frame in minima:
        try:
            frame.info.setdefault('REF_energy', frame.get_potential_energy())
        except Exception:
            pass
    return minima


def run_enhanced_sampling_frames(
    init_conf: Atoms,
    method: str,
    settings: dict,
    t_start: float,
    md_params: dict,
    calculator_factory: Callable[[], Calculator] | None = None,
) -> list[Atoms]:
    """Dispatch to the trajectory-replacing methods (replica exchange / minima hopping)."""
    if calculator_factory is None:
        base = init_conf.calc
        calculator_factory = lambda: base  # noqa: E731 - shared calc is fine here

    timestep_fs = md_params['timestep_duration_ps'] * 1000
    n_steps = md_params['num_steps']

    if method == 'replica_exchange':
        temps = settings.get('temperatures') or replica_temperature_ladder(
            t_start,
            settings.get('n_replicas', 4),
            settings.get('max_multiplier', md_params.get('max_temp_multiplier', 2.0)),
        )
        return run_replica_exchange_md(
            atoms=init_conf,
            calculator_factory=calculator_factory,
            temperatures=temps,
            n_steps=n_steps,
            swap_interval=settings.get('swap_interval', 100),
            timestep_fs=timestep_fs,
            friction_ps=md_params.get('langevin_friction_ps-1', 100.0),
        )
    if method == 'minima_hopping':
        return run_minima_hopping(
            atoms=init_conf,
            calculator=calculator_factory(),
            total_steps=settings.get('minima_hopping_steps', 10),
            temperature=t_start,
            ediff0=settings.get('minima_hopping_ediff0', 0.5),
        )
    raise ValueError(f'Unsupported trajectory-replacing method: {method}')
