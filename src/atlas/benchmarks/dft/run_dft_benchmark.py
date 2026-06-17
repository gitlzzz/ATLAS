"""CLI entry point for DFT parameter benchmarking.

Usage::

    atl_dft_benchmark -c benchmark_config.toml
"""

from __future__ import annotations

import argparse
import tomllib


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark VASP DFT parameters (kspacing, ENCUT, ISPIN, ...) '
        'to find optimal settings per crystallographic phase.',
    )
    parser.add_argument(
        '--config',
        '-c',
        required=True,
        help='Path to the benchmark TOML configuration file.',
        metavar='FILE',
    )
    return parser.parse_args()


def load_config(config_file: str) -> dict:
    with open(config_file, 'rb') as f:
        return tomllib.load(f)


def main() -> int:
    args = parse_arguments()

    from pathlib import Path

    from ase.io import read as ase_read

    from atlas.core import code_utils as atl_cut
    from atlas.core.command_line.command_line_utils import validate_config_file

    config = load_config(args.config)
    log_path = config.get('general', {}).get('log_path', '/tmp/')
    logger, log_file_path = atl_cut.init_logger(
        source='atl_dft_benchmark',
        log_path=log_path,
    )

    validate_config_file(config_dict=config, config_type='dft_benchmark')

    # --- Load database and select representatives ---
    db_path = config.get('database', {}).get('db_path')
    if not db_path:
        atl_cut.custom_print(
            'No database path specified in [database].db_path', 'error'
        )
        return 1

    atl_cut.custom_print(f'Loading database from {db_path}...', 'info')
    database = ase_read(db_path, format='extxyz', index=':')
    atl_cut.custom_print(f'Loaded {len(database)} structures.', 'info')

    from atlas.benchmarks.dft.dft_benchmark_core import (
        analyze_convergence,
        build_benchmark_calculations,
        generate_toml_snippet,
        monitor_and_collect,
        select_representative_structures,
        submit_benchmark_calculations,
    )
    from atlas.benchmarks.dft.dft_benchmark_report import generate_full_report

    representatives = select_representative_structures(database)
    atl_cut.custom_print(
        f'Selected {len(representatives)} phases: {", ".join(sorted(representatives))}',
        'info',
    )
    for phase, atoms in sorted(representatives.items()):
        atl_cut.custom_print(
            f'  {phase}: {atoms.get_chemical_formula()} ({len(atoms)} atoms)',
            'debug',
        )

    # --- Build and submit ---
    benchmark_config = config.get('benchmark', {})
    base_incar = config.get('incar', {}).get('bulk', {})
    base_kspacing = config.get('kpoints', {}).get('kspacing', {})
    dry_run = config.get('general', {}).get('dry_run', False)

    descriptors = build_benchmark_calculations(
        representatives,
        benchmark_config,
        base_incar,
        base_kspacing,
    )
    atl_cut.custom_print(f'Built {len(descriptors)} benchmark calculations.', 'info')

    if dry_run:
        atl_cut.custom_print('Dry run — listing calculations:', 'info')
        for d in descriptors:
            ref_tag = ' [REF]' if d['is_reference'] else ''
            atl_cut.custom_print(
                f'  {d["calc_label"]}: {d["param_name"]}={d["param_value"]}{ref_tag}',
                'debug',
            )
        atl_cut.custom_print(
            f'Would submit {len(descriptors)} calculations. Exiting (dry run).',
            'info',
        )
        return 0

    # Load AiiDA
    from aiida import load_profile

    load_profile()

    submitted = submit_benchmark_calculations(descriptors, config, dry_run=False)

    # --- Monitor and collect ---
    check_interval = config.get('general', {}).get('queue_check_interval_seconds', 240)
    submitted = monitor_and_collect(submitted, check_interval=check_interval)

    # --- Analyse and report ---
    convergence = analyze_convergence(submitted, benchmark_config)
    threshold = benchmark_config.get('threshold_meV', 1.0)
    toml_snippet = generate_toml_snippet(convergence)
    output_dir = Path(benchmark_config.get('output_dir', './dft_benchmark_results'))

    generate_full_report(convergence, threshold, output_dir, toml_snippet)
    atl_cut.custom_print(f'Benchmark complete. Results in {output_dir}', 'info')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
