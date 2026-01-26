#!/usr/bin/env python3
"""
Script to evaluate and compare the performance of MLIPs using MD simulations.

This script can take one or more MLIP models, trained via active learning or
otherwise, and evaluates their performance on a configurable benchmark system.
The benchmark system is a metal surface slab generated using ASE.

The script can load models from:
- AiiDA SimpleActiveLearningBaseWorkChain output (given a workchain pk/uuid).
- A user-specified path to a .model file.

The evaluation consists of running an MD simulation for each model and plotting
the energy evolution for comparison.
"""

import pathlib as pl
import sys
import time
import warnings

import MatDBForge.benchmarks.mlip.mlip_benchmark_utils as mdb_b_ut
import MatDBForge.benchmarks.mlip.mlip_benchmarks as mdb_benchmarks
from MatDBForge.core.code_utils import (
    custom_print,
    init_logger,
)
from MatDBForge.core.command_line.command_line_utils import validate_config_file

# Ignore all warnings
warnings.filterwarnings('ignore')


def main():
    """Main function to run the evaluation."""
    parsed_args = mdb_b_ut.parse_arguments()
    init_logger(source='mdb_benchmark_mlip')

    # Determine config path
    config_path = parsed_args.config_path
    if config_path is None:
        default_config = pl.Path('./mdb_benchmark_settings.toml')
        if default_config.exists():
            config_path = default_config
            custom_print(f'Found default config file: {default_config}', 'info')
        else:
            custom_print(
                'No configuration file specified and mdb_benchmark_settings.toml '
                'not found in current directory.',
                'error',
            )
            custom_print(
                'Please provide a configuration file or create '
                'mdb_benchmark_settings.toml',
                'error',
            )
            return

    # Check if config file exists
    if not config_path.exists():
        custom_print(f'Configuration file not found: {config_path}', 'error')
        return

    custom_print(f'Loading configuration from: {config_path}', 'info')

    # Validate the config file
    any_errors_found, errors, warnings = validate_config_file(
        config_path=config_path,
        config_type='mlip_benchmarks',
        run_mode='workflow',
    )

    # Load TOML config and create args
    toml_dict = mdb_b_ut.load_toml_config(config_path)
    args = mdb_b_ut.create_args_from_toml(toml_dict)

    if not any_errors_found:
        custom_print('Configuration loaded and validated successfully!', 'done')
    else:
        sys.exit(1)

    print()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    # Determine which benchmarks to run
    benchmarks_to_run = []
    benchmark_functions = {}

    if args.run_accuracy_test_set:
        benchmarks_to_run.append('Accuracy Test Set')
        benchmark_functions['Accuracy Test Set'] = (
            lambda: mdb_benchmarks.run_accuracy_test_set_benchmark(args, model_paths)
        )

    if args.run_elastic_properties:
        benchmarks_to_run.append('Elastic Properties')
        benchmark_functions['Elastic Properties'] = (
            lambda: mdb_benchmarks.run_elastic_properties_benchmark(args, model_paths)
        )

    if args.run_defect_formation_energy:
        benchmarks_to_run.append('Defect Formation Energy')
        benchmark_functions['Defect Formation Energy'] = (
            lambda: mdb_benchmarks.run_defect_formation_energy_benchmark(
                args, model_paths
            )
        )

    if args.run_surface_energies:
        benchmarks_to_run.append('Surface Energies')
        benchmark_functions['Surface Energies'] = (
            lambda: mdb_benchmarks.run_surface_energies_benchmark(args, model_paths)
        )

    if args.run_phonon_dispersion:
        benchmarks_to_run.append('Phonon Dispersion')
        benchmark_functions['Phonon Dispersion'] = (
            lambda: mdb_benchmarks.run_phonon_dispersion_benchmark(args, model_paths)
        )

    if args.run_energy_md:
        benchmarks_to_run.append('Energy MD')
        benchmark_functions['Energy MD'] = (
            lambda: mdb_benchmarks.run_energy_md_benchmark(args, model_paths)
        )

    if args.run_high_temp_md:
        benchmarks_to_run.append('High Temperature MD')
        benchmark_functions['High Temperature MD'] = (
            lambda: mdb_benchmarks.run_high_temp_md_benchmark(args, model_paths)
        )

    if args.run_melting_point:
        benchmarks_to_run.append('Melting Point')
        benchmark_functions['Melting Point'] = (
            lambda: mdb_benchmarks.run_melting_point_benchmark(args, model_paths)
        )

    if args.run_gsfe:
        benchmarks_to_run.append('GSFE')
        benchmark_functions['GSFE'] = lambda: mdb_benchmarks.run_gsfe_benchmark(
            args, model_paths
        )

    if args.run_learning_curves:
        benchmarks_to_run.append('Learning Curves')
        benchmark_functions['Learning Curves'] = (
            lambda: mdb_benchmarks.run_learning_curves_benchmark(args, model_paths)
        )

    if args.run_final_db_size:
        benchmarks_to_run.append('Final DB Size')
        benchmark_functions['Final DB Size'] = (
            lambda: mdb_benchmarks.run_final_db_size_benchmark(args, model_paths)
        )

    if args.run_md_count:
        benchmarks_to_run.append('MD Count')
        benchmark_functions['MD Count'] = lambda: mdb_benchmarks.run_md_count_benchmark(
            args, model_paths
        )

    if args.run_evaluate_database:
        benchmarks_to_run.append('Evaluate Database')
        benchmark_functions['Evaluate Database'] = (
            lambda: mdb_benchmarks.run_evaluate_database(args, model_paths)
        )

    if args.run_magic_cluster:
        benchmarks_to_run.append('Magic Cluster')
        benchmark_functions['Magic Cluster'] = (
            lambda: mdb_benchmarks.run_magic_cluster_benchmark(args, model_paths)
        )

    if not benchmarks_to_run:
        print('No benchmarks selected. Use --help to see available options.')
        return

    # Check if any models are specified
    if not args.model_files and not args.aiida_pks and not args.foundation_models:
        print(
            'No models specified. Please provide --model_files, --aiida_pks, '
            'or --foundation_models. Use --help for more information.'
        )
        return

    # Load models
    model_paths = [pl.Path(p) for p in args.model_files]

    # Add foundation models
    if args.foundation_models:
        mdb_b_ut.custom_print(
            f'Loading {len(args.foundation_models)} foundation model(s)...', 'info'
        )
        foundation_model_paths = mdb_b_ut.create_foundation_model_paths(
            args.foundation_models
        )
        for foundation_path in foundation_model_paths:
            mdb_b_ut.custom_print(f'  - {foundation_path.name}', 'info')
        model_paths.extend(foundation_model_paths)

    # For models loaded from files, use the filename stem as display name
    for model_path in model_paths:
        if hasattr(model_path, 'foundation_model_spec'):
            # Foundation model
            mdb_b_ut.set_model_display_name(str(model_path), model_path.name)
        else:
            # File-based model
            mdb_b_ut.set_model_display_name(str(model_path), model_path.stem)

    if not args.no_rich_ui:
        # Use Rich UI
        ui_manager = mdb_b_ut.RichUIManager(benchmarks_to_run)
        mdb_b_ut.set_ui_manager(ui_manager)

        with ui_manager:
            mdb_b_ut.custom_print('Initializing MLIP Benchmark Suite...', 'info')
            mdb_b_ut.custom_print(f'Output directory: {args.output_dir}', 'empty')
            mdb_b_ut.custom_print(
                f'Selected benchmarks: {", ".join(benchmarks_to_run)}', 'empty'
            )

            # Load models
            if args.aiida_pks:
                ui_manager.log('Loading models from MDB Workchain...')
                for pk in args.aiida_pks:
                    path = mdb_b_ut.load_model_from_aiida(pk, args.output_dir)
                    if path:
                        model_paths.append(path)

            if not model_paths:
                ui_manager.log('No models specified. Exiting.')
                return

            # Initialize model data with consistent colors
            mdb_b_ut.initialize_model_data(model_paths)
            mdb_b_ut.custom_print(f'Initialized {len(model_paths)} model(s).', 'info')

            # Run selected benchmarks
            for benchmark_name in benchmarks_to_run:
                try:
                    ui_manager.start_benchmark(benchmark_name)
                    benchmark_functions[benchmark_name]()
                    ui_manager.complete_benchmark(benchmark_name)
                except Exception as e:
                    ui_manager.log(f"Benchmark '{benchmark_name}' failed: {e}")
                    ui_manager.complete_benchmark(benchmark_name)

            ui_manager.current_benchmark = 'All Benchmarks Completed'
            mdb_b_ut.custom_print(' ', 'empty')
            mdb_b_ut.custom_print('All selected benchmarks finished!', 'done')

            # Create final multi-panel plot if any data was collected
            ui_manager.current_benchmark = 'Generating Summary Plot'
            mdb_b_ut.create_final_multi_panel_plot(args)

            time.sleep(2)
    else:
        # Use plain text output (original behavior)
        mdb_b_ut.custom_print('Initializing MLIP Benchmark Suite...', 'info')
        mdb_b_ut.custom_print(f'Output directory: {args.output_dir}', 'info')
        mdb_b_ut.custom_print(
            f'Selected benchmarks: {", ".join(benchmarks_to_run)}', 'info'
        )

        # Load models
        if args.aiida_pks:
            mdb_b_ut.custom_print('Loading models from MDB Workchain...', 'info')
            for pk in args.aiida_pks:
                path = mdb_b_ut.load_model_from_aiida(pk, args.output_dir)
                if path:
                    model_paths.append(path)

        if not model_paths:
            mdb_b_ut.custom_print('No models specified. Exiting.', 'error')
            return

        mdb_b_ut.custom_print(f'Loaded {len(model_paths)} model(s)', 'info')
        for path in model_paths:
            display_name = mdb_b_ut.get_model_display_name(path)
            mdb_b_ut.custom_print(f'  - {display_name}', 'info')

        # Initialize model data with consistent colors
        mdb_b_ut.initialize_model_data(model_paths)

        # Run selected benchmarks
        for benchmark_name in benchmarks_to_run:
            try:
                mdb_b_ut.custom_print(f'Starting benchmark: {benchmark_name}', 'info')
                benchmark_functions[benchmark_name]()
                mdb_b_ut.custom_print(f'Completed benchmark: {benchmark_name}', 'done')
            except Exception as e:
                raise e
                mdb_b_ut.custom_print(
                    f"Benchmark '{benchmark_name}' failed: {e}", 'error'
                )

        print()
        mdb_b_ut.custom_print('All selected benchmarks finished!', 'done')

        # Create final multi-panel plot if any data was collected
        mdb_b_ut.create_final_multi_panel_plot(args)

    # Cleanup log file
    mdb_b_ut._cleanup_log_file()


if __name__ == '__main__':
    main()
