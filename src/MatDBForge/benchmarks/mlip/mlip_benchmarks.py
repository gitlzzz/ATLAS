"""Collection of MDB-trained benchmarks for MLIPs."""

import copy
import json
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.build import bulk, surface
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.trajectory import TrajectoryWriter
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import LBFGS

import MatDBForge.active_learning.active_learning_utils as mdb_al_ut
import MatDBForge.benchmarks.mlip.mlip_benchmark_utils as mdb_b_ut
from MatDBForge.active_learning.report_utils import (
    get_loop_report,
)


def run_final_db_size_benchmark(args, model_paths: list[pl.Path]):
    """Compares the final training database size for each AL run."""
    mdb_b_ut.custom_print('Running Final Database Size Benchmark', 'info')
    benchmark_dir = args.output_dir / 'final_db_size'
    benchmark_dir.mkdir(exist_ok=True)

    # This benchmark only works with AiiDA PKs
    if not args.aiida_pks:
        mdb_b_ut.custom_print(
            'No AiiDA PKs provided. This benchmark requires AiiDA workchain PKs.',
            'warn',
        )
        return

    results = {}

    # Process each AiiDA workchain
    for pk in args.aiida_pks:
        try:
            from aiida import load_profile
            from aiida.orm import load_node

            load_profile()

            # Load the base workchain
            base_workchain = load_node(pk)

            # Get run name for display
            run_name = None
            try:
                run_name = base_workchain.inputs.active_learning.run_name.value
            except (AttributeError, KeyError):
                run_name = f'workchain_{pk}'

            mdb_b_ut.custom_print(f'Processing workchain {pk} ({run_name})...', 'info')

            # Get all children workchains to find the final one
            all_children = base_workchain.called
            if not all_children:
                mdb_b_ut.custom_print(f'No children found for workchain {pk}', 'warn')
                continue

            # The final workchain should be the last one
            final_workchain = all_children[-1]
            mdb_b_ut.custom_print(
                f'Found final workchain: {final_workchain.pk}', 'debug'
            )

            # Extract database paths from the final workchain
            seed_db_size = 0
            final_db_size = 0

            # Get seed database size
            try:
                seed_db_path = final_workchain.inputs.seed_db_path.value
                mdb_b_ut.custom_print(
                    f'Loading seed database from: {seed_db_path}', 'debug'
                )

                # Load and count structures in seed database
                seed_structures = ase_read(seed_db_path, index=':')
                seed_db_size = len(seed_structures)

            except (AttributeError, KeyError, Exception) as e:
                mdb_b_ut.custom_print(
                    f'Could not load seed database for {run_name}: {e}', 'warn'
                )

            # Get final training database size
            try:
                training_db_path = final_workchain.inputs.training_db_path.value
                mdb_b_ut.custom_print(
                    f'Loading training database from: {training_db_path}', 'debug'
                )

                # Load and count structures in training database
                training_structures = ase_read(training_db_path, index=':')
                final_db_size = len(training_structures)

            except (AttributeError, KeyError, Exception) as e:
                mdb_b_ut.custom_print(
                    f'Could not load training database for {run_name}: {e}', 'warn'
                )

            # Store results
            results[run_name] = {
                'pk': pk,
                'seed_db_size': seed_db_size,
                'final_db_size': final_db_size,
                'structures_added': final_db_size - seed_db_size,
            }

            mdb_b_ut.custom_print(
                f'{run_name}: Seed DB = {seed_db_size}, Final DB = {final_db_size}, '
                f'Added = {final_db_size - seed_db_size} structures',
                'info',
            )

        except Exception as e:
            mdb_b_ut.custom_print(f'Failed to process workchain {pk}: {e}', 'error')
            continue

    # Save results to file
    if results:
        results_file = benchmark_dir / 'database_sizes.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Store plot data for final multi-panel figure
        run_names = list(results.keys())
        seed_sizes = [results[name]['seed_db_size'] for name in run_names]
        final_sizes = [results[name]['final_db_size'] for name in run_names]
        added_sizes = [results[name]['structures_added'] for name in run_names]

        mdb_b_ut.set_plot_data(
            'database_sizes',
            {
                'type': 'database_sizes',
                'run_names': run_names,
                'seed_sizes': seed_sizes,
                'final_sizes': final_sizes,
                'added_sizes': added_sizes,
                'title': 'Final Training Database Sizes',
            },
        )

        # Print summary
        mdb_b_ut.custom_print('Database Size Summary:', 'info')
        for name in run_names:
            data = results[name]
            mdb_b_ut.custom_print(
                f'  {name}: {data["seed_db_size"]} → {data["final_db_size"]} '
                f'(+{data["structures_added"]} structures)',
                'empty',
            )

        mdb_b_ut.custom_print(f'Results saved to {results_file}', 'info')
    else:
        mdb_b_ut.custom_print('No database size results to save.', 'warn')


def run_md_count_benchmark(args, model_paths: list[pl.Path]):
    """Count total MD calculations performed during Active Learning loops."""
    mdb_b_ut.custom_print('Running MD Count Benchmark', 'info')
    benchmark_dir = args.output_dir / 'md_count'
    benchmark_dir.mkdir(exist_ok=True)

    results = {}

    # Process AiiDA workchains if provided
    if args.aiida_pks:
        for pk in args.aiida_pks:
            try:
                from aiida import load_profile
                from aiida.orm import load_node

                load_profile()

                # Load the base workchain
                base_workchain = load_node(pk)

                # Get run name for display
                run_name = None
                try:
                    run_name = base_workchain.inputs.active_learning.run_name.value
                except (AttributeError, KeyError):
                    run_name = f'workchain_{pk}'

                mdb_b_ut.custom_print(
                    f'Processing workchain {pk} ({run_name})...', 'info'
                )

                # Check cache file first
                cache_file = benchmark_dir / f'{run_name}_md_count.json'
                if cache_file.exists():
                    mdb_b_ut.custom_print(
                        f"Loading cached results for '{run_name}'", 'info'
                    )
                    with open(cache_file) as f:
                        results[run_name] = json.load(f)
                    continue

                # Count ProcessMDSeedStructCalculation children
                mdb_b_ut.custom_print(
                    f'Counting MD calculations for {run_name}...', 'info'
                )

                md_count = 0
                total_children = 0

                # Get all descendant nodes
                all_descendants = base_workchain.called_descendants
                total_children = len(all_descendants)

                # Count ProcessMDSeedStructCalculation nodes
                for node in all_descendants:
                    if hasattr(node, 'process_class'):
                        process_class_name = node.process_class.__name__
                        if process_class_name == 'ProcessMDSeedStructCalculation':
                            md_count += 1

                # Store results
                results[run_name] = {
                    'pk': pk,
                    'md_calculations': md_count,
                    'total_children': total_children,
                }

                # Cache results
                with open(cache_file, 'w') as f:
                    json.dump(results[run_name], f, indent=2)

                mdb_b_ut.custom_print(
                    f'{run_name}: {md_count} MD calculations '
                    f'(out of {total_children} total children)',
                    'info',
                )

            except Exception as e:
                mdb_b_ut.custom_print(f'Failed to process workchain {pk}: {e}', 'error')
                continue

    # Process model files (show 0 MD calculations for each)
    # Exclude foundation models since they don't have MD calculation history
    for model_path in model_paths:
        # Skip foundation models - they don't have AL runs with MD calculations
        if hasattr(model_path, 'foundation_model_spec'):
            continue

        model_name = mdb_b_ut.get_model_display_name(model_path)
        if model_name not in results:  # Only add if not already processed from AiiDA
            results[model_name] = {
                'pk': None,
                'md_calculations': 0,
                'total_children': 0,
            }
            mdb_b_ut.custom_print(
                f'{model_name}: 0 MD calculations (loaded from file)', 'info'
            )

    # Save combined results and store plot data
    if results:
        # Save all results
        all_results_file = benchmark_dir / 'all_md_counts.json'
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Store plot data for final multi-panel figure
        run_names = list(results.keys())
        md_counts = [results[name]['md_calculations'] for name in run_names]

        mdb_b_ut.set_plot_data(
            'md_count',
            {
                'type': 'bar',
                'model_names': run_names,
                'values': md_counts,
                'title': 'Total MD Calculations per AL Run',
                'ylabel': 'Number of MD Calculations',
                'value_format': '{:.0f}',
            },
        )

        # Print summary
        mdb_b_ut.custom_print('MD Count Summary:', 'info')
        for name in run_names:
            data = results[name]
            if data['pk']:
                mdb_b_ut.custom_print(
                    f'  {name}: {data["md_calculations"]} MD calculations', 'empty'
                )
            else:
                mdb_b_ut.custom_print(
                    f'  {name}: {data["md_calculations"]} MD calculations (from file)',
                    'empty',
                )

        mdb_b_ut.custom_print(f'Results saved to {all_results_file}', 'info')
    else:
        mdb_b_ut.custom_print('No MD count results to save.', 'warn')


def run_melting_point_benchmark(args, model_paths: list[pl.Path]):
    mdb_b_ut.custom_print('Running Melting Point Benchmark', 'info')
    benchmark_dir = args.output_dir / 'melting_point'
    benchmark_dir.mkdir(exist_ok=True)

    # Get settings from args (already loaded from TOML)
    benchmark_settings = {
        'supercell_size': args.melting_point_supercell_size,
        'solid_temp_K': args.melting_point_solid_temp_K,
        'liquid_temp_K': args.melting_point_liquid_temp_K,
        'nve_initial_T_test_K': args.melting_point_nve_initial_T_test_K,
        'supercell_path': args.melting_point_supercell_path,
    }

    # Results storage
    results = {}

    # For each model, prepare structure and test at different temperatures
    for _, model_path in enumerate(model_paths):
        print()
        model_name = mdb_b_ut.get_model_display_name(model_path)
        mdb_b_ut.custom_print(
            f"Testing melting point for model: '{model_name}'", 'info'
        )

        # Skip if already calculated
        results_file = benchmark_dir / f'{model_name}_melting_point.json'
        if results_file.exists():
            mdb_b_ut.custom_print(
                f"Results for '{model_name}' already exist. Loading from file.", 'warn'
            )
            with open(results_file) as f:
                results[model_name] = json.load(f)
            continue

        try:
            # Set up calculator for this model
            if 'lammps.pt' not in model_path.name:
                calculator = mdb_b_ut.create_calculator_for_model(
                    model_path, device=args.device, dtype=args.dtype, enable_cueq=True
                )

            # Create model-specific directory for structures and trajectories
            model_benchmark_dir = benchmark_dir / model_name
            model_benchmark_dir.mkdir(exist_ok=True)

            # Prepare coexistence structure for this specific model
            mdb_b_ut.custom_print(
                f'Preparing coexistence structure for model: {model_name}', 'info'
            )

            # 1. Prepare initial cell, and heat to temperature below melting point
            # 2. Prepare initial cell, and heat to temperature above melting point
            # 3. Join two cells together, and run 10 steps at solid_temp_K
            # 4. Run NVE, checking for coexistence. Average temperature will be
            # the melting temperature.
            melting_temperature, coexistence_path = coexistence_structure_melting_point(
                metal=args.metal,
                benchmark_dir=model_benchmark_dir,
                calculator=calculator,
                config_dict=benchmark_settings,
            )

            results[model_name] = {
                'estimated_melting_point': melting_temperature,
                'coexistence_structure_path': str(coexistence_path),
            }

        except Exception as exc:
            mdb_b_ut.custom_print(
                f"Failed to calculate melting point for '{model_name}': {exc}",
                'error',
            )
            continue

    # Save combined results and store plot data
    if results:
        # Save all results
        all_results_file = benchmark_dir / 'all_melting_points.json'
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Store plot data for final multi-panel figure
        model_names = list(results.keys())
        melting_points = []

        for model_name in model_names:
            tm = results[model_name].get('estimated_melting_point')
            melting_points.append(tm if tm is not None else 0.0)

        mdb_b_ut.set_plot_data(
            'melting_point',
            {
                'type': 'bar',
                'model_names': model_names,
                'values': melting_points,
                'title': f'Melting Point - {args.metal}',
                'ylabel': 'Temperature (K)',
                'value_format': '{:.0f}',
            },
        )

        # Print summary
        mdb_b_ut.custom_print('Melting Point Summary:', 'info')
        for name, tm in zip(model_names, melting_points, strict=True):
            if tm > 0:
                mdb_b_ut.custom_print(f'  {name}: {tm:.1f} K', 'info')
            else:
                mdb_b_ut.custom_print(f'  {name}: Unable to determine', 'warning')

        mdb_b_ut.custom_print(f'Results saved to {all_results_file}', 'info')
    else:
        mdb_b_ut.custom_print('No melting point results to save.', 'warn')


def run_energy_md_benchmark(args, model_paths: list[pl.Path]):
    """
    Run the energy MD benchmark.

    Generates a surface slab, runs MD for each model, and plots the results.
    """
    benchmark_dir = args.output_dir / 'energy_md_benchmark'
    benchmark_dir.mkdir(exist_ok=True)

    # Generate Benchmark Structure
    slab = surface(
        args.metal,
        tuple(args.surface_indices),
        args.supercell_size[2],
        vacuum=args.vacuum,
    )
    slab = slab.repeat(tuple(args.supercell_size[:2]) + (1,))
    slab.center(vacuum=args.vacuum, axis=2)
    slab.pbc = True
    structure_path = benchmark_dir / 'initial_structure.xyz'
    slab.write(structure_path, format='extxyz')

    # Run MD for each model
    energy_data = {}
    force_data = {}
    for _, model_path in enumerate(model_paths):
        model_name = mdb_b_ut.get_model_display_name(model_path)
        mdb_b_ut.custom_print(f'Running MD for model: {model_name}', 'info')

        # Skip if already run
        energy_file = benchmark_dir / f'{model_name}_energies.npy'
        force_file = benchmark_dir / f'{model_name}_max_forces.npy'
        if energy_file.exists() and force_file.exists():
            mdb_b_ut.custom_print(
                f'Results for {model_name} already exist. Skipping MD.', 'warn'
            )
            energy_data[model_name] = np.load(energy_file)
            force_data[model_name] = np.load(force_file)
            continue

        # Load structure and set calculator
        atoms = ase_read(structure_path, format='extxyz')
        calculator = mdb_b_ut.create_calculator_for_model(
            model_path, device=args.device, dtype=args.dtype
        )
        atoms.set_calculator(calculator)

        # Set up Langevin dynamics
        dyn = Langevin(
            atoms,
            args.timestep * units.fs,
            temperature_K=args.temp,
            friction=args.friction,
        )

        # Set up trajectory writer and logger
        traj_path = benchmark_dir / f'{model_name}_md.traj'
        traj = TrajectoryWriter(traj_path, 'w', atoms)
        dyn.attach(traj.write, interval=10)
        dyn.attach(
            MDLogger(dyn, atoms, '-', header=True, stress=False, mode='a'),
            interval=100,
        )

        # Store energies and forces
        energies = []
        max_forces = []

        def log_data(a=atoms, e_list=energies, f_list=max_forces):
            e_list.append(a.get_potential_energy())
            f_list.append(np.max(np.linalg.norm(a.get_forces(), axis=1)))

        dyn.attach(log_data, interval=1)

        # Run dynamics
        try:
            dyn.run(args.n_steps)
            mdb_b_ut.custom_print(f'MD for {model_name} finished!', 'done')
            np.save(energy_file, np.array(energies))
            np.save(force_file, np.array(max_forces))
            energy_data[model_name] = np.array(energies)
            force_data[model_name] = np.array(max_forces)
        except Exception as e:
            mdb_b_ut.custom_print(f'MD for {model_name} failed: {e}', 'error')

    if hasattr(args, 'mode'):
        if args.mode == 'high_energy':
            plot_data_key = 'energy_md_high_energy'
            title = f'High Temperature MD Benchmark ({args.temp} K)'
    else:
        plot_data_key = 'energy_md'
        title = 'Energy MD Benchmark'
    # Store plot data for final multi-panel figure
    mdb_b_ut.set_plot_data(
        plot_data_key,
        {
            'type': plot_data_key,
            'energy_data': energy_data,
            'force_data': force_data,
            'timestep': args.timestep,
            'title': title,
        },
    )

    mdb_b_ut.custom_print('Energy MD data collected for final plot', 'info')


def run_elastic_properties_benchmark(args, model_paths: list[pl.Path]):
    """Calculates and plots elastic constants and bulk modulus."""
    mdb_b_ut.custom_print('Running Elastic Properties Benchmark', 'info')
    benchmark_dir = args.output_dir / 'elastic_properties'
    benchmark_dir.mkdir(exist_ok=True)

    # 1. Create an equilibrium bulk structure (e.g., Cu FCC).
    # from ase.build import bulk

    # 2. For each model:
    #    - Attach the calculator to the bulk structure.
    #    - Use ASE's Elasticity module to calculate constants.
    #    from ase.constraints import ExpCellFilter
    #    from ase.optimize import LBFGS
    #    from ase.elasticity import Elasticity

    # 3. Extract C11, C12, C44 and Bulk Modulus.

    # 4. Save results to a file.

    # 5. Generate bar plots comparing the elastic properties for each model.
    mdb_b_ut.custom_print('Elastic properties benchmark not implemented yet.', 'warn')


def run_defect_formation_energy_benchmark(args, model_paths: list[pl.Path]):
    """Calculates and plots the monovacancy formation energy."""
    mdb_b_ut.custom_print('Running Defect Formation Energy Benchmark', 'info')
    benchmark_dir = args.output_dir / 'defect_formation_energy'
    benchmark_dir.mkdir(exist_ok=True)

    # Results storage
    results = {}

    # Create a large perfect supercell
    # (4x4x4 should be sufficient)
    supercell_size = [4, 4, 4]
    # mdb_b_ut.custom_print(
    #     f'Creating {supercell_size} supercell for defect calculations...', 'debug'
    # )

    perfect_bulk = bulk(args.metal, cubic=True)
    perfect_supercell = perfect_bulk.repeat(supercell_size)

    # Save the perfect structure
    perfect_structure_path = benchmark_dir / 'perfect_supercell.xyz'
    perfect_supercell.write(perfect_structure_path, format='extxyz')

    # Create vacancy structure by removing the central atom
    # Remove roughly central atom
    defect_supercell = perfect_supercell.copy()
    central_index = len(defect_supercell) // 2
    del defect_supercell[central_index]

    # Save the defect structure
    defect_structure_path = benchmark_dir / 'vacancy_supercell.xyz'
    defect_supercell.write(defect_structure_path, format='extxyz')

    n_atoms_perfect = len(perfect_supercell)
    n_atoms_defect = len(defect_supercell)

    # For each model, calculate formation energies
    for _, model_path in enumerate(model_paths):
        model_name = mdb_b_ut.get_model_display_name(model_path)

        # Skip if already calculated
        results_file = benchmark_dir / f'{model_name}_vacancy_formation_energy.json'
        if results_file.exists():
            mdb_b_ut.custom_print(
                f"Results for '{model_name}' already exist. Loading from file.", 'warn'
            )
            with open(results_file) as f:
                results[model_name] = json.load(f)
            continue

        try:
            # Set up calculator
            calculator = mdb_b_ut.create_calculator_for_model(
                model_path, device=args.device, dtype=args.dtype
            )

            # Load and relax perfect supercell
            perfect_atoms = ase_read(perfect_structure_path, format='extxyz')
            perfect_atoms.set_calculator(calculator)

            # Relax the perfect structure
            perfect_optimizer = LBFGS(
                perfect_atoms, logfile=benchmark_dir / f'{model_name}_perfect_relax.log'
            )
            perfect_optimizer.run(fmax=0.01, steps=500)
            e_perfect = perfect_atoms.get_potential_energy()

            # Save relaxed perfect structure
            perfect_relaxed_path = benchmark_dir / f'{model_name}_perfect_relaxed.xyz'
            perfect_atoms.write(perfect_relaxed_path, format='extxyz')

            # Load and relax defect supercell
            defect_atoms = ase_read(defect_structure_path, format='extxyz')
            defect_atoms.set_calculator(calculator)

            # Relax the defect structure
            defect_optimizer = LBFGS(
                defect_atoms, logfile=benchmark_dir / f'{model_name}_defect_relax.log'
            )
            defect_optimizer.run(fmax=0.01, steps=500)
            e_defect = defect_atoms.get_potential_energy()

            # Save relaxed defect structure
            defect_relaxed_path = benchmark_dir / f'{model_name}_defect_relaxed.xyz'
            defect_atoms.write(defect_relaxed_path, format='extxyz')

            # Calculate formation energy
            # E_formation = E_defect - (N-1)/N * E_perfect
            # This is equivalent to: E_defect - E_perfect + E_perfect/N
            # where E_perfect/N is the energy per atom in the perfect crystal
            e_per_atom_perfect = e_perfect / n_atoms_perfect
            formation_energy = e_defect - e_perfect + e_per_atom_perfect

            # Store results
            model_results = {
                'formation_energy_eV': formation_energy,
                'perfect_energy_eV': e_perfect,
                'defect_energy_eV': e_defect,
                'perfect_energy_per_atom_eV': e_per_atom_perfect,
                'n_atoms_perfect': n_atoms_perfect,
                'n_atoms_defect': n_atoms_defect,
            }

            results[model_name] = model_results

            # Save individual results
            with open(results_file, 'w') as f:
                json.dump(model_results, f, indent=2)

            mdb_b_ut.custom_print(
                f"Vacancy formation energy for '{model_name}': "
                f'{formation_energy:.3f} eV',
                'done',
            )

        except Exception as e:
            mdb_b_ut.custom_print(
                f"Failed to calculate vacancy formation energy for '{model_name}': {e}",
                'error',
            )
            continue

    # Save combined results and store plot data
    if results:
        # Save all results
        all_results_file = benchmark_dir / 'all_vacancy_formation_energies.json'
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Store plot data for final multi-panel figure
        model_names = list(results.keys())
        formation_energies = [
            results[name]['formation_energy_eV'] for name in model_names
        ]

        mdb_b_ut.set_plot_data(
            'defect_formation',
            {
                'type': 'bar',
                'model_names': model_names,
                'values': formation_energies,
                'title': f'Monovacancy Formation Energy - {args.metal}',
                'ylabel': 'Formation Energy (eV)',
                'value_format': '{:.3f}',
            },
        )

        # Print summary
        mdb_b_ut.custom_print('Vacancy Formation Energy Summary:', 'info')
        for name, energy in zip(model_names, formation_energies, strict=True):
            mdb_b_ut.custom_print(f'  {name}: {energy:.3f} eV', 'empty')
    else:
        mdb_b_ut.custom_print(
            'No results to plot for vacancy formation energy benchmark.', 'warn'
        )


def run_surface_energies_benchmark(args, model_paths: list[pl.Path]):
    """
    Calculates and plots energies for low-index surfaces.

    The surface energy represents the excess energy per unit
    area due to the creation of a surface. Lower values indicate
    more stable surfaces.

    This benchmark validates whether the explored MLIPs reproduce the correct
    energetic ordering and magnitudes of surface energies compared to
    DFT or experimental values.

    """
    mdb_b_ut.custom_print('Running Surface Energies Benchmark', 'info')
    benchmark_dir = args.output_dir / 'surface_energies'
    benchmark_dir.mkdir(exist_ok=True)

    # Print information about user-provided DFT structures if any
    bulk_structure_arg = getattr(args, 'surf_ene_benchmark_bulk_structure', None)
    slab_structures_arg = getattr(args, 'surf_ene_benchmark_slab_structures', None)
    dft_refs_arg = getattr(args, 'surf_ene_benchmark_dft_refs', None)

    if bulk_structure_arg or slab_structures_arg or dft_refs_arg:
        mdb_b_ut.custom_print(
            'Surface Energy Benchmark - DFT Structure Options:', 'info'
        )
        if bulk_structure_arg:
            mdb_b_ut.custom_print(f'  DFT bulk structure: {bulk_structure_arg}', 'info')
        if slab_structures_arg:
            mdb_b_ut.custom_print(
                f'  DFT slab structures: {len(slab_structures_arg)} provided', 'info'
            )
        if dft_refs_arg:
            mdb_b_ut.custom_print(f'  DFT reference energies: {dft_refs_arg}', 'info')

    # Define surfaces to test (Miller indices)
    surfaces_to_test = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]

    # Number of layers in the slab
    slab_layers = 7

    # Supercell size in x and y directions
    slab_size = [3, 3]

    # Vacuum thickness in Angstroms
    vacuum_thickness = 15.0

    # Results storage
    results = {}

    # Generate bulk primitive cell
    bulk_structure_arg = getattr(args, 'surf_ene_benchmark_bulk_structure', None)
    if hasattr(args, 'surf_ene_benchmark_bulk_structure') and bulk_structure_arg:
        # Use user-provided DFT-optimized bulk structure
        bulk_structure_path = args.surf_ene_benchmark_bulk_structure
        if not bulk_structure_path.exists():
            mdb_b_ut.custom_print(
                f'DFT bulk structure file not found: {bulk_structure_path}', 'error'
            )
            return

        # Handle common formats
        struct_format = mdb_b_ut._get_structure_format(bulk_structure_path)

        bulk_primitive = ase_read(bulk_structure_path, format=struct_format)
        mdb_b_ut.custom_print(
            f'Using user-provided DFT-optimized bulk structure: {bulk_structure_path}',
            'info',
        )

        # Save the bulk structure to benchmark directory
        bulk_primitive_path = benchmark_dir / 'bulk_primitive.xyz'
        bulk_primitive.write(bulk_primitive_path, format='extxyz')
        print('#@# bulk_atoms (from DFT): ', len(bulk_primitive))
    else:
        # Create bulk primitive cell (no supercell)
        bulk_primitive = bulk(args.metal, cubic=True)

        # Save the bulk primitive structure
        bulk_primitive_path = benchmark_dir / 'bulk_primitive.xyz'
        print('#@# benchmark_dir: ', benchmark_dir)
        bulk_primitive.write(bulk_primitive_path, format='extxyz')

    # Handle slab structures (if user-provided DFT slabs are available)
    user_provided_slabs = {}

    # Parse user-provided slab structures if available
    slab_structures_arg = getattr(args, 'surf_ene_benchmark_slab_structures', None)
    if hasattr(args, 'surf_ene_benchmark_slab_structures') and slab_structures_arg:
        for slab_spec in args.surf_ene_benchmark_slab_structures:
            print('#@# slab_spec: ', slab_spec)
            if ':' not in slab_spec:
                mdb_b_ut.custom_print(
                    f'Invalid slab specification format: {slab_spec}. '
                    'Expected format: "surface_indices:path_to_structure"',
                    'error',
                )
                continue

            surface_indices_str, slab_path_str = slab_spec.split(':', 1)
            slab_path = pl.Path(slab_path_str)

            if not slab_path.exists():
                mdb_b_ut.custom_print(
                    f'DFT slab structure file not found: {slab_path}', 'error'
                )
                continue

            # Convert surface indices string to tuple (e.g., "100" -> (1, 0, 0))
            try:
                surface_indices = tuple(int(x) for x in surface_indices_str)
                surface_name = ''.join(map(str, surface_indices))
                user_provided_slabs[surface_name] = slab_path
                mdb_b_ut.custom_print(
                    f'Using user-provided DFT-optimized slab for '
                    f'{args.metal}({surface_name}): {slab_path}',
                    'info',
                )
            except ValueError:
                mdb_b_ut.custom_print(
                    f'Invalid surface indices format: {surface_indices_str}. '
                    'Expected numeric indices like "100", "110", "111"',
                    'error',
                )
                continue

    # Load DFT reference data if provided
    dft_references = {}
    dft_surface_energies = {}
    dft_refs_arg = getattr(args, 'surf_ene_benchmark_dft_refs', None)
    if hasattr(args, 'surf_ene_benchmark_dft_refs') and dft_refs_arg:
        if args.surf_ene_benchmark_dft_refs.exists():
            try:
                with open(args.surf_ene_benchmark_dft_refs) as f:
                    dft_references = json.load(f)

                mdb_b_ut.custom_print(
                    f'Loaded DFT reference data from '
                    f'{args.surf_ene_benchmark_dft_refs}',
                    'info',
                )

                # Calculate DFT surface energies if bulk energy is provided
                if 'bulk' in dft_references:
                    # Use num_atoms from JSON if provided, otherwise use ASE bulk atoms
                    if isinstance(dft_references['bulk'], dict):
                        # Bulk entry is a dictionary with energy and num_atoms
                        bulk_total_energy = dft_references['bulk']['energy']
                        bulk_num_atoms = dft_references['bulk']['num_atoms']
                        bulk_energy_per_atom = bulk_total_energy / bulk_num_atoms
                    else:
                        # Bulk entry is just a number (backwards compatible)
                        bulk_total_energy = dft_references['bulk']
                        bulk_num_atoms = len(bulk_primitive)
                        bulk_energy_per_atom = bulk_total_energy / bulk_num_atoms
                        mdb_b_ut.custom_print(
                            f'Using ASE bulk primitive cell with {bulk_num_atoms} '
                            'atoms. Consider specifying num_atoms in JSON.',
                            'warn',
                        )

                    mdb_b_ut.custom_print(
                        f'DFT bulk energy per atom: {bulk_energy_per_atom:.6f} eV '
                        f'(total: {bulk_total_energy:.6f} eV, {bulk_num_atoms} atoms)',
                        'info',
                    )

                    for surface_indices in surfaces_to_test:
                        surface_name = ''.join(map(str, surface_indices))

                        if surface_name in dft_references:
                            print('#@# surface_name: ', surface_name)

                            # Handle surface entry - can be number or dict
                            if isinstance(dft_references[surface_name], dict):
                                # Surface entry has energy and optional num_atoms
                                surface_data = dft_references[surface_name]
                                slab_total_energy = surface_data['energy']
                                n_atoms = surface_data.get('num_atoms')
                                if n_atoms is None:
                                    mdb_b_ut.custom_print(
                                        f'Surface {surface_name}: num_atoms not '
                                        'specified in DFT reference. Will use actual '
                                        'slab atom count.',
                                        'warn',
                                    )
                                    continue
                            else:
                                # Surface entry is just a number (backward compatible)
                                slab_total_energy = dft_references[surface_name]
                                mdb_b_ut.custom_print(
                                    f'Surface {surface_name}: using number format. '
                                    'Consider using dict format with num_atoms.',
                                    'warn',
                                )
                                continue

                            print('#@# slab_total_energy: ', slab_total_energy)
                            print('#@# n_atoms (DFT): ', n_atoms)

                            # Calculate approximate area for DFT reference
                            # (will be updated with actual area when slabs are created)
                            temp_slab = surface(
                                args.metal,
                                surface_indices,
                                slab_layers,
                                vacuum=vacuum_thickness,
                            )
                            temp_slab = temp_slab.repeat(slab_size + [1])
                            cell = temp_slab.get_cell()
                            area = cell[0, 0] * cell[1, 1]

                            # Surface energy: γ = (E_slab - N * E_bulk) / (2 * A)
                            # Factor of 2 because slab has two surfaces
                            surface_energy_eV_per_A2 = (
                                slab_total_energy - n_atoms * bulk_energy_per_atom
                            ) / (2 * area)
                            print(
                                '#@# surface_energy_eV_per_A2: ',
                                surface_energy_eV_per_A2,
                            )

                            # Convert eV/Å² to J/m²
                            surface_energy_J_per_m2 = surface_energy_eV_per_A2 * 16.0218
                            print(
                                '#@# surface_energy_J_per_m2: ', surface_energy_J_per_m2
                            )

                            dft_surface_energies[surface_name] = surface_energy_J_per_m2

                            mdb_b_ut.custom_print(
                                f'DFT {args.metal}({surface_name}) surface energy: '
                                f'{surface_energy_J_per_m2:.3f} J/m² '
                                f'(slab: {slab_total_energy:.3f} eV, {n_atoms} atoms)',
                                'info',
                            )
                else:
                    mdb_b_ut.custom_print(
                        'No bulk energy found in DFT references. '
                        'Cannot calculate surface energies.',
                        'warn',
                    )
                    mdb_b_ut.custom_print(
                        'Expected JSON format: '
                        '{"100": {"energy": -1231.2, "num_atoms": 63}, '
                        '"110": -1432.3, "111": -1441.1, '
                        '"bulk": {"energy": -89.1, "num_atoms": 4}}',
                        'info',
                    )

            except Exception as e:
                mdb_b_ut.custom_print(f'Error loading DFT reference file: {e}', 'error')
        else:
            mdb_b_ut.custom_print(
                f'DFT reference file not found: {args.surf_ene_benchmark_dft_refs}',
                'warn',
            )

    # For each model, calculate surface energies
    for _, model_path in enumerate(model_paths):
        model_name = mdb_b_ut.get_model_display_name(model_path)

        # Skip if already calculated
        results_file = benchmark_dir / f'{model_name}_surface_energies.json'
        if results_file.exists():
            mdb_b_ut.custom_print(
                f"Results for '{model_name}' already exist. Loading from file.", 'warn'
            )
            with open(results_file) as f:
                results[model_name] = json.load(f)
            continue

        try:
            # Set up calculator
            calculator = mdb_b_ut.create_calculator_for_model(
                model_path, device=args.device, dtype=args.dtype
            )

            # Step 1: Relax the bulk structure with this model
            struct_format = mdb_b_ut._get_structure_format(bulk_primitive_path)
            bulk_atoms = ase_read(bulk_primitive_path, format=struct_format)
            bulk_atoms.set_calculator(calculator)

            # Check if bulk structure is DFT-provided (skip relaxation) or ASE-generated
            bulk_structure_arg = getattr(
                args, 'surf_ene_benchmark_bulk_structure', None
            )
            if bulk_structure_arg and bulk_structure_arg.exists():
                # DFT-provided bulk structure - evaluate energy without relaxation
                mdb_b_ut.custom_print(
                    f'Using DFT-optimized bulk structure for {model_name} '
                    '(skipping relaxation)',
                    'info',
                )
                e_bulk_total = bulk_atoms.get_potential_energy()
                e_bulk_per_atom = e_bulk_total / len(bulk_atoms)

                # Save structure with model name for consistency
                bulk_relaxed_path = benchmark_dir / f'{model_name}_bulk_dft.xyz'
                bulk_atoms.write(bulk_relaxed_path, format='extxyz')
            else:
                # ASE-generated bulk structure - relax with MLIP
                mdb_b_ut.custom_print(
                    f'Relaxing bulk structure with {model_name}', 'info'
                )
                bulk_optimizer = LBFGS(
                    bulk_atoms, logfile=benchmark_dir / f'{model_name}_bulk_relax.log'
                )
                bulk_optimizer.run(fmax=0.01, steps=500)
                e_bulk_total = bulk_atoms.get_potential_energy()
                e_bulk_per_atom = e_bulk_total / len(bulk_atoms)

                # Save relaxed bulk structure
                bulk_relaxed_path = benchmark_dir / f'{model_name}_bulk_relaxed.xyz'
                bulk_atoms.write(bulk_relaxed_path, format='extxyz')

            # Step 2: Create supercell from relaxed bulk for surface creation
            bulk_supercell = bulk_atoms.copy()
            # Create a moderate supercell for surface cutting
            supercell_for_surface = [2, 2, 2]
            bulk_supercell = bulk_supercell.repeat(supercell_for_surface)

            # Save the supercell
            bulk_supercell_path = benchmark_dir / f'{model_name}_bulk_supercell.xyz'
            bulk_supercell.write(bulk_supercell_path, format='extxyz')

            # Initialize model results
            model_results = {
                'bulk_energy_per_atom_eV': e_bulk_per_atom,
                'bulk_total_energy_eV': e_bulk_total,
                'surfaces': {},
            }

            # Step 3: For each surface, create slab from the relaxed bulk supercell
            # and relax it
            for surface_indices in surfaces_to_test:
                surface_name = ''.join(map(str, surface_indices))

                # Check if user provided DFT slab for this surface
                if surface_name in user_provided_slabs:
                    # Use user-provided DFT-optimized slab
                    slab_path = user_provided_slabs[surface_name]
                    struct_format = mdb_b_ut._get_structure_format(slab_path)
                    slab_atoms = ase_read(slab_path, format=struct_format)

                    # Copy to benchmark directory for consistency
                    benchmark_slab_path = (
                        benchmark_dir / f'{model_name}_slab_{surface_name}_dft.xyz'
                    )
                    slab_atoms.write(benchmark_slab_path, format='extxyz')

                    # Evaluate energy without relaxation (DFT structure)
                    slab_atoms.set_calculator(calculator)
                    e_slab = slab_atoms.get_potential_energy()

                    mdb_b_ut.custom_print(
                        f'Using DFT-optimized slab {surface_name} for {model_name} '
                        '(skipping relaxation)',
                        'info',
                    )

                    slab_source = 'dft_provided'
                else:
                    # Create slab from the model's relaxed bulk structure
                    # Use the original metal name to create surface, but with
                    # lattice parameter from relaxed bulk
                    struct_format = mdb_b_ut._get_structure_format(bulk_relaxed_path)
                    bulk_for_surface = ase_read(bulk_relaxed_path, format=struct_format)

                    # Create surface using the metal name but with relaxed lattice
                    # parameters
                    slab_atoms = surface(
                        # args.metal,
                        lattice=bulk_for_surface,
                        indices=surface_indices,
                        layers=slab_layers,
                        vacuum=vacuum_thickness,
                        # lattice=bulk_for_surface.get_cell(),
                    )
                    slab_atoms = slab_atoms.repeat(slab_size + [1])
                    slab_atoms.center(vacuum=vacuum_thickness, axis=2)
                    slab_atoms.pbc = True

                    # Save initial slab structure
                    initial_slab_path = (
                        benchmark_dir / f'{model_name}_slab_{surface_name}_initial.xyz'
                    )
                    slab_atoms.write(initial_slab_path, format='extxyz')

                    # Relax the slab
                    slab_atoms.set_calculator(calculator)
                    mdb_b_ut.custom_print(
                        f'Relaxing slab {surface_name} with {model_name}', 'info'
                    )
                    slab_optimizer = LBFGS(
                        slab_atoms,
                        logfile=benchmark_dir
                        / f'{model_name}_slab_{surface_name}_relax.log',
                    )
                    slab_optimizer.run(fmax=0.01, steps=500)
                    e_slab = slab_atoms.get_potential_energy()

                    # Save relaxed slab structure
                    benchmark_slab_path = (
                        benchmark_dir / f'{model_name}_slab_{surface_name}_relaxed.xyz'
                    )
                    slab_atoms.write(benchmark_slab_path, format='extxyz')

                    slab_source = 'created_from_relaxed_bulk'

                # Calculate surface energy
                # Surface Energy = (E_slab - N_slab * E_bulk_per_atom) / (2 * Area)
                # Factor of 2 because there are two surfaces in a slab
                n_atoms_slab = len(slab_atoms)
                cell = slab_atoms.get_cell()
                area = cell[0, 0] * cell[1, 1]

                surface_energy_total = e_slab - n_atoms_slab * e_bulk_per_atom

                # eV/Å² units
                surface_energy_per_area = surface_energy_total / (2 * area)

                # Convert to more common units (J/m²)
                # 1 eV/Å² = 16.02176 J/m²
                surface_energy_j_m2 = surface_energy_per_area * 16.02176
                print('#@# mlip surface_energy_j_m2: ', surface_energy_j_m2)

                model_results['surfaces'][surface_name] = {
                    'surface_energy_eV_per_A2': surface_energy_per_area,
                    'surface_energy_J_per_m2': surface_energy_j_m2,
                    'slab_energy_eV': e_slab,
                    'slab_n_atoms': n_atoms_slab,
                    'surface_area_A2': area,
                    'slab_source': slab_source,
                }

                mdb_b_ut.custom_print(
                    f'{model_name} - {args.metal}({surface_name}): '
                    f'{surface_energy_j_m2:.3f} J/m² '
                    f'(slab: {e_slab:.3f} eV, {n_atoms_slab} atoms, '
                    f'source: {slab_source})',
                    'info',
                )

            results[model_name] = model_results

            # Save individual results
            with open(results_file, 'w') as f:
                json.dump(model_results, f, indent=2)

            mdb_b_ut.custom_print(
                f"Surface energies calculated for '{model_name}'", 'done'
            )

        except Exception as e:
            mdb_b_ut.custom_print(
                f"Failed to calculate surface energies for '{model_name}': {e}",
                'error',
            )
            continue

    # Save combined results and store plot data
    if results:
        # Save all results
        all_results_file = benchmark_dir / 'all_surface_energies.json'
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Store plot data for final multi-panel figure
        model_names = list(results.keys())
        surface_names = list(surfaces_to_test)

        # Calculate statistics for each surface to determine if zooming is needed
        zoom_info = {}
        all_energies = []  # Collect all surface energies across all surfaces
        surface_stats = {}

        # First pass: collect statistics for each surface
        for surface_indices in surfaces_to_test:
            surface_name = ''.join(map(str, surface_indices))

            # Collect energies for this surface across all models
            surface_energies = []
            for model_name in model_names:
                if surface_name in results[model_name]['surfaces']:
                    energy = results[model_name]['surfaces'][surface_name][
                        'surface_energy_J_per_m2'
                    ]
                    surface_energies.append(energy)
                    all_energies.append(energy)  # Add to global collection

            # Also include DFT energy for this surface if available
            if surface_name in dft_surface_energies:
                dft_energy = dft_surface_energies[surface_name]
                surface_energies.append(dft_energy)
                all_energies.append(dft_energy)  # Add DFT to global collection

            if len(surface_energies) >= 2:  # Need at least 2 models for statistics
                energy_array = np.array(surface_energies)
                std_dev = np.std(energy_array)
                max_energy = np.max(energy_array)
                min_energy = np.min(energy_array)
                mean_energy = np.mean(energy_array)

                # Check if standard deviation is less than 10% of the highest value
                should_zoom = std_dev < (0.1 * max_energy) if max_energy > 0 else False

                surface_stats[surface_name] = {
                    'should_zoom': should_zoom,
                    'std_dev': std_dev,
                    'max_energy': max_energy,
                    'min_energy': min_energy,
                    'mean_energy': mean_energy,
                    'std_dev_percent': (std_dev / max_energy * 100)
                    if max_energy > 0
                    else 0,
                    'energies': surface_energies,
                }

                mdb_b_ut.custom_print(
                    f'Surface {args.metal}({surface_name}): '
                    f'std_dev = {std_dev:.3f} J/m² '
                    f'({std_dev / max_energy * 100:.1f}% of max), zoom = {should_zoom}',
                    'info',
                )

        # Second pass: determine global zoom settings
        should_zoom_globally = False
        global_y_min = None
        global_y_max = None

        if all_energies and len(surface_stats) > 0:
            # Check if ALL surfaces should zoom (conservative approach)
            all_surfaces_should_zoom = all(
                stats['should_zoom'] for stats in surface_stats.values()
            )

            if all_surfaces_should_zoom:
                should_zoom_globally = True

                # Calculate tight y-axis limits based on all energies
                all_energies_array = np.array(all_energies)
                global_min = np.min(all_energies_array)
                global_max = np.max(all_energies_array)
                global_range = global_max - global_min

                # Add 10% margin on both sides for better visualization
                margin = global_range * 0.1 if global_range > 0 else global_max * 0.01
                global_y_min = global_min - margin
                global_y_max = global_max + margin

                mdb_b_ut.custom_print(
                    f'Global zoom enabled: y-axis range [{global_y_min:.3f}, '
                    f'{global_y_max:.3f}] J/m²',
                    'info',
                )
            else:
                mdb_b_ut.custom_print(
                    'Global zoom disabled: not all surfaces meet zoom criteria', 'info'
                )

        # Prepare zoom_info with global settings
        zoom_info = {
            'global_zoom': {
                'enabled': should_zoom_globally,
                'y_min': global_y_min,
                'y_max': global_y_max,
            },
            'surface_stats': surface_stats,
        }

        mdb_b_ut.set_plot_data(
            'surface_energies',
            {
                'type': 'surface_energies',
                'model_names': model_names,
                'surface_names': surface_names,
                'results': results,
                'metal': args.metal,
                'title': 'Surface Energies',
                'zoom_info': zoom_info,
                'dft_surface_energies': dft_surface_energies,
            },
        )

        # Print summary
        mdb_b_ut.custom_print('Surface Energy Summary:', 'info')
        for model_name in model_names:
            mdb_b_ut.custom_print(f'  {model_name}:', 'empty')
            for surface_indices in surface_names:
                surface_name = ''.join(map(str, surface_indices))
                if surface_name in results[model_name]['surfaces']:
                    energy = results[model_name]['surfaces'][surface_name][
                        'surface_energy_J_per_m2'
                    ]
                    mdb_b_ut.custom_print(
                        f'    {args.metal}({surface_name}): {energy:.2f} J/m²', 'empty'
                    )
    else:
        mdb_b_ut.custom_print(
            'No results to plot for surface energies benchmark.', 'warn'
        )


def run_phonon_dispersion_benchmark(args, model_paths: list[pl.Path]):
    """Calculates and plots the phonon dispersion curves."""
    mdb_b_ut.custom_print('Running Phonon Dispersion Benchmark', 'info')
    benchmark_dir = args.output_dir / 'phonon_dispersion'
    benchmark_dir.mkdir(exist_ok=True)

    # 1. Create a primitive cell for the material.
    # 2. Define the high-symmetry path in the Brillouin zone.

    # 3. For each model:
    #    - Use a library like phonopy or ase.phonons.
    #    - Create a supercell for force calculations.
    #    - Calculate the phonon dispersion.

    # 4. Plot the dispersion curves for each model, perhaps overlaying them.
    #    Check for imaginary frequencies.
    mdb_b_ut.custom_print('Phonon dispersion benchmark not implemented yet.', 'warn')


def run_high_temp_md_benchmark(args, model_paths: list[pl.Path]):
    """Runs a high-temperature MD simulation to test stability."""
    benchmark_dir = args.output_dir / 'high_temp_md'
    benchmark_dir.mkdir(exist_ok=True)

    # Create a copy of args with modified parameters for high-temperature MD
    high_temp_args = copy.deepcopy(args)

    # Modify parameters for high-temperature benchmark
    # High temperature (1000 K)
    high_temp_args.temp = 1000.0

    # Longer simulation (100 ps at 2 fs timestep)
    high_temp_args.n_steps = 50000

    # Larger supercell for stability
    high_temp_args.supercell_size = [4, 4, 5]

    # Use high-temp specific directory
    high_temp_args.output_dir = benchmark_dir

    # Indicate high-energy MD mode
    high_temp_args.mode = 'high_energy'

    mdb_b_ut.custom_print(
        f'Running high-temperature MD at {high_temp_args.temp} K', 'info'
    )
    mdb_b_ut.custom_print(
        f'Using {high_temp_args.n_steps} steps with supercell '
        f'{high_temp_args.supercell_size}',
        'info',
    )

    # Call the existing energy MD benchmark with modified parameters
    run_energy_md_benchmark(high_temp_args, model_paths)


def coexistence_structure_melting_point(
    metal: str,
    benchmark_dir: pl.Path,
    calculator,
    config_dict: dict,
) -> pl.Path:
    """
    Prepare a solid-liquid coexistence structure using proper equilibration.

    This function implements the multi-step preparation process:
    1. Equilibrate solid phase at T < T_melt using NPT
    2. Take the equilibrated cell, and heat at T > T_melt with NPAT to get liquid
    3. Combine solid and liquid phases into coexistence structure, run short NPAT
    4. Relax interfaces using NVE

    Returns path to the final coexistence structure.
    """
    # Check if final structure already exists
    final_coexistence_path = benchmark_dir / '4_coexistence_structure_final.xyz'
    if final_coexistence_path.exists():
        mdb_b_ut.custom_print(
            f'Using existing prepared coexistence structure: {final_coexistence_path}',
            'info',
        )
        return final_coexistence_path

    mdb_b_ut.custom_print(
        'Preparing coexistence structure through multi-step process...', 'info'
    )

    # Parsing settings dict
    supercell_path = config_dict.get('supercell_path')
    supercell_size = config_dict.get('supercell_size')
    solid_temp_K = config_dict.get('solid_temp_K')
    liquid_temp_K = config_dict.get('liquid_temp_K')

    # Phase 1: Prepare equilibrated solid phase
    solid_path = benchmark_dir / '1_equilibrated_solid.xyz'
    step_1_nsteps = 333334
    step_1_tstep_fs = 3.0

    if not solid_path.exists():
        mdb_b_ut.custom_print('Phase 1: Equilibrating solid phase...', 'info')

        # Create initial solid structure
        if supercell_path is None:
            solid_bulk = bulk(metal, cubic=True)
            solid_bulk.pbc = True
            solid_supercell = solid_bulk.repeat(supercell_size)
            mdb_b_ut.custom_print(
                f'Initial solid supercell created with {len(solid_supercell)} atoms',
                'debug',
            )
        else:
            solid_supercell = ase_read(supercell_path, format='extxyz')
            solid_supercell.pbc = True
            mdb_b_ut.custom_print(
                f"Initial solid supercell read from '{supercell_path}' "
                f'with {len(solid_supercell)} atoms.',
            )
        solid_supercell.calc = calculator
        solid_supercell.write(
            benchmark_dir / 'init_equilibrated_solid.xyz', format='extxyz'
        )

        # Initialize velocities for solid temperature
        MaxwellBoltzmannDistribution(solid_supercell, temperature_K=solid_temp_K)

        # 1. Equilibrate solid using NPT at the solid temperature.
        solid_npt = NPT(
            solid_supercell,
            timestep=step_1_tstep_fs * units.fs,
            temperature_K=solid_temp_K,
            externalstress=1.0 * units.bar,
            ttime=50 * units.fs,
            pfactor=75 * units.fs**2,
            trajectory=benchmark_dir / '2_melted_liquid.traj',
            loginterval=10,
        )

        mdb_b_ut.custom_print(
            f'Equilibrating solid at {solid_temp_K} K'
            f' for {step_1_nsteps * step_1_tstep_fs * 0.001:.1f} ps...',
            'info',
        )

        solid_npt.attach(
            mdb_al_ut.manual_progress_display,
            interval=1000,
            dyn=solid_npt,
        )
        solid_npt.run(step_1_nsteps)

        # Save equilibrated solid
        solid_supercell.write(solid_path, format='extxyz')
        mdb_b_ut.custom_print(f'Equilibrated solid saved to {solid_path}', 'done')
    else:
        solid_supercell = ase_read(solid_path, format='extxyz')
        mdb_b_ut.custom_print(
            f'Phase 1: Loaded existing equilibrated solid from {solid_path}', 'done'
        )

    print()

    # Phase 2: Prepare equilibrated liquid phase
    melted_liquid_path = benchmark_dir / '2_melted_liquid.xyz'

    stage_2_nsteps = 333334
    stage_2_timestep_fs = 3.0

    mdb_b_ut.custom_print('Phase 2: Creating and melting liquid phase...', 'info')

    # Perform NPAT melting and equilibration
    if melted_liquid_path.exists():
        mdb_b_ut.custom_print(
            f'Using existing intermediate structure: {melted_liquid_path}',
            'info',
        )
        liquid_supercell = ase_read(
            melted_liquid_path,
            format='extxyz',
        )
    else:
        liquid_supercell = ase_read(solid_path, format='extxyz')
        mdb_b_ut.custom_print(
            f'Melting liquid part at {liquid_temp_K:.1f} K '
            f'for {stage_2_nsteps * stage_2_timestep_fs * 0.001} ps...',
            'info',
        )
        # Heat to high temperature for melting
        MaxwellBoltzmannDistribution(liquid_supercell, temperature_K=liquid_temp_K)

        # Melt using NPAT
        liquid_supercell.calc = calculator
        liquid_npt_melt = NPT(
            liquid_supercell,
            timestep=stage_2_timestep_fs * units.fs,
            temperature_K=liquid_temp_K,
            ttime=25 * units.fs,
            # pfactor=ptime^2*B (B = Bulk Modulus)
            pfactor=(75 * units.fs**2) * 100,
            externalstress=(1.01325 * units.bar),  # 1 atm in bar
            mask=(0, 0, 1),
            trajectory=benchmark_dir / '2_melted_liquid.traj',
            loginterval=10,
        )

        liquid_npt_melt.attach(
            mdb_al_ut.manual_progress_display,
            interval=1000,
            dyn=liquid_npt_melt,
        )

        # Run melting
        liquid_npt_melt.run(stage_2_nsteps)
        liquid_supercell.write(melted_liquid_path, format='extxyz')

        print()

    cooled_liquid_path = benchmark_dir / '3_1_cooled_liquid.xyz'
    coexistence_initial_path = benchmark_dir / '3_2_coexistence_initial.xyz'
    coexistence_relaxed_path = benchmark_dir / '3_3_coexistence_relaxed.xyz'
    # low_energy_liquid_path = benchmark_dir / '2_low_energy_liquid.xyz'
    mdb_b_ut.custom_print('Phase 3: Cooling down liquid phase...', 'info')

    if not coexistence_relaxed_path.exists():
        # 3.1 Cool from liquid_temp_K to solid_temp_K using NPAT
        # Use a temperature ramp do decrease temperature continously
        # This is to lower the energy of the liquid, while keeping liquid-like structure
        # so that it won't break the structure of the solid later

        if not cooled_liquid_path.exists():
            # Make a short temperature ramp to raise T from solid_temp_K to
            # liquid temp_K in 3 ps
            num_steps_ramp = 1000
            timestep_ramp_fs = 3.0
            liquid_supercell.pbc = True

            MaxwellBoltzmannDistribution(
                liquid_supercell,
                temperature_K=solid_temp_K,
            )

            mdb_b_ut.custom_print('Heating solid to liquid temperature...', 'info')

            liquid_supercell.calc = calculator
            liquid_nvt_heating_ramp = NPT(
                liquid_supercell,
                timestep=timestep_ramp_fs * units.fs,
                temperature_K=solid_temp_K,
                ttime=25 * units.fs,
                # pfactor=ptime^2*B (B = Bulk Modulus)
                pfactor=(75 * units.fs**2) * 100,
                externalstress=(1.01325 * units.bar),  # 1 atm in bar
                mask=(0, 0, 1),
                trajectory=benchmark_dir / '3_1_heating_liquid.traj',
                loginterval=10,
            )

            liquid_nvt_heating_ramp.attach(
                mdb_al_ut.manual_progress_display,
                interval=10,
                dyn=liquid_nvt_heating_ramp,
            )

            T_list_heating_ramp = []
            liquid_nvt_heating_ramp.attach(
                interval=1,
                function=mdb_al_ut.md_apply_temperature_ramp,
                dyn=liquid_nvt_heating_ramp,
                total_steps=num_steps_ramp,
                T_start=solid_temp_K,
                T_end=liquid_temp_K,
                T_list=T_list_heating_ramp,
            )

            # Run cooling
            liquid_nvt_heating_ramp.run(num_steps_ramp)

            # 100 ps cooling
            num_steps_cool = 33334
            timestep_cool_fs = 3.0

            liquid_supercell.pbc = True

            mdb_b_ut.custom_print(
                f'Reducing energy of liquid by changing T from {liquid_temp_K} K to '
                f'{solid_temp_K} K for '
                f'{num_steps_cool * timestep_cool_fs * 0.001} ps...',
                'info',
            )
            T_list = []
            liquid_supercell.calc = calculator
            liquid_nvt_ramp = NPT(
                liquid_supercell,
                timestep=timestep_cool_fs * units.fs,
                temperature_K=liquid_temp_K,
                ttime=25 * units.fs,
                # pfactor=ptime^2*B (B = Bulk Modulus)
                pfactor=(75 * units.fs**2) * 100,
                externalstress=(1.01325 * units.bar),  # 1 atm in bar
                mask=(0, 0, 1),
                trajectory=benchmark_dir / '3_1_cooling_liquid.traj',
                loginterval=100,
            )

            liquid_nvt_ramp.attach(
                interval=1,
                function=mdb_al_ut.md_apply_temperature_ramp,
                dyn=liquid_nvt_ramp,
                total_steps=num_steps_cool,
                T_start=liquid_temp_K,
                T_end=solid_temp_K,
                T_list=T_list,
            )

            liquid_nvt_ramp.attach(
                mdb_al_ut.manual_progress_display,
                interval=1000,
                dyn=liquid_nvt_ramp,
            )

            # Run cooling
            liquid_nvt_ramp.run(num_steps_cool)

            # Save equilibrated liquid
            liquid_supercell.write(cooled_liquid_path, format='extxyz')
            mdb_b_ut.custom_print(
                f'Saved cooled liquid to {cooled_liquid_path}', 'info'
            )
        else:
            liquid_supercell = ase_read(cooled_liquid_path, format='extxyz')
            mdb_b_ut.custom_print(
                f'Loaded existing cooled liquid from {cooled_liquid_path}', 'info'
            )

        # 3.2 Join solid and liquid phase in same cell,
        # and run 10 steps at solid_temp_K

        # Create the coexistence structure by combining solid and liquid
        coexistence_atoms = solid_supercell.copy()

        mdb_b_ut.custom_print(
            'Combining solid and liquid phases into one cell...',
            'info',
        )

        # Add the z length of the liquid cell to the solid cell to make it fit
        gap = 1.75
        new_cell = solid_supercell.get_cell().copy()
        new_cell[2, 2] += liquid_supercell.get_cell()[2, 2] + gap

        coexistence_atoms.set_cell(new_cell, scale_atoms=False)

        # Add the shifted liquid atoms to the coexistence structure
        liquid_atoms_shifted = liquid_supercell.copy()

        # Wrap coordinates
        liquid_atoms_shifted.wrap()

        # Translate the liquid atoms to sit on top of the solid slab.
        # The starting z-position for the liquid will be the z-height of the solid cell.
        liquid_atoms_shifted.translate([0, 0, solid_supercell.get_cell()[2, 2]])

        # Add the shifted liquid atoms to the coexistence structure
        coexistence_atoms.extend(liquid_atoms_shifted)

        # The new structure is now in liquid_supercell for the next steps
        liquid_supercell = coexistence_atoms
        liquid_supercell.pbc = True

        # Save the combined structure for inspection
        liquid_supercell.write(coexistence_initial_path, format='extxyz')
        mdb_b_ut.custom_print('Combined solid and liquid phases into one cell.', 'info')

        # 3.3 Relax the combined structure with short NPT run at solid_temp_K
        num_steps_relax = 10
        timestep_relax_fs = 2.0
        T_mult = 1.0

        liquid_supercell.calc = calculator
        cool_low_energy_liquid = NPT(
            liquid_supercell,
            timestep=timestep_relax_fs * units.fs,
            temperature_K=solid_temp_K * T_mult,
            ttime=25 * units.fs,
            # pfactor=ptime^2*B (B = Bulk Modulus)
            pfactor=(75 * units.fs**2) * 100,
            externalstress=(1.01325 * units.bar),  # 1 atm in bar
            mask=(0, 0, 1),
        )

        cool_low_energy_liquid.attach(
            mdb_al_ut.manual_progress_display,
            interval=250,
            dyn=cool_low_energy_liquid,
        )

        cool_low_energy_liquid.run(num_steps_relax)

        # Save equilibrated liquid
        liquid_supercell.write(coexistence_relaxed_path, format='extxyz')
        mdb_b_ut.custom_print(
            f'Saved equilibrated coexistence to {coexistence_relaxed_path}', 'info'
        )

    else:
        liquid_supercell = ase_read(coexistence_relaxed_path, format='extxyz')
        mdb_b_ut.custom_print(
            'Phase 2 and 3: Loaded equilibrated coexistence from '
            f'{coexistence_relaxed_path}',
            'info',
        )

    print()

    # Phase 4: Coexistence of liquid and solid phases with NVE
    mdb_b_ut.custom_print('Phase 4: Coexistence of liquid and solid (NVE)...', 'info')

    nve_initial_T_test_K = config_dict.get('nve_initial_T_test_K')
    mdb_b_ut.custom_print(
        f'Rescaling velocities to a starting T of {nve_initial_T_test_K} K', 'info'
    )
    MaxwellBoltzmannDistribution(
        liquid_supercell, temperature_K=nve_initial_T_test_K, force_temp=True
    )
    mdb_b_ut.custom_print(
        'Velocities rescaled. '
        f'Actual starting T: {liquid_supercell.get_temperature():.1f} K',
        'info',
    )

    # 10 ps at 1 fs timestep
    n_steps_nve = 333334
    timestep_nve_fs = 3.0

    # Use NVE dynamics to relax interfaces at fixed volume
    liquid_supercell.calc = calculator
    interface_relaxer = VelocityVerlet(
        liquid_supercell,
        timestep=timestep_nve_fs * units.fs,  # Smaller timestep for stability
        trajectory=benchmark_dir
        / f'4_1_nve_{liquid_supercell.get_temperature():.1f}K.traj',
        loginterval=1000,
    )

    mdb_b_ut.custom_print(
        f'Relaxing interface (NVE) for '
        f'{n_steps_nve * timestep_nve_fs * 0.001:.1f} ps...',
        'info',
    )

    nve_log_path = (
        benchmark_dir / f'4_1_nve_{liquid_supercell.get_temperature():.1f}K.log'
    )
    interface_relaxer.attach(
        MDLogger(
            dyn=interface_relaxer,
            atoms=liquid_supercell,
            logfile=nve_log_path,
            header=True,
            stress=False,
            peratom=False,
            mode='a',
        ),
        interval=100,
    )

    interface_relaxer.run(n_steps_nve)

    # Save final prepared structure
    liquid_supercell.write(final_coexistence_path, format='extxyz')

    mdb_b_ut.custom_print(
        f'Final coexistence structure saved to {final_coexistence_path}', 'info'
    )

    md_log = np.loadtxt(
        nve_log_path,
        skiprows=1,
        dtype={
            'names': ('time_ps', 'e_tot_eV', 'e_pot_eV', 'e_kin_eV', 'temp_K'),
            'formats': (np.float32, np.float32, np.float32, np.float32, np.float32),
        },
    )
    step_4_temperatures_K = md_log['temp_K']
    potential_energies = md_log['e_pot_eV']
    total_energies = md_log['e_tot_eV']

    average_T_K = np.average(
        step_4_temperatures_K[int(step_4_temperatures_K.shape[0] * 0.25) :]
    )

    mdb_b_ut.custom_print(
        f'Average temperature during final interface relaxation: {average_T_K:.1f} K'
        f' ({average_T_K - 273.15:.1f} °C)',
        'done',
    )

    # Create multipanel figure with temperature, potential energy, and total energy
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Temperature vs timestep
    ax = axes[0]
    ax.plot(
        range(step_4_temperatures_K.shape[0]),
        step_4_temperatures_K,
        'b-',
        linewidth=1.5,
    )
    ax.axhline(
        y=average_T_K,
        color='k',
        linestyle='--',
        linewidth=2,
        label=f'Average: {average_T_K:.1f} K',
    )
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature during Interface Relaxation')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 2: Potential energy vs timestep
    ax = axes[1]
    ax.plot(range(len(potential_energies)), potential_energies, 'r-', linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Potential Energy (eV)')
    ax.set_title('Potential Energy during Interface Relaxation')
    ax.grid(True, alpha=0.3)

    # Panel 3: Total energy vs timestep
    ax = axes[2]
    ax.plot(range(len(total_energies)), total_energies, 'g-', linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Total Energy (eV)')
    ax.set_title('Total Energy during Interface Relaxation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the multipanel plot
    temp_plot_path = benchmark_dir / 'coexistence_interface_relaxation_multipanel'
    plt.savefig(temp_plot_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(temp_plot_path.with_suffix('.svg'), dpi=300, bbox_inches='tight')
    plt.close()

    return average_T_K, final_coexistence_path


def _get_lammps_calculator(
    mace_model_path: pl.Path,
    lammps_commands: list[str],
    metal_symbol: str,
    tmp_dir: pl.Path,
) -> LAMMPS:
    """
    Helper function to configure the ASE LAMMPS calculator for MACE.

    Args:
        mace_model_path: Path to the MACE model file.
        lammps_commands: A list of LAMMPS commands to execute.
        metal_symbol: The chemical symbol of the metal (e.g., 'Ni').
        tmp_dir: Temporary directory for LAMMPS input/output.

    Returns
    -------
        An ASE LAMMPS calculator instance.
    """
    # Ensure the directory for LAMMPS temp files exists
    tmp_dir.mkdir(exist_ok=True)

    parameters = {
        'pair_style': 'mace',
        'pair_coeff': f'* * {mace_model_path.resolve()} {metal_symbol}',
        'mass': '1 *',  # Mass will be inferred from ASE atoms
        'thermo_style': 'custom step temp pe etotal press vol',
        'thermo': '100',  # Print thermo data every 100 steps
    }

    # ASE's LAMMPS calculator needs a unique label for each run to avoid
    # overwriting temp files in the same directory.
    label = f'lammps_{np.random.randint(1e9)}'

    lammps = LAMMPS(
        label=label,
        tmp_dir=str(tmp_dir),
        files=[str(mace_model_path.resolve())],
        specorder=[metal_symbol],
        lammps_header=[
            'units metal',
            'atom_style atomic',
            'atom_modify map array',
        ],
        # The 'run 0' is a trick to make ASE write the data file without
        # starting a long run, allowing us to add custom velocity commands.
        lammps_main=['run 0'],
        # Keep log file for parsing temperature data in the NVE step
        keep_log_file=True,
        # Post-commands are executed after the data file is written
        # but before the main run command.
        post_changebox_commands=lammps_commands,
        **parameters,
    )
    return lammps


def prepare_coexistence_structure_lammps(
    metal: str,
    supercell_size: list[int],
    benchmark_dir: pl.Path,
    mace_model_path: pl.Path,
    solid_temp_K: float,
    liquid_temp_K: float,
) -> pl.Path:
    """
    Prepare a solid-liquid coexistence structure using LAMMPS for dynamics.

    This function implements the multi-step preparation process:
    1. Equilibrate solid phase at T < T_melt using NPT.
    2. Melt a copy of the solid at T > T_melt.
    3. Cool the liquid down to the solid temperature.
    4. Combine solid and liquid phases into a coexistence structure.
    5. Relax the interface using NVE dynamics.

    Returns
    -------
        Path to the final coexistence structure.
    """
    final_coexistence_path = benchmark_dir / 'coexistence_structure_prepared.xyz'
    if final_coexistence_path.exists():
        mdb_b_ut.custom_print(
            f'Using existing prepared coexistence structure: {final_coexistence_path}',
            'info',
        )
        return final_coexistence_path

    mdb_b_ut.custom_print('Preparing coexistence structure via LAMMPS...', 'info')

    # --- Phase 1: Prepare equilibrated solid phase ---
    solid_path = benchmark_dir / 'equilibrated_solid.xyz'
    if not solid_path.exists():
        mdb_b_ut.custom_print('Phase 1: Equilibrating solid phase (NPT)...', 'info')
        solid_bulk = bulk(metal, cubic=True)
        solid_supercell = solid_bulk.repeat(supercell_size)
        MaxwellBoltzmannDistribution(solid_supercell, temperature_K=solid_temp_K)

        step_1_nsteps = 33334
        step_1_tstep_fs = 3.0
        ttime_fs = 100.0 * step_1_tstep_fs
        ptime_fs = 1000.0 * step_1_tstep_fs

        mdb_b_ut.custom_print(
            f'Equilibrating solid at {solid_temp_K} K for '
            f'{step_1_nsteps * step_1_tstep_fs * 0.001:.1f} ps...',
            'info',
        )

        solid_commands = [
            f'timestep {step_1_tstep_fs / 1000.0}',  # Timestep in ps
            f'fix 1 all npt temp {solid_temp_K} {solid_temp_K} {ttime_fs / 1000.0} '
            f'iso 1.0 1.0 {ptime_fs / 1000.0}',
            f'run {step_1_nsteps}',
        ]
        solid_supercell.calc = _get_lammps_calculator(
            mace_model_path, solid_commands, metal, benchmark_dir
        )
        # This triggers the LAMMPS run
        solid_supercell.get_potential_energy()

        ase_write(solid_path, solid_supercell, format='extxyz')
        mdb_b_ut.custom_print(f'Equilibrated solid saved to {solid_path}', 'done')
    else:
        solid_supercell = ase_read(solid_path, format='extxyz')
        mdb_b_ut.custom_print(
            f'Phase 1: Loaded existing solid from {solid_path}', 'done'
        )
    print()

    # --- Phase 2: Prepare equilibrated liquid phase ---
    cooled_liquid_path = benchmark_dir / 'equilibrated_liquid.xyz'
    if not cooled_liquid_path.exists():
        # --- 2a: Melt the solid ---
        mdb_b_ut.custom_print('Phase 2a: Melting the structure (NPT)...', 'info')
        liquid_supercell = ase_read(solid_path, format='extxyz')
        MaxwellBoltzmannDistribution(liquid_supercell, temperature_K=liquid_temp_K)

        step_2_nsteps = 33334
        step_2_tstep_fs = 3.0
        ttime_fs = 100.0 * step_2_tstep_fs
        ptime_fs = 1000.0 * step_2_tstep_fs

        mdb_b_ut.custom_print(
            f'Melting at {liquid_temp_K} K for '
            f'{step_2_nsteps * step_2_tstep_fs * 0.001:.1f} ps...',
            'info',
        )

        melt_commands = [
            f'timestep {step_2_tstep_fs / 1000.0}',
            f'fix 1 all npt temp {liquid_temp_K} {liquid_temp_K} {ttime_fs / 1000.0} '
            f'iso 1.0 1.0 {ptime_fs / 1000.0}',
            f'run {step_2_nsteps}',
        ]
        liquid_supercell.calc = _get_lammps_calculator(
            mace_model_path, melt_commands, metal, benchmark_dir
        )
        liquid_supercell.get_potential_energy()

        # --- 2b: Cool the liquid ---
        mdb_b_ut.custom_print('Phase 2b: Cooling the liquid (NPT)...', 'info')
        num_steps_cool = 11334
        tstep_cool_fs = 3.0
        mdb_b_ut.custom_print(
            f'Cooling from {liquid_temp_K} K to {solid_temp_K} K for '
            f'{num_steps_cool * tstep_cool_fs * 0.001:.1f} ps...',
            'info',
        )

        cool_commands = [
            f'timestep {tstep_cool_fs / 1000.0}',
            f'fix 1 all npt temp {liquid_temp_K} {solid_temp_K} {ttime_fs / 1000.0} '
            f'iso 1.0 1.0 {ptime_fs / 1000.0}',
            f'run {num_steps_cool}',
        ]
        liquid_supercell.calc = _get_lammps_calculator(
            mace_model_path, cool_commands, metal, benchmark_dir
        )
        liquid_supercell.get_potential_energy()
        ase_write(cooled_liquid_path, liquid_supercell, format='extxyz')
    else:
        liquid_supercell = ase_read(cooled_liquid_path, format='extxyz')
        mdb_b_ut.custom_print(
            f'Phase 2: Loaded existing liquid from {cooled_liquid_path}', 'done'
        )
    print()

    # --- Phase 3: Combine solid and liquid and relax briefly ---
    mdb_b_ut.custom_print('Phase 3: Assembling coexistence structure...', 'info')
    coexistence_atoms = solid_supercell.copy()
    cell = coexistence_atoms.get_cell()
    cell[2, 2] *= 2.0
    coexistence_atoms.set_cell(cell, scale_atoms=False)

    liquid_atoms_shifted = liquid_supercell.copy()
    liquid_atoms_shifted.translate([0, 0, cell[2, 2] / 2.0])
    coexistence_atoms.extend(liquid_atoms_shifted)
    ase_write(
        benchmark_dir / 'coexistence_structure_combined.xyz',
        coexistence_atoms,
        format='extxyz',
    )

    mdb_b_ut.custom_print(
        'Running brief NPAT relaxation on combined structure...', 'info'
    )
    npt_z_commands = [
        'timestep 0.001',  # 1 fs
        f'fix 1 all npt temp {solid_temp_K} {solid_temp_K} 0.1 '
        'x NULL NULL y NULL NULL z 1.0 1.0 1.0',
        'run 1000',  # 1 ps relaxation
    ]
    coexistence_atoms.calc = _get_lammps_calculator(
        mace_model_path, npt_z_commands, metal, benchmark_dir
    )
    coexistence_atoms.get_potential_energy()
    print()

    # --- Phase 4: Coexistence of liquid and solid phases with NVE ---
    mdb_b_ut.custom_print('Phase 4: Coexistence interface relaxation (NVE)...', 'info')
    n_steps_nve = 33334
    timestep_nve_fs = 3.0
    mdb_b_ut.custom_print(
        f'Relaxing interface (NVE) for '
        f'{n_steps_nve * timestep_nve_fs * 0.001:.1f} ps...',
        'info',
    )

    nve_commands = [
        f'timestep {timestep_nve_fs / 1000.0}',
        'fix 1 all nve',
        f'run {n_steps_nve}',
    ]
    # Re-use the calculator object to access the log file after the run
    nve_calc = _get_lammps_calculator(
        mace_model_path, nve_commands, metal, benchmark_dir
    )
    coexistence_atoms.calc = nve_calc
    coexistence_atoms.get_potential_energy()

    # Parse the LAMMPS log file for temperature data
    log_file = pl.Path(nve_calc.tmpdir) / nve_calc.label / 'log.lammps'
    log_data = np.genfromtxt(log_file, comments='#', skip_header=1, invalid_raise=False)
    step_4_temperatures_K = log_data[:, 1]  # Temperature is the 2nd column

    np.savetxt(
        benchmark_dir / 'coexistence_interface_relaxation_temperature.txt',
        np.array(step_4_temperatures_K),
        header='Temperature (K) during interface relaxation',
    )

    ase_write(final_coexistence_path, coexistence_atoms, format='extxyz')
    mdb_b_ut.custom_print(
        f'Final coexistence structure saved to {final_coexistence_path}', 'info'
    )

    # Average over the last 75% of the run
    start_index = int(len(step_4_temperatures_K) * 0.25)
    average_T_K = np.mean(step_4_temperatures_K[start_index:])
    print()
    mdb_b_ut.custom_print(
        f'Average temperature during NVE relaxation: {average_T_K:.1f} K '
        f'({average_T_K - 273.15:.1f} °C)',
        'done',
    )

    return final_coexistence_path


def run_gsfe_benchmark(args, model_paths: list[pl.Path]):
    """Calculates and plots the Generalized Stacking Fault Energy (GSFE) curve."""
    mdb_b_ut.custom_print('Running GSFE Benchmark', 'info')
    benchmark_dir = args.output_dir / 'gsfe'
    benchmark_dir.mkdir(exist_ok=True)

    # 1. Create a slab oriented with the (111) plane.
    # 2. Define the slip path along <112>.

    # 3. For each model:
    #    - For a series of displacements along the path:
    #      - Displace the top half of the slab.
    #      - Relax the atomic positions, constraining them to move only perpendicular
    #        to the slip plane.
    #      - Calculate the energy.
    #    - The GSFE is the energy difference per unit area relative to the perfect slab.

    # 4. Plot the GSFE curves for each model.
    mdb_b_ut.custom_print('GSFE benchmark not implemented yet.', 'warn')


def run_learning_curves_benchmark(args, model_paths: list[pl.Path]):
    """Plots learning curves (e.g., test RMSE vs. number of DFT calls)."""
    mdb_b_ut.custom_print('Running Learning Curves Benchmark', 'info')
    benchmark_dir = args.output_dir / 'learning_curves'
    benchmark_dir.mkdir(exist_ok=True)

    # This benchmark only works with AiiDA PKs
    if not args.aiida_pks:
        mdb_b_ut.custom_print(
            'No AiiDA PKs provided. This benchmark requires AiiDA workchain PKs.',
            'warn',
        )
        return

    # Results storage
    learning_curves_data = {}

    # 1. Use the get_loop_report() function from report_utils.py to get
    # the AL workchain data.
    for pk in args.aiida_pks:
        try:
            mdb_b_ut.custom_print(
                f'Processing learning curve for workchain {pk}...', 'info'
            )

            # Get loop report data
            title, al_loop_node, ini_db_size, model_acc_multiplier, stats_dict = (
                get_loop_report(loop_id=pk)
            )

            # Extract iteration data
            iterations = []
            train_db_sizes = []
            energy_rmse = []
            force_rmse = []

            for it_key in sorted(stats_dict.keys()):
                it_data = stats_dict[it_key]
                if (
                    it_data is None
                    or it_data['train_db_size'] is None
                    or it_data['mace_e'] is None
                ):
                    continue

                iterations.append(it_data['it_idx'])
                train_db_sizes.append(it_data['train_db_size'])

                # Handle None values for RMSE (e.g., iteration 0)
                e_rmse = it_data.get('mace_e')
                f_rmse = it_data.get('mace_f')
                energy_rmse.append(e_rmse if e_rmse is not None else 0.0)
                force_rmse.append(f_rmse if f_rmse is not None else 0.0)

            # Store results using the run name as identifier
            run_name = title if title else f'workchain_{pk}'
            learning_curves_data[run_name] = {
                'pk': pk,
                'iterations': iterations,
                'train_db_sizes': train_db_sizes,
                'energy_rmse': energy_rmse,
                'force_rmse': force_rmse,
                'ini_db_size': ini_db_size,
            }

        except Exception as e:
            mdb_b_ut.custom_print(f'Failed to process workchain {pk}: {e}', 'error')
            continue

    # Save results to file
    if learning_curves_data:
        results_file = benchmark_dir / 'learning_curves.json'
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for run_name, data in learning_curves_data.items():
                json_data[run_name] = {
                    'pk': data['pk'],
                    'iterations': data['iterations'],
                    'train_db_sizes': data['train_db_sizes'],
                    'energy_rmse': data['energy_rmse'],
                    'force_rmse': data['force_rmse'],
                    'ini_db_size': data['ini_db_size'],
                }
            json.dump(json_data, f, indent=2)

        # Store plot data for final multi-panel figure
        mdb_b_ut.set_plot_data(
            'learning_curves',
            {
                'type': 'learning_curves',
                'data': learning_curves_data,
                'title': 'Active Learning Curves',
            },
        )

        # Print summary
        mdb_b_ut.custom_print('Learning Curves Summary:', 'info')
        for run_name, data in learning_curves_data.items():
            final_energy = data['energy_rmse'][-1] if data['energy_rmse'] else 0
            final_force = data['force_rmse'][-1] if data['force_rmse'] else 0
            final_db_size = data['train_db_sizes'][-1] if data['train_db_sizes'] else 0
            mdb_b_ut.custom_print(
                f'  {run_name}: {len(data["iterations"])} iterations, '
                f'final DB size: {final_db_size}, '
                f'final E RMSE: {final_energy:.3f} meV/atom, '
                f'final F RMSE: {final_force:.3f} meV/Å',
                'empty',
            )

        mdb_b_ut.custom_print(f'Results saved to {results_file}', 'info')
    else:
        mdb_b_ut.custom_print('No learning curve results to save.', 'warn')


def run_evaluate_database(args, model_paths: list[pl.Path]):
    """
    Evaluate MLIP models against a user-provided structure database.

    This benchmark loads structures from a database file and compares
    MLIP predictions with reference values stored in the structures.
    Reference energies should be in atoms.info['REF_energy'] and
    reference forces in atoms.arrays['REF_forces'].
    """
    mdb_b_ut.custom_print('Running Database Evaluation Benchmark', 'info')

    if not hasattr(args, 'database_path') or not args.database_path:
        mdb_b_ut.custom_print(
            'No database path provided. Use --database_path to specify '
            'the database file.',
            'error',
        )
        return

    if not args.database_path.exists():
        mdb_b_ut.custom_print(f'Database file not found: {args.database_path}', 'error')
        return

    benchmark_dir = args.output_dir / 'evaluate_database'
    benchmark_dir.mkdir(exist_ok=True)

    # Load database structures
    mdb_b_ut.custom_print(f'Loading database from: {args.database_path}', 'info')
    try:
        structures = ase_read(args.database_path, index=':')
        mdb_b_ut.custom_print(f'Loaded {len(structures)} structures', 'info')
    except Exception as e:
        mdb_b_ut.custom_print(f'Failed to load database: {e}', 'error')
        return

    # Validate reference data
    valid_structures = []
    for i, atoms in enumerate(structures):
        has_ref_energy = 'REF_energy' in atoms.info
        has_ref_forces = 'REF_forces' in atoms.arrays

        if not has_ref_energy:
            mdb_b_ut.custom_print(
                f'Structure {i}: Missing REF_energy in info. Skipping.', 'warn'
            )
            continue

        if not has_ref_forces:
            mdb_b_ut.custom_print(
                f'Structure {i}: Missing REF_forces in arrays. Skipping.', 'warn'
            )
            continue

        valid_structures.append(atoms)

    if not valid_structures:
        mdb_b_ut.custom_print(
            'No valid structures found with reference data. Exiting.', 'error'
        )
        return

    mdb_b_ut.custom_print(
        f'{len(valid_structures)} structures contain reference data', 'info'
    )

    # Results storage
    results = {}

    # Evaluate each model
    for _, model_path in enumerate(model_paths):
        model_name = mdb_b_ut.get_model_display_name(model_path)
        mdb_b_ut.custom_print(f'Evaluating model: {model_name}', 'info')

        # Skip if already calculated
        results_file = benchmark_dir / f'{model_name}_database_evaluation.json'
        if results_file.exists():
            mdb_b_ut.custom_print(
                f"Results for '{model_name}' already exist. Loading from file.", 'warn'
            )
            with open(results_file) as f:
                results[model_name] = json.load(f)
            continue

        try:
            # Set up calculator
            calculator = mdb_b_ut.create_calculator_for_model(
                model_path, device=args.device, dtype=args.dtype
            )

            # Storage for this model's results
            energy_errors = []
            force_errors = []
            ref_energies = []
            pred_energies = []
            structure_indices = []

            # Evaluate each structure
            for struct_idx, atoms in enumerate(valid_structures):
                # Make a copy to avoid modifying original
                test_atoms = atoms.copy()
                test_atoms.set_calculator(calculator)

                # Get reference values
                ref_energy = atoms.info['REF_energy']
                ref_forces = atoms.arrays['REF_forces']

                # Get MLIP predictions
                try:
                    pred_energy = test_atoms.get_potential_energy()
                    pred_forces = test_atoms.get_forces()

                    # Calculate errors
                    # Energy error per atom (in meV/atom)
                    energy_error = abs(pred_energy - ref_energy) / len(atoms) * 1000

                    # Force error (RMSE in eV/Å)
                    force_diff = pred_forces - ref_forces
                    force_error = np.sqrt(np.mean(force_diff**2))

                    # Store results
                    energy_errors.append(energy_error)
                    force_errors.append(force_error)
                    ref_energies.append(ref_energy)
                    pred_energies.append(pred_energy)
                    structure_indices.append(struct_idx)

                except Exception as e:
                    mdb_b_ut.custom_print(
                        f'Failed to evaluate structure {struct_idx}: {e}', 'warn'
                    )
                    continue

            if not energy_errors:
                mdb_b_ut.custom_print(
                    f'No successful evaluations for model {model_name}', 'error'
                )
                continue

            # Calculate statistics
            mean_energy_error = np.mean(energy_errors)
            std_energy_error = np.std(energy_errors)
            mean_force_error = np.mean(force_errors)
            std_force_error = np.std(force_errors)

            # Store results for this model
            model_results = {
                'n_structures_evaluated': len(energy_errors),
                'energy_errors_meV_per_atom': energy_errors,
                'force_errors_eV_per_A': force_errors,
                'ref_energies_eV': ref_energies,
                'pred_energies_eV': pred_energies,
                'structure_indices': structure_indices,
                'mean_energy_error_meV_per_atom': mean_energy_error,
                'std_energy_error_meV_per_atom': std_energy_error,
                'mean_force_error_eV_per_A': mean_force_error,
                'std_force_error_eV_per_A': std_force_error,
            }

            results[model_name] = model_results

            # Save individual results
            with open(results_file, 'w') as f:
                json.dump(model_results, f, indent=2)

            mdb_b_ut.custom_print(
                f'{model_name}: Evaluated {len(energy_errors)} structures. '
                f'Mean energy error: {mean_energy_error:.2f}±{std_energy_error:.2f} '
                f'meV/atom, Mean force error: {mean_force_error:.3f}±'
                f'{std_force_error:.3f} eV/Å',
                'done',
            )

        except Exception as e:
            mdb_b_ut.custom_print(
                f"Failed to evaluate database for '{model_name}': {e}", 'error'
            )
            continue

    # Save combined results and generate plots
    if results:
        # Save all results
        all_results_file = benchmark_dir / 'all_database_evaluations.json'
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate plots
        mdb_b_ut.custom_print('Generating evaluation plots...', 'info')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Color map for different models
        model_names = list(results.keys())
        colors = mdb_b_ut.get_model_colors_by_names(model_names)

        # Plot energy errors
        for i, (model_name, model_data) in enumerate(results.items()):
            struct_indices = model_data['structure_indices']
            energy_errors = model_data['energy_errors_meV_per_atom']

            ax1.plot(
                struct_indices,
                energy_errors,
                label=model_name,
                color=colors[i],
                marker='o',
                markersize=3,
                linewidth=1,
            )

        ax1.set_xlabel('Structure Index')
        ax1.set_ylabel('Energy Error (meV/atom)')
        ax1.set_title('Energy Prediction Errors vs Reference Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Plot force errors
        for i, (model_name, model_data) in enumerate(results.items()):
            struct_indices = model_data['structure_indices']
            force_errors = model_data['force_errors_eV_per_A']

            ax2.plot(
                struct_indices,
                force_errors,
                label=model_name,
                color=colors[i],
                marker='o',
                markersize=3,
                linewidth=1,
            )

        ax2.set_xlabel('Structure Index')
        ax2.set_ylabel('Force Error (eV/Å)')
        ax2.set_title('Force Prediction Errors vs Reference Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()

        # Save in both PNG and SVG formats
        base_path = benchmark_dir / 'database_evaluation_errors'
        png_path, svg_path = mdb_b_ut.save_plot_dual_format(base_path, dpi=300)

        mdb_b_ut.custom_print(f'Plot saved to {png_path}', 'done')
        mdb_b_ut.custom_print(f'Plot saved to {svg_path}', 'done')
        plt.close()

        # Store plot data for final multi-panel figure
        model_names = list(results.keys())
        mean_energy_errors = [
            results[name]['mean_energy_error_meV_per_atom'] for name in model_names
        ]
        mean_force_errors = [
            results[name]['mean_force_error_eV_per_A'] for name in model_names
        ]

        title_general = 'Database Evaluation Results'

        mdb_b_ut.set_plot_data(
            'database_evaluation',
            {
                'type': 'database_evaluation',
                'model_names': model_names,
                'mean_energy_errors': mean_energy_errors,
                'mean_force_errors': mean_force_errors,
                'results': results,
                'title': title_general + ' - RMSE',
            },
        )

        # Store detailed structure-by-structure plot data
        mdb_b_ut.set_plot_data(
            'database_evaluation_detailed',
            {
                'type': 'database_evaluation_detailed',
                'model_names': model_names,
                'results': results,
                'title': title_general + ' - Structure-by-Structure Errors',
            },
        )

        # Print summary
        mdb_b_ut.custom_print('Database Evaluation Summary:', 'info')
        for name in model_names:
            data = results[name]
            mdb_b_ut.custom_print(
                f'  {name}: {data["n_structures_evaluated"]} structures, '
                f'Energy RMSE: {data["mean_energy_error_meV_per_atom"]:.2f} meV/atom, '
                f'Force RMSE: {data["mean_force_error_eV_per_A"]:.3f} eV/Å',
                'empty',
            )

        mdb_b_ut.custom_print(f'Results saved to {all_results_file}', 'info')
    else:
        mdb_b_ut.custom_print('No database evaluation results to save.', 'warn')


def run_magic_cluster_benchmark(args, model_paths: list[pl.Path]):
    """
    Calculates and plots energies for magic number cluster structures.

    Magic number clusters are particularly stable cluster sizes that often
    correspond to closed-shell electronic configurations or geometric
    completeness (e.g., icosahedral or cuboctahedral structures).

    This benchmark validates whether MLIPs can reproduce the correct
    ordering and stability of these special cluster sizes.
    """
    mdb_b_ut.custom_print('Running Magic Cluster Benchmark', 'info')
    benchmark_dir = args.output_dir / 'magic_cluster'
    benchmark_dir.mkdir(exist_ok=True)

    # Define common magic numbers for different cluster types
    # These are well-known magic numbers for various Ih cluster geometries
    default_magic_numbers = [13, 19, 55, 147, 309, 561]

    # User can override with custom magic numbers or provide structures
    magic_numbers = getattr(args, 'magic_cluster_sizes', default_magic_numbers)

    # Load DFT reference energies if provided
    dft_references = {}
    dft_refs_arg = getattr(args, 'magic_cluster_dft_refs', None)
    if hasattr(args, 'magic_cluster_dft_refs') and dft_refs_arg:
        try:
            with open(dft_refs_arg) as f:
                dft_references = json.load(f)
                mdb_b_ut.custom_print(
                    f'Loaded DFT reference energies for '
                    f'{len(dft_references)} cluster sizes',
                    'info',
                )
                # Convert string keys to integers for consistency
                dft_references = {int(k): v for k, v in dft_references.items()}
        except Exception as e:
            mdb_b_ut.custom_print(
                f'Failed to load DFT reference energies: {e}', 'error'
            )
            dft_references = {}

    # Results storage
    results = {}

    # For each model, calculate cluster energies
    for _, model_path in enumerate(model_paths):
        model_name = mdb_b_ut.get_model_display_name(model_path)

        # Skip if already calculated
        results_file = benchmark_dir / f'{model_name}_magic_cluster_energies.json'
        if results_file.exists():
            mdb_b_ut.custom_print(
                f"Results for '{model_name}' already exist. Loading from file.", 'warn'
            )
            with open(results_file) as f:
                loaded_results = json.load(f)
                # Convert string keys back to integers for consistency
                if 'cluster_energies' in loaded_results:
                    loaded_results['cluster_energies'] = {
                        int(k): v for k, v in loaded_results['cluster_energies'].items()
                    }
                if 'cluster_energies_per_atom' in loaded_results:
                    loaded_results['cluster_energies_per_atom'] = {
                        int(k): v
                        for k, v in loaded_results['cluster_energies_per_atom'].items()
                    }
                results[model_name] = loaded_results

                # Debug: Print loaded cluster sizes for verification
                if 'cluster_energies' in loaded_results:
                    cluster_sizes = list(loaded_results['cluster_energies'].keys())
                    mdb_b_ut.custom_print(
                        f'Loaded cluster energies for sizes: {cluster_sizes}', 'debug'
                    )
            continue

        try:
            # Set up calculator
            calculator = mdb_b_ut.create_calculator_for_model(
                model_path, device=args.device, dtype=args.dtype
            )

            model_results = {
                'cluster_energies': {},
                'cluster_energies_per_atom': {},
                'magic_numbers': magic_numbers,
            }

            # Calculate energies for each magic number cluster
            for n_atoms in magic_numbers:
                mdb_b_ut.custom_print(
                    f'Calculating energy for {model_name} - {n_atoms} atom cluster...',
                    'debug',
                )

                # Create cluster structure
                cluster_structure_path = benchmark_dir / f'cluster_{n_atoms}_atoms.xyz'

                if cluster_structure_path.exists():
                    # Load existing structure
                    cluster = ase_read(cluster_structure_path, format='extxyz')
                    mdb_b_ut.custom_print(
                        f'Loaded existing cluster structure with {n_atoms} atoms',
                        'debug',
                    )
                else:
                    # Generate cluster structure using ASE
                    cluster = _generate_magic_cluster(
                        args.metal, n_atoms, benchmark_dir
                    )

                    # Verify we got the correct number of atoms
                    actual_atoms = len(cluster)
                    if actual_atoms != n_atoms:
                        mdb_b_ut.custom_print(
                            f'Error: Generated cluster has {actual_atoms} atoms, '
                            f'expected {n_atoms}. Skipping this cluster size.',
                            'error',
                        )
                        continue

                    cluster.write(cluster_structure_path, format='extxyz')
                    mdb_b_ut.custom_print(
                        f'Successfully generated cluster structure with '
                        f'{n_atoms} atoms',
                        'debug',
                    )

                # Set calculator and optimize structure
                cluster.set_calculator(calculator)

                # Relax the cluster structure
                optimizer = LBFGS(
                    cluster,
                    logfile=benchmark_dir / f'{model_name}_cluster_{n_atoms}_relax.log',
                )
                optimizer.run(fmax=0.01, steps=500)

                # Get final energy
                energy = cluster.get_potential_energy()
                energy_per_atom = energy / n_atoms

                # Save relaxed structure
                relaxed_path = (
                    benchmark_dir / f'{model_name}_cluster_{n_atoms}_relaxed.xyz'
                )
                cluster.write(relaxed_path, format='extxyz')

                # Store results
                model_results['cluster_energies'][n_atoms] = energy
                model_results['cluster_energies_per_atom'][n_atoms] = energy_per_atom

                mdb_b_ut.custom_print(
                    f'Cluster {n_atoms} atoms - {model_name}: {energy:.3f} eV '
                    f'({energy_per_atom:.3f} eV/atom)',
                    'debug',
                )

            results[model_name] = model_results

            # Save individual results
            with open(results_file, 'w') as f:
                json.dump(model_results, f, indent=2)

            mdb_b_ut.custom_print(
                f"Magic cluster energies calculated for '{model_name}'", 'done'
            )

        except Exception as e:
            mdb_b_ut.custom_print(
                f"Failed to calculate magic cluster energies for '{model_name}': {e}",
                'error',
            )
            continue

    # Save combined results and store plot data
    if results:
        # Debug: Print overall results structure
        mdb_b_ut.custom_print(f'Results loaded for {len(results)} models:', 'debug')
        for model_name, model_data in results.items():
            cluster_count = len(model_data.get('cluster_energies', {}))
            mdb_b_ut.custom_print(
                f'  {model_name}: {cluster_count} cluster sizes', 'debug'
            )

        # Save all results
        all_results_file = benchmark_dir / 'all_magic_cluster_energies.json'
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Prepare data for plotting
        model_names = list(results.keys())

        # Add DFT reference if available
        all_model_names = model_names.copy()
        if dft_references:
            all_model_names.append('DFT')

        # Prepare energy data for each cluster size
        cluster_energies_data = {}
        for n_atoms in magic_numbers:
            cluster_energies_data[n_atoms] = []

            # Add MLIP model energies
            for model_name in model_names:
                if n_atoms in results[model_name]['cluster_energies']:
                    cluster_data = results[model_name]['cluster_energies_per_atom']
                    energy_per_atom = cluster_data[n_atoms]
                    cluster_energies_data[n_atoms].append(energy_per_atom)
                else:
                    cluster_energies_data[n_atoms].append(None)  # Missing data

            # Add DFT reference if available
            if dft_references and n_atoms in dft_references:
                dft_energy_per_atom = dft_references[n_atoms] / n_atoms
                cluster_energies_data[n_atoms].append(dft_energy_per_atom)
            elif dft_references:
                # DFT data not available for this size
                cluster_energies_data[n_atoms].append(None)

        # Print cluster energies data for verification
        mdb_b_ut.custom_print('Cluster energies data for plotting:', 'debug')
        for n_atoms, energies in cluster_energies_data.items():
            non_none_count = sum(1 for e in energies if e is not None)
            mdb_b_ut.custom_print(
                f'  {n_atoms} atoms: {non_none_count}/{len(energies)} non-None values',
                'debug',
            )

        # Store plot data for final multi-panel figure
        mdb_b_ut.set_plot_data(
            'magic_cluster',
            {
                'type': 'magic_cluster',
                'model_names': all_model_names,
                'magic_numbers': magic_numbers,
                'cluster_energies_data': cluster_energies_data,
                'title': f'Magic Number Cluster Energies - {args.metal}',
                'ylabel': 'Energy per Atom (eV/atom)',
                'has_dft_reference': bool(dft_references),
            },
        )

        # Print summary
        mdb_b_ut.custom_print('Magic Cluster Energy Summary:', 'info')
        for n_atoms in magic_numbers:
            mdb_b_ut.custom_print(f'  Cluster size {n_atoms} atoms:', 'empty')
            for model_name in model_names:
                if n_atoms in results[model_name]['cluster_energies_per_atom']:
                    cluster_data = results[model_name]['cluster_energies_per_atom']
                    energy_per_atom = cluster_data[n_atoms]
                    mdb_b_ut.custom_print(
                        f'    {model_name}: {energy_per_atom:.3f} eV/atom', 'empty'
                    )
            if dft_references and n_atoms in dft_references:
                dft_energy_per_atom = dft_references[n_atoms] / n_atoms
                mdb_b_ut.custom_print(
                    f'    DFT: {dft_energy_per_atom:.3f} eV/atom', 'empty'
                )

        mdb_b_ut.custom_print(f'Results saved to {all_results_file}', 'info')
    else:
        mdb_b_ut.custom_print('No magic cluster results to save.', 'warn')


def gen_magic_number_list_chini(n_max: int = 25) -> list[float]:
    """Return a list of magic numbers using the Chini series.

    Ref: 10.1134/S0022476617070149
    """
    magic_numbers = []

    for n in range(1, n_max + 1):
        magic_number = ((10 * n**3 + 15 * n**2 + 11 * n) // 3) + 1
        magic_numbers.append(magic_number)

    return magic_numbers


def _generate_magic_cluster(metal: str, n_atoms: int, benchmark_dir: pl.Path):
    """
    Generate a cluster structure with the specified number of atoms.

    This function creates cluster structures using geometric algorithms
    suitable for magic number clusters, ensuring exact atom count.
    """
    import numpy as np
    from ase.build import bulk
    from ase.cluster import Icosahedron

    magic_numbers = gen_magic_number_list_chini()

    # Try different approaches to get exactly n_atoms
    cluster = None

    # For magic numbers, use the Icosahedron class
    cluster = Icosahedron(metal, noshells=magic_numbers.index(n_atoms) + 1)

    # Check if we got the right number of atoms from icosahedral generation
    if cluster is not None and len(cluster) != n_atoms:
        mdb_b_ut.custom_print(
            f'Icosahedral generation gave {len(cluster)} atoms, expected {n_atoms}. '
            f'Falling back to spherical construction.',
            'debug',
        )
        cluster = None

    # If icosahedral approach didn't work or isn't applicable, use spherical
    # construction
    if cluster is None:
        # Create a roughly spherical cluster by carving from bulk
        bulk_structure = bulk(metal, cubic=True)

        # Estimate required supercell size
        vol_per_atom = bulk_structure.get_volume() / len(bulk_structure)
        target_volume = n_atoms * vol_per_atom
        radius = (3 * target_volume / (4 * np.pi)) ** (1 / 3)

        # Create a large enough supercell (make it generous)
        cell_min = min(bulk_structure.get_cell().diagonal())
        # Ensure large enough supercell
        repeat = max(5, int(np.ceil(3 * radius / cell_min)))
        supercell = bulk_structure.repeat([repeat, repeat, repeat])

        # Center the supercell
        positions = supercell.get_positions()
        center = positions.mean(axis=0)
        positions -= center
        supercell.set_positions(positions)

        # Select atoms by distance to get exactly n_atoms
        distances = np.linalg.norm(positions, axis=1)
        sorted_indices = np.argsort(distances)

        # Take exactly n_atoms closest to center
        selected_indices = sorted_indices[:n_atoms]
        cluster = supercell[selected_indices]

        # Center the cluster
        cluster.center()

    # Final verification and correction
    current_natoms = len(cluster)
    if current_natoms != n_atoms:
        if current_natoms > n_atoms:
            # Trim excess atoms (keep those closest to center)
            positions = cluster.get_positions()
            distances = np.linalg.norm(positions, axis=1)
            sorted_indices = np.argsort(distances)
            cluster = cluster[sorted_indices[:n_atoms]]
            mdb_b_ut.custom_print(
                f'Trimmed cluster from {current_natoms} to {n_atoms} atoms',
                'debug',
            )
        else:
            # Need more atoms - add them at reasonable positions
            mdb_b_ut.custom_print(
                f'Need to add {n_atoms - current_natoms} atoms to reach target size',
                'debug',
            )

            # Get existing positions and find good spots for new atoms
            existing_positions = cluster.get_positions()

            # Create additional atoms at positions that maintain roughly spherical shape
            from ase import Atoms

            atoms_needed = n_atoms - current_natoms

            # Find the maximum distance from center to guide placement
            max_dist = np.max(np.linalg.norm(existing_positions, axis=1))

            # Add atoms in shells around the existing cluster
            additional_positions = []
            for i in range(atoms_needed):
                # Use spherical coordinates to place atoms
                phi = np.random.uniform(0, 2 * np.pi)
                costheta = np.random.uniform(-1, 1)
                theta = np.arccos(costheta)

                # Place at distance slightly larger than existing atoms
                r = max_dist + 1.0 + 0.5 * i

                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)

                additional_positions.append([x, y, z])

            additional_atoms = Atoms(
                metal * atoms_needed, positions=additional_positions
            )
            cluster += additional_atoms

    # Final verification
    if len(cluster) != n_atoms:
        mdb_b_ut.custom_print(
            f'Warning: Final cluster has {len(cluster)} atoms, expected {n_atoms}',
            'warn',
        )

    # Remove any periodic boundary conditions
    cluster.set_pbc(False)

    return cluster
