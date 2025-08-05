"""Collection of MDB-trained benchmarks for MLIPs."""

import copy
import json
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.build import bulk, surface
from ase.io import read as ase_read
from ase.io.trajectory import TrajectoryWriter
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import LBFGS
from mace.calculators import MACECalculator

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


def run_melting_point_benchmark(args, model_paths: list[pl.Path]):
    """
    Calculates the melting point using the Two-Phase Coexistence Method.

    This method creates a solid-liquid interface and monitors its stability
    at different temperatures to determine the melting point.
    """
    mdb_b_ut.custom_print('Running Melting Point Benchmark', 'info')
    benchmark_dir = args.output_dir / 'melting_point'
    benchmark_dir.mkdir(exist_ok=True)

    # Two-phase coexistence parameters
    # Temperature range to test (around typical Cu melting point ~1358 K)
    base_temp = 1358  # Approximate experimental melting point for Cu
    temp_range = 200  # ±200 K range
    temperatures = [
        base_temp - (temp_range * 1.5),
        base_temp - temp_range,
        base_temp - temp_range // 2,
        base_temp,
        base_temp + temp_range // 2,
        base_temp + temp_range,
        base_temp + (temp_range * 1.5),
    ]

    # Simulation parameters
    # Large cell to accommodate interface - [4,4,10]
    supercell_size = [4, 4, 10]

    # 100 ps at 2 fs timestep
    sim_steps = 50000

    # 20 ps equilibration
    equilibration_steps = 10000

    # 1 bar (in ASE units: eV/Å³)
    pressure = 1.0

    # Results storage
    results = {}

    # For each model, prepare structure and test at different temperatures
    for _, model_path in enumerate(model_paths):
        model_name = mdb_b_ut.get_model_display_name(model_path)
        mdb_b_ut.custom_print(f'Testing melting point for model: {model_name}', 'info')

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
            calculator = MACECalculator(
                model_paths=str(model_path),
                device=args.device,
                default_dtype=args.dtype,
                enable_cueq=True,
            )

            # Create model-specific directory for structures and trajectories
            model_benchmark_dir = benchmark_dir / model_name
            model_benchmark_dir.mkdir(exist_ok=True)

            # Prepare coexistence structure for this specific model
            mdb_b_ut.custom_print(
                f'Preparing coexistence structure for model: {model_name}', 'info'
            )
            coexistence_path = prepare_coexistence_structure(
                args.metal,
                supercell_size,
                model_benchmark_dir,
                calculator,
                base_temp,
            )

            model_results = {
                'temperatures_tested': temperatures,
                'interface_stability': [],
                'potential_energies': [],
                'estimated_melting_point': None,
                'temperature_analysis': {},
                'coexistence_structure_path': str(coexistence_path),
            }

            # Get cell dimensions for interface analysis
            test_structure = ase_read(coexistence_path, format='extxyz')
            cell = test_structure.get_cell()
            z_max = cell[2, 2]

            # Test at each temperature
            for temp in temperatures:
                mdb_b_ut.custom_print(f'Testing temperature: {temp} K', 'info')

                traj_path = model_benchmark_dir / f'T{temp}K.traj'

                # Load fresh coexistence structure
                test_atoms = ase_read(coexistence_path, format='extxyz')
                test_atoms.set_calculator(calculator)

                # Data storage for this temperature
                energies = []
                interface_positions = []

                def collect_melting_point_req_data(
                    atoms=test_atoms,
                    energy_list=energies,
                    interface_list=interface_positions,
                    z_max_val=z_max,
                ):
                    """Collect energy and interface position data."""
                    energy_list.append(atoms.get_potential_energy())

                    # Analyze interface position by finding density transitions
                    positions = atoms.get_positions()
                    z_coords = positions[:, 2]

                    # Bin atoms along z-axis and calculate density
                    z_bins = np.linspace(0, z_max_val, 50)
                    hist, _ = np.histogram(z_coords, bins=z_bins)
                    density = hist / (len(atoms) / len(z_bins))

                    # Find interfaces as locations where density changes rapidly
                    density_gradient = np.gradient(density)
                    threshold = np.std(density_gradient) * 2
                    interface_indices = np.where(np.abs(density_gradient) > threshold)[
                        0
                    ]

                    if len(interface_indices) >= 2:
                        # Record positions of solid-liquid interfaces
                        interface_z = z_bins[interface_indices]
                        interface_list.append([interface_z[0], interface_z[-1]])
                    else:
                        # No clear interface found
                        interface_list.append([None, None])

                if traj_path.exists():
                    mdb_b_ut.custom_print(
                        f'Trajectory for {temp} K already exists. Loading...',
                        'warn',
                    )
                    md_traj = ase_read(traj_path, format='traj', index=':')
                    for frame_idx in range(len(md_traj)):
                        curr_idx = md_traj[frame_idx]
                        collect_melting_point_req_data(atoms=curr_idx)

                else:
                    # Initialize velocities for the target temperature
                    MaxwellBoltzmannDistribution(test_atoms, temperature_K=temp)

                    # Set up NPT dynamics
                    npt = NPT(
                        test_atoms,
                        timestep=args.timestep * units.fs,
                        temperature_K=temp,
                        externalstress=pressure * units.GPa,  # Convert to ASE units
                        ttime=50 * units.fs,  # Thermostat time constant
                        pfactor=75 * units.fs**2,  # Barostat time constant
                    )

                    # Set up trajectory and data collection
                    traj = TrajectoryWriter(traj_path, 'w', test_atoms)

                    # Attach data collection and trajectory writing
                    npt.attach(collect_melting_point_req_data, interval=100)
                    npt.attach(traj.write, interval=100)

                    # Equilibration phase
                    mdb_b_ut.custom_print(f'Equilibrating at {temp} K...', 'info')
                    npt.run(equilibration_steps)

                    # Production phase
                    mdb_b_ut.custom_print(f'Production run at {temp} K...', 'info')

                    # Clear equilibration data
                    energies.clear()
                    interface_positions.clear()

                    # Run simulation
                    npt.run(sim_steps)

                # Analyze results for this temperature
                avg_energy = np.mean(energies)
                energy_std = np.std(energies)

                # Analyze interface stability
                valid_interfaces = [
                    pos for pos in interface_positions if pos[0] is not None
                ]
                # Need sufficient data
                if len(valid_interfaces) > 10:
                    interface_lower = [pos[0] for pos in valid_interfaces]
                    interface_upper = [pos[1] for pos in valid_interfaces]

                    # Calculate interface drift (indicates melting/freezing)
                    lower_drift = np.polyfit(
                        range(len(interface_lower)), interface_lower, 1
                    )[0]
                    upper_drift = np.polyfit(
                        range(len(interface_upper)), interface_upper, 1
                    )[0]
                    total_drift = abs(lower_drift) + abs(upper_drift)

                    # Threshold for stability
                    interface_stable = total_drift < 0.01
                else:
                    interface_stable = False
                    total_drift = float('inf')

                # Store results for this temperature
                temp_analysis = {
                    'avg_energy': avg_energy,
                    'energy_std': energy_std,
                    'interface_stable': interface_stable,
                    'interface_drift': total_drift,
                    'num_interface_points': len(valid_interfaces),
                }

                model_results['potential_energies'].append(avg_energy)
                model_results['interface_stability'].append(interface_stable)
                model_results['temperature_analysis'][temp] = temp_analysis

                mdb_b_ut.custom_print(
                    f'T={temp}K: Energy={avg_energy:.3f}±{energy_std:.3f} eV, '
                    f'Interface stable: {interface_stable}, Drift: {total_drift:.4f}',
                    'info',
                )

                # Clean up trajectory if it was created
                if 'traj' in locals():
                    traj.close()

            # Estimate melting point from results
            stable_temps = [
                temp
                for i, temp in enumerate(temperatures)
                if model_results['interface_stability'][i]
            ]

            if stable_temps:
                # Take the average of stable temperatures as melting point estimate
                estimated_tm = np.mean(stable_temps)
                model_results['estimated_melting_point'] = estimated_tm
                mdb_b_ut.custom_print(
                    f'Estimated melting point for {model_name}: {estimated_tm:.1f} K',
                    'done',
                )
            else:
                mdb_b_ut.custom_print(
                    f'Could not determine melting point for {model_name} - '
                    'no stable interfaces found',
                    'warn',
                )

            results[model_name] = model_results

            # Serialize bool types in results dict
            results[model_name]['interface_stability'] = [
                str(x).lower() for x in results[model_name]['interface_stability']
            ]

            for key in results[model_name]['temperature_analysis']:
                results[model_name]['temperature_analysis'][key]['interface_stable'] = (
                    str(
                        results[model_name]['temperature_analysis'][key][
                            'interface_stable'
                        ]
                    ).lower()
                )

            # Save individual results
            with open(results_file, 'w') as f:
                json.dump(model_results, f, indent=2)

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
        calculator = MACECalculator(
            model_paths=str(model_path),
            device=args.device,
            default_dtype=args.dtype,
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
            calculator = MACECalculator(
                model_paths=str(model_path),
                device=args.device,
                default_dtype=args.dtype,
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

    # Create bulk reference structure
    bulk_atoms = bulk(args.metal, cubic=True)

    # Create a small supercell for bulk calculations
    bulk_supercell = bulk_atoms.repeat([2, 2, 2])
    bulk_structure_path = benchmark_dir / 'bulk_reference.xyz'
    bulk_supercell.write(bulk_structure_path, format='extxyz')

    # Create slab structures for each surface
    slab_structures = {}
    for surface_indices in surfaces_to_test:
        surface_name = ''.join(map(str, surface_indices))

        # Create slab
        slab = surface(
            args.metal,
            surface_indices,
            slab_layers,
            vacuum=vacuum_thickness,
        )
        slab = slab.repeat(slab_size + [1])
        slab.center(vacuum=vacuum_thickness, axis=2)
        slab.pbc = True

        # Save slab structure
        slab_path = benchmark_dir / f'slab_{surface_name}.xyz'
        slab.write(slab_path, format='extxyz')
        slab_structures[surface_name] = {
            'path': slab_path,
            # Surface area
            'area': slab.get_cell()[0, 0] * slab.get_cell()[1, 1],
            'n_atoms': len(slab),
        }

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
            calculator = MACECalculator(
                model_paths=str(model_path),
                device=args.device,
                default_dtype=args.dtype,
            )

            # Calculate bulk energy per atom
            bulk_atoms = ase_read(bulk_structure_path, format='extxyz')
            bulk_atoms.set_calculator(calculator)

            # Relax bulk structure
            bulk_optimizer = LBFGS(
                bulk_atoms, logfile=benchmark_dir / f'{model_name}_bulk_relax.log'
            )
            bulk_optimizer.run(fmax=0.01, steps=500)
            e_bulk_total = bulk_atoms.get_potential_energy()
            e_bulk_per_atom = e_bulk_total / len(bulk_atoms)

            # Save relaxed bulk structure
            bulk_relaxed_path = benchmark_dir / f'{model_name}_bulk_relaxed.xyz'
            bulk_atoms.write(bulk_relaxed_path, format='extxyz')

            # Calculate surface energies for each surface
            model_results = {
                'bulk_energy_per_atom_eV': e_bulk_per_atom,
                'bulk_total_energy_eV': e_bulk_total,
                'surfaces': {},
            }

            for surface_name, slab_info in slab_structures.items():
                # Load and relax slab
                slab_atoms = ase_read(slab_info['path'], format='extxyz')
                slab_atoms.set_calculator(calculator)

                # Relax slab structure
                slab_optimizer = LBFGS(
                    slab_atoms,
                    logfile=benchmark_dir
                    / f'{model_name}_slab_{surface_name}_relax.log',
                )
                slab_optimizer.run(fmax=0.01, steps=500)
                e_slab = slab_atoms.get_potential_energy()

                # Save relaxed slab structure
                slab_relaxed_path = (
                    benchmark_dir / f'{model_name}_slab_{surface_name}_relaxed.xyz'
                )
                slab_atoms.write(slab_relaxed_path, format='extxyz')

                # Calculate surface energy
                # Surface Energy = (E_slab - N_slab * E_bulk_per_atom) / (2 * Area)
                # Factor of 2 because there are two surfaces in a slab
                n_atoms_slab = slab_info['n_atoms']
                area = slab_info['area']

                surface_energy_total = e_slab - n_atoms_slab * e_bulk_per_atom
                # eV/Å² units
                surface_energy_per_area = surface_energy_total / (2 * area)

                # Convert to more common units (J/m²)
                # 1 eV/Å² = 16.02176 J/m²
                surface_energy_j_m2 = surface_energy_per_area * 16.02176

                model_results['surfaces'][surface_name] = {
                    'surface_energy_eV_per_A2': surface_energy_per_area,
                    'surface_energy_J_per_m2': surface_energy_j_m2,
                    'slab_energy_eV': e_slab,
                    'slab_n_atoms': n_atoms_slab,
                    'surface_area_A2': area,
                }

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

        mdb_b_ut.set_plot_data(
            'surface_energies',
            {
                'type': 'surface_energies',
                'model_names': model_names,
                'surface_names': surface_names,
                'results': results,
                'metal': args.metal,
                'title': 'Surface Energies',
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


def prepare_coexistence_structure(
    metal: str,
    supercell_size: list[int],
    benchmark_dir: pl.Path,
    calculator,
    base_temp: float,
) -> pl.Path:
    """
    Prepare a solid-liquid coexistence structure using proper equilibration.

    This function implements the multi-step preparation process:
    1. Equilibrate solid phase at T < T_melt
    2. Create and equilibrate liquid phase at T > T_melt
    3. Combine solid and liquid phases into coexistence structure
    4. Relax interfaces

    Returns path to the final coexistence structure.
    """
    # Check if final structure already exists
    final_coexistence_path = benchmark_dir / 'coexistence_structure_prepared.xyz'
    if final_coexistence_path.exists():
        mdb_b_ut.custom_print(
            f'Using existing prepared coexistence structure: {final_coexistence_path}',
            'info',
        )
        return final_coexistence_path

    mdb_b_ut.custom_print(
        'Preparing coexistence structure through multi-step process...', 'info'
    )

    # Phase 1: Prepare equilibrated solid phase
    mdb_b_ut.custom_print('Phase 1: Equilibrating solid phase...', 'info')

    solid_path = benchmark_dir / 'equilibrated_solid.xyz'
    if not solid_path.exists():
        # Use temperature well below base_temp for solid equilibration

        # 200K below melting point
        solid_temp = base_temp - 200

        # Create initial solid structure
        solid_bulk = bulk(metal, cubic=True)
        solid_supercell = solid_bulk.repeat(supercell_size)
        mdb_b_ut.custom_print(
            f'Initial solid supercell created with {len(solid_supercell)} atoms',
            'debug',
        )
        solid_supercell.set_calculator(calculator)
        solid_supercell.write(
            benchmark_dir / 'init_equilibrated_solid.xyz', format='extxyz'
        )

        # Initialize velocities for solid temperature
        MaxwellBoltzmannDistribution(solid_supercell, temperature_K=solid_temp)

        # Equilibrate solid using NPT
        solid_npt = NPT(
            solid_supercell,
            timestep=2.0 * units.fs,
            temperature_K=solid_temp,
            externalstress=1.0 * units.bar,
            ttime=50 * units.fs,
            pfactor=75 * units.fs**2,
        )

        mdb_b_ut.custom_print(
            f'Equilibrating solid at {solid_temp} K for 20 ps...', 'info'
        )

        # 20 ps equilibration
        solid_npt.run(10000)

        # Save equilibrated solid
        solid_supercell.write(solid_path, format='extxyz')
        mdb_b_ut.custom_print(f'Equilibrated solid saved to {solid_path}', 'info')
    else:
        solid_supercell = ase_read(solid_path, format='extxyz')
        mdb_b_ut.custom_print(
            f'Loaded existing equilibrated solid from {solid_path}', 'info'
        )

    # Phase 2: Prepare equilibrated liquid phase
    liquid_path = benchmark_dir / 'equilibrated_liquid.xyz'
    if not liquid_path.exists():
        mdb_b_ut.custom_print(
            'Phase 2: Creating and equilibrating liquid phase...', 'info'
        )

        # 200K above base_temp for melting
        liquid_temp_high_K = base_temp + 200

        # Target temperature for coexistence
        liquid_temp_target_K = base_temp

        # Start with solid structure for melting
        liquid_supercell = solid_supercell.copy()
        liquid_supercell.set_calculator(calculator)
        solid_supercell.write(
            benchmark_dir / 'init_equilibrated_liquid.xyz', format='extxyz'
        )

        # Heat to high temperature for melting
        MaxwellBoltzmannDistribution(liquid_supercell, temperature_K=liquid_temp_high_K)

        # Melt using NVT dynamics (fixed volume to preserve density)
        liquid_nvt_melt = Langevin(
            liquid_supercell,
            timestep=2.0 * units.fs,
            temperature_K=liquid_temp_high_K,
            friction=0.01,  # Strong coupling for rapid equilibration
        )

        # Equilibrate solid using NPT
        mdb_b_ut.custom_print(
            f'Melting structure at {liquid_temp_high_K} K for 20 ps...', 'info'
        )

        # 20 ps melting
        liquid_nvt_melt.run(10000)

        # Cool down to target temperature and equilibrate
        MaxwellBoltzmannDistribution(
            liquid_supercell, temperature_K=liquid_temp_target_K
        )

        liquid_nvt_cool = Langevin(
            liquid_supercell,
            timestep=2.0 * units.fs,
            temperature_K=liquid_temp_target_K,
            friction=0.005,  # Moderate coupling for equilibration
        )

        mdb_b_ut.custom_print(
            f'Equilibrating liquid at {liquid_temp_target_K} K for 20 ps...',
            'info',
        )

        # 20 ps equilibration
        liquid_nvt_cool.run(10000)

        # Save equilibrated liquid
        liquid_supercell.write(liquid_path, format='extxyz')
        mdb_b_ut.custom_print(f'Equilibrated liquid saved to {liquid_path}', 'info')
    else:
        liquid_supercell = ase_read(liquid_path, format='extxyz')
        mdb_b_ut.custom_print(
            f'Loaded existing equilibrated liquid from {liquid_path}', 'info'
        )

    # Phase 3: Combine solid and liquid phases
    mdb_b_ut.custom_print('Phase 3: Combining solid and liquid phases...', 'info')

    # Get positions and cell info
    solid_positions = solid_supercell.get_positions()
    liquid_positions = liquid_supercell.get_positions()
    cell = solid_supercell.get_cell()

    # Create extended cell in z-direction to accommodate both phases
    z_extend_factor = 2.0
    new_cell = cell.copy()
    new_cell[2, 2] *= z_extend_factor

    # Select atoms from each phase (use z-coordinate sorting)
    solid_z_sorted = np.argsort(solid_positions[:, 2])
    liquid_z_sorted = np.argsort(liquid_positions[:, 2])

    # Take half the atoms from each phase
    n_atoms_half = len(solid_supercell) // 2

    # Bottom half
    solid_half_indices = solid_z_sorted[:n_atoms_half]

    # Top half
    liquid_half_indices = liquid_z_sorted[n_atoms_half:]

    # Create combined structure
    combined_positions = []
    combined_symbols = []

    # Add solid atoms (bottom region)
    for idx in solid_half_indices:
        pos = solid_positions[idx].copy()

        # Center in cell, extend towards bottom
        pos[2] = (pos[2]) + (0.25 * new_cell[2, 2])
        combined_positions.append(pos)
        combined_symbols.append(solid_supercell[idx].symbol)

    # Add liquid atoms (top region)
    for idx in liquid_half_indices:
        pos = liquid_positions[idx].copy()

        # Center in cell, extend towards bottom
        pos[2] = (pos[2]) + (0.30 * new_cell[2, 2])
        combined_positions.append(pos)
        combined_symbols.append(liquid_supercell[idx].symbol)

    # Create coexistence structure
    from ase import Atoms

    coexistence_atoms = Atoms(
        symbols=combined_symbols,
        positions=combined_positions,
        cell=new_cell,
        pbc=True,
    )

    # Save initial combined structure
    initial_combined_path = benchmark_dir / 'initial_combined.xyz'
    coexistence_atoms.write(initial_combined_path, format='extxyz')
    mdb_b_ut.custom_print(
        f'Initial combined structure saved to {initial_combined_path}', 'info'
    )

    # Phase 4: Relax interfaces
    mdb_b_ut.custom_print('Phase 4: Relaxing interfaces...', 'info')

    coexistence_atoms.set_calculator(calculator)
    MaxwellBoltzmannDistribution(coexistence_atoms, temperature_K=liquid_temp_target_K)

    # Use NVE dynamics to relax interfaces at fixed volume
    interface_relaxer = VelocityVerlet(
        coexistence_atoms,
        timestep=1.0 * units.fs,  # Smaller timestep for stability
    )

    mdb_b_ut.custom_print('Relaxing interfaces with NVE dynamics for 10 ps...', 'info')

    # 10 ps interface relaxation
    interface_relaxer.run(10000)

    # Save final prepared structure
    coexistence_atoms.write(final_coexistence_path, format='extxyz')
    mdb_b_ut.custom_print(
        f'Final coexistence structure saved to {final_coexistence_path}', 'info'
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
            calculator = MACECalculator(
                model_paths=str(model_path),
                device=args.device,
                default_dtype=args.dtype,
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
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

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
        plot_path = benchmark_dir / 'database_evaluation_errors.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mdb_b_ut.custom_print(f'Plot saved to {plot_path}', 'done')
        plt.close()

        # Store plot data for final multi-panel figure
        model_names = list(results.keys())
        mean_energy_errors = [
            results[name]['mean_energy_error_meV_per_atom'] for name in model_names
        ]
        mean_force_errors = [
            results[name]['mean_force_error_eV_per_A'] for name in model_names
        ]

        mdb_b_ut.set_plot_data(
            'database_evaluation',
            {
                'type': 'database_evaluation',
                'model_names': model_names,
                'mean_energy_errors': mean_energy_errors,
                'mean_force_errors': mean_force_errors,
                'results': results,
                'title': 'Database Evaluation Results',
            },
        )

        # Store detailed structure-by-structure plot data
        mdb_b_ut.set_plot_data(
            'database_evaluation_detailed',
            {
                'type': 'database_evaluation_detailed',
                'model_names': model_names,
                'results': results,
                'title': 'Structure-by-Structure Errors',
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
