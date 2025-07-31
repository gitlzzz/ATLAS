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

import argparse
import atexit
import copy
import json
import math
import pathlib as pl
import re
import shutil
import time
import warnings

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
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

from MatDBForge.active_learning.report_utils import (
    get_loop_report,
)
from MatDBForge.core.code_utils import (
    custom_print as _original_custom_print,
)
from MatDBForge.core.code_utils import (
    init_logger,
)

# Global UI manager instance
_ui_manager = None
_log_file = None
_plot_data = {}  # Store data for final multi-panel plot
_model_display_names = {}  # Maps model path to display name


def _initialize_log_file():
    """Initialize the log file with a timestamp."""
    global _log_file
    if _log_file is None:
        timestamp = int(time.time())
        log_filename = f'mdb_run_mlip_benchmark_{timestamp}.log'
        _log_file = open(log_filename, 'w', encoding='utf-8')  # noqa
        print(f'Logging to file: {log_filename}')
        # Register cleanup function
        atexit.register(_cleanup_log_file)


def _cleanup_log_file():
    """Close the log file properly."""
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None


def _write_to_log_file(message):
    """Write a message to the log file."""
    global _log_file
    if _log_file:
        # Create a clean version without Rich markup for file logging
        clean_message = re.sub(r'\[/?[a-zA-Z0-9_ ]+\]', '', message)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        _log_file.write(f'[{timestamp}] {clean_message}\n')
        _log_file.flush()


def custom_print(message, level='info'):
    """Custom print function that integrates with Rich UI and logs to file."""
    global _ui_manager, _log_file

    # Initialize log file if not already done
    _initialize_log_file()

    # Write to log file with clean message
    _write_to_log_file(f'{level.upper()}: {message}' if level != 'empty' else message)

    if _ui_manager:
        # Format message with level for Rich UI
        if level == 'info':
            formatted = f'[blue]INFO[/blue]: {message}'
        elif level == 'warn':
            formatted = f'[yellow]WARN[/yellow]: {message}'
        elif level == 'error':
            formatted = f'[red]ERROR[/red]: {message}'
        elif level == 'done':
            formatted = f'[green]DONE[/green]: {message}'
        elif level == 'debug':
            formatted = f'[bright_black]DEBUG[/bright_black]: {message}'
        else:
            formatted = message

        _ui_manager.log(formatted)
    else:
        # Fallback to original function
        _original_custom_print(message, level)


# Ignore all warnings
warnings.filterwarnings('ignore')

COLORS = [
    '#fe8019',
    '#038999',
    '#689d6a',
    '#b16286',
    '#458588',
    '#d79921',
    '#cc241d',
    '#98971a',
]


class RichUIManager:
    """Manages the Rich CLI interface for the benchmark suite."""

    def __init__(self, benchmarks_to_run):
        self.console = Console()
        self.benchmarks_to_run = benchmarks_to_run
        self.current_benchmark = None
        self.completed_benchmarks = set()
        self.logs = []
        self.live = None
        self.progress = None
        self.overall_task = None

    def create_layout(self):
        """Create the main layout structure."""
        layout = Layout()

        # Split into header and body
        layout.split_column(Layout(name='header', size=3), Layout(name='body'))

        # Split body into sidebar and main content
        layout['body'].split_row(Layout(name='sidebar', size=30), Layout(name='main'))

        return layout

    def create_progress_panel(self):
        """Create the progress bar panel."""
        if self.progress is None:
            self.progress = Progress(
                TextColumn('[bold blue]{task.description}'),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                console=self.console,
                expand=True,
            )
            self.overall_task = self.progress.add_task(
                'Overall Progress', total=len(self.benchmarks_to_run)
            )

        return Panel(
            self.progress,
            title='[bold cyan]MatDBForge - MLIP Benchmark Suite',
            border_style='blue',
            expand=True,
        )

    def create_sidebar_panel(self):
        """Create the sidebar showing benchmark status."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column('Status', width=3)
        table.add_column('Benchmark', min_width=20)

        for benchmark in self.benchmarks_to_run:
            if benchmark in self.completed_benchmarks:
                status = '[green]✓[/green]'
                style = 'green'
            elif benchmark == self.current_benchmark:
                status = '[yellow]►[/yellow]'
                style = 'yellow bold'
            else:
                status = '[dim]○[/dim]'
                style = 'dim'

            table.add_row(status, f'[{style}]{benchmark}[/{style}]')

        return Panel(table, title='[bold]Benchmarks', border_style='green')

    def create_log_panel(self):
        """Create the log display panel."""
        # Calculate available height for logs based on terminal size
        # Account for: header panel (5 lines), sidebar, panel borders, padding
        terminal_height = self.console.size.height
        # Reserve 10 lines for UI elements
        available_height = max(10, terminal_height - 10)

        # Show the most recent logs that fit in the available display space
        total_logs = len(self.logs)
        max_logs = min(total_logs, available_height)
        visible_logs = self.logs[-max_logs:] if max_logs > 0 else self.logs
        log_text = '\n'.join(visible_logs) if visible_logs else 'Waiting for logs...'

        # Add indicator if there are more logs above
        if len(self.logs) > max_logs:
            earlier_count = len(self.logs) - max_logs
            log_text = f'[dim]... ({earlier_count} earlier logs) ...[/dim]\n' + log_text

        benchmark_name = self.current_benchmark or 'Processing'
        title = f'[bold]Logs - {benchmark_name}'

        return Panel(
            Text.from_markup(log_text),
            title=title,
            border_style='cyan',
        )

    def update_display(self):
        """Update the entire display."""
        if self.live is None:
            return

        layout = self.create_layout()
        layout['header'].update(self.create_progress_panel())
        layout['sidebar'].update(self.create_sidebar_panel())
        layout['main'].update(self.create_log_panel())

        self.live.update(layout)

    def start_benchmark(self, benchmark_name):
        """Mark a benchmark as started."""
        self.current_benchmark = benchmark_name
        custom_print('', 'empty')
        custom_print(
            f'Starting benchmark: [bold blue]{benchmark_name}[/bold blue]', 'info'
        )
        self.update_display()

    def complete_benchmark(self, benchmark_name):
        """Mark a benchmark as completed."""
        self.completed_benchmarks.add(benchmark_name)
        if self.progress and self.overall_task is not None:
            self.progress.update(self.overall_task, advance=1)
        custom_print(f'Completed benchmark: {benchmark_name}', 'done')
        self.current_benchmark = None
        self.update_display()

    def log(self, message):
        """Add a log message."""
        self.logs.append(
            f'[dim bright_black]{len(self.logs) + 1:03d}[/dim bright_black] {message}'
        )
        self.update_display()

    def __enter__(self):
        """Enter context manager."""
        layout = self.create_layout()
        layout['header'].update(self.create_progress_panel())
        layout['sidebar'].update(self.create_sidebar_panel())
        layout['main'].update(self.create_log_panel())

        self.live = Live(layout, console=self.console, refresh_per_second=4)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.live:
            self.live.stop()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate and compare MLIPs performance.'
    )
    parser.add_argument(
        '--model_files',
        nargs='+',
        help='Paths to .model files.',
        default=[],
    )
    parser.add_argument(
        '--aiida_pks',
        nargs='+',
        type=int,
        help='AiiDA workchain PKs/UUIDs to load models from.',
        default=[],
    )
    parser.add_argument(
        '--output_dir',
        type=pl.Path,
        default=pl.Path('./mlip_evaluation'),
        help='Directory to save results.',
    )

    # Slab generation arguments
    slab_group = parser.add_argument_group('Slab Generation')
    slab_group.add_argument(
        '--metal',
        type=str,
        default='Cu',
        help='Metal symbol for the benchmark systems.',
    )
    slab_group.add_argument(
        '--surface_indices',
        nargs=3,
        type=int,
        default=[1, 1, 1],
        help='Miller indices for the surface.',
    )
    slab_group.add_argument(
        '--supercell_size',
        nargs=3,
        type=int,
        default=[3, 3, 4],
        help='Size of the supercell (e.g., 3 3 4).',
    )
    slab_group.add_argument(
        '--vacuum', type=float, default=10.0, help='Vacuum layer in Angstrom.'
    )

    # MD arguments
    md_group = parser.add_argument_group('MD Parameters')
    md_group.add_argument(
        '--temp', type=float, default=300.0, help='MD temperature in Kelvin.'
    )
    md_group.add_argument(
        '--n_steps', type=int, default=10000, help='Number of MD steps.'
    )
    md_group.add_argument(
        '--timestep', type=float, default=2.0, help='MD timestep in fs.'
    )
    md_group.add_argument(
        '--friction', type=float, default=5e-3, help='Friction for Langevin dynamics.'
    )
    md_group.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run the calculations on.',
    )
    md_group.add_argument(
        '--dtype',
        type=str,
        default='float64',
        choices=['float32', 'float64'],
        help='Data type for the calculations.',
    )

    # Benchmark selection
    benchmark_group = parser.add_argument_group('Benchmark Selection')
    benchmark_group.add_argument(
        '--run_energy_md',
        action='store_true',
        help='Run the energy MD benchmark.',
    )
    benchmark_group.add_argument(
        '--test_set_path',
        type=pl.Path,
        help='Path to the held-out test set for accuracy benchmarks.',
    )
    benchmark_group.add_argument(
        '--run_accuracy_test_set',
        action='store_true',
        help='Run energy and force error benchmark on a test set.',
    )
    benchmark_group.add_argument(
        '--run_elastic_properties',
        action='store_true',
        help='Run elastic properties benchmark.',
    )
    benchmark_group.add_argument(
        '--run_defect_formation_energy',
        action='store_true',
        help='Run defect formation energy benchmark.',
    )
    benchmark_group.add_argument(
        '--run_surface_energies',
        action='store_true',
        help='Run surface energies benchmark.',
    )
    benchmark_group.add_argument(
        '--run_phonon_dispersion',
        action='store_true',
        help='Run phonon dispersion benchmark.',
    )
    benchmark_group.add_argument(
        '--run_high_temp_md',
        action='store_true',
        help='Run high-temperature MD benchmark.',
    )
    benchmark_group.add_argument(
        '--run_melting_point',
        action='store_true',
        help='Run melting point calculation benchmark.',
    )
    benchmark_group.add_argument(
        '--run_gsfe',
        action='store_true',
        help='Run Generalized Stacking Fault Energy (GSFE) benchmark.',
    )
    benchmark_group.add_argument(
        '--run_learning_curves',
        action='store_true',
        help='Plot learning curves from AL runs.',
    )
    benchmark_group.add_argument(
        '--run_final_db_size',
        action='store_true',
        help='Compare final database sizes from AL runs.',
    )

    # UI options
    ui_group = parser.add_argument_group('UI Options')
    ui_group.add_argument(
        '--no_rich_ui',
        action='store_true',
        help='Disable Rich UI and use plain text output.',
    )

    return parser.parse_args()


def load_model_from_aiida(identifier: int, output_dir: pl.Path):
    """Load a model from an MDB AL Workchain."""
    from aiida import load_profile
    from aiida.orm import load_node

    load_profile()

    try:
        # Loading model from last workchain step
        base_workchain = load_node(identifier)
        last_workchain = base_workchain.called[-1]

        # Extract run name from the base workchain inputs
        run_name = None
        try:
            run_name = base_workchain.inputs.active_learning.run_name.value
            custom_print(
                f"Extracted run name: '{run_name}' from workchain {identifier}", 'debug'
            )
        except (AttributeError, KeyError) as e:
            custom_print(
                f'Could not extract run name from workchain {identifier}: {e}', 'warn'
            )
            # Fallback to using the identifier as the name
            run_name = f'workchain_{identifier}'

        # Saving model file to output directory
        with last_workchain.outputs.m0_model_file.as_path() as model_file:
            model_file_path = pl.Path(model_file)

            # Copying model to cwd
            model_file_path = shutil.copy(
                model_file_path, output_dir / model_file_path.name
            )

        # Store the mapping of model path to display name
        global _model_display_names
        _model_display_names[str(model_file_path)] = run_name

        custom_print(
            f"Loaded model '{run_name}' (file: {model_file_path.name}) "
            f'using MDB AL workchain PK/UUID: {identifier}',
            'info',
        )
        return model_file_path
    except Exception as e:
        custom_print(
            f'Failed to load model from Workchain pk {identifier}: {e}', 'error'
        )
        return None


def get_model_display_name(model_path: pl.Path) -> str:
    """
    Get the display name for a model.

    Returns run name if available, otherwise filename stem.
    """
    global _model_display_names
    path_str = str(model_path)
    if path_str in _model_display_names:
        return _model_display_names[path_str]
    else:
        # For models not loaded from AiiDA, use filename stem
        return model_path.stem


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
        model_name = get_model_display_name(model_path)
        custom_print(f'Running MD for model: {model_name}', 'info')

        # Skip if already run
        energy_file = benchmark_dir / f'{model_name}_energies.npy'
        force_file = benchmark_dir / f'{model_name}_max_forces.npy'
        if energy_file.exists() and force_file.exists():
            custom_print(
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
            custom_print(f'MD for {model_name} finished!', 'done')
            np.save(energy_file, np.array(energies))
            np.save(force_file, np.array(max_forces))
            energy_data[model_name] = np.array(energies)
            force_data[model_name] = np.array(max_forces)
        except Exception as e:
            custom_print(f'MD for {model_name} failed: {e}', 'error')

    if hasattr(args, 'mode'):
        if args.mode == 'high_energy':
            plot_data_key = 'energy_md_high_energy'
            title = f'High Temperature MD Benchmark ({args.temp} K)'
    else:
        plot_data_key = 'energy_md'
        title = 'Energy MD Benchmark'
    # Store plot data for final multi-panel figure
    global _plot_data
    _plot_data[plot_data_key] = {
        'type': plot_data_key,
        'energy_data': energy_data,
        'force_data': force_data,
        'timestep': args.timestep,
        'title': title,
    }

    custom_print('Energy MD data collected for final plot', 'info')


def run_accuracy_test_set_benchmark(args, model_paths: list[pl.Path]):
    """Calculates and plots energy and force RMSE on a held-out test set."""
    custom_print('Running Accuracy Benchmark on Test Set', 'info')
    if not args.test_set_path or not args.test_set_path.exists():
        custom_print('Test set path not provided or does not exist. Skipping.', 'error')
        return

    benchmark_dir = args.output_dir / 'accuracy_test_set'
    benchmark_dir.mkdir(exist_ok=True)

    # 1. Load the test set structures.
    # test_structures = ase_read(args.test_set_path, index=':')

    # 2. For each model, iterate through the test set.
    #    - Set the calculator on each structure.
    #    - Calculate MLIP energy and forces.
    #    - Assume DFT energy/forces are in structure.info/arrays.
    #    - Store the differences.

    # 3. Calculate RMSE for energies (meV/atom) and forces (eV/A).

    # 4. Save results to a file (e.g., CSV or JSON).

    # 5. Generate a bar plot comparing the RMSE values for each model.
    custom_print('Accuracy test set benchmark not implemented yet.', 'warn')


def run_elastic_properties_benchmark(args, model_paths: list[pl.Path]):
    """Calculates and plots elastic constants and bulk modulus."""
    custom_print('Running Elastic Properties Benchmark', 'info')
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
    custom_print('Elastic properties benchmark not implemented yet.', 'warn')


def run_defect_formation_energy_benchmark(args, model_paths: list[pl.Path]):
    """Calculates and plots the monovacancy formation energy."""
    custom_print('Running Defect Formation Energy Benchmark', 'info')
    benchmark_dir = args.output_dir / 'defect_formation_energy'
    benchmark_dir.mkdir(exist_ok=True)

    # Results storage
    results = {}

    # Create a large perfect supercell
    # (4x4x4 should be sufficient)
    supercell_size = [4, 4, 4]
    # custom_print(
    #     f'Creating {supercell_size} supercell for defect calculations...', 'debug'
    # )

    perfect_bulk = bulk(args.metal, cubic=True)
    perfect_supercell = perfect_bulk.repeat(supercell_size)

    # Save the perfect structure
    perfect_structure_path = benchmark_dir / 'perfect_supercell.xyz'
    perfect_supercell.write(perfect_structure_path, format='extxyz')
    # custom_print(f"Perfect supercell saved to '{perfect_structure_path}'", 'debug')

    # Create vacancy structure by removing the central atom
    # Remove roughly central atom
    defect_supercell = perfect_supercell.copy()
    central_index = len(defect_supercell) // 2
    del defect_supercell[central_index]

    # Save the defect structure
    defect_structure_path = benchmark_dir / 'vacancy_supercell.xyz'
    defect_supercell.write(defect_structure_path, format='extxyz')
    # custom_print(f"Vacancy supercell saved to '{defect_structure_path}'", 'debug')

    n_atoms_perfect = len(perfect_supercell)
    n_atoms_defect = len(defect_supercell)

    # For each model, calculate formation energies
    for _, model_path in enumerate(model_paths):
        model_name = get_model_display_name(model_path)
        # custom_print(
        #     f"Calculating vacancy formation energy for model: '{model_name}'", 'debug'
        # )

        # Skip if already calculated
        results_file = benchmark_dir / f'{model_name}_vacancy_formation_energy.json'
        if results_file.exists():
            custom_print(
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

            custom_print(
                f"Vacancy formation energy for '{model_name}': "
                f'{formation_energy:.3f} eV',
                'done',
            )

        except Exception as e:
            custom_print(
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
        global _plot_data
        model_names = list(results.keys())
        formation_energies = [
            results[name]['formation_energy_eV'] for name in model_names
        ]

        _plot_data['defect_formation'] = {
            'type': 'bar',
            'model_names': model_names,
            'values': formation_energies,
            'title': f'Monovacancy Formation Energy - {args.metal}',
            'ylabel': 'Formation Energy (eV)',
            'value_format': '{:.3f}',
        }

        # Print summary
        custom_print('Vacancy Formation Energy Summary:', 'info')
        for name, energy in zip(model_names, formation_energies, strict=True):
            custom_print(f'  {name}: {energy:.3f} eV', 'empty')
    else:
        custom_print(
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
    custom_print('Running Surface Energies Benchmark', 'info')
    benchmark_dir = args.output_dir / 'surface_energies'
    benchmark_dir.mkdir(exist_ok=True)

    # Define surfaces to test (Miller indices)
    surfaces_to_test = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    slab_layers = 7  # Number of layers in the slab
    slab_size = [3, 3]  # Supercell size in x and y directions
    vacuum_thickness = 15.0  # Vacuum thickness in Angstroms

    # Results storage
    results = {}

    # Create bulk reference structure
    # custom_print(f'Creating bulk reference structure for {args.metal}...', 'debug')
    bulk_atoms = bulk(args.metal, cubic=True)

    # Create a small supercell for bulk calculations
    bulk_supercell = bulk_atoms.repeat([2, 2, 2])
    bulk_structure_path = benchmark_dir / 'bulk_reference.xyz'
    bulk_supercell.write(bulk_structure_path, format='extxyz')
    # custom_print(f"Bulk reference saved to '{bulk_structure_path}'", 'debug')

    # Create slab structures for each surface
    slab_structures = {}
    for surface_indices in surfaces_to_test:
        surface_name = ''.join(map(str, surface_indices))
        # custom_print(f'Creating {surface_name} surface slab...', 'debug')

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
        # custom_print(f"Slab {surface_name} saved to '{slab_path}'", 'debug')

    # For each model, calculate surface energies
    for _, model_path in enumerate(model_paths):
        model_name = get_model_display_name(model_path)

        # Skip if already calculated
        results_file = benchmark_dir / f'{model_name}_surface_energies.json'
        if results_file.exists():
            custom_print(
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
            # custom_print(f"Relaxing bulk structure for '{model_name}'...", 'debug')
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

            custom_print(f"Surface energies calculated for '{model_name}'", 'done')

        except Exception as e:
            custom_print(
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
        global _plot_data
        model_names = list(results.keys())
        surface_names = list(surfaces_to_test)

        _plot_data['surface_energies'] = {
            'type': 'surface_energies',
            'model_names': model_names,
            'surface_names': surface_names,
            'results': results,
            'metal': args.metal,
            'title': 'Surface Energies',
        }

        # Print summary
        custom_print('Surface Energy Summary:', 'info')
        for model_name in model_names:
            custom_print(f'  {model_name}:', 'empty')
            for surface_indices in surface_names:
                surface_name = ''.join(map(str, surface_indices))
                if surface_name in results[model_name]['surfaces']:
                    energy = results[model_name]['surfaces'][surface_name][
                        'surface_energy_J_per_m2'
                    ]
                    custom_print(
                        f'    {args.metal}({surface_name}): {energy:.2f} J/m²', 'empty'
                    )
    else:
        custom_print('No results to plot for surface energies benchmark.', 'warn')


def run_phonon_dispersion_benchmark(args, model_paths: list[pl.Path]):
    """Calculates and plots the phonon dispersion curves."""
    custom_print('Running Phonon Dispersion Benchmark', 'info')
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
    custom_print('Phonon dispersion benchmark not implemented yet.', 'warn')


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

    custom_print(f'Running high-temperature MD at {high_temp_args.temp} K', 'info')
    custom_print(
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
        custom_print(
            f'Using existing prepared coexistence structure: {final_coexistence_path}',
            'info',
        )
        return final_coexistence_path

    custom_print(
        'Preparing coexistence structure through multi-step process...', 'info'
    )

    # Phase 1: Prepare equilibrated solid phase
    custom_print('Phase 1: Equilibrating solid phase...', 'info')

    solid_path = benchmark_dir / 'equilibrated_solid.xyz'
    if not solid_path.exists():
        # Use temperature well below base_temp for solid equilibration
        solid_temp = base_temp - 200  # 200K below melting point

        # Create initial solid structure
        solid_bulk = bulk(metal, cubic=True)
        solid_supercell = solid_bulk.repeat(supercell_size)
        custom_print(
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

        custom_print(f'Equilibrating solid at {solid_temp} K for 20 ps...', 'info')

        # 20 ps equilibration
        solid_npt.run(10000)

        # Save equilibrated solid
        solid_supercell.write(solid_path, format='extxyz')
        custom_print(f'Equilibrated solid saved to {solid_path}', 'info')
    else:
        solid_supercell = ase_read(solid_path, format='extxyz')
        custom_print(f'Loaded existing equilibrated solid from {solid_path}', 'info')

    # Phase 2: Prepare equilibrated liquid phase
    liquid_path = benchmark_dir / 'equilibrated_liquid.xyz'
    if not liquid_path.exists():
        custom_print('Phase 2: Creating and equilibrating liquid phase...', 'info')

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
            # Strong coupling for rapid equilibration
            friction=0.01,
        )

        # Equilibrate solid using NPT
        custom_print(
            f'Melting structure at {liquid_temp_high_K} K for 20 ps...', 'info'
        )
        liquid_nvt_melt.run(10000)  # 20 ps melting

        # Cool down to target temperature and equilibrate
        MaxwellBoltzmannDistribution(
            liquid_supercell, temperature_K=liquid_temp_target_K
        )

        liquid_nvt_cool = Langevin(
            liquid_supercell,
            timestep=2.0 * units.fs,
            temperature_K=liquid_temp_target_K,
            # Moderate coupling for equilibration
            friction=0.005,
        )

        custom_print(
            f'Equilibrating liquid at {liquid_temp_target_K} K for 20 ps...',
            'info',
        )
        liquid_nvt_cool.run(10000)  # 20 ps equilibration

        # Save equilibrated liquid
        liquid_supercell.write(liquid_path, format='extxyz')
        custom_print(f'Equilibrated liquid saved to {liquid_path}', 'info')
    else:
        liquid_supercell = ase_read(liquid_path, format='extxyz')
        custom_print(f'Loaded existing equilibrated liquid from {liquid_path}', 'info')

    # Phase 3: Combine solid and liquid phases
    custom_print('Phase 3: Combining solid and liquid phases...', 'info')

    # Load the equilibrated phases
    # solid_supercell = ase_read(solid_path, format='extxyz')
    # liquid_supercell = ase_read(liquid_path, format='extxyz')

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
    solid_half_indices = solid_z_sorted[:n_atoms_half]  # Bottom half
    liquid_half_indices = liquid_z_sorted[n_atoms_half:]  # Top half

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
    custom_print(f'Initial combined structure saved to {initial_combined_path}', 'info')

    # Phase 4: Relax interfaces
    custom_print('Phase 4: Relaxing interfaces...', 'info')

    coexistence_atoms.set_calculator(calculator)
    MaxwellBoltzmannDistribution(coexistence_atoms, temperature_K=liquid_temp_target_K)

    # Use NVE dynamics to relax interfaces at fixed volume
    interface_relaxer = VelocityVerlet(
        coexistence_atoms,
        timestep=1.0 * units.fs,  # Smaller timestep for stability
    )

    custom_print('Relaxing interfaces with NVE dynamics for 10 ps...', 'info')
    interface_relaxer.run(10000)  # 10 ps interface relaxation

    # Save final prepared structure
    coexistence_atoms.write(final_coexistence_path, format='extxyz')
    custom_print(
        f'Final coexistence structure saved to {final_coexistence_path}', 'info'
    )

    return final_coexistence_path


def run_melting_point_benchmark(args, model_paths: list[pl.Path]):
    """
    Calculates the melting point using the Two-Phase Coexistence Method.

    This method creates a solid-liquid interface and monitors its stability
    at different temperatures to determine the melting point.
    """
    custom_print('Running Melting Point Benchmark', 'info')
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
        model_name = get_model_display_name(model_path)
        custom_print(f'Testing melting point for model: {model_name}', 'info')

        # Skip if already calculated
        results_file = benchmark_dir / f'{model_name}_melting_point.json'
        if results_file.exists():
            custom_print(
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
            custom_print(
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
                custom_print(f'Testing temperature: {temp} K', 'info')

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
                    custom_print(
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
                    custom_print(f'Equilibrating at {temp} K...', 'info')
                    npt.run(equilibration_steps)

                    # Production phase
                    custom_print(f'Production run at {temp} K...', 'info')

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

                custom_print(
                    f'T={temp}K: Energy={avg_energy:.3f}±{energy_std:.3f} eV, '
                    f'Interface stable: {interface_stable}, Drift: {total_drift:.4f}',
                    'info',
                )

                if 'traj' in locals():
                    # Clean up trajectory
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
                custom_print(
                    f'Estimated melting point for {model_name}: {estimated_tm:.1f} K',
                    'done',
                )
            else:
                custom_print(
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
            custom_print(
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
        global _plot_data
        model_names = list(results.keys())
        melting_points = []

        for model_name in model_names:
            tm = results[model_name].get('estimated_melting_point')
            melting_points.append(tm if tm is not None else 0.0)

        _plot_data['melting_point'] = {
            'type': 'bar',
            'model_names': model_names,
            'values': melting_points,
            'title': f'Melting Point - {args.metal}',
            'ylabel': 'Temperature (K)',
            'value_format': '{:.0f}',
        }

        # Print summary
        custom_print('Melting Point Summary:', 'info')
        for name, tm in zip(model_names, melting_points, strict=True):
            if tm > 0:
                custom_print(f'  {name}: {tm:.1f} K', 'info')
            else:
                custom_print(f'  {name}: Unable to determine', 'warning')

        custom_print(f'Results saved to {all_results_file}', 'info')
    else:
        custom_print('No melting point results to save.', 'warn')


def run_gsfe_benchmark(args, model_paths: list[pl.Path]):
    """Calculates and plots the Generalized Stacking Fault Energy (GSFE) curve."""
    custom_print('Running GSFE Benchmark', 'info')
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
    custom_print('GSFE benchmark not implemented yet.', 'warn')


def run_learning_curves_benchmark(args, model_paths: list[pl.Path]):
    """Plots learning curves (e.g., test RMSE vs. number of DFT calls)."""
    custom_print('Running Learning Curves Benchmark', 'info')
    benchmark_dir = args.output_dir / 'learning_curves'
    benchmark_dir.mkdir(exist_ok=True)

    # This benchmark only works with AiiDA PKs
    if not args.aiida_pks:
        custom_print(
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
            custom_print(f'Processing learning curve for workchain {pk}...', 'info')

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
            custom_print(f'Failed to process workchain {pk}: {e}', 'error')
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
        global _plot_data
        _plot_data['learning_curves'] = {
            'type': 'learning_curves',
            'data': learning_curves_data,
            'title': 'Active Learning Curves',
        }

        # Print summary
        custom_print('Learning Curves Summary:', 'info')
        for run_name, data in learning_curves_data.items():
            final_energy = data['energy_rmse'][-1] if data['energy_rmse'] else 0
            final_force = data['force_rmse'][-1] if data['force_rmse'] else 0
            final_db_size = data['train_db_sizes'][-1] if data['train_db_sizes'] else 0
            custom_print(
                f'  {run_name}: {len(data["iterations"])} iterations, '
                f'final DB size: {final_db_size}, '
                f'final E RMSE: {final_energy:.3f} meV/atom, '
                f'final F RMSE: {final_force:.3f} meV/Å',
                'empty',
            )

        custom_print(f'Results saved to {results_file}', 'info')
    else:
        custom_print('No learning curve results to save.', 'warn')


def create_final_multi_panel_plot(args):
    """Creates a single multi-panel figure with all benchmark plots."""
    if not _plot_data:
        custom_print('No plot data available for multi-panel figure.', 'warn')
        return

    # Count available plot types
    plot_types = list(_plot_data.keys())
    n_plots = len(plot_types)

    if n_plots == 0:
        custom_print('No plots to display.', 'warn')
        return

    # Calculate grid layout (try to make it roughly square)
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_plots > 1 else axes

    # Plot each benchmark
    plot_idx = 0
    for plot_type, plot_info in _plot_data.items():
        ax = axes_flat[plot_idx]

        if plot_type in ['energy_md', 'energy_md_high_energy']:
            # Energy/Force time series plot
            energy_data = plot_info['energy_data']
            timestep = plot_info['timestep']
            model_names = list(energy_data.keys())

            for i, model_name in enumerate(model_names):
                energies = energy_data[model_name]
                # Generate time data from timestep and number of points
                time_data = np.arange(len(energies)) * timestep
                color = COLORS[i % len(COLORS)]
                ax.plot(time_data, energies, label=model_name, color=color, linewidth=2)

            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Energy (eV)')
            ax.set_title(plot_info['title'])
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        elif plot_type == 'defect_formation':
            # Bar chart for defect formation energies
            model_names = plot_info['model_names']
            formation_energies = plot_info['values']

            # Create bar plot
            colors = [COLORS[i % len(COLORS)] for i in range(len(model_names))]
            bars = ax.bar(model_names, formation_energies, color=colors)

            ax.set_xlabel('Model')
            ax.set_ylabel(plot_info['ylabel'])
            ax.set_title(plot_info['title'])
            ax.grid(True, linestyle='--', alpha=0.6)

            # Add value labels
            for bar, energy in zip(bars, formation_energies, strict=True):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(formation_energies) * 0.01,
                    plot_info['value_format'].format(energy),
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )

        elif plot_type == 'surface_energies':
            # Multiple surface energies subplot
            model_names = plot_info['model_names']
            surface_names = plot_info['surface_names']
            results = plot_info['results']

            # If multiple surfaces, create mini-subplots within this panel
            n_surfaces = len(surface_names)
            if n_surfaces == 1:
                # Single surface - bar chart
                surface_name = ''.join(map(str, surface_names[0]))
                energies = []
                for model_name in model_names:
                    if surface_name in results[model_name]['surfaces']:
                        energies.append(
                            results[model_name]['surfaces'][surface_name][
                                'surface_energy_J_per_m2'
                            ]
                        )
                    else:
                        energies.append(0)

                colors = [COLORS[i % len(COLORS)] for i in range(len(model_names))]
                bars = ax.bar(model_names, energies, color=colors)
                ax.set_title(f'{plot_info["metal"]}({surface_name}) Surface Energy')
                ax.set_ylabel('Surface Energy (J/m²)')

                # Add value labels
                for bar, energy in zip(bars, energies, strict=True):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(energies) * 0.01,
                        f'{energy:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                    )
            else:
                # Multiple surfaces - grouped bar chart
                bar_width = 0.8 / len(surface_names)
                x_pos = range(len(model_names))

                for i, surface_indices in enumerate(surface_names):
                    surface_name = ''.join(map(str, surface_indices))
                    energies = []
                    for model_name in model_names:
                        if surface_name in results[model_name]['surfaces']:
                            energies.append(
                                results[model_name]['surfaces'][surface_name][
                                    'surface_energy_J_per_m2'
                                ]
                            )
                        else:
                            energies.append(0)

                    x_offset = [x + i * bar_width for x in x_pos]
                    ax.bar(
                        x_offset,
                        energies,
                        bar_width,
                        label=f'({surface_name})',
                        color=COLORS[i % len(COLORS)],
                    )

                ax.set_xlabel('Model')
                ax.set_ylabel('Surface Energy (J/m²)')
                ax.set_title(f'{plot_info["metal"]} Surface Energies')
                tick_positions = [
                    x + bar_width * (len(surface_names) - 1) / 2 for x in x_pos
                ]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(model_names)
                ax.legend()

            ax.grid(True, linestyle='--', alpha=0.6)

        elif plot_type == 'high_temp_md':
            # High-temperature MD heatmap
            results = plot_info['results']
            model_names = list(results.keys())

            # Get temperatures from the stored data (fallback to default range)
            if results and model_names:
                first_result = results[model_names[0]]
                if 'defects_detected' in first_result:
                    temperatures = list(range(len(first_result['defects_detected'])))
                else:
                    temperatures = [1000]  # Default high temp
            else:
                temperatures = [1000]  # Default high temp

            # Prepare defect matrix
            defect_matrix = []
            for model_name in model_names:
                if 'defects_detected' in results[model_name]:
                    defect_matrix.append(results[model_name]['defects_detected'])
                else:
                    defect_matrix.append([False] * len(temperatures))

            # Create heatmap
            ax.imshow(
                defect_matrix,
                cmap='RdYlGn_r',
                aspect='auto',
                vmin=0,
                vmax=1,
                interpolation='nearest',
            )

            # Set labels
            ax.set_xticks(range(len(temperatures)))
            ax.set_xticklabels([f'{temp}K' for temp in temperatures], rotation=45)
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels(model_names)

            # Add text annotations
            for i in range(len(model_names)):
                for j in range(len(temperatures)):
                    text = 'FAIL' if defect_matrix[i][j] else 'PASS'
                    color = 'white' if defect_matrix[i][j] else 'black'
                    ax.text(
                        j,
                        i,
                        text,
                        ha='center',
                        va='center',
                        color=color,
                        fontweight='bold',
                    )

            ax.set_xlabel('Temperature')
            ax.set_ylabel('MLIP Model')
            ax.set_title(plot_info['title'])

        elif plot_type == 'database_sizes':
            # Database size comparison bar chart (final sizes only)
            run_names = plot_info['run_names']
            final_sizes = plot_info['final_sizes']

            # Create simple bar chart for final database sizes
            colors = [COLORS[i % len(COLORS)] for i in range(len(run_names))]
            bars = ax.bar(run_names, final_sizes, color=colors, alpha=0.8)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(final_sizes) * 0.01,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )

            ax.set_xlabel('AL Run')
            ax.set_ylabel('Number of Structures')
            ax.set_title(plot_info['title'])
            ax.set_xticklabels(run_names, rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.6)

        elif plot_type == 'learning_curves':
            # Learning curves plot - RMSE vs training database size
            learning_data = plot_info['data']

            # For learning curves, we need two subplots but tight_layout has issues
            # with nested gridspecs, so we'll create a simple dual plot
            ax.clear()  # Clear the main axis

            # Create two subplots manually within the axes area
            pos = ax.get_position()
            fig = ax.figure

            # Create energy subplot (top half)
            ax_energy = fig.add_axes(
                [pos.x0, pos.y0 + pos.height * 0.55, pos.width, pos.height * 0.4]
            )

            # Create force subplot (bottom half)
            ax_force = fig.add_axes(
                [pos.x0, pos.y0 + pos.height * 0.05, pos.width, pos.height * 0.4]
            )

            # Plot each AL run
            for i, (run_name, data) in enumerate(learning_data.items()):
                color = COLORS[i % len(COLORS)]
                iterations = data['iterations']
                energy_rmse = data['energy_rmse']
                force_rmse = data['force_rmse']

                # Skip iteration 0 (usually has RMSE = 0)
                if len(iterations) > 1:
                    iterations_plot = iterations[1:]
                    energy_rmse_plot = energy_rmse[1:]
                    force_rmse_plot = force_rmse[1:]
                else:
                    iterations_plot = iterations
                    energy_rmse_plot = energy_rmse
                    force_rmse_plot = force_rmse

                ax_energy.plot(
                    iterations_plot,
                    energy_rmse_plot,
                    'o-',
                    color=color,
                    label=run_name,
                    linewidth=2,
                    markersize=4,
                )
                ax_force.plot(
                    iterations_plot,
                    force_rmse_plot,
                    'o-',
                    color=color,
                    label=run_name,
                    linewidth=2,
                    markersize=4,
                )

            # Configure energy subplot
            ax_energy.set_ylabel('Energy RMSE\n(meV/atom)', fontsize=10)
            ax_energy.set_title('Active Learning Curves', fontsize=11, pad=10)
            ax_energy.grid(True, linestyle='--', alpha=0.6)
            ax_energy.legend(fontsize=8)
            ax_energy.tick_params(axis='both', which='major', labelsize=8)
            ax_energy.set_xticklabels([])  # Remove x-axis labels from top plot
            ax_energy.set_xlim(left=0)  # Start x-axis at 0

            # Configure force subplot
            ax_force.set_xlabel('Active Learning Iteration', fontsize=10)
            ax_force.set_ylabel('Force RMSE\n(meV/Å)', fontsize=10)
            ax_force.grid(True, linestyle='--', alpha=0.6)
            ax_force.legend(fontsize=8)
            ax_force.tick_params(axis='both', which='major', labelsize=8)
            ax_force.set_xlim(left=0)  # Start x-axis at 0

            # Hide the main axis since we're using subplots
            ax.set_visible(False)

        elif plot_type == 'melting_point':
            # Bar chart for melting points
            model_names = plot_info['model_names']
            melting_points = plot_info['values']

            # Create bar plot
            colors = [COLORS[i % len(COLORS)] for i in range(len(model_names))]
            bars = ax.bar(model_names, melting_points, color=colors)

            ax.set_xlabel('Model')
            ax.set_ylabel(plot_info['ylabel'])
            ax.set_title(plot_info['title'])
            ax.grid(True, linestyle='--', alpha=0.6)

            # Add value labels
            for bar, temp in zip(bars, melting_points, strict=True):
                if temp > 0:  # Only show label if melting point was determined
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(melting_points) * 0.01,
                        plot_info['value_format'].format(temp),
                        ha='center',
                        va='bottom',
                        fontsize=9,
                    )

        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Adjust layout and save
    plt.tight_layout()
    plot_path = args.output_dir / 'all_benchmarks_summary.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    custom_print(f'Multi-panel summary plot saved to {plot_path}', 'success')

    # Clear the plot to free memory
    plt.close(fig)


def run_final_db_size_benchmark(args, model_paths: list[pl.Path]):
    """Compares the final training database size for each AL run."""
    custom_print('Running Final Database Size Benchmark', 'info')
    benchmark_dir = args.output_dir / 'final_db_size'
    benchmark_dir.mkdir(exist_ok=True)

    # This benchmark only works with AiiDA PKs
    if not args.aiida_pks:
        custom_print(
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

            custom_print(f'Processing workchain {pk} ({run_name})...', 'info')

            # Get all children workchains to find the final one
            all_children = base_workchain.called
            if not all_children:
                custom_print(f'No children found for workchain {pk}', 'warn')
                continue

            # The final workchain should be the last one
            final_workchain = all_children[-1]
            custom_print(f'Found final workchain: {final_workchain.pk}', 'debug')

            # Extract database paths from the final workchain
            seed_db_size = 0
            final_db_size = 0

            # Get seed database size
            try:
                seed_db_path = final_workchain.inputs.seed_db_path.value
                custom_print(f'Loading seed database from: {seed_db_path}', 'debug')

                # Load and count structures in seed database
                seed_structures = ase_read(seed_db_path, index=':')
                seed_db_size = len(seed_structures)

            except (AttributeError, KeyError, Exception) as e:
                custom_print(
                    f'Could not load seed database for {run_name}: {e}', 'warn'
                )

            # Get final training database size
            try:
                training_db_path = final_workchain.inputs.training_db_path.value
                custom_print(
                    f'Loading training database from: {training_db_path}', 'debug'
                )

                # Load and count structures in training database
                training_structures = ase_read(training_db_path, index=':')
                final_db_size = len(training_structures)

            except (AttributeError, KeyError, Exception) as e:
                custom_print(
                    f'Could not load training database for {run_name}: {e}', 'warn'
                )

            # Store results
            results[run_name] = {
                'pk': pk,
                'seed_db_size': seed_db_size,
                'final_db_size': final_db_size,
                'structures_added': final_db_size - seed_db_size,
            }

            custom_print(
                f'{run_name}: Seed DB = {seed_db_size}, Final DB = {final_db_size}, '
                f'Added = {final_db_size - seed_db_size} structures',
                'info',
            )

        except Exception as e:
            custom_print(f'Failed to process workchain {pk}: {e}', 'error')
            continue

    # Save results to file
    if results:
        results_file = benchmark_dir / 'database_sizes.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Store plot data for final multi-panel figure
        global _plot_data
        run_names = list(results.keys())
        seed_sizes = [results[name]['seed_db_size'] for name in run_names]
        final_sizes = [results[name]['final_db_size'] for name in run_names]
        added_sizes = [results[name]['structures_added'] for name in run_names]

        _plot_data['database_sizes'] = {
            'type': 'database_sizes',
            'run_names': run_names,
            'seed_sizes': seed_sizes,
            'final_sizes': final_sizes,
            'added_sizes': added_sizes,
            'title': 'Final Training Database Sizes',
        }

        # Print summary
        custom_print('Database Size Summary:', 'info')
        for name in run_names:
            data = results[name]
            custom_print(
                f'  {name}: {data["seed_db_size"]} → {data["final_db_size"]} '
                f'(+{data["structures_added"]} structures)',
                'empty',
            )

        custom_print(f'Results saved to {results_file}', 'info')
    else:
        custom_print('No database size results to save.', 'warn')


def main():
    """Main function to run the evaluation."""
    global _ui_manager

    args = parse_arguments()
    init_logger(source='mdb_mlip_eval')

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    # Determine which benchmarks to run
    benchmarks_to_run = []
    benchmark_functions = {}

    if args.run_accuracy_test_set:
        benchmarks_to_run.append('Accuracy Test Set')
        benchmark_functions['Accuracy Test Set'] = (
            lambda: run_accuracy_test_set_benchmark(args, model_paths)
        )

    if args.run_elastic_properties:
        benchmarks_to_run.append('Elastic Properties')
        benchmark_functions['Elastic Properties'] = (
            lambda: run_elastic_properties_benchmark(args, model_paths)
        )

    if args.run_defect_formation_energy:
        benchmarks_to_run.append('Defect Formation Energy')
        benchmark_functions['Defect Formation Energy'] = (
            lambda: run_defect_formation_energy_benchmark(args, model_paths)
        )

    if args.run_surface_energies:
        benchmarks_to_run.append('Surface Energies')
        benchmark_functions['Surface Energies'] = (
            lambda: run_surface_energies_benchmark(args, model_paths)
        )

    if args.run_phonon_dispersion:
        benchmarks_to_run.append('Phonon Dispersion')
        benchmark_functions['Phonon Dispersion'] = (
            lambda: run_phonon_dispersion_benchmark(args, model_paths)
        )

    if args.run_energy_md:
        benchmarks_to_run.append('Energy MD')
        benchmark_functions['Energy MD'] = lambda: run_energy_md_benchmark(
            args, model_paths
        )

    if args.run_high_temp_md:
        benchmarks_to_run.append('High Temperature MD')
        benchmark_functions['High Temperature MD'] = lambda: run_high_temp_md_benchmark(
            args, model_paths
        )

    if args.run_melting_point:
        benchmarks_to_run.append('Melting Point')
        benchmark_functions['Melting Point'] = lambda: run_melting_point_benchmark(
            args, model_paths
        )

    if args.run_gsfe:
        benchmarks_to_run.append('GSFE')
        benchmark_functions['GSFE'] = lambda: run_gsfe_benchmark(args, model_paths)

    if args.run_learning_curves:
        benchmarks_to_run.append('Learning Curves')
        benchmark_functions['Learning Curves'] = lambda: run_learning_curves_benchmark(
            args, model_paths
        )

    if args.run_final_db_size:
        benchmarks_to_run.append('Final DB Size')
        benchmark_functions['Final DB Size'] = lambda: run_final_db_size_benchmark(
            args, model_paths
        )

    if not benchmarks_to_run:
        print('No benchmarks selected. Use --help to see available options.')
        return

    # Load models
    model_paths = [pl.Path(p) for p in args.model_files]

    # For models loaded from files, use the filename stem as display name
    global _model_display_names
    for model_path in model_paths:
        _model_display_names[str(model_path)] = model_path.stem

    if not args.no_rich_ui:
        # Use Rich UI
        _ui_manager = RichUIManager(benchmarks_to_run)

        with _ui_manager:
            custom_print('Initializing MLIP Benchmark Suite...', 'info')
            custom_print(f'Output directory: {args.output_dir}', 'empty')
            custom_print(
                f'Selected benchmarks: {", ".join(benchmarks_to_run)}', 'empty'
            )

            # Load models
            if args.aiida_pks:
                _ui_manager.log('Loading models from MDB Workchain...')
                for pk in args.aiida_pks:
                    path = load_model_from_aiida(pk, args.output_dir)
                    if path:
                        model_paths.append(path)

            if not model_paths:
                _ui_manager.log('No models specified. Exiting.')
                return

            # Run selected benchmarks
            for benchmark_name in benchmarks_to_run:
                try:
                    _ui_manager.start_benchmark(benchmark_name)
                    benchmark_functions[benchmark_name]()
                    _ui_manager.complete_benchmark(benchmark_name)
                except Exception as e:
                    _ui_manager.log(f"Benchmark '{benchmark_name}' failed: {e}")
                    _ui_manager.complete_benchmark(benchmark_name)

            _ui_manager.current_benchmark = 'All Benchmarks Completed'
            custom_print(' ', 'empty')
            custom_print('All selected benchmarks finished!', 'done')

            # Create final multi-panel plot if any data was collected
            _ui_manager.current_benchmark = 'Generating Summary Plot'
            create_final_multi_panel_plot(args)

            time.sleep(2)
    else:
        # Use plain text output (original behavior)
        custom_print('Initializing MLIP Benchmark Suite...', 'info')
        custom_print(f'Output directory: {args.output_dir}', 'info')
        custom_print(f'Selected benchmarks: {", ".join(benchmarks_to_run)}', 'info')

        # Load models
        if args.aiida_pks:
            custom_print('Loading models from MDB Workchain...', 'info')
            for pk in args.aiida_pks:
                path = load_model_from_aiida(pk, args.output_dir)
                if path:
                    model_paths.append(path)

        if not model_paths:
            custom_print('No models specified. Exiting.', 'error')
            return

        custom_print(f'Loaded {len(model_paths)} model(s)', 'info')
        for path in model_paths:
            display_name = get_model_display_name(path)
            custom_print(f'  - {display_name}', 'info')

        # Run selected benchmarks
        for benchmark_name in benchmarks_to_run:
            try:
                custom_print(f'Starting benchmark: {benchmark_name}', 'info')
                benchmark_functions[benchmark_name]()
                custom_print(f'Completed benchmark: {benchmark_name}', 'done')
            except Exception as e:
                custom_print(f"Benchmark '{benchmark_name}' failed: {e}", 'error')

        custom_print('', 'empty')
        custom_print('All selected benchmarks finished!', 'done')

        # Create final multi-panel plot if any data was collected
        create_final_multi_panel_plot(args)

    # Cleanup log file
    _cleanup_log_file()


if __name__ == '__main__':
    main()
