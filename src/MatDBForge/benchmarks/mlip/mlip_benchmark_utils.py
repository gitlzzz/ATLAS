"""General utilities for the MLIP benchmark suite in MDBForge."""
import argparse
import atexit
import math
import pathlib as pl
import re
import shutil
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
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

from MatDBForge.core.code_utils import (
    custom_print as _original_custom_print,
)

# Global UI manager instance
_ui_manager = None
_log_file = None
_plot_data = {}  # Store data for final multi-panel plot
_model_display_names = {}  # Maps model path to display name

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


def get_model_display_names():
    """Get the global model display names dictionary."""
    global _model_display_names
    return _model_display_names


def set_model_display_name(path, name):
    """Set a model display name."""
    global _model_display_names
    _model_display_names[path] = name


def get_plot_data():
    """Get the global plot data dictionary."""
    global _plot_data
    return _plot_data


def set_plot_data(key, value):
    """Set a value in the global plot data dictionary."""
    global _plot_data
    _plot_data[key] = value


def clear_plot_data():
    """Clear the global plot data dictionary."""
    global _plot_data
    _plot_data.clear()


def get_ui_manager():
    """Get the global UI manager instance."""
    global _ui_manager
    return _ui_manager


def set_ui_manager(ui_manager):
    """Set the global UI manager instance."""
    global _ui_manager
    _ui_manager = ui_manager


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

            # Use matplotlib's gridspec for proper subplot management
            from matplotlib.gridspec import GridSpecFromSubplotSpec

            # Clear and hide the main axis to prevent background interference
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Create a 2x1 gridspec within this subplot
            gs = GridSpecFromSubplotSpec(
                2, 1, ax.get_subplotspec(), height_ratios=[1, 1], hspace=0.3
            )

            # Create the two subaxes
            ax_energy = ax.figure.add_subplot(gs[0])
            ax_force = ax.figure.add_subplot(gs[1])

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
            ax_energy.set_ylabel('Energy RMSE\n(meV/atom)', fontsize=9, labelpad=2)
            ax_energy.set_title('Active Learning Curves', fontsize=11, pad=10)
            ax_energy.grid(True, linestyle='--', alpha=0.6)
            ax_energy.legend(fontsize=8)
            ax_energy.tick_params(axis='both', which='major', labelsize=8)
            ax_energy.set_xticklabels([])  # Remove x-axis labels from top plot
            ax_energy.set_xlim(left=0)  # Start x-axis at 0

            # Configure force subplot
            ax_force.set_xlabel('Active Learning Iteration', fontsize=10)
            ax_force.set_ylabel('Force RMSE\n(meV/Å)', fontsize=9, labelpad=2)
            ax_force.grid(True, linestyle='--', alpha=0.6)
            ax_force.legend(fontsize=8)
            ax_force.tick_params(axis='both', which='major', labelsize=8)
            ax_force.set_xlim(left=0)  # Start x-axis at 0

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
