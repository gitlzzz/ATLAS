"""Reporting and plotting for DFT benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PHASE_COLORS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]


def plot_convergence(
    convergence_results: dict[str, dict[str, Any]],
    threshold_meV: float,
    output_dir: Path,
) -> list[Path]:
    """Generate one convergence plot per benchmark parameter.

    Each plot shows energy difference vs parameter value with one line
    per phase, a threshold line, and a marker on the selected value.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for param_name, phases in convergence_results.items():
        if not phases:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        phase_names = sorted(phases.keys())

        for i, phase in enumerate(phase_names):
            data = phases[phase]
            vals = data['values']
            diffs = data['diffs_meV']
            color = PHASE_COLORS[i % len(PHASE_COLORS)]

            plot_vals = [v for v, d in zip(vals, diffs, strict=False) if d is not None]
            plot_diffs = [d for d in diffs if d is not None]

            ax.plot(plot_vals, plot_diffs, 'o-', color=color, label=phase, markersize=5)

            conv_val = data['converged_value']
            if conv_val in plot_vals:
                idx = plot_vals.index(conv_val)
                ax.plot(
                    conv_val,
                    plot_diffs[idx],
                    's',
                    color=color,
                    markersize=10,
                    zorder=5,
                    markeredgecolor='black',
                    markeredgewidth=1.5,
                )

        ax.axhline(
            y=threshold_meV,
            color='red',
            linestyle='--',
            linewidth=1.2,
            label=f'Threshold ({threshold_meV} meV)',
        )

        param_label = (
            param_name.upper() if param_name.lower() != 'kspacing' else 'K-spacing'
        )
        units = _param_units(param_name)
        ax.set_xlabel(f'{param_label}{units}', fontsize=12)
        ax.set_ylabel('|ΔE| (meV/atom)', fontsize=12)
        ax.set_title(f'{param_label} convergence test', fontsize=14)
        ax.legend(fontsize=9, loc='best')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = output_dir / f'convergence_{param_name}.png'
        fig.savefig(path, dpi=300)
        plt.close(fig)
        saved.append(path)

    return saved


def _param_units(param_name: str) -> str:
    units = {
        'kspacing': ' (2π/Å)',
        'encut': ' (eV)',
        'sigma': ' (eV)',
        'ediff': ' (eV)',
    }
    return units.get(param_name.lower(), '')


def print_summary_table(
    convergence_results: dict[str, dict[str, Any]],
    threshold_meV: float,
) -> None:
    """Print a summary table to the console."""
    for param_name, phases in convergence_results.items():
        if not phases:
            continue

        param_label = (
            param_name.upper() if param_name.lower() != 'kspacing' else 'K-spacing'
        )
        units = _param_units(param_name)

        print(f'\n{"=" * 70}')
        print(
            f'  {param_label} benchmark results (threshold: {threshold_meV} meV/atom)'
        )
        print(f'{"=" * 70}')
        print(f'  {"Phase":<20} {"Reference":<12} {"Selected":<12} {"ΔE (meV)":<10}')
        print(f'  {"-" * 54}')

        for phase in sorted(phases.keys()):
            data = phases[phase]
            ref = data['reference_value']
            conv = data['converged_value']
            # Find diff at converged value
            try:
                idx = data['values'].index(conv)
                diff = data['diffs_meV'][idx]
                diff_str = f'{diff:.3f}' if diff is not None else 'N/A'
            except (ValueError, IndexError):
                diff_str = 'N/A'

            ref_str = f'{ref}{units}'
            conv_str = f'{conv}{units}'
            print(f'  {phase:<20} {ref_str:<12} {conv_str:<12} {diff_str:<10}')

        print()


def generate_full_report(
    convergence_results: dict[str, dict[str, Any]],
    threshold_meV: float,
    output_dir: Path,
    toml_snippet: str,
) -> Path:
    """Orchestrate all reporting: plots, tables, and TOML snippet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = plot_convergence(convergence_results, threshold_meV, output_dir)
    for p in plot_paths:
        print(f'  Plot saved: {p}')

    print_summary_table(convergence_results, threshold_meV)

    print('\n' + '=' * 70)
    print('  Recommended settings (copy to your DFT config):')
    print('=' * 70)
    print(toml_snippet)

    snippet_path = output_dir / 'recommended_settings.toml'
    snippet_path.write_text(toml_snippet)
    print(f'\n  Saved to: {snippet_path}')

    return output_dir
