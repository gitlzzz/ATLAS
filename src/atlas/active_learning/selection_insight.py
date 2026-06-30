"""Quantify what the active-learning loop selects and why.

Pure, dependency-light helpers (numpy/ase/shapely only) for summarizing a batch
of selected structures and the latent-space domain. Used by the reporting layer
to track, per AL iteration: latent-space hull area/coverage, and a breakdown of
the selected structures by selection reason, structure type, and composition.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms
from shapely.geometry import MultiPolygon, Polygon

# Structure-type info flags, in priority order.
_STRUCT_TYPE_FLAGS = ('bulk', 'surface', 'cluster', 'isolated_atom', 'init_md')


def hull_area(hull) -> float:
    """Return the area of a concave hull.

    Accepts a shapely ``Polygon``/``MultiPolygon`` (uses ``.area``) or an
    ``(N, 2)`` array / list of vertex coordinates (builds a polygon). Returns
    ``0.0`` for degenerate input (fewer than 3 vertices).
    """
    if isinstance(hull, (Polygon, MultiPolygon)):
        return float(hull.area)

    coords = np.asarray(hull, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 3:
        return 0.0
    return float(Polygon(coords).area)


def coverage(area: float, n_structures: int) -> float:
    """Latent-space area normalized by dataset size (area per structure)."""
    if n_structures <= 0:
        return 0.0
    return float(area) / float(n_structures)


def _structure_type(struct: Atoms) -> str:
    """Infer the structure type from ASE ``.info`` flags, else 'other'."""
    info = struct.info
    for flag in _STRUCT_TYPE_FLAGS:
        if info.get(flag):
            return flag
    return 'other'


def compute_selection_insights(selected_structures: list[Atoms]) -> dict:
    """Summarize a batch of selected structures.

    Parameters
    ----------
    selected_structures : list[Atoms]
        Structures chosen in one AL iteration. Each may carry a
        ``selection_reason`` key in ``.info`` (set by the selection routines).

    Returns
    -------
    dict
        ``{'n_selected', 'by_reason', 'by_type', 'composition'}`` where the
        latter three are plain ``{label: count}`` dicts.
    """
    by_reason: Counter = Counter()
    by_type: Counter = Counter()
    composition: Counter = Counter()

    for struct in selected_structures:
        by_reason[struct.info.get('selection_reason', 'unspecified')] += 1
        by_type[_structure_type(struct)] += 1
        composition.update(struct.get_chemical_symbols())

    return {
        'n_selected': len(selected_structures),
        'by_reason': dict(by_reason),
        'by_type': dict(by_type),
        'composition': dict(composition),
    }


def format_selection_report(insights: dict) -> str:
    """Render a compact one-batch text summary from ``compute_selection_insights``."""
    lines = [f'Selected {insights["n_selected"]} structures:']
    for title, key in (
        ('  by reason', 'by_reason'),
        ('  by type', 'by_type'),
        ('  composition', 'composition'),
    ):
        items = ', '.join(f'{k}={v}' for k, v in sorted(insights.get(key, {}).items()))
        lines.append(f'{title}: {items or "-"}')
    return '\n'.join(lines)
