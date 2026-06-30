"""Tests for the selection-insight quantification helpers."""

import numpy as np
from ase import Atoms
from shapely.geometry import Polygon

from atlas.active_learning.selection_insight import (
    compute_selection_insights,
    coverage,
    format_selection_report,
    hull_area,
)


class TestHullArea:
    """hull_area on polygons and raw coordinates."""

    def test_polygon_area(self):
        square = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
        assert np.isclose(hull_area(square), 4.0)

    def test_coords_array_area(self):
        coords = [[0, 0], [0, 1], [1, 1], [1, 0]]
        assert np.isclose(hull_area(coords), 1.0)

    def test_degenerate_returns_zero(self):
        assert hull_area([[0, 0], [1, 1]]) == 0.0


class TestCoverage:
    def test_normalizes_by_count(self):
        assert coverage(10.0, 5) == 2.0

    def test_zero_structures_safe(self):
        assert coverage(10.0, 0) == 0.0


class TestComputeSelectionInsights:
    """Aggregation of a selected batch."""

    def _batch(self):
        a = Atoms('Cu2', positions=[[0, 0, 0], [1, 1, 1]])
        a.info.update({'selection_reason': 'fps', 'bulk': True})
        b = Atoms('CuO', positions=[[0, 0, 0], [1, 0, 0]])
        b.info.update({'selection_reason': 'extrapolating', 'surface': True})
        c = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1]])
        c.info.update({'selection_reason': 'fps', 'cluster': True})
        return [a, b, c]

    def test_counts(self):
        ins = compute_selection_insights(self._batch())
        assert ins['n_selected'] == 3
        assert ins['by_reason'] == {'fps': 2, 'extrapolating': 1}
        assert ins['by_type'] == {'bulk': 1, 'surface': 1, 'cluster': 1}
        assert ins['composition'] == {'Cu': 3, 'O': 3}

    def test_missing_reason_is_unspecified(self):
        plain = Atoms('Cu', positions=[[0, 0, 0]])
        ins = compute_selection_insights([plain])
        assert ins['by_reason'] == {'unspecified': 1}
        assert ins['by_type'] == {'other': 1}

    def test_empty_batch(self):
        ins = compute_selection_insights([])
        assert ins == {
            'n_selected': 0,
            'by_reason': {},
            'by_type': {},
            'composition': {},
        }

    def test_format_report_is_string(self):
        text = format_selection_report(compute_selection_insights(self._batch()))
        assert 'Selected 3 structures' in text
        assert 'fps=2' in text
