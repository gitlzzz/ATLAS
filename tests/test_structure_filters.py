"""Tests for the structure-filter utilities (pure ASE/geometry)."""

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111

from atlas.core.filtering.structure_filters import (
    apply_filter_evaporation,
    get_available_filters,
    get_coord_nums,
    get_max_layer_distance,
)


class TestGetAvailableFilters:
    """The dynamic filter registry."""

    def test_returns_callable_dict_with_known_filters(self):
        filters = get_available_filters()
        assert isinstance(filters, dict)
        assert all(callable(f) for f in filters.values())
        # Names have the 'apply_filter_' prefix stripped.
        assert 'evaporation' in filters
        assert 'exploding_structures' in filters


class TestGetCoordNums:
    """Coordination numbers via ASE NeighborList."""

    def test_bulk_fcc_is_fully_connected(self):
        cu = bulk('Cu', 'fcc', a=3.6, cubic=True)
        _conn, has_disconnected, coord_nums = get_coord_nums(cu)
        assert has_disconnected is False or has_disconnected == np.False_
        assert len(coord_nums) == len(cu)
        assert np.all(coord_nums > 0)

    def test_isolated_atom_is_disconnected(self):
        atom = Atoms('Cu', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
        _conn, has_disconnected, coord_nums = get_coord_nums(atom)
        assert bool(has_disconnected) is True
        assert coord_nums.tolist() == [0]


class TestApplyFilterEvaporation:
    """Fast z-thickness evaporation check."""

    def test_thin_structure_passes(self):
        atoms = Atoms('H3', positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        # Returns a numpy bool; assert via truthiness.
        assert not apply_filter_evaporation(atoms, max_allowed_thickness=5.0)

    def test_thick_structure_flagged(self):
        atoms = Atoms('H3', positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        # z thickness is 2.0; threshold 1.0 -> flagged for removal.
        assert apply_filter_evaporation(atoms, max_allowed_thickness=1.0)


class TestGetMaxLayerDistance:
    """Inter-layer spacing of a slab."""

    def test_positive_spacing_for_slab(self):
        slab = fcc111('Cu', size=(1, 1, 4), vacuum=10.0)
        dist = get_max_layer_distance(slab)
        assert isinstance(dist, float)
        assert dist > 0.0
