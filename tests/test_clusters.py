"""Tests for the clusters module."""

from unittest.mock import Mock

import pytest

# Import atlas first to resolve circular imports
import atlas  # noqa: F401
import atlas.core.clusters as atl_clusters
from atlas.core import exceptions as exc


class TestAtomData:
    """Tests for ATOM_DATA constant."""

    def test_atom_data_has_cu(self):
        """Cu is present in the cluster atom data table."""
        assert 'Cu' in atl_clusters.ATOM_DATA

    def test_atom_data_cu_lattice(self):
        """Cu entry has a positive lattice parameter 'a'."""
        assert 'a' in atl_clusters.ATOM_DATA['Cu']
        assert atl_clusters.ATOM_DATA['Cu']['a'] > 0

    def test_atom_data_cu_dimer_dist(self):
        """Cu entry has a positive dimer distance."""
        assert 'dimer_dist' in atl_clusters.ATOM_DATA['Cu']
        assert atl_clusters.ATOM_DATA['Cu']['dimer_dist'] > 0

    def test_atom_data_is_dict(self):
        """ATOM_DATA is a dictionary."""
        assert isinstance(atl_clusters.ATOM_DATA, dict)


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_clust_list_exists(self):
        """CLUST_LIST is a list of cluster data folders."""
        assert hasattr(atl_clusters, 'CLUST_LIST')
        assert isinstance(atl_clusters.CLUST_LIST, list)

    def test_data_path_exists(self):
        """DATA_PATH points to the cluster data directory."""
        assert hasattr(atl_clusters, 'DATA_PATH')

    def test_latt_constant(self):
        """LATT is a 3-element tuple of lattice constants."""
        assert hasattr(atl_clusters, 'LATT')
        assert len(atl_clusters.LATT) == 3

    def test_max_vac_constant(self):
        """MAX_VAC is a positive vacuum thickness."""
        assert hasattr(atl_clusters, 'MAX_VAC')
        assert atl_clusters.MAX_VAC > 0


class TestMakeCleanCluster:
    """Tests for make_clean_cluster function."""

    def test_make_clean_cluster_unknown_element_raises(self):
        """Test that unknown element raises AtomNotFoundForCluster."""
        mock_phase = Mock()
        mock_phase.cluster_elem.symbol = 'Fe'
        with pytest.raises(exc.AtomNotFoundForCluster):
            atl_clusters.make_clean_cluster(None, 2, mock_phase)


class TestMakeCleanDimer:
    """Tests for make_clean_dimer function."""

    def test_make_clean_dimer_unknown_element_raises(self):
        """Test that unknown element raises AtomNotFoundForCluster."""
        mock_phase = Mock()
        mock_phase.cluster_elem.symbol = 'Fe'
        with pytest.raises(exc.AtomNotFoundForCluster):
            atl_clusters.make_clean_dimer(None, mock_phase)


class TestApplyGaussPerturbList:
    """Tests for apply_gauss_perturb_list function."""

    def test_apply_gauss_perturb_list_returns_list(self):
        """Test that perturbation returns a list of Structure objects."""
        from pymatgen.core import Structure as pmg_struct

        # Create a pymatgen structure and wrap it in ATL Structure
        lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
        coords = [[0, 0, 0], [1, 0, 0]]
        pmg = pmg_struct(lattice, ['Cu', 'Cu'], coords)
        atl_struct = atl_clusters.atl_struct.Structure(structure=pmg)
        result = atl_clusters.apply_gauss_perturb_list(
            repeat=2, cluster_list=[atl_struct], center=0.04
        )
        assert isinstance(result, list)
        assert len(result) == 2

    def test_apply_gauss_perturb_list_preserves_atoms(self):
        """Test that perturbed structures have same number of atoms."""
        from pymatgen.core import Structure as pmg_struct

        lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
        coords = [[0, 0, 0], [1, 0, 0]]
        pmg = pmg_struct(lattice, ['Cu', 'Cu'], coords)
        atl_struct = atl_clusters.atl_struct.Structure(structure=pmg)
        result = atl_clusters.apply_gauss_perturb_list(
            repeat=1, cluster_list=[atl_struct], center=0.04
        )
        # Each Structure has a .structure attribute (pymatgen)
        assert len(result[0].structure) == len(atl_struct.structure)
