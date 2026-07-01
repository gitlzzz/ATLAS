"""Tests for the adsorption structure-building helpers (pure ASE/pymatgen)."""

import numpy as np
import pytest
from ase.build import bulk, fcc111

from atlas.core.adsorption import (
    build_adsorbate_molecule,
    find_adsorption_sites,
    generate_adsorbed_structures,
    place_cluster_on_slab,
    place_molecule_on_slab,
)


class TestBuildAdsorbateMolecule:
    def test_custom_library_oh(self):
        mol = build_adsorbate_molecule('OH')
        assert mol.get_chemical_symbols() == ['O', 'H']
        # Anchored at z = 0.
        assert np.isclose(mol.get_positions()[:, 2].min(), 0.0)

    def test_g2_fallback_h2o(self):
        mol = build_adsorbate_molecule('H2O')
        assert sorted(mol.get_chemical_symbols()) == ['H', 'H', 'O']
        assert np.isclose(mol.get_positions()[:, 2].min(), 0.0)

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError):
            build_adsorbate_molecule('Zzz9')


class TestPlacement:
    def test_place_molecule_above_slab(self):
        slab = fcc111('Cu', size=(2, 2, 3), vacuum=8.0)
        mol = build_adsorbate_molecule('O')
        n_before = len(slab)
        out = place_molecule_on_slab(slab, mol, site_xy=np.array([1.0, 1.0]), height=2.0)
        assert len(out) == n_before + 1
        slab_top = slab.get_positions()[:, 2].max()
        # The added atom (last) sits ~height above the slab top.
        assert out.get_positions()[-1, 2] >= slab_top + 1.9

    def test_place_cluster_above_slab(self):
        slab = fcc111('Cu', size=(3, 3, 3), vacuum=10.0)
        clus = bulk('Cu', 'fcc', a=3.6, cubic=True)  # small 4-atom cell as a stand-in
        n_before = len(slab)
        out = place_cluster_on_slab(clus, slab, height=2.5)
        assert len(out) == n_before + len(clus)
        slab_top = slab.get_positions()[:, 2].max()
        cluster_z = out.get_positions()[n_before:, 2]
        assert cluster_z.min() >= slab_top + 2.4


class TestSiteFinding:
    def test_find_sites_returns_coords(self):
        slab = fcc111('Cu', size=(2, 2, 3), vacuum=8.0)
        sites = find_adsorption_sites(slab, site_types=['ontop'])
        assert 'ontop' in sites
        assert len(sites['ontop']) >= 1
        # Each site is a 3-vector.
        assert np.asarray(sites['ontop'][0]).shape == (3,)


class TestGenerateAdsorbedStructures:
    def test_generates_tagged_structures(self):
        slab = fcc111('Cu', size=(2, 2, 3), vacuum=8.0)
        results = generate_adsorbed_structures(
            slab,
            species_list=['O', 'H'],
            site_types=['ontop'],
            height=1.8,
            max_per_slab=3,
            rng=np.random.default_rng(0),
        )
        assert 1 <= len(results) <= 3
        for atoms, ads_type, site_type in results:
            assert len(atoms) == len(slab) + len(build_adsorbate_molecule(ads_type))
            assert ads_type in ('O', 'H')
            assert site_type == 'ontop'
