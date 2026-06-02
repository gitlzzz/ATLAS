"""Tests for the Structure class hierarchy."""

import uuid

import pytest
from ase import Atoms

# Import atlas first to fully resolve all modules
import atlas  # noqa: F401

# Now we can import from submodules without circular import issues
import atlas.core.structure as atl_struct


class TestStructureInitialization:
    """Tests for Structure base class construction and default values."""

    def test_default_unique_id(self):
        """A new Structure without unique_id gets a UUID4 assigned."""
        s = atl_struct.Structure()
        assert s.unique_id is not None
        uuid.UUID(s.unique_id)

    def test_provided_unique_id(self):
        """A custom unique_id passed to the constructor is preserved."""
        s = atl_struct.Structure(unique_id='my-custom-id')
        assert s.unique_id == 'my-custom-id'

    def test_accepts_pmg_structure(self, mock_pmg_structure):
        """A pymatgen Structure can be passed as the structure argument."""
        s = atl_struct.Structure(structure=mock_pmg_structure)
        assert s.structure is mock_pmg_structure

    def test_rejects_non_pmg_structure(self):
        """Passing a non-pymatgen structure raises TypeError."""
        with pytest.raises(TypeError, match='not a pymatgen structure'):
            atl_struct.Structure(structure='not_a_structure')

    def test_none_structure(self):
        """Passing structure=None leaves the structure attribute as None."""
        s = atl_struct.Structure(structure=None)
        assert s.structure is None

    def test_default_flags(self):
        """All boolean flags default to False, perturb/base default to None."""
        s = atl_struct.Structure()
        assert s.bulk is False
        assert s.surface is False
        assert s.cluster is False
        assert s.isolated_atom is False
        assert s.calc_performed is False
        assert s.perturb is None
        assert s.base is None

    def test_formula_set_from_structure(self, mock_pmg_structure):
        """The formula attribute is derived from the pymatgen structure."""
        s = atl_struct.Structure(structure=mock_pmg_structure)
        assert s.formula is not None

    def test_no_formula_without_structure(self):
        """Without a structure, the formula attribute is None."""
        s = atl_struct.Structure()
        assert s.formula is None

    def test_targeted_modification(self):
        """The targeted_modification attribute can be set via constructor."""
        s = atl_struct.Structure(targeted_modification='substitution')
        assert s.targeted_modification == 'substitution'

    def test_al_loop_step_default(self):
        """The al_loop_step defaults to 0."""
        s = atl_struct.Structure()
        assert s.al_loop_step == 0

    def test_al_loop_step_custom(self):
        """A custom al_loop_step is preserved."""
        s = atl_struct.Structure(al_loop_step=5)
        assert s.al_loop_step == 5


class TestBulk:
    """Tests for the Bulk subclass (bulk=True flag, surface/cluster=False)."""

    def test_bulk_defaults(self, mock_pmg_structure):
        """Bulk sets bulk=True and surface/cluster=False."""
        b = atl_struct.Bulk(structure=mock_pmg_structure)
        assert b.bulk is True
        assert b.surface is False
        assert b.cluster is False

    def test_bulk_with_kwargs(self, mock_pmg_structure):
        """Bulk accepts extra kwargs like phase and base."""
        b = atl_struct.Bulk(structure=mock_pmg_structure, phase='alpha', base=True)
        assert b.bulk is True
        assert b.phase == 'alpha'
        assert b.base is True


class TestSurface:
    """Tests for the Surface subclass (surface=True flag, surface_miller parsing)."""

    def test_surface_defaults(self, mock_pmg_structure):
        """Surface sets surface=True and bulk/cluster=False."""
        s = atl_struct.Surface(structure=mock_pmg_structure)
        assert s.surface is True
        assert s.bulk is False
        assert s.cluster is False

    def test_surface_miller_from_string(self, mock_pmg_structure):
        """A space-separated Miller string '1 1 1' is parsed to [1, 1, 1]."""
        s = atl_struct.Surface(structure=mock_pmg_structure, surface_miller='1 1 1')
        assert s.surface_miller == [1, 1, 1]

    def test_surface_miller_from_tuple(self, mock_pmg_structure):
        """A tuple Miller index (1, 1, 0) is stored as-is."""
        s = atl_struct.Surface(structure=mock_pmg_structure, surface_miller=(1, 1, 0))
        assert s.surface_miller == (1, 1, 0)

    def test_surface_with_kwargs(self, mock_pmg_structure):
        """Surface accepts phase, base, and surface_miller kwargs."""
        s = atl_struct.Surface(
            structure=mock_pmg_structure,
            phase='alpha',
            base=True,
            surface_miller='1 0 0',
        )
        assert s.surface is True
        assert s.phase == 'alpha'
        assert s.base is True
        assert s.surface_miller == [1, 0, 0]


class TestCluster:
    """Tests for the Cluster subclass (cluster=True flag)."""

    def test_cluster_defaults(self, mock_pmg_structure):
        """Cluster sets cluster=True and surface/bulk=False."""
        c = atl_struct.Cluster(structure=mock_pmg_structure)
        assert c.cluster is True
        assert c.surface is False
        assert c.bulk is False

    def test_cluster_with_kwargs(self, mock_pmg_structure):
        """Cluster accepts extra kwargs like phase and perturb."""
        c = atl_struct.Cluster(
            structure=mock_pmg_structure, phase='alpha', perturb=True
        )
        assert c.cluster is True
        assert c.phase == 'alpha'
        assert c.perturb is True


class TestIsolatedAtom:
    """Tests for the IsolatedAtom subclass (isolated_atom=True flag)."""

    def test_isolated_atom_defaults(self):
        """IsolatedAtom sets isolated_atom=True and other type flags=False."""
        ia = atl_struct.IsolatedAtom()
        assert ia.isolated_atom is True
        assert ia.cluster is False
        assert ia.surface is False
        assert ia.bulk is False


class TestToBulk:
    """Tests for the to_bulk() conversion method on Structure."""

    def test_converts_to_bulk(self, mock_pmg_structure):
        """to_bulk() returns a Bulk instance with the same unique_id and phase."""
        s = atl_struct.Structure(structure=mock_pmg_structure, phase='alpha', base=True)
        b = s.to_bulk()
        assert isinstance(b, atl_struct.Bulk)
        assert b.bulk is True
        assert b.phase == 'alpha'
        assert b.base is True
        assert b.unique_id == s.unique_id

    def test_preserves_attributes(self, mock_pmg_structure):
        """to_bulk() preserves material_name, material_id, calc_energy, etc."""
        s = atl_struct.Structure(
            structure=mock_pmg_structure,
            material_name='test_mat',
            material_id='mp-123',
            phase='beta',
            perturb=True,
            calc_performed=True,
            calc_energy=-10.5,
        )
        b = s.to_bulk()
        assert b.material_name == 'test_mat'
        assert b.material_id == 'mp-123'
        assert b.phase == 'beta'
        assert b.perturb is True
        assert b.calc_performed is True
        assert b.calc_energy == -10.5


class TestToSurface:
    """Tests for the to_surface() conversion method on Structure."""

    def test_converts_to_surface(self, mock_pmg_structure):
        """to_surface() returns a Surface instance with surface=True."""
        s = atl_struct.Structure(structure=mock_pmg_structure, phase='alpha')
        surf = s.to_surface()
        assert isinstance(surf, atl_struct.Surface)
        assert surf.surface is True


class TestToCluster:
    """Tests for the to_cluster() conversion method on Structure."""

    def test_converts_to_cluster(self, mock_pmg_structure):
        """to_cluster() returns a Cluster instance with cluster=True."""
        s = atl_struct.Structure(structure=mock_pmg_structure, phase='alpha')
        c = s.to_cluster()
        assert isinstance(c, atl_struct.Cluster)
        assert c.cluster is True


class TestFromAseAtoms:
    """Tests for Structure.from_ase_atoms() factory method."""

    def test_basic_conversion(self, mock_atoms_simple):
        """from_ase_atoms builds a Structure from an ASE Atoms with .info metadata."""
        s = atl_struct.Structure.from_ase_atoms(mock_atoms_simple)
        assert isinstance(s, atl_struct.Structure)
        assert s.unique_id == 'test-uuid-0001'
        assert s.material_name == 'test_structure'
        assert s.phase == 'alpha'
        assert s.bulk is True
        assert s.base is True

    def test_isolated_atom_detection(self, mock_atoms_simple):
        """The atl_struct_type='isolated_atom' flag sets isolated_atom=True."""
        mock_atoms_simple.info['atl_struct_type'] = 'isolated_atom'
        s = atl_struct.Structure.from_ase_atoms(mock_atoms_simple)
        assert s.isolated_atom is True

    def test_extra_metadata(self, mock_atoms_simple):
        """Metadata like perturb/deformation/vacancy is carried over from .info."""
        mock_atoms_simple.info['perturb'] = True
        mock_atoms_simple.info['deformation'] = True
        mock_atoms_simple.info['vacancy'] = False
        s = atl_struct.Structure.from_ase_atoms(mock_atoms_simple)
        assert s.perturb is True
        assert s.deformation is True
        assert s.vacancy is False

    def test_defaults_for_missing_info(self, mock_atoms_simple):
        """Missing .info keys default to False rather than raising."""
        del mock_atoms_simple.info['base']
        s = atl_struct.Structure.from_ase_atoms(mock_atoms_simple)
        assert s.base is False


class TestToAseAtoms:
    """Tests for Structure.to_ase_atoms() conversion method."""

    def test_returns_ase_atoms(self, mock_pmg_structure):
        """to_ase_atoms returns an ASE Atoms object."""
        s = atl_struct.Structure(structure=mock_pmg_structure, material_name='test')
        ase_atoms = s.to_ase_atoms()
        assert isinstance(ase_atoms, Atoms)

    def test_metadata_in_info(self, mock_pmg_structure):
        """Structure attributes are copied into the ASE Atoms .info dict."""
        s = atl_struct.Structure(
            structure=mock_pmg_structure,
            material_name='test_mat',
            phase='alpha',
            bulk=True,
            unique_id='test-uuid',
        )
        ase_atoms = s.to_ase_atoms()
        assert ase_atoms.info['unique_id'] == 'test-uuid'
        assert ase_atoms.info['material_name'] == 'test_mat'
        assert ase_atoms.info['phase'] == 'alpha'
        assert ase_atoms.info['bulk'] is True

    def test_roundtrip_preserves_info(self, mock_pmg_structure):
        """to_ase_atoms -> from_ase_atoms preserves phase, bulk and base flags.

        Note: unique_id and material_name do not survive the roundtrip
        because to_ase_atoms stores self.__dict__ directly while
        from_ase_atoms reads specific keys ('atl_id', 'struct_name').
        """
        original = atl_struct.Structure(
            structure=mock_pmg_structure,
            material_name='roundtrip_test',
            phase='alpha',
            bulk=True,
            base=True,
            perturb=True,
            calc_performed=True,
            calc_energy=-15.0,
        )
        ase_atoms = original.to_ase_atoms()
        reconstructed = atl_struct.Structure.from_ase_atoms(ase_atoms)
        assert reconstructed.phase == original.phase
        assert reconstructed.bulk == original.bulk
        assert reconstructed.base == original.base


class TestSaveToDb:
    """Tests for Structure.save_to_db() DataFrame serialization."""

    def test_saves_to_empty_dataframe(self, mock_pmg_structure):
        """A single structure can be saved into an empty DataFrame."""
        import pandas as pd

        s = atl_struct.Structure(
            structure=mock_pmg_structure, material_name='test', phase='alpha'
        )
        db_obj = s.save_to_db(pd.DataFrame())
        assert isinstance(db_obj, pd.DataFrame)
        assert len(db_obj) == 1
        assert db_obj.iloc[0]['material_name'] == 'test'

    def test_appends_to_existing_dataframe(self, mock_pmg_structure):
        """Multiple structures can be appended to an existing DataFrame."""
        import pandas as pd

        s1 = atl_struct.Structure(
            structure=mock_pmg_structure, material_name='first', phase='alpha'
        )
        s2 = atl_struct.Structure(
            structure=mock_pmg_structure, material_name='second', phase='beta'
        )
        df = s1.save_to_db(pd.DataFrame())
        df = s2.save_to_db(df)
        assert len(df) == 2
        assert df.iloc[0]['material_name'] == 'first'
        assert df.iloc[1]['material_name'] == 'second'

    def test_bool_columns_are_bool(self, mock_pmg_structure):
        """Boolean metadata columns have dtype bool after save_to_db."""
        import pandas as pd

        s = atl_struct.Structure(
            structure=mock_pmg_structure,
            bulk=True,
            surface=False,
            cluster=False,
            perturb=False,
            calc_performed=True,
        )
        db_obj = s.save_to_db(pd.DataFrame())
        bool_cols = [
            'perturb',
            'deformation',
            'base',
            'isolated_atom',
            'bulk',
            'surface',
            'cluster',
            'calc_performed',
            'replacement',
            'vacancy',
            'init_md',
        ]
        for col in bool_cols:
            assert db_obj[col].dtype == bool

    def test_phase_from_string(self, mock_pmg_structure):
        """Phase is stored as a string in the DataFrame."""
        import pandas as pd

        s = atl_struct.Structure(
            structure=mock_pmg_structure, material_name='test', phase='alpha'
        )
        db_obj = s.save_to_db(pd.DataFrame())
        assert db_obj.iloc[0]['phase'] == 'alpha'


class TestRepr:
    """Tests for Structure.__repr__ string representation."""

    def test_repr_contains_class_name(self, mock_pmg_structure):
        """Repr includes the class name."""
        s = atl_struct.Structure(structure=mock_pmg_structure)
        assert 'Structure' in repr(s)

    def test_repr_contains_unique_id(self, mock_pmg_structure):
        """Repr includes the structure's unique_id."""
        s = atl_struct.Structure(structure=mock_pmg_structure, unique_id='my-test-id')
        assert 'my-test-id' in repr(s)

    def test_repr_shows_no_calc(self, mock_pmg_structure):
        """Repr shows 'No calc performed' when no DFT data is present."""
        s = atl_struct.Structure(structure=mock_pmg_structure)
        assert 'No calc performed' in repr(s)

    def test_repr_shows_calc_info(self, mock_pmg_structure):
        """Repr shows energy and calc type when calc_performed=True."""
        s = atl_struct.Structure(
            structure=mock_pmg_structure,
            calc_performed=True,
            calc_energy=-10.5,
            calc_type='SP',
        )
        r = repr(s)
        assert '-10.5' in r
        assert 'SP' in r
