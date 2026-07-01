"""Tests for the pluggable MLIP backend abstraction."""

import pytest

from atlas.active_learning.backends import (
    MaceBackend,
    MLIPBackend,
    available_backends,
    get_backend,
    register_backend,
)


class TestRegistry:
    def test_mace_registered_by_default(self):
        assert 'mace' in available_backends()

    def test_get_backend_returns_mace(self):
        backend = get_backend('mace')
        assert isinstance(backend, MaceBackend)
        assert backend.name == 'mace'
        assert backend.train_calcjob_entry_point() == 'mace-train'

    def test_get_backend_is_case_insensitive_and_defaults(self):
        assert isinstance(get_backend('MACE'), MaceBackend)
        assert isinstance(get_backend(None), MaceBackend)
        assert isinstance(get_backend(), MaceBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match='Unknown MLIP model_type'):
            get_backend('does-not-exist')

    def test_register_custom_backend(self):
        class DummyBackend(MLIPBackend):
            name = 'dummy'

            def build_calculator(self, model=None, **kwargs):
                return 'dummy-calc'

            def train_calcjob_entry_point(self):
                return 'dummy-train'

        register_backend('dummy', DummyBackend)
        assert 'dummy' in available_backends()
        assert get_backend('dummy').build_calculator() == 'dummy-calc'


class TestMaceFoundationParsing:
    def test_mp_variant(self):
        assert get_backend('mace').parse_foundation_id('mace:mp-small') == ('mp', 'small')

    def test_off_variant(self):
        assert get_backend('mace').parse_foundation_id('mace:off-medium') == (
            'off',
            'medium',
        )

    def test_plain_path_is_not_foundation(self):
        assert get_backend('mace').parse_foundation_id('/models/curr_model.model') is None

    def test_non_string_is_not_foundation(self):
        assert get_backend('mace').parse_foundation_id(None) is None


class TestFrameworkBackends:
    """The framework backends register and fail gracefully when absent."""

    ALL = ['mace', 'nequip', 'allegro', 'deepmd', 'equiformer', 'orb', 'lasp']

    def test_all_registered(self):
        for name in self.ALL:
            assert name in available_backends()

    def test_each_has_train_entry_point(self):
        for name in self.ALL:
            ep = get_backend(name).train_calcjob_entry_point()
            assert isinstance(ep, str) and ep

    def test_absent_frameworks_raise_informative_error(self):
        # For any framework whose package is NOT installed, building a calculator
        # must raise a clear, actionable error naming the backend (not an opaque
        # error deep in the framework). Backends whose package IS installed are
        # skipped here (they'd try to actually load a model).
        import importlib.util

        backend_pkg = {
            'nequip': 'nequip',
            'allegro': 'nequip',
            'deepmd': 'deepmd',
            'equiformer': 'fairchem',
            'orb': 'orb_models',
        }
        checked = 0
        for name, pkg in backend_pkg.items():
            if importlib.util.find_spec(pkg) is not None:
                continue  # framework present in this env; covered by live tests
            checked += 1
            with pytest.raises(ImportError, match=name):
                get_backend(name).build_calculator('some_model')
        assert checked >= 1  # at least one framework is absent in a base env

    def test_lasp_is_scaffold(self):
        with pytest.raises(NotImplementedError):
            get_backend('lasp').build_calculator('model.pot')

    def test_orb_foundation_parsing(self):
        assert get_backend('orb').parse_foundation_id('orb-v2') == ('pretrained', 'orb-v2')
        assert get_backend('orb').parse_foundation_id('/ckpt.ckpt') is None


class TestOrbLive:
    """Live Orb backend check (skipped unless orb-models is installed).

    Verified against orb-models v0.5.5: builds a real ORBCalculator from a
    pretrained potential and runs a single-point energy/force evaluation.
    """

    def test_orb_single_point(self):
        pytest.importorskip('orb_models')
        import numpy as np
        from ase.build import bulk

        calc = get_backend('orb').build_calculator(
            'orb-d3-xs-v2', device='cpu', compile=False
        )
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        assert np.isfinite(energy)
        assert forces.shape == (len(atoms), 3)


class TestInterface:
    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            MLIPBackend()

    def test_default_optionals(self):
        backend = get_backend('mace')
        # LAMMPS unsupported returns None by default only when overridden;
        # MACE overrides get_lammps_model, so just check descriptors fallback.
        with pytest.raises(NotImplementedError):
            MLIPBackend.get_descriptors(backend)
