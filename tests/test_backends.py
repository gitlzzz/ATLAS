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
