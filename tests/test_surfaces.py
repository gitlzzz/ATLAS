"""Tests for surface-generation helper functions."""

import numpy as np

from atlas.core.surfaces import get_miller_index_str


class TestGetMillerIndexStr:
    """Miller-index string formatting used for structure naming."""

    def test_tuple(self):
        assert get_miller_index_str((1, 0, 0)) == '100'
        assert get_miller_index_str((1, 1, 1)) == '111'

    def test_list(self):
        assert get_miller_index_str([2, 1, 0]) == '210'

    def test_ndarray(self):
        assert get_miller_index_str(np.array([1, 0, 0])) == '100'

    def test_negative_indices_keep_sign(self):
        assert get_miller_index_str((-1, 1, 0)) == '-110'

    def test_string_passthrough_strips_brackets(self):
        assert get_miller_index_str('(1, 1, 1)') == '111'

    def test_unsupported_source_returns_none(self):
        assert get_miller_index_str(42) is None
