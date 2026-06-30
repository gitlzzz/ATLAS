"""Tests for the Vendi-score diversity metric (pure numpy path)."""

import numpy as np

from atlas.core.database.diversity_metrics import get_vendi_score


class TestVendiScore:
    """get_vendi_score on controlled feature matrices."""

    def test_identical_points_score_is_one(self):
        # All-identical embeddings -> effective diversity of 1.
        X = np.ones((5, 3))
        score = get_vendi_score(X, sigma=1.0)
        assert np.isclose(score, 1.0, atol=1e-6)

    def test_well_separated_points_approach_n(self):
        # Far-apart points with a small bandwidth -> kernel ~ identity,
        # Vendi score approaches the number of points.
        X = np.array([[0.0], [100.0], [200.0], [300.0]])
        score = get_vendi_score(X, sigma=0.1)
        assert np.isclose(score, 4.0, atol=1e-3)

    def test_score_between_one_and_n(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 4))
        score = get_vendi_score(X, sigma=1.0)
        assert 1.0 <= score <= 10.0 + 1e-6
