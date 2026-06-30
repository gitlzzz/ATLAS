"""Tests for the concave-hull / alpha-shape geometry utilities."""

import numpy as np
from shapely.geometry import LineString, Polygon

from atlas.active_learning.extrapolation.concave_hull import (
    alpha_shape,
    check_traj_in_domain,
)


class TestAlphaShape:
    """Tests for alpha_shape degenerate and small-point-set cases."""

    def test_two_points_returns_linestring(self):
        result = alpha_shape(np.array([[0.0, 0.0], [1.0, 0.0]]), alpha=1.0)
        assert isinstance(result, LineString)

    def test_three_points_returns_triangle_polygon(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = alpha_shape(pts, alpha=1.0)
        assert isinstance(result, Polygon)
        # Right triangle with legs of length 1 -> area 0.5
        assert np.isclose(result.area, 0.5, atol=1e-6)


class TestCheckTrajInDomain:
    """Tests for the point-in-hull membership check."""

    def test_points_inside_and_outside_unit_square(self):
        # Single-polygon hull -> exercises the len()==1 list path.
        square = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
        # latent_space is a list of frames; each frame is a list of 2D points.
        descriptor_dict = {
            'struct_a': {'latent_space': [[[0.5, 0.5], [5.0, 5.0]]]},
        }
        inside, outside, flags, _ = check_traj_in_domain(
            concave_hull=[square], descriptor_dict=descriptor_dict
        )
        # One point inside, one outside.
        assert len(inside) == 1
        assert len(outside) == 1
        # Per-frame boolean array: [inside, outside] -> [True, False]
        assert flags[0].tolist() == [True, False]
