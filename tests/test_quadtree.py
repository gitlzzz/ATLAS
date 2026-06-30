"""Tests for the QuadTree spatial-partitioning utilities."""

from shapely.geometry import Point

from atlas.active_learning.extrapolation.quadtree import QuadTree, Rectangle


class TestRectangle:
    """Tests for the axis-aligned Rectangle helper."""

    def test_bounds(self):
        r = Rectangle(x=0.0, y=0.0, w=1.0, h=2.0)
        assert r.xmin == -1.0
        assert r.xmax == 1.0
        assert r.ymin == -2.0
        assert r.ymax == 2.0

    def test_area(self):
        # Area uses full width/height: (2w) * (2h)
        assert Rectangle(x=0.0, y=0.0, w=1.0, h=1.0).get_area() == 4.0
        assert Rectangle(x=5.0, y=5.0, w=2.0, h=3.0).get_area() == 24.0

    def test_contains_interior_and_exterior(self):
        r = Rectangle(x=0.0, y=0.0, w=1.0, h=1.0)
        assert r.contains(Point(0.0, 0.0)) is True
        assert r.contains(Point(2.0, 2.0)) is False

    def test_contains_is_half_open(self):
        # xmin/ymin inclusive, xmax/ymax exclusive
        r = Rectangle(x=0.0, y=0.0, w=1.0, h=1.0)
        assert r.contains(Point(-1.0, -1.0)) is True
        assert r.contains(Point(1.0, 1.0)) is False


class TestQuadTree:
    """Tests for QuadTree insertion and subdivision."""

    def _tree(self, capacity=4):
        return QuadTree(Rectangle(x=0.0, y=0.0, w=1.0, h=1.0), capacity=capacity)

    def test_insert_within_capacity_does_not_subdivide(self):
        qt = self._tree(capacity=4)
        for i in range(4):
            assert qt.insert(Point(0.1 * i, 0.1 * i)) is True
        assert qt.divided is False
        assert len(qt.points) == 4

    def test_insert_beyond_capacity_subdivides(self):
        qt = self._tree(capacity=4)
        for i in range(5):
            assert qt.insert(Point(0.01 * i, 0.01 * i)) is True
        assert qt.divided is True
        # Parent clears its own points after redistributing to children.
        assert qt.points == []
        assert qt.northeast is not None and qt.southwest is not None

    def test_insert_outside_boundary_rejected(self):
        qt = self._tree()
        assert qt.insert(Point(10.0, 10.0)) is False
