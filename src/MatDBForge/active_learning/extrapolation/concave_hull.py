"""Utilities to compute the concave hull for a point cloud of atomic descriptors."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

try:
    import numba

    njit = numba.njit(cache=True, fastmath=True)
except ImportError:  # fall back gracefully

    def njit(x=None, **kw):  # decorator that returns the function unchanged
        return x if x is not None else (lambda f: f)


def get_concave_hull_julia(latent_space: np.ndarray) -> np.ndarray:
    from juliacall import Main as jl
    from juliacall import convert as jl_convert

    # Load the required Julia modules.
    jl.seval('using GMT')

    # Convert the latent space to a Julia array.
    # As it is necessary for the concave hull function.
    latent_space = jl_convert(jl.Matrix[jl.Float32], latent_space)

    # Compute the concave hull using the GMT.jl package.
    concave_hull = jl.concavehull(latent_space, 0.075)

    # Convert the result back to a NumPy array.
    concave_hull = np.array(concave_hull)

    return concave_hull


@njit
def _edges_from_triangles(tris):
    """
    Return the set of boundary oriented edges from the subset of
    triangles that satisfy the alpha condition.
    """
    edges = set()
    for t0, t1, t2 in tris:
        for i, j in ((t0, t1), (t1, t2), (t2, t0)):
            if (j, i) in edges:  # internal edge → remove
                edges.remove((j, i))
            else:  # external edge → add
                edges.add((i, j))
    return edges


def alpha_shape(points, alpha: float, only_outer: bool = True):
    """
    Compute the 2-D alpha-shape (concave hull) of a set of points.

    Parameters
    ----------
    points : (N, 2) array-like
        Input coordinates.
    alpha : float
        Inverse length scale.  Smaller alpha ⟶ coarser (more concave) hull.
        A good starting point is alpha ≈ 1 / (average edge length).
    only_outer : bool, default True
        If True, return only the outer boundary.  If False, keep holes.

    Returns
    -------
    shapely.geometry.Polygon | MultiPolygon
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.shape[0] < 4:  # degenerate cases
        return Polygon(pts).convex_hull

    tri = Delaunay(pts)
    simplices = tri.simplices  # (M, 3) int array

    # circum-radii of all Delaunay triangles
    pa, pb, pc = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
    a = np.linalg.norm(pb - pc, axis=1)
    b = np.linalg.norm(pc - pa, axis=1)
    c = np.linalg.norm(pa - pb, axis=1)
    s = (a + b + c) * 0.5
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))
    circum_r = a * b * c / (4.0 * area)  # R = abc / 4A

    # keep triangles whose circum-radius satisfies R < 1/alpha
    keep = circum_r < 1.0 / alpha
    kept_simplices = simplices[keep]

    # boundary edges
    boundary_edges = _edges_from_triangles(kept_simplices)

    # build shapely geometry
    segments = [LineString([pts[i], pts[j]]) for (i, j) in boundary_edges]
    mgeom = unary_union(segments)
    polygons = list(polygonize(mgeom))

    if not polygons:
        return MultiPolygon()  # empty shape

    if only_outer:
        # Return the polygon with the largest area (outer shell)
        return max(polygons, key=lambda p: p.area)

    return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]


def get_concave_hull_python(latent_space: np.ndarray) -> np.ndarray:
    """Compute the concave hull of a set of points using the alpha shape algorithm."""
    from scipy.spatial import KDTree

    tree = KDTree(latent_space)

    # k=2 → 1-NN
    mean_nn_dist = np.mean(tree.query(latent_space, k=2)[0][:, 1])
    print('mean_nn_dist: ', mean_nn_dist)
    # alpha = 1.5 / mean_nn_dist
    alpha = 1.5

    # Get alpha shape
    exterior_xy = alpha_shape(
        latent_space, alpha=alpha, only_outer=True
    ).exterior.coords.xy

    alpha_shape_arr = np.stack((exterior_xy[0], exterior_xy[1]), axis=1)
    return alpha_shape_arr


def check_atom_in_domain(
    concave_hull: np.ndarray,
    descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_inside = []
    point_outside = []
    all_points_in_out = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    polygon = Polygon(concave_hull)
    for point in descriptors:
        p = Point(point)
        if polygon.contains(p):
            point_inside.append(point)
            all_points_in_out.append(True)
        else:
            point_outside.append(point)
            all_points_in_out.append(False)

    point_inside = np.array(point_inside)
    point_outside = np.array(point_outside)
    all_points_in_out = np.array(all_points_in_out)
    return point_inside, point_outside, all_points_in_out


def plot_concave_hull(
    concave_hull: np.ndarray,
    latent_space: np.ndarray = None,
    point_inside: np.ndarray = None,
    point_outside: np.ndarray = None,
    filename: str = None,
):
    if not filename:
        filename = 'concave_hull.png'

    # Plotting the concave hull in 2D space using lines
    plt.plot(concave_hull[:, 0], concave_hull[:, 1], 'r-')
    if latent_space is not None and latent_space.size != 0:
        plt.plot(
            latent_space[:, 0],
            latent_space[:, 1],
            'o',
            markersize=3.5,
            alpha=0.5,
            label='Descriptor in database',
            markeredgewidth=0,
            color='#b16286',
        )

    if point_inside is not None and point_inside.size != 0:
        plt.plot(
            point_inside[:, 0],
            point_inside[:, 1],
            's',
            label='Structure in domain',
            color='#8ec07c',
            markersize=5,
            markeredgewidth=1.5,
            markeredgecolor='#282828',
        )
    if point_outside is not None and point_outside.size != 0:
        plt.plot(
            point_outside[:, 0],
            point_outside[:, 1],
            's',
            label='Structure out of domain',
            color='#fb4934',
            markersize=5,
            markeredgewidth=1.5,
            markeredgecolor='#282828',
        )

    plt.title('Concave Hull')
    plt.xlabel('Embedded dimension 1')
    plt.ylabel('Embedded dimension 2')
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show(block=False)
    plt.clf()


if __name__ == '__main__':
    plot_hull = True
    latent_space_file = 'latent_space.npy'
    descriptors_file = 'descriptors.npz'

    # Set the random seed for reproducibility.
    rng = np.random.default_rng(seed=420)

    # Gather the latent space from the autoencoder.
    latent_space = np.load(latent_space_file)

    # Get descriptors array
    descriptors = np.load(descriptors_file)

    # Compute the concave hull using Julia.
    concave_hull = get_concave_hull_julia(latent_space)

    # Check if the random points are inside the concave hull.
    point_inside, point_outside, all_points = check_atom_in_domain(
        concave_hull, descriptors
    )

    np.save('point_inside.npy', point_inside)
    np.save('point_outside.npy', point_outside)
    np.save('all_points.npy', all_points)

    if plot_hull:
        plot_concave_hull(
            concave_hull=concave_hull,
            point_inside=point_inside,
            point_outside=point_outside,
            latent_space=latent_space,
            filename='/tmp/concave_hull.png',
        )
