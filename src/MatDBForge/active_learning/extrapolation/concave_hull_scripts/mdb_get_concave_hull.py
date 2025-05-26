#!/usr/bin/env python3
"""Utilities to compute the concave hull for a point cloud of atomic descriptors."""

import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, KDTree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

try:
    import numba

    njit = numba.njit(cache=True, fastmath=True)
except ImportError:  # fall back gracefully

    def njit(x=None, **kw):  # decorator that returns the function unchanged
        return x if x is not None else (lambda f: f)


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
    # for degenerate cases
    if pts.shape[0] < 4:
        return Polygon(pts).convex_hull

    tri = Delaunay(pts)
    # simplices are the indices of the points that form the Delaunay triangles
    # (M, 3) int array
    simplices = tri.simplices

    # circum-radii of all Delaunay triangles
    pa, pb, pc = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
    a = np.linalg.norm(pb - pc, axis=1)
    b = np.linalg.norm(pc - pa, axis=1)
    c = np.linalg.norm(pa - pb, axis=1)
    s = (a + b + c) * 0.5
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))

    # Ignore division by zero and invalid operations
    with np.errstate(divide='ignore', invalid='ignore'):
        # R = abc / 4A
        circum_r = a * b * c / (4.0 * area)

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


def get_concave_hull_python(
    latent_space: np.ndarray,
    target_alpha_range: tuple[float, float] = (3.0, 8.0),
    default_alpha_if_issues: float = 5,
    nn_dist_scale_factor: float = 1.5,
) -> np.ndarray:
    """
    Compute the concave hull of a set of points using the alpha-shape algorithm.

    Parameters
    ----------
    latent_space : np.ndarray
        The input points (N, 2) for which to compute the concave hull.
    target_alpha_range : tuple[float, float], optional
        The desired (min_alpha, max_alpha) range for the alpha parameter.
        Alpha will be clipped to this range. Defaults to (3.0, 8.0).
    default_alpha_if_issues : float, optional
        Default alpha value to use if nearest neighbor distance calculation
        is not possible (e.g., too few points) or other issues arise.
        Defaults to 5 (midpoint of common 3-8 range).
    nn_dist_scale_factor : float, optional
        Scaling factor for the alpha candidate calculation:
        `alpha_candidate = nn_dist_scale_factor / mean_nn_dist`.
        Defaults to 1.5.

    Returns
    -------
    np.ndarray
        The coordinates of the vertices of the concave hull (N, 2).
        Returns an empty array np.empty((0,2)) if a hull cannot be formed.
    """
    num_points = latent_space.shape[0]
    alpha = default_alpha_if_issues

    try:
        tree = KDTree(latent_space)

        # k=2 for 1st nearest neighbor (k=1 is the point itself)
        distances, _ = tree.query(latent_space, k=2, workers=-1)
        nn_distances = distances[:, 1]

        # Filter out zero distances if any point is duplicated or coincident
        # Use a small epsilon for floating point comparisons
        valid_nn_distances = nn_distances[nn_distances > 1e-9]

        # If all points are coincident or extremely close
        if valid_nn_distances.size == 0:
            print(
                'All points are likely duplicates or extremely close. '
                'Using max alpha from target range.',
            )
            # Use max alpha, tends to shrink
            alpha = target_alpha_range[1]

        else:
            mean_nn_dist = np.mean(valid_nn_distances)

            # Effectively zero
            if mean_nn_dist < 1e-7:
                # This case implies extremely dense points.
                # A very large alpha candidate will be clipped to max of range.
                alpha_candidate = np.inf
                print(
                    'Mean nearest neighbor distance is very small '
                    f'({mean_nn_dist:.2e}). Alpha candidate set to infinity'
                    ' before clipping.',
                    'debug',
                )
            else:
                alpha_candidate = nn_dist_scale_factor / mean_nn_dist
                print(
                    f'Mean NN dist: {mean_nn_dist:.4f}, '
                    f'NN Distance Scale Factor: {nn_dist_scale_factor:.1f}, '
                    f'Alpha candidate (factor/mean_nn_dist): {alpha_candidate:.4f}',
                )

            # Limiting alpha to the target range
            alpha = np.clip(
                alpha_candidate, target_alpha_range[0], target_alpha_range[1]
            )
            print(
                f'Calculated alpha: {alpha:.2f}'
                f' (clipped to range {target_alpha_range})',
                'info',
            )

    except Exception as e:
        print(
            f'Error during KDTree query or mean_nn_dist calculation: {e}. '
            f'Using default alpha: {default_alpha_if_issues}',
        )
        alpha = default_alpha_if_issues

    # Get alpha shape using the determined alpha
    shape = alpha_shape(latent_space, alpha=alpha, only_outer=True)

    if shape.is_empty:
        print(
            f'Alpha shape with alpha={alpha:.4f} resulted in an empty geometry. '
            'This can happen if alpha is too large (too restrictive) for the point set.'
            ' Attempting to return convex hull as a fallback.',
        )
        # Fallback to convex hull if alpha shape is empty
        if num_points >= 3:
            try:
                hull = ConvexHull(latent_space)
                return latent_space[hull.vertices]
            except Exception as e_cvx:
                print(f'Convex hull fallback also failed: {e_cvx}', 'error')
                return np.empty((0, 2))
        else:
            # Should have been caught earlier, but as a safeguard
            return np.array(latent_space) if num_points > 0 else np.empty((0, 2))

    if hasattr(shape.exterior, 'coords'):
        exterior_xy = shape.exterior.coords.xy
        alpha_shape_arr = np.stack((exterior_xy[0], exterior_xy[1]), axis=1)
    else:
        print(
            f'Alpha shape (alpha={alpha:.4f}) did not return a simple polygon '
            'with an exterior. '
            f'Shape type: {type(shape)}. This is unexpected with only_outer=True. '
            'Attempting to return convex hull as a fallback.',
        )
        if num_points >= 3:
            try:
                hull = ConvexHull(latent_space)
                return latent_space[hull.vertices]
            except Exception as e_cvx2:
                print(f'Convex hull fallback also failed: {e_cvx2}', 'error')
                return np.empty((0, 2))
        else:
            return np.array(latent_space) if num_points > 0 else np.empty((0, 2))

    return alpha_shape_arr


def check_atom_in_domain(
    concave_hull: np.ndarray, descriptors: np.ndarray
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
    latent_space: np.ndarray,
    filename: str = 'concave_hull.png',
):
    # Plotting the concave hull in 2D space using lines
    plt.plot(concave_hull[:, 0], concave_hull[:, 1], 'r-')
    plt.plot(
        latent_space[:, 0],
        latent_space[:, 1],
        'o',
        markersize=2,
        alpha=0.5,
        label='Descriptor in database',
        markeredgewidth=0,
        color='#b16286',
    )
    plt.title('Concave Hull')
    plt.xlabel('x')
    plt.legend()
    plt.tight_layout()

    # Get file path
    filename = pl.Path(filename)

    # Save original format
    plt.savefig(filename, dpi=300)

    # Save as SVG
    filename_svg = filename.with_suffix('.svg')
    plt.savefig(filename_svg, dpi=300)

    plt.clf()


if __name__ == '__main__':
    plot_hull = True
    latent_space_file = 'latent_space.npy'

    # Gather the latent space from the autoencoder.
    print('Reading latent space...')
    latent_space = np.load(latent_space_file)
    print('Latent space read.')

    # Compute the concave hull using Julia.
    print('Computing concave hull...')
    concave_hull = get_concave_hull_python(latent_space)
    np.save('concave_hull', concave_hull)
    print("Concave hull computed, saved to 'concave_hull.npy'.")

    print('Plotting concave hull...')
    if plot_hull:
        plot_concave_hull(
            concave_hull=concave_hull,
            latent_space=latent_space,
            filename='concave_hull.png',
        )
    print('Concave hull plotted.')
    print('Calculation done.')
