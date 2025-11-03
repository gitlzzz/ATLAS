"""Utilities to compute the concave hull for a point cloud of atomic descriptors."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, KDTree
from scipy.stats import gaussian_kde
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

from MatDBForge.core import code_utils as mdb_cut

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
    frac_points_allowed_out: float = 0.002,
    n_attempts: int = 20,
    decrease_factor_multiplier: float = 0.95,
    use_alpha: float = None,
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
    frac_points_allowed_out: float, optional
        The maximum fraction of points allowed to be outside the concave hull.
        If the fraction of points outside the hull exceeds this value,
        alpha will be decreased iteratively until the condition is met or
        alpha reaches zero. Defaults to 0.002 (0.2%).
    n_attempts : int, optional
        Number of attempts to compute the concave hull by adjusting alpha.

    Returns
    -------
    np.ndarray
        The coordinates of the vertices of the concave hull (N, 2).
        Returns an empty array np.empty((0,2)) if a hull cannot be formed.
    float
        The alpha value used to compute the concave hull.
    """
    total_num_points = latent_space.shape[0]

    if use_alpha is not None:
        mdb_cut.custom_print(f'Using user-defined alpha: {use_alpha}', 'info')
        # Get alpha shape using the user-defined alpha
        shape = alpha_shape(latent_space, alpha=use_alpha, only_outer=True)

        exterior_xy = shape.exterior.coords.xy
        alpha_shape_arr = np.stack((exterior_xy[0], exterior_xy[1]), axis=1)

        # Check status of the points
        _, point_outside, _ = check_atom_in_domain(
            concave_hull=alpha_shape_arr, descriptors=latent_space
        )
        points_to_check = point_outside.shape[0]

        # Compute fraction of points outside the hull
        frac_outside = points_to_check / total_num_points

        mdb_cut.custom_print(
            f'Current fraction of points outside hull is {frac_outside:.4f}.',
            'info',
        )

        return alpha_shape_arr, use_alpha

    else:
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
                mdb_cut.custom_print(
                    'All points are likely duplicates or extremely close. '
                    'Using max alpha from target range.',
                    'warn',
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
                    mdb_cut.custom_print(
                        'Mean nearest neighbor distance is very small '
                        f'({mean_nn_dist:.2e}). Alpha candidate set to infinity'
                        ' before clipping.',
                        'debug',
                    )
                else:
                    alpha_candidate = nn_dist_scale_factor / mean_nn_dist
                    mdb_cut.custom_print(
                        f'Mean NN dist: {mean_nn_dist:.4f}, '
                        f'NN Distance Scale Factor: {nn_dist_scale_factor:.1f}, '
                        f'Alpha candidate (factor/mean_nn_dist): {alpha_candidate:.4f}',
                        'debug',
                    )

                # Limiting alpha to the target range
                alpha = np.clip(
                    alpha_candidate, target_alpha_range[0], target_alpha_range[1]
                )
                mdb_cut.custom_print(
                    f'Calculated alpha: {alpha:.4f}'
                    f' (clipped to range {target_alpha_range})',
                    'info',
                )

        except Exception as e:
            mdb_cut.custom_print(
                f'Error during KDTree query or mean_nn_dist calculation: {e}. '
                f'Using default alpha: {default_alpha_if_issues}',
                'warn',
            )
            alpha = default_alpha_if_issues

        frac_outside = 1.0
        change_factor = alpha * 0.1
        min_change_factor = change_factor * 0.01
        num_attempts = 0
        alpha_lower_bound = 0.0

        while num_attempts <= n_attempts:
            print()
            mdb_cut.custom_print(
                f'Computing alpha shape with alpha: {alpha:.4f}', 'info'
            )

            # Get alpha shape using the determined alpha
            shape = alpha_shape(latent_space, alpha=alpha, only_outer=True)

            if shape.is_empty:
                mdb_cut.custom_print(
                    f'Alpha shape with alpha={alpha:.4f} resulted in an empty geometry.'
                    ' This can happen if alpha is too large (too restrictive)'
                    ' for the point set. Attempting to return convex hull'
                    ' as a fallback.',
                    'warn',
                )
                # Fallback to convex hull if alpha shape is empty
                if total_num_points >= 3:
                    try:
                        hull = ConvexHull(latent_space)
                        return latent_space[hull.vertices]
                    except Exception as e_cvx:  # pragma: no cover
                        mdb_cut.custom_print(
                            f'Convex hull fallback also failed: {e_cvx}', 'error'
                        )
                        return np.empty((0, 2))
                else:
                    # Should have been caught earlier, but as a safeguard
                    return (
                        np.array(latent_space)
                        if total_num_points > 0
                        else np.empty((0, 2))
                    )

            if hasattr(shape.exterior, 'coords'):
                exterior_xy = shape.exterior.coords.xy
                alpha_shape_arr = np.stack((exterior_xy[0], exterior_xy[1]), axis=1)
            else:
                mdb_cut.custom_print(
                    f'Alpha shape (alpha={alpha:.4f}) did not return a simple polygon '
                    'with an exterior. '
                    f'Shape type: {type(shape)}. This is unexpected '
                    'with only_outer=True.  Attempting to return convex'
                    ' hull as a fallback.',
                    'error',
                )
                if total_num_points >= 3:
                    try:
                        hull = ConvexHull(latent_space)
                        return latent_space[hull.vertices]
                    except Exception as e_cvx2:  # pragma: no cover
                        mdb_cut.custom_print(
                            f'Convex hull fallback also failed: {e_cvx2}', 'error'
                        )
                        return np.empty((0, 2))
                else:
                    return (
                        np.array(latent_space)
                        if total_num_points > 0
                        else np.empty((0, 2))
                    )

            # Check status of the points
            _, point_outside, _ = check_atom_in_domain(
                concave_hull=alpha_shape_arr, descriptors=latent_space
            )
            points_to_check = point_outside.shape[0]

            # Compute fraction of points outside the hull
            prev_frac_outside = frac_outside
            frac_outside = points_to_check / total_num_points
            prev_distance_to_target = abs(prev_frac_outside - frac_points_allowed_out)
            distance_to_target = abs(frac_outside - frac_points_allowed_out)

            # Decrease alpha to make hull less strict
            last_alpha = alpha

            # Change the sign of the change factor if we overshot
            if distance_to_target <= prev_distance_to_target:
                change_factor = abs(change_factor)
            else:
                change_factor = -1 * abs(change_factor)

            # alpha -= change_factor
            alpha -= change_factor

            # Decrease the factor for more gradual changes next iteration
            change_factor = np.max(
                (change_factor * decrease_factor_multiplier, min_change_factor)
            )

            if alpha <= alpha_lower_bound:
                mdb_cut.custom_print(
                    f'Alpha has reached the minimum threshold of {alpha_lower_bound}.'
                    'Stopping adjustments.',
                    'warn',
                )
                break

            if frac_outside >= frac_points_allowed_out:
                mdb_cut.custom_print(
                    (
                        f'Current fraction of points outside hull is'
                        f' {frac_outside:.4f}, above the allowed'
                        f' {frac_points_allowed_out:.4f} threshold.'
                    ),
                    'warn',
                )
                mdb_cut.custom_print(f'Decreasing alpha to: {alpha:.4f}', 'info')
                last_alpha = alpha
                num_attempts += 1
            else:
                mdb_cut.custom_print(
                    (
                        f'Current fraction of points outside hull'
                        f' is {frac_outside:.4f}, '
                        f' below the allowed {frac_points_allowed_out:.4f} threshold.'
                    ),
                    'info',
                )
                break

        # After optimizing alpha, print final status.
        print()
        mdb_cut.custom_print(
            (
                f'Current fraction of points outside hull is {frac_outside:.4f}, '
                f'after {num_attempts} attempts.'
            ),
            'info',
        )
        mdb_cut.custom_print(f'Final alpha: {last_alpha:.4f}', 'done')
        print()

    return alpha_shape_arr, last_alpha


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
    alpha: float = None,
    plot_density: bool = False,
):
    """
    Generate a plot for the concave hull area in 2D space, including in and out
    of domain points if provided.
    """
    if not filename:
        filename = 'concave_hull.png'

    fig, ax = plt.subplots()

    # Concave hull plots may come from md simulations or other sources.
    # This block stacks descriptors for density coloring in all types of concave hull
    # plots
    if latent_space is None and (point_inside is not None or point_outside is not None):
        if point_outside.size == 0:
            point_outside = point_outside.reshape((0, 2))
        if point_inside.size == 0:
            point_inside = point_inside.reshape((0, 2))
        latent_space_dens = np.vstack((point_inside, point_outside))
    else:
        latent_space_dens = latent_space

    x = latent_space_dens[:, 0]
    y = latent_space_dens[:, 1]

    if plot_density:
        # Get point density for coloring
        xy = np.vstack((x, y))
        z = gaussian_kde(xy)(xy)
    else:
        z = '#b16286'

    # Plotting the concave hull in 2D space using lines
    if latent_space is not None and latent_space.size != 0:
        scatter = ax.scatter(
            x,
            y,
            marker='o',
            s=25,
            alpha=0.5,
            label='Descriptor in database',
            linewidths=0,
            c=z,
        )
        fig.colorbar(scatter, ax=ax, label='Density')

    if point_inside is not None and point_inside.size != 0:
        ax.scatter(
            x,
            y,
            marker='s',
            label='Structure in domain',
            color='#8ec07c',
            s=3.5,
            linewidths=1.5,
            edgecolors='#282828',
        )
    if point_outside is not None and point_outside.size != 0:
        ax.scatter(
            x,
            y,
            marker='s',
            label='Structure out of domain',
            color='#fb4934',
            s=3.5,
            linewidths=1.5,
            edgecolors='#282828',
        )

    ax.plot(
        concave_hull[:, 0],
        concave_hull[:, 1],
        '-',
        color='#cc241d',
        lw=1,
        label='Concave hull',
    )

    polygon = Polygon(concave_hull)
    hull_area = polygon.area

    if alpha is None:
        alpha = 'unknown'
    if isinstance(alpha, (float, int)):
        alpha = f'{alpha:.2f}'

    # Write area and alpha in a text box
    plt.text(
        0.05,
        0.95,
        f'Alpha: {alpha}\nHull area: {hull_area:.2e}',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    plt.title('Concave Hull')
    plt.xlabel('Embedded dimension 1')
    plt.ylabel('Embedded dimension 2')
    plt.legend()

    # Save as PNG
    plt.savefig(filename, dpi=300)

    # Save as SVG
    filename_svg = filename.with_suffix('.svg')
    plt.savefig(filename_svg, dpi=300)

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

    # Compute the concave hull.
    concave_hull, alpha = get_concave_hull_python(latent_space)

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
