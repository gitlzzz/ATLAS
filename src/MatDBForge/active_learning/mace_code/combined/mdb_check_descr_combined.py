#!/usr/bin/env python3
"""Script to handle descriptor gathering for an AL Loop iteration."""

import pathlib as pl
import pickle
import tomllib

import matplotlib.pyplot as plt
import numpy as np
from aiida.common.extendeddicts import AttributeDict
from ase.io import read as ase_read
from scipy.spatial import ConvexHull, Delaunay, KDTree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

from MatDBForge.active_learning.active_learning_utils import generate_descriptors
from MatDBForge.active_learning.extrapolation import train_autoencoder as mdb_train_ae
from MatDBForge.core import code_utils as mdb_cut

try:
    import numba

    njit = numba.njit(cache=True, fastmath=True)
except ImportError:  # fall back gracefully

    def njit(x=None, **kw):  # decorator that returns the function unchanged
        return x if x is not None else (lambda f: f)


def is_advanced_extrapolation_set(input_dict: dict):
    """Check if the advanced extrapolation check is enabled."""
    extr_type = input_dict.get('extrapolation', {}).get('check_extrapolation_type')
    if extr_type == 'advanced':
        return True
    return


def has_latent_space(settings_dict: dict):
    """Check if the latent space was computed for the current iteration."""
    return settings_dict.get('latent_space', False)


def can_do_advanced_extrapolation():
    """Check if the advanced extrapolation can be done."""
    return has_latent_space() and is_advanced_extrapolation_set()


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


@njit
def _edges_from_triangles(tris):
    """
    Return the set of boundary oriented edges from the subset of
    triangles that satisfy the alpha condition.
    """
    edges = set()
    for t0, t1, t2 in tris:
        for i, j in ((t0, t1), (t1, t2), (t2, t0)):
            # remove if internal edge
            if (j, i) in edges:
                edges.remove((j, i))
            else:
                # add if external edge
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
                f'Calculated alpha: {alpha:.2f}'
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

    # Get alpha shape using the determined alpha
    shape = alpha_shape(latent_space, alpha=alpha, only_outer=True)

    if shape.is_empty:
        mdb_cut.custom_print(
            f'Alpha shape with alpha={alpha:.4f} resulted in an empty geometry. '
            'This can happen if alpha is too large (too restrictive) for the point set.'
            ' Attempting to return convex hull as a fallback.',
            'warn',
        )
        # Fallback to convex hull if alpha shape is empty
        if num_points >= 3:
            try:
                hull = ConvexHull(latent_space)
                return latent_space[hull.vertices]
            except Exception as e_cvx:
                mdb_cut.custom_print(
                    f'Convex hull fallback also failed: {e_cvx}', 'error'
                )
                return np.empty((0, 2))
        else:
            # Should have been caught earlier, but as a safeguard
            return np.array(latent_space) if num_points > 0 else np.empty((0, 2))

    if hasattr(shape.exterior, 'coords'):
        exterior_xy = shape.exterior.coords.xy
        alpha_shape_arr = np.stack((exterior_xy[0], exterior_xy[1]), axis=1)
    else:
        mdb_cut.custom_print(
            f'Alpha shape (alpha={alpha:.4f}) did not return a simple polygon '
            'with an exterior. '
            f'Shape type: {type(shape)}. This is unexpected with only_outer=True. '
            'Attempting to return convex hull as a fallback.',
            'error',
        )
        if num_points >= 3:
            try:
                hull = ConvexHull(latent_space)
                return latent_space[hull.vertices]
            except Exception as e_cvx2:
                mdb_cut.custom_print(
                    f'Convex hull fallback also failed: {e_cvx2}', 'error'
                )
                return np.empty((0, 2))
        else:
            return np.array(latent_space) if num_points > 0 else np.empty((0, 2))

    return alpha_shape_arr


if __name__ == '__main__':
    # The /mdb_data directory should only exist in the containerized version
    # of the code. This conditional statement will get the correct path for
    # input and output files.
    if pl.Path('/mdb_data').exists():
        prepend_path = pl.Path('/mdb_data')
    else:
        prepend_path = pl.Path('.')

    # Define results folder
    res_folder = pl.Path(prepend_path) / pl.Path('./results')
    res_folder.mkdir(exist_ok=True)

    # Initialize the logger
    log_folder = prepend_path / pl.Path('./logs')
    log_folder.mkdir(exist_ok=True)
    logger = mdb_cut.init_logger(
        source='check_extrapolation_combined', log_path=log_folder
    )

    mdb_cut.custom_print(
        'Initializing descriptor generation, dimensionality reduction, '
        'and concave hull computation for the structure database.'
    )
    mdb_cut.custom_print(f"Storing results in '{res_folder.resolve()}'.")

    # Read TOML file with settings
    with open(prepend_path / 'settings.toml', 'rb') as f:
        settings = tomllib.load(f)

    descriptor_settings = settings.get('descriptors', {})
    device = descriptor_settings.get('device', 'cpu')
    dtype = descriptor_settings.get('dtype', 'float32')
    extrapolation_settings = settings.get('extrapolation', {})
    auto_settings = descriptor_settings.get('autoencoder', {})
    auto_train_settings = auto_settings.get('train_settings', {})
    auto_path = auto_train_settings.get('model_path', 'autoencoder_model.pth')
    mdb_cut.custom_print(f"Using device '{device}' and dtype '{dtype}'", 'info')

    # Get settings for concave hull if available
    concave_hull_settings = extrapolation_settings.get('concave_hull', {})
    target_alpha_range_min = concave_hull_settings.get('target_alpha_range_min', 3.0)
    target_alpha_range_max = concave_hull_settings.get('target_alpha_range_max', 8.0)
    default_alpha_if_issues = concave_hull_settings.get('default_alpha_if_issues', 5.0)
    nn_dist_scale_factor = concave_hull_settings.get('nn_dist_scale_factor', 1.5)

    # Ensure target alpha range is a tuple
    target_alpha_range = (target_alpha_range_min, target_alpha_range_max)

    mdb_cut.custom_print(
        f'Read concave hull settings. Target alpha range: {target_alpha_range}, '
        f'default alpha if issues: {default_alpha_if_issues}, '
        f'NN distance scale factor: {nn_dist_scale_factor}',
        'info',
    )

    # Load data
    structs_database = ase_read(
        prepend_path / 'training_db.xyz', index=':', format='extxyz'
    )

    # Generate descriptors
    if not pl.Path(res_folder / 'curr_it_db_descriptors.pkl').exists():
        descriptor_type = descriptor_settings.get('descriptor_type', 'mace')
        descriptor_dict, descriptor_arr = generate_descriptors(
            descriptor_type=descriptor_type,
            database=structs_database,
            descriptor_settings=descriptor_settings,
            model_path=prepend_path / 'curr_iter_best.model',
        )
    else:
        mdb_cut.custom_print('Reading descriptors from file...')
        try:
            descriptor_arr = np.load(prepend_path / 'all_descriptors.npz')['arr_0']
        except FileNotFoundError:
            descriptor_arr = np.load(prepend_path / 'all_descriptors.npy')

        with open(res_folder / 'curr_it_db_descriptors.pkl', 'rb') as f:
            descriptor_dict = pickle.load(f)

    # Minimum and maximum values for each of the descriptors
    min_val = np.min(descriptor_arr, axis=0)
    max_val = np.max(descriptor_arr, axis=0)

    # Storing arrays into a numpy file to be later gathered by the workchain
    np.save(file=res_folder / 'curr_it_db_max', arr=max_val)
    np.save(file=res_folder / 'curr_it_db_min', arr=min_val)

    # Saving descriptor array
    np.savez_compressed(prepend_path / 'all_descriptors.npz', descriptor_arr)
    mdb_cut.custom_print('Descriptors generated.')

    latent_space = None
    latent_space_all = None

    # Proceed with advanced extrapolation if enabled
    if is_advanced_extrapolation_set(input_dict=settings):
        # Get latent space
        latent_space_file = res_folder / 'latent_space.npy'

        if not pl.Path(prepend_path / auto_path).exists():
            mdb_cut.custom_print('Training the autoencoder model...')

            # Check if the dataset path exists, if not prepend the prepend_path
            if not pl.Path(auto_train_settings['dataset']).exists():
                auto_train_settings['dataset'] = (
                    prepend_path / auto_train_settings['dataset']
                )
            # Check if the model path exists, if not prepend the prepend_path
            if not pl.Path(auto_train_settings['model_path']).exists():
                auto_path = prepend_path / auto_train_settings['model_path']
                auto_train_settings['model_path'] = auto_path

            # Train the autoencoder model
            mdb_train_ae.run_training(AttributeDict(auto_train_settings))
            mdb_cut.custom_print('Autoencoder model trained.')
        else:
            mdb_cut.custom_print('Loading Autoencoder model...')

        # Load autoencoder model
        import torch

        model = torch.load(auto_path, weights_only=False)

        # Save autoencoder model to `autoencoder_model.pth` if it does not exist
        if not pl.Path(res_folder / 'autoencoder_model.pth').exists():
            torch.save(model, res_folder / 'autoencoder_model.pth')

        # Changing device to CPU
        model.to('cpu')

        # Remember that you must call model.eval() to set dropout and batch
        # normalization layers to evaluation mode before running inference.
        # Failing to do this will yield inconsistent inference results.
        model.eval()

        # Reduce the dimensionality of the input points to 2D
        mdb_cut.custom_print('Computing latent space for all structures...')
        latent_space_all = []
        with torch.no_grad():  # No need to compute gradients for inference
            for idx, struct in enumerate(structs_database):
                mdb_cut.custom_print(
                    f'Structure {idx}/{len(structs_database)}', 'debug'
                )

                # Get descriptors
                if struct.info.get('mdb_id'):
                    curr_struct_id = struct.info.get('mdb_id')
                else:
                    curr_struct_id = struct.info.get('aiida_uuid')

                curr_descriptors = descriptor_dict[curr_struct_id]['descriptors'][0]

                # Get latent space
                latent_space = (
                    model.encoder(torch.Tensor(curr_descriptors)).cpu().numpy()
                )

                # Save latent space
                descriptor_dict[curr_struct_id]['latent_space'].extend(latent_space)
                latent_space_all.append(latent_space)

        # Saving latent space
        latent_space_all = np.vstack(latent_space_all)
        np.save(latent_space_file, latent_space_all)

        # No way to store all of the descriptors in a single array,
        # as the n_atom dimension will change according to the structure
        # This pickle object will contain a list of length n_struct,
        # that will have n_at lists inside, each containing model_size
        # lists of descriptor values.
        with open(res_folder / 'curr_it_db_descriptors.pkl', 'wb') as f:
            pickle.dump(descriptor_dict, f)

    # If latent space was successfully calculated
    # Check for the existence of the latent_space variable
    if latent_space is not None or latent_space_all is not None:
        # Get concave hull
        match descriptor_settings.get('dimensionality_reduction_method'):
            case 'autoencoder':
                mdb_cut.custom_print('Computing concave hull...', 'info')
                concave_hull = get_concave_hull_python(
                    latent_space_all,
                    target_alpha_range=target_alpha_range,
                    default_alpha_if_issues=default_alpha_if_issues,
                    nn_dist_scale_factor=nn_dist_scale_factor,
                )
                concave_hull_path = res_folder / 'concave_hull.npy'
                np.save(file=concave_hull_path, arr=concave_hull)

                mdb_cut.custom_print(
                    f"Concave hull computed, saved to '{concave_hull_path}'.", 'info'
                )

                plot_hull = True

                if plot_hull:
                    plot_img_path = res_folder / 'concave_hull.png'
                    plot_concave_hull(
                        concave_hull=concave_hull,
                        latent_space=latent_space_all,
                        filename=plot_img_path,
                    )

                    mdb_cut.custom_print(
                        f"Concave hull plotted, saved to '{plot_img_path}'.", 'info'
                    )
            case 'pca':
                raise NotImplementedError('PCA not implemented yet')
            case 'none':
                mdb_cut.custom_print(
                    'No dimensionality reduction method specified. '
                    'Skipping concave hull computation.',
                    'warn',
                )
                concave_hull = None
    else:
        mdb_cut.custom_print(
            'Latent space was not computed. Skipping concave hull computation.',
            'warn',
        )
    mdb_cut.custom_print('Calculation done.', 'done')
