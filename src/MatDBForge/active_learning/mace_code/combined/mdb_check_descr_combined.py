#!/usr/bin/env python3
"""Script to handle descriptor gathering for an AL Loop iteration."""

import pathlib as pl
import pickle
import tomllib

import numpy as np
from aiida.common.extendeddicts import AttributeDict
from ase.io import read as ase_read
from shapely.geometry import Point, Polygon

from MatDBForge.active_learning.active_learning_utils import generate_descriptors
from MatDBForge.active_learning.extrapolation import train_autoencoder as mdb_train_ae
from MatDBForge.active_learning.extrapolation.concave_hull import (
    get_concave_hull_python,
    plot_concave_hull,
)
from MatDBForge.core import code_utils as mdb_cut


def is_advanced_extrapolation_set(input_dict: dict):
    """Check if the advanced extrapolation check is enabled."""
    extr_type = input_dict.get('extrapolation', {}).get('check_extrapolation_type')
    if extr_type in ['advanced', 'alpha-shape']:
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
    frac_points_allowed_out = concave_hull_settings.get(
        'frac_points_allowed_out', 0.002
    )

    # Ensure target alpha range is a tuple
    target_alpha_range = (target_alpha_range_min, target_alpha_range_max)

    mdb_cut.custom_print(
        f'Read concave hull settings. Target alpha range: {target_alpha_range}, '
        f'default alpha if issues: {default_alpha_if_issues}, '
        f'NN distance scale factor: {nn_dist_scale_factor}',
        f'Fraction of points allowed outside: {frac_points_allowed_out}',
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
                concave_hull, final_alpha = get_concave_hull_python(
                    latent_space_all,
                    target_alpha_range=target_alpha_range,
                    default_alpha_if_issues=default_alpha_if_issues,
                    nn_dist_scale_factor=nn_dist_scale_factor,
                    frac_points_allowed_out=frac_points_allowed_out,
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
                        alpha=final_alpha,
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
