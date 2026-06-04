"""Compute diversity/coverage metrics from atlas database extxyz files."""

import argparse
import json
import logging
import pathlib
import warnings

import numpy as np
import torch
from ase.io import read
from rich.console import Console
from rich.table import Table
from shapely.geometry import Polygon

import atlas.active_learning.active_learning_utils as atl_al_ut
from atlas.active_learning.extrapolation import autoencoder as atl_ae
from atlas.active_learning.extrapolation import morphological_closing as atl_morph
from atlas.core.code_utils import custom_print
from atlas.core.database import diversity_metrics as atl_div

warnings.filterwarnings('ignore', category=UserWarning, message='.*weights_only.*')
logging.getLogger('mace').setLevel(logging.ERROR)
logging.getLogger('e3nn').setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute metrics from atlas extxyz files.'
    )
    parser.add_argument(
        'datasets',
        nargs='+',
        type=str,
        help='Path(s) to extxyz file(s).',
    )
    parser.add_argument(
        '--descriptor_type',
        type=str,
        default='mace',
        help='Descriptor type (default: mace).',
    )
    parser.add_argument(
        '--mace_model',
        type=str,
        default='mace:mp-small',
        help='MACE model name (default: mace:mp-small).',
    )
    parser.add_argument(
        '--descriptor_settings_soap',
        type=str,
        default=None,
        help='JSON string for SOAP descriptor settings (optional).',
    )
    parser.add_argument(
        '--descriptor_settings_mace',
        type=str,
        default='{"device": "cuda", "default_dtype": "float32",'
        ' "outer_average": false}',
        help='JSON string for MACE settings (default: cuda, float32, no outer avg).',
    )
    parser.add_argument(
        '--autoencoder_model_path',
        type=str,
        default='/BACKUP/tests/p2/get_metrics_old_runs/autoencoder/zan_autoencoder.pth',
        help='Path to the autoencoder .pth file.',
    )
    parser.add_argument(
        '--selected_steps',
        type=str,
        default='all',
        help='Comma-separated list of AL step numbers to evaluate, or "all". '
        'Example: "0,1,2,3,4,5,7,10,20,30,40,-1" (default: all).',
    )
    parser.add_argument(
        '--vendi_subset_size',
        type=int,
        default=10000,
        help='Subset size for Vendi score sampling (default: 10000).',
    )
    parser.add_argument(
        '--vendi_sigma',
        type=float,
        default=0.0868,
        help='Sigma for Vendi score RBF kernel (default: 0.0868).',
    )
    parser.add_argument(
        '--vendi_k',
        type=int,
        default=1,
        help='k for Vendi score (default: 1).',
    )
    parser.add_argument(
        '--vendi_num_iter',
        type=int,
        default=2,
        help='Number of Vendi score subsampling iterations (default: 2).',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional path to save results as a JSON file.',
    )
    return parser.parse_args()


# Metric functions


def compute_vendi_score(
    latent_space_dict,
    step_to_uuid_mapping,
    selected_steps,
    sigma=None,
    subset_size=10000,
    k=1,
    number_iterations=4,
):
    vendi_scores = {}
    cumulative_descriptors = []
    for step, uuids in step_to_uuid_mapping.items():
        # First accumulate the descriptors for the current step
        # otherwise, skipping steps would lead to incorrect calculations
        for uuid in uuids:
            curr_descriptor = latent_space_dict[uuid]['descriptors'][0].squeeze()
            cumulative_descriptors.append(curr_descriptor)

        # Then check if we need to compute metrics for this step, and
        # skip if necessary.
        if step not in selected_steps:
            continue

        custom_print(
            f'Processing step {step} that incorporated {len(uuids)} structures...',
            'empty',
        )

        # Stack the list into a numpy array once per step
        if cumulative_descriptors:
            descriptor_array = np.vstack(cumulative_descriptors)
        else:
            descriptor_array = np.empty(
                (0, latent_space_dict[uuids[0]]['descriptors'][0].shape[1])
            )
        custom_print(
            f'Getting Vendi score for array {descriptor_array.shape}...', 'empty'
        )
        vendi_score = atl_div.get_vendi_score_subsampling(
            feature_matrix=descriptor_array,
            subset_size=subset_size,
            n_iterations=number_iterations,
            sigma=sigma,
            k=k,
        )
        vendi_scores[str(step)] = {
            'mean_vendi_score': vendi_score[-2],
            'number_iterations': number_iterations,
            'sigma': sigma,
            'k': k,
            'std_dev': vendi_score[-1],
            'normalized_vendi_score': vendi_score[-2] / subset_size
            if subset_size > 0
            else 0.0,
            'normalized_std_dev': vendi_score[-1] / subset_size
            if subset_size > 0
            else 0.0,
        }
        print()
    return vendi_scores


def process_data_boundaries(uuid_mapping, latent_space_dict, selected_steps):
    data_boundaries = {}
    uuids_to_check = []
    for step, uuids in uuid_mapping.items():
        # Always accumulate the UUIDs to avoid issues when skipping steps
        uuids_to_check.extend(uuids)

        # Then check if we need to compute metrics
        if step not in selected_steps:
            continue

        custom_print(
            f'Processing step {step} that incorporated {len(uuids)} structures...',
            'empty',
        )
        latent_spaces = []
        for curr_uuid in uuids_to_check:
            curr_latent_space = latent_space_dict[curr_uuid]['latent_space'].squeeze()
            latent_spaces.append(curr_latent_space)
        latent_space_arr = np.vstack(latent_spaces)
        results = atl_morph.process_morphological_closing(
            latent_space_arr[:, 0], latent_space_arr[:, 1], disk_size=1
        )
        curr_area = 0.0
        for boundary in results['data_boundaries']:
            boundary = Polygon(boundary)
            curr_area += boundary.area
        data_boundaries[step] = {
            'data_boundaries': results['data_boundaries'],
            'uuids': uuids,
            'boundary_area': curr_area,
        }
    return data_boundaries


def compute_area_difference(boundary_information, run_results=None):
    area_differences = {}
    if '0' not in boundary_information and (
        run_results and run_results.get('0', {}).get('Boundary Area', None)
    ):
        boundary_information[0] = {'boundary_area': run_results['0']['Boundary Area']}
        boundary_information = dict(sorted(boundary_information.items()))
    for step, info in boundary_information.items():
        if step == 0:
            initial_area = info['boundary_area']
        curr_area = info['boundary_area']
        area_differences[step] = {
            'current_area': curr_area,
            'area_difference': curr_area - initial_area,
        }
    return area_differences


def compute_area_growth_ratio(area_differences, run_results=None):
    area_ratios = {}
    if '0' not in area_differences and (
        run_results and run_results.get('0', {}).get('Boundary Area', None)
    ):
        area_differences[0] = {'current_area': run_results['0']['Boundary Area']}
        area_differences = dict(sorted(area_differences.items()))
    for step, area in area_differences.items():
        area_ratios[step] = {}
        if step == 0:
            initial_area = area['current_area']
        current_area = area['current_area']
        area_ratios[step]['area_ratio'] = (
            current_area / initial_area if initial_area > 0 else float('inf')
        )
    return area_ratios


def associate_steps_to_uuids(database, latent_space_dict):
    last_step = max([struct.info.get('atl_al_step', 0) for struct in database])
    step_to_uuid = {step: [] for step in range(last_step + 1)}
    for struct in database:
        step = struct.info.get('atl_al_step', 0)
        uuid = struct.info.get('atl_id')
        if uuid in latent_space_dict:
            step_to_uuid[step].append(uuid)
    return step_to_uuid


def run_database_analysis():
    args = parse_args()

    descriptor_settings_mace = json.loads(args.descriptor_settings_mace)

    for dataset_path_str in args.datasets:
        dataset_path = pathlib.Path(dataset_path_str).resolve()
        if not dataset_path.exists():
            custom_print(f'File not found: {dataset_path}', 'error')
            continue

        run_name = dataset_path.stem
        custom_print(f"Processing: '{dataset_path.name}'.", 'info')

        # Load frames
        curr_db = read(str(dataset_path), format='extxyz', index=':')
        custom_print(f"Loaded '{len(curr_db)}' frames.", 'info')

        # Resolve selected_steps, "all" means every step present in the dataset
        if args.selected_steps.strip().lower() == 'all':
            all_steps = sorted(
                {struct.info.get('atl_al_step', 0) for struct in curr_db}
            )
            selected_steps = all_steps
            custom_print(
                f'Using all {len(selected_steps)} steps '
                f'({selected_steps[0]}-{selected_steps[-1]}).',
                'info',
            )
        else:
            selected_steps = [int(s.strip()) for s in args.selected_steps.split(',')]

        # Generate descriptors
        descr_dict, descr_arr, uuids = atl_al_ut.generate_descriptors(
            database=curr_db,
            descriptor_type=args.descriptor_type,
            model_path=args.mace_model,
            descriptor_settings=descriptor_settings_mace,
            verbose=True,
        )

        # Load autoencoder
        autoencoder_model = torch.load(args.autoencoder_model_path, weights_only=False)

        # Get latent space
        latent_space_dict = atl_ae.get_latent_space_autoencoder(
            model=autoencoder_model,
            descriptor_dict=descr_dict,
            device='cuda',
            dtype=torch.float32,
            standardize_data=True,
            autoencoder_path=args.autoencoder_model_path,
        )
        print()

        step_to_uuid_mapping = associate_steps_to_uuids(
            database=curr_db, latent_space_dict=latent_space_dict
        )

        # Resolve -1 sentinel
        resolved_steps = [
            max(step_to_uuid_mapping.keys()) if s == -1 else s for s in selected_steps
        ]

        custom_print('Processing data boundaries and areas...', 'info')
        boundary_information = process_data_boundaries(
            uuid_mapping=step_to_uuid_mapping,
            latent_space_dict=latent_space_dict,
            selected_steps=resolved_steps,
        )
        print()

        custom_print('Obtaining Vendi Scores...', 'info')
        vendi_score_information = compute_vendi_score(
            latent_space_dict=latent_space_dict,
            step_to_uuid_mapping=step_to_uuid_mapping,
            selected_steps=resolved_steps,
            number_iterations=args.vendi_num_iter,
            subset_size=args.vendi_subset_size,
            sigma=args.vendi_sigma,
            k=args.vendi_k,
        )
        custom_print('Done obtaining Vendi Scores!', 'info')
        print()

        # Area metrics
        area_differences = compute_area_difference(
            boundary_information=boundary_information, run_results=None
        )
        area_growth_ratios = compute_area_growth_ratio(
            area_differences=area_differences, run_results=None
        )

        # Assemble metrics
        metrics = {}
        for step in resolved_steps:
            if step not in boundary_information:
                custom_print(
                    f'Warning: Step {step} not found in boundary information.'
                    ' Skipping.',
                    'warning',
                )
                continue

            vendi_score = vendi_score_information.get(str(step), {}).get(
                'mean_vendi_score', None
            )
            norm_vendi_score = vendi_score_information.get(str(step), {}).get(
                'normalized_vendi_score', None
            )
            boundary_area = boundary_information.get(step, {}).get('boundary_area')
            boundary_area_diff = area_differences.get(step, {}).get('area_difference')
            boundary_area_growth_ratio = area_growth_ratios.get(step, {}).get(
                'area_ratio'
            )

            metrics[step] = {
                'Vendi Score': vendi_score,
                'Normalized Vendi Score': norm_vendi_score,
                'Boundary Area': boundary_area,
                'Boundary Area Difference': boundary_area_diff,
                'Boundary Area Growth Ratio': boundary_area_growth_ratio,
            }

        custom_print('Printing final results...', 'info')
        print()

        # Print table
        console = Console()
        first_metrics = next(iter(metrics.values()))
        columns = ['Step', *first_metrics.keys()]
        table = Table(
            title=run_name.title(), show_header=True, header_style='bold magenta'
        )
        for col in columns:
            if col == 'Step':
                table.add_column(col, justify='right', style='bold cyan')
            else:
                table.add_column(col)

        for step in sorted(metrics.keys()):
            row_values = [str(step)]
            for key in first_metrics:
                val = metrics[step].get(key, '')
                if isinstance(val, float):
                    row_values.append(f'{val:.6f}')
                else:
                    row_values.append(str(val))
            table.add_row(*row_values)

        console.print(table)
        print()

        if args.output:
            output_path = pathlib.Path(args.output)
            # If output is a directory, write per-dataset files inside it
            if (
                args.output.endswith('/')
                or args.output.endswith('\\')
                or (len(args.datasets) > 1 and not output_path.suffix)
            ):
                output_dir = output_path.resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / f'{run_name}.json').write_text(
                    json.dumps(metrics, indent=4)
                )
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    run_database_analysis()
