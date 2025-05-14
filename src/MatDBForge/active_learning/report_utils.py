"""Utils for generating reports."""

import datetime
import json
import os
import pickle
import re
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from aiida import load_profile, orm
from aiida.cmdline.utils.common import get_workchain_report
from ase.io import read as ase_read
from ase.io import write as ase_write
from matplotlib import colormaps, gridspec
from plotly.subplots import make_subplots

from MatDBForge.active_learning.active_learning_utils import (
    generate_descriptors,
    remove_isolated_atoms,
    simplify_forces_struct,
)
from MatDBForge.active_learning.extrapolation.autoencoder import (
    get_latent_space_autoencoder,
    load_autoencoder_model,
)
from MatDBForge.active_learning.extrapolation.concave_hull import get_concave_hull_julia
from MatDBForge.core.code_utils import custom_print, init_logger

# Define colors from the gruvbox palette
COLORS = [
    '#6aa1f4',
    '#ffa6c8',
    '#3ed04e',
    '#e50e3f',
    '#fed65b',
    '#7532c8',
    '#ff8834',
]
LINE_COLOR = '#28282855'

# Define the process labels for the different calculation types
TRAINING_LABEL = 'TrainMACEModelCalculation'
DESCRIPTORS_LABEL = 'GetDescriptorsCombinedCalculation'
MD_LABEL = 'ProcessMDSeedStructCalculation'
DFT_LABEL = 'EvaluateMACEConfigsCalculation'
AL_STEP_WORKCHAIN_LABEL = (
    'SimpleActiveLearningWorkChain'  # Ensure this matches your WorkChain label
)

# Define stage keys for iteration
STAGE_KEYS: list[str] = [
    'training',
    'descriptors',
    'md',
    'dft',
]


def gen_al_loop_report(
    loop_id: int | str = None,
    log_path: str = None,
    get_error_plot: bool = False,
    device='cpu',
    model_path: str | Path = None,
    database_path: str | Path = None,
    threshold_E_meV: float = None,
    threshold_F_meV: float = None,
    remove_outliers: bool = False,
    title: str = None,
    get_latent_space: bool = False,
    autoencoder_path: str = None,
    limit_num_steps: int = None,
):
    # Init logger
    init_logger(source='al_loop_report_gen')

    # Get report data and stats
    (
        title,
        al_loop_node,
        ini_db_size,
        model_acc_multiplier,
        stats_dict,
    ) = get_loop_report(loop_id, log_path, title)

    # Parse dict
    it_idx = [stats_dict[i]['it_idx'] for i in stats_dict]
    seed_gen_db_sizes = [stats_dict[i]['seed_gen_db_size'] for i in stats_dict]
    train_db_sizes = [stats_dict[i]['train_db_size'] for i in stats_dict]
    mace_e = [stats_dict[i]['mace_e'] for i in stats_dict]
    mace_f = [stats_dict[i]['mace_f'] for i in stats_dict]

    # Adjust number of panels according to options
    num_rows = 2
    num_cols = 2

    if get_error_plot:
        num_cols += 1
    if get_latent_space:
        num_rows += 1

    # Automatically adjust figure size
    fig_width = num_cols * 5
    fig_height = num_rows * 5

    fig = plt.figure(layout='tight', figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols, figure=fig)

    # Get report
    filename: Path = plot_al_loop_report(
        limit_num_steps=limit_num_steps,
        seed_gen_db_sizes=seed_gen_db_sizes,
        train_db_sizes=train_db_sizes,
        mace_e=mace_e,
        mace_f=mace_f,
        it_idx=it_idx,
        ax=gs,
        model_acc_multiplier=model_acc_multiplier,
    )

    # bold title
    fig.suptitle(f"Results for run: '{title}'", fontsize=14, fontweight='bold')

    if get_error_plot:
        custom_print(
            (
                'Generating database error plot. '
                f'Remove outliers: {bool(remove_outliers)}'
            ),
            'info',
        )

        if not database_path:
            try:
                database_path = al_loop_node.inputs.active_learning.init_db_path.value
                custom_print(
                    'Database path not provided. Using the one from the AL loop.',
                    'warn',
                )
            except Exception:
                custom_print(
                    (
                        'Database path not provided and unable to gather initial db '
                        'from AL workchain. Skipping error plot.'
                    ),
                    'error',
                )

        if database_path:
            generate_error_plot(
                al_loop_node=al_loop_node,
                device_str=device,
                ax=gs,
                database_path=database_path,
                model_path=model_path,
                threshold_E_meV=threshold_E_meV,
                threshold_F_meV=threshold_F_meV,
                remove_outliers=remove_outliers,
                # train_db=ase_read(database_path, format='extxyz', index=':'),
            )

    if get_latent_space:
        # raise NotImplementedError("Latent space evolution plot not implemented yet.")
        custom_print(
            'Plotting latent space evolution...',
            'info',
        )

        # TODO: Get the autoencoder model from the first iteration
        if not autoencoder_path:
            ...
        else:
            autoencoder_model = list(Path(autoencoder_path).glob('*.pth'))[0]
            database_files = list(Path(autoencoder_path).glob('database*.xyz'))
            database_files = [file.name for file in database_files]
            database_files.sort(
                key=lambda x: int(str(x).replace('.xyz', '').split('_')[-1])
            )

        generate_latent_space_evol(
            al_loop_node=al_loop_node,
            device_str=device,
            ax=gs,
            database_path=database_path,
            model_path=model_path,
            autoencoder_model=autoencoder_model,
            autoencoder_path=autoencoder_path,
            databases=database_files,
        )

    plt.tight_layout()
    fig.subplots_adjust(top=0.925, bottom=0.0725, right=0.95, left=0.075)

    # Saving both png and svg
    plt.savefig(filename.with_suffix('.png'), dpi=300)
    plt.savefig(filename.with_suffix('.svg'), dpi=300)
    custom_print(f"Saved report to '{filename.with_suffix('.png')}'.", 'info')

    plt.clf()

    custom_print('Report generation complete.', 'done')


def get_loop_report(
    loop_id=None,
    log_path=None,
    title=None,
    model_acc_multiplier=None,
):
    """
    Parses an active learning loop report to extract summary statistics.

    Args:
        loop_id: Identifier for the loop (currently not used for parsing).
        log_path: Filesystem path to the log file. Used if report_str is not provided.
        title: Optional title for the run. If None, attempts to extract from report.
        model_acc_multiplier: Multiplier for model accuracy (can be None).

    Returns
    -------
        A tuple containing:
        - title (str): The title of the run.
        - al_loop_node (int): The active learning loop node ID.
        - ini_db_size (int): The initial database size.
        - model_acc_multiplier (float or None): The model accuracy multiplier.
        - stats_dict (dict): A dictionary with statistics for each iteration.
            Example: {
                1: {
                    "it_idx": 1, "mace_e": 0.778, "mace_f": 8.204,
                    "train_db_size": 3706, "seed_gen_db_sizes": 3689, "finished": True
                }, ...
            }
    """
    if loop_id:
        al_loop_node = orm.load_node(loop_id)
        report = get_workchain_report(al_loop_node, levelname='REPORT')
    if log_path:
        with open(log_path) as f:
            report = f.read()
        loop_id = re.compile(r'.*ActiveLearningBaseWorkChain\|setup].*').findall(report)
        loop_id = int(loop_id[0].split('|')[0].replace('[', '').strip())
        try:
            al_loop_node = orm.load_node(loop_id)
        except Exception:
            al_loop_node = loop_id

    try:
        ini_db_line = re.compile(r'initial database containing.*').findall(report)
        ini_db_size = int(ini_db_line[0].split()[3].replace("'", ''))
    except IndexError:
        ini_db_size = 0

    try:
        if isinstance(al_loop_node, orm.Node):
            model_acc_multiplier = (
                al_loop_node.inputs.active_learning.model_acc_multiplier.value
            )
    except AttributeError:
        model_acc_multiplier = 1.0

    stats_dict = {
        0: {
            'it_idx': 0,
            'mace_e': None,
            'mace_f': None,
            'train_db_size': ini_db_size,  # Uses the pre-defined ini_db_size
            'seed_gen_db_size': ini_db_size,  # Uses the pre-defined ini_db_size
            'finished': True,
        }
    }

    # Extract iteration indices from "Starting AL Loop iteration X/Y..."
    it_idx_list = []
    start_loop_matches = re.finditer(r'Starting AL Loop iteration (\d+)/\d+', report)
    for match in start_loop_matches:
        it_idx_list.append(int(match.group(1)))

    # Extract DB sizes from "Iteration X: seed_gen_db Y, training_db: Z, entries"
    # Store these in a dictionary keyed by iteration number for easy lookup.
    db_data_by_iter = {}
    db_lines_matches = re.finditer(
        r'Iteration (\d+): seed_gen_db (\S+), training_db: (\S+) entries', report
    )
    for match in db_lines_matches:
        iter_num_db = int(match.group(1))
        db_data_by_iter[iter_num_db] = {
            'seed_gen_db_size': int(match.group(2).replace(',', '')),
            'train_db_size': int(match.group(3).replace(',', '')),
        }

    # Extract MACE performance from "Best model of current step..."
    # This is collected sequentially and assumed to correspond to
    # the sequence of started iterations.
    mace_data_list = []
    best_model_matches = re.finditer(
        r'Best model of current step .* RMSE E: (\S+) meV/at, RMSE F: (\S+) meV/Å',
        report,
    )
    for match in best_model_matches:
        mace_data_list.append(
            {
                'mace_e': float(match.group(1)),
                'mace_f': float(match.group(2)),
            }
        )

    # Populate stats_dict for iterations 1 through N
    for i, current_iter_num in enumerate(it_idx_list):
        stats_dict[current_iter_num] = {
            'it_idx': current_iter_num,
            'mace_e': None,
            'mace_f': None,
            'train_db_size': None,
            'seed_gen_db_size': None,
            'finished': False,  # Default, updated based on db_data_by_iter
        }

        # Assign MACE data if available for this sequential position
        if i < len(mace_data_list):
            stats_dict[current_iter_num]['mace_e'] = mace_data_list[i]['mace_e']
            stats_dict[current_iter_num]['mace_f'] = mace_data_list[i]['mace_f']

        # Check for the "Iteration X: seed_gen_db..." line for this current_iter_num
        # If found, the step is considered "finished" and DB sizes are populated.
        # The "Starting AL Loop iteration X..." condition is met by iterating
        # it_idx_list.
        if current_iter_num in db_data_by_iter:
            stats_dict[current_iter_num]['train_db_size'] = db_data_by_iter[
                current_iter_num
            ]['train_db_size']
            stats_dict[current_iter_num]['seed_gen_db_size'] = db_data_by_iter[
                current_iter_num
            ]['seed_gen_db_size']
            stats_dict[current_iter_num]['finished'] = True

    # Get run name
    if not title:  # If title wasn't passed in or set in the omitted part
        try:
            title_regex = r'Active Learning Inputs:\s*\n({.*?})\n'
            match_obj = re.search(title_regex, report, re.DOTALL)
            if match_obj:
                settings_dict_string = match_obj.group(1)
                # Convert Python dict-like string to valid JSON string
                settings_dict_string = (
                    settings_dict_string.replace("'", '"')
                    .replace('True', 'true')
                    .replace('False', 'false')
                    .replace('None', 'null')
                )
                settings_dict = json.loads(settings_dict_string)
                title = settings_dict.get('run_name')
        except Exception:
            # If parsing fails, title remains None or its previous value.
            # The original code had a generic Exception, kept here.
            # Consider more specific error handling if needed.
            pass

    if not title:
        title = f'{al_loop_node}'  # Assumes al_loop_node is defined

    return (
        title,
        al_loop_node,
        ini_db_size,
        model_acc_multiplier,
        stats_dict,
    )


def get_mace_eval_results(
    al_loop_node, database_path, device_str='cpu', model_path=None, folder_path=None
):
    from mace.calculators import MACECalculator
    from mace.tools import torch_tools
    from rich.progress import track

    torch_tools.set_default_dtype('float32')

    # Create tmp folder in the current directory

    with tempfile.TemporaryDirectory(dir=folder_path) as tmp_dir:
        # Get last iteration from aiida
        try:
            all_iters = [
                stp
                for stp in al_loop_node.called
                if stp.process_type == 'aiida.workflows:mdb-active-learning'
                or stp.process_type == 'aiida.workflows:mdb-simple-active-learning'
            ]
            last_iter = all_iters[-1]
        except AttributeError:
            last_iter = al_loop_node
            custom_print(
                'No nodes found for the AL loop. Is it stored in this machine?',
                'warning',
            )

        # Copy the training database to the tmp folder
        if database_path:
            training_db_path = Path(database_path)
        else:
            training_db_path = Path(last_iter.inputs.training_db_path.value)

        training_db_cp_path = shutil.copy(training_db_path, tmp_dir)
        if not model_path:
            custom_print('Using best model from last workchain iteration...', 'info')
            if not last_iter.is_finished and len(all_iters) > 1:
                model_file: orm.SinglefileData = all_iters[-2].outputs.m0_model_file
            else:
                model_file: orm.SinglefileData = last_iter.outputs.m0_model_file
            with model_file.as_path() as f:
                model_path = shutil.copy(f, tmp_dir)
        else:
            custom_print(f'Using model from: {model_path}', 'info')

        # Load the training database
        train_db = ase_read(training_db_cp_path, format='extxyz', index=':')

        # Load model
        mace_model = MACECalculator(
            model_paths=model_path,
            device=device_str,
            default_dtype='float32',
        )

        if not folder_path:
            folder_path = Path('.')

        files = (folder_path / 'E_nn_list.npy', folder_path / 'F_nn_list.npy')
        file_names = (files[0].name, files[1].name)

        E_nn_list = []
        F_nn_list = []
        # Iterate over the data loader to get the model energies
        # Use a rich progress bar to show the progress
        # Only run if any of the two files are missing
        if len(set(file_names) - set(os.listdir(folder_path))) != 0:
            for struct in track(
                train_db,
                description='Evaluating structures...',
            ):
                struct.calc = mace_model
                energy_output = struct.get_potential_energy()
                E_nn_list.append(energy_output)

                # Getting forces
                # Shape: (n_atoms, 3)
                forces_output = struct.get_forces()
                forces_max, _ = simplify_forces_struct(forces_output)
                F_nn_list.append(forces_max)

        else:
            custom_print('Loading energies and forces from file...', 'info')
            E_nn_list = np.load(files[0])
            F_nn_list = np.load(files[1])

        np.save(files[0], E_nn_list)
        np.save(files[1], F_nn_list)

    return E_nn_list, F_nn_list


def mark_and_remove_outliers(
    E_sorted_diff_list_meV: list,
    F_sorted_diff_list_meV: list,
    E_diff_list_meV: list,
    F_diff_list_meV: list,
    E_nn_list_per_at: list,
    F_nn_list_per_at: list,
    E_dft_list_per_at: list,
    F_dft_list_per_at: list,
    threshold_E_meV: float,
    threshold_F_meV: float,
    remove_outliers: bool,
    train_db: list,
):
    outlier_list = []
    remove_indices = []

    # Mark outliers, appending the E difference in meV to the structure info dict.
    # This will raise an error if the number of elements in the lists are not the same.
    # This could happen if E or F were not calculated for all structures.
    for E_val, F_val in zip(
        E_sorted_diff_list_meV[::-1], F_sorted_diff_list_meV[::-1], strict=False
    ):
        if (
            abs(E_diff_list_meV[E_val]) > threshold_E_meV
            or abs(F_diff_list_meV[F_val]) > threshold_F_meV
        ):
            struct = train_db[int(E_val)]
            struct.info['E_dft_nn_diff_meV'] = E_diff_list_meV[E_val]
            struct.info['F_dft_nn_diff_meV'] = F_diff_list_meV[F_val]
            outlier_list.append(struct)

            # Add outler indices to list for removal
            if remove_outliers:
                remove_indices.append(E_val)

    if remove_outliers:
        train_db = [
            struct
            for struct in train_db
            if train_db.index(struct) not in remove_indices
        ]

        # Energies
        E_diff_list_meV = np.delete(E_diff_list_meV, remove_indices)
        E_nn_list_per_at = np.delete(E_nn_list_per_at, remove_indices)
        E_dft_list_per_at = np.delete(E_dft_list_per_at, remove_indices)

        # Forces
        F_diff_list_meV = np.delete(F_diff_list_meV, remove_indices)
        F_nn_list_per_at = np.delete(F_nn_list_per_at, remove_indices)
        F_dft_list_per_at = np.delete(F_dft_list_per_at, remove_indices)

        # Save the outliers to a file,
        if len(outlier_list) > 0:
            ase_write('outliers.xyz', outlier_list, format='extxyz')
            custom_print(
                (
                    f"Saved {len(outlier_list)} outliers to 'outliers.xyz'. "
                    f'Used E: {threshold_E_meV} meV and F: {threshold_F_meV} meV'
                    ' as threshold.'
                ),
                'info',
            )

    return (
        E_diff_list_meV,
        E_nn_list_per_at,
        E_dft_list_per_at,
        F_diff_list_meV,
        F_nn_list_per_at,
        F_dft_list_per_at,
        outlier_list,
        remove_indices,
    )


def generate_error_plot(
    al_loop_node,
    device_str: str = 'cpu',
    ax=None,
    database_path: str = None,
    model_path: str = None,
    threshold_E_meV: float = None,
    threshold_F_meV: float = None,
    remove_outliers: bool = False,
    # train_db:list=None,
):
    E_nn_list, F_nn_list = get_mace_eval_results(
        al_loop_node=al_loop_node,
        device_str=device_str,
        database_path=database_path,
        model_path=model_path,
    )

    train_db = ase_read(database_path, format='extxyz', index=':')

    E_dft_list = np.array([atoms.info['REF_energy'] for atoms in train_db])
    F_dft_list = np.array(
        [simplify_forces_struct(atoms.arrays['REF_forces'])[0] for atoms in train_db]
    )

    # Getting atom count
    struct_len_list = np.array([len(atoms) for atoms in train_db])
    # Getting energy per atom
    E_nn_list_per_at = E_nn_list / struct_len_list
    E_dft_list_per_at = E_dft_list / struct_len_list

    # Getting force per atom
    F_nn_list_per_at = F_nn_list / struct_len_list
    F_dft_list_per_at = F_dft_list / struct_len_list

    E_diff_list_meV = (E_nn_list_per_at - E_dft_list_per_at) * 1000
    # F_diff_list_meV = (F_nn_list_per_at - F_dft_list_per_at) * 1000
    F_diff_list_meV = (F_nn_list - F_dft_list) * 1000

    # Sort diff_list_meV, keeping the original indices
    E_sorted_diff_list_meV = np.argsort(np.abs(E_diff_list_meV))
    F_sorted_diff_list_meV = np.argsort(np.abs(E_diff_list_meV))

    (
        E_diff_list_meV,
        E_nn_list_per_at,
        E_dft_list_per_at,
        F_diff_list_meV,
        F_nn_list_per_at,
        F_dft_list_per_at,
        outlier_list,
        remove_indices,
    ) = mark_and_remove_outliers(
        E_sorted_diff_list_meV,
        F_sorted_diff_list_meV,
        E_diff_list_meV,
        F_diff_list_meV,
        E_nn_list_per_at,
        F_nn_list_per_at,
        E_dft_list_per_at,
        F_dft_list_per_at,
        threshold_E_meV,
        threshold_F_meV,
        remove_outliers,
        train_db,
    )

    # save stats into a numpy array file
    export_list = []
    for idx in range(len(E_dft_list_per_at)):
        export_list.append(
            [
                train_db[idx].info.get('mdb_id'),
                E_dft_list_per_at[idx],
                E_nn_list_per_at[idx],
                F_dft_list_per_at[idx],
                F_nn_list_per_at[idx],
                E_diff_list_meV[idx],
                F_diff_list_meV[idx],
            ]
        )
    export_array = np.array(export_list, dtype=object)
    np.savetxt(
        'stats.out',
        export_array,
        fmt='%s %.8f %.8f %.8f %.8f %.8f %.8f',
        header='mdb_id E_dft_per_at E_nn_per_at F_dft_per_at F_nn_per_at'
        ' E_diff_meV F_diff_meV',
        comments='# ',
    )

    # Get RMSD
    E_rmsd = np.sqrt(np.mean(E_diff_list_meV**2))  # meV/atom
    F_rmsd = np.sqrt(np.mean(F_diff_list_meV**2))  # meV/A

    # Get MAE
    E_mae = np.mean(np.abs(E_diff_list_meV))  # meV/atom
    F_mae = np.mean(np.abs(F_diff_list_meV))  # meV/A

    # Remove isolated atoms
    (
        train_db,
        E_dft_list_per_at,
        E_nn_list_per_at,
        F_dft_list_per_at,
        F_nn_list_per_at,
        E_diff_list_meV,
        F_diff_list_meV,
    ) = remove_isolated_atoms(
        train_db,
        E_dft_list_per_at,
        E_nn_list_per_at,
        F_dft_list_per_at,
        F_nn_list_per_at,
        E_diff_list_meV,
        F_diff_list_meV,
    )

    tex_x_pos = 0.1
    tex_y_pos = 0.80
    save_fig_now = False

    if ax is None:
        save_fig_now = True
        ax = plt.subplots(2, 3, figsize=(18, 12))[1]

    inner_grid = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=ax[0, 2], hspace=0.5
    )
    ax1_top = ax.figure.add_subplot(inner_grid[0, 0])

    ax1_top.text(
        x=tex_x_pos,
        y=tex_y_pos,
        s=f'MAE: {E_mae:.3f} meV/atom\nRMSD: {E_rmsd:.3f} meV/atom',
        transform=ax1_top.transAxes,
    )

    ax1_top.plot(
        E_dft_list_per_at,
        E_nn_list_per_at,
        'o',
        color='#83a598',
        alpha=0.5,
        markeredgewidth=0.5,
        markeredgecolor='#282828',
        markersize=3,
    )
    ax1_top.plot(
        E_dft_list_per_at,
        E_dft_list_per_at,
        color='#b16286',
        linestyle='--',
    )
    ax1_top.set_xlabel('DFT Energy [eV/atom]')
    ax1_top.set_ylabel('NN Energy [eV/atom]')
    ax1_top.set_title('Energy comparison')

    ax1_top.annotate(
        'e)',
        xy=(0, 1),
        xycoords='axes fraction',
        xytext=(+0.5, -0.5),
        textcoords='offset fontsize',
        fontsize='medium',
        verticalalignment='top',
        fontfamily='sans-serif',
        bbox=dict(facecolor='1', edgecolor='none', pad=3.0),
    )

    ax1_bottom = ax.figure.add_subplot(inner_grid[1, 0])
    ax1_bottom.annotate(
        'f)',
        xy=(0, 1),
        xycoords='axes fraction',
        xytext=(+0.5, -0.5),
        textcoords='offset fontsize',
        fontsize='medium',
        verticalalignment='top',
        bbox=dict(facecolor='1', edgecolor='none', pad=3.0),
    )
    ax1_bottom.text(
        x=tex_x_pos,
        y=tex_y_pos,
        s=f'MAE: {F_mae:.3f} meV/A\nRMSD: {F_rmsd:.3f} meV/A',
        transform=ax1_bottom.transAxes,
    )

    ax1_bottom.plot(
        F_dft_list_per_at,
        F_nn_list_per_at,
        'o',
        color='#83a598',
        alpha=0.5,
        markeredgewidth=0.5,
        markeredgecolor='#282828',
        markersize=3,
    )
    ax1_bottom.plot(
        F_dft_list_per_at, F_dft_list_per_at, color='#b16286', linestyle='--'
    )
    ax1_bottom.set_xlabel('DFT Forces [eV/A]')
    ax1_bottom.set_ylabel('NN Forces [eV/A]')
    ax1_bottom.set_title('Forces comparison')

    inner_grid_diff = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=ax[1, 2], hspace=0.5
    )
    ax2_top = ax.figure.add_subplot(inner_grid_diff[0, 0])
    ax2_bottom = ax.figure.add_subplot(inner_grid_diff[1, 0])

    # Plotting the EDFT, ENN and their difference in a line plot, using the
    # structure index as the x-axis.
    ax2_top.plot(E_dft_list, color='#83a598', label='DFT Energy', alpha=0.5)
    ax2_top.plot(E_nn_list, color='#b16286', label='NN Energy', alpha=0.5)

    ax2_bottom.plot(F_dft_list, color='#83a598', label='DFT Forces', alpha=0.5)
    ax2_bottom.plot(F_nn_list, color='#b16286', label='NN Forces', alpha=0.5)

    # Use secondary y-axis for the difference
    ax_twin = ax2_top.twinx()
    ax_twin_b = ax2_bottom.twinx()
    ax_twin.plot(
        np.abs(E_diff_list_meV),
        color='#fb4934',
        label='Difference',
        linewidth=0.75,
        alpha=0.75,
    )
    ax_twin_b.plot(
        np.abs(F_diff_list_meV),
        color='#fb4934',
        label='Difference',
        linewidth=0.75,
        alpha=0.75,
    )
    ax_twin.set_ylabel('Abs. E diff [meV/atom]')
    ax_twin_b.set_ylabel('Abs. F diff [meV/A]')

    ax_twin.spines['right'].set_color('#fb4934')
    ax_twin.tick_params(axis='y', colors='#fb4934')
    ax_twin.yaxis.label.set_color('#fb4934')
    ax_twin.title.set_color('#fb4934')
    ax_twin_b.spines['right'].set_color('#fb4934')
    ax_twin_b.tick_params(axis='y', colors='#fb4934')
    ax_twin_b.yaxis.label.set_color('#fb4934')
    ax_twin_b.title.set_color('#fb4934')

    # Set labels and title
    ax2_top.set_xlabel('Structure index')
    ax2_top.set_ylabel('Energy [eV]')
    ax2_top.set_title('Energy comparison')
    ax2_bottom.set_xlabel('Structure index')
    ax2_bottom.set_ylabel('Forces [eV/A]')
    ax2_bottom.set_title('Forces comparison')

    lines, labels = ax2_top.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_twin.legend(lines + lines2, labels + labels2)
    lines_b, labels_b = ax2_bottom.get_legend_handles_labels()
    lines2_b, labels2_b = ax_twin_b.get_legend_handles_labels()
    ax_twin_b.legend(lines_b + lines2_b, labels_b + labels2_b)

    ax2_top.annotate(
        'g)',
        xy=(0, 1),
        xycoords='axes fraction',
        xytext=(+0.5, -0.5),
        textcoords='offset fontsize',
        fontsize='medium',
        verticalalignment='top',
        bbox=dict(facecolor='1', edgecolor='none', pad=3.0),
        zorder=102,
    )
    ax2_bottom.annotate(
        'h)',
        xy=(0, 1),
        xycoords='axes fraction',
        xytext=(+0.5, -0.5),
        textcoords='offset fontsize',
        fontsize='medium',
        verticalalignment='top',
        bbox=dict(facecolor='1', edgecolor='none', pad=3.0),
        zorder=102,
    )

    if save_fig_now:
        plt.tight_layout()
        filename = Path('./energy_difference.png').resolve()
        plt.savefig(filename, dpi=300)
        custom_print(f"Saved report to '{filename}'.", 'info')


def gen_batch_report(
    training_db_path: str,
    threshold_meV: float = 100,
    remove_outliers: bool = False,
    root_path: str | Path = '.',
    device_str='cuda',
):
    init_logger(source='gen_batch_report')
    custom_print('Generating al_loop batch report...', 'info')

    # Plot seed and train db sizes as a stacked bar chart over every iteration
    width = 0.25

    # Get unix timestamp for filename
    timestamp = int(time.time())
    filename = Path(f'al_loop_batch_report_{timestamp}').resolve()

    # Adjust number of panels according to options
    num_rows = 2
    num_cols = 2

    # Automatically adjust figure size
    fig_width = num_cols * 5
    fig_height = num_rows * 5

    fig = plt.figure(layout='tight', figsize=(fig_width, fig_height))
    ax = gridspec.GridSpec(nrows=num_rows, ncols=num_cols, figure=fig)

    root_path = Path(root_path).absolute()

    # Load the initial training database
    train_db = ase_read(training_db_path, format='extxyz', index=':')

    # Getting DFT energy and forces
    E_dft_list = np.array([atoms.info['REF_energy'] for atoms in train_db])
    F_dft_list = np.array(
        [simplify_forces_struct(atoms.arrays['REF_forces'])[0] for atoms in train_db]
    )

    # Get all directories in the root path
    dirs = [d for d in root_path.iterdir() if d.is_dir()]
    ind = np.array(range(len(dirs)))

    train_db_sizes_plot, seed_gen_db_sizes_plot = [], []
    mace_e_plot, mace_f_plot = [], []
    E_rmsd_plot, F_rmsd_plot = [], []
    E_mae_plot, F_mae_plot = [], []

    # Iterate over all directories to gather data
    for dir_path in dirs:
        model_file = list(dir_path.glob('*.model'))
        db_file = list(dir_path.glob('*.xyz'))

        # Check if the directories contain the necessary .log files
        log_file = list(dir_path.glob('*.log'))

        if len(log_file) == 0 or len(model_file) == 0 or len(db_file) == 0:
            custom_print(f'Missing an output file in {dir_path}.', 'error')
            continue
        if len(list(log_file)) > 1:
            custom_print(f'Repeated output files in {dir_path}.', 'error')
            continue

        (
            title,
            al_loop_node,
            ini_db_size,
            model_acc_multiplier,
            stats_dict,
        ) = get_loop_report(
            log_path=log_file[0],
        )

        # Parse dict
        # it_idx = [stats_dict[i]['it_idx'] for i in stats_dict]
        seed_gen_db_sizes = [stats_dict[i]['seed_gen_db_size'] for i in stats_dict]
        train_db_sizes = [stats_dict[i]['train_db_size'] for i in stats_dict]
        mace_e = [stats_dict[i]['mace_e'] for i in stats_dict]
        mace_f = [stats_dict[i]['mace_f'] for i in stats_dict]

        raise NotImplementedError(
            'This report type must be updated for its use with the'
            'new `get_loop_report` function.'
        )

        train_db_sizes_plot.append(train_db_sizes[-1])
        seed_gen_db_sizes_plot.append(seed_gen_db_sizes[-1])

        ## Model performance
        mace_e_plot.append(mace_e[-1])
        mace_f_plot.append(mace_f[-1])

        ## MAE and RMSD
        E_nn_list, F_nn_list = get_mace_eval_results(
            al_loop_node=al_loop_node,
            device_str=device_str,
            folder_path=dir_path,
            model_path=model_file[0],
            database_path=training_db_path,
        )
        E_dft_list = np.array([atoms.info['REF_energy'] for atoms in train_db])
        F_dft_list = np.array(
            [
                simplify_forces_struct(atoms.arrays['REF_forces'])[0]
                for atoms in train_db
            ]
        )

        # Getting atom count
        struct_len_list = np.array([len(atoms) for atoms in train_db])

        # Getting energy per atom
        E_nn_list_per_at = E_nn_list / struct_len_list
        E_dft_list_per_at = E_dft_list / struct_len_list

        # Getting force per atom
        F_nn_list_per_at = F_nn_list / struct_len_list
        F_dft_list_per_at = F_dft_list / struct_len_list

        E_diff_list_meV = (E_nn_list_per_at - E_dft_list_per_at) * 1000
        F_diff_list_meV = (F_nn_list_per_at - F_dft_list_per_at) * 1000

        # Sort diff_list_meV, keeping the original indices
        E_sorted_diff_list_meV = np.argsort(np.abs(E_diff_list_meV))
        F_sorted_diff_list_meV = np.argsort(np.abs(E_diff_list_meV))

        (
            E_diff_list_meV,
            E_nn_list_per_at,
            E_dft_list_per_at,
            F_diff_list_meV,
            F_nn_list_per_at,
            F_dft_list_per_at,
            outlier_list,
            remove_indices,
        ) = mark_and_remove_outliers(
            E_sorted_diff_list_meV,
            F_sorted_diff_list_meV,
            E_diff_list_meV,
            F_diff_list_meV,
            E_nn_list_per_at,
            F_nn_list_per_at,
            E_dft_list_per_at,
            F_dft_list_per_at,
            threshold_meV,
            remove_outliers,
        )

        # Get RMSD
        E_rmsd = np.sqrt(np.mean(E_diff_list_meV**2))
        F_rmsd = np.sqrt(np.mean(F_diff_list_meV**2))

        # Get MAE
        E_mae = np.mean(np.abs(E_diff_list_meV))
        F_mae = np.mean(np.abs(F_diff_list_meV))

        E_rmsd_plot.append(E_rmsd)
        F_rmsd_plot.append(F_rmsd)
        E_mae_plot.append(E_mae)
        F_mae_plot.append(F_mae)

    # Iterate over all directories
    # for dir_ind, dir_path in enumerate(dirs):

    run_color = np.array([mpl.colors.to_rgba(c) for c in COLORS])
    run_color_seed = [np.fmin(c + 0.15, np.ones(len(c))) for c in np.array(run_color)]
    clr_short = COLORS[: len(ind)]

    ax1 = ax.figure.add_subplot(ax[0, 0])
    ax1.set_xticks(ind + width / 2, labels=[d.name for d in dirs])
    ax1.bar(
        ind,
        train_db_sizes_plot,
        width=width,
        label=title + '_train',
        color=COLORS,
    )
    ax1.bar(
        ind + width,
        seed_gen_db_sizes_plot,
        width=width,
        label=title + '_seed',
        color=run_color_seed,
    )

    # Model performance
    # ax2 = ax.figure.add_subplot(ax[0, 1])
    inner_grid_1 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=ax[0, 1], hspace=0.5
    )
    ax2_top = ax.figure.add_subplot(inner_grid_1[0, 0])
    ax2_bottom = ax.figure.add_subplot(inner_grid_1[1, 0])

    ax2_top.set_xticks(ind, labels=[d.name for d in dirs])
    ax2_bottom.set_xticks(ind, labels=[d.name for d in dirs])

    ax2_top.bar(ind, mace_e_plot, width=width, label=title + '_E', color=clr_short)
    ax2_bottom.bar(ind, mace_f_plot, width=width, label=title + '_F', color=clr_short)

    # MAE and RMSD
    inner_grid_2 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=ax[1, 0], hspace=0.5
    )
    ax3_top = ax.figure.add_subplot(inner_grid_2[0, 0])
    ax3_bottom = ax.figure.add_subplot(inner_grid_2[1, 0])
    ax3_top.set_xticks(ind, labels=[d.name for d in dirs])
    ax3_bottom.set_xticks(ind, labels=[d.name for d in dirs])

    ax3_top_twin = ax3_top.twinx()
    ax3_bottom_twin = ax3_bottom.twinx()
    ax3_top.scatter(
        ind, E_rmsd_plot, label='E RMSD', marker='o', c=clr_short, alpha=0.5
    )
    ax3_top_twin.scatter(ind, E_mae_plot, label='E MAE', marker='x', c=clr_short)
    ax3_bottom.scatter(
        ind, F_rmsd_plot, label='F RMSD', marker='o', c=clr_short, alpha=0.5
    )
    ax3_bottom_twin.scatter(ind, F_mae_plot, label='F MAE', marker='x', c=clr_short)

    # TODO: Prepare panel 4 with autoencoder
    # ax4 = ax.figure.add_subplot(ax[1, 1])

    ax1.set_xlabel('AL Loop Run')
    ax1.set_ylabel('Number of structures')
    ax1.set_title('Seed and Train Database per Run')
    # ax1.legend()

    # ax2_top.set_xlabel('AL Loop Run')
    ax2_top.set_ylabel('RMSE Energy [meV/atom]')
    ax2_top.set_title('Model Performance')
    ax2_bottom.set_xlabel('AL Loop Run')
    ax2_bottom.set_ylabel('RMSE Forces [meV/A]')
    ax2_bottom.set_title('Model Performance')

    # ax3_top.set_xlabel('AL Loop Run')
    ax3_top.set_ylabel('RMSE E [meV/atom]')
    ax3_top.set_title('Energy Metrics per Run')
    ax3_bottom.set_xlabel('AL Loop Run')
    ax3_bottom.set_ylabel('RMSE F [meV/A]')
    ax3_bottom.set_title('Force Metrics per Run')
    ax3_top_twin.set_ylabel('MAE E [meV/atom]')
    ax3_bottom_twin.set_ylabel('MAE F [meV/A]')

    plt.tight_layout()
    fig.subplots_adjust(top=0.925, bottom=0.0725, right=0.95, left=0.075)

    # Saving both png and svg
    plt.savefig(filename.with_suffix('.png'), dpi=300)
    plt.savefig(filename.with_suffix('.svg'), dpi=300)
    custom_print(f"Saved report to '{filename.with_suffix('.png')}'.", 'info')
    custom_print('Report generation complete.', 'done')


def gen_init_db_report(
    train_db_path,
    threshold_E: float = None,
    threshold_F: float = None,
    remove_outliers: bool = False,
    color_type: str = None,
    per_atom: bool = False,
):
    init_logger(source='gen_init_db_report')
    custom_print('Generating initial database report...', 'info')
    train_db = ase_read(train_db_path, format='extxyz', index=':')

    E_dft_list = np.array([atoms.info['REF_energy'] for atoms in train_db])
    F_dft_list_max = np.array(
        [simplify_forces_struct(atoms.arrays['REF_forces'])[0] for atoms in train_db]
    )
    F_dft_list_avg = np.array(
        [simplify_forces_struct(atoms.arrays['REF_forces'])[1] for atoms in train_db]
    )

    # Set standard units
    E_unit = 'eV'
    F_unit = 'eV/A'

    # Apply per atom scaling
    if per_atom:
        for idx, _ in enumerate(train_db):
            E_dft_list[idx] /= len(train_db[idx])
            F_dft_list_max[idx] /= len(train_db[idx])
            F_dft_list_avg[idx] /= len(train_db[idx])

        # Update units
        E_unit = 'eV/atom'
        F_unit = 'eV/atom'

    # Create masks (True for values that should be removed)
    E_mask = np.full(len(E_dft_list), False)
    F_mask = np.full(len(E_dft_list), False)
    if threshold_E:
        E_mask = E_dft_list > float(threshold_E)
    if threshold_F:
        F_mask = np.abs(F_dft_list_max) < float(threshold_F)

    # Combine masks
    combined_mask = E_mask | F_mask

    if np.any(combined_mask) and remove_outliers:
        # Apply combined_mask to E_dft_list and F_dft_list
        indices = np.array(range(len(E_dft_list)))[~combined_mask]
        E_dft_list = E_dft_list[~combined_mask]
        F_dft_list_max = F_dft_list_max[~combined_mask]
        F_dft_list_avg = F_dft_list_avg[~combined_mask]

        valid_idxs = np.array(range(len(train_db)))
        structs = [train_db[struct_idx] for struct_idx in valid_idxs[~combined_mask]]

        custom_print(
            f'Filtered {len(train_db) - len(structs)} structures based on thresholds.',
            'warn',
        )
        ase_write('filtered_structs.xyz', structs, format='extxyz')
        custom_print(
            f'Saved {len(structs)} filtered structures to "filtered_structs.xyz".',
            'info',
        )
    else:
        indices = list(range(len(E_dft_list)))

    color_dict = {'bulk': '#cc241d', 'surface': '#158588', 'cluster': '#28cc10'}

    formulas = [struct.get_chemical_formula() for struct in train_db]
    phases = [struct.info.get('phase', 'unknown') for struct in train_db]
    struct_type = [struct.info.get('mdb_struct_type', 'unknown') for struct in train_db]
    uuids = [struct.info.get('mdb_id', 'unknown') for struct in train_db]

    energy_line_color = 'rgba(177,98,134, 0.25)'
    forces_max_line_color = 'rgba(25, 95, 180, 0.50)'
    forces_avg_line_color = 'rgba(80, 180, 100, 0.50)'

    if color_type == 'struct_type':
        type_color = [color_dict.get(struct, '#282828') for struct in struct_type]
        E_type_color, F_max_type_color, F_avg_type_color = (
            type_color,
            type_color,
            type_color,
        )
    elif color_type == 'phase':
        type_color = [color_dict.get(struct, '#282828') for struct in phases]
        E_type_color, F_max_type_color, F_avg_type_color = (
            type_color,
            type_color,
            type_color,
        )
    else:
        type_color = None
        E_type_color = energy_line_color
        F_max_type_color = forces_max_line_color
        F_avg_type_color = forces_avg_line_color

    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('DFT Energy', 'DFT Forces'),
    )

    # Energy chart
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=E_dft_list,
            name='DFT Energy',
            hoverinfo='x+y',
            mode='lines+markers',
            marker=dict(size=3, color=E_type_color),
            line=dict(
                width=0.85,
                color=energy_line_color,
            ),
            customdata=np.stack((formulas, phases, struct_type, uuids), axis=-1),
            hovertemplate="""
            <b>Energy</b><br><br>
            <b>Structure:</b> %{x}<br>
            <b>ID:</b> %{customdata[3]}<br>
            <b>Energy:</b> %{y} eV<br>
            <b>Formula:</b> %{customdata[0]}<br>
            <b>Phase:</b> %{customdata[1]}<br>
            <b>Struct type:</b> %{customdata[2]}<br>
            <extra></extra>
            """.replace('eV', E_unit),
        ),
        row=1,
        col=1,
    )

    # Forces chart
    forces_hovertemplate = """
            <b>NAME</b><br><br>
            <b>Structure:</b> %{x}<br>
            <b>ID:</b> %{customdata[3]}<br>
            <b>Force:</b> %{y} eV<br>
            <b>Formula:</b> %{customdata[0]}<br>
            <b>Phase:</b> %{customdata[1]}<br>
            <b>Struct type:</b> %{customdata[2]}<br>
            <extra></extra>
        """.replace('eV', F_unit)
    curr_trace: str = 'DFT Forces Max'
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=F_dft_list_max,
            name=curr_trace,
            # marker_color='#83a598',
            hoverinfo='x+y',
            mode='lines+markers',
            marker=dict(size=2, color=F_max_type_color),
            line=dict(width=0.85, color=forces_max_line_color),
            customdata=np.stack((formulas, phases, struct_type, uuids), axis=-1),
            hovertemplate=forces_hovertemplate.replace('NAME', curr_trace),
        ),
        row=2,
        col=1,
    )
    curr_trace: str = 'DFT Forces Average'
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=F_dft_list_avg,
            name=curr_trace,
            hoverinfo='x+y',
            mode='lines+markers',
            marker=dict(size=2, color=F_avg_type_color),
            line=dict(
                width=0.85,
                color=forces_avg_line_color,
            ),
            customdata=np.stack((formulas, phases, struct_type, uuids), axis=-1),
            hovertemplate=forces_hovertemplate.replace('NAME', curr_trace),
        ),
        row=2,
        col=1,
    )

    if threshold_E:
        fig.add_shape(
            type='line',
            x0=0,
            x1=max(indices),
            y0=threshold_E,
            y1=threshold_E,
            line=dict(color='#282828', width=2, dash='dash'),
            yref='y1',
        )
        fig.add_annotation(text='E threshold', x=0, y=threshold_E, yref='y1')
    if threshold_F:
        fig.add_shape(
            type='line',
            x0=0,
            x1=max(indices),
            y0=threshold_F,
            y1=threshold_F,
            line=dict(color='#282828', width=2, dash='dash'),
            yref='y2',
        )
        fig.add_annotation(text='F threshold', x=0, y=threshold_F, yref='y2')

    yaxis_label = f'Energy [{E_unit}]'
    yaxis2_label = f'Forces [{F_unit}]'

    # Customize layout
    fig.update_layout(
        title='DFT Energy and Forces per Structure',
        yaxis=dict(autorange='reversed', title=yaxis_label),
        yaxis2=dict(title=yaxis2_label),
        xaxis2=dict(title='Structure Index'),
        template='simple_white',
    )

    plt.tight_layout()
    timestamp = int(time.time())
    filename = Path(f'al_loop_report_{timestamp}.html').resolve()
    fig.write_html(filename)

    print()
    custom_print(f"Saved interactive report to '{filename}'.", 'info')

    plt.clf()


def plot_al_loop_report(
    limit_num_steps: bool,
    seed_gen_db_sizes: list[int],
    train_db_sizes: list[int],
    mace_e: list[float],
    mace_f: list[float],
    it_idx: list[int],
    ax,
    model_acc_multiplier: float = None,
):
    bar_line_color = '#282828'
    bar_line_width = 1.5
    # Get unix timestamp for filename
    timestamp = int(time.time())
    filename = Path(f'al_loop_report_{timestamp}').resolve()

    # Iterate in reverse over train_db_sizes and seed_gen_db_sizes,
    # if there are None valiues, delete that index and the corresponding
    # index in ind
    ind = it_idx.copy()
    for idx, seed, train in zip(ind, seed_gen_db_sizes, train_db_sizes, strict=False):
        if seed is None or train is None:
            seed_gen_db_sizes.remove(seed)
            train_db_sizes.remove(train)
            ind.remove(idx)
    ind = np.array(ind)

    # If the number of steps in the AL loop is greater than limit_num_steps,
    # sample steps to stay below that number, always including the first
    # and last steps
    if limit_num_steps:
        # Sample steps equispaced
        ind = np.linspace(0, ind[-1], num=limit_num_steps, dtype=int).tolist()
        seed_gen_db_sizes = [seed_gen_db_sizes[i] for i in ind]
        train_db_sizes = [train_db_sizes[i] for i in ind]

    # Plot seed and train db sizes as a stacked bar chart over every iteration
    ax1 = ax.figure.add_subplot(ax[0, 0])

    # x_axis data should use a range of the length of the data
    # so it looks equispaced. The labels are set to the actual
    # iteration index later.
    top_xaxis = np.arange(len(ind))

    # Get the width as a function of the x axis data length
    width = 0.3 + (1 / (len(ind) * 0.75))

    ax1.bar(
        top_xaxis,
        train_db_sizes,
        width=width,
        label='train_db',
        color=COLORS[0],
        edgecolor=bar_line_color,
        linewidth=bar_line_width,
    )
    ax1.bar(
        top_xaxis + width,
        seed_gen_db_sizes,
        width=width,
        label='seed_gen_db',
        color=COLORS[1],
        edgecolor=bar_line_color,
        linewidth=bar_line_width,
    )
    ax1.set_xticks(top_xaxis + width / 2, labels=ind)
    ax1.set_xlabel('AL Loop Step')
    ax1.set_ylabel('Number of structures')
    ax1.legend()
    ax1.set_title('Seed and Train Database Evolution')

    ax1.annotate(
        'a)',
        xy=(0, 1),
        xycoords='axes fraction',
        xytext=(+0.5, -0.5),
        textcoords='offset fontsize',
        fontsize='medium',
        verticalalignment='top',
        bbox=dict(facecolor='1', edgecolor='none', pad=3.0),
    )

    # Add text labels to top left figure bars
    for idx, seed, train in zip(
        top_xaxis, seed_gen_db_sizes, train_db_sizes, strict=False
    ):
        ax1.text(
            idx,
            train / 2,
            train,
            ha='center',
            va='bottom',
            rotation=90,
            fontweight=700,
        )
        ax1.text(
            idx + width,
            seed / 2,
            seed,
            ha='center',
            va='bottom',
            rotation=90,
            fontweight=700,
        )

    ax2 = ax.figure.add_subplot(ax[0, 1])

    ax2.annotate(
        'b)',
        xy=(0, 1),
        xycoords='axes fraction',
        xytext=(+0.5, -0.5),
        textcoords='offset fontsize',
        fontsize='medium',
        verticalalignment='top',
        bbox=dict(facecolor='1', edgecolor='none', pad=3.0),
    )

    # Plot seed size delta as a bar chart over every iteration
    seed_gen_db_diff, train_db_diff = [], []
    for idx, seed, train in zip(
        it_idx, seed_gen_db_sizes, train_db_sizes, strict=False
    ):
        idx = it_idx.index(idx)

        if idx == 0:
            seed_gen_db_diff.append(0)
            train_db_diff.append(0)
        else:
            seed_gen_db_diff.append(seed - seed_gen_db_sizes[idx - 1])
            train_db_diff.append(train - train_db_sizes[idx - 1])

    # Add text labels to bars
    for idx, seed, train in zip(
        top_xaxis, seed_gen_db_diff, train_db_diff, strict=False
    ):
        if idx == 0:
            continue

        # Add sign
        seed_txt = f'{seed}' if seed < 0 else f'+{seed}'
        train_txt = f'{train}' if train < 0 else f'+{train}'

        ax2.text(
            idx,
            train / 2,
            train_txt,
            ha='center',
            va='bottom',
            rotation=90,
            fontweight=700,
        )
        ax2.text(
            idx + width,
            seed / 2,
            seed_txt,
            ha='center',
            va='bottom',
            rotation=90,
            fontweight=700,
        )

    ax2.bar(
        top_xaxis,
        train_db_diff,
        width=width,
        label='train_db',
        color=COLORS[0],
        edgecolor=bar_line_color,
        linewidth=bar_line_width,
    )
    ax2.bar(
        top_xaxis + width,
        seed_gen_db_diff,
        width=width,
        label='seed_gen_db',
        color=COLORS[1],
        edgecolor=bar_line_color,
        linewidth=bar_line_width,
    )
    ax2.axhline(y=0, color=LINE_COLOR, linestyle='--')
    ax2.set_xticks(top_xaxis + width / 2, ind)
    ax2.set_xlabel('AL Loop Step')
    ax2.set_ylabel(r'$\Delta$ Number of structures')
    ax2.set_title('Structure count change over iteration (accumul.)')
    ax2.legend()

    # Plot MACE model energy performance
    ax3 = ax.figure.add_subplot(ax[1, 0])

    # Iterate in reverse over train_db_sizes and seed_gen_db_sizes,
    # if there are None valiues, delete that index and the corresponding
    # index in ind
    ind_short = it_idx.copy()

    # This also implies mace_e and mace_f are non-empty if they have the same length
    if ind_short:
        for i in range(len(ind_short) - 1, -1, -1):
            # Access elements by index i from mace_e and mace_f
            if mace_e[i] is None or mace_f[i] is None:
                # Remove elements at index `i` from each list.
                # Using .pop(i) or del list[i] is safe when iterating backwards.
                mace_e.pop(i)
                mace_f.pop(i)
                ind_short.pop(i)

    # If the number of steps in the AL loop is greater than limit_num_steps,
    # sample steps to stay below that number, always including the first
    # and last steps
    if limit_num_steps:
        # Sample steps equispaced
        ind_short = np.linspace(
            0, ind_short[-1], num=limit_num_steps, dtype=int
        ).tolist()
        mace_e = [mace_e[i] for i in range(len(ind_short))]
        mace_f = [mace_f[i] for i in range(len(ind_short))]

    # Convert lists to numpy arrays
    ind_short = np.array(ind_short)
    mace_e = np.array(mace_e)
    mace_f = np.array(mace_f)

    # x_axis data should use a range of the length of the data
    # so it looks equispaced. The labels are set to the actual
    # iteration index later.
    bottom_xaxis = np.arange(start=1, stop=len(ind_short) + 1)
    breakpoint()

    ax3.plot(bottom_xaxis, mace_e, label='MACE Energy', color=COLORS[2], marker='o')

    if model_acc_multiplier:
        thresh_color = COLORS[-1]
        ax3.plot(
            bottom_xaxis,
            mace_e * model_acc_multiplier,
            label='Energy threshold',
            color=COLORS[-1],
            marker='^',
        )
        ax3.set_ylabel('Energy threshold [meV]')

    ax3.set_xticks(bottom_xaxis, labels=ind_short)
    ax3.set_xlabel('AL Loop Step')
    ax3.set_ylabel('RMSE E per atom [meV]')
    ax3.set_title('Evolution of best MACE Model Energy RMSE')
    ax3.legend()

    ax3.annotate(
        'c)',
        xy=(0, 1),
        xycoords='axes fraction',
        xytext=(+0.5, -0.5),
        textcoords='offset fontsize',
        fontsize='medium',
        verticalalignment='top',
        bbox=dict(facecolor='1', edgecolor='none', pad=3.0),
    )

    # Add a horizontal line to mark chemical accuracy for energy and forces
    chem_acc = 43.37  # meV
    ax3.axhline(y=chem_acc, color=LINE_COLOR, linestyle='--')
    ax3.text(x=1.5, y=chem_acc + 0.5, s='Chem. Acc.', color=LINE_COLOR)

    # Plot MACE model force performance
    ax4 = ax.figure.add_subplot(ax[1, 1])
    ax4.plot(bottom_xaxis, mace_f, label='MACE Forces', color=COLORS[3], marker='o')

    if model_acc_multiplier:
        thresh_color = COLORS[-1]
        ax4.plot(
            bottom_xaxis,
            mace_f * model_acc_multiplier,
            label='Forces threshold',
            color=thresh_color,
            marker='^',
        )

    ax4.set_xticks(bottom_xaxis, labels=ind_short)
    ax4.set_xlabel('AL Loop Step')
    ax4.set_ylabel('RMSE F [meV / A]')
    ax4.set_title('Evolution of best MACE Model Force RMSE')
    ax4.legend()

    ax4.annotate(
        'd)',
        xy=(0, 1),
        xycoords='axes fraction',
        xytext=(+0.5, -0.5),
        textcoords='offset fontsize',
        fontsize='medium',
        verticalalignment='top',
        bbox=dict(facecolor='1', edgecolor='none', pad=3.0),
    )

    return filename


def generate_latent_space_evol(
    al_loop_node: orm.Node,
    device_str: str,
    ax,
    database_path: str,
    model_path: str,
    autoencoder_path: str,
    autoencoder_model: str,
    databases: list[str],
):
    # all_steps = [
    #     stp
    #     for stp in al_loop_node.called
    #     if stp.process_type == 'aiida.workflows:mdb-active-learning'
    #     or stp.process_type == 'aiida.workflows:mdb-simple-active-learning'
    # ]

    # auto_steps = []

    # for node in all_steps:
    #     for substep in node.called:
    #         if substep.process_type == 'aiida.calculations:mdb-descriptors-combined':
    #             auto_steps.append(substep)

    # if len(auto_steps) == 0:
    #     return

    # # Get autoencoder for first step
    # main_auto_model: orm.SinglefileData = auto_steps[0].outputs.autoencoder_model

    # for step in auto_steps:
    #     ...
    #     # TODO: Database is not stored in any of the aiida nodes

    autoencoder_file_path = Path(autoencoder_model)
    autoencoder = None
    descr_dict = None

    # Load pickle file
    if Path('./latent_spaces.pkl').exists():
        with open('latent_spaces.pkl', 'rb') as f:
            latent_spaces = pickle.load(f)
    else:
        latent_spaces = []

    # Get number of steps saved
    num_steps_saved = len(latent_spaces)

    # Computing latent space for all structures
    if num_steps_saved == 0:
        for step_idx, database in enumerate(databases):
            custom_print(
                f"Getting latent space for the database at step '{step_idx}'...",
                'info',
            )

            curr_db = ase_read(
                Path(autoencoder_path) / database, format='extxyz', index=':'
            )
            descr_dict, arr = generate_descriptors(
                model_path=model_path,
                database=curr_db,
                device=device_str,
                descriptor_dict=descr_dict,
            )

            if not autoencoder:
                autoencoder = load_autoencoder_model(
                    model_path=autoencoder_file_path,
                    data_arr=arr,
                )

            latent_space_dict = get_latent_space_autoencoder(
                model=autoencoder,
                descriptor_dict=descr_dict,
                device=device_str,
                quiet=True,
            )
            latent_spaces.append(latent_space_dict)

        # Saving latent spaces to pickle file
        if len(latent_spaces) > 0:
            with open('latent_spaces.pkl', 'wb') as f:
                pickle.dump(latent_spaces, file=f)

    # Getting colormap for step number
    cmap = colormaps.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num_steps_saved))

    # Plotting latent space
    ax1 = ax.figure.add_subplot(ax[2, :])
    for idx, latent_dict in enumerate(latent_spaces):
        latent_space_vals = []

        for structure in latent_dict:
            curr_struct_latent = latent_dict[structure]['latent_space'][0]
            latent_space_vals.append(curr_struct_latent)
        latent_space_vals = np.vstack(latent_space_vals)

        # Get concave hull if step is 0
        if idx == 0:
            concave_hull = get_concave_hull_julia(latent_space_vals)
            ax1.plot(
                concave_hull[:, 0],
                concave_hull[:, 1],
                color='#fb4934',
                linestyle='solid',
                zorder=10,
                label='Concave Hull',
                linewidth=1,
            )

        ax1.set_title('Latent Space Evolution')
        ax1.set_xlabel('Reduced dimension 1')
        ax1.set_ylabel('Reduced dimension 2')
        ax1.set_title('Latent Space Evolution')
        ax1.scatter(
            x=latent_space_vals[:, 0],
            y=latent_space_vals[:, 1],
            color=colors[idx],
            alpha=0.1,
            # label=f'Step {idx}',
            s=2,
            edgecolors='#282828',
            linewidth=0.25,
        )
        ax1.legend()

    # Adding colorbar once all iterations are plotted
    plt.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(1, num_steps_saved),
            cmap=cmap,
        ),
        label='AL Loop Step',
        pad=0.01,
        ax=ax1,
    )


def get_al_loop_performance(al_loop_pk: int | list[int]) -> dict:
    # Get all the directly called AL Step WorkChains
    al_steps = []

    for pk in al_loop_pk:
        base_node = orm.load_node(pk)
        custom_print(f'Loaded base node: {base_node.process_label}<{base_node.pk}>')
        curr_loop_al_steps = [
            node
            for node in base_node.called
            if isinstance(node, orm.WorkChainNode)
            and node.process_label == AL_STEP_WORKCHAIN_LABEL
            and node.is_finished_ok
        ]
        al_steps.extend(curr_loop_al_steps)

    custom_print(f"Found {len(al_steps)} '{AL_STEP_WORKCHAIN_LABEL}' steps.")

    if not al_steps:
        # Maybe the structure is different, try descendants?
        custom_print(
            f"No direct children matched '{AL_STEP_WORKCHAIN_LABEL}'."
            ' Checking descendants...'
        )
        al_steps = [
            node
            for node in base_node.called_descendants
            if isinstance(node, orm.WorkChainNode)
            and node.process_label == AL_STEP_WORKCHAIN_LABEL
            and node.is_finished_ok
        ]
        custom_print(f'Found {len(al_steps)} matching descendant steps.')

    if not al_steps:
        custom_print(
            f"Error: Could not find any AL steps ('{AL_STEP_WORKCHAIN_LABEL}') called"
            f' by {base_node.pk}.',
            'error',
        )
        exit()

    # Get performance information about the AL Steps
    all_performance_data = []
    for i, al_step in enumerate(al_steps):
        custom_print(
            f'Analyzing AL Step {i + 1}/{len(al_steps)} (PK={al_step.pk})', 'debug'
        )
        performance_dict = get_al_step_performance(al_step)
        if performance_dict:
            # pprint(performance_dict)
            all_performance_data.append(performance_dict)
        else:
            custom_print(f'Could not analyze step {al_step.pk}.')

    return all_performance_data


def get_ncores_from_calcjob(calcjob: orm.CalcJobNode) -> int:
    """
    Get the number of cores used by a CalcJob.

    Parameters
    ----------
    calcjob : orm.CalcJobNode
        The CalcJob node to analyze.

    Returns
    -------
    int
        Number of cores used by the CalcJob.
    """
    stdout, scheduler = gather_stdout_and_scheduler(calcjob)

    if stdout is None or scheduler is None:
        return 0

    if 'slurm' in scheduler:
        header: list[str] = stdout.splitlines()[0].split('|')
        n_cpu_posc: int = header.index('ReqTRES')
        content: list[str] = stdout.splitlines()[1].split('|')
        req_cpu_data = content[n_cpu_posc]
        n_cpus = int(req_cpu_data.split(',')[1].split('=')[1])
    elif 'sge' in scheduler:
        for entry in stdout.splitlines():
            if 'slots ' in entry:
                n_cpus = int(entry.split()[1])
                break
    else:
        custom_print(
            f'Unrecognized scheduler for CalcJob {calcjob.pk}: {scheduler}', 'error'
        )
        n_cpus = 0
    return n_cpus


def gather_stdout_and_scheduler(calcjob: orm.CalcJobNode) -> tuple[str, str]:
    try:
        scheduler = str(calcjob.computer.get_scheduler()).lower()

        # Get job information
        stdout = calcjob.get_detailed_job_info().get('stdout')

    except Exception as e:
        custom_print(f'Error getting scheduler for CalcJob {calcjob.pk}: {e}', 'error')
        stdout = None
        scheduler = None

    return stdout, scheduler


def get_runtime_from_calcjob(calcjob: orm.CalcJobNode) -> datetime.timedelta:
    stdout, scheduler = gather_stdout_and_scheduler(calcjob)

    if stdout is None or scheduler is None:
        return 0

    if 'slurm' in scheduler:
        header: list[str] = stdout.splitlines()[0].split('|')
        n_cpu_posc: int = header.index('CPUTimeRAW')
        content: list[str] = stdout.splitlines()[1].split('|')
        cpu_time = datetime.timedelta(seconds=int(content[n_cpu_posc]))
    elif 'sge' in scheduler:
        for entry in stdout.splitlines():
            if 'wallclock ' in entry:
                cpu_time = datetime.timedelta(seconds=int(entry.split()[1]))
                break
    else:
        custom_print(
            f'Unrecognized scheduler for CalcJob {calcjob.pk}: {scheduler}', 'error'
        )
        cpu_time = 0
    return cpu_time


def get_al_step_performance(al_step: orm.WorkChainNode) -> dict:
    """
    Get performance information about an AL Step WorkChain.

    Calculates the total duration of the step and the elapsed time
    spent within each major stage (Training, Descriptors, MD, DFT)
    by finding the time between the start of the first calculation
    of that type and the end of the last calculation of that type
    within the step. Also collects individual calculation durations.

    :param al_step: The AL Step WorkChain node.
    :return: A dictionary with performance information.
    """
    if not isinstance(al_step, orm.WorkChainNode):
        custom_print(f'Warning: Node {al_step.pk} is not a WorkChainNode. Skipping.')
        return None
    if al_step.process_label != AL_STEP_WORKCHAIN_LABEL:
        custom_print(
            f"Warning: Node {al_step.pk} has label '{al_step.process_label}',"
            f" expected '{AL_STEP_WORKCHAIN_LABEL}'. Skipping."
        )
        return None

    ctime = al_step.ctime
    mtime = al_step.mtime
    step_duration = mtime - ctime if mtime and ctime else datetime.timedelta(0)

    # Initialize dictionaries to store times for each stage
    stage_stats = {
        'training': {'ctimes': [], 'mtimes': [], 'durations': [], 'cores': []},
        'descriptors': {'ctimes': [], 'mtimes': [], 'durations': [], 'cores': []},
        'md': {'ctimes': [], 'mtimes': [], 'durations': [], 'cores': []},
        'dft': {'ctimes': [], 'mtimes': [], 'durations': [], 'cores': []},
    }

    # Get all descendant CalcJob nodes
    # called_descendants includes CalcJobs run by sub-workchains within this step
    descendant_calcjobs = [
        node for node in al_step.called_descendants if isinstance(node, orm.CalcJobNode)
    ]

    # Categorize calculations and collect their times
    for job in descendant_calcjobs:
        label = job.process_label
        job_ctime = job.ctime
        job_mtime = job.mtime
        job_duration = get_runtime_from_calcjob(job)

        # Get the number of cores used
        n_cores = get_ncores_from_calcjob(job)

        if label == TRAINING_LABEL:
            stage_stats['training']['ctimes'].append(job_ctime)
            stage_stats['training']['mtimes'].append(job_mtime)
            stage_stats['training']['durations'].append(job_duration)
            stage_stats['training']['cores'] = n_cores
        elif label == DESCRIPTORS_LABEL:
            stage_stats['descriptors']['ctimes'].append(job_ctime)
            stage_stats['descriptors']['mtimes'].append(job_mtime)
            stage_stats['descriptors']['durations'].append(job_duration)
            stage_stats['descriptors']['cores'] = n_cores
        elif label == MD_LABEL:
            stage_stats['md']['ctimes'].append(job_ctime)
            stage_stats['md']['mtimes'].append(job_mtime)
            stage_stats['md']['durations'].append(job_duration)
            stage_stats['md']['cores'] = n_cores
        elif label == DFT_LABEL:
            stage_stats['dft']['ctimes'].append(job_ctime)
            stage_stats['dft']['mtimes'].append(job_mtime)
            stage_stats['dft']['durations'].append(job_duration)
            stage_stats['dft']['cores'] = n_cores

    # Calculate total elapsed time for each stage
    performance_info = {
        'step': {
            'pk': al_step.pk,
            'ctime': ctime,
            'mtime': mtime,
            'step_duration': step_duration,
            'state': al_step.process_state.value,
            'exit_status': al_step.exit_status,
        },
        'stages': {
            'training': {
                'duration_list': [],
                'total_elapsed_time': datetime.timedelta(0),
            },
            'descriptors': {
                'duration_list': [],
                'total_elapsed_time': datetime.timedelta(0),
            },
            'md': {'duration_list': [], 'total_elapsed_time': datetime.timedelta(0)},
            'dft': {'duration_list': [], 'total_elapsed_time': datetime.timedelta(0)},
        },
        'counts': {  # Add counts for clarity
            'training': 0,
            'descriptors': 0,
            'md': 0,
            'dft': 0,
        },
    }

    for stage_name, stats in stage_stats.items():
        # Check if any calculations of this type were run
        if stats['ctimes']:
            min_ctime = min(stats['ctimes'])
            max_mtime = max(stats['mtimes'])
            total_elapsed_time = max_mtime - min_ctime
            performance_info['stages'][stage_name]['total_elapsed_time'] = (
                total_elapsed_time
            )
            performance_info['stages'][stage_name]['duration_list'] = stats['durations']
            performance_info['stages'][stage_name]['cores'] = stats['cores']
            performance_info['counts'][stage_name] = len(stats['durations'])
        else:
            performance_info['counts'][stage_name] = 0
            performance_info['stages'][stage_name] = {}
            performance_info['stages'][stage_name]['duration_list'] = []
            performance_info['stages'][stage_name]['cores'] = 0
            performance_info['stages'][stage_name]['total_elapsed_time'] = (
                datetime.timedelta(0)
            )
    return performance_info


def plot_performance_stacked_bar(
    all_performance_data: list[dict], output_filename: str = 'al_loop_performance.png'
):
    """
    Generates a stacked bar chart of stage elapsed times per AL step.

    Parameters
    ----------
    all_performance_data : List[Dict]
        A list containing performance dictionaries for each AL step.
    output_filename : Optional[str], optional
        The filename to save the plot. If None, the plot is displayed
        instead of being saved. Defaults to "al_loop_performance.png".
    """
    if not all_performance_data:
        custom_print('No performance data available to plot.', 'warning')
        return

    num_steps = len(all_performance_data)
    step_indices = np.arange(num_steps) + 1
    step_pks = [
        step['step']['pk'] for step in all_performance_data
    ]  # For x-axis labels

    # Prepare data for stacking (convert timedelta to seconds)
    stage_data_seconds = {key: np.zeros(num_steps) for key in STAGE_KEYS}
    stage_data_core_h = {key: np.zeros(num_steps) for key in STAGE_KEYS}
    for i, step_data in enumerate(all_performance_data):
        for key in STAGE_KEYS:
            n_cores = step_data['stages'][key]['cores']
            elapsed_time = step_data['stages'][key]['total_elapsed_time']
            stage_data_seconds[key][i] = elapsed_time.total_seconds()
            stage_data_core_h[key][i] = (elapsed_time.total_seconds() / 3600) * n_cores

    # Define colors for stages (adjust as needed)
    colors = {
        'training': '#ffa6c8',
        'descriptors': '#ff8834',
        'md': '#6aa1f4',
        'dft': '#3ed04e',
    }
    fig, ax = plt.subplots(
        nrows=2, figsize=(max(10, num_steps * 0.6), 6)
    )  # Adjust figure size dynamically

    bottoms = np.zeros(num_steps)
    bottoms_core = np.zeros(num_steps)
    for key in STAGE_KEYS:
        ax[0].bar(
            step_indices,
            stage_data_seconds[key] / 3600,  # Convert seconds to hours
            bottom=bottoms,
            label=key,
            color=colors.get(key),  # Use defined color or default
            edgecolor='#282828',
            linewidth=1,
        )
        bottoms += (
            stage_data_seconds[key] / 3600
        )  # Add current stage height to bottom for next stack
        ax[1].bar(
            step_indices,
            stage_data_core_h[key],  # Convert seconds to hours
            bottom=bottoms_core,
            label=key,
            color=colors.get(key),  # Use defined color or default
            edgecolor='#282828',
            linewidth=1,
        )
        bottoms_core += stage_data_core_h[key]

    ax[0].set_ylabel('Elapsed Time (hours)')
    ax[0].set_xlabel('AL Step Index')
    ax[0].set_title('AL Loop Performance: Elapsed Time per Stage')
    ax[0].set_xticks(step_indices)

    # Use PKs as labels, rotate if many steps
    ax[0].set_xticklabels(
        [f'{i + 1}' for i, pk in enumerate(step_pks)],
        rotation=45 if num_steps > 15 else 0,
        ha='center',
    )
    ax[0].legend(title='Stages')
    ax[0].grid(axis='y', linestyle='--', alpha=0.3)

    # Set y-limit for better visibility
    ax[0].set_ylim(0, max(bottoms) * 1.1)

    ax[1].set_ylabel(r'Resource consumption ($\mathrm{core} \cdot \mathrm{h}$)')
    ax[1].set_xlabel('AL Step Index')
    ax[1].set_title('AL Loop Performance: Resource consumption per Stage')
    ax[1].set_xticks(step_indices)
    ax[1].set_xticklabels(
        [f'{i + 1}' for i, pk in enumerate(step_pks)],
        rotation=45 if num_steps > 15 else 0,
        ha='center',
    )
    ax[1].legend(title='Stages')
    ax[1].grid(axis='y', linestyle='--', alpha=0.3)

    # Set y-limit for better visibility
    ax[1].set_ylim(0, max(bottoms_core) * 1.1)

    # Adjust layout to prevent labels overlapping
    fig.tight_layout()

    if output_filename:
        output_path = Path(output_filename)
        try:
            plt.savefig(output_path, dpi=300)
            svg_path = Path(output_path.stem).with_suffix('.svg')
            plt.savefig(svg_path, dpi=300, format='svg')
            custom_print(f"Performance plot saved to '{output_path.resolve()}'")
        except Exception as e:
            custom_print(f'Error saving plot to {output_path.resolve()}: {e}', 'error')
    else:
        custom_print('Displaying performance plot...')
        plt.show()


def print_performance_report(all_performance_data):
    custom_print('Performance data for all AL steps:')
    total_md_time = datetime.timedelta(0)
    total_md_core_h = 0.0

    total_dft_time = datetime.timedelta(0)
    total_dft_core_h = 0.0

    total_training_time = datetime.timedelta(0)
    total_training_core_h = 0.0

    total_descriptors_time = datetime.timedelta(0)
    total_descriptors_core_h = 0.0

    final_time = datetime.timedelta(0)
    for step_idx, step in enumerate(all_performance_data):
        custom_print(f'Step {step_idx + 1} PK: {step["step"]["pk"]}:')
        custom_print(f'  Step duration: {step["step"]["step_duration"]}')

        training_time = step['stages']['training']['total_elapsed_time']
        training_core_h = (training_time.total_seconds() / 3600) * step['stages'][
            'training'
        ].get('cores', 0)

        descriptors_time = step['stages']['descriptors']['total_elapsed_time']
        descriptors_core_h = (descriptors_time.total_seconds() / 3600) * step['stages'][
            'descriptors'
        ].get('cores', 0)

        md_time = step['stages']['md']['total_elapsed_time']
        md_core_h = (md_time.total_seconds() / 3600) * step['stages']['md'].get(
            'cores', 0
        )

        dft_time = step['stages']['dft']['total_elapsed_time']
        dft_core_h = (dft_time.total_seconds() / 3600) * step['stages']['dft'].get(
            'cores', 0
        )

        custom_print(
            f'  Training stage - time: {training_time},'
            f' core hours: {training_core_h:.1f}'
        )
        custom_print(
            f'  Descriptors stage stats - time: {descriptors_time},'
            f' core hours: {descriptors_core_h:.1f}'
        )
        custom_print(f'  MD stage stats - time: {md_time}, core hours: {md_core_h:.1f}')
        custom_print(
            f'  DFT stage stats - time: {dft_time} core hours: {dft_core_h:.1f}'
        )

        total_md_time += md_time
        total_md_core_h += md_core_h

        total_dft_time += dft_time
        total_dft_core_h += dft_core_h

        total_training_time += training_time
        total_training_core_h += training_core_h

        total_descriptors_time += descriptors_time
        total_descriptors_core_h += descriptors_core_h

        final_time += step['step']['step_duration']
        print()

    custom_print('Timings for the entire AL Loop:')
    custom_print(
        f'  Training totals - time: {total_training_time},'
        f' core hours: {total_training_core_h:.1f}'
    )
    custom_print(
        f'  Descriptors totals - time: {total_descriptors_time},'
        f'core hours: {total_descriptors_core_h:.1f}'
    )
    custom_print(
        f'  MD totals - time: {total_md_time} s, core hours: {total_md_core_h:.1f}'
    )
    custom_print(
        f'  DFT totals - time: {total_dft_time}, core hours: {total_dft_core_h:.1f}'
    )
    print()
    custom_print(f'  Total loop duration: {final_time}')


def gen_performance_report(al_loop_pk: int | list[int], output_filename: str = None):
    init_logger(source='get_run_performance')

    try:
        load_profile()
        custom_print(f"AiiDA profile '{load_profile().name}' loaded.")
    except Exception as e:
        custom_print(f'Error loading AiiDA profile: {e}')
        exit()

    try:
        # Get the performance data
        all_performance_data: list[dict] = get_al_loop_performance(al_loop_pk)

    except orm.NotExistent as e:
        custom_print(f'Node with PK={al_loop_pk} does not exist.', 'error')
        custom_print(e)
    except Exception as e:
        custom_print(f'An unexpected error occurred: {e}', 'error')
        import traceback

        traceback.print_exc()

    print()

    # Print the performance data
    print_performance_report(all_performance_data)

    # Plot the performance data
    plot_performance_stacked_bar(
        all_performance_data,
        output_filename=output_filename,
    )
