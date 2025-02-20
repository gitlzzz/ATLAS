"""Utils for generating reports."""

import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from aiida import orm
from ase.io import read as ase_read
from ase.io import write as ase_write
from matplotlib import colormaps, gridspec
from plotly.subplots import make_subplots

from MatDBForge.active_learning.active_learning_utils import (
    generate_descriptors,
    simplify_forces_struct,
)
from MatDBForge.active_learning.extrapolation.autoencoder import (
    get_latent_space_autoencoder,
    load_autoencoder_model,
)
from MatDBForge.active_learning.extrapolation.concave_hull import get_concave_hull_julia
from MatDBForge.core.code_utils import custom_print, init_logger


def gen_al_loop_report(
    loop_id: int | str = None,
    log_path: str = None,
    get_error_plot: bool = False,
    device='cpu',
    model_path: str | Path = None,
    database_path: str | Path = None,
    threshold_meV: float = None,
    remove_outliers: bool = False,
    title: str = None,
    get_latent_space: bool = False,
    autoencoder_path: str = None,
):
    import re

    from aiida.cmdline.utils.common import get_workchain_report

    # Init logger
    init_logger(source='al_loop_report_gen')

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

    # Match all lines containing the seed_gen_db and train_db sizes
    seed_gen_db_sizes, train_db_sizes, it_idx = [], [], []
    db_lines_re = re.compile(r'Iteration \d\d?: seed_gen_db.*').findall(report)

    # Prepare a list of all seed_gen_db and train_db sizes from db_lines
    for line in db_lines_re:
        it_idx.append(int(line.split()[1].replace(':', '')))
        seed_gen_db_sizes.append(int(line.split()[3].replace(',', '')))
        train_db_sizes.append(int(line.split()[5].replace(',', '')))

    # Match all lines containing the M0 model performance
    mace_e, mace_f = [], []
    best_model_lines = re.compile(r'Best model of current step.*').findall(report)

    # Prepare a list of all mace models generated from lammps_lines
    for line in best_model_lines:
        mace_e.append(float(line.split()[12]))
        mace_f.append(float(line.split()[16]))

    # Get run name
    if not title:
        try:
            title_regex = r'Active Learning Inputs:\s*\n({.*?})\n'
            matches = [match for match in re.finditer(title_regex, report, re.DOTALL)]
            settings_dict_string = matches[0].group(1)
            import json

            settings_dict_string = (
                settings_dict_string.replace("'", '"')
                .replace('True', '"True"')
                .replace('False', '"False"')
                .replace('None', '"None"')
            )

            settings_dict = json.loads(settings_dict_string)
            title = settings_dict.get('run_name')
        except Exception:
            title = None

    if not title:
        title = f'{al_loop_node}'

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

    filename = plot_al_loop_report(
        ini_db_size=ini_db_size,
        seed_gen_db_sizes=seed_gen_db_sizes,
        train_db_sizes=train_db_sizes,
        mace_e=mace_e,
        mace_f=mace_f,
        it_idx=it_idx,
        ax=gs,
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

        generate_error_plot(
            al_loop_node=al_loop_node,
            device_str=device,
            ax=gs,
            database_path=database_path,
            model_path=model_path,
            threshold_meV=threshold_meV,
            remove_outliers=remove_outliers,
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
    plt.savefig(filename, dpi=300)
    custom_print(f"Saved report to '{filename}'.", 'info')
    plt.clf()

    custom_print('Report generation complete.', 'done')


def generate_error_plot(
    al_loop_node,
    device_str: str = 'cpu',
    ax=None,
    database_path: str = None,
    model_path: str = None,
    threshold_meV: float = None,
    remove_outliers: bool = False,
):
    from mace.calculators import MACECalculator
    from mace.tools import torch_tools
    from rich.progress import track

    torch_tools.set_default_dtype('float32')

    # Create tmp folder in the current directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Get last iteration from aiida
        try:
            last_iter = [
                stp
                for stp in al_loop_node.called
                if stp.process_type == 'aiida.workflows:mdb-active-learning'
                or stp.process_type == 'aiida.workflows:mdb-simple-active-learning'
            ][-1]
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
            model_file: orm.SinglefileData = last_iter.outputs.m0_model_file
            with model_file.as_path() as f:
                model_path = shutil.copy(f, tmp_dir)

        # Load the training database
        train_db = ase_read(training_db_cp_path, format='extxyz', index=':')

        # Filter reference energies
        train_db = [
            at
            for at in train_db
            if at.info.get('config_type', '').lower() != 'isolatedatom'
        ]

        # Load model
        mace_model = MACECalculator(
            model_paths=model_path,
            device=device_str,
            default_dtype='float32',
        )

        files = ('E_nn_list.npy', 'F_nn_list.npy')
        E_nn_list = []
        F_nn_list = []
        # Iterate over the data loader to get the model energies
        # Use a rich progress bar to show the progress
        # Only run if any of the two files are missing
        if len(set(files) - set(os.listdir())) != 0:
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
                forces_std = simplify_forces_struct(forces_output)
                F_nn_list.append(forces_std)

        else:
            custom_print('Loading energies and forces from file...', 'info')
            E_nn_list = np.load('E_nn_list.npy')
            F_nn_list = np.load('F_nn_list.npy')

        np.save('E_nn_list.npy', E_nn_list)
        np.save('F_nn_list.npy', F_nn_list)
        E_dft_list = np.array([atoms.info['REF_energy'] for atoms in train_db])
        F_dft_list = np.array(
            [simplify_forces_struct(atoms.arrays['REF_forces']) for atoms in train_db]
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
        # F_sorted_diff_list_meV = np.argsort(np.abs(E_diff_list_meV))

        outlier_list = []
        remove_indices = []
        # Mark outliers, appending the E difference in meV to the structure info dict.
        for val in E_sorted_diff_list_meV[::-1]:
            if abs(E_diff_list_meV[val]) > threshold_meV:
                struct = train_db[int(val)]
                struct.info['E_dft_nn_diff_meV'] = E_diff_list_meV[val]
                outlier_list.append(struct)

                # Add outler indices to list for removal
                if remove_outliers:
                    remove_indices.append(val)

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
                    f'Used {threshold_meV} meV as threshold.'
                ),
                'info',
            )

        # Get RMSD
        E_rmsd = np.sqrt(np.mean(E_diff_list_meV**2))
        F_rmsd = np.sqrt(np.mean(F_diff_list_meV**2))

        # Get MAE
        E_mae = np.mean(np.abs(E_diff_list_meV))
        F_mae = np.mean(np.abs(F_diff_list_meV))

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
            E_dft_list_per_at, E_dft_list_per_at, color='#b16286', linestyle='--'
        )
        ax1_top.set_xlabel('DFT Energy [meV/atom]')
        ax1_top.set_ylabel('NN Energy [meV/atom]')
        ax1_top.set_title('Energy comparison')

        ax1_bottom = ax.figure.add_subplot(inner_grid[1, 0])
        ax1_bottom.text(
            x=tex_x_pos,
            y=tex_y_pos,
            s=f'MAE: {F_mae:.3f} meV/atom\nRMSD: {F_rmsd:.3f} meV/atom',
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
        ax1_bottom.set_xlabel('DFT Forces [meV/A]')
        ax1_bottom.set_ylabel('NN Forces [meV/A]')
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
        ax_twin_b.set_ylabel('Abs. F diff [meV/atom]')

        # Set labels and title
        ax2_top.set_xlabel('Structure index')
        ax2_top.set_ylabel('Energy [eV]')
        ax2_top.set_title('Energy comparison')
        ax2_bottom.set_xlabel('Structure index')
        ax2_bottom.set_ylabel('Forces [eV]')
        ax2_bottom.set_title('Forces comparison')

        lines, labels = ax2_top.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines + lines2, labels + labels2)
        lines_b, labels_b = ax2_bottom.get_legend_handles_labels()
        lines2_b, labels2_b = ax_twin_b.get_legend_handles_labels()
        ax_twin_b.legend(lines_b + lines2_b, labels_b + labels2_b)

        if save_fig_now:
            plt.tight_layout()
            filename = Path('./energy_difference.png').resolve()
            plt.savefig(filename, dpi=300)
            custom_print(f"Saved report to '{filename}'.", 'info')


def gen_init_db_report(
    train_db_path,
    threshold_E: float = None,
    threshold_F: float = None,
    remove_outliers: bool = False,
    color_type: str = None,
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

    # Create masks
    E_mask = np.full(len(E_dft_list), False)
    F_mask = np.full(len(E_dft_list), False)
    if threshold_E:
        E_mask = np.abs(E_dft_list) < float(threshold_E)
    if threshold_F:
        F_mask = np.abs(F_dft_list_max) < float(threshold_F)

    combined_mask = E_mask & F_mask

    if np.any(combined_mask) and remove_outliers:
        # Apply combined_mask to E_dft_list and F_dft_list
        indices = np.array(range(len(E_dft_list)))[combined_mask]
        E_dft_list = E_dft_list[combined_mask]
        F_dft_list_max = F_dft_list_max[combined_mask]
        F_dft_list_avg = F_dft_list_avg[combined_mask]

        valid_idxs = np.array(range(len(train_db)))
        structs = [train_db[struct_idx] for struct_idx in valid_idxs[combined_mask]]

        custom_print(
            f'Filtered {len(train_db)-len(structs)} structures based on thresholds.',
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

    # Energy bar chart
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
            customdata=np.stack((formulas, phases, struct_type), axis=-1),
            hovertemplate="""
            <b>Energy</b><br><br>
            <b>Structure:</b> %{x}<br>
            <b>Energy:</b> %{y} eV<br>
            <b>Formula:</b> %{customdata[0]}<br>
            <b>Phase:</b> %{customdata[1]}<br>
            <b>Struct type:</b> %{customdata[2]}<br>
            <extra></extra>
            """,
        ),
        row=1,
        col=1,
    )

    # Forces bar chart
    forces_hovertemplate = """
            <b>Forces</b><br><br>
            <b>Structure:</b> %{x}<br>
            <b>Force std. dev.:</b> %{y} eV<br>
            <b>Formula:</b> %{customdata[0]}<br>
            <b>Phase:</b> %{customdata[1]}<br>
            <b>Struct type:</b> %{customdata[2]}<br>
            <extra></extra>
        """
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=F_dft_list_max,
            name='DFT Forces Max',
            # marker_color='#83a598',
            hoverinfo='x+y',
            mode='lines+markers',
            marker=dict(size=2, color=F_max_type_color),
            line=dict(width=0.85, color=forces_max_line_color),
            customdata=np.stack((formulas, phases, struct_type), axis=-1),
            hovertemplate=forces_hovertemplate,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=F_dft_list_avg,
            name='DFT Forces Average',
            hoverinfo='x+y',
            mode='lines+markers',
            marker=dict(size=2, color=F_avg_type_color),
            line=dict(
                width=0.85,
                color=forces_avg_line_color,
            ),
            customdata=np.stack((formulas, phases, struct_type), axis=-1),
            hovertemplate=forces_hovertemplate,
        ),
        row=2,
        col=1,
    )

    if threshold_E:
        fig.add_shape(
            type='line',
            x0=0,
            x1=max(indices),
            y0=-threshold_E,
            y1=-threshold_E,
            line=dict(color='#282828', width=2, dash='dash'),
            yref='y1',
        )
        fig.add_annotation(text='E threshold', x=0, y=-threshold_E, yref='y1')
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

    # Customize layout
    fig.update_layout(
        title='DFT Energy and Forces per Structure',
        yaxis=dict(autorange='reversed', title='Energy [eV]'),
        yaxis2=dict(title='Forces [eV]'),
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
    ini_db_size: int,
    seed_gen_db_sizes: list[int],
    train_db_sizes: list[int],
    mace_e: list[float],
    mace_f: list[float],
    it_idx: list[int],
    ax,
):
    # Get unix timestamp for filename
    timestamp = int(time.time())
    filename = Path(f'al_loop_report_{timestamp}.png').resolve()

    # Define colors from the gruvbox palette
    colors = [
        '#83a598',
        '#b16286',
        '#98971a',
        '#fb4934',
    ]
    line_color = '#28282855'

    # Plot seed and train db sizes as a stacked bar chart over every iteration
    width = 0.3

    # Adding inital database size as iteration 0
    it_idx = [0] + it_idx
    ind = np.array(it_idx)
    train_db_sizes = [ini_db_size] + train_db_sizes
    seed_gen_db_sizes = [ini_db_size] + seed_gen_db_sizes

    # Plotting seed and train db sizes
    ax1 = ax.figure.add_subplot(ax[0, 0])
    ax1.bar(ind, train_db_sizes, width=width, label='train_db', color=colors[0])
    ax1.bar(
        ind + width,
        seed_gen_db_sizes,
        width=width,
        label='seed_gen_db',
        color=colors[1],
    )
    ax1.set_xticks(ind + width / 2, ind)
    ax1.set_xlabel('AL Loop Step')
    ax1.set_ylabel('Number of structures')
    ax1.legend()
    ax1.set_title('Seed and Train Database Evolution')

    # Add text labels to top left figure bars
    for idx, seed, train in zip(
        it_idx, seed_gen_db_sizes, train_db_sizes, strict=False
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
    for idx, seed, train in zip(it_idx, seed_gen_db_diff, train_db_diff, strict=False):
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

    ax2.bar(ind, train_db_diff, width=width, label='train_db', color=colors[0])
    ax2.bar(
        ind + width,
        seed_gen_db_diff,
        width=width,
        label='seed_gen_db',
        color=colors[1],
    )
    ax2.axhline(y=0, color=line_color, linestyle='--')
    ax2.set_xticks(ind + width / 2, ind)
    ax2.set_xlabel('AL Loop Step')
    ax2.set_ylabel(r'$\Delta$ Number of structures')
    ax2.set_title('Structure count change over iteration')
    ax2.legend()

    # Plot MACE model energy performance
    ax3 = ax.figure.add_subplot(ax[1, 0])
    ind = np.arange(len(mace_e)) + 1
    ax3.plot(ind, mace_e, label='MACE Energy', color=colors[2], marker='o')
    ax3.set_xticks(ind, ind)
    ax3.set_xlabel('AL Loop Step')
    ax3.set_ylabel('RMSE E per atom [meV]')
    ax3.set_title('Evolution of best MACE Model Energy RMSE')

    # Add a horizontal line to mark chemical accuracy for energy and forces
    chem_acc = 43.37  # meV
    ax3.axhline(y=chem_acc, color=line_color, linestyle='--')

    # Plot MACE model force performance
    ax4 = ax.figure.add_subplot(ax[1, 1])
    ax4.plot(ind, mace_f, label='MACE Forces', color=colors[3], marker='o')
    ax4.set_xticks(ind, ind)
    ax4.set_xlabel('AL Loop Step')
    ax4.set_ylabel('RMSE F [meV / A]')
    ax4.text(x=1.5, y=chem_acc + 0.5, s='Chem. Acc.', color=line_color)
    ax4.set_title('Evolution of best MACE Model Force RMSE')

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
