"""Run active learning loop from a TOML configuration file."""

import argparse
import contextlib
import pathlib as pl
import time
import tomllib
import warnings
from argparse import RawTextHelpFormatter

from aiida.orm import Bool, Dict, Int, Str
from rich.traceback import install as traceback_install

from MatDBForge.core.code_utils import (
    check_mdb_version,
    custom_print,
    get_mdb_version_info,
    init_logger,
)

warnings.filterwarnings('ignore')


def create_active_learning_builder(
    toml_dict: dict,
    toml_dict_path: pl.Path = None,
    complete: bool = False,
    log_file_path: pl.Path = None,
    al_start_mode: str = 'normal',
):
    """
    Create builder object for the ActiveLearningWorkChain.

    Parameters
    ----------
    toml_dict : dict
        Dictionary coming from parsing an MDB TOML settings file.

    Returns
    -------
    ProcessBuilder
        A process builder that helps setting up the inputs for
        an ActiveLearningWorkChain.
    """
    from aiida.plugins import WorkflowFactory

    # Getting builder for workchain
    if complete:
        al_calculation = WorkflowFactory('mdb-active-learning-base')
    else:
        al_calculation = WorkflowFactory('mdb-simple-active-learning-base')
    builder = al_calculation.get_builder()

    ## General AL settings
    al_conf = toml_dict['active_learning']

    if toml_dict_path:
        builder.active_learning.toml_file = str(toml_dict_path)

    builder.active_learning.enable_ntfysh = Bool(al_conf.get('enable_ntfysh', False))

    builder.active_learning.mdb_working_directory = Str(pl.Path.cwd().resolve())
    builder.active_learning.run_name = al_conf['run_name']
    builder.active_learning.load_init_models = al_conf.get('load_init_models')
    builder.active_learning.load_descriptor_calc = al_conf.get('load_descriptor_calc')
    builder.active_learning.load_md_calcs = al_conf.get('load_md_calcs')
    builder.active_learning.init_db_path = str(
        pl.Path(al_conf['init_db_path']).resolve()
    )

    if al_conf.get('results_dir'):
        results_dir = pl.Path(al_conf['results_dir']).resolve()
    else:
        results_dir = pl.Path('./results').resolve()
    builder.active_learning.results_dir = str(results_dir)

    if al_conf.get('log_path'):
        log_path = pl.Path(al_conf['log_path']).resolve()
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_path = pl.Path(f'mdb_output_{timestamp}.log').resolve()

    # Getting contents from previous log file and appending to new log file
    # to conserve the history of the previous run.
    if log_file_path:
        with open(log_file_path) as f:
            old_log: str = f.read()

        with open(log_path, 'a+') as f:
            f.write(old_log)
            f.write('\n')

    builder.log_path = str(log_path)

    builder.active_learning.final_db_name = al_conf['final_db_name']
    builder.active_learning.max_iterations = Int(int(al_conf['max_iterations']))
    builder.active_learning.al_mode = al_conf.get('al_mode', 'md')

    # Code settings
    code_settings = toml_dict.get('code', {})
    builder.active_learning.container_settings = Dict(
        code_settings.get('container', {})
    )

    # Getting extrapolation settings
    builder.active_learning.check_extrapolation_type = toml_dict.get(
        'extrapolation', {}
    ).get('check_extrapolation_type', 'advanced')

    ## AL seed settings
    builder.active_learning.seed_size_frac = float(
        toml_dict['al_seed']['seed_size_frac']
    )
    builder.active_learning.seed_min_num_structs = int(
        toml_dict['al_seed']['seed_min_num_structs']
    )
    builder.active_learning.seed_max_num_structs = int(
        toml_dict['al_seed']['seed_max_num_structs']
    )

    builder.active_learning.seed_select_settings = toml_dict['al_seed']

    # Delete structures from MD seed if they are well represented
    # Default is True
    builder.active_learning.delete_seed_structs = toml_dict.get('al_seed', {}).get(
        'delete_seed_structs', True
    )

    builder.active_learning.committee_num_models = int(
        toml_dict['committee_eval']['committee_num_models']
    )
    builder.active_learning.model_acc_multiplier = float(
        toml_dict['interpolation']['model_acc_multiplier']
    )

    ## Data reduction settings
    data_reduction_settings: dict = toml_dict.get('data_reduction', {})
    builder.active_learning.data_reduction_settings = Dict(
        value=data_reduction_settings
    )

    ## MD settings
    md_params = toml_dict['md']['parameters']
    builder.active_learning.al_keep_struct_every_n_ps = float(
        md_params['al_keep_struct_every_n_ps']
    )
    builder.active_learning.md_temperature_list_K = md_params['temperature_list_K']
    builder.active_learning.md_max_temp_multiplier = md_params['max_temp_multiplier']
    builder.active_learning.md_num_steps = int(md_params['num_steps'])
    builder.active_learning.md_timestep_duration_ps = float(
        md_params['timestep_duration_ps']
    )
    builder.active_learning.gather_traj_cnt_lattice = md_params[
        'gather_traj_cnt_lattice'
    ]
    builder.active_learning.use_kokkos = md_params['use_kokkos']

    # MACE MD Settings
    builder.active_learning.md_parameters = Dict(value=toml_dict['md']['parameters'])

    # MD filters
    builder.active_learning.md_filters = Dict(value=toml_dict['md'].get('filters'))

    ## MACE training settings
    builder.active_learning.mace_train = Dict(value=toml_dict['mace_train'])

    ## Committee Evaluation Settings
    builder.active_learning.committee_eval = Dict(value=toml_dict['committee_eval'])

    ## DFT method selection and settings
    dft_method = toml_dict['dft'].get('dft_method', 'mace')
    builder.active_learning.dft_method = dft_method
    if dft_method == 'vasp':
        selected_settings_dict = toml_dict['dft']['vasp']
    elif dft_method == 'mace':
        # Make sure the path to the MACE potential is absolute
        mace_potential_path = pl.Path(
            toml_dict['dft']['mace']['mace_potential_path']
        ).absolute()
        toml_dict['dft']['mace']['mace_potential_path'] = str(mace_potential_path)

        if not mace_potential_path.is_absolute():
            raise ValueError(
                'The path to the MACE potential must be absolute.'
                f'Current path: {mace_potential_path}'
            )
        selected_settings_dict = toml_dict['dft']['mace']

    # Adding ignore container options if available
    selected_settings_dict['ignore_container'] = toml_dict.get('dft', {}).get(
        'ignore_container', False
    )
    builder.active_learning.dft_settings = Dict(value=selected_settings_dict)
    builder.active_learning.dft_calc_limit = toml_dict['dft'].get('dft_calc_limit')

    ## Descriptor settings
    builder.active_learning.descriptor_settings = Dict(value=toml_dict['descriptors'])

    ## Start mode
    builder.active_learning.al_start_mode = Str(al_start_mode)

    return builder


def resume_al_loop_builder(
    prev_run_dir: pl.Path, toml_dict_path: pl.Path = None, log_file_path: pl.Path = None
):
    # Resume dictionary. This will be used to pass the last iteration and
    # the paths to the train_db and seed_db files to the base workchain
    # to resume the active learning loop from the beginning of the last iteration.
    resume_dict = {'last_iteration': None, 'train_db_path': None, 'seed_db_path': None}

    wk_uuid = None
    wk_node = None

    # Check if the provided path is a directory
    if prev_run_dir and not prev_run_dir.is_dir():
        raise NotADirectoryError(
            f"The path '{prev_run_dir}' is not a directory. "
            "Please provide a valid active learning 'run_<UUID>' directory."
        )

    # Get pk/uuid of the base workchain using one of two approaches:
    # 1. Get the pk/uuid from the prev_run_dir folder name
    with contextlib.suppress(Exception):
        wk_uuid = prev_run_dir.stem.split('_')[-1]

    # 2. Get the pk/uuid from the log file in the prev_run_dir folder
    if not wk_uuid:
        with contextlib.suppress(Exception):
            for log in prev_run_dir.glob('*.log'):
                with open(log) as f:
                    log_file = f.readlines()
            wk_uuid = log_file[0].split('|')[0].strip('[')

    # Load aiida node if found
    with contextlib.suppress(Exception):
        from aiida.orm import load_node

        wk_node = load_node(wk_uuid)

    resume_dict['prev_workchain_uuid'] = wk_uuid

    # REMOVE: Disable wk_node until implemented
    # Node found, reading settings from aiida node
    if wk_node and False:  # noqa
        # TODO: Implement this part
        ...
    # Node not found, using file-based approach
    else:
        # Checking that the previous run directory exists
        if not prev_run_dir.exists():
            raise FileNotFoundError(
                f"The directory '{prev_run_dir}' does not exist. "
                'Please make sure that is the correct path.'
            )

        # Read toml settings file
        if not toml_dict_path:
            for toml in prev_run_dir.glob('*.toml'):
                toml_dict_path = toml

        toml_dict = read_toml_config(toml_dict_path)

        # Populating resume dictionary with last iteration
        run_tmp_path = prev_run_dir / 'run_tmp_data'
        run_tmp_files = [f for f in run_tmp_path.glob('*.pkl')]
        run_tmp_files.sort(key=lambda x: int(x.stem.split('-')[-1]))
        for it_file in run_tmp_files:
            last_iteration = int(it_file.stem.split('-')[-1])
        resume_dict['last_iteration'] = last_iteration

        # Getting train_db and seed_db paths
        train_db_path = toml_dict['active_learning'].get('init_db_path')

        should_reset_seed_db = toml_dict['active_learning'].get('reset_seed_db', False)

        if (
            prev_run_dir.exists()
            and (prev_run_dir / 'mdb_seed_db.xyz').exists()
            and not should_reset_seed_db
        ):
            seed_db_path = prev_run_dir / 'mdb_seed_db.xyz'
        else:
            custom_print(
                'Seed database is intialized as a copy of the training database. '
                "This behavior can be changed using 'active_learning.reset_seed_db'.",
                'warning',
            )
            seed_db_path = toml_dict['active_learning'].get('init_db_path')

        custom_print(f'Reading training database from: {train_db_path}', 'warning')
        custom_print(
            'Please, make sure that this file corresponds with the complete database '
            '(Dt) that you wish to resume from. You can do this by setting the path in '
            f"the '{toml_dict_path}' file, in key"
            " 'active_learning.init_db_path'.",
            'warning',
        )
        print()

        resume_dict['train_db_path'] = str(train_db_path)
        resume_dict['seed_db_path'] = str(seed_db_path)

    # Getting builder for workchain
    builder = create_active_learning_builder(
        toml_dict=toml_dict,
        toml_dict_path=toml_dict_path,
        log_file_path=log_file_path,
        al_start_mode='resume',
    )

    # Setting resume dictionary and updating builder inputs
    builder.resume_dict = resume_dict
    builder.active_learning.run_name = (
        builder.active_learning.run_name.value + '_resumed'
    )
    builder.active_learning.init_db_path = resume_dict['train_db_path']

    # Store resume dictionary and run name
    # TODO: Check if necessary to store manually in this case
    builder.resume_dict.store()
    builder.active_learning.run_name.store()
    builder.active_learning.init_db_path.store()

    # Check for any model files in the folder and return an error if not found
    # builder.dft_settings["mace_potential_path"]
    if toml_dict.get('dft', {}).get('dft_method') == 'mace':
        mace_model_path = pl.Path(
            builder.active_learning.dft_settings['mace_potential_path']
        )
        if not mace_model_path.exists():
            raise FileNotFoundError(
                'No model files found in the run directory.\n'
                f"Check that '{mace_model_path}' exists."
            )

    return builder


def run_active_learning():
    # Use rich formatter for tracebacks
    traceback_install(
        width=88,
    )

    # Create the top-level parser
    parser = argparse.ArgumentParser(
        prog='run_active_learning',
        description=(
            'Launch a MDB active learning loop.\n'
            'Provide a TOML settings file to start the active learning loop, '
            'or use any of the available commands.'
        ),
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version='MatDBForge version: ' + str(get_mdb_version_info()[0]),
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode. This will run the workchain in the foreground.',
        default=False,
    )

    # Create a subparsers object
    subparsers = parser.add_subparsers(
        dest='command', help='List of available commands'
    )

    # Create the subparser for the 'run' command
    al_loop_parser = subparsers.add_parser(
        'run',
        help='Run the active learning loop.',
        usage=(
            'run_active_learning run [-h]\n'
            'Run the active learning loop with the specified settings.'
        ),
    )

    al_loop_parser.add_argument(
        '-c',
        '--config_file',
        help=(
            'path pointing to a TOML settings file.\n'
            'By default `active_learning_settings.toml` will be searched in the CWD.'
        ),
        type=pl.Path,
        default='./active_learning_settings.toml',
        # required=True,
        metavar='PATH',
    )

    al_loop_parser.add_argument(
        '--complete',
        help='Use the old version of the active learning workchain.',
        action='store_const',
        const=True,
    )

    al_loop_parser.add_argument(
        '--dashboard',
        help='Launch a dashboard to keep track of the active learning loop.',
        action='store_const',
        const=True,
    )

    # Add arguments specific to the 'dashboard' subcommand
    al_loop_parser.add_argument(
        '--update_interval',
        help=('Dashboard refresh time interval in seconds.'),
        type=int,
        default=60,
        metavar='n_sec',
    )
    al_loop_parser.add_argument(
        '--port',
        help=('Dashboard port to use for the webapp.'),
        type=int,
        default=8000,
        metavar='port',
    )
    al_loop_parser.add_argument(
        '--debug',
        help=('Enable Flask debug for the dashboard.'),
        action='store_const',
        const=True,
        default=False,
    )
    al_loop_parser.add_argument(
        '--online',
        help=('Enable online dashboard.'),
        action='store_const',
        const=True,
        default=False,
    )

    # Create the subparser for the 'report' command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate reports for MatDBForge.',
        usage=(
            'run_active_learning report [-h]\n'
            'Generate a report for a MatDBForge active learning loop or '
            'a MatDBForge initial database.'
        ),
    )
    report_subparsers = report_parser.add_subparsers(dest='subcommand', required=True)

    # Create the subparser for the 'report' command
    init_db_parser = report_subparsers.add_parser(
        'init_db',
        help='Generate a report for a MatDBForge initial database.',
        usage=(
            'run_active_learning report init_db [-h] --db_path <PATH>\n'
            'Generate a report for a MatDBForge initial database.'
        ),
    )

    init_db_parser.add_argument(
        '--db_path',
        '-d',
        help=('Path to the database file.'),
        metavar='<PATH>',
        default=None,
        required=True,
    )

    init_db_parser.add_argument(
        '--threshold_E',
        help=(
            'Threshold for E to consider a structure as outlier, '
            'in eV, or eV/atom if enabled'
        ),
        metavar='<FLOAT>',
        default=None,
        required=False,
        type=float,
    )

    init_db_parser.add_argument(
        '--threshold_F',
        help=(
            'Threshold for F to consider a structure as outlier, '
            'in eV, or eV/atom if enabled'
        ),
        metavar='<FLOAT>',
        default=None,
        required=False,
        type=float,
    )

    init_db_parser.add_argument(
        '--remove_outliers',
        help=('Remove outliers from the plot'),
        action='store_const',
        const=True,
    )
    init_db_parser.add_argument(
        '--color_type',
        help=('What to use for coloring the plot'),
        choices=['phase', 'struct_type'],
        type=str,
    )

    init_db_parser.add_argument(
        '--per_atom',
        help=('Display energy and force values per atom'),
        action='store_const',
        const=True,
        default=False,
    )

    # Create the subparser for the 'al_loop' subcommand
    al_loop_report_parser = report_subparsers.add_parser(
        'al_loop',
        help=(
            'Generate a report for an active learning loop '
            'by providing an AiiDA PK/UUID or a log file path.'
        ),
        usage=(
            'run_active_learning report al_loop (--loop_id <ID> | --log_path <PATH>)'
            '\nGenerate a report for an active learning loop by providing an AiiDA '
            'PK/UUID or a log file path.'
        ),
    )

    # Add mutually exclusive group for the 'report' command
    report_group = al_loop_report_parser.add_mutually_exclusive_group(required=True)

    # Add arguments specific to the 'report' subcommand
    report_group.add_argument(
        '--loop_id',
        '-i',
        help=('AiiDA PK/UUID of the active learning loop.'),
        metavar='<ID>',
    )
    report_group.add_argument(
        '--log_path',
        '-l',
        help=('MatDBForge log of the active learning loop.'),
        metavar='<PATH>',
    )
    al_loop_report_parser.add_argument(
        '--device',
        '-d',
        help=('String representing a device to run inference on.'),
        metavar='<DEVICE>',
        default='cpu',
        choices=['cpu', 'cuda'],
    )
    al_loop_report_parser.add_argument(
        '--get_error_plot',
        help=('Get error plot'),
        action='store_const',
        const=True,
    )
    al_loop_report_parser.add_argument(
        '--mace_enable_cueq',
        help=('Enable CUEQ for MACE'),
        action='store_const',
        const=True,
        default=False,
    )

    al_loop_report_parser.add_argument(
        '--remove_outliers',
        help=('Remove outliers from the error plot'),
        action='store_const',
        const=True,
    )

    al_loop_report_parser.add_argument(
        '--model',
        help=('Path to the model file.'),
        metavar='<PATH>',
        default=None,
        required=False,
    )

    al_loop_report_parser.add_argument(
        '--database',
        help=('Path to the database file.'),
        metavar='<PATH>',
        default=None,
        required=False,
    )
    al_loop_report_parser.add_argument(
        '--limit_num_steps',
        help=('Number of steps to limit the report to.'),
        metavar='<INT>',
        default=None,
        required=False,
        type=int,
    )

    al_loop_report_parser.add_argument(
        '--threshold_E_meV',
        help=(
            'Threshold for the difference between E_DFT and E_NN to mark a structure as'
            ' an outlier, in meV. Default is 150 meV.'
        ),
        metavar='<FLOAT>',
        default=150.0,
        required=False,
    )
    al_loop_report_parser.add_argument(
        '--threshold_F_meV',
        help=(
            'Threshold for the difference between F_DFT and F_NN to mark a structure as'
            ' an outlier, in meV. Default is 25000 meV.'
        ),
        metavar='<FLOAT>',
        default=2.5e4,
        required=False,
    )

    al_loop_report_parser.add_argument(
        '--title',
        help=(
            'Run name to be used as title. '
            'If not provided, a name will be gathered from the calculation files.'
        ),
        metavar='<TITLE>',
        required=False,
    )

    al_loop_report_parser.add_argument(
        '--db_latent_space_evolution',
        help=('Plot database evolution over time'),
        action='store_const',
        const=True,
    )

    al_loop_report_parser.add_argument(
        '--autoencoder_path',
        help=(
            'Path to the autoencoder folder. '
            'Should contain an autoencoder model and the database files for every step'
        ),
        metavar='<PATH>',
        default=None,
        required=False,
    )

    # Create the subparser for the 'al_loop' subcommand
    al_loop_batch_parser = report_subparsers.add_parser(
        'al_loop_batch',
        help=(
            'Generate a report for a series of active learning loops, '
            'as a way of comparing them by providing an AiiDA PK/UUID or '
            ' a log file path.'
        ),
        usage=(
            'run_active_learning report al_loop (--loop_id <ID> | --log_path <PATH>)'
            '\nGenerate a report for an active learning loop by providing an AiiDA '
            'PK/UUID or a log file path.'
        ),
    )

    al_loop_batch_parser.add_argument(
        '--db_path',
        '--d',
        help='Path to the database file.',
        metavar='<PATH>',
    )
    al_loop_batch_parser.add_argument(
        '--threshold_meV',
        help=('Threshold to consider a structure as outlier, in meV. Default is 100.'),
        metavar='<FLOAT>',
        default=100.0,
        required=False,
    )

    al_loop_batch_parser.add_argument(
        '--remove_outliers',
        help=('Remove outliers from the error plot'),
        action='store_const',
        const=True,
    )

    # Create the subparser for the 'al_loop' subcommand
    al_loop_performance_report_parser = report_subparsers.add_parser(
        'al_loop_performance',
        help=(
            'Generate a performance report for an active learning loop '
            'by providing an AiiDA PK/UUID or a log file path.'
        ),
        usage=(
            'run_active_learning report al_loop_performance '
            '(--loop_id <ID> | --log_path <PATH>)'
            '\nGenerate a report for an active learning loop by providing an AiiDA '
            'PK/UUID or a log file path.'
        ),
    )

    # Add arguments specific to the 'report' subcommand
    al_loop_performance_report_parser.add_argument(
        '--loop_ids',
        '-i',
        nargs='+',
        help=(
            'AiiDA PK/UUIDs of the active learning loop. '
            'Several IDs can be provided for multi-stage loops'
            ' (e.g., when using resume)'
        ),
        metavar='<ID>',
    )
    # Add arguments specific to the 'report' subcommand
    al_loop_performance_report_parser.add_argument(
        '--output_filename',
        '-o',
        help=('Filename for the report plot to generate'),
        metavar='<PATH>',
        default=None,
        required=False,
    )

    # Create the subparser for the 'resume' command
    resume_parser = subparsers.add_parser(
        'resume',
        help='Resume an active learning loop using a results folder.',
        usage=(
            'run_active_learning resume [-h] --dir_resume <PATH> '
            '[--config_file <PATH>]\n\n'
            'Resume an active learning loop using a results folder.'
        ),
    )

    resume_parser.add_argument(
        '--dir_resume',
        '-d',
        help=('Path to the results directory of a previous active learning loop run.'),
        metavar='<PATH>',
        required=True,
    )

    resume_parser.add_argument(
        '--config_file',
        '-c',
        help=(
            'Path pointing to a TOML settings file.\n'
            'Optional, as all calculation folders should contain A TOML file.'
        ),
        type=pl.Path,
        metavar='<PATH>',
        required=False,
        default=None,
    )
    resume_parser.add_argument(
        '--old_log_path',
        '-l',
        help=(
            'Path pointing to the previous run log file.\n'
            'Optional. Will prepend the previous contents to the new calculation log.'
        ),
        type=pl.Path,
        metavar='<PATH>',
        required=False,
        default=None,
    )

    # Create the subparser for the 'dashboard' command
    dashboard_parser = subparsers.add_parser(
        'dashboard', help='Launch a dashboard to keep track of the active learning loop'
    )

    # Add arguments specific to the 'dashboard' subcommand
    dashboard_parser.add_argument(
        '--update_interval',
        help=('Refresh time interval in seconds'),
        type=int,
        default=60,
        metavar='n_sec',
    )
    dashboard_parser.add_argument(
        '--port',
        help=('Port to use for the webapp'),
        type=int,
        default=8000,
        metavar='port',
    )

    dashboard_parser.add_argument(
        '--online',
        help=('Enable online'),
        action='store_const',
        const=True,
        default=False,
    )
    dashboard_parser.add_argument(
        '-i',
        '--pk',
        '--loop_id',
        help=('AiiDA PK/UUID of the active learning loop.'),
        metavar='<ID>',
    )

    # Getting CLI arguments
    args = parser.parse_args()

    logger = init_logger('active_learning', show_log_path=False)

    # Checking version
    check_mdb_version(logger)

    from MatDBForge.core.command_line.command_line_utils import validate_config_file

    # Initialize ntfysh variable
    ntfysh_topic = None

    # Check if all required sections are present
    if args.command == 'run' or args.command == 'resume':
        from aiida.engine import run, submit
        from aiida.orm import Bool, Str

        # Check if TOML file is correct
        validate_config_file(
            config_path=args.config_file, config_type='active_learning'
        )
        print()

    if args.command == 'report':
        from MatDBForge.active_learning import report_utils as mdb_report

        if args.subcommand == 'al_loop':
            mdb_report.gen_al_loop_report(
                args.loop_id,
                args.log_path,
                device=args.device,
                get_error_plot=args.get_error_plot,
                model_path=args.model,
                database_path=args.database,
                threshold_E_meV=float(args.threshold_E_meV),
                threshold_F_meV=float(args.threshold_F_meV),
                remove_outliers=args.remove_outliers,
                title=args.title,
                get_latent_space=args.db_latent_space_evolution,
                autoencoder_path=args.autoencoder_path,
                limit_num_steps=args.limit_num_steps,
                enable_cueq=args.mace_enable_cueq,
            )
            return
        elif args.subcommand == 'init_db':
            # Generating a report for an initial database
            mdb_report.gen_init_db_report(
                train_db_path=args.db_path,
                threshold_E=args.threshold_E,
                threshold_F=args.threshold_F,
                remove_outliers=args.remove_outliers,
                color_type=args.color_type,
                per_atom=args.per_atom,
            )
            return
        elif args.subcommand == 'al_loop_batch':
            # Generating a report for an initial database
            mdb_report.gen_batch_report(training_db_path=args.db_path)

        elif args.subcommand == 'al_loop_performance':
            # Generating a report for an initial database
            mdb_report.gen_performance_report(
                al_loop_pk=args.loop_ids,
                output_filename=args.output_filename,
            )

    # Resume a previous calculation
    elif args.command == 'resume':
        # Getting path for config file if provided
        config_file = pl.Path(args.config_file).resolve() if args.config_file else None

        builder = resume_al_loop_builder(
            prev_run_dir=pl.Path(args.dir_resume).resolve(),
            toml_dict_path=config_file,
            log_file_path=args.old_log_path,
        )

    elif args.command == 'dashboard':
        from MatDBForge.core.command_line.cli_dashboard import run_dashboard_app

        run_dashboard_app(
            process_id=str(args.pk),
            port=args.port,
            update_interval=args.update_interval,
            debug=args.debug,
            online=args.online,
        )

    # Start a new al loop
    elif args.command == 'run':
        # Loading TOML config file
        toml_dict = read_toml_config(args.config_file)

        from aiida import load_profile

        try:
            # Loading default aiida profile
            load_profile(profile=toml_dict['active_learning']['aiida_profile'])
        except Exception as e:
            custom_print(f"Error loading aiida profile: '{e}'", 'error')

        # Parsing settings from TOML and creating builder for aiida
        builder = create_active_learning_builder(
            toml_dict,
            toml_dict_path=pl.Path(args.config_file).resolve(),
            complete=args.complete,
        )

    # Start a new al loop
    if (
        args.command in ['run', 'resume']
        and builder.active_learning.get('enable_ntfysh', Bool(False)).value
    ):
        # Enable ntfysh if specified in the config file
        # if builder.active_learning.get('enable_ntfysh', Bool(False)).value:
        from MatDBForge.active_learning.active_learning_utils import (
            generate_model_name,
        )
        from MatDBForge.core.code_utils import display_qr_in_cli, save_qr_to_file

        ntfysh_topic = 'mdb_' + generate_model_name()

        custom_print(
            f'ntfy.sh notifications enabled. '
            f"Subscribe to 'https://ntfy.sh/{ntfysh_topic}'"
        )

        # Do not display the QR in resume mode
        if args.command in ['run']:
            custom_print('Displaying QR code for ntfy.sh subscription:')
            print()
            save_qr_to_file(
                data=f'https://ntfy.sh/{ntfysh_topic}',
                filename=f'qr_{ntfysh_topic}.png',
            )
            display_qr_in_cli(f'ntfy.sh/{ntfysh_topic}')
        print()

    if ntfysh_topic:
        builder.active_learning.ntfysh_topic = Str(ntfysh_topic)

    # Check if dashboard is enabled
    if hasattr(args, 'dashboard'):
        dashboard_enabled = args.dashboard is not None
    else:
        dashboard_enabled = False

    # Launch dashboard
    if dashboard_enabled and args.command in ['run', 'resume']:
        from MatDBForge.core.command_line.cli_dashboard import run_dashboard_app

        node = submit(builder)
        time.sleep(1)

        run_dashboard_app(
            process_id=str(node.pk),
            port=args.port,
            update_interval=args.update_interval,
            debug=args.debug,
            online=args.online,
        )

    # Launch normal CLI or resume run, without dashboard
    elif not dashboard_enabled and args.command in ['run', 'resume']:
        if not args.debug:
            builder.active_learning.debug_mode = Bool(False)

            # Submit workchain to the daemon
            node = submit(builder)

            # Pretty-print details of the submitted active learning loop
            active_learning_run_print_details(
                process_pk=str(node.pk),
                log_path=builder.log_path.value,
                process_uuid=str(node.uuid),
                ntifysh_topic=ntfysh_topic,
            )
        else:
            # Run workchain in the foreground
            builder.active_learning.debug_mode = Bool(True)
            node = run(builder)


def active_learning_run_print_details(
    process_pk: str,
    log_path: str,
    process_uuid: str = None,
    ntifysh_topic: str = None,
):
    """Prints pretty output for active learning using the rich library."""
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from MatDBForge.core.command_line.command_line_utils import MDB_LOGO

    console = Console(record=True)

    store_output_file = f'mdb_active_learning_run_details_{process_pk}.out'

    # Creating Text containing the process information
    process_text = Text.assemble(
        (
            f'{MDB_LOGO}',
            'bold magenta',
        ),
        justify='full',
    )
    process_text.append(
        Text.assemble(
            '\n• The active learning loop has been launched (PK: ',
            (f'{process_pk}', 'bold blue'),
            ') and is now handled by the AiiDA daemon.\n\n',
            '• You can monitor its progress with the following command: ',
            (f'verdi process report {process_pk}', 'bold blue'),
            '\n\n',
            '• The output of the calculation is located in: ',
            (f'\n{log_path}', 'bold blue'),
            '\n\n',
            '• You can check this information again in:',
            (f'\n{store_output_file}', 'bold blue'),
            '\n\n',
            '• Use ',
            (f'verdi process kill {process_pk}', 'bold red'),
            ' to stop the calcualtion.\n',
            justify='full',
        ),
    )

    # Create a table
    table = Table(
        show_header=True,
        header_style='bold magenta',
        # highlight=True,
        title='Summary Information',
        expand=True,
    )
    table.add_column('Info Type', style='dim', width=20)
    table.add_column('Details', style='bold')

    # Add rows with the information
    table.add_row('Status', '[green]Running :heavy_check_mark:')
    table.add_row('Process PK', f'{process_pk}')
    if process_uuid is not None:
        table.add_row('Process UUID', f'{process_uuid}')
    table.add_row(
        'MDB Log File',
        f'[bold]{pl.Path(log_path).name}',
    )

    if ntifysh_topic is not None:
        table.add_row(
            'ntfy.sh topic',
            f'[bold]{ntifysh_topic}',
        )
        table.add_row(
            'ntfy.sh link',
            f'[bold]https://ntfy.sh/{ntifysh_topic}',
        )

    # Create a group with the text and the table
    grp = Group(process_text, table)

    # Create a panel with the group inside
    panel = Panel(
        grp,
        title='MatDBForge Active Learning Run Details',
        border_style='magenta',
        expand=False,
        padding=(2, 2),
    )

    # Print the panel
    console.print(panel)
    console.save_text(store_output_file)


def read_toml_config(config_file: pl.Path | str):
    try:
        with open(config_file, 'rb') as f:
            toml_dict = tomllib.load(f)
    except FileNotFoundError as e:
        error_message = (
            f"The config file '{config_file}' does not exist. "
            'Please make sure that is the correct name or input a different path.'
        )
        raise FileNotFoundError(error_message) from e
    return toml_dict
