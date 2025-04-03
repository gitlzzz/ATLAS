"""Run active learning loop from a TOML configuration file."""

import argparse
import contextlib
import pathlib as pl
import time
import tomllib
import warnings
from argparse import RawTextHelpFormatter

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
    from aiida.orm import Dict, Int, Str
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

    builder.active_learning.run_name = al_conf['run_name']
    builder.active_learning.load_init_models = al_conf.get('load_init_models')
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

    builder.active_learning.seed_select_settings = toml_dict['al_seed'][
        'seed_select_settings'
    ]

    # Delete structures from MD seed if they are well represented
    # Default is True
    builder.active_learning.delete_seed_structs = toml_dict.get('al_seed', {}).get(
        'delete_seed_structs', True
    )

    builder.active_learning.committee_num_models = int(
        toml_dict['committee_eval']['committee_num_models']
    )
    builder.active_learning.model_acc_multiplier = float(
        al_conf['model_acc_multiplier']
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
        builder.active_learning.dft_settings = Dict(value=toml_dict['dft']['vasp'])
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
        builder.active_learning.dft_settings = Dict(value=toml_dict['dft']['mace'])

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

    # REMOVE: Disable wk_node until implemented
    wk_node = None

    # Node found, reading settings from aiida node
    if wk_node:
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
        run_tmp_files.sort(key=lambda x: x.stem.split('-')[-1])
        for it_file in run_tmp_files:
            last_iteration = int(it_file.stem.split('-')[-1])
        resume_dict['last_iteration'] = last_iteration

        # Getting train_db and seed_db paths
        for it_file in prev_run_dir.glob('*.xyz'):
            if 'mdb_train_db' in it_file.stem:
                resume_dict['train_db_path'] = str(it_file)
            elif 'mdb_seed_db' in it_file.stem:
                resume_dict['seed_db_path'] = str(it_file)

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

    parser.add_argument(
        '--complete',
        help='Use the old version of the active learning workchain.',
        action='store_const',
        const=True,
    )

    parser.add_argument(
        '--gui',
        help='Launch a dashboard to keep track of the active learning loop.',
        action='store_const',
        const=True,
    )

    # Create a subparsers object
    subparsers = parser.add_subparsers(
        dest='command', help='List of available commands'
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
        '-log',
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
        '--threshold_meV',
        help=('Threshold to consider a structure as outlier, in meV. Default is 100.'),
        metavar='<FLOAT>',
        default=100.0,
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
        '--autoencoder_folder',
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

    # Create the subparser for the 'gui' command
    gui_parser = subparsers.add_parser(
        'gui', help='Launch a dashboard to keep track of the active learning loop'
    )

    # Add arguments specific to the 'gui' subcommand
    gui_parser.add_argument(
        '--update_interval',
        help=('Refresh time interval in seconds'),
        type=int,
        default=60,
        metavar='n_sec',
    )
    gui_parser.add_argument(
        '--port',
        help=('Port to use for the webapp'),
        type=int,
        default=8000,
        metavar='port',
    )

    gui_parser.add_argument(
        '--debug',
        help=('Enable Flask debug'),
        action='store_const',
        const=True,
        default=False,
    )
    gui_parser.add_argument(
        '--online',
        help=('Enable online'),
        action='store_const',
        const=True,
        default=False,
    )
    gui_parser.add_argument(
        '-i',
        '--pk',
        '--loop_id',
        help=('AiiDA PK/UUID of the active learning loop.'),
        metavar='<ID>',
    )

    # Getting CLI arguments
    args = parser.parse_args()

    # Checking version
    from MatDBForge.core.code_utils import check_mdb_version

    check_mdb_version()

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
                threshold_meV=float(args.threshold_meV),
                remove_outliers=args.remove_outliers,
                title=args.title,
                get_latent_space=args.db_latent_space_evolution,
                autoencoder_path=args.autoencoder_folder,
            )
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
        elif args.subcommand == 'al_loop_batch':
            # Generating a report for an initial database
            mdb_report.gen_batch_report(training_db_path=args.db_path)

    # Resume a previous calculation
    elif args.command == 'resume':
        from aiida.engine import run

        # Getting path for config file if provided
        config_file = pl.Path(args.config_file).resolve() if args.config_file else None

        builder = resume_al_loop_builder(
            prev_run_dir=pl.Path(args.dir_resume).resolve(),
            toml_dict_path=config_file,
            log_file_path=args.old_log_path,
        )

        # Running the workchain
        node = run(builder)
    elif args.command == 'gui':
        from MatDBForge.core.command_line.cli_dashboard import run_dashboard_app

        run_dashboard_app(
            process_id=str(args.pk),
            port=args.port,
            update_interval=args.update_interval,
            debug=args.debug,
            online=args.online,
        )

    # Start a new al loop
    else:
        from aiida.engine import run, submit

        # Loading TOML config file
        toml_dict = read_toml_config(args.config_file)

        from aiida import load_profile

        try:
            # Loading default aiida profile
            load_profile(profile=toml_dict['active_learning']['aiida_profile'])
        except Exception as e:
            from MatDBForge.core.code_utils import custom_print

            custom_print(f"Error loading aiida profile: '{e}'", 'error')

        # Parsing settings from TOML and creating builder for aiida
        builder = create_active_learning_builder(
            toml_dict,
            toml_dict_path=pl.Path(args.config_file).resolve(),
            complete=args.complete,
        )

        if not args.gui:
            node = run(builder)
        else:
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
