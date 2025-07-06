"""Dashboard for monitoring and reviewing training Active Learning WorkChains."""

import pathlib as pl
import re
import time

import pandas as pd
from aiida import load_profile, orm
from aiida.cmdline.utils.common import get_workchain_report
from flask import Flask, render_template


def get_report(node):
    """Parses and styles the report from an AiiDA workchain node."""
    if isinstance(node, (int, str)):
        node = orm.load_node(node)

    report = get_workchain_report(node, levelname='REPORT')
    report_lines = report.split('\n')
    pattern = r'\[\d+\|[A-Za-z]+\|[A-Za-z_]+\]'

    styled_report = []
    for line_idx, line in enumerate(report_lines):
        # Catch error strings until the next aiida report
        if '|on_except]:' in line:
            report_part = []
            cnt = line_idx + 1

            parts = line.split(']: ')
            date_part, id_part, label_part = '', '', ''
            if len(parts) > 1:
                date_part = parts[0].split(' ')[0]
                date_part += ' ' + parts[0].split(' ')[1]
                id_part = str(parts[1].split('|')[0]).replace('[', '')
                id_part = id_part.replace(' ', '')
                id_part = f' [{id_part}] '

                label_part = (
                    '['
                    + str(parts[1].split('|')[1])
                    .replace('[', '')
                    .replace("'", '')
                    .replace(',', '')
                    + ']: '
                )

            # Check bounds to prevent IndexError
            while cnt < len(report_lines) and not re.search(pattern, report_lines[cnt]):
                report_part.append(report_lines[cnt])
                cnt += 1

            report_part = '\n'.join(report_part)
            styled_report.append(
                {
                    'date_part': date_part,
                    'id_part': id_part,
                    'label_part': '[ERROR]',
                    'report_part': report_part,
                }
            )
        elif re.search(pattern, line):
            parts = line.split(']: ')
            if len(parts) > 1:
                date_part = parts[0].split(' ')[0]
                date_part += ' ' + parts[0].split(' ')[1]
                id_part = str(parts[1].split('|')[0]).replace('[', '')
                id_part = id_part.replace(' ', '')
                id_part = f' [{id_part}] '

                label_part = (
                    '['
                    + str(parts[1].split('|')[1])
                    .replace('[', '')
                    .replace("'", '')
                    .replace(',', '')
                    + ']: '
                )
                report_part = ' '.join(parts[2:])
                styled_report.append(
                    {
                        'date_part': date_part,
                        'id_part': id_part,
                        'label_part': label_part,
                        'report_part': report_part,
                    },
                )
            else:
                styled_report.append(line)

    return styled_report


def get_complete_steps_uuid(node):
    """Retrieves UUIDs and details of completed steps in a workchain."""
    node: orm.WorkChainNode = orm.load_node(node)
    children = node.called
    completed_iterations_it = []
    exit_statuses = []
    completed_iterations_uuid = []
    for child in children:
        if child.is_finished_ok:
            completed_iterations_uuid.append(child.uuid)
            completed_iterations_it.append(child.inputs.al_loop_iteration.value)
            exit_statuses.append(child.exit_status)

    if not completed_iterations_it:
        completed_iterations_it = [0]
    return (
        children,
        completed_iterations_uuid,
        max(completed_iterations_it),
        exit_statuses,
    )


def get_missing_cache_steps_uuid(node, cache: pd.DataFrame):
    """Identifies workchain steps that are not yet in the cache."""
    missing_uuid_list = []
    children = orm.load_node(node).called
    children = [
        child
        for child in children
        if child.process_label
        in ('ActiveLearningWorkChain', 'SimpleActiveLearningWorkChain')
    ]

    for child in children:
        if child.uuid not in cache['uuid'].values:
            missing_uuid_list.append(child.uuid)
        if child.uuid in cache['uuid'].values:
            target_row = cache[cache['uuid'] == child.uuid]
            if target_row['progbar_class_name'].values[0] == 'workchain-progbar':
                missing_uuid_list.append(child.uuid)

    return missing_uuid_list


def get_step_child_info(node):
    """Gathers information about the direct children of a given node."""
    children_info = []
    children = node.called
    if len(children) > 0:
        for child in children:
            if isinstance(child, orm.CalcJobNode):
                if child.exit_status == 0:
                    exit_status = '󰱒'  # Success icon
                elif child.exit_status is None:
                    exit_status = ''  # Running icon
                else:
                    exit_status = ''  # Error icon

                child_dict = {
                    'exit_status_str': str(exit_status),
                    'exit_status': child.exit_status,
                    'child_pk': child.pk,
                    'child_process_label': child.process_label,
                }
                children_info.append(
                    {
                        'children': child_dict,
                        'curr_it': node.inputs.al_loop_iteration.value + 1,
                    }
                )

    return children_info


def get_model_stats(node):
    """Extracts model training statistics (RMSE) if available."""
    has_model = False
    rmse_e, rmse_f = None, None
    for child in node.called:
        if child.process_label == 'TrainMACEModelCalculation':
            try:
                rmse_e = child.outputs.m_rmse_e.value
                rmse_f = child.outputs.m_rmse_f.value
                has_model = True
            except AttributeError:
                pass

    if has_model:
        model_stats = {'energy': rmse_e, 'forces': rmse_f}
    else:
        model_stats = {'energy': None, 'forces': None}
    return model_stats


def get_progbar_class_name(node):
    """Determines the CSS class for a progress bar based on node state."""
    if node.is_finished_ok:
        class_name = 'workchain-progbar-done'
    elif node.is_excepted or node.is_failed or node.is_killed:
        class_name = 'workchain-progbar-error'
    else:
        class_name = 'workchain-progbar'
    return class_name


def update_cache(
    cache: pd.DataFrame, missing_uuid_list: list, db_size_dict: dict
) -> pd.DataFrame:
    """Updates the cache DataFrame with information from new UUIDs."""
    # The attributes must be managed within the loop because pd.concat creates
    # a new DataFrame object, discarding the attributes of the old one.
    for uuid in missing_uuid_list:
        # Preserve attributes before any potential modification
        original_attrs = cache.attrs.copy()

        curr_calc = orm.load_node(uuid)
        child_info_list = get_step_child_info(curr_calc)
        train_info_dict = get_model_stats(curr_calc)
        progbar_class = get_progbar_class_name(curr_calc)

        step_num = curr_calc.inputs.al_loop_iteration.value

        cache_df_row = pd.DataFrame(
            [
                {
                    'step': step_num + 1,
                    'pk': curr_calc.pk,
                    'uuid': uuid,
                    'children': child_info_list,
                    'training': train_info_dict,
                    'progbar_class_name': progbar_class,
                    'exit_status': curr_calc.exit_status,
                    'train_db_size': db_size_dict.get(step_num + 1, {}).get(
                        'train_db_size'
                    ),
                    'seed_db_size': db_size_dict.get(step_num + 1, {}).get(
                        'seed_db_size'
                    ),
                }
            ]
        )

        if uuid not in cache['uuid'].values:
            # Reassigning `cache` here loses attributes.
            cache = pd.concat([cache, cache_df_row], ignore_index=True)
            # Restore attributes immediately to the new DataFrame object.
            cache.attrs = original_attrs
        else:
            # In-place update does not lose attributes.
            idx = cache.index[cache['uuid'] == uuid][0]
            cache.loc[idx] = cache_df_row.loc[0]

    # At this point, `cache` is guaranteed to have its attributes.
    al_loop_steps = [
        c
        for c in orm.load_node(cache.attrs['base_workchain']).called
        if c.process_label
        in ('ActiveLearningWorkChain', 'SimpleActiveLearningWorkChain')
    ]
    if al_loop_steps:
        cache.attrs['curr_iter'] = al_loop_steps[-1].inputs.al_loop_iteration.value + 1

    return cache


def create_cache(workchain_node_id):
    """Initializes a new, empty cache DataFrame."""
    cache = pd.DataFrame(
        columns=[
            'step',
            'pk',
            'uuid',
            'children',
            'training',
            'progbar_class_name',
            'exit_status',
            'train_db_size',
            'seed_db_size',
        ]
    )

    cache.attrs['base_workchain'] = workchain_node_id
    base_workchain = orm.load_node(workchain_node_id)
    try:
        max_iters = base_workchain.inputs.active_learning.max_iterations.value
    except AttributeError:
        # Fallback
        max_iters = base_workchain.inputs.max_iterations.value
    cache.attrs['max_iters'] = max_iters

    cache.attrs['curr_iter'] = 0
    return cache


def gather_information(workchain_node_id, app):
    """Main function to gather all dashboard information, using a cache file."""
    app.logger.info(f"Checking workchain '{workchain_node_id}'.")

    cache_path = '/tmp'
    cache_filename = f'mdb_cache_{workchain_node_id}.pkl'
    cache_full_path = pl.Path(cache_path) / cache_filename

    wkc = orm.load_node(workchain_node_id)
    t_ini = time.time_ns()
    report = get_report(wkc)
    app.logger.info(f'Report gather time: {(time.time_ns() - t_ini) * 1e-9:.1f} s')

    db_size_dict = {}
    for line in report:
        db_size_line = line['report_part'].split()
        if 'seed_gen_db' in db_size_line:
            db_size_dict[int(db_size_line[1].replace(':', ''))] = {
                'seed_db_size': int(db_size_line[3].replace(',', '')),
                'train_db_size': int(db_size_line[5].replace(',', '')),
            }

    if cache_full_path.exists():
        try:
            cache = pd.read_pickle(cache_full_path)
            # Check for stale cache file from older versions that didn't save attributes
            if 'base_workchain' not in cache.attrs:
                app.logger.warning(
                    'Cache file is stale (missing attributes). Recreating cache.'
                )
                cache = create_cache(workchain_node_id)
            else:
                app.logger.info(f"Read cache file: '{cache_full_path}'")
        except (EOFError, pd.errors.EmptyDataError):
            cache = create_cache(workchain_node_id)
            app.logger.warning('Cache file was corrupted or empty. Created new cache.')
    else:
        cache = create_cache(workchain_node_id)
        app.logger.info(f"Created new cache file: '{cache_full_path}'")

    missing_uuid_list = get_missing_cache_steps_uuid(workchain_node_id, cache=cache)
    app.logger.info(f'Missing processes in cache: {missing_uuid_list}')

    if missing_uuid_list:
        cache = update_cache(cache, missing_uuid_list, db_size_dict)
        # Always save the cache after updating it
        cache.to_pickle(cache_full_path)
        app.logger.info(f"Updated cache file: '{cache_full_path}'")
    else:
        app.logger.info('Cache file up-to-date.')

    curr_iter = cache.attrs.get('curr_iter', 0)
    max_iters = cache.attrs.get('max_iters', 0)
    progbar_iter = max(curr_iter - 1, 0)

    if not wkc.is_finished:
        # Check the last running step
        if not cache.empty and cache.iloc[-1]['exit_status'] is None:
            progbar_iter = max(curr_iter - 1, 0)
        else:
            progbar_iter = curr_iter

    iter_text = f'{progbar_iter} / {max_iters}'

    # Get model stats
    model_stats_str = 'No model available yet.'
    if not cache.empty:
        # Try to find the latest valid training stats
        for i in range(len(cache) - 1, -1, -1):
            training_info = cache.iloc[i]['training']
            if training_info and training_info.get('energy') is not None:
                model_stats_str = (
                    f'RMSE E: {training_info["energy"]:.2f} meV/at - '
                    f'RMSE F: {training_info["forces"]:.2f} meV/A '
                    f'(from step: {cache.iloc[i]["step"]}).'
                )
                break

    if any([wkc.is_excepted, wkc.is_killed]):
        progbar_class_name = 'workchain-progbar-error'
        iter_text = f'ERROR ({curr_iter})'
        progbar_iter = max_iters  # Fill the bar on error
    elif wkc.is_finished_ok:
        progbar_class_name = 'workchain-progbar-done'
        iter_text = f'DONE ({curr_iter})'
        progbar_iter = max_iters
    else:
        progbar_class_name = 'workchain-progbar'

    app.logger.info(f"Active learning loop currently in iteration: '{curr_iter}'.")
    app.logger.info(f"Current iteration training stats: '{model_stats_str}'.")

    return (
        model_stats_str,
        report,
        iter_text,
        progbar_iter,
        cache,
        curr_iter,
        progbar_class_name,
    )


def _get_subprocess_list(cache: pd.DataFrame) -> list:
    """
    Extracts and flattens the subprocess list from the cache.
    This is used for the Fancytree UI component.
    """
    subprocess_list = []
    if not cache.empty and 'children' in cache.columns:
        # The 'children' column contains lists of dictionaries. We flatten them.
        for item_list in cache['children'].dropna():
            for item in item_list:
                if 'children' in item and 'child_process_label' in item['children']:
                    subprocess_list.append(item['children'])
    return subprocess_list


def run_training_dashboard(workchain_node_id, refresh_interval=60, port=8000):
    """Sets up and runs the Flask application for the dashboard."""
    app = Flask(__name__)
    load_profile()

    @app.route('/')
    @app.route('/status')
    def training_dashboard():
        """Renders the main status page of the dashboard."""
        (
            model_stats,
            report,
            iter_text,
            progbar_iter,
            cache,
            curr_iter,
            progbar_class_name,
        ) = gather_information(workchain_node_id, app)

        # The main part of the training_dashboard.html template expects a nested
        # list of subprocesses (a list of steps, where each step is a list of children).
        # This structure is taken directly from the 'children' column of the cache.
        subprocess_list_nested = []
        if not cache.empty and 'children' in cache.columns:
            subprocess_list_nested = cache['children'].dropna().tolist()

        return render_template(
            'training_dashboard.html',
            refresh_interval=refresh_interval,
            update_time=time.strftime('%H:%M:%S'),
            workchain_node_id=workchain_node_id,
            model_stats=model_stats,
            report=report,
            iter_text=iter_text,
            max_iters=cache.attrs.get('max_iters', 0),
            curr_iter=curr_iter,
            progbar_iter=progbar_iter,
            progbar_class_name=progbar_class_name,
            subprocess_list=subprocess_list_nested,  # Pass the correct nested list
            cache=cache,  # The template might still use the full cache object
        )

    @app.route('/results')
    def results_dashboard():
        """Renders the results page with graphs."""
        (
            model_stats,
            report,
            iter_text,
            progbar_iter,
            cache,
            curr_iter,
            progbar_class_name,
        ) = gather_information(workchain_node_id, app)

        return render_template(
            'results_dashboard.html',
            refresh_interval=refresh_interval,
            update_time=time.strftime('%H:%M:%S'),
            workchain_node_id=workchain_node_id,
            model_stats=model_stats,
            report='',  # Report not needed for results page
            iter_text=iter_text,
            max_iters=cache.attrs.get('max_iters', 0),
            curr_iter=curr_iter,
            progbar_iter=progbar_iter,
            progbar_class_name=progbar_class_name,
            subprocess_list=_get_subprocess_list(
                cache
            ),  # The flat list is correct for the fancytree
            cache=cache.to_json(orient='split'),
        )

    return app
