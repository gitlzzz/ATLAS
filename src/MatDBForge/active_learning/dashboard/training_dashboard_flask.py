"""Dashboard for monitoring and reviewing training Active Learning WorkChains."""

import pathlib as pl
import re
import time

import pandas as pd
from aiida import load_profile, orm
from aiida.cmdline.utils.common import get_workchain_report
from flask import Flask, render_template


def get_report(node):
    if isinstance(node, str):
        node = orm.load_node(node)

    report = get_workchain_report(node, levelname="REPORT")
    report_lines = report.split("\n")
    pattern = r"\[\d+\|[A-Za-z]+\|[A-Za-z_]+\]"

    styled_report = []
    for line_idx, line in enumerate(report_lines):
        # Catch error strings until the next aiida report
        if "|on_except]:" in line:
            report_part = []
            cnt = line_idx + 1

            parts = line.split("]: ")
            if len(parts) > 1:
                date_part = parts[0].split(" ")[0]
                date_part += " " + parts[0].split(" ")[1]
                id_part = str(parts[1].split("|")[0]).replace("[", "")
                id_part = id_part.replace(" ", "")
                id_part = f" [{id_part}] "

                label_part = (
                    "["
                    + str(parts[1].split("|")[1])
                    .replace("[", "")
                    .replace("'", "")
                    .replace(",", "")
                    + "]: "
                )

            while not re.search(pattern, report_lines[cnt]):
                report_part.append(report_lines[cnt])
                cnt += 1

            report_part = "\n".join(report_part)
            styled_report.append(
                {
                    "date_part": date_part,
                    "id_part": id_part,
                    "label_part": "[ERROR]",
                    "report_part": report_part,
                }
            )
        elif re.search(pattern, line):
            parts = line.split("]: ")
            if len(parts) > 1:
                date_part = parts[0].split(" ")[0]
                date_part += " " + parts[0].split(" ")[1]
                id_part = str(parts[1].split("|")[0]).replace("[", "")
                id_part = id_part.replace(" ", "")
                id_part = f" [{id_part}] "

                label_part = (
                    "["
                    + str(parts[1].split("|")[1])
                    .replace("[", "")
                    .replace("'", "")
                    .replace(",", "")
                    + "]: "
                )
                report_part = " ".join(parts[2:])
                styled_report.append(
                    {
                        "date_part": date_part,
                        "id_part": id_part,
                        "label_part": label_part,
                        "report_part": report_part,
                    },
                )
            else:
                styled_report.append(line)

    return styled_report


def get_complete_steps_uuid(node):
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
    missing_uuid_list = []
    children = orm.load_node(node).called
    children = [
        child for child in children if child.process_label == "ActiveLearningWorkChain"
    ]

    for child in children:
        if child.uuid not in cache["uuid"].values:
            missing_uuid_list.append(child.uuid)
        if child.uuid in cache["uuid"].values:
            target_row = cache[cache["uuid"] == child.uuid]
            if target_row["progbar_class_name"].values[0] == "workchain-progbar":
                missing_uuid_list.append(child.uuid)

    return missing_uuid_list


def get_step_child_info(node):
    children_info = []
    children = node.called
    if len(children) > 0:
        for child in children:
            if isinstance(child, orm.CalcJobNode):
                if child.exit_status == 0:
                    exit_status = "󰱒"
                elif not child.exit_status:
                    exit_status = ""
                else:
                    exit_status = child.exit_status

                child_dict = {
                    "exit_status_str": str(exit_status),
                    "exit_status": child.exit_status,
                    "child_pk": child.pk,
                    "child_process_label": child.process_label,
                }
                children_info.append(
                    {
                        "children": child_dict,
                        "curr_it": node.inputs.al_loop_iteration.value + 1,
                    }
                )

    return children_info


def get_model_stats(node):
    has_model = False
    for child in node.called:
        if child.process_label == "TrainMACEModelCalculation":
            try:
                rmse_e = child.outputs.m_rmse_e.value
                rmse_f = child.outputs.m_rmse_f.value
                has_model = True
            except Exception:
                pass

    if has_model:
        model_stats = {"energy": rmse_e, "forces": rmse_f}
    else:
        model_stats = {"energy": None, "forces": None}
    return model_stats


def get_progbar_class_name(node):
    if node.is_finished:
        # Filling up and restyling the progress bar.
        class_name = "workchain-progbar-done"
    elif node.is_excepted or node.is_failed:
        class_name = "workchain-progbar-error"
    else:
        class_name = "workchain-progbar"

    return class_name


def update_cache(cache: pd.DataFrame, missing_uuid_list) -> pd.DataFrame:
    for uuid in missing_uuid_list:
        # Get information from uuid
        curr_calc = orm.load_node(uuid)

        # Add information to cache
        child_info_list = get_step_child_info(curr_calc)
        train_info_dict = get_model_stats(curr_calc)

        progbar_class = get_progbar_class_name(curr_calc)
        cache_df_row = pd.Series(
            data={
                "step": curr_calc.inputs.al_loop_iteration.value + 1,
                "pk": curr_calc.pk,
                "uuid": uuid,
                "children": child_info_list,
                "training": train_info_dict,
                "progbar_class_name": progbar_class,
                "exit_status": curr_calc.exit_status,
            },
        )
        cache_df_row = cache_df_row.to_frame().T
        cache_df_row.attrs = cache.attrs

        if len(cache) > 0:
            cache_df_row.columns = cache.columns

            # If entry not in cache, add it
            if uuid not in cache["uuid"].values:
                # Add dict to dataframe
                cache = pd.concat(
                    [cache, cache_df_row],
                    ignore_index=True,
                    keys=cache.columns,
                )

            # UUID already in cache. Updating entry.
            else:
                target_row = cache[cache["uuid"] == uuid]
                for col in cache_df_row.columns:
                    target_row[col] = cache_df_row[col].values
                cache[cache["uuid"] == uuid] = target_row

        # First entry in cache
        else:
            # Add dict to dataframe
            cache = pd.concat(
                [cache, cache_df_row],
                ignore_index=True,
                keys=cache.columns,
            )

    # Gathering AL steps
    al_loop_steps = [
        c
        for c in orm.load_node(cache.attrs["base_workchain"]).called
        if c.process_label == "ActiveLearningWorkChain"
    ]
    # Update extra information
    cache.attrs["curr_iter"] = al_loop_steps[-1].inputs.al_loop_iteration.value + 1

    return cache


def create_cache(workchain_node_id):
    cache = pd.DataFrame(
        columns=[
            "step",
            "pk",
            "uuid",
            "children",
            "training",
            "progbar_class_name",
            "exit_status",
        ]
    )

    # Adding extra information
    cache.attrs["base_workchain"] = workchain_node_id

    base_workchain = orm.load_node(workchain_node_id)
    cache.attrs["max_iters"] = (
        base_workchain.inputs.active_learning.max_iterations.value
    )

    # Gathering AL steps
    al_loop_steps = [
        c for c in base_workchain.called if c.process_label == "ActiveLearningWorkChain"
    ]
    cache.attrs["curr_iter"] = al_loop_steps[-1].inputs.al_loop_iteration.value

    return cache


def gather_information(workchain_node_id, app):
    cache_path = "/tmp"
    cache_filename = f"mdb_cache_{workchain_node_id}.pkl"
    cache_full_path = pl.Path(cache_path) / cache_filename

    if cache_full_path.exists():
        # Load cache file
        cache = pd.read_pickle(cache_full_path)
        app.logger.info(f"Read cache file: '{cache_full_path}'")
    else:
        # Create cache  as a dataframe
        cache = create_cache(workchain_node_id)
        app.logger.info(f"Created new cache file: '{cache_full_path}'")

    # Get AL steps not found in cache file
    missing_uuid_list = get_missing_cache_steps_uuid(
        workchain_node_id,
        cache=cache,
    )
    app.logger.info(f"Missing processes in cache: {missing_uuid_list}")

    if len(missing_uuid_list) > 0:
        # Save information into cache df
        cache = update_cache(cache, missing_uuid_list)

        # Save df as pickle file
        cache.to_pickle(cache_full_path)
        app.logger.info(f"Updated cache file: '{cache_full_path}'")
    else:
        app.logger.info("Cache file up-to-date.")

    # Get status for the current iteration from cache
    curr_step_status = cache.iloc[-1]["exit_status"]
    curr_iter = cache.attrs["curr_iter"]

    # If current step is finished, get current iteration
    if curr_step_status:
        progbar_iter = cache.attrs["curr_iter"]
    # If current step is not finished, get previous iteration
    else:
        progbar_iter = cache.attrs["curr_iter"] - 1

    iter_text = f"{progbar_iter} / {cache.attrs['max_iters']}"

    # Get model stats
    energy_avail = False
    if len(cache) > 0:
        # Getting model stats from current iteration
        if cache.iloc[curr_iter - 1]["training"]["energy"]:
            model_stats_dict = cache.iloc[curr_iter - 1]["training"]
            extra_text = f"from curr. step: {curr_iter}"
            energy_avail = True

        # If training not yet completed, get model stats from previous iteration
        elif len(cache) >= 2 and cache.iloc[curr_iter - 2]["training"]["energy"]:
            model_stats_dict = cache.iloc[curr_iter - 2]["training"]
            extra_text = f"from prev. step: {curr_iter-1}"
            energy_avail = True

    if energy_avail:
        model_stats = (
            f"RMSE E: {model_stats_dict['energy']:.2f} meV/at - "
            f"RMSE F: {model_stats_dict['forces']:.2f} meV/A ({extra_text})."
        )
    else:
        model_stats = "No model available yet."

    # Get report from AiiDA
    t_ini = time.time_ns()
    report = get_report(workchain_node_id)
    app.logger.info(f"Report gather time: {(time.time_ns() - t_ini) * 1e-9:.1f} s")

    # Changing progress bar style
    if orm.load_node(workchain_node_id).is_excepted:
        progbar_class_name = "workchain-progbar-error"
        iter_text = f"ERROR ({cache.attrs['curr_iter']})"
        curr_iter = cache.attrs["max_iters"]
        progbar_iter = cache.attrs["max_iters"]

    elif orm.load_node(workchain_node_id).is_finished:
        progbar_class_name = "workchain-progbar-done"
        iter_text = f"DONE ({cache.attrs['curr_iter']})"
        curr_iter = cache.attrs["max_iters"]
        progbar_iter = cache.attrs["max_iters"]

    else:
        progbar_class_name = "workchain-progbar"

    return (
        model_stats,
        report,
        iter_text,
        progbar_iter,
        cache,
        curr_iter,
        progbar_class_name,
    )


def run_training_dashboard(workchain_node_id, refresh_interval=60, port=8000):
    app = Flask(__name__)
    load_profile()

    @app.route("/")
    @app.route("/status")
    def training_dashboard():
        (
            model_stats,
            report,
            iter_text,
            progbar_iter,
            cache,
            curr_iter,
            progbar_class_name,
        ) = gather_information(workchain_node_id, app)
        print("cache: ", cache["exit_status"])
        print("cache: ", cache.columns)

        return render_template(
            "training_dashboard.html",
            refresh_interval=refresh_interval,
            update_time=time.strftime("%H:%M:%S"),
            workchain_node_id=workchain_node_id,
            model_stats=model_stats,
            report=report,
            iter_text=iter_text,
            max_iters=cache.attrs["max_iters"],
            curr_iter=curr_iter,
            progbar_iter=progbar_iter,
            progbar_class_name=progbar_class_name,
            subprocess_list=cache.children,
            cache=cache,
        )

    @app.route("/results")
    def results_dashboard():
        # return render_template("results_dashboard.html")

        model_stats, report, iter_text, cache, curr_iter, progbar_class_name = (
            gather_information(workchain_node_id, app)
        )

        return render_template(
            "results_dashboard.html",
            refresh_interval=refresh_interval,
            update_time=time.strftime("%H:%M:%S"),
            workchain_node_id=workchain_node_id,
            model_stats=model_stats,
            report="",
            iter_text=iter_text,
            max_iters=cache.attrs["max_iters"],
            curr_iter=curr_iter,
            progbar_class_name=progbar_class_name,
            subprocess_list=cache.children,
            cache=cache,
        )

    return app
