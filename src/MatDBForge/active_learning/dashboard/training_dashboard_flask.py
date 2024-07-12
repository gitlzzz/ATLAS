import re
import time

from aiida import load_profile, orm
from aiida.cmdline.utils.common import get_workchain_report
from flask import Flask, render_template


def gather_information(workchain_node_id):
    node: orm.WorkChainNode = orm.load_node(workchain_node_id)
    children = node.called_descendants

    model_stats = get_model_stats(children)
    report = get_report(node)
    iter_text, progbar_class_name, max_iters, curr_iter = get_iteration_info(node)
    subprocess_list = display_subprocesses(children)

    return (
        model_stats,
        report,
        iter_text,
        progbar_class_name,
        max_iters,
        curr_iter,
        subprocess_list,
    )


def get_model_stats(children):
    has_model = False
    for child in children:
        if child.process_label == "create_mace_lammps_model":
            try:
                rmse_e = child.inputs.rmse_e.value
                rmse_f = child.inputs.rmse_f.value
                has_model = True
            except Exception:
                pass

    if has_model:
        model_stats = f"RMSE E: {rmse_e:.2f} meV/at - RMSE F: {rmse_f:.2f} meV/A"
    else:
        model_stats = "No model available yet."
    return model_stats


def get_iteration_info(node):
    max_iters = str(node.inputs.active_learning.max_iterations.value)
    class_name = "workchain-progbar"
    iter_text = "1 / ?"

    if node.is_finished:
        # Filling up and restyling the progress bar.
        curr_iter = max_iters
        class_name = "workchain-progbar-done"
        iter_text = f"{curr_iter} / {max_iters}"
    elif node.is_excepted or node.is_failed:
        curr_iter = max_iters
        class_name = "workchain-progbar-error"
        iter_text = "ERROR"
    else:
        children = [
            child
            for child in node.called_descendants
            if child.process_label == "ActiveLearningWorkChain"
        ]
        if len(children) > 0:
            curr_iter = children[-1].inputs.al_loop_iteration.value
        else:
            curr_iter = 0

        iter_text = f"{curr_iter} / {max_iters}"
    return iter_text, class_name, max_iters, curr_iter


def display_subprocesses(children):
    subprocesses = []

    if len(children) > 0:
        for child in children:
            if isinstance(child, orm.CalcJobNode):
                if child.exit_status == 0:
                    exit_status = "󰱒"
                elif not child.exit_status:
                    exit_status = ""
                else:
                    exit_status = child.exit_status

                child_str = f"{exit_status} - ({child.pk}) {child.process_label}"
                button = {
                    "child_str": child_str,
                    "id": f"button-{child.pk}",
                    "type": "calcjob-button",
                    "value": str(child.pk),
                    "n_clicks": 0,
                }
                subprocesses.append(button)

    else:
        subprocesses.append("No children spawned yet.")

    return subprocesses


def get_report(node):
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


def run_training_dashboard(workchain_node_id, refresh_interval=60, port=8000):
    app = Flask(__name__)
    load_profile()

    @app.route("/")
    @app.route("/status")
    def training_dashboard():
        # workchain_node_id = 1388283
        (
            model_stats,
            report,
            iter_text,
            progbar_class_name,
            max_iters,
            curr_iter,
            subprocess_list,
        ) = gather_information(workchain_node_id)
        return render_template(
            "training_dashboard.html",
            refresh_interval=refresh_interval,
            update_time=time.strftime("%H:%M:%S"),
            workchain_node_id=workchain_node_id,
            model_stats=model_stats,
            report=report,
            iter_text=iter_text,
            max_iters=max_iters,
            curr_iter=curr_iter,
            progbar_class_name=progbar_class_name,
            subprocess_list=subprocess_list,
        )

    @app.route("/results")
    def results_dashboard():
        # return render_template("results_dashboard.html")

        # REMOVE
        (
            model_stats,
            report,
            iter_text,
            progbar_class_name,
            max_iters,
            curr_iter,
            subprocess_list,
        ) = gather_information(workchain_node_id)
        return render_template(
            "results_dashboard.html",
            refresh_interval=refresh_interval,
            update_time=time.strftime("%H:%M:%S"),
            workchain_node_id=workchain_node_id,
            model_stats=model_stats,
            report=report,
            iter_text=iter_text,
            max_iters=max_iters,
            curr_iter=curr_iter,
            progbar_class_name=progbar_class_name,
            subprocess_list=subprocess_list,
        )


    return app
