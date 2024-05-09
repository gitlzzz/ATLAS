from aiida import load_profile, orm
from aiida.cmdline.utils.common import get_workchain_report, get_calcjob_report
from dash import Dash, Input, Output, State, callback, dcc, html, ALL, MATCH
import dash_daq as daq


def run_training_dashboard(workchain_node_id, n_sec, port=8050):
    load_profile()

    @callback(
        Output("interval-general", "interval"),
        Output("interval-model-score", "interval"),
        Input("progress-update", "value"),
        Input("progress-update", "max"),
        State("interval-model-score", "disabled"),
        State("interval-general", "disabled"),
    )
    def toggle_interval(value, max_value, state_model, state_general):
        if value == max_value:
            return 100000, 100000
        else:
            return n_sec * 1000, n_sec * 1000

    # @callback(
    #     Output("progress-update", "className"),
    #     Input("progress-update", "value"),
    #     Input("progress-update", "max"),
    # )
    # def progressbar_done(value, max_value):
    #     if value == max_value:
    #         return "workchain-progbar-done"
    #     else:
    #         return "workchain-progbar"

    @callback(
        Output("model-score-text", "children"),
        Input("interval-model-score", "n_intervals"),
    )
    def update_dynamic_text(n):
        # Placeholder function to generate dynamic text

        node = orm.load_node(workchain_node_id)
        children = node.called_descendants

        result = []

        has_model = False
        for child in children:
            if child.process_label == "create_mace_lammps_model":
                has_model = True
                rmse_e = child.inputs.rmse_e.value
                rmse_f = child.inputs.rmse_f.value

        if has_model:
            result.append(
                html.P(f"RMSE E: {rmse_e:.2f} meV/at - RMSE F: {rmse_f:.2f} meV/at"),
            )
        else:
            result.append("No model available yet.")

        return result

    @callback(
        Output("subprocess-list", "children"),
        Input("interval-model-score", "n_intervals"),
    )
    def display_subprocesses(n):
        node = orm.load_node(workchain_node_id)
        children = node.called_descendants

        result = []

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
                    button = html.Button(
                        child_str,
                        id=f"button-{child.pk}",
                        type="calcjob-button",
                        value=str(child.pk),
                        n_clicks=0,
                    )
                    result.append(button)

        else:
            result.append("No children spawned yet.")

        return result

    # @callback(
    #     Output(
    #         {"id": "live-update-text", "n_clicks": MATCH},
    #         "children",
    #         allow_duplicate=True,
    #     ),
    #     Input(
    #         {"type": "calcjob-button", "n_clicks": MATCH},
    #         "n_clicks",
    #     ),
    #     State({"type": "calcjob-button", "n_clicks": MATCH}, "value"),
    #     # Input({"id": "ALL", "type": ALL}, "value"),
    #     prevent_initial_call=True,
    # )
    # def display_calcjob(n_clicks, value):
    #     print("n_clicks: ", n_clicks)
    #     print("value: ", value)
    #     # [ ] Stop right column from updating
    #     # [ ] Reenable right column update
    #     # node = orm.load_node(value)
    #     # report = get_calcjob_report(node)
    #     # print("report: ", report)
    #     # return report
    #     print(n_clicks)

    @callback(
        Output("live-update-text", "children", allow_duplicate=True),
        Output("progress-update", "max"),
        Output("progress-update", "value"),
        Output("iter-text", "children"),
        Output("progress-update", "className"),
        Input("interval-general", "n_intervals"),
        prevent_initial_call="initial_duplicate",
    )
    def update_metrics(n):
        node = orm.load_node(workchain_node_id)
        report = get_workchain_report(node, levelname="REPORT")

        styled_report = []
        for line in report.split("\n"):
            parts = line.split("]: ")
            if len(parts) > 1:
                date_part = parts[0].split(" ")[0]
                date_part += " " + parts[0].split(" ")[1]
                id_part = str(parts[1].split("|")[0]).replace("[", "")
                num_spaces = id_part.count(" ")
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
                report_part = parts[2]
                styled_report.append(
                    html.P(
                        [
                            html.Span(
                                date_part,
                                className="date-part",
                                style={
                                    "padding-left": f"{num_spaces//2}em",
                                },
                            ),
                            html.Span(
                                id_part,
                                className="id-part",
                            ),
                            html.Span(
                                label_part,
                                className="label-part",
                            ),
                            html.Span(
                                report_part,
                                className="report-text",
                            ),
                        ],
                    )
                )
            else:
                styled_report.append(html.P(line))

        max_iters = str(node.inputs.active_learning.max_iterations.value)
        class_name = "workchain-progbar"

        if node.is_finished:
            # Filling up and restyling the progress bar.
            curr_iter = max_iters
            class_name = "workchain-progbar-done"
        if node.is_excepted or node.is_failed:
            curr_iter = max_iters
            class_name = "workchain-progbar-error"
            iter_text = f"ERROR"
        else:
            children = [
                child
                for child in node.called_descendants
                if child.process_label == "ActiveLearningWorkChain"
            ]
            if len(children) > 0:
                curr_iter = str(children[-1].inputs.al_loop_iteration.value + 1)
            else:
                curr_iter = 0

            iter_text = f"{curr_iter} / {max_iters}"

        return (
            styled_report,
            max_iters,
            curr_iter,
            iter_text,
            class_name,
        )

    app = Dash(__name__)
    app.title = "MDB AL Loop"
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1(
                        children="Active Learning Loop",
                        className="title",
                    ),
                    html.Img(src="./assets/logo_dark.png", className="logo-left"),
                    html.Img(
                        src="./assets/toyoshima_logo_alt.svg", className="logo-right"
                    ),
                ],
                className="header",
            ),
            # New top box across both columns, now split into two sections
            html.Div(
                [
                    # Left section (80%)
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        f"Target workchain - {workchain_node_id}",
                                        className="title-upper",
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                "Total progress:",
                                                className="top-text",
                                            ),
                                            html.Progress(
                                                className="workchain-progbar",
                                                id="progress-update",
                                            ),
                                            html.P(
                                                "1 / ?",
                                                id="iter-text",
                                                className="top-text",
                                            ),
                                        ],
                                        className="workchain-progress-container",
                                    ),
                                ],
                                className="top-box-left-headers",
                            ),
                        ],
                        className="top-box left-box",
                    ),
                    # Right section (20%) for dynamic text
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        "Current iteration perfomance",
                                        className="title-upper",
                                    ),
                                    html.P(
                                        "No model available yet.",
                                        id="model-score-text",
                                        className="top-text",
                                    ),
                                ],
                                className="top-box-left-headers",
                            ),
                        ],
                        className="top-box right-box",
                    ),
                ],
                className="top-box",
            ),
            # Container Div for bottom columns
            html.Div(
                [
                    # Left Column
                    html.Div(
                        [
                            # Empty box with black border
                            html.Div(id="subprocess-list", className="subprocess-list")
                        ],
                        className="column left-column",
                    ),
                    # Right Column
                    html.Div(
                        [
                            # Box containing text and with a black border
                            html.Div(
                                children="Loading...",
                                id="live-update-text",
                                className="live-text",
                            ),
                        ],
                        className="column right-column",
                    ),
                ],
                className="column-container",
            ),
            dcc.Interval(
                id="interval-general",
                interval=n_sec * 1000,  # in milliseconds
                n_intervals=0,
            ),
            dcc.Interval(
                id="interval-model-score",
                interval=n_sec * 1000 * 2,  # in milliseconds
                n_intervals=0,
            ),
        ]
    )

    app.run(debug=True, port=port)
