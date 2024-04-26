from aiida import load_profile, orm
from aiida.cmdline.utils.common import get_workchain_report
from dash import Dash, Input, Output, State, callback, dcc, html


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
            return 10000, 10000
        else:
            return n_sec, n_sec

    @callback(
        Output("progress-update", "className"),
        Input("progress-update", "value"),
        Input("progress-update", "max"),
    )
    def progressbar_done(value, max_value):
        if value == max_value:
            return "workchain-progbar-done"
        else:
            return "workchain-progbar"

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
        Output("live-update-text", "children"),
        Output("progress-update", "max"),
        Output("progress-update", "value"),
        Input("interval-general", "n_intervals"),
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
                report_part = parts[2:]
                styled_report.append(
                    html.P(
                        [
                            html.Span(
                                date_part + " ",
                                className="date-part",
                            ),
                            html.Span(
                                parts[1] + "]: ",
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

        max_iters = str(node.inputs.max_iterations.value)

        if node.is_finished:
            # Filling up and restyling the progress bar.
            curr_iter = max_iters
        else:
            children = [
                child
                for child in node.called_descendants
                if child.process_label == "ActiveLearningWorkChain"
            ]
            curr_iter = str(children[-1].inputs.al_loop_iteration.value + 1)

        return styled_report, max_iters, curr_iter

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
                            # Your empty box with black border here
                            html.Div()
                        ],
                        className="column left-column",
                    ),
                    # Right Column
                    html.Div(
                        [
                            # Box containing text and with a black border
                            html.Div(
                                id="live-update-text",
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

    app.run(debug=False, port=port)
