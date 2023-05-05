import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

CWD = os.path.abspath(
    # "/home/psanz/teklahome/tests_dir_4c898d1cba504efc860ab75a6dc20919"
    # "/home/psanz/teklahome/projects/p2-CuZn/initial_tests/tests_dir_4678bcb9da984d13a824a4426f3a66c8_test_batch_2"
    # "/home/psanz/teklahome/projects/p2-CuZn/initial_tests/test_batch_4/kpoints_test_conventional/kpoints_sp_conventional"
    "/home/psanz/teklahome/projects/p2-CuZn/initial_tests/final_tests_plot"
)

axis_tick_scale = 0

gruvbox = [
    "#3c3836",
    "#cc241d",
    "#98971a",
    "#d79921",
    "#458588",
    "#b16286",
    "#689d6a",
    "#7c6f64",
]

res_dict = defaultdict(lambda: defaultdict(dict))
test_types = [
    "kpoints_newstructures",
    "ispin",
    "encut",
    "kpoints_conv_cell",
    "kpoints_old",
    "kpoints_pot_Zn_pv_2005",
    "kpoints_pot_Zn_2000",
    "kpoints_encut_500",
]


# df = pd.read_pickle(os.path.join(CWD, "result_df_test007.pkl"))
df = pd.read_pickle(os.path.join(CWD, "result_df_test007_new.pkl"))

phase_list = df["phase"].values
extend_phase_list = ["all"]
extend_phase_list.extend(phase_list)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.H2(children="DFT Benchmarks"),
        html.Div(
            children=(
                html.Div(
                    children=(
                        "Select a test",
                        dcc.Dropdown(
                            options=test_types,
                            value=test_types[0],
                            id="test-type",
                            placeholder="Select a test type",
                            style={
                                "width": "30vw",
                            },
                        ),
                    ),
                ),
                html.Div(style=dict(width="2vw")),
                html.Div(
                    children=(
                        "Select a phase",
                        dcc.Dropdown(
                            options=extend_phase_list,
                            value=extend_phase_list[0],
                            id="phase-type-sel",
                            placeholder="Select a phase",
                            style={
                                "width": "30vw",
                            },
                        ),
                        dcc.Checklist(
                            options=[
                                {"label": "Enable text labels", "value": "text"},
                                {
                                    "label": "Enable Reference Datapoint",
                                    "value": "reference",
                                },
                            ],
                            value=["text", "reference"],
                            id="enable-ref-datapoint",
                        ),
                    )
                ),
                html.Div(style=dict(width="2vw")),
                html.Div(
                    children=(
                        # "Current Scale",
                        # html.Div(""),
                        dcc.RadioItems(
                            id="scale-selector",
                            options=[
                                {"label": "1 meV", "value": "1"},
                                {"label": "2 meV", "value": "2"},
                                {"label": "5 meV", "value": "5"},
                                {"label": "10 meV", "value": "10"},
                                {"label": "100 meV", "value": "100"},
                            ],
                            value="1",
                            inline=True,
                        ),
                        html.Div(id="scale-indicator"),
                        html.Div(id="distance-indicator"),
                    ),
                    style={"margin-top": "0.75cm"},
                ),
            ),
            style={"display": "flex", "horizontal_align": "top"},
        ),
        dcc.Graph(
            id="vasp-graph",
            style={
                "width": "98vw",
                "height": "80vh",
            },
            config={
                "toImageButtonOptions": {
                    "format": "svg",  # one of png, svg, jpeg, webp
                    "filename": "custom_image",
                    "width": 1280,
                    "height": 720,
                    "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
                }
            },
        ),
    ],
)


@app.callback(
    Output("vasp-graph", "figure"),
    Output("scale-indicator", "children"),
    Input("test-type", "value"),
    Input("phase-type-sel", "value"),
    Input("enable-ref-datapoint", "value"),
    Input("scale-selector", "value"),
)
def update_figure(selected_test, selected_phase, enable_ref_datapoint, scale_selector):
    print('selected_test: ', selected_test)
    # Searching for matching tests inside the dataframe
    matching_tests_kpt = [
        col
        for col in df.columns
        if selected_test in col and col.endswith("kpt_density")
    ]
    # Matching only columns with the energy per atom.
    # TODO: This conditional is a little bit sketchy. Make it more robust?
    matching_tests = [
        # _kpt_
        # col for col in df.columns if selected_test in col and (col[-1].isdigit() or ispincol[5].isdigit())
        col
        for col in df.columns
        if selected_test in col and ("_kpt_" not in col)
    ]
    print("matching_tests: ", matching_tests)

    matching_vectors = [
        col for col in df.columns if selected_test in col and col.endswith("kpt_vector")
    ]

    # Checking if the reference datapoint is disabled
    if (
        "reference" not in enable_ref_datapoint
        and "kpoints_conv_cell_006" in matching_tests
    ):
        reference_ind = matching_tests.index("kpoints_conv_cell_006")
        matching_tests.pop(reference_ind)
        matching_tests_kpt.pop(reference_ind)
        matching_vectors.pop(reference_ind)

    # Chart settings if selected phase is 'all'
    if selected_phase == "all":
        fig = go.Figure(
            layout={
                "xaxis": {
                    "title": r"k-point spacing (2&#960;&#183;&#8491;<sup>-1</sup>)"
                },
                "yaxis": {"title": r"Energy per atom (eV)"},
            },
        )
        for idx, test_name in enumerate(matching_tests):
            x_vals = df[matching_tests_kpt[idx]].values
            y_vals = df[matching_tests[idx]].values

            fig.add_scatter(
                x=x_vals,
                y=y_vals,
                name=test_name,
                marker=dict(
                    size=14,
                    line_width=0,
                    opacity=1,
                    color=gruvbox,  # px.colors.qualitative.Prism
                ),
                line=dict(width=0),
                customdata=phase_list,
                hovertemplate="<b>Energy</b>: %{y:.8f}"
                + "<br><b>Kpt spacing</b>: %{x:.8f}<br>"
                + "<br>%{customdata}<br>",
            )
        change_ticks = False
        axis_tick_scale = "default"

    # Chart settings if any specific phase is selected
    else:
        # Creating empty lists for storing values later
        y_val_list = []
        x_val_list = []
        name_list = []
        kpt_vect_list = []

        # Creating figure with named axis
        fig = go.Figure(
            layout={
                "xaxis": {
                    "title": r"k-point spacing (2&#960;&#183;&#8491;<sup>-1</sup>)"
                },
                "yaxis": {"title": r"Energy per atom (eV)"},
            }
        )

        # Going over the matching tests to gather their values on the dataframe
        for idx, test_names in enumerate(
            zip(matching_tests, matching_tests_kpt, matching_vectors)
        ):
            test_name = test_names[0]
            kpt_dens_name = test_names[1]
            vec = test_names[2]

            # Omitting some undesired datapoints
            if "kpoints_old_conv" in test_names:
                continue

            if selected_phase == "m3_old" and "kpoints" not in test_names:
                continue

            # Gathering values for the two axis and the k-point vectors
            # the k-point vectors will be shown on the hover text
            x_val = df.loc[df["phase"] == selected_phase][kpt_dens_name].values
            y_val = df.loc[df["phase"] == selected_phase][test_name].values
            kpt_vector = df.loc[df["phase"] == selected_phase][vec].values

            # Adding the obtained values to the lists
            y_val_list.append(y_val[0])
            x_val_list.append(x_val[0])
            name_list.append(test_name)
            kpt_vect_list.append(str(list(kpt_vector)))

        # Converting value lists to arrays
        y_val_list = np.array(y_val_list)
        x_val_list = np.array(x_val_list)

        # Removing nan from the arrays, caused by missing points
        # from unfinished calculations
        y_val_list_nonan = y_val_list[~np.isnan(y_val_list)]
        x_val_list_nonan = x_val_list[~np.isnan(x_val_list)]

        # Getting minimum and maximum
        y_val_max = np.max(y_val_list_nonan)
        y_val_min = np.min(y_val_list_nonan)
        x_val_max = np.max(x_val_list_nonan)
        x_val_min = np.min(x_val_list_nonan)

        # Generating y axis ticks in intervals of 1000, to get
        # a scale of 'scale_selector' meV.
        # This value can be chosen with the selector on top of the chart.
        axis_tick_scale = int(scale_selector) / 1000
        ticks = np.arange(y_val_min, y_val_max, axis_tick_scale)

        # if selected_phase == "m3":
        #     if len(ticks) > 50:
        #         axis_tick_scale = 1 / 100

        ticks = np.arange(y_val_min, y_val_max, axis_tick_scale)
        change_ticks = True

        # Creating an array of extra data to be shown on the hoverlabel
        customdata_arr = np.stack((name_list, kpt_vect_list), axis=-1)

        if "text" in enable_ref_datapoint:
            mode_str = "markers+text"
        else:
            mode_str = "markers"

        # Plotting the actual graph
        fig.add_scatter(
            x=x_val_list,
            y=y_val_list,
            marker=dict(
                size=14,
                line_width=2,
                opacity=1,
                colorscale="Viridis",
                color=x_val_list_nonan,
            ),
            text=y_val_list,
            textposition="top center",
            textfont=dict(
                # family="Fira Code Bold",
                size=20,
                # color="#282828",
            ),
            texttemplate="<b>%{customdata[0]}:</b> %{y:.6f} eV",
            mode=mode_str,
            customdata=customdata_arr,
            hovertemplate="""<br><b>%{customdata[0]}</b>
            <br><br><b>Energy</b>: %{y:.8f}
            <br><b>k-point spacing (&#8491;<sup>-1</sup>)</b>: %{x:.8f}
            <br><b>k-point array</b>: %{customdata[1]}
            """,
            hoverlabel=dict(
                font_size=16,
            ),
        )

    if change_ticks:
        # Scaling x axis ticks to a size that actually fits
        # the screen.
        # This was implemented due to some weird display issues,
        # but maybe it could be fixed by changing different
        # parameters.
        fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=ticks,
                tickformat=".3f",
                gridcolor="#c3cad4",
            ),
            xaxis=dict(
                range=[x_val_min * 0.9, x_val_max * 1.1],
                gridcolor="#c3cad4",
            ),
        )

    # Choosing font and some other settings
    fig.update_layout(
        # font_family="Fira Code Regular",
        font_size=30,
        margin=dict(l=10, r=5, b=10, t=25, pad=2),
        transition_duration=300,
        uniformtext_minsize=12,
        uniformtext_mode="hide",
        yaxis=dict(gridcolor="#c3cad4"),
        xaxis=dict(gridcolor="#c3cad4"),
    )

    fig.update_layout(clickmode="event+select")

    # Changing marker sizes
    fig.update_traces(
        marker_size=15,
        marker_line_width=2,
    )

    # Generating label for the scale output.
    if isinstance(axis_tick_scale, float):
        axis_tick_scale = 1000 * axis_tick_scale
        final_tick_scale = f" Scale: {axis_tick_scale} meV"
    else:
        final_tick_scale = f" Scale: {axis_tick_scale}"

    return fig, final_tick_scale


@app.callback(
    Output("distance-indicator", "children"), Input("vasp-graph", "selectedData")
)
def display_click_data(selectedData):
    if selectedData:
        if len(selectedData["points"]) == 2:
            val_1 = selectedData["points"][0]["y"]
            val_2 = selectedData["points"][1]["y"]
            energ_diff_mev = (val_1 - val_2) * 1000

            return f"Energy Difference: {energ_diff_mev:.3f} meV"

        else:
            return "Select an additional point to get the energy difference."

    else:
        return "Select two points to get the energy difference."


if __name__ == "__main__":
    app.run(host="0.0.0.0", use_reloader=False, debug=True)
