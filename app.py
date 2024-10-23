import os
from pathlib import Path

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, callback_context, State
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).parent
experiment = 202409182205  # directory where results are logged
# tmp_dir = PROJECT_ROOT / "outputs/tmp"
# tmp_dir.mkdir(parents=True, exist_ok=True)


output_dir = PROJECT_ROOT / "outputs" / str(experiment) / "pareto_set"
pareto_set = []
for i in output_dir.iterdir():
    pareto_set.append(pd.read_csv(i, index_col="geo_code"))
configuration_columns = ["signature_type", "use", "greenspace", "job_types"]
indicator_columns = [
    "air_quality",
    "house_affordability",
    "job_accessibility",
    "greenspace_accessibility",
]
pareto_evals = pd.DataFrame([p[indicator_columns].mean() for p in pareto_set])
pareto_evals[:] = MinMaxScaler().fit_transform(pareto_evals)
pareto_evals.loc[:, "hypervolume"] = (
    (pareto_evals**2).sum(axis="columns").apply(np.sqrt)
)
pareto_evals = pareto_evals.head(10)
pareto_evals[:] = MinMaxScaler().fit_transform(pareto_evals)


def plot_signal_on_geography(
    signal: pd.Series, geography: gpd.GeoDataFrame = None
) -> plotly.basedatatypes.BaseTraceType:
    """Plot a signal on the geography.

    :param signal: The signal. Should be indexed by Output Area codes.
    :param geography: The geography. Should be indexed by Output Area
        codes.
    :return: A plotly trace of the signal on the geography.
    """
    if not geography:
        with open(
            PROJECT_ROOT / "outputs/geography.json"
        ) as f:
            geo_data = json.load(f)
        geography = gpd.GeoDataFrame.from_features(geo_data["features"]).set_index(
            "OA11CD"
        )
    gdf = geography.merge(signal, left_on="OA11CD", right_index=True, how="left")
    return go.Choropleth(
        geojson=json.loads(gdf["geometry"].to_json()),
        locations=gdf.index,
        z=gdf[signal.name],
        coloraxis="coloraxis",
        marker=dict(line=dict(color="rgba(0,0,0,0)")),
        # colorbar=dict(
        # tickvals=np.arange(-4, 4.5, 0.5),  # Keep the detailed tickvals
    )


fig = px.scatter_matrix(pareto_evals)
fig.update_traces(showupperhalf=False)
# fig.show()
# fig.write_html(tmp_dir / "pareto_scatter_matrix.html")

fig = px.line_polar(
    pareto_evals.iloc[:1].T.reset_index(), r=0, theta="index", line_close=True
)
# fig.write_html(tmp_dir / "pareto_line_polar.html")

# Initialize the Dash app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
# Create the layout for Dash
app.layout = html.Div(
    children=[
        html.H1(children="Trade-Off Explorer"),
        html.Div(children="The Demoland simulator divides Newcastle into 3795 zones (based on the UK Census 'Output Areas')."),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="six columns",
                    children=[
                        html.Div(children="An urban configuration (or urban plan) sets, for each zone:"),
                        html.Div(children=html.Ul([
                            html.Li("Spatial signature (degree of urbanity) (0 - 15)."),
                            html.Li("Use (Residential vs Commerical) (-1 - 1)."),
                            html.Li("Greenspace (Area of zone that is greenspace) (0 - 1)."),
                            html.Li("Job types (Manual labour vs office work) (0 - 1).")
                        ]))
                    ]
                ),
                html.Div(
                    className="six columns",
                    children=[
                        html.Div(children="Demoland then predicts urban quality indicators, for each zone:"),
                        html.Div(children=html.Ul([
                            html.Li("Air quality."),
                            html.Li("House affordability."),
                            html.Li("Job accessibility."),
                            html.Li("Greenspace accessibility.")
                        ]))
                    ]
                ),
            ]
        ),
        html.H2(children="The Best Urban Plans Found"),
        html.Div(children="These plots show different views of the same points. Each point represents an urban configuration or plan. Each plot shows the plans' peformance on a pair of indicators (averaged over the zones). (Hypervolume tries to combine all the indicators into one.) The red cross shows the plan being visualised below. Blue crosses show the previously visualised plans."),
        html.Div(children=[dcc.Graph(figure={}, id="scatter-matrix", style={"height": "75vh"})]),
        html.H2(children="Visualisation of a Specific Urban Plan"),
        html.Div(children="A visualisation of the urban plan corresponding to the red cross in the plots above. The map shows the plan, the spider plot shows the plan's predicted urban quality indicators (averaged over the zones). The histograms show the distribution of indicator values over the zones."),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="six columns",
                    children=["Select an aspect of the urban configuration/plan to display on the map:"],
                ),
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(className="six columns", children=[dcc.RadioItems(
                    id="configuration_type",
                    options=configuration_columns,
                    value="signature_type",
                )]),
                # html.Div(className="six columns", children=[dcc.RadioItems(
                #     id="sort_by",
                #     options=["hypervolume"] + indicator_columns,
                #     value="hypervolume",
                # )]),
            ],
        ),
        # html.Div(
        #     className="row",
        #     children=[
        #         html.Div(
        #             className="six columns",
        #             children=[
        #                 ""
        #             ],
        #         ),
        #         html.Div(
        #             className="six columns",
        #             children=[
        #                 dcc.Slider(0, len(pareto_set) - 1, 1, value=0, id="rank")
        #             ],
        #         ),
        #     ],
        # ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="six columns",
                    children=[dcc.Graph(figure={}, id="configuration")],
                ),
                html.Div(
                    className="six columns", children=[dcc.Graph(figure={}, id="radar")]
                ),
            ],
        ),
        html.Div(className="row", children=[dcc.Graph(figure={}, id="indicators")]),
        # Add dcc.Store component to store the 'selected' value
        dcc.Store(id='selected-store', data=0),
    ]
)


@callback(
    Output("radar", "figure"),
    Output("configuration", "figure"),
    Output("indicators", "figure"),
    Output("scatter-matrix", "figure"),
    Output('selected-store', 'data'),  # Output to store 'selected' value
    # Input("sort_by", "value"),
    # Input("rank", "value"),
    Input("configuration_type", "value"),
    Input("scatter-matrix", "clickData"),
    State('selected-store', 'data')  # State to get the stored 'selected' value
)
# def update_plots(sort_by, rank, configuration_type, clickData, selected_store):
def update_plots(configuration_type, clickData, selected_store):
    ctx = callback_context
    if not ctx.triggered:
        triggered_input = "No input triggered yet"
    else:
        triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]
        
    if triggered_input == "sort_by" or triggered_input == "rank" or triggered_input == "No input triggered yet":
        # sorting = pareto_evals.sort_values(sort_by, ascending=False).index
        # selected = sorting[rank]
        selected = 0
    elif triggered_input == "scatter-matrix":
        clicked_values = [
            clickData["points"][0]['dimensions[0].values'],
            clickData["points"][0]['dimensions[1].values'],
            clickData["points"][0]['dimensions[2].values'],
            clickData["points"][0]['dimensions[3].values'],
            clickData["points"][0]['dimensions[4].values']
        ]

        columns = ['air_quality', 'house_affordability', 'job_accessibility', 'greenspace_accessibility', 'hypervolume']

        distances = np.sqrt(
            (pareto_evals[columns[0]] - clicked_values[0]) ** 2 +
            (pareto_evals[columns[1]] - clicked_values[1]) ** 2 +
            (pareto_evals[columns[2]] - clicked_values[2]) ** 2 +
            (pareto_evals[columns[3]] - clicked_values[3]) ** 2 +
            (pareto_evals[columns[4]] - clicked_values[4]) ** 2
        )

        selected = distances.idxmin()
    elif triggered_input == "configuration_type":
        selected = selected_store  # Keep the previous 'selected' value
    else:
        selected = selected_store  # Default to previous 'selected' value for any other input

    figs = []

    figs.append(
        px.line_polar(
            pareto_evals.loc[selected].T.reset_index(),
            r=selected,
            theta="index",
            line_close=True,
        )
    )
    figs[-1].update_traces(fill="toself")

    trace = plot_signal_on_geography(pareto_set[selected][configuration_type])
    figs.append(go.Figure(data=trace))
    figs[-1].update_geos(fitbounds="locations", visible=False)

    figs.append(
        make_subplots(
            rows=1,
            cols=len(indicator_columns),
            shared_yaxes=True,
            subplot_titles=indicator_columns,
        )
    )
    for i, ic in enumerate(indicator_columns):
        figs[-1].add_trace(
            go.Histogram(x=pareto_set[selected][ic].to_numpy()),#, histnorm="probability"),
            row=1,
            col=i + 1,
        )
    figs[-1].update_layout(showlegend=False)
    
    fig0 = go.Figure()
    for i in range(pareto_evals.shape[0]):
        fig1 = px.scatter_matrix(pareto_evals.loc[[i]])
    
        if i == selected:
            mysymbol = 'x'
            mycolor = 'red'
        else:
            mysymbol = 'circle'
            mycolor = 'blue'

        fig1.update_traces(marker=dict(size=8, color=mycolor, symbol=mysymbol, opacity=0.7))

        for trace in fig1.data:
            fig0.add_trace(trace)

    figs.append(fig0)
    figs[-1].update_traces(showupperhalf=False)
    
    # Return the figures and update the 'selected' value in the store
    return tuple(figs) + (selected,)


# Run the Dash app
if __name__ == "__main__":
    app.run(debug=False, port=os.environ.get("PORT", 8050))
