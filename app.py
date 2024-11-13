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
from scipy.special import softmax
from scipy.spatial.distance import cosine


PROJECT_ROOT = Path(__file__).parent
experiment = 202411071548
# tmp_dir = PROJECT_ROOT / "outputs/tmp"
# tmp_dir.mkdir(parents=True, exist_ok=True)


output_dir = PROJECT_ROOT / "outputs" / str(experiment) / "pareto_set"
configuration_columns = ["Urban Density", "Land Use", "Public Greenspace", "Job Mix"]
indicator_columns = [
    "Air Purity",
    "House Affordability",
    "Job Accessibility",
    "Greenspace Accessibility",
]
pareto_set = []
for i in output_dir.iterdir():
    configuration = pd.read_csv(i, index_col="geo_code")
    configuration.columns = configuration_columns + indicator_columns
    pareto_set.append(configuration)
# current = pd.DataFrame([[15.501794, -7.411663, 3505.254282, 480699.676225]], columns=indicator_columns)
pareto_evals = pd.DataFrame([p[indicator_columns].mean() for p in pareto_set])
# pareto_evals = pd.concat([current, pareto_evals], ignore_index=True)
scaler = MinMaxScaler()
pareto_evals[:] = scaler.fit_transform(pareto_evals)
hypervolume = pareto_evals.prod(axis="columns")
pareto_evals.loc[:, "Unified Score"] = hypervolume / hypervolume.max()
pareto_evals = pareto_evals[pareto_evals.columns[-1:].to_list() + pareto_evals.columns[:-1].to_list()]
for c in pareto_set:
    c.loc[:, indicator_columns] = scaler.transform(c.loc[:, indicator_columns])
# pareto_evals = pareto_evals.head(10)
    
colours = [
    "#5FB6E3",
    "#655C56",
    "#C5B6A8",
    "#D6534C",
    "#396CAF",
    "#D8A2FF",
    "#743F9A",
    "#51A220",
    "#ECA98A",
    "#CEC11F"
]
colour_names = [
    "Cyan",
    "Grey",
    "Beige",
    "Red",
    "Blue",
    "Pink",
    "Purple",
    "Green",
    "Orange",
    "Yellow"
]
    

def get_options_from_preferences(preferences, pareto_evals, num_options=10):
    distances = pareto_evals.loc[:, indicator_columns].apply(lambda row: cosine(preferences, row), axis=1)
    return distances.argsort().to_numpy()[:num_options]


def plot_options(pareto_evals):
    rows = 2
    cols = 5
    fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "polar"} for _ in range(cols)] for _ in range(rows)])
    for i, (_, c) in enumerate(pareto_evals.iterrows()):
        fig.add_trace(
            go.Scatterpolar(r=c.to_numpy()[1:], theta=["AP", "HA", "JA", "GA"], mode="lines", line=dict(color=colours[i]), fill="toself"),
            row=i // cols + 1,
            col=i % cols + 1
        )
    fig.update_polars(
        # angularaxis_showticklabels=False,
        radialaxis_showticklabels=False,
        radialaxis_range=[0, 1]
    )
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False)
    return fig


colour_text = {
    "Urban Density": ([1, 12], ["Rural", "Urban"]),
    "Land Use": ([-1, 1], ["More Residential", "More Commercial"]),
    "Public Greenspace": ([0, 1], ["Less Greenspace", "More Greenspace"]),
    "Job Mix": ([0, 1], ["Blue Collar", "White Collar"])
}
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
        # coloraxis="coloraxis",
        marker=dict(line=dict(color="rgba(0,0,0,0)")),
        colorbar=dict(
            tickmode="array",
            tickvals=colour_text[signal.name][0],
            ticktext=colour_text[signal.name][1],
        )
    )


# Initialize the Dash app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
# Create the layout for Dash
app.layout = html.Div(
    children=[
        html.H1(children="Trade-off Explorer"),
        html.Div(children="The Demoland simulator divides Newcastle into 3795 zones (based on the UK Census 'Output Areas'). A scenario is specified by setting a value for Urban Density, Land Use, Public Greenspace, and Job Mix for each zone. Demoland then predicts, for each zone, urban quality indicators - Air Purity, House Affordability, Greenspace Accessibility and Job Accessibility. Our optimisation alorithm finds the set of scenarios that achieve the best trade-offs between the indicators, called the Pareto set. This tool explores the set found by our algorithm."),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="six columns",
                    children=[
                        html.H3(children="Indicate your preferences"),
                        html.Div(children="Drag the sliders to indicate how important each urban quality indicator is to your plan."),
                        html.B(children=["Air Purity", dcc.Slider(0, 1, value=0.5, id="air_quality")]),
                        html.B(children=["House Affordability", dcc.Slider(0, 1, value=0.5, id="house_affordability")]),
                        html.B(children=["Job Accessibility", dcc.Slider(0, 1, value=0.5, id="job_accessibility")]),
                        html.B(children=["Greenspace Accessibility", dcc.Slider(0, 1, value=0.5, id="greenspace_accessibility")]),
                        # html.Div("."),
                        html.Div(children=["The spiderplots on the right show the scenarios that most closely match your preferences. Use the dropdown menu to choose the colour of the scenario you would like to explore further. This is the scenario that will be shown in the Scenario Explorer below.", dcc.Dropdown(colour_names, "Cyan", id="colour")]),
                    ]
                ),
                html.Div(
                    className="six columns",
                    children=[
                        html.H3(children="Browse your options"),
                        html.Div(children="These spiderplots represent the scenarios in the Pareto set that best match your preferences."),
                        html.Div(children=dcc.Graph(figure={}, id="radar-options"))
                    ]
                ),
            ]
        ),
        html.H2(children="Scenario Explorer"),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="six columns",
                    children=[
                        html.H3(children="Scenario Map"),
                        html.Div(children=[
                            "The Scenario Map makes it easy to see your city's story come to life by showing how different aspects of an urban plan come together. Choose an aspect to see it mapped out, helping you understand patterns and distributions at a glance. Use this to explore urban density, land use, greenspace, and job composition, making complex data easy to grasp.",
                            html.Ul([
                                html.Li("Urban Density: represents the degree of urbanity, showing a spectrum from mostly rural to highly urban areas."),
                                html.Li("Land Use: measures the ratio of commercial to residential space, indicating deviations from the baseline distribution."),
                                html.Li("Public Greenspace: displays the percentage of publicly accessible greenspace, ranging from minimal to full-area coverage."),
                                html.Li("Job Mix: illustrates the balance between office (white-collar) and manual (blue-collar) jobs in the area.")
                            ])
                        ]),
                        html.Div(children=
                            ["Press a button to update the map: "]
                            + [html.Button(i, id=i, n_clicks=0) for i in configuration_columns]
                        ),
                        html.Div(children=dcc.Graph(figure={}, id="configuration")),
                    ]
                ),
                html.Div(
                    className="six columns",
                    children=[
                        html.H3(children="Average Values"),
                        html.Div(children=[
                            "The spider plot shows the scenario's predicted urban quality indicators (averaged over the zones). A value of 1 indicates the best performance, 0 the worst.",
                            html.Ul([
                                html.Li("Air Purity: summarizes pollution levels based on WHO guidelines, combining five key pollutants into a weighted score. Higher values indicate cleaner air."),
                                html.Li("House Affordability: reflects the average cost of housing per square metre across different zones. Higher values indicate more affordable housing."),
                                html.Li("Greenspace Accessibility: assesses the availability of parks that can be reached within a 15-minute drive from any location in the zone. Higher values indicate more accessible greenspace."),
                                html.Li("Job Accessibility: measures how many jobs are accessible within a 15-minute drive from any point in a given zone. Higher values indicate better job accessibility (and therefore lower commute times).")
                            ])
                        ]),
                        html.Div(children=dcc.Graph(figure={}, id="radar"))
                    ]
                ),
            ]
        ),
        html.H3(children="Distributions"),
        html.Div(children="The histograms show the distribution of indicator values over the city's zones for this scenario."),
        html.Div(children=dcc.Graph(figure={}, id="indicators")),
        # Add dcc.Store component to store the 'selected' value``
        dcc.Store(id='selected-store', data=0),
        html.H3(children="Performance: Indicator vs Indicator"),
        html.Div(children="These plots show how the indicator values of your chosen scenario compare to the indicator values of the other options. The point highlighted with the larger circle is your chosen scenario."),
        html.Div(children=[dcc.Graph(figure={}, id="scatter-matrix", style={"height": "75vh"})]),
    ]
)


@callback(
    Output("radar-options", "figure"),
    Output("radar", "figure"),
    Output("configuration", "figure"),
    Output("indicators", "figure"),
    Output("scatter-matrix", "figure"),
    Output('selected-store', 'data'),  # Output to store 'selected' value
    Input("air_quality", "value"),
    Input("house_affordability", "value"),
    Input("job_accessibility", "value"),
    Input("greenspace_accessibility", "value"),
    Input("colour", "value"),
    Input("Urban Density", "n_clicks"),
    Input("Land Use", "n_clicks"),
    Input("Public Greenspace", "n_clicks"),
    Input("Job Mix", "n_clicks"),
)
# def update_plots(sort_by, rank, configuration_type, clickData, selected_store):
def update_plots(air_quality, house_affordability, job_accessibility, greenspace_accessibility, colour, signature_type, use, greenspace, job_types):
    ctx = callback_context
    configuration_type = "Urban Density" 
    for c in configuration_columns:
        if ctx.triggered_id == c:
            configuration_type = c

    option_idx = colour_names.index(colour)

    figs = []

    options = get_options_from_preferences(
        softmax([air_quality, house_affordability, job_accessibility, greenspace_accessibility]),
        pareto_evals
    )
    figs.append(plot_options(pareto_evals.loc[options]))
    selected = options[option_idx]

    figs.append(
        go.Figure(data=go.Scatterpolar(r=pareto_evals.loc[selected].to_numpy()[1:], theta=indicator_columns, mode="lines", line=dict(color=colours[option_idx]), fill="toself"))
    )
    figs[-1].update_layout(polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False)

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
    figs[-1].update_xaxes(showticklabels=False)
    figs[-1].update_yaxes(showticklabels=False)
    figs[-1].update_layout(showlegend=False)
    
    fig = go.Figure()
    df = pareto_evals.loc[options, indicator_columns]
    for i, (_, c) in enumerate(df.iterrows()):
        if i == option_idx:
            fig.add_trace(
                go.Splom(
                    dimensions=[
                        dict(label=col_name, values=[c.loc[col_name]]) for col_name in indicator_columns
                    ],
                    marker=dict(size=20, color="#3490FF", opacity=0.2, showscale=False)
                )
            )
        fig.add_trace(
            go.Splom(
                dimensions=[
                    dict(label=col_name, values=[c.loc[col_name]]) for col_name in indicator_columns
                ],
                marker=dict(size=8, color=colours[i], showscale=False)
            )
        )
    # df = pareto_evals.drop(options)
    # fig.add_trace(
    #     go.Splom(
    #         dimensions=[
    #             dict(label=col_name, values=df[col_name]) for col_name in indicator_columns
    #         ],
    #         marker=dict(size=4, color="blue", showscale=False)
    #     )
    # )
    fig.update_layout({"xaxis"+str(i+1): dict(range = [0, 1]) for i in range(7)})
    fig.update_layout({"yaxis"+str(i+1): dict(range = [0, 1]) for i in range(7)})
    fig.update_traces(diagonal_visible=False, showlegend=False)
    figs.append(fig)
    
    # Return the figures and update the 'selected' value in the store
    return tuple(figs) + (selected,)


# Run the Dash app
if __name__ == "__main__":
    app.run(debug=False, port=os.environ.get("PORT", 8050))
