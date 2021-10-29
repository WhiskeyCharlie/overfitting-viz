"""
Everything needed to add the layout (structure) to the app
"""
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import dash_reusable_components as drc
import tooltip_data as ttd

EXTERNAL_CSS = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    dbc.themes.BOOTSTRAP
]
MIN_NUM_POINTS_SLIDER, DEFAULT_NUM_POINTS_SLIDER = 10, 30
APP_TITLE = 'Vizibly'


def add_layout_to_app(app) -> None:
    """
    Defines and adds the layout (physical properties) of the given app.
    This includes buttons, sliders, graphs, etc.
    :param app: Dash app object
    :return: None
    """
    app.layout = html.Div(children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(className="banner", children=[
            html.Div(className='container scalable', children=[
                html.H2(html.A(
                    APP_TITLE,
                    style={'text-decoration': 'none', 'color': 'inherit'}
                ))
            ]),
        ]),

        html.Div(id='body', className='container scalable', children=[
            html.Div(id='custom-data-storage', style={'display': 'none'}),
            html.Div(
                className='two columns',
                style={
                    'min-width': '15%',
                    'max-height': 'calc(100vh - 85px)',
                    'overflow-y': 'hidden',
                    'overflow-x': 'hidden',
                },
                children=[
                    html.Br(), html.Br(), html.Br(), html.Br(),
                    drc.named_dropdown(
                        name='Dataset',
                        id='dropdown-dataset',
                        options=[
                            {'label': 'Dataset Degree 0', 'value': 'degree_0'},
                            {'label': 'Dataset Degree 1', 'value': 'degree_1'},
                            {'label': 'Dataset Degree 2', 'value': 'degree_2'},
                            {'label': 'Dataset Degree 3', 'value': 'degree_3'},
                            {'label': 'Dataset Degree 4', 'value': 'degree_4'},
                            {'label': 'Dataset Degree 5', 'value': 'degree_5'},
                            {'label': 'Dataset Degree 6', 'value': 'degree_6'},
                            {'label': 'Dataset Degree 7', 'value': 'degree_7'},
                            {'label': 'Dataset Degree 8', 'value': 'degree_8'},
                            {'label': 'Dataset Degree 9', 'value': 'degree_9'},
                            {'label': 'Dataset Degree 10', 'value': 'degree_10'},

                        ],
                        value='degree_1',
                        clearable=False,
                        searchable=False,
                        style={
                            'color': 'rgb(0, 0, 0)',
                            'width': '100%'
                        }

                    ),
                    html.Br(),
                    drc.named_input_area(
                        name="Noise Factor",
                        min=0,
                        value=0,
                        id='slider-dataset-noise',
                        type='number',
                        style={'width': '100%'}
                    ),
                    html.Br(),
                    drc.named_input_area(
                        name="Dataset Sample Size",
                        min=MIN_NUM_POINTS_SLIDER,
                        value=DEFAULT_NUM_POINTS_SLIDER,
                        id="slider-sample-size",
                        type='number',
                        style={'width': '100%'}
                    ),
                    html.Br(),
                    drc.named_slider(
                        name='Model Polynomial Degree',
                        min=1,
                        max=10,
                        value=1,
                        step=1,
                        id="slider-polynomial-degree",
                        marks={str(i): str(i) for i in range(1, 11)}
                    ),
                    html.Br(),
                    html.Br(),
                    html.Button('Resample Train/Test',
                                id='resample-btn',
                                style={'color': 'rgb(200, 200, 200)', 'width': '100%'}),
                    html.Br()
                ]
            ),
            html.Div(
                id='div-graphs',
                className='five columns',
                children=[
                    dcc.Graph(
                        id='graph-regression-display',
                        figure=dict(
                            layout=dict(
                                plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                            )
                        ),
                        className='row',
                        style={
                            'height': 'calc(100vh - 160px)',
                        },
                        config={'modeBarButtonsToRemove': [
                            'pan2d',
                            'lasso2d',
                            'select2d',
                            'autoScale2d',
                            'hoverClosestCartesian',
                            'hoverCompareCartesian',
                            'toggleSpikelines'
                        ]}
                    )],
                style={'display': 'inline-block'}),
            html.Div(
                id='div-fitting',
                className='four columns',
                children=[
                    dcc.Graph(
                        id='graph-fitting-display',
                        className='row',
                        style={
                            'height': 'calc(100vh - 160px)',
                        },
                        figure=dict(
                            layout=dict(
                                plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                            )
                        ),
                        config={'modeBarButtonsToRemove': [
                            'pan2d',
                            'lasso2d',
                            'select2d',
                            'autoScale2d',
                            'hoverClosestCartesian',
                            'hoverCompareCartesian',
                            'toggleSpikelines'
                        ]}
                    )]
            ),
        ]),
        drc.custom_tooltip(ttd.SLIDER_DATASET_NOISE, target='slider-dataset-noise'),
        drc.custom_tooltip(ttd.RESAMPLE_BUTTON, target='resample-btn'),
    ])
