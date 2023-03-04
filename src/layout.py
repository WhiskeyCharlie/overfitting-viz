"""
Everything needed to add the layout (structure) to the app
"""
import os
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

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

_MIN_NUM_POINTS_SLIDER = os.environ.get('MIN_NUM_POINTS_SLIDER', default=10)
_DEFAULT_NUM_POINTS_SLIDER = os.environ.get('DEFAULT_NUM_POINTS_SLIDER', default=30)
_MAX_NUM_POINTS_SLIDER = os.environ.get('MAX_NUM_POINTS_SLIDER', default=300)

_APP_TITLE = 'Vizibly'
_GITHUB_URL = 'https://github.com/WhiskeyCharlie/overfitting-viz'


def add_layout_to_app(app) -> None:
    """
    Defines and adds the layout (physical properties) of the given app.
    This includes buttons, sliders, graphs, etc.
    :param app: Dash app object
    :return: None
    """
    app.layout = html.Div(children=[
        # For storing information on the client-side
        dcc.Store(id='session', storage_type='session'),
        # .container class is fixed, .container.scalable is scalable
        html.Div(className="banner", children=[
            html.Div(className='container scalable', children=[
                html.H2(html.A(
                    _APP_TITLE,
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
                        options=[dict(label=f'Dataset Degree {i}', value=f'degree_{i}') for i in range(11)],
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
                        min=_MIN_NUM_POINTS_SLIDER,
                        value=_DEFAULT_NUM_POINTS_SLIDER,
                        max=_MAX_NUM_POINTS_SLIDER,
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
                        marks={mark: mark for mark in map(str, range(1, 11))}
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
        html.Center(
            children=html.Footer(
                id='footer',
                children=[
                    html.Div(
                        children=[
                            html.I(
                                className='fa fa-github',
                                style={'margin-right': '5px'}
                            ),
                            html.A(
                                children='Get the code',
                                href=_GITHUB_URL
                            )
                        ]
                    )
                ]
            )
        )
    ])
