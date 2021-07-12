import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import dash_reusable_components as drc
import tooltip_data as ttd
from dataset_generation import DatasetGenerator
from general_utils import format_yhat

RANDOM_STATE = 718

EXTERNAL_CSS = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    dbc.themes.BOOTSTRAP
]

app = dash.Dash(__name__,
                external_stylesheets=EXTERNAL_CSS)
server = app.server

# This block of code defines the layout of the app, meaning its physical structure: sliders, buttons, etc.
app.layout = html.Div(children=[
    # .container class is fixed, .container.scalable is scalable
    html.Div(className="banner", children=[
        html.Div(className='container scalable', children=[
            html.H2(html.A(
                'Overfitting Explorer',
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
                        # {'label': 'Dataset #1', 'value': 'dataset #1'},
                        # {'label': 'Dataset #2', 'value': 'dataset #2'},
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
                drc.input_area(
                    name="Noise Factor",
                    min=0,
                    value=0,
                    id='slider-dataset-noise',
                    type='number',
                    style={'width': '100%'}
                ),
                html.Br(),
                drc.input_area(
                    name="Dataset Sample Size",
                    min=10,
                    value=300,
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
                            id='resample-btn', style={'color': 'rgb(200, 200, 200)', 'width': '100%'}),
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
                        'height': 'calc(50vh - 160px)',
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
    # drc.custom_tooltip(ttd.SLIDER_POLYNOMIAL_DEGREE, target='slider-polynomial-degree'),
    drc.custom_tooltip(ttd.RESAMPLE_BUTTON, target='resample-btn'),
])


@app.callback(Output('graph-regression-display', 'figure'),
              [Input('dropdown-dataset', 'value'),
               Input('slider-sample-size', 'value'),
               Input('slider-polynomial-degree', 'value'),
               Input('slider-dataset-noise', 'value'),
               Input('resample-btn', 'n_clicks')])
def update_graph(dataset, sample_size, degree, noise_factor, n_clicks=0,
                 split_random_state=RANDOM_STATE):
    """
    Function called any time the graph needs to be updated. We redraws the graph from scratch
    :param dataset: Name of the dataset to generate
    :param sample_size: How many points to generate
    :param degree: Degree of the polynomial to fit to the graph
    :param noise_factor: How much noise should be added to the data (how much it deviates from the true function)
    :param n_clicks: How many times has the resample button been pressed
    :param split_random_state: The random state under which to split the data into training and test
    :return: The figure, essentially the main graph to display
    """
    ctx = dash.callback_context
    if ctx.triggered:
        button_is_event = ctx.triggered[0]['prop_id'].split('.')[0] == 'resample-btn'
        if button_is_event:
            np.random.seed(n_clicks or RANDOM_STATE)  # This hack takes RANDOM_STATE if n_clicks is 0, else n_clicks
            split_random_state = np.random.randint(100)
    generator = DatasetGenerator(dataset, RANDOM_STATE, sample_size, noise_factor)
    X, y, X_out_range, y_out_range = generator.make_dataset(use_random_seed=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=int(X.shape[0] * 0.15),
                                                        random_state=split_random_state)

    X_range = np.linspace(min(X.min(), X_out_range.min()) - 0.5,
                          max(X.max(), X_out_range.max()) + 0.5, sample_size).reshape(-1, 1)

    # Create Polynomial Features so that linear regression is actually polynomial regression
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_range = poly.fit_transform(X_range)

    model = LinearRegression()

    # Train model and predict
    model.fit(X_train_poly, y_train)
    y_pred_range = model.predict(poly_range)
    train_error = mean_squared_error(y_train, model.predict(X_train_poly))
    test_error = mean_squared_error(y_test, model.predict(X_test_poly))

    # Create figure
    trace_train_in_range = go.Scatter(
        x=X_train.squeeze(),
        y=y_train,
        name='Training Data',
        mode='markers',
        opacity=0.7,
        marker=dict(size=8)
    )
    trace_test_in_range = go.Scatter(
        x=X_test.squeeze(),
        y=y_test,
        name='Test Data',
        mode='markers',
        opacity=0.7,
        marker=dict(size=8)
    )

    trace_test_out_range = go.Scatter(
        x=X_out_range.squeeze(),
        y=y_out_range,
        name='Out Of Range Test Data',
        mode='markers',
        opacity=0.7,
        marker=dict(size=8, color='yellow')
    )

    trace_prediction = go.Scatter(
        x=X_range.squeeze(),
        y=y_pred_range,
        name='Prediction',
        mode='lines',
        hovertext=format_yhat(model),
        marker=dict(color='#27ab22'),
        line=dict(width=4)
    )
    data = [trace_train_in_range, trace_test_in_range, trace_test_out_range, trace_prediction]

    layout = go.Layout(
        title=f"MSE: {train_error:.3f} (Train Data) "
              f"\n MSE: {test_error:.3f} (Test Data)",
        legend=dict(orientation='h'),
        margin=dict(l=25, r=25),
        hovermode='closest',
        plot_bgcolor="#cbd3f2",
        paper_bgcolor="#282b38",
        font=dict(color='rgb(200, 200, 200)', size=15)

    )

    return go.Figure(data=data, layout=layout)


@app.callback(Output('graph-fitting-display', 'figure'),
              [Input('dropdown-dataset', 'value'),
               Input('slider-sample-size', 'value'),
               Input('slider-polynomial-degree', 'value'),
               Input('slider-dataset-noise', 'value'),
               Input('resample-btn', 'n_clicks')])
def update_fitting_graph(dataset, sample_size, chosen_degree, noise_factor, n_clicks=0):
    """
    Function called any time the graph needs to be updated. We redraws the graph from scratch
    :param dataset: Name of the dataset to generate
    :param sample_size: How many points to generate
    :param chosen_degree: The degree of the polynomial the user is trying to fit to the dataset (draws vertical line)
    :param noise_factor: How much noise should be added to the data (how much it deviates from the true function)
    :param n_clicks: How many times has the resample button been pressed
    :return: The figure, essentially the main graph to display
    """
    max_degree_to_check = 10
    generator = DatasetGenerator(dataset, RANDOM_STATE, sample_size, noise_factor)
    X, y, X_out_range, y_out_range = generator.make_dataset(True)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=int(X.shape[0] * 0.15), random_state=n_clicks or RANDOM_STATE)

    train_errors = []
    test_errors = []
    out_of_range_test_errors = []
    degrees = list(range(1, max_degree_to_check + 1))
    for deg in degrees:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        X_test_out_of_range_poly = poly.transform(X_out_range)

        model = LinearRegression()

        # Train model and predict
        model.fit(X_train_poly, y_train)
        train_error = mean_squared_error(y_train, model.predict(X_train_poly))
        test_error = mean_squared_error(y_test, model.predict(X_test_poly))
        out_of_range_test_error = mean_squared_error(y_out_range, model.predict(X_test_out_of_range_poly))
        train_errors.append(train_error)
        test_errors.append(test_error)
        out_of_range_test_errors.append(out_of_range_test_error)

    trace_train = go.Scatter(
        x=degrees,
        y=train_errors,
        name='Training MSE',
        opacity=0.7,
        marker=dict(color='blue'),
        line=dict(width=4)
    )

    trace_test = go.Scatter(
        x=degrees,
        y=test_errors,
        name='Testing MSE',
        opacity=0.7,
        marker=dict(color='red'),
        line=dict(width=4)
    )

    trace_test_out_of_range = go.Scatter(
        x=degrees,
        y=out_of_range_test_errors,
        name='OOR Testing MSE',
        opacity=0.7,
        marker=dict(color='yellow'),
        line=dict(width=4)
    )

    # noinspection PyTypeChecker
    layout = go.Layout(
        title='',
        legend=dict(orientation='h',
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01),
        margin=dict(l=25, r=25),
        hovermode='closest',
        plot_bgcolor="#cbd3f2",
        paper_bgcolor="#282b38",
        font=dict(color='rgb(200, 200, 200)', size=15),
        xaxis=dict(tickvals=degrees),
        xaxis_title='Polynomial Degree',
        yaxis_title='Mean Squared Error'
    )
    fig = go.Figure(data=[trace_train, trace_test, trace_test_out_of_range], layout=layout)
    fig.add_vline(x=chosen_degree, line_width=3, line_color='#27ab22',
                  annotation=dict(text='Current Degree', textangle=-90, font=dict(color='rgb(0, 0, 0)'),
                                  yshift=-100))
    return fig


# Running the server
if __name__ == '__main__':
    app.run_server(port=2522, debug=True)

# TODO: Noise seems broken to me. It seems to add far too much noise for low values for instance FIXED?
# TODO: dataset generation should be all in one place DONE
# TODO: Easy-to-use documentation for entire project
# TODO: Commented code everywhere with good explanations IN PROGRESS
