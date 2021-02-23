import json

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import dash_reusable_components as drc
from generate_regression_data import reg_functions, gen_regression_symbolic

RANDOM_STATE = 718
DS_NAME_TO_DEGREE = {'degree_0': 0, 'degree_1': 1, 'degree_2': 2, 'degree_3': 3,
                     'degree_4': 4, 'degree_5': 5, 'degree_6': 6, 'degree_7': 7, 'degree_8': 8,
                     'degree_9': 9, 'degree_10': 10}

EXTERNAL_CSS = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    # Custom Stylesheet
    # "https://cdn.rawgit.com/plotly/dash-regression/98b5a541/custom-styles.css"
]

app = dash.Dash(__name__,
                external_stylesheets=EXTERNAL_CSS)
server = app.server

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
                drc.card([
                    html.Br(), html.Br(), html.Br(), html.Br(),
                    drc.named_dropdown(
                        name='Select Dataset',
                        id='dropdown-dataset',
                        options=[
                            {'label': 'Dataset #1', 'value': 'dataset #1'},
                            {'label': 'Dataset #2', 'value': 'dataset #2'},
                            # {'label': 'Custom Data', 'value': 'custom'},
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
                        }

                    ),
                    html.Br(),
                    drc.input_area(
                        name="Select Noise Factor",
                        min=0,
                        value=0,
                        id='slider-dataset-noise',
                        type='number'
                    ),
                    html.Br(),
                    drc.input_area(
                        name="Select Dataset Sample Size",
                        min=10,
                        value=300,
                        id="slider-sample-size",
                        type='number'
                    ),
                    html.Br(),
                    drc.input_area(
                        name='Select Model Polynomial Degree',
                        min=0,
                        max=10,
                        value=1,
                        id="slider-polynomial-degree",
                        type='number'
                    ),
                    html.Br(),
                    html.Br(),
                    html.Button('Resample Train/Test', id='resample-btn'),
                    html.Br(),
                ]),
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
])


def make_dataset(name, random_state, sample_size, noise_factor):
    np.random.seed(random_state)

    if name == 'dataset #1':
        X = load_boston().data[:, -1].reshape(-1, 1)
        y = (load_boston().target + 23) * (noise_factor + 1) * -3.75
        if noise_factor:
            y[:100] += (1 + noise_factor * 10)
            y[250:350] += (20 + noise_factor * 30)
            y[400:] += (1 + noise_factor * 15)
        return X, y

    elif name == 'dataset #2':
        X = load_boston().data[:, -1].reshape(-1, 1)
        y = load_boston().target
        if noise_factor:
            y[:50] += (5 + noise_factor * 13)
            y[250:350] -= (20 + noise_factor * 3)
            y[450:] += (7 + noise_factor * 25)
        return X, y

    else:
        ds_degree = DS_NAME_TO_DEGREE[name]
        regression_func = reg_functions[ds_degree]
        return gen_regression_symbolic(m=regression_func, n_samples=sample_size, noise=noise_factor)


def format_yhat(model):
    coefficients = model.coef_
    intercept = model.intercept_
    model_values = np.insert(coefficients, 0, intercept)
    coefficient_string = "yhat = "

    for order, coefficient in enumerate(model_values):
        if coefficient >= 0:
            sign = ' + '
        else:
            sign = ' - '
        if order == 0:
            coefficient_string += f'{coefficient}'
        elif order == 1:
            coefficient_string += sign + f'{abs(coefficient):.3f}*x'
        else:
            coefficient_string += sign + f'{abs(coefficient):.3f}*x^{order}'

    return coefficient_string


@app.callback(Output('custom-data-storage', 'children'),
              [Input('graph-regression-display', 'clickData')])
def update_custom_storage(data):
    if data is None:
        data = {
            'train_X': [1, 2],
            'train_y': [1, 2],
            'test_X': [3, 4],
            'test_y': [3, 4],
        }
    else:
        data = json.loads(data)
    return json.dumps(data)


@app.callback(Output('graph-regression-display', 'figure'),
              [Input('dropdown-dataset', 'value'),
               Input('slider-sample-size', 'value'),
               Input('slider-polynomial-degree', 'value'),
               Input('slider-dataset-noise', 'value'),
               Input('resample-btn', 'n_clicks')])
def update_graph(dataset, sample_size, degree, noise_factor, n_clicks=0,
                 split_random_state=RANDOM_STATE):
    ctx = dash.callback_context
    if ctx.triggered:
        button_is_event = ctx.triggered[0]['prop_id'].split('.')[0] == 'resample-btn'
        if button_is_event:
            np.random.seed(n_clicks or RANDOM_STATE)
            split_random_state = np.random.randint(100)
    # Generate base data
    X, y = make_dataset(dataset, RANDOM_STATE, sample_size, noise_factor)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=int(X.shape[0] * 0.15), random_state=split_random_state)

    X_range = np.linspace(X.min() - 0.5, X.max() + 0.5, sample_size).reshape(-1, 1)

    # Create Polynomial Features
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
    trace0 = go.Scatter(
        x=X_train.squeeze(),
        y=y_train,
        name='Training Data',
        mode='markers',
        opacity=0.7,
        marker=dict(size=8)
    )
    trace1 = go.Scatter(
        x=X_test.squeeze(),
        y=y_test,
        name='Test Data',
        mode='markers',
        opacity=0.7,
        marker=dict(size=8)
    )
    trace2 = go.Scatter(
        x=X_range.squeeze(),
        y=y_pred_range,
        name='Prediction',
        mode='lines',
        hovertext=format_yhat(model),
        marker=dict(color='#27ab22'),
        line=dict(width=4)
    )
    data = [trace0, trace1, trace2]

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
               Input('slider-dataset-noise', 'value'),
               Input('resample-btn', 'n_clicks')])
def update_fitting_graph(dataset, sample_size, noise_factor, n_clicks=0):
    max_degree_to_check = 10
    X, y = make_dataset(dataset, RANDOM_STATE, sample_size, noise_factor)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=int(X.shape[0] * 0.15), random_state=n_clicks or RANDOM_STATE)

    train_errors = []
    test_errors = []
    degrees = list(range(1, max_degree_to_check + 1))
    for deg in degrees:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()

        # Train model and predict
        model.fit(X_train_poly, y_train)
        train_error = mean_squared_error(y_train, model.predict(X_train_poly))
        test_error = mean_squared_error(y_test, model.predict(X_test_poly))
        train_errors.append(train_error)
        test_errors.append(test_error)

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
    return go.Figure(data=[trace_train, trace_test], layout=layout)


# Running the server
if __name__ == '__main__':
    app.run_server(port=2522, debug=True)
