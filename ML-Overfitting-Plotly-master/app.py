import json

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
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
        html.Div(className='row', children=[
            html.Div(
                id='div-graphs',
                children=dcc.Graph(
                    id='graph-regression-display',
                    figure=dict(
                        layout=dict(
                            plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                        )
                    ),
                    className='row',
                    style={
                        'height': 'calc(100vh - 160px)',
                        'width': 'calc(70vw - 160px)'
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
                ),
                style={'display': 'inline-block'}),

            html.Div(
                className='three columns',
                style={
                    'min-width': '24.5%',
                    'max-height': 'calc(100vh - 85px)',
                    'overflow-y': 'auto',
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
                                {'label': 'Custom Data', 'value': 'custom'},
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
                                'color': 'rgb(0, 0, 0)'
                            }
                        ),
                        drc.named_dropdown(
                            name='Click Mode (Select Custom Data to enable)',
                            id='dropdown-custom-selection',
                            options=[
                                {'label': 'Add Training Data Points', 'value': 'training'},
                                {'label': 'Add Test Data Points', 'value': 'test'},
                                {'label': 'Remove Data point', 'value': 'remove'},
                                {'label': 'Do Nothing', 'value': 'nothing'},
                            ],
                            value='training',
                            clearable=False,
                            searchable=False,
                            style={
                                'color': 'rgb(0, 0, 0)'
                            }
                        ),
                        drc.input_area(
                            name="Select Dataset Noise Factor",
                            min=0,
                            value=0,
                            id='slider-dataset-noise',
                            type='number'
                        ),
                        drc.input_area(
                            name="Select Dataset Sample Size",
                            min=10,
                            value=300,
                            id="slider-sample-size",
                            type='number'
                        ),
                        drc.input_area(
                            name='Select Model Polynomial Degree',
                            min=0,
                            max=10,
                            value=1,
                            id="slider-polynomial-degree",
                            type='number'
                        )
                    ]),
                ]
            ),
        ]),
    ])
])


def make_dataset(name, random_state, sample_size, noise_factor):
    np.random.seed(random_state)

    if name == 'dataset #1':
        X = load_boston().data[:, -1].reshape(-1, 1)
        print(X)
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


@app.callback(Output('dropdown-custom-selection', 'disabled'),
              [Input('dropdown-dataset', 'value')])
def disable_custom_selection(dataset):
    return dataset != 'custom'


@app.callback(Output('custom-data-storage', 'children'),
              [Input('graph-regression-display', 'clickData')],
              [State('dropdown-custom-selection', 'value'),
               State('custom-data-storage', 'children'),
               State('dropdown-dataset', 'value')])
def update_custom_storage(clickData, selection, data, dataset):
    if data is None:
        data = {
            'train_X': [1, 2],
            'train_y': [1, 2],
            'test_X': [3, 4],
            'test_y': [3, 4],
        }
    else:
        data = json.loads(data)
        if clickData and dataset == 'custom':
            selected_X = clickData['points'][0]['x']
            selected_y = clickData['points'][0]['y']

            if selection == 'training':
                data['train_X'].append(selected_X)
                data['train_y'].append(selected_y)
            elif selection == 'test':
                data['test_X'].append(selected_X)
                data['test_y'].append(selected_y)
            elif selection == 'remove':
                while selected_X in data['train_X'] and selected_y in data['train_y']:
                    data['train_X'].remove(selected_X)
                    data['train_y'].remove(selected_y)
                while selected_X in data['test_X'] and selected_y in data['test_y']:
                    data['test_X'].remove(selected_X)
                    data['test_y'].remove(selected_y)

    return json.dumps(data)


@app.callback(Output('graph-regression-display', 'figure'),
              [Input('dropdown-dataset', 'value'),
               Input('slider-sample-size', 'value'),
               Input('slider-polynomial-degree', 'value'),
               Input('slider-dataset-noise', 'value'),
               Input('custom-data-storage', 'children')])
def update_graph(dataset, sample_size, degree, noise_factor, custom_data=None):
    # Generate base data
    if dataset == 'custom':
        custom_data = json.loads(custom_data)
        X_train = np.array(custom_data['train_X']).reshape(-1, 1)
        y_train = np.array(custom_data['train_y'])
        X_test = np.array(custom_data['test_X']).reshape(-1, 1)
        y_test = np.array(custom_data['test_y'])
        X_range = np.linspace(-5, 5, 300).reshape(-1, 1)

        trace_contour = go.Contour(
            x=np.linspace(-5, 5, 300),
            y=np.linspace(-5, 5, 300),
            z=np.ones(shape=(300, 300)),
            showscale=False,
            hoverinfo='none',
            contours=dict(coloring='lines'),
        )
    else:
        X, y = make_dataset(dataset, RANDOM_STATE, sample_size, noise_factor)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=int(X.shape[0] * 0.15), random_state=RANDOM_STATE)

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
    if dataset == 'custom':
        data.insert(0, trace_contour)

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


# Running the server
if __name__ == '__main__':
    app.run_server(port=2522, debug=True)
