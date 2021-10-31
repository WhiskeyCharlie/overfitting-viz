"""
The main driver of the app
"""
import os
import sys
from pathlib import Path

import dash
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from dataset_generation import DatasetGenerator
from general_utils import format_yhat, get_y_limits, form_error_bars_from_x_y
from layout import add_layout_to_app, EXTERNAL_CSS, APP_TITLE

RANDOM_STATE = 718
TESTING_DATA_PROPORTION = 0.2
NUM_RESAMPLES_TO_DO = 10
MIN_SAMPLE_SIZE = DatasetGenerator.min_sample_size

# Check if the code is running "inside" an executable file
if getattr(sys, 'frozen', False):
    ASSETS_FOLDER = Path(getattr(sys, '_MEIPASS', '.')) / 'assets'
else:
    ASSETS_FOLDER = str(Path(__file__).absolute().parent.parent / 'assets')

# This helps work in various environments
IP_TO_LISTEN_ON = os.getenv('IP_TO_LISTEN_ON', '127.0.0.1')
PORT = 2522

app = dash.Dash(__name__, external_stylesheets=EXTERNAL_CSS, assets_folder=ASSETS_FOLDER,
                title=APP_TITLE)
server = app.server
add_layout_to_app(app)


@app.callback(Output('graph-regression-display', 'figure'),
              Output('session', 'data'),
              [Input('dropdown-dataset', 'value'),
               Input('slider-sample-size', 'value'),
               Input('slider-polynomial-degree', 'value'),
               Input('slider-dataset-noise', 'value'),
               Input('resample-btn', 'n_clicks'),
               Input('session', 'data')]
              )
def update_graph(dataset, sample_size, degree, noise_factor, n_clicks, session):
    """
    Function called any time the graph needs to be updated. We redraw the graph from scratch
    :param session: Dictionary stored client-side to keep track of randomness
    :param dataset: Name of the dataset to generate
    :param sample_size: How many points to generate
    :param degree: Degree of the polynomial to fit to the graph
    :param noise_factor: How much noise to add to the data (how much deviation from true function)
    :param n_clicks: How many times has the resample button been pressed
    :return: The figure, essentially the main graph to display
    """

    if None in [sample_size, noise_factor] or sample_size < MIN_SAMPLE_SIZE:
        raise PreventUpdate
    context = dash.callback_context
    if session is None:
        session = {'random-state': RANDOM_STATE}
    # If we're updating because the dropdown-dataset was changed, select new randomness for the polynomials
    if context.triggered and context.triggered[0]['prop_id'].split('.')[0] == 'dropdown-dataset':
        session['random-state'] = np.random.randint(0, 1_000_000)

    generator = DatasetGenerator(dataset, sample_size, noise_factor,
                                 random_state=session.get('random-state', RANDOM_STATE))
    x_values, y_values, x_values_out_of_range, y_values_out_range = \
        generator.make_dataset().introduce_noise().get_dataset()
    x_train, x_test, y_train, y_test = \
        train_test_split(x_values, y_values,
                         test_size=int(x_values.shape[0] * TESTING_DATA_PROPORTION),
                         random_state=n_clicks or RANDOM_STATE)

    x_range = np.linspace(min(x_values.min(),
                              x_values_out_of_range.min()) - 0.5,
                          max(x_values.max(), x_values_out_of_range.max()) + 0.5,
                          sample_size).reshape(-1, 1)

    # Create Polynomial Features so that linear regression is actually polynomial regression
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)
    poly_range = poly.fit_transform(x_range)

    model = LinearRegression()

    # Train model and predict
    model.fit(x_train_poly, y_train)
    y_pred_range = model.predict(poly_range)
    train_error = mean_squared_error(y_train, model.predict(x_train_poly))
    test_error = mean_squared_error(y_test, model.predict(x_test_poly))

    # Create figure
    trace_train_in_range = go.Scatter(
        x=x_train.squeeze(),
        y=y_train,
        name='Training Data',
        mode='markers',
        opacity=0.7,
        marker=dict(size=12)
    )
    trace_test_in_range = go.Scatter(
        x=x_test.squeeze(),
        y=y_test,
        name='Test Data',
        mode='markers',
        opacity=0.7,
        marker=dict(size=12)
    )

    trace_test_out_range = go.Scatter(
        x=x_values_out_of_range.squeeze(),
        y=y_values_out_range,
        name='Out Of Range Test Data',
        mode='markers',
        opacity=0.7,
        marker=dict(size=12, color='yellow')
    )

    trace_prediction = go.Scatter(
        x=x_range.squeeze(),
        y=y_pred_range,
        name='Prediction',
        mode='lines',
        hovertext=format_yhat(model),
        marker=dict(color='#27ab22'),
        line=dict(width=4)
    )
    data = [trace_train_in_range, trace_test_in_range, trace_test_out_range, trace_prediction]
    rounded_train_error = round(train_error, 3)
    rounded_test_error = round(test_error, 3)

    if min(rounded_test_error, rounded_train_error) < 0.0001:
        rounded_test_error = f'{test_error:.2e}'
        rounded_train_error = f'{train_error:.2e}'
    inequality_symbol = '>' if train_error > test_error else '<'
    layout = go.Layout(
        title=f"MSE: {rounded_train_error} (Train Data) "
              f"{inequality_symbol} MSE: {rounded_test_error} (Test Data)",
        legend=dict(orientation='h'),
        margin=dict(l=25, r=25),
        hovermode='closest',
        plot_bgcolor="#cbd3f2",
        paper_bgcolor="#282b38",
        font=dict(color='rgb(200, 200, 200)', size=15)

    )

    return go.Figure(data=data, layout=layout,
                     layout_yaxis_range=get_y_limits(y_values, y_values_out_range)), session


@app.callback(Output('graph-fitting-display', 'figure'),
              [Input('dropdown-dataset', 'value'),
               Input('slider-sample-size', 'value'),
               Input('slider-polynomial-degree', 'value'),
               Input('slider-dataset-noise', 'value'),
               Input('session', 'data')])
def update_fitting_graph(dataset, sample_size, chosen_degree, noise_factor, session):
    """
    Function called any time the graph needs to be updated. We redraws the graph from scratch
    :param session: Dictionary stored client-side to keep track of randomness
    :param dataset: Name of the dataset to generate
    :param sample_size: How many points to generate
    :param chosen_degree: Degree of polynomial user fits to the dataset (draws vertical line)
    :param noise_factor: How much noise to add to data (deviation from the true function)
    :return: The figure, essentially the main graph to display
    """
    if None in [sample_size, noise_factor] or sample_size < MIN_SAMPLE_SIZE:
        raise PreventUpdate
    max_degree_to_check = 10

    if session is None:
        session = {'random-state': RANDOM_STATE}

    degrees = np.array(range(1, max_degree_to_check + 1))
    error_data = {'train': [], 'test': [], 'out-of-range': []}
    generator = DatasetGenerator(dataset, sample_size, noise_factor,
                                 random_state=session.get('random-state', RANDOM_STATE))

    for i in range(NUM_RESAMPLES_TO_DO):
        x_values, y_values, x_values_out_range, y_out_range = \
            generator.make_dataset().introduce_noise().get_dataset()
        x_train, x_test, y_train, y_test = \
            train_test_split(x_values, y_values,
                             test_size=int(x_values.shape[0] * TESTING_DATA_PROPORTION),
                             random_state=i + 1)
        train_errors = []
        test_errors = []
        out_of_range_test_errors = []
        for deg in degrees:
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            x_train_poly = poly.fit_transform(x_train)
            x_test_poly = poly.transform(x_test)
            x_test_out_of_range_poly = poly.transform(x_values_out_range)

            model = LinearRegression()

            # Train model and predict
            model.fit(x_train_poly, y_train)
            train_error = mean_squared_error(y_train, model.predict(x_train_poly))
            test_error = mean_squared_error(y_test, model.predict(x_test_poly))
            out_of_range_test_error = \
                mean_squared_error(y_out_range, model.predict(x_test_out_of_range_poly))
            train_errors.append(train_error)
            test_errors.append(test_error)
            out_of_range_test_errors.append(out_of_range_test_error)
        error_data['train'].append(train_errors)
        error_data['test'].append(test_errors)
        error_data['out-of-range'].append(out_of_range_test_errors)

    mean_train_errors = np.mean(error_data['train'], axis=0)
    mean_test_errors = np.mean(error_data['test'], axis=0)
    mean_out_of_range_errors = np.mean(error_data['out-of-range'], axis=0)

    std_train_errors = np.std(error_data['train'], axis=0)
    std_test_errors = np.std(error_data['test'], axis=0)
    std_out_of_range_errors = np.std(error_data['out-of-range'], axis=0)

    trace_train = go.Scatter(
        x=degrees,
        y=mean_train_errors,
        name='MSE',
        legendgroup='Train',
        legendgrouptitle=dict(text='Training', font=dict(color='black')),
        opacity=0.7,
        marker=dict(color='blue'),
        line=dict(width=4)
    )
    train_error_x, train_error_y = form_error_bars_from_x_y(degrees, mean_train_errors,
                                                            std_train_errors)
    trace_train_error = go.Scatter(
        x=train_error_x,
        y=train_error_y,
        name='1 std',
        legendgroup='Train',
        fill='toself',
        fillcolor='blue',
        opacity=0.2,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip'
    )

    trace_test = go.Scatter(
        x=degrees,
        y=mean_test_errors,
        name='MSE',
        legendgroup='Test',
        legendgrouptitle=dict(text='Testing', font=dict(color='black')),
        opacity=0.7,
        marker=dict(color='red'),
        line=dict(width=4)
    )
    test_error_x, test_error_y = form_error_bars_from_x_y(degrees, mean_test_errors,
                                                          std_test_errors)
    trace_test_error = go.Scatter(
        x=test_error_x,
        y=test_error_y,
        name='1 std',
        legendgroup='Test',
        fill='toself',
        fillcolor='red',
        opacity=0.2,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip'
    )
    trace_test_out_of_range = go.Scatter(
        x=degrees,
        y=mean_out_of_range_errors,
        name='OOR Test',
        legendgroup='Test OOR',
        legendgrouptitle=dict(text='Testing OOR', font=dict(color='black')),
        opacity=0.7,
        marker=dict(color='yellow'),
        line=dict(width=4),
        visible='legendonly'  # This makes the out-or-range plot off by default, toggle in legend
    )

    test_oor_error_x, test_oor_error_y = form_error_bars_from_x_y(degrees,
                                                                  mean_out_of_range_errors,
                                                                  std_out_of_range_errors)
    trace_test_out_of_range_error = go.Scatter(
        x=test_oor_error_x,
        y=test_oor_error_y,
        name='OOR Test 1 std',
        legendgroup='Test OOR',
        fill='toself',
        fillcolor='yellow',
        opacity=0.2,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        visible='legendonly'
    )

    # noinspection PyTypeChecker
    layout = go.Layout(
        title='MSE Behavior vs. Polynomial Degree',
        legend=dict(orientation='v',
                    yanchor="top",
                    y=-0.3,
                    xanchor="left",
                    x=0.01,
                    bgcolor='#cbd3f2',
                    font=dict(color='black')),
        margin=dict(l=25, r=25),
        hovermode='closest',
        plot_bgcolor="#cbd3f2",
        paper_bgcolor="#282b38",
        font=dict(color='rgb(200, 200, 200)', size=15),
        xaxis=dict(tickvals=degrees),
        xaxis_title='Polynomial Degree',
        yaxis_title='Mean Squared Error'
    )
    jitter = 0.05
    concept_degree = DatasetGenerator.dataset_name_to_degree[dataset]
    max_y_val = max(max(mean_test_errors + std_test_errors),
                    max(mean_train_errors + std_train_errors))

    # noinspection PyTypeChecker
    concept_vertical_line = go.Scatter(
        x=(concept_degree, concept_degree),
        y=(0, max_y_val),
        name='Dataset Degree',
        marker=dict(color='black'),
        line_dash='dash',
        legendgroup='Misc.'
        )

    fig = go.Figure(data=[trace_train, trace_train_error, trace_test, trace_test_error,
                          trace_test_out_of_range, trace_test_out_of_range_error,
                          concept_vertical_line], layout=layout)

    fig.add_vline(x=chosen_degree + jitter, line_width=3, line_color='#27ab22',
                  annotation=dict(text='Model Degree', textangle=-90,
                                  font=dict(color='rgb(0, 0, 0)'),
                                  yshift=-50))
    return fig


# Running the server
if __name__ == '__main__':
    # NOTE: do not set debug=True in the below function call if you intend to compile it!
    # The executable will crash!
    app.run_server(host=IP_TO_LISTEN_ON, port=PORT, dev_tools_silence_routes_logging=True)
