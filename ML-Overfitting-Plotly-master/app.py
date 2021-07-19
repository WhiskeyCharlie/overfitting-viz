"""
The main driver of the app
"""
import dash
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from dataset_generation import DatasetGenerator
from general_utils import format_yhat, get_y_limits
from layout import add_layout_to_app, EXTERNAL_CSS

RANDOM_STATE = 718
TESTING_DATA_PROPORTION = 0.2

app = dash.Dash(__name__, external_stylesheets=EXTERNAL_CSS)
server = app.server
add_layout_to_app(app)


@app.callback(Output('graph-regression-display', 'figure'),
              [Input('dropdown-dataset', 'value'),
               Input('slider-sample-size', 'value'),
               Input('slider-polynomial-degree', 'value'),
               Input('slider-dataset-noise', 'value'),
               Input('resample-btn', 'n_clicks')])
def update_graph(dataset, sample_size, degree, noise_factor, n_clicks=0):
    """
    Function called any time the graph needs to be updated. We redraw the graph from scratch
    :param dataset: Name of the dataset to generate
    :param sample_size: How many points to generate
    :param degree: Degree of the polynomial to fit to the graph
    :param noise_factor: How much noise to add to the data (how much deviation from true function)
    :param n_clicks: How many times has the resample button been pressed
    :return: The figure, essentially the main graph to display
    """
    generator = DatasetGenerator(dataset, sample_size, noise_factor, random_state=RANDOM_STATE)
    X, y, X_out_range, y_out_range = generator.make_dataset()
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=int(X.shape[0] * TESTING_DATA_PROPORTION),
                         random_state=n_clicks or RANDOM_STATE)

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

    return go.Figure(data=data, layout=layout, layout_yaxis_range=get_y_limits(y, y_out_range))


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
    :param chosen_degree: Degree of polynomial user fits to the dataset (draws vertical line)
    :param noise_factor: How much noise to add to data (deviation from the true function)
    :param n_clicks: How many times has the resample button been pressed
    :return: The figure, essentially the main graph to display
    """
    max_degree_to_check = 10
    generator = DatasetGenerator(dataset, sample_size, noise_factor, random_state=RANDOM_STATE)
    X, y, X_out_range, y_out_range = generator.make_dataset()
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=int(X.shape[0] * TESTING_DATA_PROPORTION),
                         random_state=n_clicks or RANDOM_STATE)

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
        out_of_range_test_error = \
            mean_squared_error(y_out_range, model.predict(X_test_out_of_range_poly))
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
        line=dict(width=4),
        visible='legendonly'  # This makes the out-or-range plot off by default, toggle in legend
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
                  annotation=dict(text='Current Degree', textangle=-90,
                                  font=dict(color='rgb(0, 0, 0)'),
                                  yshift=-100))
    return fig


# Running the server
if __name__ == '__main__':
    # NOTE: do not set debug=True in the below function call if you intend to compile it!
    # The executable will crash!
    app.run_server(port=2522, dev_tools_silence_routes_logging=True)
