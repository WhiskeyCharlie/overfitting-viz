"""
Useful functions that don't belong anywhere else
"""
import os
from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from dataset_generation import DatasetGenerator


NUM_RESAMPLES_TO_DO = os.getenv('NUM_RESAMPLES_TO_DO', default=10)


def get_y_limits(y_arr: np.array, y_arr_out_or_range: np.array) -> Tuple[float, float]:
    """
    Finds the edges of a "window" around data such that there is a 5% empty region around the data
    :param y_arr: y_values in the data
    :param y_arr_out_or_range: out of range y_values in the data
    :return: Appropriate y_limits to show all datapoint
    """
    lowest_val = np.minimum(np.amin(y_arr, initial=np.inf), np.amin(y_arr_out_or_range, initial=np.inf))
    highest_val = np.maximum(np.amax(y_arr, initial=-np.inf), np.amax(y_arr_out_or_range, initial=-np.inf))
    delta = highest_val - lowest_val

    distance_to_stretch = delta * 0.05
    return lowest_val - distance_to_stretch, highest_val + distance_to_stretch


def format_yhat(model):
    """
    Make a textual representation of the passed model
    :param model: sklearn LinearRegression object
    :return: str representing the model passed
    """
    coefficients = model.coef_
    intercept = model.intercept_
    formula_components = []
    variable_powers = dict([(1, 'x')] + [(i, f'x^{i}') for i in range(2, len(coefficients) + 1)])
    for order, coefficient in enumerate(coefficients, start=1):
        sign = '-' if coefficient < 0 else '+'
        formula_components.append(f'{sign} {abs(coefficient):.2f}*{variable_powers[order]}')
    return f'y = {intercept:.2f} ' + ' '.join(formula_components)


def form_error_bars_from_x_y(x_array: np.array, y_array: np.array, std_array: np.array) -> \
        Tuple[np.array, np.array]:
    """
    Creates the x and y values for the continuous error bars around lines.
    Lines will not go below zero.
    :param x_array: values along x-axis
    :param y_array: heights above x-axis
    :param std_array: standard deviation of each point.
    :return: x_values, y_values compliant with continuous error bar generation for go.Scatter
    """
    x_values = np.concatenate((x_array, x_array[::-1]))
    y_values = np.concatenate((y_array + std_array, np.clip((y_array - std_array)[::-1], 0, None)))
    return x_values, y_values


def multiple_model_error_data(degrees: np.array, generator: DatasetGenerator) -> dict[str, list]:
    error_data = {'train': [], 'test': [], 'out-of-range': []}
    for i in range(NUM_RESAMPLES_TO_DO):
        dataset = generator.get_dataset(train_test_split_randomness=i + 1)
        train_errors = []
        test_errors = []
        out_of_range_test_errors = []
        for deg in degrees:
            out_of_range_test_error, test_error, train_error = train_model(dataset, deg)
            train_errors.append(train_error)
            test_errors.append(test_error)
            out_of_range_test_errors.append(out_of_range_test_error)
        error_data['train'].append(train_errors)
        error_data['test'].append(test_errors)
        error_data['out-of-range'].append(out_of_range_test_errors)
    return error_data


def train_model(dataset, deg):
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    x_train_poly = poly.fit_transform(dataset.x.train)
    x_test_poly = poly.transform(dataset.x.test)
    x_test_out_of_range_poly = poly.transform(dataset.x.out_of_range)
    model = LinearRegression()
    # Train model and predict
    model.fit(x_train_poly, dataset.y.train)
    train_error = mean_squared_error(dataset.y.train, model.predict(x_train_poly))
    test_error = mean_squared_error(dataset.y.test, model.predict(x_test_poly))
    out_of_range_test_error = \
        mean_squared_error(dataset.y.out_of_range, model.predict(x_test_out_of_range_poly))
    return out_of_range_test_error, test_error, train_error
