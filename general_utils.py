"""
This is where odd-one-out functions go
"""

from typing import Tuple

import numpy as np


def get_y_limits(y_arr: np.array, y_arr_out_or_range: np.array) -> Tuple[float, float]:
    """
    Does some transformations to "enlarge" the viewing window by 5% on top and bottom.
    If this function seems too complex, maybe it is :) but it gets good, easy-to-visualise results.
    :param y_arr: y_values in the data
    :param y_arr_out_or_range: out of range y_values in the data
    :return: Appropriate y_limits to show all datapoint
    """
    y_all = np.concatenate((y_arr, y_arr_out_or_range))
    y_min = np.min(y_all, initial=np.inf)
    y_max = np.max(y_all, initial=-np.inf)
    old_diff = y_max - y_min
    # Start off with the viewing window being [y_min, y_max] shifted down by y_min
    y_limits = np.array([0, old_diff])
    # "Stretch" the viewing window to 110% of its original size
    y_limits *= 1.1
    # Move the viewing window so the bottom is at y_min
    y_limits += y_min
    # Move the window down a further 5% of the old difference (half of what was added)
    y_limits -= old_diff * 0.05
    return y_limits[0], y_limits[1]


def format_yhat(model):
    """
    Make a textual representation of the passed model
    :param model: sklearn LinearRegression object
    :return: str representing the model passed
    """
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
            coefficient_string += f'{coefficient:.3f}'
        elif order == 1:
            coefficient_string += sign + f'{abs(coefficient):.3f}*x'
        else:
            coefficient_string += sign + f'{abs(coefficient):.3f}*x^{order}'

    return coefficient_string


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
