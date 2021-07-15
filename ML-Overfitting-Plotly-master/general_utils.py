from typing import Tuple

import numpy as np


def get_y_limits(y: np.array, y_out_or_range: np.array) -> Tuple[float, float]:
    """
    Does some transformations to "enlarge" the viewing window by 5% on top and bottom.
    If this function seems too complex, maybe it is :) but it gets good, easy-to-visualise results.
    :param y: y_values in the data
    :param y_out_or_range: out of range y_values in the data
    :return: Appropriate y_limits to show all datapoint
    """
    ys = np.concatenate((y, y_out_or_range))
    y_min = ys.min(initial=np.inf)
    y_max = ys.max(initial=-np.inf)
    old_diff = y_max - y_min
    y_limits = np.array([0, old_diff])  # Start off with the viewing window being [y_min, y_max] shifted down by y_min
    y_limits *= 1.1  # "Stretch" the viewing window to 110% of its original size
    y_limits += y_min  # Move the viewing window so the bottom is at y_min
    y_limits -= old_diff * 0.05  # Move the window down a further 5% of the old difference (half of what was added)
    return y_limits[0], y_limits[1]


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
            coefficient_string += f'{coefficient:.3f}'
        elif order == 1:
            coefficient_string += sign + f'{abs(coefficient):.3f}*x'
        else:
            coefficient_string += sign + f'{abs(coefficient):.3f}*x^{order}'

    return coefficient_string
