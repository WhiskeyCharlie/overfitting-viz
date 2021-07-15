import numpy as np


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
