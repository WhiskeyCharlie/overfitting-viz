from typing import List, Tuple

import numpy as np


def generate_list_of_regression_functions(number_of_functions=11) -> List[np.polynomial.Polynomial]:
    """
    Make a list of random polynomial functions against which data will be generated.
    :param number_of_functions: How many functions should we generate (In general, don't change this)
    :return: A list of numpy Polynomials ranging from degree 1 to number_of_functions (inclusively)
    """
    return [np.polynomial.Polynomial(np.random.uniform(-1, 1, size=i)) for i in range(1, number_of_functions + 1)]


REG_FUNCTIONS = generate_list_of_regression_functions()


def eval_polynomial(polynomial: np.polynomial.Polynomial, values: np.array) -> np.array:
    """
    Evaluate the given polynomial at all points in values given (vectorized)
    :param polynomial: Given polynomial
    :param values: Values at which to evaluate polynomial
    :return: Numpy array of Evaluations, cast to np.float64
    """
    return polynomial(values).squeeze().astype(np.float64)


def generate_regression_data(polynomial: np.polynomial.Polynomial, n_samples=100, noise=0.0,
                             n_out_of_range_samples=10) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Generates the actual data for to fit including (possibly) noise
    :param polynomial: Polynomial to generate the data for
    :param n_samples: Number of points to generate
    :param noise: Amount of gaussian noise to introduce
    :param n_out_of_range_samples: How many out of range samples to add to the data (points appearing on the extremes)
    :return: X, y, X_out_of_range, y_out_of_range
    """

    lst_features = [np.sort(np.random.uniform(-2, 2, size=n_samples))]
    samples_per_side = n_out_of_range_samples // 2
    lst_features_out_of_range = [np.concatenate((np.sort(np.random.uniform(-2.25, -2, size=samples_per_side)),
                                                 np.sort(np.random.uniform(2, 2.25, size=samples_per_side))))]

    lst_features = np.array(lst_features).T

    lst_features_out_of_range = np.array(lst_features_out_of_range).T

    evaluations = eval_polynomial(polynomial, lst_features)
    evaluations_out_of_range = eval_polynomial(polynomial, lst_features_out_of_range)

    noise_sample = np.random.normal(loc=0, scale=noise, size=n_samples)
    noise_sample_out_of_range = np.random.normal(loc=0, scale=noise, size=len(evaluations_out_of_range))

    evaluations = evaluations + noise_sample
    evaluations_out_of_range = evaluations_out_of_range + noise_sample_out_of_range

    return lst_features, evaluations, lst_features_out_of_range, evaluations_out_of_range
