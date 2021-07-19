"""
Home of anything pertaining to generating the dataset, including noise generation
"""
from typing import Tuple, List

import numpy as np


class DatasetGenerator:
    """
    Class handling all dataset generation, can operate "deterministically" or "randomly";
    generating either fixed datasets or random datasets respectively, given the parameters.
    """
    dataset_name_to_degree = {f'degree_{i}': i for i in range(0, 11)}

    def __init__(self, name: str, sample_size: int, noise_factor: float,
                 out_of_range_proportion=0.025, random_state=None):
        self.__name = name
        # Random number generator (rng) for all randomness
        self.__rng = np.random.default_rng(random_state)
        self.__sample_size = sample_size
        # Right now, 0.5 is arbitrary so let's fix this in future
        self.__noise_factor = noise_factor * 0.5
        self.__out_of_range_proportion = out_of_range_proportion
        # Max with 2 to ensure at least 1 out-of-range point on either side of the function
        self.__n_out_of_range_samples = max(2, round(self.__out_of_range_proportion * sample_size))
        self.__regression_functions = self.__generate_list_of_regression_functions()

    def make_dataset(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Create and return a new dataset
        :return: X, y, X_out_of_range, y_out_of_range
        """

        ds_degree = DatasetGenerator.dataset_name_to_degree[self.__name]
        regression_func = self.__regression_functions[ds_degree]
        return self.__generate_regression_data(regression_func)

    def __generate_regression_data(self, polynomial: np.polynomial.Polynomial) -> \
            Tuple[np.array, np.array, np.array, np.array]:
        """
        Generates the actual data for to fit including (possibly) noise
        :param polynomial: Polynomial to generate the data for
        :return: X, y, X_out_of_range, y_out_of_range
        """

        lst_features = np.sort(self.__rng.uniform(-2, 2, size=self.__sample_size)).reshape(-1, 1)
        samples_per_side = self.__n_out_of_range_samples // 2
        lower_half = np.sort(self.__rng.uniform(-2.25, -2, size=samples_per_side))
        upper_half = np.sort(self.__rng.uniform(2, 2.25, size=samples_per_side))
        lst_features_out_of_range = np.concatenate((lower_half, upper_half)).reshape(-1, 1)

        evaluations = DatasetGenerator.__eval_polynomial(polynomial, lst_features)
        evaluations_out_of_range = DatasetGenerator.__eval_polynomial(polynomial,
                                                                      lst_features_out_of_range)

        noise_sample = self.__rng.normal(loc=0, scale=self.__noise_factor, size=self.__sample_size)
        noise_sample_out_of_range = self.__rng.normal(loc=0, scale=self.__noise_factor,
                                                      size=len(evaluations_out_of_range))

        evaluations = evaluations + noise_sample
        evaluations_out_of_range = evaluations_out_of_range + noise_sample_out_of_range

        return lst_features, evaluations, lst_features_out_of_range, evaluations_out_of_range

    def __generate_list_of_regression_functions(self, number_of_functions=11) -> \
            List[np.polynomial.Polynomial]:
        """
        Make a list of random polynomial functions against which data will be generated.
        :param number_of_functions: How many functions should we generate (In general don't modify)
        :return: list of numpy Polynomials ranging from degree 1 to number_of_functions (inclusive)
        """
        return [np.polynomial.Polynomial(self.__rng.uniform(-1, 1, size=i+1)) for i in
                range(0, number_of_functions)]

    @staticmethod
    def __eval_polynomial(polynomial: np.polynomial.Polynomial, values: np.array) -> np.array:
        """
        Evaluate the given polynomial at all points in values given (vectorized)
        :param polynomial: Given polynomial
        :param values: Values at which to evaluate polynomial
        :return: Numpy array of Evaluations, cast to np.float64
        """
        return polynomial(values).squeeze().astype(np.float64)
