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
    min_sample_size = 10

    def __init__(self, name: str, sample_size: int, noise_factor: float,
                 out_of_range_proportion=0.025, random_state=None):
        self.__name = name
        # Random number generator (rng) for all randomness
        self.__rng = np.random.default_rng(random_state or None)
        # The below or ensures that if the frontend passes None as sample_size, we don't crash
        self.__sample_size = sample_size or self.min_sample_size
        # Right now, 0.5 is arbitrary so let's fix this in future
        self.__noise_factor = noise_factor * 0.5
        self.__out_of_range_proportion = out_of_range_proportion
        # Max with 2 to ensure at least 1 out-of-range point on either side of the function
        self.__n_out_of_range_samples = max(2, round(self.__out_of_range_proportion *
                                                     self.__sample_size))
        self.__regression_functions = self.__generate_list_of_regression_functions()
        self.__lst_features: np.array = None
        self.__lst_features_out_of_range: np.array = None
        self.__evaluations: np.array = None
        self.__evaluations_out_of_range: np.array = None
        self.__has_data = False

    def make_dataset(self) -> 'DatasetGenerator':
        """
        Create and store internally a new dataset
        :return: self to allow for chaining
        """

        ds_degree = DatasetGenerator.dataset_name_to_degree[self.__name]
        regression_func = self.__regression_functions[ds_degree]
        self.__generate_regression_data(regression_func)
        return self

    def introduce_noise(self) -> 'DatasetGenerator':
        """
        Add noise to the data most recently created by this object.
        Noise amount is determined both by the "noise_factor" parameter
        passed at construction time, and the dataset itself.
        :return: self for chaining
        """
        assert self.__has_data, 'make_dataset must be called before introduce_noise'
        ds_degree = DatasetGenerator.dataset_name_to_degree[self.__name]
        std_of_in_range_data = np.std(self.__evaluations) if ds_degree != 0 else 0.1
        scale_of_noise = self.__noise_factor * std_of_in_range_data * 0.1
        noise_sample = self.__rng.normal(loc=0, scale=scale_of_noise, size=self.__sample_size)
        size = len(self.__evaluations_out_of_range)
        noise_sample_out_of_range = self.__rng.normal(loc=0, scale=scale_of_noise, size=size)
        self.__evaluations += noise_sample
        self.__evaluations_out_of_range += noise_sample_out_of_range

        return self

    def get_dataset(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Get (a copy of) the dataset stored internally in this object.
        :return: X, y (possibly with noise), X_out_of_range, y_out_of_range (possibly with noise)
        """
        assert self.__has_data, 'make_dataset must be called before get_dataset'
        return np.copy(self.__lst_features), np.copy(self.__evaluations), \
               np.copy(self.__lst_features_out_of_range), np.copy(self.__evaluations_out_of_range)

    def __generate_regression_data(self, polynomial: np.polynomial.Polynomial) -> \
            Tuple[np.array, np.array, np.array, np.array]:
        """
        Generates the actual data for to fit WITHOUT ANY NOISE!
        :param polynomial: Polynomial to generate the data for
        :return: X, y, X_out_of_range, y_out_of_range
        """

        lst_features = np.sort(self.__rng.uniform(-2, 2, size=self.__sample_size)).reshape(-1, 1)
        samples_per_side = self.__n_out_of_range_samples // 2
        lower_half = np.sort(self.__rng.uniform(-2.125, -2, size=samples_per_side))
        upper_half = np.sort(self.__rng.uniform(2, 2.125, size=samples_per_side))
        lst_features_out_of_range = np.concatenate((lower_half, upper_half)).reshape(-1, 1)

        evaluations, norm = DatasetGenerator.__eval_polynomial(polynomial, lst_features, normalize=True)
        evaluations_out_of_range, _ = DatasetGenerator.__eval_polynomial(polynomial,
                                                                         lst_features_out_of_range)
        evaluations_out_of_range /= norm

        self.__lst_features = lst_features
        self.__evaluations = evaluations
        self.__lst_features_out_of_range = lst_features_out_of_range
        self.__evaluations_out_of_range = evaluations_out_of_range
        self.__has_data = True

        return lst_features, evaluations, lst_features_out_of_range, evaluations_out_of_range

    def __generate_list_of_regression_functions(self, number_of_functions=11) -> \
            List[np.polynomial.Polynomial]:
        """
        Make a list of random polynomial functions against which data will be generated.
        :param number_of_functions: How many functions should we generate (In general don't modify)
        :return: list of numpy Polynomials ranging from degree 1 to number_of_functions (inclusive)
        """
        return [np.polynomial.Polynomial(
            np.polynomial.polynomial.polyfromroots(self.__rng.uniform(-2, 2, size=i)))
            for i in range(0, number_of_functions)]

    @staticmethod
    def __eval_polynomial(polynomial: np.polynomial.Polynomial, values: np.array, normalize=False) -> np.array:
        """
        Evaluate the given polynomial at all points in values given (vectorized)
        :param polynomial: Given polynomial
        :param values: Values at which to evaluate polynomial
        :return: Numpy array of Evaluations, cast to np.float64
        """
        evaluations = polynomial(values).squeeze().astype(np.float64)
        divide_by = 1
        if normalize:
            divide_by = np.max(np.abs(evaluations))
        return (2 * evaluations / divide_by), divide_by
