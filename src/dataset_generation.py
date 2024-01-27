"""
Home of anything pertaining to generating the dataset, including noise generation
"""
import logging
from typing import NamedTuple

import numpy as np
import numpy.polynomial.polynomial as poly
from sklearn.model_selection import train_test_split


TESTING_DATA_PROPORTION = 0.2


class HalfDataset(NamedTuple):
    all_values: np.array
    train: np.array
    test: np.array
    out_of_range: np.array


class Dataset(NamedTuple):
    x: HalfDataset
    y: HalfDataset


class DatasetGenerator:
    """
    Class handling all dataset generation, can operate "deterministically" or "randomly";
    generating either fixed datasets or random datasets respectively, given the parameters.
    """
    dataset_name_to_degree = {f'degree_{i}': i for i in range(0, 11)}
    min_sample_size = 10

    def __init__(self, name: str, sample_size: int, noise_factor: float,
                 out_of_range_proportion=0.025, random_state=None):
        self._name = name

        # Random number generator (rng) for all randomness
        self._rng = np.random.default_rng(random_state or None)

        # The below or ensures that if the frontend passes None as sample_size, we don't crash
        self._sample_size = sample_size or self.min_sample_size

        # TODO: 0.5 is arbitrary so let's fix this in future
        self._noise_factor = noise_factor * 0.5

        self._out_of_range_proportion = out_of_range_proportion

        # Max with 2 to ensure at least 1 out-of-range point on either side of the function
        self._n_out_of_range_samples = max(2, round(self._out_of_range_proportion *
                                                    self._sample_size))

        # NOTE: 'oor' is 'out of range', i.e. data occurring outside the desired window of (-2, 2)
        self._x_values: np.array = None
        self._x_values_oor: np.array = None
        self._y_values: np.array = None
        self._y_values_oor: np.array = None

    def make_dataset(self) -> 'DatasetGenerator':
        """
        Create and store a new dataset
        :return: self to allow for chaining
        """

        ds_degree = DatasetGenerator.dataset_name_to_degree[self._name]
        regression_func = self._generate_regression_function(ds_degree)
        self._generate_regression_data(regression_func)
        return self

    def introduce_noise(self) -> 'DatasetGenerator':
        """
        Add noise to the data most recently created by this object.
        Noise amount is determined both by the "noise_factor" parameter
        passed at construction time, and the dataset itself.
        :return: self for chaining
        """
        self._ensure_data_exists()
        ds_degree = DatasetGenerator.dataset_name_to_degree[self._name]
        std_of_in_range_data = np.std(self._y_values) if ds_degree != 0 else 0.1
        scale_of_noise = self._noise_factor * std_of_in_range_data * 0.1
        noise_sample = self._rng.normal(loc=0, scale=scale_of_noise, size=self._sample_size)
        size = len(self._y_values_oor)
        noise_sample_out_of_range = self._rng.normal(loc=0, scale=scale_of_noise, size=size)
        self._y_values += noise_sample
        self._y_values_oor += noise_sample_out_of_range

        return self

    def get_dataset(self, train_test_split_randomness: int | None = None) -> Dataset:
        """
        :param train_test_split_randomness: determines how the data gets divided into train/test sets
        :return Dataset object containing this objects data split into train/test sets
        """
        x_values, y_values, x_out_range, y_out_range =\
            self.make_dataset().introduce_noise()._get_dataset()

        x_train, x_test, y_train, y_test = \
            train_test_split(x_values, y_values,
                             test_size=int(x_values.shape[0] * TESTING_DATA_PROPORTION),
                             random_state=train_test_split_randomness)
        return Dataset(x=HalfDataset(all_values=x_values, train=x_train, test=x_test, out_of_range=x_out_range),
                       y=HalfDataset(all_values=y_values, train=y_train, test=y_test, out_of_range=y_out_range))

    def _get_dataset(self) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Get (a copy of) the dataset stored internally in this object.
        :return: X, y (possibly with noise), X_out_of_range, y_out_of_range (possibly with noise)
        """
        self._ensure_data_exists()
        return np.copy(self._x_values), np.copy(self._y_values), \
            np.copy(self._x_values_oor), np.copy(self._y_values_oor)

    def _generate_regression_data(self, polynomial: np.polynomial.Polynomial) -> \
            tuple[np.array, np.array, np.array, np.array]:
        """
        Generates points from the given polynomial, divided into in-range and out-of-range
        :param polynomial: Polynomial from which points will be sampled
        :return: X, y, X_out_of_range, y_out_of_range
        """

        x_values = np.sort(self._rng.uniform(-2, 2, size=self._sample_size)).reshape(-1, 1)
        # NOTE: 'oor' is 'out of range', i.e. data occurring outside the desired window of (-2, 2)
        oor_samples_per_side = self._n_out_of_range_samples // 2
        left_oor_x_values = np.sort(self._rng.uniform(-2.125, -2, size=oor_samples_per_side))
        right_oor_x_values = np.sort(self._rng.uniform(2, 2.125, size=oor_samples_per_side))
        all_oor_x_values = np.concatenate((left_oor_x_values, right_oor_x_values)).reshape(-1, 1)

        evaluations, norm = DatasetGenerator._eval_polynomial(polynomial, x_values, normalize=True)
        evaluations_out_of_range, _ = DatasetGenerator._eval_polynomial(polynomial, all_oor_x_values)
        evaluations_out_of_range /= norm

        self._x_values = x_values
        self._y_values = evaluations
        self._x_values_oor = all_oor_x_values
        self._y_values_oor = evaluations_out_of_range

        return x_values, evaluations, all_oor_x_values, evaluations_out_of_range

    @property
    def has_data(self) -> bool:
        """
        :return: True if all data this object requires has been generated otherwise false
        """
        data = [
            self._x_values, self._x_values_oor,
            self._y_values, self._y_values_oor
        ]
        return all(d is not None for d in data)

    def _generate_regression_function(self, degree) -> np.polynomial.Polynomial:
        """
        Create a random numpy polynomial.
        This object uses polynomials to generate random datasets of points that appear along a "target" polynomial
        """
        sign = self._rng.choice([-1, 1])  # we will randomly multiply the polynomial by +1 or -1
        return poly.Polynomial(poly.polyfromroots(self._rng.uniform(-2, 2, size=degree))) * sign

    @staticmethod
    def _eval_polynomial(polynomial: np.polynomial.Polynomial, values: np.array,
                         normalize=False) -> tuple[np.array, float]:
        """
        Evaluate the given polynomial at all points in values given (evaluation is vectorized)
        :param polynomial: Given polynomial
        :param values: Values at which to evaluate polynomial
        :param normalize Determines whether the evaluations will be scaled to the range [-1, 1]
        :return: both the evaluations and the amount by which the evaluations were scaled
        """
        evaluations = polynomial(values).squeeze().astype(np.float64)
        divide_by = 1.0
        if normalize:
            divide_by = np.max(np.abs(evaluations))
        return (2 * evaluations / divide_by), divide_by

    def _ensure_data_exists(self):
        """
        If required data has not already been generated, log a warning and generate it.
        The warning indicates misuse of this object, but we can infer the user's intent
        """
        if not self.has_data:
            logger = logging.getLogger()
            logger.warning('Data does not exist. Generating now')
            self.make_dataset()
