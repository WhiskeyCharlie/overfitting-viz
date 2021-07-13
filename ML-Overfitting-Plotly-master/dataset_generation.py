from typing import Tuple

import numpy as np

from generate_regression_data import REG_FUNCTIONS, generate_regression_data


class DatasetGenerator:
    """
    Class handling all dataset generation, can operate "deterministically" or "randomly"; generating either fixed
    datasets or random datasets respectively, given the parameters.
    """
    dataset_name_to_degree = {f'degree_{i}': i for i in range(0, 11)}

    def __init__(self, name: str, random_state: int, sample_size: int, noise_factor: float,
                 out_of_range_proportion=0.025):
        self.__name = name
        self.__random_state = random_state
        self.__sample_size = sample_size
        self.__noise_factor = noise_factor * 0.5  # TODO: make this scaling more sensible, possibly move to make_dataset
        self.__out_of_range_proportion = out_of_range_proportion
        self.__n_out_of_range_samples = round(out_of_range_proportion * sample_size)

    def make_dataset(self, use_random_seed=False) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Create and return a new dataset, possibly setting the numpy seed to be the seed passed at construction
        :param use_random_seed: Boolean, whether to use the seed
        :return: X, y, X_out_of_range, y_out_of_range
        """
        if use_random_seed:
            self.set_seed()

        ds_degree = DatasetGenerator.dataset_name_to_degree[self.__name]
        regression_func = REG_FUNCTIONS[ds_degree]
        return generate_regression_data(regression_func, n_samples=self.__sample_size, noise=self.__noise_factor,
                                        n_out_of_range_samples=self.__n_out_of_range_samples)

    def set_seed(self):
        np.random.seed(self.__random_state)
