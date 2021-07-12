import numpy as np

from generate_regression_data import reg_functions, gen_regression_symbolic


class DatasetGenerator:
    dataset_name_to_degree = {f'degree_{i}': i for i in range(0, 11)}

    def __init__(self, name, random_state, sample_size, noise_factor, out_of_range_proportion=0.025):
        self.__name = name
        self.__random_state = random_state
        self.__sample_size = sample_size
        self.__noise_factor = noise_factor * 0.5  # TODO: make this scaling more sensible, possibly move to make_dataset
        self.__out_of_range_proportion = out_of_range_proportion
        self.__n_out_of_range_samples = round(out_of_range_proportion * sample_size)

    def make_dataset(self, use_random_seed=False):
        if use_random_seed:
            self.set_seed()

        ds_degree = DatasetGenerator.dataset_name_to_degree[self.__name]
        regression_func = reg_functions[ds_degree]
        return gen_regression_symbolic(m=regression_func, n_samples=self.__sample_size, noise=self.__noise_factor,
                                       n_out_of_range_samples=self.__n_out_of_range_samples)

    def set_seed(self):
        np.random.seed(self.__random_state)
