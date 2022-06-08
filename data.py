from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import special
from sklearn.model_selection import train_test_split


class DatasetConfig(NamedTuple):
    data_path: str
    window_size: int
    boxcox_lambda: float = 2.5
    split_size: float = 0.3


class Dataset:
    def __init__(self, config: DatasetConfig):
        self.config: DatasetConfig = config
        self.x_train: np.array = None
        self.y_train: np.array = None
        self.x_test: np.array = None
        self.y_test: np.array = None

    def build_dataset(self):
        df = pd.read_csv(self.config.data_path)
        data_shape = df.shape
        data = df.values

        data = self._inv_boxcox(data.reshape(-1))
        data = data.reshape(*data_shape)

        x = self._to_windows(data[:, :-1])
        y = data[:, self.config.window_size :]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.config.split_size
        )

        x_train = x_train.reshape(-1, self.config.window_size)
        y_train = y_train.reshape(-1, 1)
        x_train = x_train[:, np.newaxis, :]

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def _inv_boxcox(self, x: np.array) -> np.array:
        return special.inv_boxcox(x, self.config.boxcox_lambda)

    def _to_windows(self, x: np.array) -> np.array:
        x = np.lib.stride_tricks.sliding_window_view(x, [1, self.config.window_size])
        x = np.squeeze(x)
        return x
