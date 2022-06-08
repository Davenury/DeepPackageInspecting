from data import DatasetConfig, Dataset
from model import TrafficModelConfig, TrafficModel

WINDOW_SIZE = 2
BOXCOX_LAMBDA = 2.5


def build_model_and_dataset():
    dataset_config = DatasetConfig(
        data_path="../datasets/zh_df.csv",
        window_size=WINDOW_SIZE,
        boxcox_lambda=BOXCOX_LAMBDA,
        split_size=0.3,
    )

    model_config = TrafficModelConfig(
        window_size=WINDOW_SIZE,
        boxcox_lambda=BOXCOX_LAMBDA,
        lstm_n_neurons=64,
        dense_n_neurons=128,
        dropout_ratio=0.3,
        optimizer="adam",
        loss="mse",
        metrics=["mse", "mae"],
    )

    model = TrafficModel(config=model_config)
    dataset = Dataset(config=dataset_config)

    return model, dataset
