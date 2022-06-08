from typing import NamedTuple, Tuple

from data import DatasetConfig, Dataset
from model import TrafficModelConfig, TrafficModel


class TrainingConfig(NamedTuple):
    batch_size: int
    epochs: int
    dataset_config: DatasetConfig = None
    model_config: TrafficModelConfig = None


# FIXME change to logging
def train(
    config: TrainingConfig, model: TrafficModel = None, dataset: Dataset = None
) -> Tuple[TrafficModel, Dataset]:

    if not dataset:
        if not config.dataset_config:
            raise RuntimeError("Dataset config is not specified")
        print("Building dataset")
        dataset = Dataset(config=config.dataset_config)
        dataset.build_dataset()

    if not model:
        if not config.model_config:
            raise RuntimeError("Model config is not specified")
        print("Building model")
        model = TrafficModel(config=config.model_config)
        model.compile(
            optimizer=config.model_config.optimizer,
            loss=config.model_config.loss,
            metrics=config.model_config.metrics,
        )
        model.build((None, 1, config.model_config.window_size))
        print(model.summary())

    model.fit(
        dataset.x_train,
        dataset.y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
    )

    return model, dataset
