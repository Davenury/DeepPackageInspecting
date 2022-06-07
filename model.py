from typing import NamedTuple, List

import tensorflow as tf


class TrafficModelConfig(NamedTuple):
    window_size: int = 2
    boxcox_lambda: float = 2.5
    lstm_n_neurons: int = 64
    dense_n_neurons: int = 128
    dropout_ratio: float = 0.3
    optimizer: str = "adam"
    loss: str = "mse"
    metrics: List[str] = ["mse", "mae"]


class TrafficModel(tf.keras.Model):
    def __init__(self, config: TrafficModelConfig):
        super().__init__()
        self.config = config
        self.lstm = tf.keras.layers.LSTM(
            units=self.config.lstm_n_neurons,
            activation="relu",
            input_shape=(None, self.config.window_size),
        )
        self.dense_1 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dense_out = tf.keras.layers.Dense(units=1)
        self.dropout = tf.keras.layers.Dropout(self.config.dropout_ratio)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_out(x)
        return x

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32)]
    )
    def predict_boxcox(self, inputs):
        inputs = inputs[:, tf.newaxis, :]
        x = self.lstm(inputs)
        x = self.dense_1(x)
        y = self.dense_out(x)
        y_shape = y.shape
        y = tf.reshape(y, [-1])
        y = tf.cast(y, tf.float32)
        if self.config.boxcox_lambda == 0:
            y = tf.math.log(y)
        else:
            y = (tf.pow(y, self.config.boxcox_lambda) - 1) / self.config.boxcox_lambda

        y = tf.reshape(y, [-1, y_shape[-1]])
        return y
