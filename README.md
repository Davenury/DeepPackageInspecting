### Train the model
```python
from instances.en import build_model_and_dataset
from train import train, TrainingConfig

model_config, dataset_config = build_model_and_dataset()

config = TrainingConfig(
    batch_size=64,
    epochs=1,
    model_config=model_config,
    dataset_config=dataset_config
)

model, dataset = train(
    config=config,
)
```

### Save model

```python
model_path = "en_model"
model.save(model_path)
```

### Load model
```python
import tensorflow as tf
model_path = "en_model"
model = tf.keras.models.load_model(model_path)
```

### Inference model
```python
X = dataset.x_test[0]  # shape = (551, window_size)
y = datset.y_test[0]  # shape = (551,)
y_pred = model.predict_boxcox(X)  # shape = (551, 1)
```
