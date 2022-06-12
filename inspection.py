import random
from datetime import datetime
import tensorflow as tf
import pandas as pd

r = random.SystemRandom()

RPS_PER_INSTANCE = 10

models = {}

three_last_predictions = {
    "zh": [r.randint(1,10), r.randint(1,10)],
    "fr": [r.randint(1,10), r.randint(1,10)],
    "en": [r.randint(1,10), r.randint(1,10)]
}

per_region_instances = {
    "zh": 1,
    "fr": 1,
    "en": 1
}

regions = ["zh", "fr", "en"]

def load_model(model_location):
    return tf.keras.models.load_model(model_location)


def model_wrapper(region):
    df = pd.DataFrame([three_last_predictions[region]], dtype=float).to_numpy()
    print(df.shape)
    return df


def predict_for_one(region):
    input = model_wrapper(region)

    model = models.get(region, None)
    if model is None:
        raise ValueError()

    rps = model.predict_boxcox(input).numpy()[0][0]
    instances_needed = int(rps / RPS_PER_INSTANCE) + 1
    three_last_predictions[region].append(rps)
    three_last_predictions[region].pop(0)

    if instances_needed > per_region_instances[region]:
        for _ in range(instances_needed - per_region_instances[region]):
            create_instance(region)
    else:
        for _ in range(per_region_instances[region] - instances_needed):
            destroy_instance(region)


def predict():
    for region in regions:
        predict_for_one(region)


def create_instance(instance_type):
    per_region_instances[instance_type] = per_region_instances[instance_type] + 1
    print(f"instance {instance_type} created")


def destroy_instance(instance_type):
    per_region_instances[instance_type] = per_region_instances[instance_type] + 1
    print(f"destroying instance {instance_type}")


def prepare_models():
    global models
    fr_model = load_model("models/fr_model.model")
    zh_model = load_model("models/zh_model.model")
    en_model = load_model("models/en_model.model")

    models = {
        "fr": fr_model,
        "zh": zh_model,
        "en": en_model
    }