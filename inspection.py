from random import randint
from datetime import datetime


RPS_PER_INSTANCE = 10

models = {}

three_last_predictions = {
    "zh": [randint(10), randint(10), randint(10)],
    "fr": [randint(10), randint(10), randint(10)],
    "en": [randint(10), randint(10), randint(10)]
}

per_region_instances = {
    "zh": 1,
    "fr": 1,
    "en": 1
}

regions = ["zh", "fr", "en"]

def load_model(model_file):
    return model_file


def model_wrapper(region):
    current = datetime.now()
    minute = current.minute
    hour = current.hour
    return [*three_last_predictions[region], minute, hour]


def predict_for_one(region):
    input = model_wrapper(region)

    model_type = models.get("region")
    model = models.get(model_type, None)
    if model is None:
        raise ValueError()

    rps = model.predict(input)
    instances_needed = rps / RPS_PER_INSTANCE
    three_last_predictions[region].append(instances_needed)
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
    fr_model = load_model("fr_model")
    zh_model = load_model("zh_model")
    en_model = load_model("en_model")

    models = {
        "fr": fr_model,
        "zh": zh_model,
        "en": en_model
    }