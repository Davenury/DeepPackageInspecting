def model_wrapper(request):
    print(request)


def predict(request):
    return model_wrapper(request)


def act(model_result):
    pass
