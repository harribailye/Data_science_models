from ML_model_playground.ml_playground.src.ml_playground.models.linear_regression import LinearRegressionModel

MODEL_REGISTRY = {
    "linear": LinearRegressionModel,
}

def get_model(name: str, **kwargs):
    return MODEL_REGISTRY[name](**kwargs)
