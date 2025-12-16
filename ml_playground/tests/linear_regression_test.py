import numpy as np
from ml_playground.models.linear_regression import LinearRegressionModel

def test_linear():
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    model = LinearRegressionModel()
    model.fit(X, y)
    preds = model.predict(X)
    metrics = model.metrics(X, y)
    assert all(preds.round() == y)

