from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from ml_playground.base import BaseModel
import numpy as np

class LinearRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def metrics(self, X, y):
        preds = self.predict(X)
        return {
            "mae" : mean_absolute_error(y, preds),
            "mse": mean_squared_error(y, preds),
            "r2": r2_score(y, preds)
        }