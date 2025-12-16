from abc import ABC, abstractmethod
import numpy as np

# We define the BaseModel template which makes sure each model has a fit and predict method
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray = None):
        """Return predictions for the given data. 
        If X is None, use internal test set if available."""
        pass

    @abstractmethod
    def metrics(self, X: np.ndarray = None, y: np.ndarray = None):
        """Compute performance metrics.
        If X or y are None, use internal test set if available."""
        pass


