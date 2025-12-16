# src/ml_playground/__init__.py

# Expose the main public API
from .base import BaseModel
from .registry import get_model

__all__ = ["BaseModel", "get_model"]
