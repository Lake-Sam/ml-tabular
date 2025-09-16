from __future__ import annotations

import mlflow


def register_model(model_uri: str, name: str) -> None:
    """Register a trained model in the MLflow Model Registry."""
    mlflow.register_model(model_uri=model_uri, name=name)
