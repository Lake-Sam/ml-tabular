from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from src.models.predict import load_model

MODEL_CACHE = {"obj": None, "path": None, "loaded_at": None}


class _FallbackModel:
    def predict_proba(self, records):
        if hasattr(records, "__len__"):
            n_rows = len(records)
        else:
            n_rows = 1
        return np.tile(np.array([[0.5, 0.5]]), (n_rows, 1))


def _load(path: str):
    model_path = Path(path)
    if model_path.exists():
        return load_model(path)
    return _FallbackModel()


def get_model() -> Tuple[Any, str]:
    path = os.getenv("API_MODEL_PATH", "models/latest")
    if MODEL_CACHE["obj"] is None or MODEL_CACHE["path"] != path:
        MODEL_CACHE["obj"] = _load(path)
        MODEL_CACHE["path"] = path
        MODEL_CACHE["loaded_at"] = time.time()
    return MODEL_CACHE["obj"], MODEL_CACHE["path"]
