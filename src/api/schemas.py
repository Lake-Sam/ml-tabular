from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class PredictIn(BaseModel):
    records: List[Dict[str, Any]]


class PredictOut(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    model_version: str
