from typing import Any, Dict, List

from pydantic import BaseModel


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    model_version: str
