from __future__ import annotations

from fastapi import FastAPI

from src.api.load_model import get_model
from src.api.schemas import PredictIn, PredictOut
from src.models.predict import predict_records

app = FastAPI(title="ml-cookiecutter-tabular")


@app.get("/health")
def health():
    _, path = get_model()
    return {"status": "ok", "model_path": path}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn) -> PredictOut:
    model, path = get_model()
    predictions, probabilities = predict_records(model, payload.records)
    version = path.split("/")[-1]
    return PredictOut(
        predictions=predictions,
        probabilities=probabilities,
        model_version=version,
    )
