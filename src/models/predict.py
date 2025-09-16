from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mlflow.sklearn
import pandas as pd


def load_model(path: str):
    return mlflow.sklearn.load_model(path)


def predict_records(
    model,
    records: List[Dict[str, Any]],
) -> Tuple[List[int], List[float]]:
    dataframe = pd.DataFrame.from_records(records)
    probabilities = model.predict_proba(dataframe)[:, 1]
    predictions = (probabilities >= 0.5).astype(int).tolist()
    return predictions, probabilities.tolist()
