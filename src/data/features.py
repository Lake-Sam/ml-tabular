from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocess(categorical, numerical):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            (
                "num",
                Pipeline([("scaler", StandardScaler(with_mean=False))]),
                numerical,
            ),
        ]
    )
