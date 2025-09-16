import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.data.features import make_preprocess


def test_train_small():
    frame = pd.DataFrame(
        {"job": ["a", "b", "a", "b"], "age": [1, 2, 3, 4], "y": [0, 1, 0, 1]}
    )
    X = frame[["job", "age"]]
    y = frame["y"]
    pipeline = Pipeline(
        [
            ("pre", make_preprocess(["job"], ["age"])),
            ("clf", LGBMClassifier(n_estimators=10)),
        ]
    )
    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X)[:, 1]
    assert probabilities.shape[0] == len(y)
