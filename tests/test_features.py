import pandas as pd

from src.data.features import make_preprocess


def test_preprocess_fit_transform():
    categorical = ["job"]
    numerical = ["age"]
    frame = pd.DataFrame({"job": ["a", "b", "a"], "age": [20, 30, 40]})
    transformer = make_preprocess(categorical, numerical)
    transformed = transformer.fit_transform(frame)
    assert transformed.shape[0] == 3
