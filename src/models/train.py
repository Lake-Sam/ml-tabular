from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from hydra import compose, initialize
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import settings
from src.data.features import make_preprocess
from src.models.calibrate import calibrate
from src.models.metrics import compute_metrics


def _load_xy(
    dataframe: pd.DataFrame,
    target: str,
    features: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if features is None:
        features = [column for column in dataframe.columns if column != target]
    X = dataframe[features]
    y = dataframe[target]
    return X, y, list(features)


def _build_estimator(cfg) -> Pipeline:
    preprocess = make_preprocess(cfg.data.categorical, cfg.data.numerical)
    model_type = cfg.model.type.lower()
    if model_type == "lightgbm":
        classifier = LGBMClassifier(**cfg.model.params)
    elif model_type == "xgboost":
        classifier = XGBClassifier(**cfg.model.params)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
    pipeline = Pipeline([("pre", preprocess), ("clf", classifier)])
    if cfg.train.calibrate:
        pipeline = calibrate(pipeline, method=cfg.train.calibration_method)
    return pipeline


def main() -> None:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")

    train_df = pd.read_parquet("data/processed/train.parquet")
    valid_df = pd.read_parquet("data/processed/valid.parquet")

    estimator = _build_estimator(cfg)

    X_train, y_train, feature_names = _load_xy(train_df, cfg.data.target)
    X_valid, y_valid, _ = _load_xy(valid_df, cfg.data.target, feature_names)

    with mlflow.start_run() as run:
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_valid)[:, 1]
        metrics = compute_metrics(y_valid, probabilities)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))

        mlflow.log_params(dict(cfg.model.params))
        mlflow.log_param("model_type", cfg.model.type)

        models_dir = Path(settings.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / run.info.run_id
        mlflow.sklearn.save_model(estimator, model_path)

        latest_symlink = models_dir / "latest"
        if latest_symlink.is_symlink() or latest_symlink.exists():
            if latest_symlink.is_symlink():
                latest_symlink.unlink()
            else:
                shutil.rmtree(latest_symlink)
        os.symlink(model_path, latest_symlink, target_is_directory=True)

        mlflow.log_artifacts(str(model_path))


if __name__ == "__main__":
    main()
