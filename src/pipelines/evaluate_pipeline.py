from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize

from src.models.explain import shap_summary
from src.models.metrics import compute_metrics
from src.models.predict import load_model
from src.utils.io import read_parquet
from src.utils.logging import configure_logging


def main() -> None:
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")
    configure_logging(cfg.logging)

    dataset = read_parquet("data/processed/test.parquet")
    y_true = dataset[cfg.data.target].values
    X = dataset.drop(columns=[cfg.data.target])

    model = load_model("models/latest")
    probabilities = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(
        y_true,
        probabilities,
        threshold=cfg.eval.threshold,
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "eval_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if cfg.eval.shap_summary:
        shap_summary(model, X, output_dir / "plots")


if __name__ == "__main__":
    main()
