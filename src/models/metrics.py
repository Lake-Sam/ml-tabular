from __future__ import annotations

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def compute_metrics(y_true, proba, threshold: float = 0.5):
    y_pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, proba),
    }
