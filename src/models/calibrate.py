from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def calibrate(model, method: str = "isotonic"):
    return CalibratedClassifierCV(model, method=method, cv=3)
