from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


def shap_summary(
    model,
    X: pd.DataFrame,
    outdir: str | os.PathLike[str],
) -> None:
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transformed = model.named_steps["pre"].transform(X)
    explainer = shap.Explainer(
        model.named_steps["clf"], transformed, check_additivity=False
    )
    values = explainer(transformed)

    plt.figure()
    shap.plots.beeswarm(values, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png")
    plt.close()
