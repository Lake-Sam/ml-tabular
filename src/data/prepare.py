from __future__ import annotations

from pathlib import Path

import pandas as pd
from hydra import compose, initialize

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")
    dataset = pd.read_csv(RAW_DIR / "bank_additional_full.csv")
    dataset.columns = [column.strip() for column in dataset.columns]
    dataset[cfg.data.target] = (
        dataset[cfg.data.target] == cfg.data.positive_class
    ).astype(int)
    dataset.to_parquet(INTERIM_DIR / "bank_interim.parquet", index=False)


if __name__ == "__main__":
    main()
