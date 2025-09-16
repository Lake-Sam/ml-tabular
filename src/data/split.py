from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import write_parquet

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(INTERIM_DIR / "bank_interim.parquet")
    train_df, test_df = train_test_split(
        dataset, test_size=0.2, random_state=42, stratify=dataset["y"]
    )
    train_df, valid_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["y"]
    )
    write_parquet(train_df, PROCESSED_DIR / "train.parquet")
    write_parquet(valid_df, PROCESSED_DIR / "valid.parquet")
    write_parquet(test_df, PROCESSED_DIR / "test.parquet")


if __name__ == "__main__":
    main()
