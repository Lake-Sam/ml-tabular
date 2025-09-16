from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
from hydra import compose, initialize

RAW_DIR = Path("data/raw")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")
    url = cfg.data.url
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    with archive.open("bank-additional/bank-additional-full.csv") as csv_file:
        df = pd.read_csv(csv_file, sep=";")
    df.to_csv(RAW_DIR / "bank_additional_full.csv", index=False)


if __name__ == "__main__":
    main()
