from __future__ import annotations

from hydra import compose, initialize

from src.models.train import main as train_main
from src.utils.logging import configure_logging


def main() -> None:
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")
    configure_logging(cfg.logging)
    train_main()


if __name__ == "__main__":
    main()
