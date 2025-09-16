import logging
import sys


def configure_logging(cfg) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.level),
        format=cfg.format,
        stream=sys.stdout,
    )
