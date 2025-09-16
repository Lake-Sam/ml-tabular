import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def log_timing(message: str) -> Generator[None, None, None]:
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info("%s finished in %.3fs", message, elapsed)
