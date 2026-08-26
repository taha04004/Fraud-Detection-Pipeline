"""Benchmark Pandas and PySpark preprocessing implementations."""

import time
from collections.abc import Callable


def elapsed_seconds(operation: Callable[[], None]) -> float:
    started = time.perf_counter()
    operation()
    return time.perf_counter() - started

