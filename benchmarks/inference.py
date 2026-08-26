"""HTTP inference latency benchmark placeholder."""

from statistics import median


def summarize(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    p95_index = max(0, int(len(ordered) * 0.95) - 1)
    return {"median_ms": median(ordered), "p95_ms": ordered[p95_index]}

