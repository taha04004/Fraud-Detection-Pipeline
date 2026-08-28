"""Benchmark FastAPI fraud-prediction latency."""

import json
import os
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
import pandas as pd

from src.data.validate import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
ENVIRONMENT = os.getenv(
    "BENCHMARK_ENVIRONMENT",
    "local",
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / f"inference_{ENVIRONMENT}_results.json"
)

API_URL = "http://127.0.0.1:8000"
WARMUP_REQUESTS = 20
MEASURED_REQUESTS = 200


def load_requests() -> list[dict[str, list[float]]]:
    """Load real transactions and convert them to API requests."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    frame = pd.read_csv(
        DATA_PATH,
        nrows=MEASURED_REQUESTS,
    )

    return [
        {
            "features": [
                float(row[column])
                for column in FEATURE_COLUMNS
            ]
        }
        for _, row in frame.iterrows()
    ]


def main() -> None:
    """Measure sequential prediction latency."""

    requests = load_requests()
    latencies_ms: list[float] = []

    with httpx.Client(
        base_url=API_URL,
        timeout=10.0,
    ) as client:
        readiness = client.get("/ready")
        readiness.raise_for_status()

        print(f"API readiness: {readiness.json()}")

        print(f"Running {WARMUP_REQUESTS} warm-up requests...")

        for index in range(WARMUP_REQUESTS):
            response = client.post(
                "/predict",
                json=requests[index],
            )
            response.raise_for_status()

        print(
            f"Running {MEASURED_REQUESTS} measured requests..."
        )

        for request in requests:
            started_at = perf_counter()

            response = client.post(
                "/predict",
                json=request,
            )

            elapsed_ms = (
                perf_counter() - started_at
            ) * 1000

            response.raise_for_status()
            latencies_ms.append(elapsed_ms)

    total_seconds = sum(latencies_ms) / 1000

    results = {
        "api_url": API_URL,
        "environment": ENVIRONMENT,
        "warmup_requests": WARMUP_REQUESTS,
        "measured_requests": MEASURED_REQUESTS,
        "request_mode": "sequential",
        "mean_latency_ms": float(
            np.mean(latencies_ms)
        ),
        "median_latency_ms": float(
            np.median(latencies_ms)
        ),
        "p95_latency_ms": float(
            np.percentile(latencies_ms, 95)
        ),
        "p99_latency_ms": float(
            np.percentile(latencies_ms, 99)
        ),
        "minimum_latency_ms": float(
            np.min(latencies_ms)
        ),
        "maximum_latency_ms": float(
            np.max(latencies_ms)
        ),
        "throughput_requests_per_second": (
            MEASURED_REQUESTS / total_seconds
        ),
        "sub_100_ms_p95": bool(
            np.percentile(latencies_ms, 95) < 100
        ),
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()