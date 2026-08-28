"""Benchmark the Pandas and Dockerized PySpark ETL pipelines."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from src.data.pandas_baseline import run as run_pandas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
TEMP_DIR = PROJECT_ROOT / "benchmarks" / ".tmp"
RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "preprocessing_results.json"


def reset_directory(path: Path) -> None:
    """Create an empty directory for one benchmark run."""

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True)


def benchmark_pandas() -> float:
    """Measure the Pandas reference pipeline."""

    output_path = TEMP_DIR / "pandas" / "transactions.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = perf_counter()
    run_pandas(INPUT_PATH, output_path)

    return perf_counter() - started_at


def benchmark_spark() -> float:
    """Measure the Dockerized PySpark pipeline."""

    project_mount = (
        f"type=bind,source={PROJECT_ROOT},target=/workspace"
    )

    command = [
        "docker",
        "run",
        "--rm",
        "--mount",
        project_mount,
        "--workdir",
        "/workspace",
        "apache/spark:4.2.0-python3",
        "/opt/spark/bin/spark-submit",
        "src/data/spark_etl.py",
        (
            "from pathlib import Path; "
            "from src.data.spark_etl import run; "
            "run("
            "Path('/workspace/data/raw/creditcard.csv'), "
            "Path('/workspace/benchmarks/.tmp/spark')"
            ")"
        ),
    ]

    started_at = perf_counter()

    subprocess.run(
        command,
        check=True,
        cwd=PROJECT_ROOT,
    )

    return perf_counter() - started_at


def main() -> None:
    """Run both benchmarks and save their measurements."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    reset_directory(TEMP_DIR)

    print("Running Pandas benchmark...")
    pandas_seconds = benchmark_pandas()
    print(f"Pandas completed in {pandas_seconds:.3f} seconds")

    print("Running Dockerized PySpark benchmark...")
    spark_seconds = benchmark_spark()
    print(f"PySpark completed in {spark_seconds:.3f} seconds")

    results = {
        "dataset": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "pandas_seconds": pandas_seconds,
        "spark_seconds": spark_seconds,
        "spark_to_pandas_ratio": spark_seconds / pandas_seconds,
        "comparison_note": (
            "Pandas writes one transformed dataset. "
            "PySpark additionally performs chronological splitting "
            "and writes train, validation, and test datasets. "
            "The Spark measurement also includes Docker and JVM startup."
        ),
        "python_version": sys.version,
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()