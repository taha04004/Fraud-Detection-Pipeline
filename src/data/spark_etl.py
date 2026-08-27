"""PySpark ETL pipeline for transaction data."""

import csv
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

TARGET_COLUMN = "Class"

TRANSACTION_SCHEMA = StructType(
    [
        *[
            StructField(column, DoubleType(), nullable=False)
            for column in FEATURE_COLUMNS
        ],
        StructField(TARGET_COLUMN, IntegerType(), nullable=False),
    ]
)

def validate_csv_header(input_path: Path) -> None:
    """Verify that the CSV header matches the expected dataset schema."""

    expected_header = [*FEATURE_COLUMNS, TARGET_COLUMN]

    with input_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.reader(csv_file)
        actual_header = next(reader, None)

    if actual_header is None:
        raise ValueError("Input dataset is empty")

    if actual_header != expected_header:
        missing_columns = sorted(
            set(expected_header) - set(actual_header)
        )
        unexpected_columns = sorted(
            set(actual_header) - set(expected_header)
        )

        raise ValueError(
            "CSV header does not match the expected schema. "
            f"Missing columns: {missing_columns}. "
            f"Unexpected columns: {unexpected_columns}. "
            "Column order must match the expected schema."
        )


def run(input_path: Path, output_dir: Path) -> None:
    """Validate, transform, and write transaction data using Spark."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    
    validate_csv_header(input_path)

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("fraud-etl")
        .getOrCreate()
    )

    try:
        frame = (
            spark.read
            .option("header", True)
            .option("mode", "FAILFAST")
            .schema(TRANSACTION_SCHEMA)
            .csv(str(input_path))
        )

        processed = (
            frame
            .dropDuplicates()
            .dropna()
            .filter(F.col("Amount") >= 0)
            .withColumn("AmountLog", F.log1p(F.col("Amount")))
            .withColumn(
                "Hour",
                ((F.col("Time") / 3600) % 24).cast("double"),
            )
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        (
            processed.write
            .mode("overwrite")
            .parquet(str(output_dir))
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    run(
        Path("data/raw/creditcard.csv"),
        Path("data/processed/spark"),
    )