"""PySpark ETL pipeline for transaction data."""

import csv
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
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

def split_by_time(
    frame: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Split transactions chronologically into train, validation, and test sets."""

    train_boundary, validation_boundary = frame.approxQuantile(
        "Time",
        [0.70, 0.85],
        0.001,
    )

    train = frame.filter(
        F.col("Time") <= train_boundary
    )

    validation = frame.filter(
        (F.col("Time") > train_boundary)
        & (F.col("Time") <= validation_boundary)
    )

    test = frame.filter(
        F.col("Time") > validation_boundary
    )

    return train, validation, test

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

        train, validation, test = split_by_time(processed)

        output_dir.mkdir(parents=True, exist_ok=True)

        datasets = {
            "train": train,
            "validation": validation,
            "test": test,
        }

        for dataset_name, dataset in datasets.items():
            dataset_path = output_dir / f"{dataset_name}.parquet"

            (
                dataset.write
                .mode("overwrite")
                .parquet(str(dataset_path))
            )
    finally:
        spark.stop()


if __name__ == "__main__":
    run(
        Path("data/raw/creditcard.csv"),
        Path("data/processed/spark"),
    )