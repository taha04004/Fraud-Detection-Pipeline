"""PySpark ETL pipeline for transaction data."""

import csv
from functools import reduce
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

def validate_transaction_data(frame: DataFrame) -> None:
    """Raise an error when transaction rows violate data-quality rules."""

    has_null = reduce(
        lambda left, right: left | right,
        [
            F.col(column).isNull()
            for column in frame.columns
        ],
    )

    summary = (
        frame
        .agg(
            F.count("*").alias("row_count"),
            F.sum(
                F.when(has_null, 1).otherwise(0)
            ).alias("null_rows"),
            F.sum(
                F.when(
                    ~F.col(TARGET_COLUMN).isin(0, 1),
                    1,
                ).otherwise(0)
            ).alias("invalid_target_rows"),
            F.sum(
                F.when(
                    F.col("Amount") < 0,
                    1,
                ).otherwise(0)
            ).alias("negative_amount_rows"),
        )
        .first()
    )

    if summary is None:
        raise RuntimeError(
            "Spark did not return a data-quality summary"
        )

    row_count = int(summary["row_count"])
    null_rows = int(summary["null_rows"] or 0)
    invalid_target_rows = int(
        summary["invalid_target_rows"] or 0
    )
    negative_amount_rows = int(
        summary["negative_amount_rows"] or 0
    )

    errors: list[str] = []

    if row_count == 0:
        errors.append("dataset contains no transaction rows")

    if null_rows:
        errors.append(
            f"{null_rows} rows contain null values"
        )

    if invalid_target_rows:
        errors.append(
            f"{invalid_target_rows} rows contain invalid targets"
        )

    if negative_amount_rows:
        errors.append(
            f"{negative_amount_rows} rows contain negative amounts"
        )

    if errors:
        raise ValueError(
            "Transaction data validation failed: "
            + "; ".join(errors)
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

        validate_transaction_data(frame)

        processed = (
            frame
            .dropDuplicates()
            .withColumn(
                "AmountLog",
                F.log1p(F.col("Amount")),
            )
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