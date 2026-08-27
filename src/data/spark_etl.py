"""PySpark ETL pipeline for transaction data."""

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


def run(input_path: Path, output_dir: Path) -> None:
    """Validate, transform, and write transaction data using Spark."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

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