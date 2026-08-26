"""PySpark ETL pipeline for transaction data."""

from pathlib import Path


def run(input_path: Path, output_dir: Path) -> None:
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.appName("fraud-etl").getOrCreate()
    try:
        frame = spark.read.option("header", True).option("inferSchema", True).csv(str(input_path))
        processed = (
            frame.dropDuplicates()
            .dropna()
            .filter(F.col("Amount") >= 0)
            .withColumn("AmountLog", F.log1p("Amount"))
            .withColumn("Hour", (F.col("Time") / 3600 % 24).cast("double"))
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        processed.write.mode("overwrite").parquet(str(output_dir))
    finally:
        spark.stop()


if __name__ == "__main__":
    run(Path("data/raw/creditcard.csv"), Path("data/processed/spark"))

