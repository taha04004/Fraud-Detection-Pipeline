from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from src.data.spark_etl import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRANSACTION_SCHEMA,
    split_by_time,
    validate_csv_header,
    validate_transaction_data,
)
from src.features.transformations import add_features


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    """Create one local Spark session for the ETL tests."""

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("fraud-etl-tests")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()

def test_add_features() -> None:
    result = add_features(
        pd.DataFrame(
            {
                "Time": [7200.0],
                "Amount": [9.0],
            }
        )
    )

    assert result.loc[0, "Hour"] == 2.0
    assert "AmountLog" in result


def test_transaction_schema_has_expected_columns() -> None:
    column_names = [field.name for field in TRANSACTION_SCHEMA.fields]

    assert column_names == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert len(FEATURE_COLUMNS) == 30
    assert len(column_names) == 31


def test_transaction_schema_has_expected_types() -> None:
    fields_by_name = {
        field.name: field
        for field in TRANSACTION_SCHEMA.fields
    }

    for column in FEATURE_COLUMNS:
        assert isinstance(
            fields_by_name[column].dataType,
            DoubleType,
        )

    assert isinstance(
        fields_by_name[TARGET_COLUMN].dataType,
        IntegerType,
    )


def test_transaction_schema_marks_columns_as_required() -> None:
    assert all(
        not field.nullable
        for field in TRANSACTION_SCHEMA.fields
    )

def test_validate_csv_header_accepts_expected_header(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "valid.csv"
    expected_header = [*FEATURE_COLUMNS, TARGET_COLUMN]

    csv_path.write_text(
        ",".join(expected_header) + "\n",
        encoding="utf-8",
    )

    validate_csv_header(csv_path)


def test_validate_csv_header_rejects_incorrect_header(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "invalid.csv"

    csv_path.write_text(
        "Time,V1,Amount,WrongTarget\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="CSV header does not match",
    ):
        validate_csv_header(csv_path)


def test_validate_csv_header_rejects_empty_file(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Input dataset is empty",
    ):
        validate_csv_header(csv_path)


def test_split_by_time_is_complete_and_non_overlapping(
    spark: SparkSession,
) -> None:
    source_times = set(range(100))

    frame = spark.range(100).select(
        F.col("id").cast("double").alias("Time"),
        (F.col("id") % 2).cast("integer").alias("Class"),
    )
    
    train, validation, test = split_by_time(frame)

    train_times = {
        int(row.Time)
        for row in train.select("Time").collect()
    }
    validation_times = {
        int(row.Time)
        for row in validation.select("Time").collect()
    }
    test_times = {
        int(row.Time)
        for row in test.select("Time").collect()
    }

    assert train_times.isdisjoint(validation_times)
    assert train_times.isdisjoint(test_times)
    assert validation_times.isdisjoint(test_times)

    combined_times = (
        train_times
        | validation_times
        | test_times
    )

    assert combined_times == source_times

    assert max(train_times) < min(validation_times)
    assert max(validation_times) < min(test_times)

def test_validate_transaction_data_accepts_valid_rows(
    spark: SparkSession,
) -> None:
    frame = spark.sql(
        """
        SELECT
            1.0 AS Time,
            10.0 AS Amount,
            0 AS Class
        UNION ALL
        SELECT
            2.0 AS Time,
            25.0 AS Amount,
            1 AS Class
        """
    )

    validate_transaction_data(frame)


def test_validate_transaction_data_rejects_invalid_rows(
    spark: SparkSession,
) -> None:
    frame = spark.sql(
        """
        SELECT
            1.0 AS Time,
            10.0 AS Amount,
            0 AS Class
        UNION ALL
        SELECT
            2.0 AS Time,
            -5.0 AS Amount,
            2 AS Class
        UNION ALL
        SELECT
            CAST(NULL AS DOUBLE) AS Time,
            20.0 AS Amount,
            1 AS Class
        """
    )

    with pytest.raises(
        ValueError,
        match="Transaction data validation failed",
    ) as error:
        validate_transaction_data(frame)

    message = str(error.value)

    assert "1 rows contain null values" in message
    assert "1 rows contain invalid targets" in message
    assert "1 rows contain negative amounts" in message