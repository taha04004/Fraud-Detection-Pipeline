import pandas as pd
from pyspark.sql.types import DoubleType, IntegerType

from src.data.spark_etl import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRANSACTION_SCHEMA,
)
from src.features.transformations import add_features


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