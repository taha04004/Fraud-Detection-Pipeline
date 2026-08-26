"""Validate the raw credit-card transaction dataset."""

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


def validate_dataframe(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` when required columns or values are invalid."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains null values")
    if not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")
    if (frame["Amount"] < 0).any():
        raise ValueError("Amount cannot be negative")


def validate_csv(path: Path) -> None:
    """Load and validate a CSV file."""
    validate_dataframe(pd.read_csv(path))

