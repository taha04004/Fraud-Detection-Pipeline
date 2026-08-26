"""Pandas reference ETL used for correctness and timing comparisons."""

from pathlib import Path

import pandas as pd

from src.data.validate import validate_dataframe
from src.features.transformations import add_features


def run(input_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(input_path)
    validate_dataframe(frame)
    transformed = add_features(frame.drop_duplicates())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_parquet(output_path, index=False)


if __name__ == "__main__":
    run(Path("data/raw/creditcard.csv"), Path("data/processed/pandas.parquet"))

