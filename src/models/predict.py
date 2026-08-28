"""Run batch predictions using saved fraud-model artifacts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.api.model_loader import load_model
from src.data.validate import FEATURE_COLUMNS
from src.features.transformations import add_features


def predict_probability(
    model: torch.nn.Module,
    features: torch.Tensor,
) -> torch.Tensor:
    """Return fraud probabilities for a feature batch."""

    model.eval()

    with torch.inference_mode():
        return torch.sigmoid(model(features))


def predict_csv(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    preprocessor_path: Path,
    batch_size: int,
) -> pd.DataFrame:
    """Score every transaction in a CSV file."""

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prediction input not found: {input_path}"
        )

    frame = pd.read_csv(input_path)

    if frame.empty:
        raise ValueError(
            "Prediction input contains no transactions"
        )

    missing_columns = sorted(
        set(FEATURE_COLUMNS) - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Prediction input is missing columns: "
            f"{missing_columns}"
        )

    raw_features = frame[FEATURE_COLUMNS].copy()

    if raw_features.isnull().any().any():
        raise ValueError(
            "Prediction input contains null feature values"
        )

    raw_values = raw_features.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(raw_values).all():
        raise ValueError(
            "Prediction input contains non-finite values"
        )

    if (raw_features["Amount"] < 0).any():
        raise ValueError(
            "Prediction input contains negative amounts"
        )

    bundle = load_model(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )

    transformed = add_features(raw_features)

    missing_model_features = sorted(
        set(bundle.feature_columns)
        - set(transformed.columns)
    )

    if missing_model_features:
        raise ValueError(
            "Prediction input cannot produce model features: "
            f"{missing_model_features}"
        )

    model_features = transformed[
        bundle.feature_columns
    ].to_numpy(dtype=np.float32)

    scaled_features = (
        model_features - bundle.mean
    ) / bundle.scale

    probabilities: list[np.ndarray] = []

    for start in range(0, len(frame), batch_size):
        end = start + batch_size

        batch = torch.from_numpy(
            scaled_features[start:end].astype(
                np.float32
            )
        )

        batch_probabilities = predict_probability(
            bundle.model,
            batch,
        )

        probabilities.append(
            batch_probabilities.cpu().numpy()
        )

    fraud_probabilities = np.concatenate(
        probabilities
    )

    predictions = pd.DataFrame(
        {
            "transaction_index": np.arange(len(frame)),
            "fraud_probability": fraud_probabilities,
            "is_fraud": (
                fraud_probabilities >= bundle.threshold
            ),
            "threshold": bundle.threshold,
            "model_version": bundle.version,
        }
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    predictions.to_csv(
        output_path,
        index=False,
    )

    return predictions


def parse_arguments() -> argparse.Namespace:
    """Parse batch-prediction command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Score a CSV file using saved fraud-model artifacts."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/raw/creditcard.csv"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "artifacts/batch_predictions.csv"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/model.pt"),
    )
    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=Path(
            "artifacts/preprocessor.json"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    return parser.parse_args()


def main() -> None:
    """Run batch prediction from the command line."""

    arguments = parse_arguments()

    predictions = predict_csv(
        input_path=arguments.input_path,
        output_path=arguments.output_path,
        model_path=arguments.model_path,
        preprocessor_path=arguments.preprocessor_path,
        batch_size=arguments.batch_size,
    )

    fraud_predictions = int(
        predictions["is_fraud"].sum()
    )

    print(f"Scored transactions: {len(predictions)}")
    print(f"Predicted fraud cases: {fraud_predictions}")
    print(f"Saved predictions: {arguments.output_path}")


if __name__ == "__main__":
    main()