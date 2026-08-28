"""Evaluate a saved fraud-classification model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.api.model_loader import load_model


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate metrics for imbalanced binary classification."""

    predictions = probabilities >= threshold

    return {
        "roc_auc": float(
            roc_auc_score(targets, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                targets,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
    }


def evaluate_saved_model(
    test_path: Path,
    model_path: Path,
    preprocessor_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Evaluate saved model artifacts against a processed test split."""

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    bundle = load_model(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )

    frame = pd.read_parquet(test_path)

    if frame.empty:
        raise ValueError("Test dataset contains no rows")

    required_columns = {
        *bundle.feature_columns,
        "Class",
    }
    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Test dataset is missing columns: {missing_columns}"
        )

    evaluation_frame = frame[
        [*bundle.feature_columns, "Class"]
    ]

    if evaluation_frame.isnull().any().any():
        raise ValueError(
            "Test dataset contains null feature or target values"
        )

    targets = evaluation_frame["Class"].to_numpy(
        dtype=np.int64
    )

    invalid_targets = sorted(
        set(np.unique(targets)) - {0, 1}
    )

    if invalid_targets:
        raise ValueError(
            f"Test dataset contains invalid targets: "
            f"{invalid_targets}"
        )

    model_features = evaluation_frame[
        bundle.feature_columns
    ].to_numpy(dtype=np.float32)

    scaled_features = (
        model_features - bundle.mean
    ) / bundle.scale

    input_tensor = torch.from_numpy(
        scaled_features.astype(np.float32)
    )

    with torch.inference_mode():
        logits = bundle.model(input_tensor)
        probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

    metrics = classification_metrics(
        targets,
        probabilities,
        bundle.threshold,
    )

    report: dict[str, object] = {
        "model_version": bundle.version,
        "threshold": bundle.threshold,
        "test_path": str(test_path),
        "row_count": len(targets),
        "fraud_count": int(targets.sum()),
        "metrics": metrics,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def parse_arguments() -> argparse.Namespace:
    """Parse evaluation command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved fraud-model artifacts "
            "against a processed test split."
        )
    )

    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path(
            "data/processed/spark/test.parquet"
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
        "--output-path",
        type=Path,
        default=Path(
            "artifacts/evaluation_report.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run saved-model evaluation from the command line."""

    arguments = parse_arguments()

    report = evaluate_saved_model(
        test_path=arguments.test_path,
        model_path=arguments.model_path,
        preprocessor_path=arguments.preprocessor_path,
        output_path=arguments.output_path,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
