"""Load model artifacts and perform transaction inference."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.validate import FEATURE_COLUMNS
from src.models.network import FraudClassifier


@dataclass(frozen=True)
class ModelBundle:
    """Model and preprocessing values needed for inference."""

    model: FraudClassifier
    threshold: float
    version: str
    feature_columns: list[str]
    mean: np.ndarray
    scale: np.ndarray


def load_model(
    model_path: Path = Path("artifacts/model.pt"),
    preprocessor_path: Path = Path(
        "artifacts/preprocessor.json"
    ),
) -> ModelBundle:
    """Load a trained model and its preprocessing metadata."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            "Preprocessor artifact not found: "
            f"{preprocessor_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model = FraudClassifier(
        input_size=int(checkpoint["input_size"]),
        dropout=float(checkpoint["dropout"]),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    metadata = json.loads(
        preprocessor_path.read_text(encoding="utf-8")
    )

    feature_columns = list(metadata["feature_columns"])
    mean = np.asarray(
        metadata["mean"],
        dtype=np.float32,
    )
    scale = np.asarray(
        metadata["scale"],
        dtype=np.float32,
    )

    expected_size = int(checkpoint["input_size"])

    if len(feature_columns) != expected_size:
        raise ValueError(
            "Feature metadata does not match model input size"
        )

    if len(mean) != expected_size:
        raise ValueError(
            "Scaler mean does not match model input size"
        )

    if len(scale) != expected_size:
        raise ValueError(
            "Scaler scale does not match model input size"
        )

    if np.any(scale == 0):
        raise ValueError(
            "Scaler metadata contains a zero scale value"
        )

    return ModelBundle(
        model=model,
        threshold=float(metadata["threshold"]),
        version=str(metadata["model_version"]),
        feature_columns=feature_columns,
        mean=mean,
        scale=scale,
    )


def predict_transaction(
    bundle: ModelBundle,
    raw_features: list[float],
) -> float:
    """Transform one raw transaction and return its fraud probability."""

    if len(raw_features) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} raw features, "
            f"received {len(raw_features)}"
        )

    values_by_name = dict(
        zip(
            FEATURE_COLUMNS,
            raw_features,
            strict=True,
        )
    )

    amount = float(values_by_name["Amount"])
    transaction_time = float(values_by_name["Time"])

    if amount < 0:
        raise ValueError(
            "Transaction amount cannot be negative"
        )

    values_by_name["AmountLog"] = float(
        np.log1p(amount)
    )
    values_by_name["Hour"] = (
        transaction_time / 3600
    ) % 24

    model_features = np.asarray(
        [
            values_by_name[column]
            for column in bundle.feature_columns
        ],
        dtype=np.float32,
    )

    scaled_features = (
        model_features - bundle.mean
    ) / bundle.scale

    input_tensor = torch.from_numpy(
        scaled_features
    ).unsqueeze(0)

    with torch.inference_mode():
        logit = bundle.model(input_tensor)
        probability = torch.sigmoid(logit).item()

    return float(probability)