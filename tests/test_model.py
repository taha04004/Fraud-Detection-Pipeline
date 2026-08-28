from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pytest import MonkeyPatch

from src.api.model_loader import ModelBundle
from src.data.validate import FEATURE_COLUMNS
from src.models import evaluate, predict
from src.models.network import FraudClassifier

MODEL_FEATURE_COLUMNS = [
    *[f"V{i}" for i in range(1, 29)],
    "AmountLog",
    "Hour",
]


def create_test_bundle() -> ModelBundle:
    """Create a deterministic model bundle for evaluation tests."""

    model = FraudClassifier(
        input_size=30,
        dropout=0.2,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    model.eval()

    return ModelBundle(
        model=model,
        threshold=0.6,
        version="test-version",
        feature_columns=MODEL_FEATURE_COLUMNS,
        mean=np.zeros(30, dtype=np.float32),
        scale=np.ones(30, dtype=np.float32),
    )


def test_model_output_shape() -> None:
    model = FraudClassifier(input_size=30)

    assert model(torch.zeros((4, 30))).shape == (4,)


def test_evaluate_saved_model(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    bundle = create_test_bundle()

    monkeypatch.setattr(
        evaluate,
        "load_model",
        lambda **_: bundle,
    )

    frame = pd.DataFrame(
        np.zeros((4, 30), dtype=np.float32),
        columns=MODEL_FEATURE_COLUMNS,
    )
    frame["Class"] = [0, 1, 0, 1]

    test_path = tmp_path / "test.parquet"
    output_path = tmp_path / "evaluation.json"
    frame.to_parquet(test_path, index=False)

    report = evaluate.evaluate_saved_model(
        test_path=test_path,
        model_path=tmp_path / "model.pt",
        preprocessor_path=tmp_path / "preprocessor.json",
        output_path=output_path,
    )

    assert report["model_version"] == "test-version"
    assert report["threshold"] == 0.6
    assert report["row_count"] == 4
    assert report["fraud_count"] == 2
    assert output_path.exists()

    metrics = report["metrics"]

    assert isinstance(metrics, dict)
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["pr_auc"] == pytest.approx(0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_evaluate_rejects_missing_features(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    bundle = create_test_bundle()

    monkeypatch.setattr(
        evaluate,
        "load_model",
        lambda **_: bundle,
    )

    test_path = tmp_path / "incomplete.parquet"

    pd.DataFrame(
        {
            "Class": [0, 1],
        }
    ).to_parquet(test_path, index=False)

    with pytest.raises(
        ValueError,
        match="Test dataset is missing columns",
    ):
        evaluate.evaluate_saved_model(
            test_path=test_path,
            model_path=tmp_path / "model.pt",
            preprocessor_path=(
                tmp_path / "preprocessor.json"
            ),
            output_path=tmp_path / "evaluation.json",
        )

def test_predict_csv(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    bundle = create_test_bundle()

    monkeypatch.setattr(
        predict,
        "load_model",
        lambda **_: bundle,
    )

    frame = pd.DataFrame(
        np.zeros(
            (3, len(FEATURE_COLUMNS)),
            dtype=np.float32,
        ),
        columns=FEATURE_COLUMNS,
    )

    input_path = tmp_path / "transactions.csv"
    output_path = tmp_path / "predictions.csv"
    frame.to_csv(input_path, index=False)

    predictions = predict.predict_csv(
        input_path=input_path,
        output_path=output_path,
        model_path=tmp_path / "model.pt",
        preprocessor_path=(
            tmp_path / "preprocessor.json"
        ),
        batch_size=2,
    )

    assert len(predictions) == 3
    assert output_path.exists()
    assert np.allclose(
        predictions["fraud_probability"],
        0.5,
    )
    assert predictions["is_fraud"].tolist() == [
        False,
        False,
        False,
    ]
    assert predictions["model_version"].tolist() == [
        "test-version",
        "test-version",
        "test-version",
    ]


def test_predict_csv_rejects_negative_amount(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        np.zeros(
            (1, len(FEATURE_COLUMNS)),
            dtype=np.float32,
        ),
        columns=FEATURE_COLUMNS,
    )
    frame.loc[0, "Amount"] = -1.0

    input_path = tmp_path / "invalid.csv"
    frame.to_csv(input_path, index=False)

    with pytest.raises(
        ValueError,
        match="negative amounts",
    ):
        predict.predict_csv(
            input_path=input_path,
            output_path=tmp_path / "predictions.csv",
            model_path=tmp_path / "model.pt",
            preprocessor_path=(
                tmp_path / "preprocessor.json"
            ),
            batch_size=2,
        )

