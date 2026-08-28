import numpy as np
import torch
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from src.api import main
from src.api.model_loader import ModelBundle
from src.models.network import FraudClassifier

MODEL_FEATURE_COLUMNS = [
    *[f"V{i}" for i in range(1, 29)],
    "AmountLog",
    "Hour",
]

client = TestClient(main.app)


def create_test_bundle() -> ModelBundle:
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


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_ready(
    monkeypatch: MonkeyPatch,
) -> None:
    bundle = create_test_bundle()

    monkeypatch.setattr(
        main,
        "get_model_bundle",
        lambda: bundle,
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_version": "test-version",
    }


def test_predict_returns_model_probability(
    monkeypatch: MonkeyPatch,
) -> None:
    bundle = create_test_bundle()

    monkeypatch.setattr(
        main,
        "get_model_bundle",
        lambda: bundle,
    )

    response = client.post(
        "/predict",
        json={
            "features": [0.0] * 30,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["fraud_probability"] == 0.5
    assert result["is_fraud"] is False
    assert result["threshold"] == 0.6
    assert result["model_version"] == "test-version"


def test_predict_rejects_incorrect_feature_count() -> None:
    response = client.post(
        "/predict",
        json={
            "features": [0.0] * 29,
        },
    )

    assert response.status_code == 422


def test_predict_rejects_negative_amount(
    monkeypatch: MonkeyPatch,
) -> None:
    bundle = create_test_bundle()

    monkeypatch.setattr(
        main,
        "get_model_bundle",
        lambda: bundle,
    )

    features = [0.0] * 30
    features[-1] = -10.0

    response = client.post(
        "/predict",
        json={
            "features": features,
        },
    )

    assert response.status_code == 422
    assert (
        response.json()["detail"]
        == "Transaction amount cannot be negative"
    )