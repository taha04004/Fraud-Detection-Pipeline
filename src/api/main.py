"""FastAPI application for online fraud inference."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from src.api.model_loader import (
    ModelBundle,
    load_model,
    predict_transaction,
)
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="Fraud Detection API",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    """Load and cache the model for repeated requests."""

    return load_model()


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the web application is running."""

    return {"status": "healthy"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Report whether the trained model is available."""

    try:
        bundle = get_model_bundle()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    return {
        "status": "ready",
        "model_version": bundle.version,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """Return a fraud prediction for one transaction."""

    try:
        bundle = get_model_bundle()
        probability = predict_transaction(
            bundle,
            request.features,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    return PredictionResponse(
        fraud_probability=probability,
        is_fraud=probability >= bundle.threshold,
        threshold=bundle.threshold,
        model_version=bundle.version,
    )