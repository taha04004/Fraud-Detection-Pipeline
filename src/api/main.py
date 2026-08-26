"""FastAPI application for online fraud inference."""

from fastapi import FastAPI

from src.api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Fraud Detection API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    del request
    return PredictionResponse(
        fraud_probability=0.0, is_fraud=False, threshold=0.5, model_version="untrained"
    )

