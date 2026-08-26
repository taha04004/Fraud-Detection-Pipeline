# Production Fraud Detection ML Pipeline

An end-to-end fraud classification project using PySpark for ETL, PyTorch for model
training, MLflow for experiment tracking, FastAPI for inference, and Docker for packaging.

## Status

The repository structure and initial interfaces are scaffolded. Dataset ingestion,
training, model registration, and production inference will be implemented incrementally.

## Quick start

1. Install Python 3.11 or 3.12, Java 17, Git, and Docker Desktop.
2. Create and activate a virtual environment.
3. Run `python -m pip install -e ".[dev]"`.
4. Place `creditcard.csv` in `data/raw/`.
5. Run `python -m pytest`.
6. Start the placeholder API with `python -m uvicorn src.api.main:app --reload`.

API documentation is available at `http://localhost:8000/docs`.

