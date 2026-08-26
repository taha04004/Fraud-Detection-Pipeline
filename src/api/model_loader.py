"""Model loading boundary for the API."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelBundle:
    model: object
    threshold: float
    version: str


def load_model() -> ModelBundle:
    raise RuntimeError("No trained model artifact is available yet")

