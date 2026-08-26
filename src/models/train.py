"""Model training entry point (implemented in the training milestone)."""

from pathlib import Path


def train(config_path: Path) -> None:
    """Train, evaluate, and log a model using the supplied configuration."""
    raise NotImplementedError("Training pipeline has not been implemented yet")


if __name__ == "__main__":
    train(Path("configs/training.yaml"))

