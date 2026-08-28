"""Train and save the PyTorch fraud-classification model."""

import json
import random
from copy import deepcopy
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from mlflow import pytorch as mlflow_pytorch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.dataset import FraudDataset
from src.models.network import FraudClassifier


def set_random_seeds(seed: int) -> None:
    """Make model training as reproducible as practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(
    path: Path,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load one processed Parquet split."""

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

    frame = pd.read_parquet(path)

    missing_columns = sorted(
        {*feature_columns, "Class"}
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    features = frame[feature_columns].to_numpy(
        dtype=np.float32
    )
    targets = frame["Class"].to_numpy(dtype=np.float32)

    return features, targets


def create_data_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create batches for PyTorch training or evaluation."""

    dataset = FraudDataset(features, targets)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def predict_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Return fraud probabilities for a dataset."""

    model.eval()
    probabilities: list[np.ndarray] = []

    with torch.inference_mode():
        for features, _ in data_loader:
            features = features.to(device)
            logits = model(features)
            batch_probabilities = torch.sigmoid(logits)

            probabilities.append(
                batch_probabilities.cpu().numpy()
            )

    return np.concatenate(probabilities)


def select_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Select the validation threshold with the highest F1 score."""

    thresholds = np.linspace(0.05, 0.95, 181)

    scores = [
        f1_score(
            targets,
            probabilities >= threshold,
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_index = int(np.argmax(scores))

    return float(thresholds[best_index])


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate imbalanced-classification metrics."""

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


def train(config_path: Path) -> None:
    """Train, evaluate, and save the fraud classifier."""

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    seed = int(config["seed"])
    set_random_seeds(seed)

    mlflow_config = config["mlflow"]

    mlflow.set_tracking_uri(
        mlflow_config["tracking_uri"]
    )
    mlflow.set_experiment(
        mlflow_config["experiment_name"]
    )

    mlflow.start_run(
        run_name="pytorch-fraud-classifier"
    )

    feature_columns = list(config["features"]["columns"])
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    learning_rate = float(
        config["training"]["learning_rate"]
    )
    weight_decay = float(
        config["training"]["weight_decay"]
    )
    patience = int(
        config["training"]["early_stopping_patience"]
    )
    dropout = float(config["model"]["dropout"])

    train_features, train_targets = load_dataset(
        Path(config["data"]["train_path"]),
        feature_columns,
    )
    validation_features, validation_targets = load_dataset(
        Path(config["data"]["validation_path"]),
        feature_columns,
    )
    test_features, test_targets = load_dataset(
        Path(config["data"]["test_path"]),
        feature_columns,
    )

    scaler = StandardScaler()

    train_features = scaler.fit_transform(
        train_features
    ).astype(np.float32)

    validation_features = scaler.transform(
        validation_features
    ).astype(np.float32)

    test_features = scaler.transform(
        test_features
    ).astype(np.float32)

    train_loader = create_data_loader(
        train_features,
        train_targets,
        batch_size,
        shuffle=True,
    )
    validation_loader = create_data_loader(
        validation_features,
        validation_targets,
        batch_size,
        shuffle=False,
    )
    test_loader = create_data_loader(
        test_features,
        test_targets,
        batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")
    print(f"Training rows: {len(train_targets)}")
    print(f"Validation rows: {len(validation_targets)}")
    print(f"Test rows: {len(test_targets)}")

    model = FraudClassifier(
        input_size=len(feature_columns),
        dropout=dropout,
    ).to(device)

    fraud_count = float(train_targets.sum())
    legitimate_count = float(
        len(train_targets) - fraud_count
    )

    if fraud_count == 0:
        raise ValueError(
            "Training dataset contains no fraud examples"
        )

    positive_weight = legitimate_count / fraud_count

    mlflow.log_params(
        {
            "seed": seed,
            "feature_count": len(feature_columns),
            "batch_size": batch_size,
            "maximum_epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": patience,
            "dropout": dropout,
            "positive_class_weight": positive_weight,
            "training_rows": len(train_targets),
            "validation_rows": len(validation_targets),
            "test_rows": len(test_targets),
            "training_fraud_cases": int(fraud_count),
            "device": str(device),
        }
    )

    mlflow.log_dict(
        config,
        "training_config.json",
    )    

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_pr_auc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(features)
            loss = loss_function(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)

        average_loss = total_loss / len(train_targets)

        validation_probabilities = predict_probabilities(
            model,
            validation_loader,
            device,
        )

        validation_pr_auc = average_precision_score(
            validation_targets,
            validation_probabilities,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={average_loss:.6f} | "
            f"validation_pr_auc={validation_pr_auc:.6f}"
        )
        mlflow.log_metrics(
            {
                "training_loss": float(average_loss),
                "validation_pr_auc": float(
                    validation_pr_auc
                ),
            },
            step=epoch,
        )

        if validation_pr_auc > best_pr_auc:
            best_pr_auc = float(validation_pr_auc)
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered")
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid model"
        )

    model.load_state_dict(best_state)

    validation_probabilities = predict_probabilities(
        model,
        validation_loader,
        device,
    )

    threshold = select_threshold(
        validation_targets,
        validation_probabilities,
    )

    validation_metrics = calculate_metrics(
        validation_targets,
        validation_probabilities,
        threshold,
    )

    test_probabilities = predict_probabilities(
        model,
        test_loader,
        device,
    )

    test_metrics = calculate_metrics(
        test_targets,
        test_probabilities,
        threshold,
    )

    model_path = Path(config["artifacts"]["model_path"])
    preprocessor_path = Path(
        config["artifacts"]["preprocessor_path"]
    )
    metrics_path = Path(
        config["artifacts"]["metrics_path"]
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(feature_columns),
            "dropout": dropout,
        },
        model_path,
    )
    
    preprocessor = {
        "feature_columns": feature_columns,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "threshold": threshold,
        "model_version": "pending",
    }

    preprocessor_path.write_text(
        json.dumps(preprocessor, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    mlflow.log_param(
        "selected_threshold",
        threshold,
    )

    mlflow.log_metrics(
        {
            **{
                f"validation_{name}": value
                for name, value
                in validation_metrics.items()
            },
            **{
                f"test_{name}": value
                for name, value
                in test_metrics.items()
            },
        }
    )

    mlflow.log_artifact(
        str(model_path),
        artifact_path="model",
    )
    mlflow.log_artifact(
        str(preprocessor_path),
        artifact_path="model",
    )
    mlflow.log_artifact(
        str(metrics_path),
        artifact_path="evaluation",
    )

    # Register the trained PyTorch model with MLflow.
    registered_model_name = mlflow_config[
        "registered_model_name"
    ]

    model.eval()

    model_info = mlflow_pytorch.log_model(
        pytorch_model=model,
        name="pytorch-model",
        registered_model_name=registered_model_name,
        input_example=train_features[:1],
    )

    registered_version = model_info.registered_model_version

    if registered_version is None:
        raise RuntimeError(
            "MLflow did not return a registered model version"
        )

    preprocessor["model_version"] = str(
        registered_version
    )
    preprocessor["mlflow_run_id"] = model_info.run_id

    preprocessor_path.write_text(
        json.dumps(preprocessor, indent=2),
        encoding="utf-8",
    )

    mlflow.log_artifact(
        str(preprocessor_path),
        artifact_path="model",
    )

    print(f"Registered model: {registered_model_name}")
    print(f"Registered version: {registered_version}")
    print(f"Model URI: {model_info.model_uri}")    
    active_run = mlflow.active_run()

    if active_run is None:
        raise RuntimeError(
            "MLflow run ended unexpectedly"
        )

    run_id = active_run.info.run_id

    mlflow.end_run()

    print(f"MLflow run ID: {run_id}")

    print(f"Selected threshold: {threshold:.4f}")
    print(
        "Validation metrics:\n"
        + json.dumps(validation_metrics, indent=2)
    )
    print(
        "Test metrics:\n"
        + json.dumps(test_metrics, indent=2)
    )
    print(f"Saved model: {model_path}")
    print(f"Saved preprocessor: {preprocessor_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    train(Path("configs/training.yaml"))