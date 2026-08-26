"""Evaluation utilities for imbalanced binary classification."""

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def classification_metrics(targets, probabilities, threshold: float = 0.5) -> dict[str, float]:
    predictions = probabilities >= threshold
    return {
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "f1": float(f1_score(targets, predictions)),
    }

