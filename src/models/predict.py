"""Reusable prediction helpers."""

import torch


def predict_probability(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(features))

