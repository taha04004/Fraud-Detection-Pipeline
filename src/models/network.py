"""Neural network architecture for binary fraud classification."""

import torch
from torch import nn


class FraudClassifier(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(-1)

