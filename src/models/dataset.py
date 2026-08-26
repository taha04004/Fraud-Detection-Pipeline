"""PyTorch dataset for tabular fraud examples."""

import numpy as np
import torch
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray | None = None) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = None if targets is None else torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        if self.targets is None:
            return self.features[index]
        return self.features[index], self.targets[index]

