import torch

from src.models.network import FraudClassifier


def test_model_output_shape() -> None:
    model = FraudClassifier(input_size=30)
    assert model(torch.zeros((4, 30))).shape == (4,)

