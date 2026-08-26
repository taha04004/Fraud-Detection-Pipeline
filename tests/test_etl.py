import pandas as pd

from src.features.transformations import add_features


def test_add_features() -> None:
    result = add_features(pd.DataFrame({"Time": [7200.0], "Amount": [9.0]}))
    assert result.loc[0, "Hour"] == 2.0
    assert "AmountLog" in result

