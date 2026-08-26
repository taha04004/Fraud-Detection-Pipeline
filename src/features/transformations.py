"""Feature transformations shared by training and inference."""

import numpy as np
import pandas as pd


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["AmountLog"] = np.log1p(result["Amount"])
    result["Hour"] = (result["Time"] / 3600) % 24
    return result

