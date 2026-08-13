import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.forecaster import ItemForecaster  # noqa: E402


def test_spread_computed_when_bid_present():
    df = pd.DataFrame({"price": [10.0, 20.0], "buff_bid": [8.0, np.nan]})
    out = ItemForecaster._compute_bid_features(df.copy())
    assert abs(out.iloc[0]["bid_ask_spread"] - 0.2) < 1e-9
    assert out.iloc[0]["bid_present"] == 1
    assert out.iloc[1]["bid_present"] == 0


def test_no_bid_column_is_safe():
    df = pd.DataFrame({"price": [10.0]})
    out = ItemForecaster._compute_bid_features(df.copy())
    assert out.iloc[0]["bid_present"] == 0
    assert pd.isna(out.iloc[0]["bid_ask_spread"])
