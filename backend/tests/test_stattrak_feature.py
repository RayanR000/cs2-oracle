import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.forecaster import ItemForecaster  # noqa: E402


def test_present_indicator():
    df = pd.DataFrame({"st_premium": [2.1, np.nan]})
    out = ItemForecaster._compute_stattrak_feature(df.copy())
    assert out.iloc[0]["st_premium_present"] == 1
    assert out.iloc[1]["st_premium_present"] == 0


def test_absent_column_is_safe():
    df = pd.DataFrame({"price": [10.0]})
    out = ItemForecaster._compute_stattrak_feature(df.copy())
    assert out.iloc[0]["st_premium_present"] == 0
