"""`engineer_features` must not depend on the row order of duplicate item-days.

Harness loaders that skip the vote (`walkforward_backtest._load_all_prices`,
which `compute_mde` and every paired gate reads) hand over several source rows
per item-day ordered only by `(item, day)`. DuckDB returns ties in any order, so
the plain-mean collapse summed the same values in a different order each run.
The last-bit difference flipped `stale_run_days`, which compares prices
bit-for-bit, and frozen-run label voiding moved with it: on the live archive
two identical h=3 builds differed by 16 rows. That is the `paired_mde`
"frame-construction nondeterminism" of the deep model review §11
(`docs/changelog/2026-08-07-paired-mde-fold-clustering.md`).
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from models.forecaster import ItemForecaster
from models.staleness import stale_run_days

_EVENTS = pd.DataFrame(columns=["date", "event_type", "name"])


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path_factory.mktemp("saved_models")))


def _multi_source(n_items=3, n_days=160, n_sources=5, start=date(2025, 1, 1)):
    """Several source rows per item-day, with runs where every source repeats.

    Prices carry many significant bits so a different summation order changes
    the mean's last bit; every fourth week the sources hold their quotes, which
    is the frozen run `stale_run_days` has to see identically on every build.
    """
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n_items):
        quotes = 3.0 + i + rng.random(n_sources) * 0.37
        for d in range(n_days):
            if (d // 7) % 4 != 3:
                quotes = quotes * (1.0 + rng.normal(0.0, 0.013, n_sources))
            for q in quotes:
                rows.append({"item_id": f"item-{i}", "date": start + timedelta(days=d), "price": q, "volume": 1.0})
    return pd.DataFrame(rows)


def test_shuffled_duplicate_rows_give_a_bit_identical_frame(forecaster):
    base = _multi_source()
    shuffled = base.sample(frac=1.0, random_state=1).sort_values(["item_id", "date"], kind="stable")
    ref = forecaster.engineer_features(base, _EVENTS).reset_index(drop=True)
    out = forecaster.engineer_features(shuffled.reset_index(drop=True), _EVENTS).reset_index(drop=True)
    pd.testing.assert_frame_equal(out, ref, check_exact=True)

    # The consequence that moved the paired frame: frozen-run voiding.
    runs = [stale_run_days(df, item_col="item_id", date_col="date", price_col="price") for df in (ref, out)]
    assert runs[0].max() > 0, "fixture lost its frozen runs, so the test proves nothing"
    np.testing.assert_array_equal(np.asarray(runs[0]), np.asarray(runs[1]))
