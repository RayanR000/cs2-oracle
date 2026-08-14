import numpy as np
import pandas as pd

from backtest.longshort import (
    decile_longshort_by_date, net_of_cost, direction_records)


def _one_date(n, rng):
    d = np.repeat(pd.to_datetime(["2026-05-01"]), n)
    return d, rng.normal(size=n), rng.normal(size=n)


def test_decile_spread_positive_when_score_matches_realised():
    rng = np.random.default_rng(0)
    n = 200
    dates = np.repeat(pd.to_datetime(["2026-05-01", "2026-05-02"]), n // 2)
    score = rng.normal(size=n)
    realised = score + 0.1 * rng.normal(size=n)  # top-score → top-realised
    out = decile_longshort_by_date(score, realised, dates)
    assert out["n_dates"] == 2
    assert out["mean"] > 0  # top decile outperforms bottom


def test_decile_spread_zero_mean_when_score_is_noise():
    rng = np.random.default_rng(1)
    n = 6000
    dates = np.repeat(pd.to_datetime([f"2026-05-{d:02d}" for d in range(1, 11)]), n // 10)
    score = rng.normal(size=n)
    realised = rng.normal(size=n)  # independent
    out = decile_longshort_by_date(score, realised, dates)
    assert abs(out["mean"]) < 0.15  # no systematic spread


def test_mask_restricts_to_tied_rows():
    rng = np.random.default_rng(2)
    n = 100
    dates = np.repeat(pd.to_datetime(["2026-05-01"]), n)
    score = rng.normal(size=n)
    realised = rng.normal(size=n)
    mask = np.zeros(n, dtype=bool)
    mask[:10] = True  # too few rows after masking → no date read
    out = decile_longshort_by_date(score, realised, dates, mask=mask,
                                   min_rows=20)
    assert out["n_dates"] == 0
    assert out["mean"] is None


def test_net_of_cost_subtracts_roundtrip():
    assert abs(net_of_cost(0.30, 0.20) - 0.10) < 1e-9


def test_direction_records_shape_and_correctness():
    # Two rows above median score → predicted "up"; realised follows.
    score = np.array([0.0, 1.0, 2.0, 3.0] * 6)  # 24 rows, one date
    realised = score.copy()
    dates = np.repeat(pd.to_datetime(["2026-05-01"]), len(score))
    recs = direction_records(score, realised, dates, min_rows=20)
    assert len(recs) == len(score)
    assert set(recs[0]) == {"predicted_direction", "actual_direction",
                            "direction_correct", "forecast_date"}
    # score == realised → every call correct
    assert all(r["direction_correct"] for r in recs)
    assert recs[0]["forecast_date"] == "2026-05-01"


def test_direction_records_skips_thin_dates():
    score = np.arange(10, dtype=float)
    realised = score.copy()
    dates = np.repeat(pd.to_datetime(["2026-05-01"]), 10)
    assert direction_records(score, realised, dates, min_rows=20) == []
