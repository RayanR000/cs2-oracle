"""The rebase guard for the centre null.

The first run of `centre_vs_lastprice.py` read `r_hat` off `base_price` and
reported the GBM centre as ~2.7x WORSE than a random walk. That number was the
anchor wedge, not the centre: `predict()` quotes the triple off `current_price`
while the outcome legs resolve off `base_price`. These tests pin the corrected
basis so the wedge cannot creep back in behind a plausible-looking figure.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "centre_vs_lastprice.py"


@pytest.fixture(scope="module")
def cvl():
    spec = importlib.util.spec_from_file_location("centre_vs_lastprice", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path, rows):
    ops = tmp_path / "ops"
    ops.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ops / "forecast_outcomes.parquet")
    return tmp_path


def _row(**kw):
    base = dict(item_id=1, forecast_date=pd.Timestamp("2026-08-04"), horizon_days=3,
                base_price=100.0, actual_price=110.0, current_price=100.0,
                predicted_price_low=95.0, predicted_price_mid=105.0,
                predicted_price_high=115.0, model_version="lgbm-v3",
                base_stale_run_days=0.0)
    base.update(kw)
    return base


def test_r_hat_is_read_off_the_quote_not_the_resolved_base(cvl, tmp_path):
    """A stale quote must not be charged to the model as a centre error."""
    # Quoted from 50 while the outcome resolved off 100: the model predicted a
    # +5% move (52.5/50), and that is what must be read, not +425% (52.5/100-1).
    d = cvl._load(_write(tmp_path, [_row(current_price=50.0,
                                         predicted_price_low=47.5,
                                         predicted_price_mid=52.5,
                                         predicted_price_high=57.5)]))
    assert d["r_hat"].iloc[0] == pytest.approx(0.05)
    assert d["r_actual"].iloc[0] == pytest.approx(0.10)
    # Half-widths ride the same basis.
    assert d["w_lo"].iloc[0] == pytest.approx(0.10)
    assert d["w_hi"].iloc[0] == pytest.approx(0.10)


def test_a_missing_quote_falls_back_to_the_base(cvl, tmp_path):
    """`_quote_basis`'s fallback: legacy rows predating the column stay in."""
    for bad in (None, 0.0):
        d = cvl._load(_write(tmp_path, [_row(current_price=bad)]))
        assert len(d) == 1
        assert d["r_hat"].iloc[0] == pytest.approx(0.05)


def test_excluded_forecast_dates_are_dropped(cvl, tmp_path):
    """The panel matches the published one, or the numbers are incomparable."""
    d = cvl._load(_write(tmp_path, [
        _row(forecast_date=pd.Timestamp("2026-07-19")),   # dead-band rule
        _row(forecast_date=pd.Timestamp("2025-12-01")),   # replay
        _row(forecast_date=pd.Timestamp("2026-08-04")),   # kept
    ]))
    assert len(d) == 1
    assert d["forecast_date"].iloc[0] == pd.Timestamp("2026-08-04").date()


def test_sub_dollar_rows_are_out_of_the_served_cohort(cvl, tmp_path):
    d = cvl._load(_write(tmp_path, [_row(base_price=0.5, current_price=0.5)]))
    assert d.empty


def test_coverage_is_measured_at_identical_width(cvl):
    """Transplanting the half-widths is what isolates the centre."""
    r = np.array([0.20, -0.20])
    w_lo = w_hi = np.array([0.10, 0.10])
    gbm = cvl._covered(r, np.array([0.15, 0.0]), w_lo, w_hi)
    naive = cvl._covered(r, np.zeros(2), w_lo, w_hi)
    assert gbm.tolist() == [True, False]     # the centre moved onto the move
    assert naive.tolist() == [False, False]  # ...and the widths never changed


def test_skill_is_zero_when_the_centre_predicts_no_move(cvl):
    r = np.array([0.05, -0.03, 0.10])
    assert cvl._mae(r, np.zeros_like(r)) == pytest.approx(np.mean(np.abs(r)))


def test_lambda_zero_wins_when_the_centre_is_pure_noise(cvl):
    """The shrinkage dial must collapse to the random walk on a noise centre."""
    import numpy as np
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.05, 5000)
    noise = rng.normal(0, 0.05, 5000)      # independent of r
    lam, _ = cvl._best_lambda(r, noise)
    assert lam == 0.0


def test_lambda_one_wins_when_the_centre_is_the_truth(cvl):
    """...and must NOT shrink a centre that actually carries the move."""
    import numpy as np
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.05, 5000)
    lam, mae = cvl._best_lambda(r, r)
    assert lam == 1.0
    assert mae == pytest.approx(0.0, abs=1e-12)


def test_lambda_recovers_a_known_over_expression(cvl):
    """A centre twice as loud as the move shrinks to about a half."""
    import numpy as np
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.05, 20000)
    lam, _ = cvl._best_lambda(r, 2.0 * r)
    assert lam == pytest.approx(0.5, abs=0.05)
