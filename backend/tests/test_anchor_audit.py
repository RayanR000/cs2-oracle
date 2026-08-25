"""The serving-anchor audit: it must observe, and it must never decide.

This exists because the served `current_price` cannot be rebuilt from the
archive afterwards, so an arm that changes the anchor is being chosen blind. Two
properties matter more than the numbers it prints: it runs BEFORE the base is
overwritten (or it records the answer, not the inputs), and it cannot take down
a forecast.
"""
import logging

import pandas as pd
import pytest

from models.forecaster import ItemForecaster


def _frame(anchor="2026-08-04", lags=(0, 0, 1, 5)):
    """One item per lag, each with three observations ending `lag` days back."""
    anchor = pd.Timestamp(anchor)
    rows = []
    for i, lag in enumerate(lags):
        last = anchor - pd.Timedelta(lag, "D")
        for k in range(3):
            rows.append({"item_id": f"item-{i}", "date": last - pd.Timedelta(k, "D"),
                         "price": 100.0 + k})
    return pd.DataFrame(rows).sort_values(["item_id", "date"])


def _audit_path():
    """Where the dump lands: the archive when one is checked out, else the cache
    dir. The archive branch is the one that matters — a file written to
    `backend/data/` dies with the CI runner."""
    from pathlib import Path as P

    root = P(__file__).resolve().parents[2]
    archive = root / "price-archive"
    base = (archive / "ops" / "anchor_audit" if archive.is_dir()
            else root / "backend" / "data")
    return base / "anchor_audit_2026-08-04.parquet"


def _latest(df):
    out = df.groupby("item_id").last().reset_index()
    out["_smoothed_price"] = out["price"]
    return out


def test_it_reports_frame_currency_at_the_anchor(caplog):
    """The summary has to answer the question it was built for: how much of the
    frame actually reaches the anchor day."""
    df = _frame(lags=(0, 0, 1, 5))
    with caplog.at_level(logging.INFO, logger="models.forecaster"):
        ItemForecaster._audit_serving_anchor(df, _latest(df), pd.Timestamp("2026-08-04"))
    line = next(r.message for r in caplog.records if "Anchor audit @" in r.message)
    assert "2026-08-04" in line
    assert "50.0% current" in line          # two of four items land on the anchor
    assert "median 0d" in line
    assert any("rows per day" in r.message for r in caplog.records)


def test_a_stale_frame_is_visible_as_lag(caplog):
    """The incompleteness hypothesis predicts a lag tail; it must not be hidden."""
    df = _frame(lags=(2, 3, 4, 5))
    with caplog.at_level(logging.INFO, logger="models.forecaster"):
        ItemForecaster._audit_serving_anchor(df, _latest(df), pd.Timestamp("2026-08-04"))
    line = next(r.message for r in caplog.records if "Anchor audit @" in r.message)
    assert "0.0% current" in line
    # No row reaches the anchor day at all — which is the state the hypothesis
    # says the serving path is quoting from.
    assert line.split("audit @")[1].strip().startswith("2026-08-04: 0 of 12")
    assert "median 4d" in line   # lags 2,3,4,5


def test_it_never_raises(caplog):
    """A diagnostic that can kill a forecast is worse than no diagnostic."""
    bad = pd.DataFrame({"nonsense": [1, 2, 3]})
    with caplog.at_level(logging.WARNING, logger="models.forecaster"):
        ItemForecaster._audit_serving_anchor(bad, bad, pd.Timestamp("2026-08-04"))
    assert any("Anchor audit failed (ignored)" in r.message for r in caplog.records)


def test_the_parquet_is_opt_in(tmp_path, monkeypatch, caplog):
    """Unconditional per-item dumps on the serving path are a cost, not a default."""
    df = _frame()
    monkeypatch.delenv("ANCHOR_AUDIT", raising=False)
    with caplog.at_level(logging.INFO, logger="models.forecaster"):
        ItemForecaster._audit_serving_anchor(df, _latest(df), pd.Timestamp("2026-08-04"))
    assert not any("wrote" in r.message for r in caplog.records)


def test_the_audit_runs_before_the_base_is_overwritten():
    """`_serving_base_price` replaces `price` with the served base, so an audit
    after it would find price == smoothed for every item and record nothing.

    Pinned by reading the source order rather than by mocking: the ordering is
    the property, and a mock would pass against a call site that no longer
    exists.
    """
    import inspect

    src = inspect.getsource(ItemForecaster.predict)
    audit = src.index("_audit_serving_anchor(df")
    overwrite = src.index('latest_rows["price"], outlier_mask = self._serving_base_price')
    assert audit < overwrite


@pytest.mark.parametrize("lag", [0, 1, 3])
def test_the_per_item_dump_records_inputs_not_the_answer(tmp_path, monkeypatch, lag):
    """The dump has to carry the raw quote AND the smoothed value separately —
    one column holding the served base would be exactly the thing that cannot be
    reconstructed today."""
    monkeypatch.setenv("ANCHOR_AUDIT", "1")
    df = _frame(lags=(lag,))
    latest = _latest(df)
    latest["_smoothed_price"] = latest["price"] * 1.5     # deliberately different
    ItemForecaster._audit_serving_anchor(df, latest, pd.Timestamp("2026-08-04"))
    got = pd.read_parquet(_audit_path())
    assert {"price", "_smoothed_price", "last_obs_date", "n_obs_in_span",
            "anchor_date", "captured_at"} <= set(got.columns)
    assert got["_smoothed_price"].iloc[0] != got["price"].iloc[0]
    _audit_path().unlink()


def test_the_dump_lands_in_the_archive_when_one_exists(monkeypatch, caplog):
    """It has to ride the daily publish. `item_forecasts.parquet` stopped at
    2026-07-29 because a CI write went somewhere the publish step never saw."""
    from pathlib import Path as P

    archive = P(__file__).resolve().parents[2] / "price-archive"
    if not archive.is_dir():
        pytest.skip("no archive checked out")
    monkeypatch.setenv("ANCHOR_AUDIT", "1")
    df = _frame()
    with caplog.at_level(logging.INFO, logger="models.forecaster"):
        ItemForecaster._audit_serving_anchor(df, _latest(df), pd.Timestamp("2026-08-04"))
    written = archive / "ops" / "anchor_audit" / "anchor_audit_2026-08-04.parquet"
    assert written.exists()
    written.unlink()
