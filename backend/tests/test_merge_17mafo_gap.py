import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_17mafo_gap as m


SAMPLE_DAY = {
    "AK-47 | Redline (Field-Tested)": {
        "steam": {"last_24h": 47.63418, "last_7d": 47.37, "last_30d": 46.6,
                  "last_90d": 48.5, "last_ever": 45.99}
    },
    "★ Karambit | Doppler (Factory New)": {  # null recent price -> skipped
        "steam": {"last_24h": None, "last_7d": None, "last_30d": None,
                  "last_90d": None, "last_ever": 1000.0}
    },
    "Broken Item": {"steam": {}},            # no last_24h -> skipped
    "Weird Item": "not-a-dict",              # non-dict value -> skipped
}


def test_transform_day_columns_and_values():
    df = m.transform_day(SAMPLE_DAY, "2026-04-16")

    assert list(df.columns) == m.PRICE_COLS
    assert len(df) == 1  # only the valid Redline row survives

    row = df.iloc[0]
    assert row["item_slug"] == "AK-47 | Redline (Field-Tested)"
    assert row["day"] == pd.Timestamp("2026-04-16")
    assert row["source"] == "aggregator_steam_17mafo"
    assert row["mean_price"] == 47.63418
    assert row["median_price"] == 47.63418
    assert row["min_price"] == 47.63418
    assert row["max_price"] == 47.63418
    assert row["volume"] == 0


def test_transform_day_coalesces_to_last_7d_then_30d():
    day = {
        "Liquid": {"steam": {"last_24h": 10.0, "last_7d": 11.0, "last_30d": 12.0}},
        "Only7d": {"steam": {"last_24h": None, "last_7d": 20.0, "last_30d": 21.0}},
        "Only30d": {"steam": {"last_24h": None, "last_7d": None, "last_30d": 30.0}},
        "NoRecent": {"steam": {"last_24h": None, "last_7d": None, "last_30d": None,
                               "last_90d": 99.0, "last_ever": 88.0}},
    }
    df = m.transform_day(day, "2026-04-16")
    prices = dict(zip(df["item_slug"], df["mean_price"]))
    assert prices["Liquid"] == 10.0     # prefers freshest
    assert prices["Only7d"] == 20.0     # falls back to 7d
    assert prices["Only30d"] == 30.0    # falls back to 30d
    assert "NoRecent" not in prices     # 90d/ever NOT used -> skipped


def test_prices_to_snapshots():
    prices = m.transform_day(SAMPLE_DAY, "2026-04-16")
    snaps = m.prices_to_snapshots(prices)

    assert list(snaps.columns) == m.SNAP_COLS
    assert len(snaps) == 1
    assert snaps.iloc[0]["price"] == 47.63418
    assert snaps.iloc[0]["volume"] == 0
    assert snaps.iloc[0]["source"] == "aggregator_steam_17mafo"


import pytest


def _prices_for_dates(dates, n_items):
    frames = []
    for d in dates:
        obj = {f"Item {i}": {"steam": {"last_24h": 1.0 + i}} for i in range(n_items)}
        frames.append(m.transform_day(obj, d))
    return pd.concat(frames, ignore_index=True)


def test_gap_dates_inclusive():
    dates = m.gap_dates("2026-04-16", "2026-07-08")
    assert dates[0] == "2026-04-16"
    assert dates[-1] == "2026-07-08"
    assert len(dates) == 84


def test_validate_coverage_passes():
    dates = ["2026-04-16", "2026-04-17"]
    prices = _prices_for_dates(dates, n_items=25)
    m.validate_coverage(prices, dates, min_items=25)  # no raise


def test_validate_coverage_missing_day_raises():
    dates = ["2026-04-16", "2026-04-17"]
    prices = _prices_for_dates(["2026-04-16"], n_items=25)  # 04-17 missing
    with pytest.raises(AssertionError, match="2026-04-17"):
        m.validate_coverage(prices, dates, min_items=25)


def test_validate_coverage_low_count_raises():
    dates = ["2026-04-16"]
    prices = _prices_for_dates(dates, n_items=5)
    with pytest.raises(AssertionError, match="item count"):
        m.validate_coverage(prices, dates, min_items=25)


import json


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload).encode()


class _FakeSession:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload or {}
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        return _FakeResp(self.status_code, self.payload)


def test_fetch_day_downloads_when_absent(tmp_path):
    sess = _FakeSession(200, {"Item A": {"steam": {"last_24h": 5.0}}})
    path = m.fetch_day("2026-04-16", tmp_path, session=sess)
    assert path.exists()
    assert sess.calls == 1
    assert m.load_day(path)["Item A"]["steam"]["last_24h"] == 5.0


def test_fetch_day_uses_cache(tmp_path):
    (tmp_path / "2026-04-16.json").write_text('{"Item A": {"steam": {"last_24h": 9.0}}}')
    sess = _FakeSession(200, {})
    path = m.fetch_day("2026-04-16", tmp_path, session=sess)
    assert sess.calls == 0                      # cache hit, no network
    assert m.load_day(path)["Item A"]["steam"]["last_24h"] == 9.0


def test_fetch_day_raises_on_error(tmp_path):
    sess = _FakeSession(404, {})
    with pytest.raises(RuntimeError, match="404"):
        m.fetch_day("2026-04-16", tmp_path, session=sess)


import duckdb


def _fake_fetch_factory(payloads_by_date, cache_dir):
    """Return a fetch_day-compatible callable backed by in-memory payloads."""
    def _fetch(date, cdir, refresh=False, session=None):
        cdir.mkdir(parents=True, exist_ok=True)
        p = cdir / f"{date}.json"
        p.write_text(json.dumps(payloads_by_date[date]))
        return p
    return _fetch


def _payloads(dates, n_items):
    return {d: {f"Item {i}": {"steam": {"last_24h": float(i + 1)}}
                for i in range(n_items)} for d in dates}


def test_run_writes_parquet_and_is_idempotent(tmp_path):
    dates = ["2026-04-16", "2026-04-17"]
    payloads = _payloads(dates, n_items=30)
    out_dir = tmp_path / "price-archive"
    cache_dir = tmp_path / "raw"
    fetch = _fake_fetch_factory(payloads, cache_dir)

    m.run("2026-04-16", "2026-04-17", out_dir, cache_dir,
          min_items=30, fetch=fetch)

    prices_path = out_dir / "prices-2026.parquet"
    snaps_path = out_dir / "snapshots-2026.parquet"
    assert prices_path.exists() and snaps_path.exists()

    con = duckdb.connect()
    n1 = con.sql(f"SELECT COUNT(*) FROM read_parquet('{prices_path}')").fetchone()[0]
    assert n1 == 60  # 2 days x 30 items

    # re-run must not duplicate (dedup on item_slug, day, source)
    m.run("2026-04-16", "2026-04-17", out_dir, cache_dir,
          min_items=30, fetch=fetch)
    n2 = con.sql(f"SELECT COUNT(*) FROM read_parquet('{prices_path}')").fetchone()[0]
    assert n2 == 60


def test_run_dry_run_writes_nothing(tmp_path):
    dates = ["2026-04-16"]
    payloads = _payloads(dates, n_items=30)
    out_dir = tmp_path / "price-archive"
    cache_dir = tmp_path / "raw"
    fetch = _fake_fetch_factory(payloads, cache_dir)

    m.run("2026-04-16", "2026-04-16", out_dir, cache_dir,
          dry_run=True, min_items=30, fetch=fetch)
    assert not (out_dir / "prices-2026.parquet").exists()


def test_compute_basis_factors_missing_file(tmp_path):
    sf, sg, ef, eg = m.compute_basis_factors(tmp_path / "nope.parquet")
    assert sf == {} and sg == 1.0 and ef == {} and eg == 1.0


def test_compute_basis_factors_start_and_end(tmp_path):
    rows = []
    # item A overlap: buff=30, csfloat=30 (basis=30); steam_7d=44;
    # plus a higher extra source csgotrader=42 so full-consensus median > buff.
    for src, px in [("aggregator_buff163", 30.0), ("aggregator_csfloat", 30.0),
                    ("aggregator_csgotrader", 42.0), ("aggregator_steam_7d", 44.0)]:
        rows.append({"item_slug": "A", "day": pd.Timestamp("2026-07-11"),
                     "source": src, "mean_price": px, "min_price": px,
                     "max_price": px, "median_price": px, "volume": 0})
    pth = tmp_path / "prices-2026.parquet"
    pd.DataFrame(rows).to_parquet(pth, index=False)

    sf, sg, ef, eg = m.compute_basis_factors(pth)
    # start = median(buff sources)/steam = 30/44
    assert sf["A"] == pytest.approx(30.0 / 44.0)
    # end = median(all non-fallback sources incl csgotrader)/steam
    #   median of [30,30,42,44] = 36 -> 36/44
    assert ef["A"] == pytest.approx(36.0 / 44.0)
    assert sg == pytest.approx(30.0 / 44.0)
    assert eg == pytest.approx(36.0 / 44.0)


def test_apply_rescale_ramps_start_to_end(tmp_path):
    start_f = {"A": 0.60}
    end_f = {"A": 0.90}
    gap = m.transform_day({"A": {"steam": {"last_24h": 100.0}}}, "2026-04-16")  # start day
    out = m.apply_rescale(gap, start_f, 0.60, end_f, 0.90,
                          start_date="2026-04-16", end_date="2026-07-08")
    assert out.loc[0, "mean_price"] == pytest.approx(100.0 * 0.60)  # t=0 -> f_start

    gap_end = m.transform_day({"A": {"steam": {"last_24h": 100.0}}}, "2026-07-08")  # end day
    out_end = m.apply_rescale(gap_end, start_f, 0.60, end_f, 0.90,
                              start_date="2026-04-16", end_date="2026-07-08")
    assert out_end.loc[0, "mean_price"] == pytest.approx(100.0 * 0.90)  # t=1 -> f_end

    # midpoint item uses global fallback + geometric interpolation
    mid_day = "2026-05-27"  # ~halfway
    gap_mid = m.transform_day({"Z": {"steam": {"last_24h": 100.0}}}, mid_day)
    out_mid = m.apply_rescale(gap_mid, start_f, 0.60, end_f, 0.90,
                              start_date="2026-04-16", end_date="2026-07-08")
    import numpy as np
    span = (pd.Timestamp("2026-07-08") - pd.Timestamp("2026-04-16")).days
    tt = (pd.Timestamp(mid_day) - pd.Timestamp("2026-04-16")).days / span
    expected = 100.0 * np.exp((1 - tt) * np.log(0.60) + tt * np.log(0.90))
    assert out_mid.loc[0, "mean_price"] == pytest.approx(expected)
