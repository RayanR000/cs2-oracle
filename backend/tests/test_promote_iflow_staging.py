import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import promote_iflow_staging as p


def _staged(days, items=("AK-47 | Redline (Field-Tested)",), price=10.0,
            source=p.SOURCE):
    rows = []
    for d in days:
        for it in items:
            rows.append({"item_slug": it, "day": pd.Timestamp(d), "source": source,
                         "mean_price": price, "volume": None,
                         "ingested_at": pd.Timestamp("2026-08-29")})
    return pd.DataFrame(rows, columns=p.PRICE_COLS)


def _write_staging(tmp_path, df, name="prices-2026-04.parquet"):
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    df.to_parquet(staging / name, index=False)
    return staging


DAYS = pd.date_range("2026-04-16", "2026-04-18").strftime("%Y-%m-%d").tolist()
ITEMS = tuple(f"Item {i}" for i in range(5))


def test_load_staged_filters_to_range(tmp_path):
    df = _staged(["2026-04-15", "2026-04-16", "2026-04-17"], ITEMS)
    staging = _write_staging(tmp_path, df)

    out = p.load_staged(staging, "2026-04-16", "2026-04-17")

    assert out["day"].nunique() == 2
    assert out["day"].min() == pd.Timestamp("2026-04-16")
    assert list(out.columns) == p.PRICE_COLS


def test_load_staged_rejects_foreign_source(tmp_path):
    staging = _write_staging(tmp_path, _staged(DAYS, ITEMS, source="aggregator_buff163"))

    with pytest.raises(ValueError, match="expected only source"):
        p.load_staged(staging, DAYS[0], DAYS[-1])


def test_load_staged_rejects_empty_range(tmp_path):
    staging = _write_staging(tmp_path, _staged(DAYS, ITEMS))

    with pytest.raises(ValueError, match="no staged rows"):
        p.load_staged(staging, "2026-06-01", "2026-06-02")


def test_validate_coverage_flags_missing_day():
    prices = _staged(["2026-04-16", "2026-04-18"], ITEMS)

    with pytest.raises(AssertionError, match="missing day 2026-04-17"):
        p.validate_coverage(prices, "2026-04-16", "2026-04-18", min_items=1)


def test_validate_coverage_flags_sparse_day():
    prices = _staged(DAYS, ITEMS)

    with pytest.raises(AssertionError, match="low item count"):
        p.validate_coverage(prices, DAYS[0], DAYS[-1], min_items=6)


def test_validate_coverage_passes_on_full_range():
    p.validate_coverage(_staged(DAYS, ITEMS), DAYS[0], DAYS[-1], min_items=5)


def _canon(tmp_path, days, items, price, source="aggregator_buff163"):
    out = tmp_path / "archive"
    out.mkdir(exist_ok=True)
    df = _staged(days, items, price=price, source=source)
    df.to_parquet(out / "prices-2026-04.parquet", index=False)
    return out


def test_cross_check_returns_none_without_overlap(tmp_path):
    out = _canon(tmp_path, DAYS, ITEMS, 10.0, source="aggregator_csfloat")

    assert p.cross_check_buff163(_staged(DAYS, ITEMS), str(out / "prices-*.parquet")) is None


def test_cross_check_returns_none_without_archive(tmp_path):
    assert p.cross_check_buff163(_staged(DAYS, ITEMS),
                                 str(tmp_path / "nope" / "prices-*.parquet")) is None


def test_cross_check_accepts_agreeing_levels(tmp_path):
    out = _canon(tmp_path, DAYS, ITEMS, 10.0)

    stats = p.cross_check_buff163(_staged(DAYS, ITEMS, price=10.1),
                                  str(out / "prices-*.parquet"))

    assert stats["rows"] == len(DAYS) * len(ITEMS)
    assert stats["median_log_ratio"] == pytest.approx(0.00995, abs=1e-4)
    assert stats["tail_frac_gt25pct"] == 0.0


def test_cross_check_rejects_biased_levels(tmp_path):
    out = _canon(tmp_path, DAYS, ITEMS, 10.0)

    with pytest.raises(AssertionError, match="suspect FX or slug mapping"):
        p.cross_check_buff163(_staged(DAYS, ITEMS, price=20.0),
                              str(out / "prices-*.parquet"))


def test_run_refuses_past_feed_end(tmp_path):
    with pytest.raises(ValueError, match="past the iflow feed end"):
        p.run("2026-05-19", "2026-05-25", tmp_path, tmp_path, dry_run=True)


def test_run_dry_run_writes_nothing(tmp_path):
    staging = _write_staging(tmp_path, _staged(DAYS, ITEMS))
    out = tmp_path / "archive"

    p.run(DAYS[0], DAYS[-1], staging, out, dry_run=True, min_items=5)

    assert not out.exists()


def test_run_appends_and_is_idempotent(tmp_path):
    staging = _write_staging(tmp_path, _staged(DAYS, ITEMS))
    out = tmp_path / "archive"

    p.run(DAYS[0], DAYS[-1], staging, out, min_items=5)
    first = pd.read_parquet(out / "prices-2026-04.parquet")
    p.run(DAYS[0], DAYS[-1], staging, out, min_items=5)
    second = pd.read_parquet(out / "prices-2026-04.parquet")

    assert len(first) == len(DAYS) * len(ITEMS)
    assert len(second) == len(first)  # dedup keys collapse the re-run


def test_run_preserves_existing_other_source_rows(tmp_path):
    staging = _write_staging(tmp_path, _staged(DAYS, ITEMS))
    out = _canon(tmp_path, DAYS, ITEMS, 10.0, source="aggregator_steam_17mafo")

    p.run(DAYS[0], DAYS[-1], staging, out, min_items=5)
    merged = pd.read_parquet(out / "prices-2026-04.parquet")

    assert set(merged["source"]) == {"aggregator_steam_17mafo", p.SOURCE}
    assert merged.groupby("source").size().to_dict()[p.SOURCE] == len(DAYS) * len(ITEMS)


CHECK_DAYS = pd.date_range("2026-04-01", "2026-04-03").strftime("%Y-%m-%d").tolist()


def test_run_validates_against_pre_gap_window(tmp_path, capsys):
    # Staged rows span the check window AND the promoted range; canonical
    # buff163 exists only in the check window, as in the real archive.
    staging = _write_staging(tmp_path, _staged(CHECK_DAYS + DAYS, ITEMS, price=10.1))
    out = _canon(tmp_path, CHECK_DAYS, ITEMS, 10.0)

    p.run(DAYS[0], DAYS[-1], staging, out, dry_run=True, min_items=5,
          check_start=CHECK_DAYS[0], check_end=CHECK_DAYS[-1])

    printed = capsys.readouterr().out
    assert "Cross-check vs aggregator_buff163 on 2026-04-01..2026-04-03" in printed
    assert "skipped" not in printed


def test_run_skips_check_when_window_absent_from_staging(tmp_path, capsys):
    staging = _write_staging(tmp_path, _staged(DAYS, ITEMS))
    out = _canon(tmp_path, CHECK_DAYS, ITEMS, 10.0)

    p.run(DAYS[0], DAYS[-1], staging, out, dry_run=True, min_items=5,
          check_start=CHECK_DAYS[0], check_end=CHECK_DAYS[-1])

    assert "Cross-check skipped" in capsys.readouterr().out


def test_run_propagates_biased_check_window(tmp_path):
    staging = _write_staging(tmp_path, _staged(CHECK_DAYS + DAYS, ITEMS, price=20.0))
    out = _canon(tmp_path, CHECK_DAYS, ITEMS, 10.0)

    with pytest.raises(AssertionError, match="suspect FX or slug mapping"):
        p.run(DAYS[0], DAYS[-1], staging, out, dry_run=True, min_items=5,
              check_start=CHECK_DAYS[0], check_end=CHECK_DAYS[-1])
