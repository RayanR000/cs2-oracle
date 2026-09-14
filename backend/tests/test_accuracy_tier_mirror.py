"""The price_tier discriminator must survive the round trip to the mirror.

Migration 0019 added ``price_tier`` to ``prediction_accuracy`` and 0020 put it
in the unique constraint, but the Parquet mirror predates both. ``_append_parquet``
intersected schemas, so every write silently dropped the column: the served file
held six indistinguishable rows per (horizon, model) — four tiers, the all-tiers
aggregate, and the tick-dominated tier 0 — with no way to tell them apart.

The headline is a separate hole: ``_score_groups`` computed it and only logged
it, so the one number quoted as "the model's accuracy" was never stored and
could not be audited.
"""

from __future__ import annotations

import pandas as pd
import pytest
from backtest.scoring import HEADLINE_MIN_TIER, HEADLINE_TIER, score_by_tier
from db.parquet import _append_parquet


def _record(tier, correct, base=5.0):
    return {
        "abs_error": 0.1,
        "pct_error": 2.0,
        "sq_error": 0.01,
        "direction_correct": correct,
        "predicted_direction": "up",
        "actual_direction": "up" if correct else "down",
        "in_interval": 1,
        "confidence": "high",
        "base_price": base,
        "actual_price": base + 0.1,
        "price_tier": tier,
        "item_id": 1,
    }


class TestMirrorSchemaWidening:
    def test_append_adds_a_column_the_existing_file_lacks(self, tmp_path):
        """The regression: a mirror written before 0019 must gain price_tier."""
        path = tmp_path / "prediction_accuracy.parquet"
        pd.DataFrame(
            [
                {"prediction_type": "forecast", "horizon_days": 3, "sample_count": 10},
            ]
        ).to_parquet(path, index=False)

        _append_parquet(
            path,
            pd.DataFrame(
                [
                    {"prediction_type": "forecast", "horizon_days": 3, "sample_count": 20, "price_tier": 1},
                ]
            ),
            ["prediction_type", "horizon_days", "price_tier"],
        )

        out = pd.read_parquet(path)
        assert "price_tier" in out.columns, "price_tier was intersected away"
        assert set(out["price_tier"].dropna()) == {1}

    def test_pre_existing_rows_survive_with_null_in_the_new_column(self, tmp_path):
        path = tmp_path / "t.parquet"
        pd.DataFrame(
            [
                {"prediction_type": "forecast", "horizon_days": 7, "sample_count": 10},
            ]
        ).to_parquet(path, index=False)

        _append_parquet(
            path,
            pd.DataFrame(
                [
                    {"prediction_type": "forecast", "horizon_days": 3, "sample_count": 20, "price_tier": 2},
                ]
            ),
            ["prediction_type", "horizon_days", "price_tier"],
        )

        out = pd.read_parquet(path).sort_values("horizon_days")
        assert len(out) == 2, "the non-matching existing row was dropped"
        assert out[out.horizon_days == 7]["price_tier"].isna().all()

    def test_a_column_the_new_rows_lack_is_not_blanked_on_survivors(self, tmp_path):
        path = tmp_path / "t.parquet"
        pd.DataFrame(
            [
                {"prediction_type": "forecast", "horizon_days": 7, "sample_count": 10, "evaluation_window_days": 30},
            ]
        ).to_parquet(path, index=False)

        _append_parquet(
            path,
            pd.DataFrame(
                [
                    {"prediction_type": "forecast", "horizon_days": 3, "sample_count": 20},
                ]
            ),
            ["prediction_type", "horizon_days"],
        )

        out = pd.read_parquet(path)
        survivor = out[out.horizon_days == 7].iloc[0]
        assert survivor["evaluation_window_days"] == 30

    def test_dedup_distinguishes_tiers(self, tmp_path):
        """Two tiers of the same (type, horizon) are distinct rows, not a collision."""
        path = tmp_path / "t.parquet"
        keys = ["prediction_type", "horizon_days", "price_tier"]
        rows = [
            {"prediction_type": "forecast", "horizon_days": 3, "sample_count": 10 * t, "price_tier": t}
            for t in (0, 1, 2)
        ]
        _append_parquet(path, pd.DataFrame(rows), keys)
        # re-writing tier 1 must replace only tier 1
        _append_parquet(
            path,
            pd.DataFrame([{"prediction_type": "forecast", "horizon_days": 3, "sample_count": 999, "price_tier": 1}]),
            keys,
        )
        out = pd.read_parquet(path).sort_values("price_tier")
        assert list(out["price_tier"]) == [0, 1, 2], "tier rows collided on dedup"
        assert out[out.price_tier == 1]["sample_count"].iloc[0] == 999
        assert out[out.price_tier == 2]["sample_count"].iloc[0] == 20


class TestHeadlineIsPersisted:
    def test_score_by_tier_emits_a_headline_row(self):
        records = (
            [_record(0, 1, base=0.5)] * 8  # penny, all correct
            + [_record(1, 0, base=2.0)] * 4  # >=$1, all wrong
            + [_record(2, 0, base=7.0)] * 4
        )
        out = score_by_tier(records)
        tiers = {t for t, _, _ in out}
        assert HEADLINE_TIER in tiers, "headline row is not persisted"

        headline = next(m for t, m, _ in out if t == HEADLINE_TIER)
        n = next(c for t, _, c in out if t == HEADLINE_TIER)
        assert n == 8, "headline must cover exactly the >=$1 records"
        assert headline["directional_accuracy"] == 0.0

        allt = next(m for t, m, _ in out if t is None)
        assert allt["directional_accuracy"] == pytest.approx(50.0)

    def test_headline_sentinel_cannot_collide_with_a_real_tier(self):
        assert HEADLINE_TIER < 0
        assert HEADLINE_TIER != HEADLINE_MIN_TIER

    def test_headline_absent_when_no_records_reach_the_min_tier(self):
        out = score_by_tier([_record(0, 1, base=0.5)] * 5)
        assert HEADLINE_TIER not in {t for t, _, _ in out}
