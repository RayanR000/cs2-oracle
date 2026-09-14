from datetime import date

import numpy as np
import pandas as pd
import scripts.replay_lambdarank as rl
from scripts.replay_lambdarank import MIN_TIED_ROWS, SERVED_WINDOW, _anchor_metrics, _retrain_points


def _synthetic_anchor(n, signal, rng):
    """One anchor date with `n` items; `signal` scales how well score predicts
    the realised outcome. Returns (val_df, lr, q50, naive, outcomes, tied)."""
    anchor = pd.Timestamp("2026-05-01")
    item_id = np.arange(n)
    current = rng.uniform(2, 50, size=n)  # all >= $1 floor
    true_rank = rng.normal(size=n)
    lr = true_rank + 0.05 * rng.normal(size=n)
    q50 = 0.3 * true_rank + rng.normal(size=n)  # weaker orderer
    naive = rng.normal(size=n)
    realised = current * (1.0 + signal * true_rank * 0.01)
    val_df = pd.DataFrame({"item_id": item_id, "current": current, "date": anchor})
    # Outcome rows land inside the h=7 resolve window (anchor, anchor+7].
    out_day = anchor + pd.Timedelta(days=7)
    outcomes = pd.DataFrame({"item_id": item_id, "day": np.repeat(out_day, n), "price": realised})
    tied = pd.Series(True, index=item_id)  # everyone tied in the synthetic
    tied.index.name = "item_id"
    return val_df, lr, q50, naive, outcomes, tied


def test_returns_none_below_min_tied(monkeypatch):
    rng = np.random.default_rng(0)
    val_df, lr, q50, naive, outcomes, tied = _synthetic_anchor(MIN_TIED_ROWS - 1, signal=1.0, rng=rng)
    out = _anchor_metrics(pd.Timestamp("2026-05-01"), 7, val_df, lr, q50, naive, outcomes, floor=1.0, tied=tied)
    assert out is None


def test_ranker_beats_q50_on_a_planted_signal():
    rng = np.random.default_rng(1)
    val_df, lr, q50, naive, outcomes, tied = _synthetic_anchor(300, signal=1.0, rng=rng)
    out = _anchor_metrics(pd.Timestamp("2026-05-01"), 7, val_df, lr, q50, naive, outcomes, floor=1.0, tied=tied)
    assert out is not None
    assert out["n_tied"] == 300
    assert out["lr_ic"] > out["q50_ic"]  # lr orders the planted signal better
    assert out["ls_spread"] > 0  # top decile outperforms bottom
    assert len(out["pt_records"]) == 300


class TestRetrainPoints:
    def test_biweekly_points_span_the_window(self):
        pts = _retrain_points(SERVED_WINDOW, cadence=14)
        assert pts[0] == date(2026, 4, 18)
        assert all((pts[i + 1] - pts[i]).days == 14 for i in range(len(pts) - 1))
        assert pts[-1] <= date(2026, 6, 8)

    def test_each_point_precedes_the_window_end(self):
        pts = _retrain_points(SERVED_WINDOW, cadence=14)
        assert pts[-1] < SERVED_WINDOW[1]


class TestFrozenAnchors:
    """frozen_anchors keeps only feed-audit-passing, cutover-free dates in the
    window, and logs the set before any scoring. Referee helpers are monkey-
    patched so the logic is exercised without the archive."""

    def _patch(self, monkeypatch, *, audit_ok_for, cutover_for):
        # _feed_profile is only fed to the (patched) audit/cutover fns, so a
        # trivial stand-in is fine.
        monkeypatch.setattr(
            rl, "_feed_profile", lambda anchor, window=3: pd.DataFrame({"day": [anchor], "items": [100]})
        )
        monkeypatch.setattr(rl, "audit_anchor_feed", lambda anchor, profile: (anchor in audit_ok_for, []))
        monkeypatch.setattr(rl, "cutovers_from_counts", lambda counts: [])
        # Return a non-empty dict (truthy) for dates we want dropped for a cutover.
        monkeypatch.setattr(
            rl,
            "cutovers_in_outcome_window",
            lambda anchor, horizons, cutovers: {horizons[0]: [anchor]} if anchor in cutover_for else {},
        )

    def test_keeps_only_clean_anchors_sorted_in_window(self, monkeypatch):
        window = (date(2026, 4, 18), date(2026, 4, 22))  # 5 days
        all_days = [date(2026, 4, d) for d in range(18, 23)]
        # 4-18 fails the feed audit; 4-20 spans a cutover; the rest are clean.
        self._patch(monkeypatch, audit_ok_for=set(all_days) - {date(2026, 4, 18)}, cutover_for={date(2026, 4, 20)})
        out = rl.frozen_anchors(fc=None, horizon=7, window=window)
        assert out == [date(2026, 4, 19), date(2026, 4, 21), date(2026, 4, 22)]

    def test_logs_frozen_set_before_returning(self, monkeypatch, caplog):
        window = (date(2026, 4, 18), date(2026, 4, 19))
        self._patch(monkeypatch, audit_ok_for={date(2026, 4, 18), date(2026, 4, 19)}, cutover_for=set())
        import logging

        with caplog.at_level(logging.INFO):
            out = rl.frozen_anchors(fc=None, horizon=7, window=window)
        assert out == [date(2026, 4, 18), date(2026, 4, 19)]
        assert any("FROZEN anchor set" in r.message for r in caplog.records)

    def test_empty_when_nothing_passes(self, monkeypatch):
        window = (date(2026, 4, 18), date(2026, 4, 20))
        self._patch(monkeypatch, audit_ok_for=set(), cutover_for=set())
        assert rl.frozen_anchors(fc=None, horizon=7, window=window) == []
