"""C2 lambdarank serving-transfer read — does the CV vs-q50 edge reach serving?

Read-only. Retrains a matched q50 + lambdarank pair at the 14-day production
cadence, scores each frozen served-window anchor date's TIED cross-section
against replay-resolved outcomes, and reports three rulers: within-date rank IC
(primary, edge lr-q50), the decile long-short spread, and a Pesaran-Timmermann
test on the ranker's within-date directional call. Writes nothing.

    venv/bin/python -m scripts.replay_lambdarank --horizon 7

Design: docs/superpowers/specs/2026-08-14-lambdarank-serving-transfer-design.md
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.directional_test import pesaran_timmermann          # noqa: E402
from backtest.longshort import (                                  # noqa: E402
    decile_longshort_by_date, direction_records, net_of_cost)
from models.forecaster import ItemForecaster, embargo_days         # noqa: E402
from scripts.replay_serving import (                              # noqa: E402
    _resolve, _tied_mask, _feed_profile, audit_anchor_feed,
    cutovers_from_counts, cutovers_in_outcome_window, _outcomes)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Fewer tied items than this on an anchor and the within-date correlations are
# noise — the same floor `_within_date_rank_ic_detail` applies per date.
MIN_TIED_ROWS = 20
# Window start / end, and the retrain cadence. Frozen: the served window every
# stored A/B of the last fortnight is read on.
SERVED_WINDOW = (date(2026, 4, 18), date(2026, 6, 8))
RETRAIN_CADENCE_DAYS = 14
# Summed long+short microstructure cost for the net-of-cost decile read: 15%
# Steam fee each way + a mid-tier ~5% spread each way. Tradeability conditioner
# only; NOT a pass/fail bar.
ROUNDTRIP_COST = 2 * (0.15 + 0.05)


def _anchor_metrics(anchor, horizon, val_df, lr_scores, q50_scores,
                    naive_scores, outcomes, floor, tied) -> Optional[dict]:
    """Score one anchor date on its TIED, served (`current >= floor`) cohort.

    Returns per-anchor rank ICs (lr / q50 / naive), the decile long-short
    spread, and the ranker's per-row PT records for cross-date pooling — or
    None when fewer than MIN_TIED_ROWS tied rows resolve. The outcome is the
    smoothed trailing median at `anchor + horizon`, strictly after the anchor
    (`_resolve(..., after=anchor)`), so a row's outcome never reaches back over
    its own anchor quote.
    """
    target = anchor.date() if isinstance(anchor, pd.Timestamp) else anchor
    realised = _resolve(outcomes, target + timedelta(days=int(horizon)),
                        after=target)
    frame = val_df.copy()
    frame["lr"] = np.asarray(lr_scores, dtype=float)
    frame["q50"] = np.asarray(q50_scores, dtype=float)
    frame["naive"] = np.asarray(naive_scores, dtype=float)
    frame = frame.merge(realised, on="item_id", how="inner")
    frame = frame[frame["current"] >= floor]
    is_tied = frame["item_id"].map(tied).eq(True).to_numpy()
    frame = frame[is_tied]
    frame = frame[np.isfinite(frame["realised"]) & (frame["current"] > 0)]
    if len(frame) < MIN_TIED_ROWS:
        return None

    ret = (frame["realised"] / frame["current"] - 1.0).to_numpy()
    dates = frame["date"].to_numpy()
    # One anchor is one date, so min_rows is the tied-count floor above and the
    # detail reader returns a single-date IC (or None if degenerate).
    def _ic(pred):
        return ItemForecaster._within_date_rank_ic_detail(
            pred, ret, dates, min_rows=MIN_TIED_ROWS)[0]

    ls = decile_longshort_by_date(frame["lr"].to_numpy(), ret, dates,
                                  min_rows=MIN_TIED_ROWS)
    return {
        "anchor": str(target),
        "n_tied": int(len(frame)),
        "lr_ic": _ic(frame["lr"].to_numpy()),
        "q50_ic": _ic(frame["q50"].to_numpy()),
        "naive_ic": _ic(frame["naive"].to_numpy()),
        "ls_spread": ls["mean"],
        "pt_records": direction_records(frame["lr"].to_numpy(), ret, dates,
                                        min_rows=MIN_TIED_ROWS),
    }


def _retrain_points(window, cadence: int = RETRAIN_CADENCE_DAYS) -> list:
    """Retrain-point dates at `cadence` days from the window start, each
    strictly before the window end (a point on the last day scores nothing)."""
    lo, hi = window
    pts, d = [], lo
    while d < hi:
        pts.append(d)
        d = d + timedelta(days=cadence)
    return pts


def frozen_anchors(fc, horizon, window=SERVED_WINDOW) -> list:
    """Every date in `window` that passes the feed audit AND has no collector
    cutover inside its `horizon` outcome window. Enumerated and LOGGED before
    any rank IC is read — the pre-registration the spec requires.
    """
    lo, hi = window
    surviving = []
    d = lo
    while d <= hi:
        profile = _feed_profile(d)
        ok, _ = audit_anchor_feed(d, profile)
        if ok:
            span = _feed_profile(d, window=horizon)
            cutovers = cutovers_from_counts(
                span.set_index(pd.to_datetime(span["day"]).dt.date)["items"])
            if not cutovers_in_outcome_window(d, [horizon], cutovers):
                surviving.append(d)
        d = d + timedelta(days=1)
    logger.info("FROZEN anchor set (h=%s): %d dates — %s", horizon,
                len(surviving), ", ".join(a.isoformat() for a in surviving))
    return surviving


def _master_frame(fc, horizon, cutoff, floor):
    """Engineered + targeted frame bounded at `cutoff`, for TRAINING one fold.

    REPLAY_ANCHOR bounds the archive read at the cutoff (no look-ahead). The
    universe is floored at the served >= $1 cohort and the feature set is
    production's allowlisted selection, so the ranker and q50 train on the same
    columns production serves. Rows are capped to the most recent
    CV_MAX_TRAIN_ROWS to hold the 30-minute cap; the embargo removes the label's
    13-day support that would otherwise overlap the val anchors.
    """
    os.environ["REPLAY_ANCHOR"] = cutoff.isoformat()
    try:
        price_df = fc.fetch_price_history(days_back=1100, backfilled_only=True)
        price_df = fc._filter_by_median_price(price_df, floor)
        events_df = fc.fetch_events()
        feat = fc.engineer_features(price_df, events_df)
        # Production's feature selection: allowlist + correlation prune → ~33 cols.
        fc.feature_cols = fc._select_feature_cols(
            feat, fc.HORIZONS, fc._active_shelved_features())
        fc._reduce_feature_cols(feat)
        tdf = fc.prepare_targets(feat, horizon)
    finally:
        os.environ.pop("REPLAY_ANCHOR", None)
    embargo = embargo_days(horizon)
    keep = pd.to_datetime(tdf["date"]) <= (
        pd.Timestamp(cutoff) - pd.Timedelta(days=embargo))
    train_df = tdf[keep].sort_values("date")
    cap = fc.CV_MAX_TRAIN_ROWS
    if len(train_df) > cap:
        train_df = train_df.tail(cap)
    return train_df.reset_index(drop=True), feat


def _val_frame(feat, anchor):
    """The served cross-section AS OF `anchor`: the engineered rows on that day.

    Taken from `engineer_features` output (causal), not `prepare_targets`, so an
    item with no resolved forward label at the anchor is still scored — its
    outcome is resolved separately from the full archive.
    """
    day = pd.Timestamp(anchor)
    v = feat[pd.to_datetime(feat["date"]) == day].copy()
    # `_anchor_metrics` reads the anchor quote as `current` (matching
    # `replay_serving.py`'s served-row contract); `engineer_features` names the
    # same column `price`.
    v["current"] = v["price"]
    return v.reset_index(drop=True)


def main() -> int:
    if "--horizon" not in sys.argv:
        logger.error("Pass --horizon 3|7|14|30 (shard by horizon for the cap).")
        return 2
    horizon = int(sys.argv[sys.argv.index("--horizon") + 1])

    from database import SessionLocal
    from api.serving_policy import MIN_SERVED_PRICE_USD
    db = SessionLocal()
    try:
        fc = ItemForecaster(db_session=db, prune_failed_groups=False)
        anchors = frozen_anchors(fc, horizon)
        if not anchors:
            logger.error("No clean anchors in %s at h=%s.", SERVED_WINDOW,
                         horizon)
            return 1
        points = _retrain_points(SERVED_WINDOW)
        # Outcomes over the whole window+horizon, resolved once from the full
        # (unbounded) archive — REPLAY_ANCHOR is cleared inside `_outcomes`.
        # `_outcomes`'s fetch span is measured back from TODAY, not forward
        # from its `anchor` arg (see replay_serving.py:940, which calls it
        # per-anchor), so passing the window END here would miss outcomes for
        # early-window anchors; the window START gives every anchor's horizon
        # enough trailing coverage since "today" is well past the window end.
        outcomes = _outcomes(fc, SERVED_WINDOW[0], [horizon])
        floor = fc._artifact_min_median_price or MIN_SERVED_PRICE_USD
        per_quantile_params = {0.5: {}}  # tuned HP not needed for the sign bar

        # The VAL feature panel, built ONCE over the full window (unbounded up
        # to today, floored). engineer_features' lags are backward-looking, so a
        # row's features as of its anchor are identical whether the frame ends
        # at that date or later — one panel supplies every anchor's cross-section.
        # Bounding val features at the retrain cutoff (the fixed bug) left every
        # anchor after the cutoff with no rows, so only the 4 cutoff dates scored.
        # Outcomes are resolved separately and train_df stays cutoff-bounded, so
        # this adds no look-ahead.
        os.environ.pop("REPLAY_ANCHOR", None)
        val_price = fc._filter_by_median_price(
            fc.fetch_price_history(days_back=1100, backfilled_only=True), floor)
        val_feat = fc.engineer_features(val_price, fc.fetch_events())

        rows = []
        all_pt = []
        for i, cutoff in enumerate(points):
            block_hi = (points[i + 1] if i + 1 < len(points)
                        else SERVED_WINDOW[1] + timedelta(days=1))
            block = [a for a in anchors if cutoff <= a < block_hi]
            if not block:
                continue
            # Anchors in this block with a large-enough cross-section, scored on
            # ONE trained pair: train_df is identical for every anchor in the
            # block and both boosters are deterministic, so a per-anchor retrain
            # is redundant. Concatenate the anchors' val rows, train once, then
            # split the scores back by anchor (concat preserves row order).
            block_vals = [(a, _val_frame(val_feat, a)) for a in block]
            block_vals = [(a, v) for a, v in block_vals
                          if len(v) >= MIN_TIED_ROWS]
            if not block_vals:
                continue
            train_df, _ = _master_frame(fc, horizon, cutoff, floor)
            logger.info("retrain @ %s: %d train rows, %d/%d anchors scorable",
                        cutoff, len(train_df), len(block_vals), len(block))
            val_all = pd.concat([v for _, v in block_vals], ignore_index=True)
            lr_all = np.asarray(fc._lambdarank_fold_scores(
                train_df, val_all, horizon, per_quantile_params))
            q50_all = np.asarray(fc._fold_q50_scores(
                train_df, val_all, horizon, per_quantile_params))
            naive_all = (-val_all["return_1d"].to_numpy(dtype=float)
                         if "return_1d" in val_all.columns
                         else np.zeros(len(val_all)))
            off = 0
            for a, v in block_vals:
                n = len(v)
                sl = slice(off, off + n)
                off += n
                tied = _tied_mask(outcomes, a)
                m = _anchor_metrics(pd.Timestamp(a), horizon, v,
                                    lr_all[sl], q50_all[sl], naive_all[sl],
                                    outcomes, floor, tied)
                if m is not None:
                    rows.append(m)
                    all_pt.extend(m["pt_records"])

        if not rows:
            logger.error("No anchor produced >= %d tied resolved rows.",
                         MIN_TIED_ROWS)
            return 1

        def _mean(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return float(np.mean(vals)) if vals else None

        lr_ic, q50_ic, naive_ic = _mean("lr_ic"), _mean("q50_ic"), _mean("naive_ic")
        edge_q50 = (None if lr_ic is None or q50_ic is None
                    else lr_ic - q50_ic)
        edge_naive = (None if lr_ic is None or naive_ic is None
                      else lr_ic - naive_ic)
        ls = _mean("ls_spread")
        pt = pesaran_timmermann(all_pt, min_dates=len(rows))

        print(f"\nC2 SERVING-TRANSFER @ h={horizon}   "
              f"(tied cohort, {len(rows)} anchor-dates, floor >= ${floor:g})")
        print(f"  lr rank IC      {lr_ic}")
        print(f"  q50 rank IC     {q50_ic}")
        print(f"  naive rank IC   {naive_ic}")
        print(f"  EDGE vs q50     {edge_q50}   <-- primary bar: > 0 confirms")
        print(f"  edge vs naive   {edge_naive}")
        print(f"  decile L/S      gross {ls}   "
              f"net {None if ls is None else net_of_cost(ls, ROUNDTRIP_COST)}")
        print(f"  PT (lr call)    excess_pp={pt['pt_excess_pp']} "
              f"t={pt['pt_t_stat']} verdict={pt['pt_verdict']} "
              f"n_dates={pt['pt_n_dates']}")
        print("\nEDGE vs q50 is the pre-registered pass/fail. Decile net and "
              "PT are descriptive; DA/MAE are undefined for a ranker.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
