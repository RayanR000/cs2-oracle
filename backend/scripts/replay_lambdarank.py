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
from models.forecaster import ItemForecaster                      # noqa: E402
from scripts.replay_serving import _resolve, _tied_mask           # noqa: E402

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
