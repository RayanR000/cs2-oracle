"""Why is the SERVED band 1.52-1.55x as wide as the same `q_hat`'s calibration band?

Measured 2026-08-13 (`changelog/2026-08-13-band-level-on-audited-anchors.md`): the
calibration line reports a median half-width of 6.41 / 9.53 / 13.66 / 20.61% of mid on
its own OOF records, and `replay_serving.py` reads 9.80 / 14.75 / 21.22 / 31.32% on the
four audited anchors. The ratio is 1.529 / 1.548 / 1.553 / 1.520 -- flat in horizon,
which is what a level defect looks like.

WHY THE RATIO IS A SIGMA RATIO. Both figures are half of `(high - low)` over the band's
own centre, and `conformal.band` sets `high - low = 2 * q_hat * sigma ** beta` in return
space, so with one `q_hat` and `beta = 1` the width ratio is the **sigma** ratio and
nothing else. (The `(1 + mid/100)` denominator in `range_pct` differs from 1 by the
predicted return, median 0.95%.) So this script decomposes exactly two things:

  WHICH items -- the served >=$1 cohort against the pooled calibration cross-section;
  WHEN       -- the same item's sigma on the anchor against its own pooled median.

AND THE RESIDUAL LEG, WHICH IS THE POINT. The conformal score is `|resid| / sigma`.
A sigma that is 1.5x higher at serving costs no coverage at all if `|resid|` is 1.5x
higher too -- the band is wider because the item is genuinely more volatile. Over-coverage
needs sigma to have risen MORE than the residual it normalises. So the reported quantity
is the RATIO OF RATIOS, and a value below 1 is the over-coverage.

READ THIS AS A STAND-IN, NOT AS THE ARTIFACT. Like `measure_conditional_qhat`, the panel
derives its own clip bounds and its own cohort, so no absolute number here is production's
-- only ratios are read, and the residual is the realised return rather than a booster's
error (justified at `attribute_marginal_coverage`'s head: predicted |return| is median
0.95% against half-widths of 10-31%).

Read-only: reads a voted price panel, writes nothing.

    venv/bin/python -m scripts.attribute_band_level --horizons 3,7,14,30
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.forecaster import ItemForecaster  # noqa: E402
from scripts.measure_conditional_qhat import (  # noqa: E402
    default_voted_panel,
    load_panel,
    score_frame,
    sigma_bounds_for_panel,
)

logger = logging.getLogger("attribute_band_level")

# The four anchors that pass `replay_serving.audit_anchor_feed`, i.e. the ones run
# `31657639707` served the 9.80 / 14.75 / 21.22 / 31.32% widths on.
AUDITED_ANCHORS = ["2026-04-22", "2026-05-16", "2026-06-16", "2026-07-06"]
# The served/calibration half-width ratios that run measured, per horizon. Printed
# beside this script's sigma ratio so the two can be compared without a second lookup.
SERVED_WIDTH_RATIO = {3: 1.529, 7: 1.548, 14: 1.553, 30: 1.520}


def _med_ratio(num: pd.Series, den: pd.Series) -> float:
    a, b = float(np.nanmedian(num)), float(np.nanmedian(den))
    return a / b if b else float("nan")


def decompose(frame: pd.DataFrame, anchors) -> dict:
    """Split the served/pooled `sigma` and `|resid|` ratios into WHICH and WHEN.

    *frame* carries `date`, `item_id`, `sigma`, `absr`; *anchors* are the served
    dates. Returns both bases:

    `pooled_*`  the served rows against every row in the panel — cohort AND date
                effects together, which is the quantity a served-vs-calibration
                width ratio actually is.
    `same_*`    each served row against its OWN item's pooled median, so the
                cross-section cancels and only the date effect survives.

    `score_*` is `|resid|` ratio over `sigma` ratio. **That is the ratio the band's
    coverage responds to**, because the conformal score is `|resid| / sigma`: a
    served `sigma` twice the norm costs nothing if the residual doubled too, and
    below 1 means the band is wider than the residual justifies.
    """
    served = frame[frame["date"].isin(set(anchors))]
    if served.empty:
        return {}
    own_sigma = frame.groupby("item_id")["sigma"].median()
    own_absr = frame.groupby("item_id")["absr"].median()
    out = {
        "n_served": int(len(served)),
        "n_pooled": int(len(frame)),
        "pooled_sigma": _med_ratio(served["sigma"], frame["sigma"]),
        "pooled_absr": _med_ratio(served["absr"], frame["absr"]),
        "same_sigma": float(np.nanmedian(
            served["sigma"] / served["item_id"].map(own_sigma))),
        "same_absr": float(np.nanmedian(
            served["absr"] / served["item_id"].map(own_absr))),
    }
    for basis in ("pooled", "same"):
        s = out[f"{basis}_sigma"]
        out[f"score_{basis}"] = out[f"{basis}_absr"] / s if s else float("nan")
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--voted", default=None)
    ap.add_argument("--horizons", default="3,7,14,30")
    ap.add_argument("--anchors", default=",".join(AUDITED_ANCHORS))
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    anchors = [pd.Timestamp(a).date() for a in args.anchors.split(",") if a.strip()]

    panel = load_panel(args.voted or default_voted_panel())
    floor, cap = sigma_bounds_for_panel(panel)
    logger.info("sigma clip [%.4f, %.4f] from this panel's own cross-section", floor, cap)

    # `__init__`, not `__new__`: `prepare_targets` reports through
    # `self.label_voiding`, which only the constructor creates. `db_session=None`
    # is what the sibling instruments pass -- nothing here touches the DB.
    fc = ItemForecaster(db_session=None)

    # Is sigma trending? If the calibration pool spans a calmer era than the anchors,
    # "which rows q_hat saw" is a time defect and not a cohort one -- and unlike the
    # refuted trailing-window class, a secular shift is not a per-date state.
    trend = panel.copy()
    trend["sigma_raw"] = (trend["price_std_60d"] / trend["price"]).clip(floor, cap)
    trend["ym"] = pd.to_datetime(trend["date"]).dt.to_period("Q").astype(str)
    logger.info("\nmedian sigma by quarter (panel, >=$1):")
    for ym, g in trend.groupby("ym"):
        logger.info("  %s  n=%7s  median sigma %.4f", ym, f"{len(g):,}",
                    float(np.nanmedian(g["sigma_raw"])))

    for h in horizons:
        frame = score_frame(fc, panel, h, floor, cap)
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["absr"] = frame["resid"].abs()
        served = frame[frame["date"].isin(anchors)]
        if served.empty:
            logger.warning("h=%s: no panel rows on %s", h, anchors)
            continue

        d = decompose(frame, anchors)
        logger.info(
            "\nh=%-2s  served n=%s of %s pooled | measured served/cal WIDTH ratio %.3f",
            h, f"{d['n_served']:,}", f"{d['n_pooled']:,}",
            SERVED_WIDTH_RATIO.get(h, float("nan")))
        logger.info("  pooled basis (which items AND when):  sigma %.3fx   "
                    "|resid| %.3fx   score |resid|/sigma %.3fx",
                    d["pooled_sigma"], d["pooled_absr"], d["score_pooled"])
        logger.info("  same items (when only):               sigma %.3fx   "
                    "|resid| %.3fx   score |resid|/sigma %.3fx",
                    d["same_sigma"], d["same_absr"], d["score_same"])
        for a in anchors:
            sa = served[served["date"] == a]
            if sa.empty:
                continue
            logger.info("    %s  n=%5s  sigma %.3fx pooled", a, f"{len(sa):,}",
                        _med_ratio(sa["sigma"], frame["sigma"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
