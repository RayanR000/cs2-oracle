"""Score the SERVING path against outcomes the archive already holds.

Walk-forward CV scores fold predictions. Four transforms run only inside
`predict()` -- the prior-day blend, the tier bias, `_recenter_on_direction`,
and the conformal band -- so a CV rank IC says nothing about what production
actually serves. The only way to measure them was to publish a forecast and
wait `horizon` days for it to mature.

This replays instead. `REPLAY_ANCHOR` rewinds the serving clock, the archive
read is bounded at the anchor, and the outcome for `anchor + horizon` is
already on disk. Nothing is written to the database.

    REPLAY_ANCHOR=2026-06-01 venv/bin/python -m scripts.replay_serving

Add `CROSS_SECTIONAL_RANK=1` (and any other instrument flag) to score an arm.
Run the control at the same anchor and compare -- never against a stored
number, per docs/changelog/2026-08-10-instrument-panel-first-read.md.

**What this does not do.** It does not retrain: it serves whatever artifact is
in the model directory, so the artifact's flags must match the flags you set or
`predict` will refuse (or worse, agree). And a replay measures serving
arithmetic on historical features, not a historical run -- anything read from
live state carries today's value unless it was pinned.
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

from database import SessionLocal                      # noqa: E402
from db.archive import prices_relation                 # noqa: E402
from models import conformal                           # noqa: E402
from models.forecaster import ItemForecaster           # noqa: E402
from api.serving_policy import MIN_SERVED_PRICE_USD    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# The resolver's tolerance, matching the backtest: an outcome is the nearest
# observation within this many days of the target, not the exact day. The
# archive has whole missing days (docs/changelog aggregator-archive-day-gaps),
# so an exact-day join silently drops items rather than scoring them.
OUTCOME_TOLERANCE_DAYS = 3

# The referee's denominator, and these are LITERALS on purpose.
#
# `backtest.price_resolution` exports `SMOOTH_WINDOW` and `MAX_WINDOW_SPAN_DAYS`
# and `models.forecaster` re-exports them, so an experiment that changes what
# price serving quotes from -- arm B of
# `docs/superpowers/plans/2026-08-11-serving-anchor-freshness.md` moves exactly
# those two constants -- would move the yardstick along with the thing being
# measured. Importing them here would make the referee follow the arm.
#
# They must equal the shipped constants TODAY, which
# `test_the_pinned_denominator_reproduces_the_shipped_one_today` asserts by
# comparing against `_smoothed_anchor_prices` itself. If that test fails, the
# shipped definition moved and this pin is now a different basis: every stored
# replay number is incomparable to the next one until it is reconciled
# deliberately, which is the point of finding out from a red test.
PINNED_SMOOTH_WINDOW = 3
PINNED_MAX_SPAN_DAYS = 7


def _outcomes(fc: ItemForecaster, anchor: date, horizons):
    """Realised prices on the SAME basis production served from.

    Not a hand-rolled archive read. `backend/AGENTS.md` invariant 2: any loader
    that globs the archive must apply `archive_universe_sql_filter()`, and a
    plain `median(mean_price)` also lets BUFF's bid and Steam's trailing-window
    means vote -- both excluded from the consensus `predict` quoted its
    `current_price` from. Scoring one basis against the other puts a systematic
    wedge in every return, which is what the first version of this script did.

    `REPLAY_ANCHOR` is cleared for this call on purpose: the outcome is exactly
    the post-anchor data the replay was forbidden to see while forecasting.
    """
    saved = os.environ.pop("REPLAY_ANCHOR", None)
    try:
        span = (date.today() - anchor).days + max(horizons) + 10
        voted = fc.fetch_price_history(days_back=span, backfilled_only=True)
    finally:
        if saved is not None:
            os.environ["REPLAY_ANCHOR"] = saved
    out = voted[["item_id", "date", "price"]].copy()
    out["day"] = pd.to_datetime(out["date"])
    return out[["item_id", "day", "price"]]


def _resolve(outcomes: pd.DataFrame, target: date, tolerance=OUTCOME_TOLERANCE_DAYS,
             after: date | None = None):
    """Smoothed price at `target` per item -- the same statistic the anchor is.

    `predict` converts return-space output to dollars against
    `_smoothed_anchor_prices`: the MEDIAN of the observations within a bounded
    window, not the nearest single quote. Resolving the outcome as a single day
    would divide a smoothed denominator into an unsmoothed numerator and book
    the difference as forecast error. The backtest's resolver smooths for the
    same reason.
    """
    t = pd.Timestamp(target)
    # TRAILING. `_smoothed_anchor_prices` takes the most recent observations
    # within a span BEFORE the anchor, and a centred window here would resolve
    # h=3 over [anchor, anchor+6] -- pulling the anchor's own price into its
    # own outcome and shrinking every return toward zero.
    lo = t - pd.Timedelta(days=int(tolerance) + 1)
    if after is not None:
        # STRICTLY after the anchor. At h=3 a 4-day trailing window reaches
        # back over the anchor itself, so an item's outcome contains the very
        # quote its own return is measured from: a noisy-high anchor day
        # inflates the realised median AND drives return_1d, manufacturing
        # correlation. It showed up as naive rank IC negative at 17 of 18
        # anchor-horizon cells against a CV baseline of +0.19, with |IC|
        # falling monotonically in horizon -- the overlap's own signature.
        lo = max(lo, pd.Timestamp(after))
    window = outcomes[(outcomes["day"] > lo) & (outcomes["day"] <= t)]
    if window.empty:
        return pd.DataFrame(columns=["item_id", "realised"])
    med = window.groupby("item_id", as_index=False)["price"].median()
    return med.rename(columns={"price": "realised"})


def _naive_baseline(fc: ItemForecaster, outcomes: pd.DataFrame, anchor: date):
    """`-return_1d` ON THE SERVED BASIS -- the one baseline that beats the
    model on rank IC at every horizon in CV.

    The realised return divides by `predict`'s span-bounded median, so a
    baseline built from the last two RAW quotes is one basis scored against
    another: a fresh jump moves `return_1d` and barely moves the median, and
    the outcome window then carries the jump. Measured at anchor 2026-06-01,
    the raw build read -0.6187 / -0.3815 / -0.3249 / -0.2610 against a CV
    baseline of about +0.19 -- sign-inverted and decaying monotonically in
    horizon, which is the wedge's signature, not market momentum. The same
    baseline on this basis reads -0.0339 / +0.0700 / +0.0797 / +0.0721. Raw and
    smoothed differ for 29% of items at that anchor.
    """
    hist = outcomes.rename(columns={"day": "date"})
    t_now = pd.Timestamp(anchor)
    t_prev = t_now - pd.Timedelta(days=1)
    # Truncate before each call, don't lean on the anchor argument: an item
    # with nothing inside the span window falls back to `last()` over the WHOLE
    # frame, which here reaches past the anchor into the outcome.
    s_now = fc._smoothed_anchor_prices(hist[hist["date"] <= t_now], t_now)
    s_prev = fc._smoothed_anchor_prices(hist[hist["date"] <= t_prev], t_prev)
    naive = pd.DataFrame({"item_id": list(s_now)})
    naive["naive"] = [
        (-(s_now[i] / s_prev[i] - 1.0)
         if s_prev.get(i) and s_prev[i] > 0 else np.nan)
        for i in naive["item_id"]
    ]
    return naive[np.isfinite(naive["naive"])].reset_index(drop=True)


def _exact_day(outcomes: pd.DataFrame, day: date) -> pd.DataFrame:
    """The price observed on exactly `day`, or nothing.

    `prepare_targets` joins the target on `date + horizon` exactly -- no
    tolerance, no smoothing -- so an item with no observation that day gets no
    label and leaves the fold. Reproducing CV's basis means reproducing that
    dropout, not filling it in: the tolerance is what makes the two measurements
    different, so silently applying it here would erase the thing being
    measured.
    """
    d = outcomes[outcomes["day"] == pd.Timestamp(day)]
    return (d.groupby("item_id", as_index=False)["price"].median()
             .rename(columns={"price": "px"}))


def _pin_matches_production() -> bool:
    """Does the frozen referee still describe what production quotes?

    `MAX_WINDOW_SPAN_DAYS` is `collectors.pipeline.FALLBACK_MAX_AGE_DAYS`, which
    reads the environment at import — so production's smoothing span can move
    without a commit in this repo. The pin must not follow it, or two arms stop
    being comparable; but a replay scored against a denominator production no
    longer uses has to say so out loud, or its numbers look citable and are not.

    Read at call time, never captured at import: the whole point is to notice a
    value that was set somewhere else.
    """
    from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS, SMOOTH_WINDOW
    return (SMOOTH_WINDOW == PINNED_SMOOTH_WINDOW
            and MAX_WINDOW_SPAN_DAYS == PINNED_MAX_SPAN_DAYS)


def _pinned_anchor(outcomes: pd.DataFrame, anchor: date) -> pd.Series:
    """`_smoothed_anchor_prices`, frozen: the median of the most recent
    PINNED_SMOOTH_WINDOW observations within PINNED_MAX_SPAN_DAYS of `anchor`.

    Deliberately NOT a call to `fc._smoothed_anchor_prices`. That method is what
    the freshness arms change; scoring against it would mean an arm that quotes
    a different price is also scored against a different number, and the
    comparison would report the redefinition as a result. Same trap the label
    arm fell into (`2026-08-11-smoothed-anchor-label-measured.md`), one layer
    down.

    Keeps the shipped fallback: an item with nothing inside the window keeps its
    latest observation, so the referee never manufactures a price and never
    silently drops an item from one arm's cohort but not the other's.
    """
    at = pd.Timestamp(anchor)
    hist = outcomes[outcomes["day"] <= at].sort_values(["item_id", "day"])
    if hist.empty:
        return pd.Series(dtype=float, name="d_fixed")

    # Explicit unit: pd.Timedelta(days=<int>) emits a NumPy generic-unit
    # DeprecationWarning, which the rest of the codebase already avoids.
    span = pd.to_timedelta(PINNED_MAX_SPAN_DAYS, unit="D")
    in_window = hist[hist["day"] >= at - span]
    smoothed = (in_window.groupby("item_id").tail(PINNED_SMOOTH_WINDOW)
                .groupby("item_id")["price"].median())
    latest = hist.groupby("item_id")["price"].last()
    pinned = pd.Series(latest.to_dict() | smoothed.to_dict(), name="d_fixed")
    pinned.index.name = "item_id"
    return pinned


def _pinned_rank_ic(frame: pd.DataFrame, pinned: pd.Series) -> float:
    """Rank IC with BOTH legs on the pinned denominator.

    The headline row divides the prediction and the outcome by the served
    `current_price` (`frame["current"]`), which is the right thing for
    "what did production experience" and the wrong thing for comparing two arms
    that quote different prices -- there the label and the prediction move
    together and the difference is not readable. Here the arm supplies only the
    dollar mid; the denominator is the same for both arms.
    """
    d = frame["item_id"].map(pinned).to_numpy(dtype=float)
    mid = frame["mid"].to_numpy(dtype=float)
    realised = frame["realised"].to_numpy(dtype=float)
    ok = np.isfinite(d) & (d > 0) & np.isfinite(mid) & np.isfinite(realised)
    if ok.sum() < 3:
        return float("nan")
    return _rank_ic(mid[ok] / d[ok] - 1.0, realised[ok] / d[ok] - 1.0)


def _rel_abs_error(pred, realised) -> tuple[float, float]:
    """Median and p90 of `|pred - realised| / realised`. BASIS-FREE.

    The metric the freshness arms are read on. Rank IC cannot referee them:
    `main()` divides the prediction and the outcome by the served
    `current_price`, so an arm that quotes a different price moves the label and
    the prediction together and the comparison measures nothing
    (`2026-08-11-smoothed-anchor-label-measured.md`, one layer down). Here the
    denominator is the REALISED price, which no arm touches, so two arms are
    expressed in the same units.

    p90 beside the median because a quoting change can leave the centre alone
    and move the tail: `_smoothed_anchor_prices` records serving and the
    backtest resolver diverging by a median 3.6% and a p90 35%.
    """
    pred = np.asarray(pred, dtype=float)
    realised = np.asarray(realised, dtype=float)
    # `realised > 0` is not defensive: an item whose outcome did not resolve
    # divides to inf, and one inf takes the p90 of the whole cohort with it.
    ok = np.isfinite(pred) & np.isfinite(realised) & (realised > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    err = np.abs(pred[ok] - realised[ok]) / realised[ok]
    return float(np.median(err)), float(np.quantile(err, 0.90))


def _tied_mask(outcomes: pd.DataFrame, anchor: date) -> pd.Series:
    """Per item: does the anchor's raw quote equal its own local median?

    ONE definition, shared by the basis sweep and the dollar table. The gate of
    the freshness experiment is dollar error on the *deviating* cohort, and the
    durable served signal is the *tied* cell — so two copies of this `isclose`
    could drift and make the two tables describe different populations while
    printing the same words.

    `np.isclose(nan, nan)` is False, deliberately: an item with no observation
    on the anchor day is not tied, it is unknown, and defaulting it into the
    clean cohort would contaminate the one number that has replicated.
    """
    raw = _exact_day(outcomes, anchor).set_index("item_id")["px"]
    pinned = _pinned_anchor(outcomes, anchor)
    idx = raw.index.union(pinned.index)
    mask = pd.Series(
        np.isclose(raw.reindex(idx).to_numpy(dtype=float),
                   pinned.reindex(idx).to_numpy(dtype=float),
                   rtol=0, atol=1e-9),
        index=idx, name="anchor_is_tied")
    mask.index.name = "item_id"
    return mask


def _dollar_error_rows(frame: pd.DataFrame, pinned: pd.Series,
                       tied: pd.Series) -> list[dict]:
    """The dollar-error table for one horizon: pooled, tied, deviating.

    Three predictions per subset, all scored against the same realised price:

      `model` — the served mid, with its p90;
      `quote` — NO CHANGE, i.e. the served `current_price` itself. This is the
                reference a freshness arm is really claiming to beat, because
                the claim is "the price we publish is closer to what happens";
      `naive` — `-return_1d`, the baseline the model loses to on rank IC at all
                four horizons, converted to dollars against the PINNED anchor.

    The naive conversion must not use `frame["current"]`. `naive` is a return,
    so it needs a base; taking the arm's own quote would move the baseline
    whenever the arm moved, and "the model beat naive" would shift for a reason
    that is not the model.

    `n` and `n_naive` are reported separately because `MIN_SERVED_PRICE_USD`
    applies to `current_price`: the two arms do not score exactly the same item
    set at the floor (hazard 4 of the plan), and the baseline drops an item with
    no prior observation.
    """
    # `.eq(True)`, not `.fillna(False).astype(bool)`: an item the mask never saw
    # maps to NaN, which makes the column object dtype, and the fillna route
    # then emits a downcasting FutureWarning on exactly that case.
    is_tied = frame["item_id"].map(tied).eq(True).to_numpy()
    d = frame["item_id"].map(pinned).to_numpy(dtype=float)
    naive_mid = np.where(np.isfinite(d) & (d > 0),
                         d * (1.0 + frame["naive"].to_numpy(dtype=float)),
                         np.nan)
    mid = frame["mid"].to_numpy(dtype=float)
    current = frame["current"].to_numpy(dtype=float)
    realised = frame["realised"].to_numpy(dtype=float)
    scorable = np.isfinite(realised) & (realised > 0)

    rows = []
    for subset, mask in (("pooled", np.ones(len(frame), dtype=bool)),
                         ("tied", is_tied),
                         ("deviating", ~is_tied)):
        med, p90 = _rel_abs_error(mid[mask], realised[mask])
        quote, _ = _rel_abs_error(current[mask], realised[mask])
        naive, _ = _rel_abs_error(naive_mid[mask], realised[mask])
        rows.append({
            "subset": subset,
            "n": int((mask & scorable & np.isfinite(mid)).sum()),
            "model": med, "p90": p90, "quote": quote, "naive": naive,
            "n_naive": int((mask & scorable & np.isfinite(naive_mid)).sum()),
        })
    return rows


def _served_rows(served: pd.DataFrame, h: int) -> pd.DataFrame:
    """The served mid AND band for one horizon, as a frame.

    `low`/`high` are carried because the band is the other half of what
    `predict` publishes, and its coverage is the only falsifiable read on q_hat:
    coverage measured against the residuals q_hat was fitted on is >= nominal by
    construction, which is why an 80%-labelled band could serve 34.6-61.8%
    unnoticed (`2026-08-10-served-classifier-scored.md` §4).

    A row whose band is missing keeps its mid: it still scores DA and rank IC,
    and dropping it here would shrink every other table in this script.
    """
    def _f(v) -> float:
        return float(v) if v is not None else float("nan")

    rows = []
    for _, r in served.iterrows():
        f = r["forecasts"].get(h)
        if not f or f.get("mid") is None or not r.get("current_price"):
            continue
        rows.append({"item_id": r["item_id"],
                     "current": float(r["current_price"]),
                     "mid": float(f["mid"]),
                     "low": _f(f.get("low")),
                     "high": _f(f.get("high"))})
    return pd.DataFrame(rows)


def _coverage_row(frame: pd.DataFrame) -> dict:
    """Share of realised prices inside the served band, and which side missed.

    `low <= realised <= high`, inclusive, so this is the same predicate the
    backtest's `in_interval` uses — two coverage figures that disagree on the
    boundary are not the same measurement.

    The below/above split is the diagnostic. Split conformal misses
    symmetrically by construction; a centre the serving path displaced after
    calibration misses with a sign, because `_recenter_on_direction` moves the
    mid without touching either half-width.

    `halfw` is the width itself, and it is here because coverage alone cannot
    read a `SIGMA_EXPONENT` arm: `q_hat` is in different units on either side of
    that flag (~5.5x apart, see `models/conformal.py`), so the only quantities
    two arms may be differenced on are coverage and width. It is taken against
    the band's own midpoint rather than the served quote so that this function
    keeps needing only `low`/`high`/`realised`; the midpoint is arm-invariant,
    so the ratio across arms is the half-width ratio.
    """
    lo = frame["low"].to_numpy(dtype=float)
    hi = frame["high"].to_numpy(dtype=float)
    real = frame["realised"].to_numpy(dtype=float)
    ok = (np.isfinite(lo) & np.isfinite(hi) & np.isfinite(real)
          & (real > 0))
    n = int(ok.sum())
    if n == 0:
        nan = float("nan")
        return {"n": 0, "cov": nan, "below": nan, "above": nan, "halfw": nan}
    centre = (hi[ok] + lo[ok]) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_half = np.where(centre > 0, (hi[ok] - lo[ok]) / 2.0 / centre,
                            np.nan)
    return {
        "n": n,
        "cov": float(((real[ok] >= lo[ok]) & (real[ok] <= hi[ok])).mean()),
        "below": float((real[ok] < lo[ok]).mean()),
        "above": float((real[ok] > hi[ok]).mean()),
        # nanmedian of an all-NaN slice warns and returns NaN, which is the
        # right answer here and must not take out the table.
        "halfw": (float("nan") if not np.isfinite(rel_half).any()
                  else float(np.nanmedian(rel_half))),
    }


COVERAGE_HEADER = (f"{'h':>4} {'n':>7} {'cov%':>8} {'target%':>8} "
                   f"{'miss<low':>9} {'miss>high':>10} {'halfw%':>8}")


def _coverage_line(h: int, row: dict) -> str:
    """One row of the coverage table. A function for the same reason
    `_dollar_line` is one: a %-formatted NaN raising would take out the run at
    the point where the table prints."""
    # `.get`, not `row['halfw']`: this formats dicts that callers built before
    # the column existed, and a KeyError here would take out the whole table.
    halfw = row.get("halfw", float("nan"))
    return (f"{h:>4} {row['n']:>7} {100 * row['cov']:>8.2f} "
            f"{100 * conformal.NOMINAL_COVERAGE:>8.2f} "
            f"{100 * row['below']:>9.2f} {100 * row['above']:>10.2f} "
            f"{100 * halfw:>8.2f}")


DOLLAR_HEADER = (f"{'h':>4} {'subset':>10} {'n':>7} {'model':>8} {'p90':>8} "
                 f"{'quote':>8} {'naive':>8} {'nNaive':>7}")


def _dollar_line(h: int, row: dict) -> str:
    """One row of the dollar table, percentages.

    A function rather than an f-string inside `main()` so the NaN case is
    covered: a cohort under three items scores NaN, and a `%`-formatted NaN
    raising would take out an hour of CI at the point where the table prints.
    """
    return (f"{h:>4} {row['subset']:>10} {row['n']:>7} "
            f"{100 * row['model']:>8.2f} {100 * row['p90']:>8.2f} "
            f"{100 * row['quote']:>8.2f} {100 * row['naive']:>8.2f} "
            f"{row['n_naive']:>7}")


def _basis_frame(fc: ItemForecaster, outcomes: pd.DataFrame, anchor: date,
                 horizon: int) -> pd.DataFrame:
    """The four label bases, per item, on one anchor.

    The serving replay and CV disagree by 0.1-0.2 rank IC on the same artifact
    (`2026-08-11-serving-transforms-are-not-the-gap.md`), and the serving
    transforms have been ruled out. Two axes remain between the two labels:

      denominator -- CV divides by the RAW price at the anchor, `predict`
                     quotes and the replay divides by the SMOOTHED anchor;
      numerator   -- CV takes the RAW price on exactly `anchor + h`, the replay
                     takes a trailing MEDIAN over (anchor, anchor + h].

    Crossing them gives four bases. `served` and `cv` are the two real ones;
    the mixed pair exists so a difference can be attributed to one axis rather
    than to "the basis" as a lump.
    """
    # PINNED, not `fc._smoothed_anchor_prices`. `anchor_smooth` defines the
    # tied/deviating split that every 2026-08-11 result rests on, so an arm that
    # changed the serving anchor would change which items count as tied -- and
    # the split would stop being a constant across arms. `fc` is still taken so
    # callers do not have to change, and is deliberately not consulted here.
    smoothed = _pinned_anchor(outcomes, anchor).rename("anchor_smooth")

    raw_anchor = _exact_day(outcomes, anchor).rename(columns={"px": "anchor_raw"})
    # The day before the anchor. NOT a clean denominator -- it is `return_1d`'s
    # own denominator, so it carries the same shared-quote channel with the sign
    # flipped. Included because it bounds the effect from the other side.
    lag_anchor = _exact_day(outcomes, anchor - timedelta(days=1)) \
        .rename(columns={"px": "anchor_lag"})
    raw_target = _exact_day(outcomes, anchor + timedelta(days=horizon)) \
        .rename(columns={"px": "out_raw"})
    med_target = _resolve(outcomes, anchor + timedelta(days=horizon), after=anchor) \
        .rename(columns={"realised": "out_med"})

    f = (raw_anchor.merge(smoothed.reset_index(), on="item_id", how="outer")
                   .merge(lag_anchor, on="item_id", how="outer")
                   .merge(raw_target, on="item_id", how="outer")
                   .merge(med_target, on="item_id", how="outer"))
    # The quote at the anchor sits in the label's denominator AND in the
    # features. Where it equals the local median there is no deviation for the
    # model to read, so this flag splits the cross-section into the half where
    # the hypothesised channel can operate and the half where it cannot.
    # `_tied_mask` and not an inline `isclose`: the dollar table reads the same
    # split, and two copies would drift.
    f = f.merge(_tied_mask(outcomes, anchor).reset_index(), on="item_id",
                how="left")
    f["anchor_is_tied"] = f["anchor_is_tied"].eq(True)
    for name, num, den in (("served", "out_med", "anchor_smooth"),
                           ("cv", "out_raw", "anchor_raw"),
                           ("num_only", "out_raw", "anchor_smooth"),
                           ("den_only", "out_med", "anchor_raw"),
                           ("cv_lag", "out_raw", "anchor_lag")):
        f[name] = np.where(f[den].to_numpy(dtype=float) > 0,
                           f[num].to_numpy(dtype=float)
                           / f[den].to_numpy(dtype=float) - 1.0, np.nan)
    return f


def _rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman across the cross-section. One date, so no averaging."""
    if len(pred) < 3:
        return float("nan")
    return float(pd.Series(pred).corr(pd.Series(actual), method="spearman"))


def _requested_horizons(argv) -> Optional[set]:
    """`--horizons 3,7` restricts which rows are scored.

    The reason is `model-diagnostics.yml`: its matrix trains ONE horizon per job
    beside boosters restored from the production cache, and a feature transform
    like `CROSS_SECTIONAL_RANK` is applied to the whole frame rather than per
    horizon. So in job `h` the other three horizons' boosters are fitted on
    untransformed features and fed transformed ones -- their rows are not a
    measurement of anything. Scoring them anyway would publish three garbage
    rows beside one real one, all formatted identically.
    """
    if "--horizons" not in argv:
        return None
    raw = argv[argv.index("--horizons") + 1]
    return {int(h) for h in raw.split(",") if h.strip()}


def main() -> int:
    want = _requested_horizons(sys.argv)
    anchor = ItemForecaster.replay_anchor()
    if anchor is None:
        logger.error("REPLAY_ANCHOR is not set. Refusing to run: without it "
                     "this would 'replay' today and score a forecast whose "
                     "outcome does not exist yet.")
        return 2

    db = SessionLocal()
    try:
        fc = ItemForecaster(db_session=db, prune_failed_groups=False)
        if not fc.load_models():
            logger.error("No usable model artifact. Train one first — the "
                         "replay serves an artifact, it does not build one.")
            return 2

        logger.info(f"Replaying the serving path at anchor {anchor}")
        logger.info(f"  artifact: xs_rank={fc._artifact_xs_rank} "
                    f"floor={fc._artifact_min_median_price} "
                    f"skipped={fc._artifact_xs_rank_skipped}")

        served = fc.predict()
        if served.empty:
            logger.error("predict() returned nothing at this anchor.")
            return 1

        horizons = sorted({h for f in served["forecasts"] for h in f})
        if want is not None:
            missing = want - set(horizons)
            if missing:
                logger.error(f"--horizons asked for {sorted(missing)}, which "
                             f"this artifact does not serve ({horizons}).")
                return 2
            horizons = sorted(want)
            logger.info(f"  scoring only {horizons} -- the rest of this "
                        f"artifact's boosters were not trained under these "
                        f"flags, so their rows would not be a measurement.")
        outcomes = _outcomes(fc, anchor, horizons)
        logger.info(f"  {len(served):,} served rows, "
                    f"{outcomes['item_id'].nunique():,} items in the outcome window")

        naive = _naive_baseline(fc, outcomes, anchor)
        logger.info(f"  -return_1d available for {len(naive):,} items")

        floor = fc._artifact_min_median_price
        if not floor:
            floor = MIN_SERVED_PRICE_USD
            logger.warning(
                f"  the artifact records no train_min_median_price; scoring the "
                f"served cohort at >= ${floor:g} instead. Pooling every item "
                f"would report the penny-item score "
                f"(docs/changelog offline-DA-inflated-by-stale-prices).")
        # One denominator for every arm, computed once per anchor. `rankIC`
        # below divides by what THIS run quoted; `pinnedIC` divides by the
        # shipped smoothed anchor whatever this run quoted, so two arms that
        # quote different prices are still comparable. They are equal today,
        # because production quotes the pinned statistic -- a divergence means
        # an arm is live.
        pinned = _pinned_anchor(outcomes, anchor)
        # Also a constant across arms, and for the same reason: the tied cohort
        # is where the durable served signal was measured and the deviating one
        # is what the freshness gate reads, so neither may be redefined by the
        # arm being scored.
        tied = _tied_mask(outcomes, anchor)
        if not _pin_matches_production():
            from backtest.price_resolution import (MAX_WINDOW_SPAN_DAYS,
                                                   SMOOTH_WINDOW)
            logger.warning(
                f"  the pinned denominator ({PINNED_SMOOTH_WINDOW} obs / "
                f"{PINNED_MAX_SPAN_DAYS}d) no longer matches production "
                f"({SMOOTH_WINDOW} obs / {MAX_WINDOW_SPAN_DAYS}d). `pinnedIC` "
                f"is still comparable across arms; it is no longer the basis "
                f"production quotes from. Check FALLBACK_MAX_AGE_DAYS.")

        print(f"\nSERVING REPLAY @ {anchor}   (cohort floor >= ${floor:g})")
        print(f"{'h':>4} {'n':>7} {'DA%':>7} {'down%':>7} {'edge':>7} "
              f"{'rankIC':>8} {'naiveIC':>8} {'vs naive':>9} {'pinnedIC':>9}")

        dollar_rows: list[tuple[int, dict]] = []
        coverage_rows: list[tuple[int, dict]] = []
        for h in horizons:
            frame = _served_rows(served, h)
            if frame.empty:
                continue

            realised = _resolve(outcomes, anchor + timedelta(days=h), after=anchor)
            frame = frame.merge(realised, on="item_id", how="inner")
            # Before the returns are computed: a merge resets the index, and
            # the return Series are positionally aligned to this frame.
            frame = frame.merge(naive, on="item_id", how="left")
            # The served cohort is the only population the headline describes.
            frame = frame[frame["current"] >= floor]
            frame = frame[np.isfinite(frame["realised"])
                          & (frame["current"] > 0)].reset_index(drop=True)
            if len(frame) < 3:
                print(f"{h:>4} {len(frame):>7}   too few resolved outcomes")
                continue

            actual_ret = frame["realised"] / frame["current"] - 1.0
            pred_ret = frame["mid"] / frame["current"] - 1.0

            hit = (np.sign(pred_ret) == np.sign(actual_ret)) & (actual_ret != 0)
            scored = (actual_ret != 0)
            da = 100.0 * hit.sum() / max(scored.sum(), 1)
            down = 100.0 * (actual_ret < 0).mean()
            # The runnable baseline is always-down, never constant_call --
            # invariant #4 in backend/AGENTS.md.
            base = max(down, 100.0 - down)
            ic = _rank_ic(pred_ret.to_numpy(), actual_ret.to_numpy())
            # The baseline the model measurably loses to on rank IC in CV.
            have_naive = frame["naive"].notna().to_numpy()
            naive_ic = (_rank_ic(frame["naive"].to_numpy()[have_naive],
                                 actual_ret.to_numpy()[have_naive])
                        if have_naive.sum() >= 3 else float("nan"))
            pinned_ic = _pinned_rank_ic(frame, pinned)
            print(f"{h:>4} {len(frame):>7} {da:>7.2f} {down:>7.2f} "
                  f"{da - base:>+7.2f} {ic:>8.4f} {naive_ic:>8.4f} "
                  f"{ic - naive_ic:>+9.4f} {pinned_ic:>9.4f}")
            dollar_rows += [(h, r) for r in
                            _dollar_error_rows(frame, pinned, tied)]
            coverage_rows.append((h, _coverage_row(frame)))

        # The gate of the freshness experiment, and the only table here that two
        # arms can be compared on: rank IC above divides by the served quote,
        # which an arm moves. Read the DEVIATING row -- on the tied cohort an arm
        # that chooses between the raw quote and its own median serves the same
        # price, so a pooled number dilutes the effect with rows that cannot
        # move (`2026-08-11-smoothed-anchor-label-measured.md`).
        print(f"\nDOLLAR ERROR @ {anchor}   "
              f"(|x - realised| / realised, %; basis-free)")
        print(DOLLAR_HEADER)
        for h, r in dollar_rows:
            print(_dollar_line(h, r))

        # The band, which this script scored nothing about until 2026-08-11.
        # q_hat is fitted to cover 80% of residuals to a centre `predict` then
        # moves (`_recenter_on_direction`), so `cov%` short of `target%` with the
        # misses stacked on one side is that displacement, not a wide market.
        # `conformal_centre` in meta.json says which centre this artifact's q_hat
        # was fitted around.
        centre = getattr(fc, "conformal_centre", {}) or {}
        centres = ", ".join(f"{h}d={centre.get(h, 'unknown')}"
                            for h, _ in coverage_rows)
        print(f"\nBAND COVERAGE @ {anchor}   "
              f"(low <= realised <= high; conformal_centre: {centres})")
        print(COVERAGE_HEADER)
        for h, r in coverage_rows:
            print(_coverage_line(h, r))

        if "--basis-sweep" in sys.argv:
            # The SAME served mids, scored against four label bases. Everything
            # else -- artifact, anchor, items, transforms -- is held fixed, so a
            # difference here is the label definition and nothing else.
            print(f"\nLABEL BASIS SWEEP @ {anchor}   "
                  f"(same served mids; 'cv' is prepare_targets' own basis)")
            print(f"{'h':>4} {'basis':>10} {'n':>7} {'rankIC':>8} {'vs served':>10}")
            for h in horizons:
                base_f = _served_rows(served, h)
                if base_f.empty:
                    continue
                base_f = base_f[base_f["current"] >= floor]
                bases = _basis_frame(fc, outcomes, anchor, h)
                merged = base_f.merge(bases, on="item_id", how="left")
                # pred_ret keeps the served denominator throughout: the model
                # quoted a dollar mid against it, and re-deriving the prediction
                # per basis would change the prediction as well as the label.
                pred = (merged["mid"] / merged["current"] - 1.0).to_numpy()
                served_ic = None
                for name in ("served", "cv", "num_only", "den_only", "cv_lag"):
                    ok = np.isfinite(merged[name].to_numpy()) & np.isfinite(pred)
                    ic_b = (_rank_ic(pred[ok], merged[name].to_numpy()[ok])
                            if ok.sum() >= 3 else float("nan"))
                    if name == "served":
                        served_ic = ic_b
                    delta = "" if name == "served" else f"{ic_b - served_ic:>+10.4f}"
                    print(f"{h:>4} {name:>10} {int(ok.sum()):>7} {ic_b:>8.4f} {delta}")

                # The decisive split. Where the anchor quote equals its own
                # local median, `cv` and `served` share a denominator exactly,
                # so the shared-quote channel cannot operate. If the gap lives
                # in the deviating half and vanishes in the tied half, the
                # deviation IS the noise the model is reading.
                # fillna before astype: the merge is a LEFT join, so an item
                # the basis frame never saw arrives as NaN and the column comes
                # back float. `~` on that raises rather than masking, which is
                # how this was caught -- but "no anchor observed" is not "tied".
                tied = merged["anchor_is_tied"].fillna(False).astype(bool).to_numpy()
                for label, mask in (("tied", tied), ("deviating", ~tied)):
                    fin = (np.isfinite(merged["cv"].to_numpy())
                           & np.isfinite(merged["served"].to_numpy())
                           & np.isfinite(pred) & mask)
                    if fin.sum() < 3:
                        print(f"{h:>4} {label:>10} {int(fin.sum()):>7}   too few")
                        continue
                    ic_cv = _rank_ic(pred[fin], merged["cv"].to_numpy()[fin])
                    ic_sv = _rank_ic(pred[fin], merged["served"].to_numpy()[fin])
                    print(f"{h:>4} {label:>10} {int(fin.sum()):>7} "
                          f"{ic_sv:>8.4f} {ic_cv - ic_sv:>+10.4f}  "
                          f"(cv {ic_cv:+.4f})")

        print("\nDA is quotable only beside down% and the PT test "
              "(backend/AGENTS.md invariant 4). One anchor is one date: this "
              "is a path check with a number attached, not an accuracy result.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
