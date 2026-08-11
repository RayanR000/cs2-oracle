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
from models.forecaster import ItemForecaster           # noqa: E402
from api.serving_policy import MIN_SERVED_PRICE_USD    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# The resolver's tolerance, matching the backtest: an outcome is the nearest
# observation within this many days of the target, not the exact day. The
# archive has whole missing days (docs/changelog aggregator-archive-day-gaps),
# so an exact-day join silently drops items rather than scoring them.
OUTCOME_TOLERANCE_DAYS = 3


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
    t = pd.Timestamp(anchor)
    hist = outcomes.rename(columns={"day": "date"})
    smoothed = pd.Series(
        fc._smoothed_anchor_prices(hist[hist["date"] <= t], t), name="anchor_smooth")
    smoothed.index.name = "item_id"

    raw_anchor = _exact_day(outcomes, anchor).rename(columns={"px": "anchor_raw"})
    raw_target = _exact_day(outcomes, anchor + timedelta(days=horizon)) \
        .rename(columns={"px": "out_raw"})
    med_target = _resolve(outcomes, anchor + timedelta(days=horizon), after=anchor) \
        .rename(columns={"realised": "out_med"})

    f = (raw_anchor.merge(smoothed.reset_index(), on="item_id", how="outer")
                   .merge(raw_target, on="item_id", how="outer")
                   .merge(med_target, on="item_id", how="outer"))
    for name, num, den in (("served", "out_med", "anchor_smooth"),
                           ("cv", "out_raw", "anchor_raw"),
                           ("num_only", "out_raw", "anchor_smooth"),
                           ("den_only", "out_med", "anchor_raw")):
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
        print(f"\nSERVING REPLAY @ {anchor}   (cohort floor >= ${floor:g})")
        print(f"{'h':>4} {'n':>7} {'DA%':>7} {'down%':>7} {'edge':>7} "
              f"{'rankIC':>8} {'naiveIC':>8} {'vs naive':>9}")

        for h in horizons:
            rows = []
            for _, r in served.iterrows():
                f = r["forecasts"].get(h)
                if not f or f.get("mid") is None or not r.get("current_price"):
                    continue
                rows.append({"item_id": r["item_id"],
                             "current": float(r["current_price"]),
                             "mid": float(f["mid"])})
            frame = pd.DataFrame(rows)
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
            print(f"{h:>4} {len(frame):>7} {da:>7.2f} {down:>7.2f} "
                  f"{da - base:>+7.2f} {ic:>8.4f} {naive_ic:>8.4f} "
                  f"{ic - naive_ic:>+9.4f}")

        if "--basis-sweep" in sys.argv:
            # The SAME served mids, scored against four label bases. Everything
            # else -- artifact, anchor, items, transforms -- is held fixed, so a
            # difference here is the label definition and nothing else.
            print(f"\nLABEL BASIS SWEEP @ {anchor}   "
                  f"(same served mids; 'cv' is prepare_targets' own basis)")
            print(f"{'h':>4} {'basis':>10} {'n':>7} {'rankIC':>8} {'vs served':>10}")
            for h in horizons:
                rows = [{"item_id": r["item_id"],
                         "current": float(r["current_price"]),
                         "mid": float(r["forecasts"][h]["mid"])}
                        for _, r in served.iterrows()
                        if r["forecasts"].get(h)
                        and r["forecasts"][h].get("mid") is not None
                        and r.get("current_price")]
                base_f = pd.DataFrame(rows)
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
                for name in ("served", "cv", "num_only", "den_only"):
                    ok = np.isfinite(merged[name].to_numpy()) & np.isfinite(pred)
                    ic_b = (_rank_ic(pred[ok], merged[name].to_numpy()[ok])
                            if ok.sum() >= 3 else float("nan"))
                    if name == "served":
                        served_ic = ic_b
                    delta = "" if name == "served" else f"{ic_b - served_ic:>+10.4f}"
                    print(f"{h:>4} {name:>10} {int(ok.sum()):>7} {ic_b:>8.4f} {delta}")

        print("\nDA is quotable only beside down% and the PT test "
              "(backend/AGENTS.md invariant 4). One anchor is one date: this "
              "is a path check with a number attached, not an accuracy result.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
