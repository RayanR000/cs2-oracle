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

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.serving_policy import MIN_SERVED_PRICE_USD
from database import SessionLocal
from db.archive import prices_relation
from models import conformal
from models.forecaster import ItemForecaster
from models.item_parser import archive_universe_sql_filter

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

# How far either side of the anchor the feed audit looks for a comparison. The
# archive drops whole calendar days (07-27, 07-30, 08-02, 08-03 in 2026), so a
# ±1 window can find nothing to compare against and would pass a bad anchor by
# default. 3 days each way survives two adjacent gaps.
ANCHOR_NEIGHBOURHOOD_DAYS = 3
# Below this share of the neighbours' median item count, the anchor is a partial
# day rather than a day. 2026-07-09 sits at 0.21 and the thin days of the
# 07-11..07-13 regime transition at 0.21-0.33, against 0.98-1.02 for an
# ordinary day.
ANCHOR_MIN_ITEM_SHARE = 0.6
# Set to run anyway. Prints the audit as a WARNING banner instead of refusing —
# for deliberately replaying a known-bad anchor, which is a diagnosis and not a
# measurement.
ALLOW_DIRTY_ANCHOR_ENV = "ALLOW_DIRTY_ANCHOR"
# Audited when the caller names no horizons. The outcome-side check runs BEFORE
# the artifact loads, so it cannot ask the artifact which horizons it serves --
# and the audit is worth more early than exact.
DEFAULT_AUDIT_HORIZONS = (3, 7, 14, 30)


def _feed_profile(
    anchor: date, archive_dir: Path | None = None, window: int = ANCHOR_NEIGHBOURHOOD_DAYS
) -> pd.DataFrame:
    """Per day around `anchor`: which sources wrote it, and how many items.

    Two columns, one aggregate query. Through `prices_relation` and
    `archive_universe_sql_filter` because `backend/AGENTS.md` invariants 1 and 2
    hold for any loader that globs the archive — a raw glob here would compare
    the anchor's source set against a set the first Parquet file's schema
    allowed, which is exactly the silent narrowing those invariants exist for.

    `source` is NULL for the whole pre-2026 series, so it is COALESCEd to a
    label: a 2024 anchor then reads `{<none>}` on every day in its window and
    compares equal, rather than every set being empty and the audit vacuous.
    """
    lo = anchor - timedelta(days=window)
    hi = anchor + timedelta(days=window)
    con = duckdb.connect()
    try:
        relation = prices_relation(con, archive_dir, columns=["item_slug", "day", "source"])
        return con.sql(f"""
            SELECT CAST(day AS DATE) AS day,
                   count(DISTINCT item_slug) AS items,
                   list_sort(list(DISTINCT COALESCE(source, '<none>'))) AS sources
            FROM {relation} sub
            WHERE day >= DATE '{lo.isoformat()}' AND day <= DATE '{hi.isoformat()}'
              AND (source IS NULL OR source NOT LIKE 'historical_fallback:%')
              AND {archive_universe_sql_filter("sub.item_slug", "sub.source")}
            GROUP BY 1 ORDER BY 1
        """).fetchdf()
    finally:
        con.close()


def audit_anchor_feed(anchor: date, profile: pd.DataFrame) -> tuple[bool, list[str]]:
    """Is `anchor` a day the archive collected the way it collected its neighbours?

    Answers the one question that cost a day of band experiments: 2026-07-09's
    usual feed (`aggregator_steam_17mafo`, ~26,170 items) is **absent**, and
    `aggregator_sync` (5,502 items) stands in quoting the served cohort 1.287x
    higher, reverting 0.773x the next day. Every horizon inherits that spike
    through the anchor, so the down-rate, the DA, the tied share and the learned
    band's width all reverse on that one date and on no other.
    `docs/changelog/2026-08-12-july-09-anchor-is-a-feed-substitution.md`.

    The test is deliberately about *collection*, not about prices: a level check
    would have to pick a denominator, and any denominator it picked would be one
    of the arms. Which feeds wrote the day and how many items they covered are
    properties of the archive alone.

    Returns (ok, lines). The caller decides whether to refuse; the lines are the
    audit either way, so a run that proceeds still records what it proceeded on.
    """
    lines: list[str] = []
    if profile.empty:
        return False, [f"the archive holds NO rows within {ANCHOR_NEIGHBOURHOOD_DAYS} days of {anchor}."]

    days = {pd.Timestamp(d).date(): row for d, row in zip(profile["day"], profile.to_dict("records"))}
    me = days.get(anchor)
    if me is None:
        return False, [
            f"{anchor} is not in the archive at all — it is one of "
            f"the dropped calendar days. Neighbours present: "
            f"{', '.join(str(d) for d in sorted(days))}."
        ]

    others = [r for d, r in days.items() if d != anchor]
    if not others:
        return False, [
            f"{anchor} is the only day the archive holds within "
            f"±{ANCHOR_NEIGHBOURHOOD_DAYS} days, so there is "
            f"nothing to compare its collection against."
        ]

    my_sources = frozenset(me["sources"])

    def modal(rows) -> frozenset | None:
        """The most common source set among *rows*, not their union.

        The union would widen the reference by any one neighbour that sat inside
        a regime transition, and let a substitution through as a subset of it.
        Ties break toward the larger set, so a two-day window that disagrees
        with itself is reported rather than resolved by dict order.
        """
        counts: dict[frozenset, int] = {}
        for r in rows:
            key = frozenset(r["sources"])
            counts[key] = counts.get(key, 0) + 1
        return max(counts, key=lambda s: (counts[s], len(s))) if counts else None

    # The two sides are checked SEPARATELY because they fail for different
    # reasons and want different words. A day whose collection differs from the
    # days BEFORE it has features -- every lag, every rolling window -- reading a
    # synthetic jump across the change; that is the 2026-03-22 consensus break.
    # A day differing from the days AFTER it resolves its outcome on a basis its
    # own quote was not measured on. 2026-07-09 is both at once, which is why it
    # reversed all four horizons rather than tilting them.
    before = [r for d, r in days.items() if d < anchor]
    after = [r for d, r in days.items() if d > anchor]
    ref_before, ref_after = modal(before), modal(after)

    matching = [
        r
        for r in others
        if frozenset(r["sources"]) in {ref_before, ref_after} and frozenset(r["sources"]) == my_sources
    ]
    pool = matching or others
    ref_items = float(np.median([r["items"] for r in pool]))
    share = me["items"] / ref_items if ref_items else float("nan")

    lines.append(
        f"anchor {anchor}: {me['items']:,} items from "
        f"{len(my_sources)} source(s); {len(before)} day(s) before it "
        f"and {len(after)} after, reference {ref_items:,.0f} items"
    )

    ok = True
    for side, ref, consequence in (
        ("BEFORE", ref_before, "the anchor's own features read a synthetic jump across the change"),
        ("AFTER", ref_after, "the outcome resolves on a basis the anchor was not quoted on"),
    ):
        if ref is None or ref == my_sources:
            continue
        ok = False
        missing, extra = ref - my_sources, my_sources - ref
        lines.append(f"  the days {side} the anchor were collected differently — {consequence}:")
        if missing:
            lines.append(f"    absent on the anchor: {', '.join(sorted(missing))}")
        if extra:
            lines.append(f"    only on the anchor:   {', '.join(sorted(extra))}")
    if ok and ref_before != ref_after:
        # Only reachable when one side is None: if both sides had days and the
        # audit still passes, both equal `my_sources` and so equal each other.
        # So this says "half the comparison was unavailable", not "a change".
        lines.append("  (only one side of the window has days to compare against; the anchor matches the side it has)")
    if np.isfinite(share) and share < ANCHOR_MIN_ITEM_SHARE:
        ok = False
        lines.append(
            f"  item count is {share:.0%} of that reference "
            f"(floor {ANCHOR_MIN_ITEM_SHARE:.0%}): a partial day, so "
            f"most items are served from a quote before the anchor."
        )
    return ok, lines


def cutovers_from_counts(counts: pd.Series) -> list:
    """Dates where the collected universe changed size abruptly.

    `ItemForecaster._collection_shift_dates`' rule, applied to a per-day item
    count the audit already has. Deliberately the SAME rule and the same
    threshold, read off the class rather than restated: if the replay and the
    label path used two definitions of "cutover", a date could be scored here and
    voided there — which is exactly the state this function exists to end.
    """
    c = counts.sort_index()
    if len(c) < 2:
        return []
    prev = c.shift(1)
    change = (c - prev).abs() / prev.replace(0, np.nan)
    return [d for d, hit in zip(c.index, change > ItemForecaster.COLLECTION_SHIFT_FRACTION) if bool(hit)]


def cutovers_in_outcome_window(anchor: date, horizons, cutovers) -> dict:
    """{horizon: cutovers inside `(anchor, anchor + horizon]`}.

    THE GAP THIS CLOSES. `audit_anchor_feed` looks ±3 days around the anchor and
    asks how the day was collected; it cannot see a basis change 30 days out. The
    label path can, and voids any label whose window spans one — so an anchor that
    passes the feed audit can still have its OUTCOME quoted on a different source
    set, and the replay, which resolves outcomes itself, would score it anyway.

    Measured 2026-08-13: `2026-03-10` passes the feed audit and its 14d and 30d
    targets land the far side of the 2026-03-22 consensus break, where the
    archive's own consensus drops 8-10%. A band read on that cell is a read on a
    synthetic crash. Half the pre-registered six-anchor set for the date-level
    rescaling was affected.

    Span rule copied from `prepare_targets`: exclusive at the anchor — a cutover
    ON the anchor is the feed audit's business — inclusive at the target.
    """
    out = {}
    for h in horizons:
        target = anchor + timedelta(days=int(h))
        hit = [c for c in cutovers if anchor < c <= target]
        if hit:
            out[int(h)] = sorted(hit)
    return out


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


def _resolve(outcomes: pd.DataFrame, target: date, tolerance=OUTCOME_TOLERANCE_DAYS, after: date | None = None):
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
        (-(s_now[i] / s_prev[i] - 1.0) if s_prev.get(i) and s_prev[i] > 0 else np.nan) for i in naive["item_id"]
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
    return d.groupby("item_id", as_index=False)["price"].median().rename(columns={"price": "px"})


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

    return SMOOTH_WINDOW == PINNED_SMOOTH_WINDOW and MAX_WINDOW_SPAN_DAYS == PINNED_MAX_SPAN_DAYS


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
    smoothed = in_window.groupby("item_id").tail(PINNED_SMOOTH_WINDOW).groupby("item_id")["price"].median()
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
        np.isclose(
            raw.reindex(idx).to_numpy(dtype=float), pinned.reindex(idx).to_numpy(dtype=float), rtol=0, atol=1e-9
        ),
        index=idx,
        name="anchor_is_tied",
    )
    mask.index.name = "item_id"
    return mask


def _dollar_error_rows(frame: pd.DataFrame, pinned: pd.Series, tied: pd.Series) -> list[dict]:
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
    naive_mid = np.where(np.isfinite(d) & (d > 0), d * (1.0 + frame["naive"].to_numpy(dtype=float)), np.nan)
    mid = frame["mid"].to_numpy(dtype=float)
    current = frame["current"].to_numpy(dtype=float)
    realised = frame["realised"].to_numpy(dtype=float)
    scorable = np.isfinite(realised) & (realised > 0)

    rows = []
    for subset, mask in (("pooled", np.ones(len(frame), dtype=bool)), ("tied", is_tied), ("deviating", ~is_tied)):
        med, p90 = _rel_abs_error(mid[mask], realised[mask])
        quote, _ = _rel_abs_error(current[mask], realised[mask])
        naive, _ = _rel_abs_error(naive_mid[mask], realised[mask])
        rows.append(
            {
                "subset": subset,
                "n": int((mask & scorable & np.isfinite(mid)).sum()),
                "model": med,
                "p90": p90,
                "quote": quote,
                "naive": naive,
                "n_naive": int((mask & scorable & np.isfinite(naive_mid)).sum()),
            }
        )
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
        rows.append(
            {
                "item_id": r["item_id"],
                "current": float(r["current_price"]),
                "mid": float(f["mid"]),
                "low": _f(f.get("low")),
                "high": _f(f.get("high")),
                # Phase A disclosure: P(upside move clears cost). NaN on an
                # artifact with no exceedance head; carried so the
                # reliability table can score it against realised outcomes.
                "exceed_p": _f(f.get("exceed_p")),
            }
        )
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
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(real) & (real > 0)
    n = int(ok.sum())
    if n == 0:
        nan = float("nan")
        return {"n": 0, "cov": nan, "below": nan, "above": nan, "halfw": nan}
    centre = (hi[ok] + lo[ok]) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_half = np.where(centre > 0, (hi[ok] - lo[ok]) / 2.0 / centre, np.nan)
    return {
        "n": n,
        "cov": float(((real[ok] >= lo[ok]) & (real[ok] <= hi[ok])).mean()),
        "below": float((real[ok] < lo[ok]).mean()),
        "above": float((real[ok] > hi[ok]).mean()),
        # nanmedian of an all-NaN slice warns and returns NaN, which is the
        # right answer here and must not take out the table.
        "halfw": (float("nan") if not np.isfinite(rel_half).any() else float(np.nanmedian(rel_half))),
    }


COVERAGE_HEADER = f"{'h':>4} {'n':>7} {'cov%':>8} {'target%':>8} {'miss<low':>9} {'miss>high':>10} {'halfw%':>8}"


def _coverage_line(h: int, row: dict) -> str:
    """One row of the coverage table. A function for the same reason
    `_dollar_line` is one: a %-formatted NaN raising would take out the run at
    the point where the table prints."""
    # `.get`, not `row['halfw']`: this formats dicts that callers built before
    # the column existed, and a KeyError here would take out the whole table.
    halfw = row.get("halfw", float("nan"))
    return (
        f"{h:>4} {row['n']:>7} {100 * row['cov']:>8.2f} "
        f"{100 * conformal.NOMINAL_COVERAGE:>8.2f} "
        f"{100 * row['below']:>9.2f} {100 * row['above']:>10.2f} "
        f"{100 * halfw:>8.2f}"
    )


SIGMA_STRATA = 5


def _coverage_by_sigma_rows(frame: pd.DataFrame, n_strata: int = SIGMA_STRATA) -> list[dict]:
    """Served coverage within strata of `sigma` — the property `SIGMA_EXPONENT`
    targets, on the path that actually serves it.

    Everything on record about the tilt is measured on the OOF calibration
    records: coverage by `sigma` decile ramps 62->93 / 60->95 / 57->96 / 52->97%
    there (`2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md`). Nothing had
    ever checked it where the band is served, which is why the 2026-08-12 paired
    read could say the arm costs 8pp of marginal coverage without being able to
    say whether it flattened the profile it was built to flatten.

    **The stratifier is the served half-width.** `conformal.band` sets it to
    `q_hat * sigma ** beta`, strictly increasing in `sigma` for any `beta > 0`,
    so under `SIGMA_EXPONENT` its quantiles ARE `sigma` quantiles and the item
    ordering is identical in both arms — which is what makes those two tables
    comparable stratum by stratum. Deriving `sigma` from the archive instead
    would risk computing a different `sigma` than the band was built from; this
    cannot.

    ⚠️ **That identical-ordering claim does NOT extend to `LEARNED_SCALE`**, and
    an earlier version of this docstring said it did. A learned scale is a
    different variable, not a transform of `sigma`, so it reorders items freely:
    its stratum 3 and a control's stratum 3 are different cohorts. The table
    still answers the question that matters — *is coverage flat across the
    widths this arm actually serves*, which is the calibration property itself —
    but a stratum-by-stratum diff against a `sigma` control is not a like-for-like
    read, and the `ramp` is the only summary that survives the reordering.

    Two caveats a reader has to have. The relative width carries a `1/(1 + mid)`
    factor, so the ordering is `sigma`'s only up to the served mid, which is
    median |return| ~0.95% and cannot reorder materially. And a 30d band whose
    lower leg was clipped at zero is no longer `q_hat * sigma ** beta` wide —
    242 of ~990 were, on 2026-08-11 — so those rows sit in the wrong stratum;
    the count is reported as `n` per stratum rather than assumed uniform.

    `n_strata` is 5 rather than the audit's 10 because one anchor serves ~1,000
    rows: quintiles put ~200 in each, where deciles put ~100 and the per-stratum
    standard error (~4pp) would swallow the ramp being measured.
    """
    lo = frame["low"].to_numpy(dtype=float)
    hi = frame["high"].to_numpy(dtype=float)
    real = frame["realised"].to_numpy(dtype=float)
    centre = (hi + lo) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_half = np.where(centre > 0, (hi - lo) / 2.0 / centre, np.nan)
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(real) & (real > 0) & np.isfinite(rel_half)
    if ok.sum() < n_strata:
        return []

    w = rel_half[ok]
    covered = (real[ok] >= lo[ok]) & (real[ok] <= hi[ok])
    # Rank rather than value: `qcut` on a width with ties (a floor or a cap
    # binding) raises on duplicate edges, and the rank has none by construction.
    order = np.argsort(np.argsort(w, kind="stable"), kind="stable")
    edges = (order * n_strata) // len(w)

    rows = []
    for s in range(n_strata):
        m = edges == s
        if not m.any():
            continue
        rows.append(
            {"stratum": s + 1, "n": int(m.sum()), "cov": float(covered[m].mean()), "halfw": float(np.median(w[m]))}
        )
    return rows


def sigma_tilt_pp(rows: list[dict]) -> float:
    """Mean |coverage - nominal| across the strata, in points.

    The same statistic `conformal.coverage_by_sigma_stratum` returns, so a served
    number and an OOF one can be read side by side. ⚠️ It is NOT level-matched —
    the OOF version forces each arm to nominal marginal coverage first, precisely
    because this statistic falls whenever marginal coverage improves. Here the
    marginal level is the thing under test and cannot be matched away, so read
    this beside `BAND COVERAGE`'s `cov%` and never alone. See the "do not compare
    coverage schemes without matching the marginal level" entry in
    `2026-08-10-next-steps.md`.
    """
    if not rows:
        return float("nan")
    return float(np.mean([abs(100 * r["cov"] - 100 * conformal.NOMINAL_COVERAGE) for r in rows]))


SIGMA_HEADER = f"{'h':>4} {'stratum':>8} {'n':>7} {'cov%':>8} {'halfw%':>8}"


def _sigma_line(h: int, row: dict, n_strata: int = SIGMA_STRATA) -> str:
    """One stratum. Same NaN-safety reason as `_coverage_line`.

    The label is built before the f-string, not nested inside it: CI runs 3.11,
    where a same-quote nesting is a syntax error rather than the 3.12 behaviour.
    """
    label = f"{row['stratum']}/{n_strata}"
    return f"{h:>4} {label:>8} {row['n']:>7} {100 * row['cov']:>8.2f} {100 * row['halfw']:>8.2f}"


RELIABILITY_BINS = 5


def _reliability_rows(frame: pd.DataFrame, actual_ret, n_bins: int = RELIABILITY_BINS) -> list[dict]:
    """Calibration of the exceedance head: predicted `exceed_p` vs the realised
    exceedance rate, per fixed-width bin of predicted probability.

    `exceed_p` is the served ONE-SIDED head probability (Phase A). The realised
    outcome is `actual_ret > actionable_threshold(tier, csfloat)` — the SAME bar
    `prepare_targets` builds the label on, one-sided (a down move never clears
    it), tier taken from the anchor quote (`frame["current"]`) via the canonical
    `scoring.price_tier`, so this cannot drift from the label's threshold.

    Fixed-width bins over [0, 1] (not quantiles) so a bin means the same thing
    across anchors and horizons — the point is whether a stated 0.8 realises ~80%
    of the time. Returns [] when the artifact carries no head (`exceed_p` absent,
    or all NaN), so the table is skipped rather than printed empty. `SPREAD_BY_TIER`
    inside the threshold is a nearest-band approximation, not measured per tier
    (backtest/friction.py) — the calibration inherits that caveat.
    """
    if "exceed_p" not in frame.columns:
        return []
    from backtest.friction import actionable_threshold
    from backtest.scoring import price_tier

    p = frame["exceed_p"].to_numpy(dtype=float)
    ar = np.asarray(actual_ret, dtype=float)
    cur = frame["current"].to_numpy(dtype=float)
    ok = np.isfinite(p) & np.isfinite(ar) & np.isfinite(cur) & (cur > 0)
    if not ok.any():
        return []
    p, ar, cur = p[ok], ar[ok], cur[ok]
    thr = np.array([actionable_threshold(price_tier(float(c)), "csfloat") for c in cur])
    realized = (ar > thr).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": int(m.sum()),
                "pred": float(p[m].mean()),
                "realized": float(realized[m].mean()),
            }
        )
    return rows


def _reliability_ece(rows: list[dict]) -> float:
    """Expected calibration error: count-weighted mean |pred - realized| over the
    populated bins. NaN when there are no rows (no head), so it never takes out
    the table."""
    n = sum(r["n"] for r in rows)
    if n == 0:
        return float("nan")
    return sum(r["n"] * abs(r["pred"] - r["realized"]) for r in rows) / n


RELIABILITY_HEADER = f"{'h':>4} {'p-bin':>10} {'n':>7} {'pred%':>8} {'realized%':>10}"


def _reliability_line(h: int, row: dict) -> str:
    """One probability bin. NaN-safe like `_coverage_line`, though a populated bin
    always has finite pred/realized."""
    label = f"{row['lo']:.1f}-{row['hi']:.1f}"
    return f"{h:>4} {label:>10} {row['n']:>7} {100 * row['pred']:>8.2f} {100 * row['realized']:>10.2f}"


DOLLAR_HEADER = f"{'h':>4} {'subset':>10} {'n':>7} {'model':>8} {'p90':>8} {'quote':>8} {'naive':>8} {'nNaive':>7}"


def _dollar_line(h: int, row: dict) -> str:
    """One row of the dollar table, percentages.

    A function rather than an f-string inside `main()` so the NaN case is
    covered: a cohort under three items scores NaN, and a `%`-formatted NaN
    raising would take out an hour of CI at the point where the table prints.
    """
    return (
        f"{h:>4} {row['subset']:>10} {row['n']:>7} "
        f"{100 * row['model']:>8.2f} {100 * row['p90']:>8.2f} "
        f"{100 * row['quote']:>8.2f} {100 * row['naive']:>8.2f} "
        f"{row['n_naive']:>7}"
    )


def _basis_frame(fc: ItemForecaster, outcomes: pd.DataFrame, anchor: date, horizon: int) -> pd.DataFrame:
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
    lag_anchor = _exact_day(outcomes, anchor - timedelta(days=1)).rename(columns={"px": "anchor_lag"})
    raw_target = _exact_day(outcomes, anchor + timedelta(days=horizon)).rename(columns={"px": "out_raw"})
    med_target = _resolve(outcomes, anchor + timedelta(days=horizon), after=anchor).rename(
        columns={"realised": "out_med"}
    )

    f = (
        raw_anchor.merge(smoothed.reset_index(), on="item_id", how="outer")
        .merge(lag_anchor, on="item_id", how="outer")
        .merge(raw_target, on="item_id", how="outer")
        .merge(med_target, on="item_id", how="outer")
    )
    # The quote at the anchor sits in the label's denominator AND in the
    # features. Where it equals the local median there is no deviation for the
    # model to read, so this flag splits the cross-section into the half where
    # the hypothesised channel can operate and the half where it cannot.
    # `_tied_mask` and not an inline `isclose`: the dollar table reads the same
    # split, and two copies would drift.
    f = f.merge(_tied_mask(outcomes, anchor).reset_index(), on="item_id", how="left")
    f["anchor_is_tied"] = f["anchor_is_tied"].eq(True)
    for name, num, den in (
        ("served", "out_med", "anchor_smooth"),
        ("cv", "out_raw", "anchor_raw"),
        ("num_only", "out_raw", "anchor_smooth"),
        ("den_only", "out_med", "anchor_raw"),
        ("cv_lag", "out_raw", "anchor_lag"),
    ):
        f[name] = np.where(
            f[den].to_numpy(dtype=float) > 0, f[num].to_numpy(dtype=float) / f[den].to_numpy(dtype=float) - 1.0, np.nan
        )
    return f


def _rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman across the cross-section. One date, so no averaging."""
    if len(pred) < 3:
        return float("nan")
    return float(pd.Series(pred).corr(pd.Series(actual), method="spearman"))


def _requested_horizons(argv) -> set | None:
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


def _model_dir() -> str | None:
    """Which artifact directory to SERVE from.

    None keeps `ItemForecaster`'s default (`models/saved_models/`), the deployed
    artifact. The same variable `forecast_prices._model_dir` reads, deliberately:
    a band-geometry arm is trained into a scratch directory with
    `FORECAST_MODEL_DIR` set, and the replay has to be pointable at that SAME
    directory or the arm and its control both replay the deployed artifact and
    return identical numbers. Nothing is written here -- the replay only reads --
    but the directory is created so a typo surfaces as an empty-artifact refusal
    rather than a silent fall back to production's model.
    """
    raw = os.environ.get("FORECAST_MODEL_DIR")
    if not raw:
        return None
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"FORECAST_MODEL_DIR override: serving the artifact under {path} — models/saved_models/ is not read")
    return str(path)


def main() -> int:
    want = _requested_horizons(sys.argv)
    anchor = ItemForecaster.replay_anchor()
    if anchor is None:
        logger.error(
            "REPLAY_ANCHOR is not set. Refusing to run: without it "
            "this would 'replay' today and score a forecast whose "
            "outcome does not exist yet."
        )
        return 2

    # Before the artifact, before predict(): a bad anchor makes every number
    # below an artifact of the archive's collection, and the run costs ~4
    # minutes per anchor to find that out afterwards.
    ok, audit = audit_anchor_feed(anchor, _feed_profile(anchor))
    allowed = os.environ.get(ALLOW_DIRTY_ANCHOR_ENV) == "1"
    for line in audit:
        (logger.info if ok else logger.warning)(line)
    if not ok and not allowed:
        logger.error(
            f"REFUSING {anchor}: the archive did not collect this day the way "
            f"it collected its neighbours, so the served anchor price is a feed "
            f"artifact and every horizon inherits it. This is how 2026-07-09 "
            f"reversed four horizons at once "
            f"(changelog/2026-08-12-july-09-anchor-is-a-feed-substitution.md). "
            f"Pick another anchor, or set {ALLOW_DIRTY_ANCHOR_ENV}=1 to replay "
            f"it deliberately as a diagnosis."
        )
        return 2
    # The OUTCOME side, which the feed audit's ±3 day window cannot reach. An
    # anchor whose own collection is ordinary can still resolve across a basis
    # change: 2026-03-10 is clean here and its 14d/30d targets land the far side
    # of the 2026-03-22 consensus break.
    audit_horizons = sorted(want) if want else sorted(DEFAULT_AUDIT_HORIZONS)
    span = _feed_profile(anchor, window=max(audit_horizons))
    spanned = cutovers_in_outcome_window(
        anchor, audit_horizons, cutovers_from_counts(span.set_index(pd.to_datetime(span["day"]).dt.date)["items"])
    )
    if spanned:
        for h, cuts in sorted(spanned.items()):
            logger.warning(
                "h=%s resolves ACROSS a collector cutover on %s — the anchor is "
                "quoted on one source basis and its outcome on another, which "
                "`prepare_targets` voids as a label.",
                h,
                ", ".join(c.isoformat() for c in cuts),
            )
        survivors = [h for h in audit_horizons if h not in spanned]
        if not survivors and not allowed:
            logger.error(
                f"REFUSING {anchor}: every requested horizon spans a cutover. "
                f"Pick another anchor, or set {ALLOW_DIRTY_ANCHOR_ENV}=1."
            )
            return 2
        if survivors:
            logger.warning(
                "dropping %s; replaying %s",
                ",".join(str(h) for h in sorted(spanned)),
                ",".join(str(h) for h in survivors),
            )
            want = set(survivors)

    if not ok:
        logger.warning(
            f"{ALLOW_DIRTY_ANCHOR_ENV}=1: proceeding on an anchor that FAILED "
            f"the feed audit above. These numbers describe the archive's "
            f"collection, not the model — do not publish them as a measurement."
        )

    db = SessionLocal()
    try:
        fc = ItemForecaster(db_session=db, prune_failed_groups=False, model_dir=_model_dir())
        if not fc.load_models():
            logger.error(
                "No usable model artifact. Train one first — the replay serves an artifact, it does not build one."
            )
            return 2

        logger.info(f"Replaying the serving path at anchor {anchor}")
        logger.info(
            f"  artifact: xs_rank={fc._artifact_xs_rank} "
            f"floor={fc._artifact_min_median_price} "
            f"skipped={fc._artifact_xs_rank_skipped}"
        )

        served = fc.predict()
        if served.empty:
            logger.error("predict() returned nothing at this anchor.")
            return 1

        horizons = sorted({h for f in served["forecasts"] for h in f})
        if want is not None:
            missing = want - set(horizons)
            if missing:
                logger.error(
                    f"--horizons asked for {sorted(missing)}, which this artifact does not serve ({horizons})."
                )
                return 2
            horizons = sorted(want)
            logger.info(
                f"  scoring only {horizons} -- the rest of this "
                f"artifact's boosters were not trained under these "
                f"flags, so their rows would not be a measurement."
            )
        outcomes = _outcomes(fc, anchor, horizons)
        logger.info(f"  {len(served):,} served rows, {outcomes['item_id'].nunique():,} items in the outcome window")

        naive = _naive_baseline(fc, outcomes, anchor)
        logger.info(f"  -return_1d available for {len(naive):,} items")

        floor = fc._artifact_min_median_price
        if not floor:
            floor = MIN_SERVED_PRICE_USD
            logger.warning(
                f"  the artifact records no train_min_median_price; scoring the "
                f"served cohort at >= ${floor:g} instead. Pooling every item "
                f"would report the penny-item score "
                f"(docs/changelog offline-DA-inflated-by-stale-prices)."
            )
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
            from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS, SMOOTH_WINDOW

            logger.warning(
                f"  the pinned denominator ({PINNED_SMOOTH_WINDOW} obs / "
                f"{PINNED_MAX_SPAN_DAYS}d) no longer matches production "
                f"({SMOOTH_WINDOW} obs / {MAX_WINDOW_SPAN_DAYS}d). `pinnedIC` "
                f"is still comparable across arms; it is no longer the basis "
                f"production quotes from. Check FALLBACK_MAX_AGE_DAYS."
            )

        print(f"\nSERVING REPLAY @ {anchor}   (cohort floor >= ${floor:g})")
        print(
            f"{'h':>4} {'n':>7} {'DA%':>7} {'down%':>7} {'edge':>7} "
            f"{'rankIC':>8} {'naiveIC':>8} {'vs naive':>9} {'pinnedIC':>9}"
        )

        dollar_rows: list[tuple[int, dict]] = []
        coverage_rows: list[tuple[int, dict]] = []
        sigma_rows: list[tuple[int, list[dict]]] = []
        reliability_rows: list[tuple[int, list[dict]]] = []
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
            frame = frame[np.isfinite(frame["realised"]) & (frame["current"] > 0)].reset_index(drop=True)
            if len(frame) < 3:
                print(f"{h:>4} {len(frame):>7}   too few resolved outcomes")
                continue

            actual_ret = frame["realised"] / frame["current"] - 1.0
            pred_ret = frame["mid"] / frame["current"] - 1.0

            hit = (np.sign(pred_ret) == np.sign(actual_ret)) & (actual_ret != 0)
            scored = actual_ret != 0
            da = 100.0 * hit.sum() / max(scored.sum(), 1)
            down = 100.0 * (actual_ret < 0).mean()
            # The runnable baseline is always-down, never constant_call --
            # invariant #4 in backend/AGENTS.md.
            base = max(down, 100.0 - down)
            ic = _rank_ic(pred_ret.to_numpy(), actual_ret.to_numpy())
            # The baseline the model measurably loses to on rank IC in CV.
            have_naive = frame["naive"].notna().to_numpy()
            naive_ic = (
                _rank_ic(frame["naive"].to_numpy()[have_naive], actual_ret.to_numpy()[have_naive])
                if have_naive.sum() >= 3
                else float("nan")
            )
            pinned_ic = _pinned_rank_ic(frame, pinned)
            print(
                f"{h:>4} {len(frame):>7} {da:>7.2f} {down:>7.2f} "
                f"{da - base:>+7.2f} {ic:>8.4f} {naive_ic:>8.4f} "
                f"{ic - naive_ic:>+9.4f} {pinned_ic:>9.4f}"
            )
            dollar_rows += [(h, r) for r in _dollar_error_rows(frame, pinned, tied)]
            coverage_rows.append((h, _coverage_row(frame)))
            # Same frame, so the marginal and conditional tables describe the
            # same rows -- two coverage figures over different cohorts would not
            # be decomposable into each other.
            sigma_rows.append((h, _coverage_by_sigma_rows(frame)))
            # Exceedance-head calibration on the SAME frame. Empty (skipped
            # below) unless the artifact carries a head — EXCEEDANCE_HEAD off
            # emits no exceed_p, and this is the pre-flip validation of it.
            rel = _reliability_rows(frame, actual_ret.to_numpy())
            if rel:
                reliability_rows.append((h, rel))

        # The gate of the freshness experiment, and the only table here that two
        # arms can be compared on: rank IC above divides by the served quote,
        # which an arm moves. Read the DEVIATING row -- on the tied cohort an arm
        # that chooses between the raw quote and its own median serves the same
        # price, so a pooled number dilutes the effect with rows that cannot
        # move (`2026-08-11-smoothed-anchor-label-measured.md`).
        print(f"\nDOLLAR ERROR @ {anchor}   (|x - realised| / realised, %; basis-free)")
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
        centres = ", ".join(f"{h}d={centre.get(h, 'unknown')}" for h, _ in coverage_rows)
        print(f"\nBAND COVERAGE @ {anchor}   (low <= realised <= high; conformal_centre: {centres})")
        print(COVERAGE_HEADER)
        for h, r in coverage_rows:
            print(_coverage_line(h, r))

        # The conditional half. `cov%` above is marginal, and a band can hit 80%
        # marginally while covering 57% of its low-sigma items and 96% of its
        # high-sigma ones -- which is what the OOF records say this band does.
        # Stratum 1 is the NARROWEST band, i.e. the lowest sigma.
        print(
            f"\nBAND COVERAGE BY SIGMA @ {anchor}   "
            f"(quintiles of the served half-width, which is monotone in "
            f"sigma; stratum 1 = lowest sigma)"
        )
        print(SIGMA_HEADER)
        for h, srows in sigma_rows:
            for r in srows:
                print(_sigma_line(h, r))
            if srows:
                covs = [100 * r["cov"] for r in srows]
                print(
                    f"{h:>4} {'tilt':>8} {'':>7} "
                    f"{sigma_tilt_pp(srows):>8.2f} "
                    f"{covs[-1] - covs[0]:>+8.2f}   "
                    f"mean |cov-nominal| pp, and the ramp (last - first). "
                    f"NOT level-matched: read beside cov% above."
                )

        # Exceedance-head calibration: does a stated P(upside move > cost)
        # realise at that rate? Printed only when the artifact carries a head
        # (EXCEEDANCE_HEAD=1 at train time); otherwise there is no exceed_p to
        # score and the block is silent. This is the pre-flip validation, the
        # exceedance analogue of BAND COVERAGE — it runs on replayed forecasts
        # because no served exceed_p exists in the outcomes store yet.
        if reliability_rows:
            print(
                f"\nEXCEEDANCE RELIABILITY @ {anchor}   "
                f"(pred vs realised P(actual_ret > cost); one-sided, "
                f"tier-thresholded; well-calibrated => pred% ~ realized%)"
            )
            print(RELIABILITY_HEADER)
            for h, rel in reliability_rows:
                for r in rel:
                    print(_reliability_line(h, r))
                print(
                    f"{h:>4} {'ECE':>10} {'':>7} "
                    f"{100 * _reliability_ece(rel):>8.2f}   "
                    f"count-weighted mean |pred-realized| pp"
                )

        if "--basis-sweep" in sys.argv:
            # The SAME served mids, scored against four label bases. Everything
            # else -- artifact, anchor, items, transforms -- is held fixed, so a
            # difference here is the label definition and nothing else.
            print(f"\nLABEL BASIS SWEEP @ {anchor}   (same served mids; 'cv' is prepare_targets' own basis)")
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
                    ic_b = _rank_ic(pred[ok], merged[name].to_numpy()[ok]) if ok.sum() >= 3 else float("nan")
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
                    fin = (
                        np.isfinite(merged["cv"].to_numpy())
                        & np.isfinite(merged["served"].to_numpy())
                        & np.isfinite(pred)
                        & mask
                    )
                    if fin.sum() < 3:
                        print(f"{h:>4} {label:>10} {int(fin.sum()):>7}   too few")
                        continue
                    ic_cv = _rank_ic(pred[fin], merged["cv"].to_numpy()[fin])
                    ic_sv = _rank_ic(pred[fin], merged["served"].to_numpy()[fin])
                    print(
                        f"{h:>4} {label:>10} {int(fin.sum()):>7} "
                        f"{ic_sv:>8.4f} {ic_cv - ic_sv:>+10.4f}  "
                        f"(cv {ic_cv:+.4f})"
                    )

        print(
            "\nDA is quotable only beside down% and the PT test "
            "(backend/AGENTS.md invariant 4). One anchor is one date: this "
            "is a path check with a number attached, not an accuracy result."
        )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
