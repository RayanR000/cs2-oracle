"""Directional accuracy on the calls that clear their own friction.

Every error metric this repo reports is blind to the same thing: a forecast can
be accurate and imply no trade. Round trip is +2.0% at CSFloat and +16.1% on
Steam, and the bid-ask spread runs 35.5% sub-$1 to 5.2% at $1000+ — so a 3%
predicted move is inside the spread for five of six tiers. Friction is not in
the loss, so no loss can see it.

    ActionableDA(v, h) = P( sign(r_act) = sign(r_hat) | |r_hat| > RT_v + s_i )

Reported as FOUR numbers together, because any one of them alone misleads:
``n_actionable / n_total``, ``ActionableDA``, ``E[net]``, and PT on the subset. A
60% ActionableDA over 11 rows is not a finding.

Pure: no I/O, no clock. Reads only ``base_price``, ``predicted_mid``,
``actual_price`` and ``price_tier`` off records, all of which are frozen columns
on ``forecast_outcomes``.
"""

from __future__ import annotations

from backtest.directional_test import pesaran_timmermann
from backtest.friction import DEFAULT_VENUE, actionable_threshold

# An actionable metric at h=3 is a category error: nothing under 8 days is
# executable at all once listing and trade-hold times are counted, so the
# conditioning event is not a decision anyone could act on.
ACTIONABLE_HORIZONS = frozenset({14, 30})


def _sign(value: float) -> int:
    """Raw sign, with an exact zero returning 0 rather than picking a side.

    Deliberately NOT ``direction_from_return``: its +-0.5% flat band is
    meaningless on a move required to clear >= 7.2%, and folding a zero into
    "flat" would let a carried-forward price match a "flat" call and score as a
    hit. sign(0) matches nothing, so a frozen price is a miss.
    """
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _prediction_base(r: dict) -> float:
    """The price the forecast was QUOTED FROM, which is what `r_hat` divides by.

    `predict()` builds `predicted_mid` as `current_price x (1 + r_hat)`, so
    dividing it by the archive-resolved `base_price` does not recover `r_hat` --
    it recovers `r_hat` plus the wedge between the two bases. That wedge is a
    median 13.74% on this metric's cohort against thresholds of 7.2-37.5%, and
    it was supplying the conviction: 1,141 rows selected on the resolved base
    against 21 on the served one, with all 1,120 differences *gained*
    (`docs/changelog/2026-08-11-actionable-selection-is-the-base-wedge.md`).

    **This does not breach "current_price is never scored on".** That rule is
    about the two legs of `actual_ret`, which must both stay on `resolve_anchors`
    -- differencing two estimators there is what let one cohort score 61.76% and
    33.74% on consecutive days. The outcome leg below is untouched.

    Falls back to `base_price` when there is no served quote. `fold_records`
    never has one and does not need one -- it builds `mid = base * (1 + mid_ret)`
    so the resolved base *is* the quote -- and legacy outcomes predate the
    column. Dropping either would shrink the cohort silently.
    """
    served = r.get("current_price")
    if served is not None and served > 0:
        return float(served)
    return float(r["base_price"])


def _empty(scope: str, venue: str) -> dict:
    """The fixed key set, with every measured field absent.

    The shape must not depend on the data: a metrics payload whose keys vary by
    cohort is what forced every nested value in this store to JSON text. And
    None rather than 0.0 throughout — a zero here reads as "the model scored
    nothing", which is a different claim from "there is no subset to score".
    """
    out = {
        "actionable_scope": scope,
        "actionable_venue": venue,
        "actionable_n": None,
        "actionable_share_pct": None,
        "actionable_da": None,
        "actionable_e_net_pct": None,
        # Which denominator formed `r_hat`, as counts rather than a label: this
        # change puts a discontinuity in the stored series, and a payload that
        # cannot say which convention produced it is not self-describing. Same
        # reasoning as `purge` / `embargo_days` on the walkforward rows.
        "actionable_n_served_basis": None,
        "actionable_n_fallback_basis": None,
    }
    out.update({f"actionable_{k}": None for k in pesaran_timmermann([], 1)})
    return out


def actionable_metrics(
    records: list[dict],
    horizon_days: int | None,
    min_dates: int,
    venue: str = DEFAULT_VENUE,
) -> dict:
    """The four friction-conditioned numbers for one cohort, as flat keys.

    ``horizon_days`` outside ``ACTIONABLE_HORIZONS`` — including None, which is
    what a record predating the field gives — returns ``out_of_scope`` with every
    value None. That is not the same state as an empty subset, and the two must
    never be reported identically.

    ``actionable_share_pct`` divides by the SCOREABLE cohort, not by
    ``len(records)``: a row with a non-positive base or no prediction leg cannot
    form ``r_hat`` at all, so counting it in the denominator would report a
    smaller actionable share on the strength of rows that were never candidates.

    Does not mutate ``records``.
    """
    if horizon_days not in ACTIONABLE_HORIZONS:
        return _empty("out_of_scope", venue)

    usable = [r for r in records if r.get("predicted_mid") is not None and (r.get("base_price") or 0) > 0]
    if records and not usable:
        # Distinguishable from "nothing cleared the threshold": these rows carry
        # no prediction leg at all, so the metric was never computable on them.
        return _empty("no_prediction_leg", venue)

    subset = []
    n_served = 0
    for r in usable:
        base = r["base_price"]
        quote = _prediction_base(r)
        n_served += quote != base
        # Two different denominators on purpose. `r_hat` is a property of the
        # PREDICTION and divides by what the prediction was quoted from; `r_act`
        # is the outcome and stays on the resolver's basis, both legs.
        r_hat = (r["predicted_mid"] - quote) / quote
        if abs(r_hat) > actionable_threshold(r["price_tier"], venue):
            subset.append((r, r_hat, (r["actual_price"] - base) / base))

    out = _empty("in_scope", venue)
    out["actionable_n"] = len(subset)
    out["actionable_n_served_basis"] = n_served
    out["actionable_n_fallback_basis"] = len(usable) - n_served
    out["actionable_share_pct"] = round(len(subset) / len(usable) * 100, 2) if usable else 0.0
    if not subset:
        return out

    hits = sum(1 for _, r_hat, r_act in subset if _sign(r_act) == _sign(r_hat))
    net = [_sign(r_hat) * r_act - actionable_threshold(r["price_tier"], venue) for r, r_hat, r_act in subset]

    out["actionable_da"] = round(hits / len(subset) * 100, 2)
    out["actionable_e_net_pct"] = round(sum(net) / len(net) * 100, 4)
    out.update({f"actionable_{k}": v for k, v in pesaran_timmermann([r for r, _, _ in subset], min_dates).items()})
    return out
