"""Paper-trading P&L: what the served forecasts would have earned net of real friction.

An honest execution harness over frozen ``forecast_outcomes`` records, built to be
*able* to report "no cash edge". ``AGENTS.md`` states this is a RANGE forecaster and
direction is not a shippable claim; every trading-adjacent arm measured so far is
CV-positive and serving-negative, and this harness exists to price that finding rather
than to overturn it.

Pure: no I/O, no clock. Reads only the frozen fields ``base_price``,
``actual_price``, ``predicted_low``, ``predicted_mid``, ``current_price``,
``price_tier``, ``horizon_days``, ``forecast_date`` and ``base_stale_run_days`` off
records — the same record shape and discipline as ``backtest/actionable.py``
(``predicted_mid`` / ``predicted_low``, not the DB's ``predicted_price_*`` column
names; the loader renames on the way in).

**Two bases, on purpose (the 2026-08-11 invariant).** P&L is executed on the
RESOLVER basis: a round trip buys at ``base_price`` and sells at ``actual_price``,
both legs resolved by ``resolve_anchors`` with one estimator — differencing two
estimators there is the bug that let one cohort score 61.76% and 33.74% on
consecutive days. The trade *decision* is made in the SERVED basis, on the quote the
forecaster actually saw (``current_price``, falling back to ``base_price``) and the
served band, because that is the only information a trader had at decision time.

**Real venue rules, from the repo's own microstructure**
(``docs/research/2026-08-07-cs2-forecasting-research.md`` section 11):

- Round trip = venue seller fee + the tier's bid-ask spread, crossed once
  (``friction.py``). CSFloat is +2%; the spread runs 35.5% sub-$1 to 5.2% at $1000+.
- **Steam Wallet is walled.** The Subscriber Agreement gives Wallet funds no cash
  value, so a Steam round trip is not real money: steam is in ``WALLED_VENUES`` and
  every payload carries ``pt_cashable`` so a Steam return can never fold into a cash
  P&L.
- **Only h in {14, 30} is executable** — the 7-day market lock plus 7-day trade
  protection mean nothing under ~8 days can be bought and sold. (The unsourced
  "8-day cooldown" is deliberately not encoded; section 11 found no source.)
- **No cross-venue arbitrage.** The ~1.43x Steam premium is a structurally
  unarbitrageable level shift and the cash venues agree within 2%, so every trade is
  a single-venue round trip.

The cross-sectional long-short read-out lives in ``backtest/longshort.py`` and is
deliberately given no P&L here: the short leg is not executable on this market.

Fills are assumed at the quoted price, which is a best case — section 11's liquidity
finding (a flagship skin trades ~96x/day, most items single-digit) means real fills
would slip and partially fill. Results are reported by tier rather than pooled, since
both spread and liquidity improve monotonically with price.
"""

from __future__ import annotations

import numpy as np

from backtest.friction import (
    DEFAULT_VENUE,
    ROUND_TRIP_COST,
    SPREAD_BY_TIER,
    actionable_threshold,
)
from backtest.scoring import BOOTSTRAP_CI, BOOTSTRAP_RNG_SEED, N_BOOTSTRAP

# 7-day market lock + 7-day trade protection: nothing under ~8 days is executable,
# so a P&L at h in {3, 7} would describe a trade nobody could place. Same frozen set
# as backtest.actionable.ACTIONABLE_HORIZONS, kept separate so a change to one is a
# deliberate change to the other.
PAPER_TRADE_HORIZONS = frozenset({14, 30})

# Venues whose proceeds cannot be withdrawn. A return earned here is not cash, so it
# is reported on its own line and never summed into a cash P&L.
WALLED_VENUES = frozenset({"steam"})


# How much of the bid-ask spread a trade crosses.
#   round_trip: buy at the ask AND sell at the bid — the full spread.
#   sell_only:  you already hold the item (the buy is sunk), so you cross only
#               the sell side of the book — half the spread. The seller fee is
#               charged once in either mode, because ROUND_TRIP_COST is already
#               just the seller commission on the cash venues (see friction.py).
_SPREAD_CROSSINGS = {"round_trip": 1.0, "sell_only": 0.5}


def net_return(record: dict, venue: str = DEFAULT_VENUE, mode: str = "round_trip") -> float:
    """Net fractional return of one long trade, on the resolver basis.

    Buy at ``base_price``, sell at ``actual_price`` — both frozen resolver legs —
    then subtract the venue seller fee plus the spread crossed. ``mode`` selects
    how much spread: ``round_trip`` crosses the full spread, ``sell_only`` crosses
    half (the buy leg is inventory you already own). The served ``current_price``
    is intentionally not read here: the outcome's two legs must share one
    estimator.

    Raises ``KeyError`` on an unknown venue, tier or mode rather than falling back
    to a default, so no downstream number inherits a friction figure nobody chose.
    """
    base = record["base_price"]
    gross = (record["actual_price"] - base) / base
    friction = ROUND_TRIP_COST[venue] + SPREAD_BY_TIER[record["price_tier"]] * _SPREAD_CROSSINGS[mode]
    return gross - friction


def _quote(record: dict) -> float:
    """The price the forecast was QUOTED FROM — what a trader saw at decision time.

    ``current_price`` is the served snapshot; it falls back to ``base_price`` when
    the serving run wrote none (legacy rows and walkforward records have no separate
    quote). Used only to DECIDE a trade, never to price one — see the module
    docstring on the two bases.
    """
    served = record.get("current_price")
    if served is not None and served > 0:
        return float(served)
    return float(record["base_price"])


def _common_candidate(record: dict) -> bool:
    """Rows that can form a round trip at all: a positive base to buy at and a
    resolved actual to sell at."""
    base = record.get("base_price")
    return base is not None and base > 0 and record.get("actual_price") is not None


def _exceedance_trigger(record: dict, venue: str) -> bool:
    """Undervalued: the served quote sits below the forecast band's low bound."""
    return _quote(record) < record["predicted_low"]


def _q50_trigger(record: dict, venue: str) -> bool:
    """The predicted median move clears the round trip plus the tier spread — the
    same bar ``backtest/actionable.py`` conditions on, now priced as P&L."""
    quote = _quote(record)
    r_hat = (record["predicted_mid"] - quote) / quote
    return r_hat > actionable_threshold(record["price_tier"], venue)


# name -> (extra-candidate predicate, trigger predicate). The extra predicate is
# what a row needs beyond ``_common_candidate`` to be judgeable by the rule at all;
# a row missing it is not a candidate, so it must not enter the share denominator.
STRATEGIES = {
    "exceedance": (
        lambda r: r.get("predicted_low") is not None,
        _exceedance_trigger,
    ),
    "q50_clears": (
        lambda r: r.get("predicted_mid") is not None and _quote(r) > 0,
        _q50_trigger,
    ),
}

# The baseline every rule is judged against: buy every candidate indiscriminately.
_BASELINE = "buy_and_hold"


def _empty(strategy: str, venue: str, scope: str, mode: str) -> dict:
    """Fixed key set, every measured field absent (None, not 0.0 — 'nothing to
    measure' is a different claim from 'measured zero')."""
    return {
        "pt_strategy": strategy,
        "pt_venue": venue,
        "pt_mode": mode,
        "pt_cashable": venue not in WALLED_VENUES,
        "pt_scope": scope,
        "pt_n": None,
        "pt_n_candidates": None,
        "pt_share_pct": None,
        "pt_mean_net_pct": None,
        "pt_total_net_pct": None,
        "pt_win_rate_pct": None,
        "pt_n_fresh": None,
        "pt_n_stale": None,
        "pt_n_stale_unknown": None,
        "pt_mean_net_fresh_pct": None,
        "pt_mean_net_ci_lower": None,
        "pt_mean_net_ci_upper": None,
        "pt_profitable": None,
    }


def _cluster_mean_ci(nets: list[float], dates: list, ci: int = BOOTSTRAP_CI) -> tuple:
    """Percentile CI for the mean net return, resampled by FORECAST DATE.

    Returns per date are correlated through the market-wide factor, so an iid
    bootstrap over rows under-disperses the interval — the same reasoning that
    makes ``paired_mde`` resample clusters, not rows. Fewer than two dates carries
    no between-date variance, so the interval is ``(None, None)``.
    """
    from collections import defaultdict

    by_date: dict = defaultdict(list)
    for net, day in zip(nets, dates):
        by_date[day].append(net)
    # Sort by date so the cluster order is fixed regardless of input row order:
    # the bootstrap RNG is seeded, so a permuted `clusters` maps the same drawn
    # indices onto different dates and returns a different CI on the same data.
    clusters = [np.array(by_date[day], dtype=float) for day in sorted(by_date)]
    if len(clusters) < 2:
        return None, None

    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    sums = np.array([c.sum() for c in clusters], dtype=float)
    counts = np.array([c.size for c in clusters], dtype=float)
    n_groups = len(clusters)
    stats = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_groups, size=n_groups)
        stats[i] = sums[idx].sum() / counts[idx].sum()

    alpha = (100 - ci) / 2
    lower = round(float(np.percentile(stats, alpha)) * 100, 4)
    upper = round(float(np.percentile(stats, 100 - alpha)) * 100, 4)
    return lower, upper


def strategy_metrics(
    records: list[dict],
    strategy: str,
    horizon_days: int | None,
    min_dates: int,
    venue: str = DEFAULT_VENUE,
    mode: str = "round_trip",
) -> dict:
    """Net-of-friction P&L for one strategy over one cohort, as flat keys.

    ``horizon_days`` outside :data:`PAPER_TRADE_HORIZONS` returns
    ``out_of_scope`` with every value None — a different state from a cohort that
    simply had no candidate rows (``no_candidates``) or a rule that selected none
    (``in_scope`` with ``pt_n == 0``). Read ``pt_scope`` before any figure.

    ``pt_share_pct`` divides by the CANDIDATE count, not ``len(records)``: a row
    that could never be judged by the rule (no band floor for exceedance, no base
    to trade) is not a denominator.

    Does not mutate ``records``.
    """
    if horizon_days not in PAPER_TRADE_HORIZONS:
        return _empty(strategy, venue, "out_of_scope", mode)

    if strategy == _BASELINE:
        extra, trigger = (lambda r: True), (lambda r, v: True)
    else:
        extra, trigger = STRATEGIES[strategy]

    candidates = [r for r in records if _common_candidate(r) and extra(r)]
    if not candidates:
        return _empty(strategy, venue, "no_candidates", mode)

    out = _empty(strategy, venue, "in_scope", mode)
    out["pt_n_candidates"] = len(candidates)

    selected = [r for r in candidates if trigger(r, venue)]
    out["pt_n"] = len(selected)
    out["pt_share_pct"] = round(len(selected) / len(candidates) * 100, 2)
    if not selected:
        return out

    nets = [net_return(r, venue, mode) for r in selected]
    dates = [r["forecast_date"] for r in selected]
    out["pt_mean_net_pct"] = round(sum(nets) / len(nets) * 100, 4)
    out["pt_total_net_pct"] = round(sum(nets) * 100, 4)
    out["pt_win_rate_pct"] = round(sum(n > 0 for n in nets) / len(nets) * 100, 2)

    # base_stale_run_days: 0 fresh, >0 stale, NULL unknown (never read as 0 —
    # every row resolved before 2026-08-08 carries NULL).
    fresh = [n for n, r in zip(nets, selected) if r.get("base_stale_run_days") == 0]
    out["pt_n_fresh"] = len(fresh)
    out["pt_n_stale"] = sum(1 for r in selected if (r.get("base_stale_run_days") or 0) > 0)
    out["pt_n_stale_unknown"] = sum(1 for r in selected if r.get("base_stale_run_days") is None)
    out["pt_mean_net_fresh_pct"] = round(sum(fresh) / len(fresh) * 100, 4) if fresh else None

    lower, upper = _cluster_mean_ci(nets, dates)
    out["pt_mean_net_ci_lower"] = lower
    out["pt_mean_net_ci_upper"] = upper
    out["pt_profitable"] = lower is not None and lower > 0
    return out


def all_strategies(
    records: list[dict],
    horizon_days: int | None,
    min_dates: int,
    venue: str = DEFAULT_VENUE,
    mode: str = "round_trip",
) -> dict:
    """Every rule plus the buy-and-hold baseline, keyed by strategy name.

    The baseline is what the rules are read against: a rule only earns its keep if
    its selected cohort beats indiscriminate buying, so the two are reported
    together and never in isolation.
    """
    names = [_BASELINE, *STRATEGIES]
    return {name: strategy_metrics(records, name, horizon_days, min_dates, venue, mode) for name in names}
