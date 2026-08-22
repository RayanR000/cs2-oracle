"""Paper-trading P&L: what the forecasts would have earned net of real venue friction.

An honest execution harness, built to be *able* to conclude "no cash edge" — the
range-forecaster stance (`AGENTS.md`) says direction is not a shippable claim, and
every trading-adjacent arm so far is CV-positive and serving-negative.

The rules encoded here are the repo's own verified microstructure
(`docs/research/2026-08-07-cs2-forecasting-research.md` section 11):

- P&L is a round trip: buy at ``base_price``, sell at ``actual_price``, both
  frozen resolver legs, minus the venue round trip plus the tier's bid-ask spread.
- Steam Wallet is walled — a Steam "profit" is not cashable, so it is flagged
  ``pt_cashable = False`` and must never fold into a cash return.
- Only h in {14, 30} is executable (7-day market lock + trade protection).

These tests pin the INSTRUMENT so its verdict is trustworthy when it arrives.
"""
from __future__ import annotations

from datetime import date

import pytest

from backtest.friction import ROUND_TRIP_COST, SPREAD_BY_TIER
from backtest.papertrade import (
    PAPER_TRADE_HORIZONS,
    STRATEGIES,
    WALLED_VENUES,
    all_strategies,
    net_return,
    strategy_metrics,
)

MIN_DATES = 20

# Fixed keys every payload must carry, so a consumer never branches on existence.
CORE_KEYS = {
    "pt_strategy", "pt_venue", "pt_cashable", "pt_scope",
    "pt_n", "pt_n_candidates", "pt_share_pct",
    "pt_mean_net_pct", "pt_total_net_pct", "pt_win_rate_pct",
    "pt_n_fresh", "pt_n_stale", "pt_n_stale_unknown", "pt_mean_net_fresh_pct",
    "pt_mean_net_ci_lower", "pt_mean_net_ci_upper", "pt_profitable",
}


def _record(tier=5, base=2000.0, gross=0.50, quote=None, low=None, day=1,
            item_id=1, stale=0, horizon=14):
    """One frozen outcome record with the round-trip legs as fractions of base."""
    return {
        "base_price": base,
        "actual_price": base * (1 + gross),
        "predicted_low": low,
        "predicted_mid": base * (1 + gross),
        "current_price": quote,
        "price_tier": tier,
        "horizon_days": horizon,
        "item_id": item_id,
        "forecast_date": date(2026, 7, day),
        "base_stale_run_days": stale,
    }


def test_net_return_charges_round_trip_and_the_tier_spread():
    """A +50% gross move at tier 5 on CSFloat keeps 50% minus 2% RT minus the
    5.2% tier-5 spread. Friction is subtracted, never ignored."""
    r = _record(tier=5, gross=0.50)
    expected = 0.50 - (ROUND_TRIP_COST["csfloat"] + SPREAD_BY_TIER[5])
    assert net_return(r, "csfloat") == pytest.approx(expected)


def test_sell_only_charges_half_the_spread_of_a_round_trip():
    """Selling inventory you already hold pays the seller fee plus only ONE side
    of the spread — the buy side is sunk. Round trip pays fee + full spread."""
    r = _record(tier=5, gross=0.50)
    round_trip = 0.50 - (ROUND_TRIP_COST["csfloat"] + SPREAD_BY_TIER[5])
    sell_only = 0.50 - (ROUND_TRIP_COST["csfloat"] + SPREAD_BY_TIER[5] / 2)
    assert net_return(r, "csfloat", mode="sell_only") == pytest.approx(sell_only)
    assert sell_only > round_trip  # strictly cheaper: the spread is the big cost


def test_net_return_defaults_to_the_round_trip():
    """Omitting mode keeps the round-trip cost, so existing callers are
    unchanged."""
    r = _record(tier=5, gross=0.50)
    assert net_return(r, "csfloat") == net_return(r, "csfloat", mode="round_trip")


def test_net_return_uses_resolver_legs_not_the_served_quote():
    """Both P&L legs stay on the resolver basis. A wildly different served
    ``current_price`` must not leak into the earned return — that split is the
    2026-08-11 basis invariant."""
    r = _record(tier=5, base=2000.0, gross=0.10, quote=5.0)  # absurd served quote
    expected = 0.10 - (ROUND_TRIP_COST["csfloat"] + SPREAD_BY_TIER[5])
    assert net_return(r, "csfloat") == pytest.approx(expected)


def test_steam_is_a_walled_venue():
    """Steam Wallet has no cash value; the harness must know steam is walled so
    a Steam return is never folded into a cash P&L."""
    assert "steam" in WALLED_VENUES
    assert "csfloat" not in WALLED_VENUES


def test_only_the_executable_horizons_are_in_scope():
    """7-day market lock + trade protection: nothing under ~8 days is tradable,
    so only h in {14, 30} may be simulated."""
    assert PAPER_TRADE_HORIZONS == frozenset({14, 30})


# --- strategy_metrics -------------------------------------------------------


def test_the_return_shape_is_fixed_and_flat():
    """A varying key set is the defect that forced a nested metrics store
    repo-wide; every value is a scalar or None, never a container. And the key
    set must not depend on the data."""
    out = strategy_metrics([_record()], "buy_and_hold", horizon_days=14,
                           min_dates=MIN_DATES)
    assert CORE_KEYS <= set(out)
    assert not any(isinstance(v, (dict, list)) for v in out.values())

    empty = strategy_metrics([], "buy_and_hold", horizon_days=14,
                             min_dates=MIN_DATES)
    assert set(out) == set(empty), "the key set must not depend on the data"


def test_horizon_outside_scope_reports_why_rather_than_zero():
    """h=7 is not executable, which is a different state from 'nothing traded'.
    The two must never read the same."""
    out = strategy_metrics([_record(horizon=7)], "buy_and_hold", horizon_days=7,
                           min_dates=MIN_DATES)
    assert out["pt_scope"] == "out_of_scope"
    assert out["pt_n"] is None


def test_no_candidates_is_distinct_from_a_rule_that_fired_on_nothing():
    """A cohort with no eligible rows (no candidate) and a cohort where the rule
    simply selected zero rows are different verdicts."""
    # Exceedance needs a band floor; a record without one is not even a candidate.
    no_band = strategy_metrics([_record(low=None)], "exceedance",
                               horizon_days=14, min_dates=MIN_DATES)
    assert no_band["pt_scope"] == "no_candidates"

    # A candidate exists but sits above its band floor, so the rule selects none.
    not_cheap = strategy_metrics(
        [_record(base=100.0, quote=100.0, low=90.0, tier=4)],
        "exceedance", horizon_days=14, min_dates=MIN_DATES)
    assert not_cheap["pt_scope"] == "in_scope"
    assert not_cheap["pt_n_candidates"] == 1
    assert not_cheap["pt_n"] == 0


def test_exceedance_selects_only_items_below_their_band_floor():
    """Undervalued = the quote a trader saw sits below the forecast band's low."""
    cheap = _record(base=100.0, quote=80.0, low=90.0, tier=4, item_id=1)
    rich = _record(base=100.0, quote=95.0, low=90.0, tier=4, item_id=2)
    out = strategy_metrics([cheap, rich], "exceedance", horizon_days=14,
                           min_dates=MIN_DATES)
    assert out["pt_n_candidates"] == 2
    assert out["pt_n"] == 1
    assert out["pt_share_pct"] == pytest.approx(50.0)


def test_the_decision_reads_the_served_quote_not_the_resolver_base():
    """Exceedance triggers on the quote the forecaster saw. If it read the
    resolver base instead, this record — base above the floor, served quote
    below it — would be missed."""
    r = _record(base=100.0, quote=80.0, low=90.0, tier=4)
    out = strategy_metrics([r], "exceedance", horizon_days=14, min_dates=MIN_DATES)
    assert out["pt_n"] == 1


def test_stale_anchors_are_bucketed_separately():
    """A frozen anchor's return is the MA artifact, not a market move; fresh,
    stale and unknown (NULL, pre-2026-08-08) rows are counted apart and the
    fresh-only mean is reported."""
    fresh = _record(gross=0.40, stale=0, item_id=1, day=1)
    stale = _record(gross=0.40, stale=5, item_id=2, day=2)
    unknown = _record(gross=0.40, stale=None, item_id=3, day=3)
    out = strategy_metrics([fresh, stale, unknown], "buy_and_hold",
                           horizon_days=14, min_dates=MIN_DATES)
    assert out["pt_n_fresh"] == 1
    assert out["pt_n_stale"] == 1
    assert out["pt_n_stale_unknown"] == 1
    # Fresh-only mean is the single fresh trade's net, in percent.
    expected = (0.40 - (ROUND_TRIP_COST["csfloat"] + SPREAD_BY_TIER[5])) * 100
    assert out["pt_mean_net_fresh_pct"] == pytest.approx(expected)


def test_steam_metrics_are_flagged_non_cashable():
    """A Steam payload must announce it is wallet-only so a consumer never sums
    it into a cash return; a cash venue is cashable."""
    steam = strategy_metrics([_record()], "buy_and_hold", horizon_days=14,
                             min_dates=MIN_DATES, venue="steam")
    cash = strategy_metrics([_record()], "buy_and_hold", horizon_days=14,
                            min_dates=MIN_DATES, venue="csfloat")
    assert steam["pt_cashable"] is False
    assert cash["pt_cashable"] is True


def test_a_profitable_cohort_has_a_ci_lower_above_zero():
    """Many trades, all clearing friction across distinct dates, resolve as
    profitable; a break-even cohort does not."""
    winners = [_record(gross=0.60, item_id=i, day=i) for i in range(1, 25)]
    win_out = strategy_metrics(winners, "buy_and_hold", horizon_days=14,
                               min_dates=MIN_DATES)
    assert win_out["pt_mean_net_ci_lower"] > 0
    assert win_out["pt_profitable"] is True

    # Gross exactly equal to friction => net ~0 => not distinguishable from zero.
    breakeven_gross = ROUND_TRIP_COST["csfloat"] + SPREAD_BY_TIER[5]
    flats = [_record(gross=breakeven_gross, item_id=i, day=i) for i in range(1, 25)]
    flat_out = strategy_metrics(flats, "buy_and_hold", horizon_days=14,
                                min_dates=MIN_DATES)
    assert flat_out["pt_profitable"] is False


def test_the_ci_is_invariant_to_input_row_order():
    """The CI bootstrap is seeded, so cluster order must be pinned: an unordered
    DB scan would otherwise return a different `pt_profitable` run to run."""
    recs = [_record(gross=0.1 + 0.05 * (i % 7), item_id=i, day=1 + (i % 9))
            for i in range(1, 40)]
    ordered = strategy_metrics(recs, "buy_and_hold", horizon_days=14,
                               min_dates=MIN_DATES)
    shuffled = strategy_metrics(list(reversed(recs)), "buy_and_hold",
                                horizon_days=14, min_dates=MIN_DATES)
    assert ordered["pt_mean_net_ci_lower"] == shuffled["pt_mean_net_ci_lower"]
    assert ordered["pt_mean_net_ci_upper"] == shuffled["pt_mean_net_ci_upper"]
    assert ordered["pt_profitable"] == shuffled["pt_profitable"]


def test_sell_only_mode_threads_through_to_the_metrics():
    """A sell-only cohort keeps more of the move than the same round-trip cohort,
    because it is charged half the spread."""
    recs = [_record(gross=0.30, item_id=i, day=i) for i in range(1, 25)]
    rt = strategy_metrics(recs, "buy_and_hold", horizon_days=14,
                          min_dates=MIN_DATES, mode="round_trip")
    so = strategy_metrics(recs, "buy_and_hold", horizon_days=14,
                          min_dates=MIN_DATES, mode="sell_only")
    assert so["pt_mean_net_pct"] > rt["pt_mean_net_pct"]
    # The gap is exactly half the tier-5 spread, in percent.
    assert so["pt_mean_net_pct"] - rt["pt_mean_net_pct"] == pytest.approx(
        SPREAD_BY_TIER[5] / 2 * 100)


def test_buy_and_hold_baseline_trades_every_candidate():
    """all_strategies runs the baseline plus each rule; buy-and-hold selects
    every candidate, which is what the rules are judged against."""
    recs = [_record(base=100.0, quote=95.0, low=90.0, tier=4, item_id=i, day=i)
            for i in range(1, 6)]
    out = all_strategies(recs, horizon_days=14, min_dates=MIN_DATES)
    assert "buy_and_hold" in out
    assert set(STRATEGIES) <= set(out)
    assert out["buy_and_hold"]["pt_n"] == 5
