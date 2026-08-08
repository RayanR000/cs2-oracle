"""ActionableDA: directional accuracy on the calls that clear their own friction.

The metric exists because a low error and zero utility are compatible. A 3%
predicted move is inside the bid-ask spread for five of six tiers, so most of
what the model says implies no trade at all — and no error metric can see that,
because friction is not in the loss.

Expected production result: failure, on ``n_actionable`` first. These tests pin
the INSTRUMENT, so that result is trustworthy when it arrives.
"""
from __future__ import annotations

from datetime import date

from backtest.actionable import ACTIONABLE_HORIZONS, actionable_metrics
from backtest.friction import actionable_threshold

MIN_DATES = 20

# Fixed keys every return must carry, so a consumer never branches on existence.
CORE_KEYS = {
    "actionable_scope", "actionable_venue", "actionable_n",
    "actionable_share_pct", "actionable_da", "actionable_e_net_pct",
}


def _record(tier=5, r_hat=0.50, r_act=0.50, day=1, item_id=1):
    """One record with the two return legs expressed as fractions of base."""
    base = 2000.0 if tier == 5 else 2.0
    return {
        "base_price": base,
        "predicted_mid": base * (1 + r_hat),
        "actual_price": base * (1 + r_act),
        "price_tier": tier,
        "horizon_days": 14,
        "item_id": item_id,
        "forecast_date": date(2026, 7, day),
        # Fields pesaran_timmermann reads. Not what this metric decides on — it
        # uses the raw signs — but the subset PT still needs them.
        "direction_correct": 1 if (r_act > 0) == (r_hat > 0) else 0,
        "predicted_direction": "up" if r_hat > 0 else "down",
        "actual_direction": "up" if r_act > 0 else "down",
    }


def test_the_return_shape_is_fixed_and_flat():
    """A varying key set is the defect that killed a nested metrics column
    store-wide; and every value must be a scalar or None, never a dict."""
    out = actionable_metrics([_record()], horizon_days=14, min_dates=MIN_DATES)
    assert CORE_KEYS <= set(out)
    assert any(k.startswith("actionable_pt_") for k in out)
    assert not any(isinstance(v, (dict, list)) for v in out.values())

    empty = actionable_metrics([], horizon_days=14, min_dates=MIN_DATES)
    assert set(out) == set(empty), "the key set must not depend on the data"


def test_horizons_outside_the_scope_report_why_rather_than_zero():
    """h=3 is a category error, not an empty result. 'We did not measure this'
    and 'nothing was actionable' must never read the same."""
    assert ACTIONABLE_HORIZONS == frozenset({14, 30})
    for h in (3, 7, None):
        out = actionable_metrics([_record()], horizon_days=h, min_dates=MIN_DATES)
        assert out["actionable_scope"] == "out_of_scope"
        assert out["actionable_n"] is None
        assert out["actionable_da"] is None
        assert out["actionable_e_net_pct"] is None


def test_in_scope_horizons_are_measured():
    for h in (14, 30):
        out = actionable_metrics([_record()], horizon_days=h, min_dates=MIN_DATES)
        assert out["actionable_scope"] == "in_scope"
        assert out["actionable_n"] == 1


def test_membership_flips_at_the_tier_threshold():
    """The threshold is per-tier, so the same predicted move is actionable on a
    liquid item and not on an illiquid one."""
    threshold = actionable_threshold(5)          # 0.020 + 0.052 = 0.072
    just_under = actionable_metrics(
        [_record(tier=5, r_hat=threshold - 0.001)], 14, MIN_DATES)
    just_over = actionable_metrics(
        [_record(tier=5, r_hat=threshold + 0.001)], 14, MIN_DATES)
    assert just_under["actionable_n"] == 0
    assert just_over["actionable_n"] == 1


def test_the_same_move_is_actionable_at_tier_5_and_not_at_tier_1():
    r_hat = 0.10   # clears 7.2% at tier 5, nowhere near 23.1% at tier 1
    assert actionable_metrics([_record(tier=5, r_hat=r_hat)], 14, MIN_DATES)["actionable_n"] == 1
    assert actionable_metrics([_record(tier=1, r_hat=r_hat)], 14, MIN_DATES)["actionable_n"] == 0


def test_direction_is_the_raw_sign_not_the_flat_band():
    """A +-0.5% flat band is meaningless on a move required to clear >=7.2%.
    A small realised rise against a large predicted rise is a HIT here, where
    direction_from_return would have labelled the outcome 'flat' and a miss."""
    out = actionable_metrics([_record(tier=5, r_hat=0.50, r_act=0.001)], 14, MIN_DATES)
    assert out["actionable_n"] == 1
    assert out["actionable_da"] == 100.0


def test_a_carried_forward_price_scores_as_a_miss():
    """r_act == 0 exactly is the archive carrying a price forward. sign(0)
    matches no sign, so it cannot be a correct call. A frozen price is not a
    prediction the model got right."""
    out = actionable_metrics([_record(tier=5, r_hat=0.50, r_act=0.0)], 14, MIN_DATES)
    assert out["actionable_n"] == 1
    assert out["actionable_da"] == 0.0


def test_share_is_the_actionable_fraction_of_the_whole_cohort():
    records = (
        [_record(tier=5, r_hat=0.50, item_id=i) for i in range(3)]
        + [_record(tier=5, r_hat=0.01, item_id=10 + i) for i in range(7)]
    )
    out = actionable_metrics(records, 14, MIN_DATES)
    assert out["actionable_n"] == 3
    assert out["actionable_share_pct"] == 30.0


def test_e_net_subtracts_the_threshold_from_the_signed_realised_return():
    """E[net] is what acting on every actionable call would have returned after
    friction: sign(r_hat) * r_act - threshold, in percent."""
    out = actionable_metrics([_record(tier=5, r_hat=0.50, r_act=0.50)], 14, MIN_DATES)
    expected = (0.50 - actionable_threshold(5)) * 100
    assert abs(out["actionable_e_net_pct"] - expected) < 1e-6


def test_e_net_is_negative_when_the_direction_is_wrong():
    out = actionable_metrics([_record(tier=5, r_hat=0.50, r_act=-0.30)], 14, MIN_DATES)
    assert out["actionable_da"] == 0.0
    assert out["actionable_e_net_pct"] < 0


def test_a_correct_call_smaller_than_its_friction_still_loses_money():
    """The whole point of the metric. Right direction, real move, net negative."""
    out = actionable_metrics([_record(tier=5, r_hat=0.50, r_act=0.03)], 14, MIN_DATES)
    assert out["actionable_da"] == 100.0
    assert out["actionable_e_net_pct"] < 0


def test_no_actionable_rows_yields_none_not_zero():
    """0.0 would read as 'the model scored nothing'; None says 'no subset'."""
    out = actionable_metrics([_record(tier=1, r_hat=0.01)], 14, MIN_DATES)
    assert out["actionable_n"] == 0
    assert out["actionable_share_pct"] == 0.0
    assert out["actionable_da"] is None
    assert out["actionable_e_net_pct"] is None


def test_records_with_no_prediction_leg_are_reported_not_silently_dropped():
    """Harness records built before predicted_mid existed cannot be scored.
    That is a different state from 'nothing was actionable'."""
    legacy = {k: v for k, v in _record().items() if k != "predicted_mid"}
    out = actionable_metrics([legacy], 14, MIN_DATES)
    assert out["actionable_scope"] == "no_prediction_leg"
    assert out["actionable_n"] is None


def test_a_non_positive_base_cannot_form_a_return_and_is_excluded():
    bad = _record()
    bad["base_price"] = 0.0
    out = actionable_metrics([bad, _record(item_id=2, day=2)], 14, MIN_DATES)
    assert out["actionable_n"] == 1
    assert out["actionable_share_pct"] == 100.0


def test_pt_runs_on_the_actionable_subset_only():
    """PT on the subset, prefixed. One date cannot form a standard error, so a
    thin subset must report insufficient_dates rather than a bare point
    estimate — the exact failure the PT headline exists to retire."""
    records = [_record(tier=5, r_hat=0.50, day=1, item_id=i) for i in range(30)]
    out = actionable_metrics(records, 14, MIN_DATES)
    assert out["actionable_pt_verdict"] == "insufficient_dates"
    assert out["actionable_pt_n_dates"] == 1
    assert "actionable_pt_t_stat" in out


def test_venue_is_recorded_so_a_threshold_change_is_visible_in_the_data():
    out = actionable_metrics([_record()], 14, MIN_DATES)
    assert out["actionable_venue"] == "csfloat"
    harsh = actionable_metrics([_record(tier=5, r_hat=0.10)], 14, MIN_DATES, venue="steam")
    assert harsh["actionable_venue"] == "steam"
    assert harsh["actionable_n"] == 0   # 0.10 < 0.161 + 0.052


def test_the_metric_does_not_mutate_its_input():
    records = [_record()]
    snapshot = [dict(r) for r in records]
    actionable_metrics(records, 14, MIN_DATES)
    assert records == snapshot
