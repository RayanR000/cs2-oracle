# Friction-Conditioned Tier Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `price_tier` above $100, add a friction-conditioned `ActionableDA` metric at
h ∈ {14, 30}, and sweep the headline at $1/$5/$20 floors — so a reported accuracy number is
never pooled across liquidity populations and never implies a trade that the round trip eats.

**Architecture:** Two new pure modules under `backend/backtest/` — `friction.py` (constants
only) and `actionable.py` (one function) — mirroring how `directional_test.py` already holds
`pesaran_timmermann` while `scoring.py` merely calls it. `scoring.py` gains one tier cut, three
floor sentinels, and one call. Nothing reads the archive: every input is a column already
frozen on `forecast_outcomes`, which is what keeps `--rescore` archive-free.

**Tech Stack:** Python 3.13 local / 3.11 CI, pytest, numpy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-07-friction-conditioned-tier-scoring-design.md`

## Global Constraints

- Run everything from `backend/` through `venv/bin/python`. `backend/.env` points at
  **production** Supabase — these tasks touch no DB, but never run a script from `backend/`
  without checking that first.
- `backend/backtest/*.py` is **pure**: no I/O, no archive access, no clock. Keep it that way.
- Nested dict/list values in a `metrics` payload serialise to JSON text store-wide
  (`db/parquet.py::_jsonify_nested`). **New metric keys must be flat**, `None`-filled where
  not computable — the shape must not vary between cohorts.
- Round-trip costs, verbatim: CSFloat **2.0%**, DMarket **2.0%**, Skinport **8.7%**,
  Steam **16.1%**.
- Measured median bid–ask spread by band, n = 22,449, verbatim: `<$1` **35.5%**,
  `$1–10` **21.1%**, `$10–50` **17.3%**, `$50–500` **10.8%**, `$1000+` **5.2%**.
- `ACTIONABLE_HORIZONS = frozenset({14, 30})`. An actionable metric at h=3 is a category error.
- `HEADLINE_TIER` stays `-1` and stays meaning `≥ $1`. `MIN_SERVED_PRICE_USD` does **not**
  move in this plan.
- Targeted test runs only: `venv/bin/python -m pytest tests/test_<name>.py -q`. A bare
  `pytest -q` collects `scripts/test_social_signal.py` and aborts on a missing `thefuzz`.

---

### Task 1: `friction.py` — round-trip and spread constants

**Files:**
- Create: `backend/backtest/friction.py`
- Test: `backend/tests/test_friction.py`

**Interfaces:**
- Consumes: `price_tier` from `backtest.scoring` (test-side only, to assert coverage).
- Produces: `ROUND_TRIP_COST: dict[str, float]`, `DEFAULT_VENUE: str`,
  `SPREAD_BY_TIER: dict[int, float]`, `SPREAD_SOURCE_BAND: dict[int, str]`,
  `actionable_threshold(price_tier: int, venue: str = DEFAULT_VENUE) -> float`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_friction.py`:

```python
"""Friction constants: the round trip and the spread a call has to clear.

These are the numbers that decide whether a forecast implies a trade at all. The
spread table is a STATED APPROXIMATION - the measurement's bands do not line up
with price_tier's cuts - so the tests pin the mapping RULE, not just the values,
because a silently re-pointed band would move every actionable number.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.friction import (  # noqa: E402
    DEFAULT_VENUE,
    ROUND_TRIP_COST,
    SPREAD_BY_TIER,
    SPREAD_SOURCE_BAND,
    actionable_threshold,
)
from backtest.scoring import price_tier  # noqa: E402


def test_round_trip_costs_are_the_measured_venue_figures():
    assert ROUND_TRIP_COST["csfloat"] == 0.020
    assert ROUND_TRIP_COST["dmarket"] == 0.020
    assert ROUND_TRIP_COST["skinport"] == 0.087
    assert ROUND_TRIP_COST["steam"] == 0.161


def test_csfloat_is_the_default_because_it_is_the_cheapest_round_trip():
    assert DEFAULT_VENUE == "csfloat"
    assert ROUND_TRIP_COST[DEFAULT_VENUE] == min(ROUND_TRIP_COST.values())


def test_every_price_tier_has_a_spread_entry():
    """A new tier without a spread must fail here, not KeyError at score time."""
    tiers = {price_tier(p) for p in (0.5, 1, 5, 20, 100, 1000, 50_000)}
    assert tiers == set(SPREAD_BY_TIER), "price_tier and SPREAD_BY_TIER disagree"
    assert set(SPREAD_BY_TIER) == set(SPREAD_SOURCE_BAND)


def test_spread_is_monotone_decreasing_in_price():
    """35.5% sub-$1 down to 5.2% at $1000+. Expensive items are the liquid ones."""
    values = [SPREAD_BY_TIER[t] for t in sorted(SPREAD_BY_TIER)]
    assert values == sorted(values, reverse=True)


def test_spread_values_are_the_measured_band_medians():
    assert SPREAD_BY_TIER[0] == 0.355   # <$1
    assert SPREAD_BY_TIER[1] == 0.211   # $1-10
    assert SPREAD_BY_TIER[2] == 0.173   # $10-50
    assert SPREAD_BY_TIER[3] == 0.173   # $10-50
    assert SPREAD_BY_TIER[4] == 0.108   # $50-500
    assert SPREAD_BY_TIER[5] == 0.052   # $1000+


def test_each_tier_borrowed_its_nearest_source_band_by_log_geometric_midpoint():
    """The mapping RULE, reproduced independently of the table.

    The source bands are not the tier cuts, so each tier takes the band whose
    geometric midpoint is nearest its own in log price. Pinning this stops a
    later editor from re-pointing a band by eye.
    """
    import math

    source_mid = {
        "<$1": 0.5,          # open at the bottom; $0.03-$1 midpoint is ~0.17,
                             # but the band is dominated by its top decade
        "$1-10": math.sqrt(1 * 10),
        "$10-50": math.sqrt(10 * 50),
        "$50-500": math.sqrt(50 * 500),
        "$1000+": 2000.0,    # open at the top
    }
    tier_mid = {
        0: 0.5, 1: math.sqrt(1 * 5), 2: math.sqrt(5 * 20),
        3: math.sqrt(20 * 100), 4: math.sqrt(100 * 1000), 5: 2000.0,
    }
    for tier, mid in tier_mid.items():
        nearest = min(source_mid, key=lambda b: abs(math.log(source_mid[b]) - math.log(mid)))
        assert SPREAD_SOURCE_BAND[tier] == nearest, f"tier {tier} borrowed the wrong band"


def test_actionable_threshold_is_the_round_trip_plus_the_tier_spread():
    assert actionable_threshold(5) == pytest.approx(0.020 + 0.052)
    assert actionable_threshold(1) == pytest.approx(0.020 + 0.211)
    assert actionable_threshold(5, venue="steam") == pytest.approx(0.161 + 0.052)


def test_the_cheapest_actionable_bar_is_still_over_seven_percent():
    """The point of the whole metric: even the most liquid tier at the cheapest
    venue needs a >7% predicted move before a call implies a trade."""
    assert min(actionable_threshold(t) for t in SPREAD_BY_TIER) > 0.07


def test_unknown_venue_raises_rather_than_defaulting():
    with pytest.raises(KeyError):
        actionable_threshold(1, venue="buff163")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && venv/bin/python -m pytest tests/test_friction.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.friction'`

- [ ] **Step 3: Write the implementation**

Create `backend/backtest/friction.py`:

```python
"""What a predicted move has to clear before it implies a trade.

Two constants, no logic beyond adding them. Kept separate from ``scoring`` so
the numbers have one home and a citation: a threshold silently edited inside a
metric function is a number nobody can audit.

**Both tables are measurements, and the spread table is a measurement AT THE
WRONG CUTS.** The bid-ask spread was measured on BUFF163's paired
``starting_at`` / ``highest_order`` over n = 22,449 items in bands
(<$1, $1-10, $10-50, $50-500, $1000+), which do not line up with
``scoring.price_tier``'s cuts (1, 5, 20, 100, 1000). Each tier therefore
borrows the band whose geometric midpoint is nearest its own in log price, and
``SPREAD_SOURCE_BAND`` records which one it borrowed. Do not read these as
measured per tier.

The honest replacement is a per-item ``buff_spread_rel`` frozen onto
``forecast_outcomes`` at resolution time. It is not built yet: the bid feed
starts 2026-07-11, so it would be NULL on almost every stored outcome, and this
module reads nothing but its own constants by design.
"""

from __future__ import annotations

# Round trip to buy and sell one item at each venue, as a fraction of price.
# Steam's 16.1% is the fee stack; the cash venues are the seller commission.
# Sourced from docs/research/2026-08-07-cs2-forecasting-research.md section 11.
ROUND_TRIP_COST = {
    "csfloat": 0.020,
    "dmarket": 0.020,
    "skinport": 0.087,
    "steam": 0.161,
}

# The cheapest round trip, so the actionable metric is reported against the most
# favourable venue available. Any harsher venue only shrinks n_actionable, and a
# metric that fails at the best venue fails everywhere.
DEFAULT_VENUE = "csfloat"

# Median relative bid-ask spread, by price_tier. See the module docstring: the
# values are measured, the tier assignment is a nearest-band rule.
#
# Spread tightens MONOTONICALLY with price - 35.5% sub-$1 to 5.2% at $1000+ -
# which inverts retail intuition: expensive items are the liquid ones, and
# sub-$1 items are midpoints of a book nobody could transact in. This is also
# the reason a DA pooled across tiers is uninterpretable.
SPREAD_BY_TIER = {
    0: 0.355,   # < $1
    1: 0.211,   # $1 - 5
    2: 0.173,   # $5 - 20
    3: 0.173,   # $20 - 100
    4: 0.108,   # $100 - 1000
    5: 0.052,   # >= $1000
}

# Which measured band each tier's spread came from. Present so the borrowing is
# visible in the data rather than buried in a comment, and so a test can pin the
# mapping rule independently of the values above.
SPREAD_SOURCE_BAND = {
    0: "<$1",
    1: "$1-10",
    2: "$10-50",
    3: "$10-50",
    4: "$50-500",
    5: "$1000+",
}


def actionable_threshold(price_tier: int, venue: str = DEFAULT_VENUE) -> float:
    """The ``|r_hat|`` a forecast must exceed to imply a trade, as a fraction.

    ``RT_venue + s_i``: the round trip plus the spread you cross to take it. At
    CSFloat this runs 37.5% at tier 0 down to 7.2% at tier 5 - so even the most
    liquid cohort at the cheapest venue needs a >7% predicted move.

    Raises KeyError on an unknown venue or tier rather than falling back to a
    default. A silent fallback here would report a threshold nobody chose, and
    every actionable number downstream would inherit it.
    """
    return ROUND_TRIP_COST[venue] + SPREAD_BY_TIER[price_tier]
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_friction.py -q`
Expected: all PASS except `test_every_price_tier_has_a_spread_entry`, which FAILS —
`price_tier` has no tier 5 yet, so the tier set is `{0,1,2,3,4}` against
`SPREAD_BY_TIER`'s `{0,1,2,3,4,5}`. Task 2 closes it. Leave it failing and do not
weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add backend/backtest/friction.py backend/tests/test_friction.py
git commit -m "feat: add round-trip and spread friction constants"
```

---

### Task 2: The sixth price band

**Files:**
- Modify: `backend/backtest/scoring.py:55-64` (`price_tier`)
- Modify: `backend/tests/test_backtest_scoring.py:108-113` (`test_price_tier_boundaries`)
- Modify: `backend/tests/test_parquet_nested_columns.py:211-219`

**Interfaces:**
- Consumes: nothing.
- Produces: `price_tier(p) == 5` for `p >= 1000`; tier 4 now means `$100–1000`.

- [ ] **Step 1: Write the failing test**

Replace `test_price_tier_boundaries` in `backend/tests/test_backtest_scoring.py`:

```python
def test_price_tier_boundaries():
    assert price_tier(0.99) == 0
    assert price_tier(1.0) == 1
    assert price_tier(5.0) == 2
    assert price_tier(20.0) == 3
    assert price_tier(100.0) == 4
    assert price_tier(999.99) == 4
    assert price_tier(1000.0) == 5
    assert price_tier(29_685.0) == 5   # the priciest name in the archive


def test_tier_4_no_longer_merges_the_two_most_liquid_cohorts():
    """The reason the cut exists: tier 4 used to hold both the 10.8%-spread
    ($50-500) and the 5.2%-spread ($1000+) populations, which are the two most
    DIFFERENT liquidity cohorts in the market. Pooling them made every tier-4
    number uninterpretable.

    Note for anyone reading the stored series: rows written before this change
    with price_tier == 4 mean >= $100, not $100-1000.
    """
    assert price_tier(200.0) == 4
    assert price_tier(2000.0) == 5
    assert price_tier(200.0) != price_tier(2000.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && venv/bin/python -m pytest tests/test_backtest_scoring.py -q -k price_tier`
Expected: FAIL — `assert price_tier(1000.0) == 5` gets `4`.

- [ ] **Step 3: Write the implementation**

In `backend/backtest/scoring.py`, replace `price_tier`:

```python
def price_tier(price: float) -> int:
    """Liquidity band, not a display bucket.

    The $1000 cut exists because the bid-ask spread is 10.8% at $50-500 and
    5.2% at $1000+ (n = 22,449) - the two most different liquidity populations
    in the market, which tier 4 used to merge. See backtest/friction.py.

    Rows stored before 2026-08-07 with price_tier == 4 mean >= $100. Tiers 0-3
    are unchanged, so HEADLINE_MIN_TIER and MIN_SERVED_PRICE_USD are unaffected.
    """
    if price >= 1000:
        return 5
    if price >= 100:
        return 4
    if price >= 20:
        return 3
    if price >= 5:
        return 2
    if price >= 1:
        return 1
    return 0
```

- [ ] **Step 4: Fix the hardcoded fan-out assertion**

In `backend/tests/test_parquet_nested_columns.py`, replace
`test_the_whole_tier_fanout_is_stable_across_runs` (lines 211-219):

```python
    def test_the_whole_tier_fanout_is_stable_across_runs(self, tmp_path):
        """score_by_tier's whole fan-out must dedup to itself on a re-run.

        Derived from FLOOR_SWEEP rather than hardcoded: the row count moved from
        7 to 10 when the $1000 band and the $5/$20 floor sentinels landed, and a
        literal here just breaks on the next legitimate cut.
        """
        from backtest.scoring import FLOOR_SWEEP

        path = tmp_path / "prediction_accuracy.parquet"
        tiers = [0, 1, 2, 3, 4, 5, *sorted(FLOOR_SWEEP), None]
        rows = pd.DataFrame([
            {"prediction_type": "forecast", "horizon_days": 7, "price_tier": t,
             "metrics": {"mae": 1.0}}
            for t in tiers
        ])
        _append_parquet(path, rows, KEYS)
        _append_parquet(path, rows, KEYS)
        assert len(pd.read_parquet(path)) == len(tiers)
```

This imports `FLOOR_SWEEP`, which Task 5 creates. It will fail until then — that is
expected and is why the two tasks are adjacent.

- [ ] **Step 5: Run the affected suites**

Run:
```bash
cd backend && venv/bin/python -m pytest tests/test_backtest_scoring.py \
  tests/test_friction.py tests/test_serving_policy.py tests/test_cv_cohort_parity.py \
  tests/test_served_cohort_weighting.py -q
```
Expected: `test_friction.py` now fully PASSES (tier 5 exists). `test_serving_policy.py`
PASSES **untouched** — it is the guard that the served population still equals the headline
population, so if it fails, the tier change leaked into the serving floor and must be undone.
`test_parquet_nested_columns.py` is expected to fail on the `FLOOR_SWEEP` import until Task 5.

- [ ] **Step 6: Commit**

```bash
git add backend/backtest/scoring.py backend/tests/test_backtest_scoring.py \
  backend/tests/test_parquet_nested_columns.py
git commit -m "feat: split price_tier above \$100 into \$100-1000 and \$1000+"
```

---

### Task 3: `actionable.py` — the friction-conditioned metric

**Files:**
- Create: `backend/backtest/actionable.py`
- Test: `backend/tests/test_actionable_da.py`

**Interfaces:**
- Consumes: `actionable_threshold`, `DEFAULT_VENUE` from `backtest.friction`;
  `pesaran_timmermann` from `backtest.directional_test`.
- Produces: `ACTIONABLE_HORIZONS: frozenset[int]`,
  `actionable_metrics(records: list[dict], horizon_days: int | None, min_dates: int, venue: str = DEFAULT_VENUE) -> dict`.
  Returns a **fixed-shape flat dict**: `actionable_scope`, `actionable_venue`,
  `actionable_n`, `actionable_share_pct`, `actionable_da`, `actionable_e_net_pct`, plus every
  key `pesaran_timmermann` returns re-prefixed `pt_` → `actionable_pt_`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_actionable_da.py`:

```python
"""ActionableDA: directional accuracy on the calls that clear their own friction.

The metric exists because a low error and zero utility are compatible. A 3%
predicted move is inside the bid-ask spread for five of six tiers, so most of
what the model says implies no trade at all - and no error metric can see that,
because friction is not in the loss.

Expected production result: failure, on n_actionable first. These tests pin the
INSTRUMENT, so that result is trustworthy when it arrives.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.actionable import ACTIONABLE_HORIZONS, actionable_metrics  # noqa: E402
from backtest.friction import actionable_threshold  # noqa: E402

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
        # Fields pesaran_timmermann reads. Not what this metric decides on - it
        # uses the raw signs - but the subset PT still needs them.
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
    assert out["actionable_share_pct"] == 50.0


def test_pt_runs_on_the_actionable_subset_only():
    """PT on the subset, prefixed. One date cannot form a standard error, so a
    thin subset must report insufficient_dates rather than a bare point
    estimate - the exact failure the PT headline exists to retire."""
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && venv/bin/python -m pytest tests/test_actionable_da.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.actionable'`

- [ ] **Step 3: Write the implementation**

Create `backend/backtest/actionable.py`:

```python
"""Directional accuracy on the calls that clear their own friction.

Every error metric this repo reports is blind to the same thing: a forecast can
be accurate and imply no trade. Round trip is +2.0% at CSFloat and +16.1% on
Steam, and the bid-ask spread runs 35.5% sub-$1 to 5.2% at $1000+ - so a 3%
predicted move is inside the spread for five of six tiers. Friction is not in
the loss, so no loss can see it.

    ActionableDA(v, h) = P( sign(r_act) = sign(r_hat) | |r_hat| > RT_v + s_i )

Reported as FOUR numbers together, because any one of them alone misleads:
n_actionable / n_total, ActionableDA, E[net], and PT on the subset. A 60%
ActionableDA over 11 rows is not a finding.

Pure: no I/O, no clock. Reads only ``base_price``, ``predicted_mid``,
``actual_price`` and ``price_tier`` off records, all of which are frozen
columns on ``forecast_outcomes``.
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


def _empty(scope: str, venue: str) -> dict:
    """The fixed key set, with every measured field absent.

    The shape must not depend on the data: a metrics payload whose keys vary by
    cohort is what forced every nested value in this store to JSON text. And
    None rather than 0.0 throughout - a zero here reads as "the model scored
    nothing", which is a different claim from "there is no subset to score".
    """
    out = {
        "actionable_scope": scope,
        "actionable_venue": venue,
        "actionable_n": None,
        "actionable_share_pct": None,
        "actionable_da": None,
        "actionable_e_net_pct": None,
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

    ``horizon_days`` outside ``ACTIONABLE_HORIZONS`` - including None, which is
    what a record predating the field gives - returns ``out_of_scope`` with
    every value None. That is not the same state as an empty subset, and the
    two must never be reported identically.

    Does not mutate ``records``.
    """
    if horizon_days not in ACTIONABLE_HORIZONS:
        return _empty("out_of_scope", venue)

    usable = [
        r for r in records
        if r.get("predicted_mid") is not None and (r.get("base_price") or 0) > 0
    ]
    if records and not usable:
        # Distinguishable from "nothing cleared the threshold": these rows carry
        # no prediction leg at all, so the metric was never computable on them.
        return _empty("no_prediction_leg", venue)

    subset = []
    for r in usable:
        base = r["base_price"]
        r_hat = (r["predicted_mid"] - base) / base
        if abs(r_hat) > actionable_threshold(r["price_tier"], venue):
            subset.append((r, r_hat, (r["actual_price"] - base) / base))

    out = _empty("in_scope", venue)
    out["actionable_n"] = len(subset)
    out["actionable_share_pct"] = (
        round(len(subset) / len(usable) * 100, 2) if usable else 0.0
    )
    if not subset:
        return out

    hits = sum(1 for _, r_hat, r_act in subset if _sign(r_act) == _sign(r_hat))
    net = [
        _sign(r_hat) * r_act - actionable_threshold(r["price_tier"], venue)
        for r, r_hat, r_act in subset
    ]

    out["actionable_da"] = round(hits / len(subset) * 100, 2)
    out["actionable_e_net_pct"] = round(sum(net) / len(net) * 100, 4)
    out.update({
        f"actionable_{k}": v
        for k, v in pesaran_timmermann([r for r, _, _ in subset], min_dates).items()
    })
    return out
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && venv/bin/python -m pytest tests/test_actionable_da.py -q`
Expected: PASS (all 17).

- [ ] **Step 5: Commit**

```bash
git add backend/backtest/actionable.py backend/tests/test_actionable_da.py
git commit -m "feat: add friction-conditioned ActionableDA at h in {14,30}"
```

---

### Task 4: Carry `predicted_mid` and `horizon_days` on every record

**Files:**
- Modify: `backend/scripts/backtest_accuracy.py:601-617` (`_records_from_frozen_outcomes`)
- Modify: `backend/backtest/walkforward_records.py:19-36` (signature) and `:97-122` (record dict)
- Test: `backend/tests/test_backtest_scoring.py`, `backend/tests/test_walkforward_records.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: every scoring record carries `predicted_mid: float` and `horizon_days: int | None`.
  `fold_records(...)` gains a keyword parameter `horizon_days: int | None = None`, placed
  **after** `fold_id` so existing positional calls are unaffected.

Note: the fresh-resolution path in `backtest_accuracy.py` writes `forecast_outcomes` rows and
then scores through `_records_from_frozen_outcomes`, so there is only **one** record builder to
change in that file. Verify this before editing: `grep -n "price_tier(base)" backend/scripts/backtest_accuracy.py`
should return exactly one hit.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_backtest_scoring.py`:

```python
def test_frozen_outcome_records_carry_the_prediction_leg_and_the_horizon():
    """ActionableDA needs r_hat, so the record needs predicted_mid; and it is
    scoped by horizon, so the record needs the horizon.

    Both ride ON the record rather than as score_cohort parameters: records are
    grouped by (horizon, model_version) so every record in a cohort shares the
    horizon, and eight test modules call score_cohort(records) positionally.
    """
    from scripts import backtest_accuracy

    session = _seed_session_with_frozen_outcomes()   # existing fixture helper
    groups = backtest_accuracy._records_from_frozen_outcomes(session)
    records = [r for rs in groups.values() for r in rs]
    assert records, "fixture produced no records"
    for r in records:
        assert r["predicted_mid"] is not None
        assert r["horizon_days"] in (3, 7, 14, 30)
```

Before writing this, find the existing fixture in `tests/test_backtest_scoring.py` that
already seeds frozen outcomes (`grep -n "_records_from_frozen_outcomes\|frozen_outcome" tests/test_backtest_scoring.py`)
and call that instead of `_seed_session_with_frozen_outcomes` if the name differs. Do not add
a second fixture.

Append to `backend/tests/test_walkforward_records.py`:

```python
def test_fold_records_carry_the_prediction_leg_and_an_optional_horizon():
    """The walkforward gate scores through score_cohort, so its records need the
    same two fields - otherwise the gate silently reports out_of_scope forever."""
    records = _fold_records(n=4, horizon_days=14)   # existing helper + new kwarg
    for r in records:
        assert r["predicted_mid"] > 0
        assert r["horizon_days"] == 14


def test_fold_records_horizon_defaults_to_none_for_existing_callers():
    records = _fold_records(n=4)
    assert all(r["horizon_days"] is None for r in records)
```

Adapt `_fold_records` to whatever the module's existing record-building helper is called.

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd backend && venv/bin/python -m pytest tests/test_backtest_scoring.py \
  tests/test_walkforward_records.py -q -k "prediction_leg"
```
Expected: FAIL with `KeyError: 'predicted_mid'`.

- [ ] **Step 3: Write the implementation**

In `backend/scripts/backtest_accuracy.py`, inside the record dict appended in
`_records_from_frozen_outcomes` (currently ending at the `"forecast_date"` key), add:

```python
            # The prediction leg, for the friction-conditioned metric:
            # r_hat = (predicted_mid - base_price) / base_price. A frozen
            # column, so this stays archive-free and --rescore keeps working.
            "predicted_mid": mid,
            # ActionableDA is scoped to h in {14, 30}. Carried on the record
            # rather than passed into score_cohort: the grouping key already
            # fixes it per cohort, and eight test modules call score_cohort
            # positionally.
            "horizon_days": r.horizon_days,
```

In `backend/backtest/walkforward_records.py`, add the parameter after `fold_id`:

```python
    fold_id=None,
    horizon_days=None,
```

and document it in the docstring:

```
    `horizon_days` is the forecast horizon these rows were built for. It is
    optional and defaults to None, which scores as out_of_scope for the
    friction-conditioned metric - correct for a caller that has not said which
    horizon it is measuring, and wrong to guess at.
```

then add to the appended record dict, beside `"actual_price"`:

```python
            "predicted_mid": float(mid[i]),
            "horizon_days": horizon_days,
```

- [ ] **Step 4: Run the tests**

Run:
```bash
cd backend && venv/bin/python -m pytest tests/test_backtest_scoring.py \
  tests/test_walkforward_records.py tests/test_walkforward_gate.py \
  tests/test_backtest_resolution.py -q
```
Expected: PASS, except `test_parquet_nested_columns.py`'s `FLOOR_SWEEP` import (Task 5).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/backtest_accuracy.py backend/backtest/walkforward_records.py \
  backend/tests/test_backtest_scoring.py backend/tests/test_walkforward_records.py
git commit -m "feat: carry predicted_mid and horizon_days on scoring records"
```

---

### Task 5: Wire the metric in, and sweep the headline at $1/$5/$20

**Files:**
- Modify: `backend/backtest/scoring.py` — `score_cohort` (call the metric), `HEADLINE_TIER`
  block (add `FLOOR_SWEEP`), `score_by_tier`, `headline_records` (add `floor_records`)
- Modify: `backend/scripts/backtest_accuracy.py:697-771` (`_headline_line`) and `:774-817`
  (`_score_groups`)
- Modify: `backend/scripts/walkforward_backtest.py:266` (tier label map)
- Test: `backend/tests/test_backtest_scoring.py`, `backend/tests/test_accuracy_tier_mirror.py`

**Interfaces:**
- Consumes: `actionable_metrics`, `ACTIONABLE_HORIZONS` from `backtest.actionable`.
- Produces: `FLOOR_SWEEP: dict[int, float]` = `{-1: 1.0, -2: 5.0, -3: 20.0}`;
  `floor_records(records: list[dict], floor: float) -> list[dict]`;
  `score_by_tier` emits one row per present band, one per floor sentinel, and the `None`
  aggregate. `score_cohort`'s metrics dict gains every `actionable_*` key.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_backtest_scoring.py`:

```python
def test_score_cohort_publishes_the_actionable_metric():
    records = [
        _record(item_id=i, base_price=2000.0, actual_price=3000.0,
                predicted_mid=3000.0, price_tier=5, horizon_days=14)
        for i in range(20)
    ]
    metrics, _ = score_cohort(records)
    assert metrics["actionable_scope"] == "in_scope"
    assert metrics["actionable_n"] == 20
    assert metrics["actionable_da"] == 100.0


def test_score_cohort_marks_short_horizons_out_of_scope():
    """A cohort at h=3 must say why the metric is absent, not report zeros."""
    records = [_record(item_id=i, horizon_days=3) for i in range(20)]
    metrics, _ = score_cohort(records)
    assert metrics["actionable_scope"] == "out_of_scope"
    assert metrics["actionable_n"] is None


def test_the_actionable_keys_are_present_on_every_cohort():
    """Fixed shape. A key set that varies by cohort is the defect that forced
    every nested metrics value in this store to JSON text."""
    in_scope, _ = score_cohort([_record(item_id=i, horizon_days=14) for i in range(20)])
    out_of, _ = score_cohort([_record(item_id=i, horizon_days=3) for i in range(20)])
    actionable = {k for k in in_scope if k.startswith("actionable_")}
    assert actionable
    assert actionable == {k for k in out_of if k.startswith("actionable_")}


def test_floor_sweep_emits_one_stored_row_per_floor():
    """The sweep answers 'where does the headline stabilise'. Stored, not just
    logged, for the reason HEADLINE_TIER is stored: a headline that exists only
    in console output cannot be audited or recomputed."""
    records = (
        [_record(item_id=i, base_price=2.0, price_tier=1) for i in range(10)]
        + [_record(item_id=20 + i, base_price=50.0, price_tier=3) for i in range(10)]
    )
    by_tier = {t: n for t, _, n in score_by_tier(records)}
    assert FLOOR_SWEEP == {-1: 1.0, -2: 5.0, -3: 20.0}
    assert by_tier[-1] == 20     # >= $1
    assert by_tier[-2] == 10     # >= $5
    assert by_tier[-3] == 10     # >= $20


def test_the_floors_nest():
    records = [
        _record(item_id=i, base_price=base, price_tier=price_tier(base))
        for i, base in enumerate([0.5, 2.0, 8.0, 50.0, 500.0, 5000.0] * 4)
    ]
    by_tier = {t: n for t, _, n in score_by_tier(records)}
    assert by_tier[-3] <= by_tier[-2] <= by_tier[-1]


def test_headline_tier_is_still_the_dollar_floor():
    """/accuracy/headline, the homepage placard and the stored series all key on
    HEADLINE_TIER. The sweep must not renumber it."""
    assert HEADLINE_TIER == -1
    assert FLOOR_SWEEP[HEADLINE_TIER] == 1.0


def test_a_floor_no_record_reaches_is_omitted_not_emitted_as_zero():
    records = [_record(item_id=i, base_price=2.0, price_tier=1) for i in range(10)]
    tiers = {t for t, _, _ in score_by_tier(records)}
    assert -1 in tiers
    assert -2 not in tiers and -3 not in tiers
```

Update the import at `tests/test_backtest_scoring.py:939` to add `FLOOR_SWEEP` and
`price_tier`, and extend the `_record` helper's defaults (around line 85) with
`"predicted_mid": 1.10` and `"horizon_days": 14` so the new overrides work.

Also update the tier-set assertion at `tests/test_backtest_scoring.py:1072`:

```python
    # A per-tier row (tier 1 for "ak", tier 2 for "awp"), the three floor
    # sentinels the fixture reaches, and the all-tiers aggregate.
    assert {None, 1, 2, HEADLINE_TIER, -2} == {t for (_, _, t) in normal_shape}
```

Run the fixture first and read the actual tier set out of the failure before committing to
`-2`: it depends on the fixture's prices, and the assertion must describe the fixture, not a
guess.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && venv/bin/python -m pytest tests/test_backtest_scoring.py -q -k "actionable or floor"`
Expected: FAIL — `ImportError: cannot import name 'FLOOR_SWEEP'`.

- [ ] **Step 3: Write the implementation**

In `backend/backtest/scoring.py`, add the import:

```python
from backtest.actionable import actionable_metrics
```

At the end of `score_cohort`, immediately before `metrics.update(pt)`:

```python
    # Friction-conditioned accuracy, on the subset whose predicted move clears
    # the round trip plus the tier's spread. Scoped to h in {14, 30}; outside
    # that it reports out_of_scope rather than zeros. See backtest/actionable.py
    # for why a low error and zero utility are compatible.
    actionable = actionable_metrics(
        records, records[0].get("horizon_days"), MIN_FORECAST_DATES
    )
```

and after `metrics.update(pt)`:

```python
    metrics.update(actionable)
```

Replace the `HEADLINE_TIER` block's trailing definition with the sweep:

```python
# price_tier sentinels for the headline floor sweep, mapping sentinel -> floor
# in USD. HEADLINE_TIER is -1 and keeps meaning >= $1: /accuracy/headline, the
# homepage placard and months of stored series all key on it, so the sweep
# extends downward from it rather than renumbering.
#
# The sweep exists because $1 is a CONVENTION, not a derivation - the spread
# evidence (35.5% sub-$1) argues the honest floor is above it. These three rows
# are what answers "where does the headline stabilise", and they are stored
# rather than logged for the same reason HEADLINE_TIER is: a headline that
# exists only in a run's console output cannot be audited or recomputed.
#
# Negative by construction so they can never collide with a band price_tier()
# returns, and the API's price_tier query bound is widened to -3 to match.
FLOOR_SWEEP = {
    -1: 1.0,
    -2: 5.0,
    -3: 20.0,
}
```

Rewrite `score_by_tier` and add `floor_records`:

```python
def score_by_tier(records: list[dict]) -> list[tuple[int | None, dict, int]]:
    """Score per price band, once per headline floor, plus an all-tiers aggregate.

    Returns [(tier, metrics, n), ..., (floor sentinels), (None, metrics, n)].
    Cohorts with no records are omitted rather than emitted as zeros, so a floor
    nothing reaches is absent instead of reporting a fabricated 0%.

    The all-tiers (None) row is retained for API defaults and for continuity of
    a series months deep. It is POOLED ACROSS LIQUIDITY POPULATIONS whose spreads
    run 35.5% to 5.2%, so nothing quotes it - the logged headline reads
    HEADLINE_TIER.
    """
    by_tier: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_tier[r["price_tier"]].append(r)

    out: list[tuple[int | None, dict, int]] = []
    for tier in sorted(by_tier):
        metrics, n = score_cohort(by_tier[tier])
        if n:
            out.append((tier, metrics, n))

    # Descending sentinel order (-1, -2, -3) so the widest cohort is emitted
    # first and the log reads as a sweep upward through the floors.
    for sentinel in sorted(FLOOR_SWEEP, reverse=True):
        metrics, n = score_cohort(floor_records(records, FLOOR_SWEEP[sentinel]))
        if n:
            out.append((sentinel, metrics, n))

    metrics, n = score_cohort(records)
    if n:
        out.append((None, metrics, n))
    return out


def floor_records(records: list[dict], floor: float) -> list[dict]:
    """The subset at or above a price floor, by resolved base price.

    Filters on base_price rather than price_tier so a floor need not be a band
    edge - which is the point of sweeping $1/$5/$20 against tier cuts at
    1/5/20/100/1000.
    """
    return [r for r in records if r["base_price"] >= floor]


def headline_records(records: list[dict]) -> list[dict]:
    """The >=$1 subset used for the headline log line."""
    return floor_records(records, FLOOR_SWEEP[HEADLINE_TIER])
```

In `backend/scripts/backtest_accuracy.py`, import `FLOOR_SWEEP` alongside `HEADLINE_TIER`,
and in `_score_groups`, after the existing `logger.log(*_headline_line(...))` call:

```python
        # Where does the headline stabilise? $1 is a convention and the spread
        # evidence argues the honest floor is above it, so the three floors are
        # printed side by side. Reading the stored rows, not re-deriving.
        sweep = [
            (FLOOR_SWEEP[t], m, n) for t, m, n in tiered if t in FLOOR_SWEEP
        ]
        if len(sweep) > 1:
            parts = " | ".join(
                f">=${floor:g}: n={n:,} DA={m['directional_accuracy']:.1f}% "
                f"PT={_pt_str(m)}"
                for floor, m, n in sweep
            )
            logger.info(f"  [{horizon}d / {model_version}] floor sweep — {parts}")
```

In `_headline_line`, append the actionable numbers to `common`:

```python
        f"Skill={metrics['skill_vs_baseline']} "
        f"Actionable={_actionable_str(metrics)}"
```

and add the formatter beside `_pt_str`:

```python
def _actionable_str(metrics) -> str:
    """The friction-conditioned numbers as one field.

    Four numbers or none: a share, a hit rate, a net return and a PT verdict. A
    hit rate on its own is what this metric exists to stop being quoted.
    """
    scope = metrics.get("actionable_scope")
    if scope != "in_scope":
        return f"n/a ({scope})"
    n = metrics["actionable_n"]
    if not n:
        return (
            f"0 of {metrics['actionable_share_pct']:.1f}% — NO call cleared its "
            f"round trip + tier spread"
        )
    return (
        f"n={n:,} ({metrics['actionable_share_pct']:.1f}% of rows) "
        f"DA={metrics['actionable_da']:.1f}% "
        f"E[net]={metrics['actionable_e_net_pct']:+.2f}% "
        f"PT={metrics.get('actionable_pt_verdict')}"
    )
```

In `backend/scripts/walkforward_backtest.py:266`, the label map turns a floor sentinel into
`tier_-2`. Replace it:

```python
        (
            "all" if tier is None
            else "headline" if tier == HEADLINE_TIER
            else f"floor_{FLOOR_SWEEP[tier]:g}" if tier in FLOOR_SWEEP
            else f"tier_{tier}"
        ):
```

and add `FLOOR_SWEEP` to that file's import from `backtest.scoring`.

- [ ] **Step 4: Run the tests**

Run:
```bash
cd backend && venv/bin/python -m pytest tests/test_backtest_scoring.py \
  tests/test_accuracy_tier_mirror.py tests/test_parquet_nested_columns.py \
  tests/test_directional_test.py tests/test_accuracy_date_clustering.py \
  tests/test_directional_ci_units.py tests/test_walkforward_gate.py \
  tests/test_serving_policy.py tests/test_actionable_da.py tests/test_friction.py -q
```
Expected: PASS. `test_parquet_nested_columns.py`'s `FLOOR_SWEEP` import now resolves.

- [ ] **Step 5: Commit**

```bash
git add backend/backtest/scoring.py backend/scripts/backtest_accuracy.py \
  backend/scripts/walkforward_backtest.py backend/tests/test_backtest_scoring.py
git commit -m "feat: publish ActionableDA and sweep the headline at \$1/\$5/\$20"
```

---

### Task 6: Widen the API's price-tier bound

**Files:**
- Modify: `backend/api/routes/accuracy.py:157-166` (`PRICE_TIER_QUERY`)
- Test: `backend/tests/test_accuracy_headline_route.py`

**Interfaces:**
- Consumes: `FLOOR_SWEEP` from `backtest.scoring`.
- Produces: `/accuracy/*` accepts `price_tier` in `[-3, 5]`. `/accuracy/headline` unchanged.

`frontend/lib/api.ts` needs no change: it types `price_tier` as an unconstrained
`number | null` and `frontend/app/accuracy/page.tsx` renders no tier labels. Confirm with
`grep -n "price_tier" frontend/lib/api.ts` before concluding that.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_accuracy_headline_route.py`:

```python
def test_price_tier_query_admits_every_cohort_score_by_tier_emits():
    """A bound narrower than the emitted sentinels makes a stored row
    unreachable through the API - a silent 422 on a row that exists."""
    from api.routes.accuracy import PRICE_TIER_QUERY
    from backtest.scoring import FLOOR_SWEEP, price_tier

    assert PRICE_TIER_QUERY.ge == min(FLOOR_SWEEP)      # -3
    assert PRICE_TIER_QUERY.le == price_tier(50_000)    # 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && venv/bin/python -m pytest tests/test_accuracy_headline_route.py -q -k price_tier_query`
Expected: FAIL — `assert -1 == -3`.

- [ ] **Step 3: Write the implementation**

In `backend/api/routes/accuracy.py`, add `FLOOR_SWEEP` to the `backtest.scoring` import and
replace `PRICE_TIER_QUERY`:

```python
PRICE_TIER_QUERY = Query(
    None,
    ge=-3,
    le=5,
    description=(
        "Price cohort: 0-5 for a single price band (5 is >=$1000, split out "
        "from tier 4 because its bid-ask spread is 5.2% against 10.8%), or a "
        "floor sentinel -1/-2/-3 for the >=$1 / >=$5 / >=$20 headline sweep. "
        "Omit for the all-tiers aggregate, which is pooled across liquidity "
        "populations and is not quotable. The cohorts overlap, so exactly one "
        "is served."
    ),
)
```

- [ ] **Step 4: Run the tests**

Run:
```bash
cd backend && venv/bin/python -m pytest tests/test_accuracy_headline_route.py \
  tests/test_accuracy_tier_mirror.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/api/routes/accuracy.py backend/tests/test_accuracy_headline_route.py
git commit -m "feat: widen the accuracy price_tier bound to the new bands and floors"
```

---

### Task 7: Full suite, then document

**Files:**
- Modify: `backend/AGENTS.md` (the tier/headline gotchas)
- Create: `docs/changelog/2026-08-07-friction-conditioned-tier-scoring.md`
- Modify: `docs/research/2026-08-07-next-steps.md` (step 3 → DONE)

- [ ] **Step 1: Run the whole backend suite**

Run: `cd backend && venv/bin/python -m pytest tests/ -q`
Expected: PASS. Baseline is **1,258 tests** as of 2026-08-07 plus the ones added here;
~45s of collection before the first test runs. Scope it to `tests/` — a bare `pytest -q`
collects `scripts/test_social_signal.py` and aborts on a missing local `thefuzz`.

Any failure outside the files this plan touched is a real regression: fix it before
proceeding, do not adjust the assertion.

- [ ] **Step 2: Update `backend/AGENTS.md`**

Extend the existing "Never quote a directional accuracy on its own" gotcha with two
sentences: that `price_tier` now has six bands and a stored `price_tier == 4` predating
2026-08-07 means `≥$100`; and that `score_by_tier` emits `-1/-2/-3` floor sentinels of which
only `-1` is the published headline. Add a new gotcha for the actionable metric:
`actionable_*` is populated only at h ∈ {14, 30}, `actionable_scope` says which state a row is
in, and `SPREAD_BY_TIER` is a nearest-band approximation and not measured at the tier cuts.

Do not restate the numbers that live in `friction.py` — reference the module.

- [ ] **Step 3: Write the changelog**

Dispatch the `changelog-writer` subagent for
`docs/changelog/2026-08-07-friction-conditioned-tier-scoring.md`. It must record:

- The `price_tier` series discontinuity, stated as a discontinuity.
- That the spread table is measured at bands that are **not** the tier cuts, with the
  nearest-band rule named.
- The two gaps shipped deliberately: the staleness axis is **2 buckets, not the review's 4**
  (the quartiles need `stale_run_days`, which step 6 owns), and `s_i` is a tier median rather
  than the per-item BUFF spread (25 days of bid history would leave it NULL on almost every
  stored outcome).
- That the pooled `price_tier = NULL` row is retained for continuity only and is not quotable.
- That nothing was re-scored: a `--rescore` populates the new keys on existing rows at no
  archive cost, but running it is an operational step.
- The expected result — failure on `n_actionable` first — and that no production number exists
  yet, because every live cohort spans 1–2 forecast dates and `actionable_pt_verdict` will read
  `insufficient_dates` exactly as `pt_verdict` does.

- [ ] **Step 4: Mark step 3 done in the next-steps list**

In `docs/research/2026-08-07-next-steps.md`, change the step 3 heading from `NOT STARTED` to
`DONE 2026-08-07`, following the format steps 1 and 2 already use: what landed, what was
refuted or narrowed, an explicit **Not done** list, and a link to the changelog entry.

- [ ] **Step 5: Commit**

```bash
git add backend/AGENTS.md docs/changelog/2026-08-07-friction-conditioned-tier-scoring.md \
  docs/research/2026-08-07-next-steps.md
git commit -m "docs: record friction-conditioned tier scoring and close step 3"
```

---

## Self-Review

**Spec coverage.** §1 six bands → Task 2. §2 `friction.py` → Task 1. §3 ActionableDA →
Tasks 3 (metric) and 4 (record fields) and 5 (wiring). §4 floor sweep → Task 5. §5 grid →
Task 5's `score_by_tier` docstring plus Task 7's changelog gap statement; the 2-bucket
staleness axis needs no code, it is the `unchanged`/`moved` partition `score_cohort` already
computes. §6 API → Task 6. §7 tests → distributed across every task. §8 out-of-scope →
nothing implements it, correctly. §9 expected result → Task 7's changelog.

**Type consistency.** `actionable_metrics(records, horizon_days, min_dates, venue=...)` is
called with the same argument order in Task 3's tests and Task 5's `score_cohort` wiring.
`FLOOR_SWEEP` maps sentinel → USD float in Tasks 2, 5, 6 and 7 consistently.
`floor_records(records, floor)` takes a float floor, not a sentinel, everywhere.
`fold_records`'s new `horizon_days` is keyword-with-default in both the signature change and
the test.

**Known cross-task failures, deliberate.** Task 1 leaves
`test_every_price_tier_has_a_spread_entry` red until Task 2. Task 2 leaves
`test_parquet_nested_columns.py`'s `FLOOR_SWEEP` import red until Task 5. Both are called out
in the task that creates them; neither may be "fixed" by weakening an assertion.

**Two assertions must be read from a failure, not guessed:** the tier set at
`test_backtest_scoring.py:1072` (Task 5, Step 1) and the existing frozen-outcome fixture's
name (Task 4, Step 1). Both are flagged in place.
