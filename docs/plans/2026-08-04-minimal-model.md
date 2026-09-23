# Minimal Model Implementation Plan

> # ✅ EXECUTED AND CLOSED (2026-08-04)
>
> All 13 tasks landed. Evidence: `backend/models/conformal.py`; `QUANTILES = [0.5]`,
> `N_ENSEMBLES = 1` and `IncompatibleModelArtifact` in `forecaster.py`;
> `backtest/walkforward_records.py` and `backtest/paired_mde.py` (`ffe5ad7`, `bcce8ea`);
> `TRAIN_HORIZON_MAX_ROWS` at `forecast_prices.py:58`. Ledger:
> `.superpowers/sdd/2026-08-04-minimal-model/progress.md` (Tasks 1–10 + fix `1890f41`).
>
> **Outcome: `docs/changelog/2026-08-04-minimal-model-results.md`.** Part 3's plumbing shipped;
> the coverage **reinvestment** was measured and then **declined on cost**. That is a decision,
> not an outstanding task.
>
> ⚠️ Two config values this plan set have since moved: `MODEL_ARTIFACT_VERSION` is now **6**
> (`8be48c5`, not 5), and the training budget is **1,200,000** rows at a **$1** floor
> (`6eb8775`, `5ebcc73`).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the forecaster from 40 LightGBM models to 8 — replacing 24 p10/p90 quantile GBMs with locally-weighted split conformal around a single median model — without losing more directional accuracy than a repaired measurement gate can actually resolve.

**Architecture:** Three sequenced commits. Part 1 repairs `walkforward_backtest.py` (routes it through the existing clustered scorer, fits the directional classifier production actually serves, establishes a clean timing baseline, and pre-registers an MDE-based acceptance bar). Part 2 performs the rewrite: a new pure `backend/models/conformal.py` module supplies σ, calibration and band construction, and `forecaster.py` config constants collapse the quantile/ensemble grid. Part 3 raises the training item budget as a separately measured change.

**Tech Stack:** Python 3.13, LightGBM, pandas, numpy, DuckDB over `price-archive/*.parquet`, SQLAlchemy, pytest.

**Spec:** `docs/specs/2026-08-04-minimal-model-design.md` (commits `46a2278`, `1a3bdb9`)

## Global Constraints

- Nominal band coverage is pinned at **80%**, `α = 0.20`. Not 90%. The old code disagreed with itself (`forecaster.py:3092` targeted 90%, the comment at `:3088` described 80%).
- Direction classes are **`0=down, 1=flat, 2=up`** (`forecaster.py:_direction_classes`, `:3319-3325`). Never re-derive this mapping.
- Flat band: `DIRECTION_FLAT_TOLERANCE_PCT = 0.5` (**percent**) in `forecaster.py`; `FLAT_TOLERANCE = 0.005` (**fraction**) in `backtest/scoring.py`. These are the same threshold in different units. `direction_from_return` takes a **fraction**; `_direction_classes` takes **percent**. `target_return_{h}d` is in **percent**.
- `pytest backend/tests/ -q` and `python3 -m py_compile` must pass before every commit (`AGENTS.md`).
- Training is **fully sequential**. Do not add multiprocessing or threading to the training path — both were removed on 2026-07-21 for OpenMP deadlocks and `AGENTS.md` forbids re-adding them.
- Bump `ItemForecaster.VOTED_CACHE_VERSION` if `_fetch_voted_price_history` or `_apply_multi_source_voting` changes. No task here should touch them; if one does, bump it.
- No feature flag for the new model class. Two artifact formats coexisting is the failure mode this project has hit three times. `git revert` plus the retained model cache is the rollback.
- Run all commands from `backend/` unless stated otherwise.

## Correction to the spec's deletion table

The spec lists `_fix_quantile_crossing` for deletion. **It cannot be deleted in Part 2** — `scripts/walkforward_backtest.py:294` calls it, and the harness's baseline arm (arm A, the current 40-model config) still needs it to build its interval.

Revised: Part 2 removes the call from **`predict()`**, and the static method survives for the harness. It becomes deletable only when arm A is retired, which is out of scope here.

## File Structure

| File | Responsibility |
|---|---|
| `backend/models/conformal.py` **(new)** | Pure: σ from a frame, σ clip bounds, `q̂` calibration, band construction. No I/O, no LightGBM. Keeps this logic out of the 4,865-line `forecaster.py`. |
| `backend/backtest/walkforward_records.py` **(new)** | Pure: fold arrays → `score_cohort` record dicts. Owns the percent→fraction conversion and the class→label mapping. |
| `backend/backtest/paired_mde.py` **(new)** | Pure: paired, date-clustered CI on the DA *difference* between two arms measured on the same folds. |
| `backend/scripts/walkforward_backtest.py` | Modify: emit records, fit the classifier per fold, aggregate via `score_cohort`, add `--arm`. |
| `backend/models/forecaster.py` | Modify: config constants, calibration call site, band construction in `predict()`, artifact version. |
| `backend/scripts/forecast_prices.py` | Modify (Part 3 only): propagate `max_rows` to `max_feature_rows`. |
| `backend/tests/test_conformal.py` **(new)** | Unit tests for the conformal module. |
| `backend/tests/test_walkforward_records.py` **(new)** | Unit tests for record construction, units, class mapping. |
| `backend/tests/test_minimal_model_shape.py` **(new)** | Model count, artifact version, band ordering. |
| `docs/architecture/model-optimization.md` | Modify: timing baseline, results. |

---

# PART 1 — The measurement rig

Nothing in Part 2 may be committed before Part 1 is merged and Task 5's bar is written down.

---

### Task 1: Pure record construction for walkforward folds

**Files:**
- Create: `backend/backtest/walkforward_records.py`
- Test: `backend/tests/test_walkforward_records.py`

**Interfaces:**
- Consumes: `backtest.scoring.direction_from_return`, `backtest.scoring.price_tier`
- Produces: `fold_records(...) -> list[dict]`. Tasks 2 and 3 call it.
- The dicts carry every key `score_cohort` reads, **plus two it does not**:
  - `item_id` — **load-bearing for Task 2.** `paired_da_difference` indexes records by `(item_id, forecast_date)`; without it, arms cannot be paired and the MDE cannot be computed. Not optional.
  - `predicted_direction` — carried for parity with the repo's existing record shape at `backtest_accuracy.py:551`, which also includes it unread by `score_cohort`. Keeps one record schema across both scoring paths.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_walkforward_records.py`:

```python
"""Record construction for the walkforward gate.

score_cohort consumes per-record dicts, not per-fold aggregates. This module
owns the two conversions that are easy to get silently wrong: percent returns
into direction_from_return's fractions, and the classifier's integer classes
into direction labels.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from backtest.walkforward_records import CLASS_TO_DIRECTION, fold_records


def _kwargs(**overrides):
    base = dict(
        item_ids=np.array(["ak47"]),
        forecast_dates=np.array([date(2026, 1, 5)], dtype=object),
        base_prices=np.array([100.0]),
        actual_returns_pct=np.array([2.0]),
        mid_returns_pct=np.array([1.0]),
        low_returns_pct=np.array([-5.0]),
        high_returns_pct=np.array([5.0]),
        predicted_classes=np.array([2]),
    )
    base.update(overrides)
    return base


def test_class_to_direction_mapping_is_down_flat_up():
    # forecaster._direction_classes: 0=down, 1=flat, 2=up.
    assert CLASS_TO_DIRECTION == {0: "down", 1: "flat", 2: "up"}


def test_returns_are_converted_from_percent_to_fraction():
    # +0.3% is inside the 0.5% flat band. If the percent value were passed
    # straight to direction_from_return (fraction, tolerance 0.005) it would
    # read as +30% and label "up".
    rec = fold_records(**_kwargs(actual_returns_pct=np.array([0.3])))[0]
    assert rec["actual_direction"] == "flat"


def test_two_percent_return_is_up():
    rec = fold_records(**_kwargs(actual_returns_pct=np.array([2.0])))[0]
    assert rec["actual_direction"] == "up"


def test_direction_correct_compares_classifier_call_to_actual():
    up = fold_records(**_kwargs(predicted_classes=np.array([2])))[0]
    down = fold_records(**_kwargs(predicted_classes=np.array([0])))[0]
    assert up["direction_correct"] == 1
    assert down["direction_correct"] == 0


def test_prices_are_reconstructed_from_base_and_return():
    rec = fold_records(**_kwargs())[0]
    assert rec["base_price"] == pytest.approx(100.0)
    assert rec["actual_price"] == pytest.approx(102.0)
    assert rec["abs_error"] == pytest.approx(1.0)      # mid 101 vs actual 102
    assert rec["sq_error"] == pytest.approx(1.0)
    assert rec["pct_error"] == pytest.approx(1.0)      # divided by BASE, not actual


def test_in_interval_uses_price_space_band():
    inside = fold_records(**_kwargs())[0]
    assert inside["in_interval"] == 1
    outside = fold_records(**_kwargs(high_returns_pct=np.array([1.5])))[0]
    assert outside["in_interval"] == 0


def test_forecast_date_is_carried_through_as_the_cluster_key():
    rec = fold_records(**_kwargs())[0]
    assert rec["forecast_date"] == date(2026, 1, 5)


def test_confidence_is_uniform_low():
    # The harness has no confidence estimator; production's comes from
    # _calibrate_confidence. Uniform "low" makes score_cohort's conf_gap_pp
    # and conf_high_interval_cov structurally zero, which is why the results
    # write-up must not read them for harness arms.
    rec = fold_records(**_kwargs())[0]
    assert rec["confidence"] == "low"


def test_rejects_mismatched_array_lengths():
    with pytest.raises(ValueError, match="equal length"):
        fold_records(**_kwargs(mid_returns_pct=np.array([1.0, 2.0])))
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_walkforward_records.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.walkforward_records'`

- [x] **Step 3: Write the implementation**

Create `backend/backtest/walkforward_records.py`:

```python
"""Fold arrays -> score_cohort records, for the walkforward gate.

Pure: no I/O, no LightGBM, no clock. Exists as its own module because it owns
two unit conversions that were previously wrong or absent in
scripts/walkforward_backtest.py:

1. `target_return_{h}d` is in PERCENT; `direction_from_return` takes a
   FRACTION. The old `_compute_metrics` never called it at all — it compared
   floats exactly, so "flat" never fired and the gate was a 2-label problem at
   50% chance while production is 3-label at ~33%.
2. The predicted direction must come from the classifier's argmax, because that
   is what production serves (forecaster.py:2981-2982). The old code took the
   sign of the p50 regression.
"""
from __future__ import annotations

import numpy as np

from backtest.scoring import direction_from_return, price_tier

# forecaster._direction_classes buckets returns as 0=down, 1=flat, 2=up.
CLASS_TO_DIRECTION = {0: "down", 1: "flat", 2: "up"}


def fold_records(
    *,
    item_ids,
    forecast_dates,
    base_prices,
    actual_returns_pct,
    mid_returns_pct,
    low_returns_pct,
    high_returns_pct,
    predicted_classes,
) -> list[dict]:
    """Build score_cohort records for one fold.

    All arrays must be the same length and aligned row-for-row. Returns are in
    percent (the units of `target_return_{h}d`); prices are reconstructed as
    `base * (1 + ret/100)`.

    `predicted_classes` are the directional classifier's argmax values. Pass
    None to score the median's sign instead — used to report both estimators
    side by side (spec Task 1b).
    """
    arrays = {
        "item_ids": item_ids,
        "forecast_dates": forecast_dates,
        "base_prices": base_prices,
        "actual_returns_pct": actual_returns_pct,
        "mid_returns_pct": mid_returns_pct,
        "low_returns_pct": low_returns_pct,
        "high_returns_pct": high_returns_pct,
    }
    n = len(base_prices)
    for name, arr in arrays.items():
        if len(arr) != n:
            raise ValueError(
                f"all inputs must be of equal length; {name} has {len(arr)}, expected {n}"
            )
    if predicted_classes is not None and len(predicted_classes) != n:
        raise ValueError(
            f"all inputs must be of equal length; predicted_classes has "
            f"{len(predicted_classes)}, expected {n}"
        )

    base = np.asarray(base_prices, dtype=float)
    actual_ret = np.asarray(actual_returns_pct, dtype=float)
    mid_ret = np.asarray(mid_returns_pct, dtype=float)
    low_ret = np.asarray(low_returns_pct, dtype=float)
    high_ret = np.asarray(high_returns_pct, dtype=float)

    actual = base * (1.0 + actual_ret / 100.0)
    mid = base * (1.0 + mid_ret / 100.0)
    low = base * (1.0 + low_ret / 100.0)
    high = base * (1.0 + high_ret / 100.0)

    records = []
    for i in range(n):
        if base[i] <= 0:
            continue
        # /100: percent -> fraction, which is what direction_from_return's
        # FLAT_TOLERANCE = 0.005 is expressed in.
        actual_direction = direction_from_return(actual_ret[i] / 100.0)
        if predicted_classes is not None:
            predicted_direction = CLASS_TO_DIRECTION[int(predicted_classes[i])]
        else:
            predicted_direction = direction_from_return(mid_ret[i] / 100.0)

        abs_error = abs(mid[i] - actual[i])
        records.append({
            "abs_error": abs_error,
            "sq_error": (mid[i] - actual[i]) ** 2,
            # Divided by the BASE leg, matching backtest_accuracy._derive_verdict.
            "pct_error": abs(abs_error / base[i]) * 100.0,
            "direction_correct": 1 if predicted_direction == actual_direction else 0,
            "predicted_direction": predicted_direction,
            "actual_direction": actual_direction,
            "in_interval": 1 if low[i] <= actual[i] <= high[i] else 0,
            # The harness has no confidence estimator, so score_cohort's
            # conf_* fields are structurally degenerate for these arms.
            "confidence": "low",
            "base_price": float(base[i]),
            "actual_price": float(actual[i]),
            "price_tier": price_tier(float(base[i])),
            "item_id": item_ids[i],
            # The clustering unit: rows sharing a date share one market move.
            "forecast_date": forecast_dates[i],
        })
    return records
```

- [x] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_walkforward_records.py -v`

Expected: PASS, 9 tests.

- [x] **Step 5: Commit**

```bash
git add backend/backtest/walkforward_records.py backend/tests/test_walkforward_records.py
git commit -m "feat: pure record construction for the walkforward gate

score_cohort consumes per-record dicts. This module owns the percent->
fraction conversion into direction_from_return and the 0=down/1=flat/2=up
class mapping, both of which the old _compute_metrics got wrong or omitted."
```

---

### Task 2: Paired, date-clustered CI on the DA difference

**Files:**
- Create: `backend/backtest/paired_mde.py`
- Test: `backend/tests/test_paired_mde.py`

**Interfaces:**
- Consumes: `backtest.scoring.BOOTSTRAP_RNG_SEED`, `N_BOOTSTRAP`, `BOOTSTRAP_CI`
- Produces: `paired_da_difference(records_a, records_b) -> dict` with keys `mean_diff_pp`, `ci_lower_pp`, `ci_upper_pp`, `mde_pp`, `n_paired`, `n_dates`. Task 5 calls it.

**Why this and not two independent CIs:** the arms run on identical folds, so pairing removes the between-date variance that dominates both arms equally. An unpaired comparison of two very wide CIs would report an MDE so large that no design could ever pass. Pairing is what makes the bar achievable and is only valid *because* the folds are shared.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_paired_mde.py`:

```python
"""Paired MDE for arm-vs-arm comparison on shared folds."""
from __future__ import annotations

from datetime import date

import pytest

from backtest.paired_mde import paired_da_difference


def _rec(item, day, correct):
    return {
        "item_id": item,
        "forecast_date": date(2026, 1, day),
        "direction_correct": correct,
    }


def _arm(pattern):
    """pattern: {day: [(item, correct), ...]}"""
    return [_rec(item, day, correct)
            for day, entries in pattern.items()
            for item, correct in entries]


def test_identical_arms_have_zero_mean_difference():
    pattern = {d: [(f"i{i}", i % 2) for i in range(10)] for d in range(1, 6)}
    out = paired_da_difference(_arm(pattern), _arm(pattern))
    assert out["mean_diff_pp"] == pytest.approx(0.0)
    assert out["n_paired"] == 50
    assert out["n_dates"] == 5


def test_uniformly_better_arm_reports_positive_difference():
    a = {d: [(f"i{i}", 0) for i in range(10)] for d in range(1, 6)}
    b = {d: [(f"i{i}", 1) for i in range(10)] for d in range(1, 6)}
    out = paired_da_difference(_arm(a), _arm(b))
    # b is arm 2; the difference is reported as b - a.
    assert out["mean_diff_pp"] == pytest.approx(100.0)


def test_only_rows_present_in_both_arms_are_paired():
    a = _arm({1: [("x", 1), ("y", 1)]})
    b = _arm({1: [("x", 0)]})
    out = paired_da_difference(a, b)
    assert out["n_paired"] == 1


def test_mde_is_the_half_width_of_the_difference_interval():
    a = {d: [(f"i{i}", 1) for i in range(10)] for d in range(1, 8)}
    b = {d: [(f"i{i}", 1 if d % 2 else 0) for i in range(10)] for d in range(1, 8)}
    out = paired_da_difference(_arm(a), _arm(b))
    expected = (out["ci_upper_pp"] - out["ci_lower_pp"]) / 2
    assert out["mde_pp"] == pytest.approx(expected)
    assert out["mde_pp"] > 0


def test_single_date_yields_no_interval():
    a = _arm({1: [(f"i{i}", 1) for i in range(10)]})
    b = _arm({1: [(f"i{i}", 0) for i in range(10)]})
    out = paired_da_difference(a, b)
    # One date carries no information about between-date variation.
    assert out["ci_lower_pp"] is None
    assert out["ci_upper_pp"] is None
    assert out["mde_pp"] is None


def test_no_overlap_raises():
    a = _arm({1: [("x", 1)]})
    b = _arm({2: [("y", 1)]})
    with pytest.raises(ValueError, match="no paired records"):
        paired_da_difference(a, b)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_paired_mde.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.paired_mde'`

- [x] **Step 3: Write the implementation**

Create `backend/backtest/paired_mde.py`:

```python
"""Paired, date-clustered confidence interval on the DA difference between two
arms measured on the SAME folds.

Why paired: the arms share folds, so most of the uncertainty in either arm's
absolute DA is between-date variance that affects both identically (the
2025-12-01 rising / 2026-07-17 falling asymmetry documented in
docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md). An
unpaired comparison of two wide intervals reports an MDE no design could pass.
Differencing within (item, date) removes the common term.

Still clustered: what remains is between-date variation in the DIFFERENCE, so
dates are the resampling unit, not rows.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backtest.scoring import BOOTSTRAP_CI, BOOTSTRAP_RNG_SEED, N_BOOTSTRAP


def paired_da_difference(
    records_a: list[dict],
    records_b: list[dict],
    n_resamples: int = N_BOOTSTRAP,
    ci: int = BOOTSTRAP_CI,
) -> dict:
    """Bootstrap the mean of (b - a) direction_correct over shared (item, date).

    Returns percentage-point figures. `mde_pp` is the half-width of the
    interval: the smallest difference this gate can resolve. None when fewer
    than 2 dates are shared.
    """
    index_a = {(r["item_id"], r["forecast_date"]): r["direction_correct"]
               for r in records_a}
    by_date: dict = defaultdict(list)
    n_paired = 0
    for r in records_b:
        key = (r["item_id"], r["forecast_date"])
        if key in index_a:
            by_date[r["forecast_date"]].append(
                r["direction_correct"] - index_a[key]
            )
            n_paired += 1

    if n_paired == 0:
        raise ValueError(
            "no paired records: the two arms share no (item_id, forecast_date) "
            "pairs, so they were not measured on the same folds"
        )

    dates = sorted(by_date)
    groups = [np.array(by_date[d], dtype=float) for d in dates]
    all_diffs = np.concatenate(groups)
    mean_diff_pp = float(all_diffs.mean()) * 100.0

    out = {
        "mean_diff_pp": round(mean_diff_pp, 4),
        "n_paired": n_paired,
        "n_dates": len(dates),
        "ci_lower_pp": None,
        "ci_upper_pp": None,
        "mde_pp": None,
    }
    if len(groups) < 2:
        return out

    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    sums = np.array([g.sum() for g in groups], dtype=float)
    counts = np.array([g.size for g in groups], dtype=float)
    n_groups = len(groups)

    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n_groups, size=n_groups)
        stats[i] = sums[idx].sum() / counts[idx].sum()
    stats *= 100.0

    alpha = (100 - ci) / 2
    lower = float(np.percentile(stats, alpha))
    upper = float(np.percentile(stats, 100 - alpha))
    out["ci_lower_pp"] = round(lower, 4)
    out["ci_upper_pp"] = round(upper, 4)
    out["mde_pp"] = round((upper - lower) / 2, 4)
    return out
```

- [x] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_paired_mde.py -v`

Expected: PASS, 6 tests.

- [x] **Step 5: Commit**

```bash
git add backend/backtest/paired_mde.py backend/tests/test_paired_mde.py
git commit -m "feat: paired date-clustered CI on the DA difference between arms

The arms share folds, so differencing within (item, date) removes the
between-date variance that dominates both. Unpaired, the MDE would be so
large no design could pass it."
```

---

### Task 3: Route walkforward through score_cohort and fit the served classifier

**Files:**
- Modify: `backend/scripts/walkforward_backtest.py:45` (add `DIRECTION_NUM_ROUNDS`), `:103-137` (delete `_compute_metrics`), `:229-328` (fold loop + aggregation)
- Test: `backend/tests/test_walkforward_gate.py`

**Interfaces:**
- Consumes: `fold_records` (Task 1), `backtest.scoring.score_by_tier`, `HEADLINE_TIER`, `ItemForecaster._fit_direction_classifier`
- Produces: `run_walkforward(...)` report gains `horizons[h]["classifier"]` and `horizons[h]["median_sign"]`, each a `score_cohort` metrics dict, plus `horizons[h]["records"]` when `return_records=True` (Task 5 needs the records to pair arms).

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_walkforward_gate.py`:

```python
"""The walkforward gate must score what production serves, with the clustered CI.

Two defects this locks down:
  - it aggregated per-fold metrics weighted by sample_count, treating
    correlated item-rows as independent observations;
  - it derived the predicted direction from the sign of the p50 regression,
    while production serves the directional classifier's argmax
    (forecaster.py:2981-2982).
"""
from __future__ import annotations

import inspect
from datetime import date

import numpy as np
import pytest

import scripts.walkforward_backtest as wf
from backtest.scoring import HEADLINE_TIER


def test_sample_count_weighted_aggregation_is_gone():
    src = inspect.getsource(wf)
    assert 'f["directional_accuracy"] * f["sample_count"]' not in src, (
        "per-fold sample_count weighting treats correlated item-rows as "
        "independent; aggregate over pooled records via score_cohort instead"
    )


def test_compute_metrics_helper_is_removed():
    assert not hasattr(wf, "_compute_metrics"), (
        "_compute_metrics had no flat band (2-label at 50% chance) and took "
        "the median's sign as the prediction; both are replaced by "
        "fold_records + score_cohort"
    )


def test_aggregate_reports_clustered_ci_and_coverage_flag():
    records = [
        {
            "abs_error": 1.0, "sq_error": 1.0, "pct_error": 1.0,
            "direction_correct": i % 2, "predicted_direction": "up",
            "actual_direction": "up" if i % 2 else "down",
            "in_interval": 1, "confidence": "low",
            "base_price": 10.0, "actual_price": 11.0, "price_tier": 2,
            "item_id": f"i{i}", "forecast_date": date(2026, 1, 1 + (i % 25)),
        }
        for i in range(200)
    ]
    agg = wf._aggregate_records(records)
    assert agg["directional_accuracy_ci_clustered_lower"] is not None
    assert agg["distinct_forecast_dates"] == 25
    assert agg["date_coverage_sufficient"] is True


def test_aggregate_uses_the_dollar_headline_tier():
    # Tier 0 (<$1) is 72% of the universe and its labels are tick-quantised.
    # The headline must be the >=$1 aggregate, matching production.
    records = []
    for i in range(60):
        records.append({
            "abs_error": 1.0, "sq_error": 1.0, "pct_error": 1.0,
            "direction_correct": 1, "predicted_direction": "up",
            "actual_direction": "up", "in_interval": 1, "confidence": "low",
            "base_price": 50.0, "actual_price": 51.0, "price_tier": 3,
            "item_id": f"rich{i}", "forecast_date": date(2026, 1, 1 + i % 25),
        })
    for i in range(60):
        records.append({
            "abs_error": 1.0, "sq_error": 1.0, "pct_error": 1.0,
            "direction_correct": 0, "predicted_direction": "up",
            "actual_direction": "down", "in_interval": 0, "confidence": "low",
            "base_price": 0.10, "actual_price": 0.09, "price_tier": 0,
            "item_id": f"penny{i}", "forecast_date": date(2026, 1, 1 + i % 25),
        })
    agg = wf._aggregate_records(records)
    # 100% on the >=$1 subset, not the 50% all-tiers figure.
    assert agg["directional_accuracy"] == pytest.approx(100.0)


def test_fold_scores_both_estimators():
    sig = inspect.signature(wf._score_fold)
    assert "predicted_classes" in sig.parameters
    assert "mid_returns_pct" in sig.parameters


def test_classifier_is_fitted_per_fold():
    src = inspect.getsource(wf)
    assert "_fit_direction_classifier" in src, (
        "the gate must fit the directional classifier production serves, not "
        "score the median's sign"
    )
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_walkforward_gate.py -v`

Expected: FAIL — `test_sample_count_weighted_aggregation_is_gone`, `test_compute_metrics_helper_is_removed`, and `AttributeError: module has no attribute '_aggregate_records'` / `'_score_fold'`.

- [x] **Step 3: Delete `_compute_metrics` and add the two new helpers**

In `backend/scripts/walkforward_backtest.py`, delete lines 103-137 (`_compute_metrics` entirely) and add these imports near line 30:

```python
from backtest.scoring import HEADLINE_TIER, score_by_tier
from backtest.walkforward_records import fold_records
```

Add after `_load_all_prices` (where `_compute_metrics` used to be):

```python
# Boosting rounds for the per-fold directional classifier. Matches
# _fit_direction_classifier's own default so the gate's classifier is
# configured like production's.
DIRECTION_NUM_ROUNDS = 200


def _score_fold(*, item_ids, forecast_dates, base_prices, actual_returns_pct,
                mid_returns_pct, low_returns_pct, high_returns_pct,
                predicted_classes):
    """Records for one fold, for both estimators.

    Returns (classifier_records, median_sign_records). Both describe the same
    rows; only `predicted_direction` and `direction_correct` differ. The
    classifier set is what the pre-registered bar governs — production serves
    the classifier's call (forecaster.py:2981-2982) — and the median-sign set
    is reported alongside because the two have never been compared on the same
    folds.
    """
    shared = dict(
        item_ids=item_ids,
        forecast_dates=forecast_dates,
        base_prices=base_prices,
        actual_returns_pct=actual_returns_pct,
        mid_returns_pct=mid_returns_pct,
        low_returns_pct=low_returns_pct,
        high_returns_pct=high_returns_pct,
    )
    return (
        fold_records(**shared, predicted_classes=predicted_classes),
        fold_records(**shared, predicted_classes=None),
    )


def _aggregate_records(records):
    """Pool records across folds and score them with the clustered scorer.

    Replaces the old sample_count-weighted per-fold average, which treated
    every item-row inside a fold as an independent observation. Directional
    outcomes are clustered by date, so the effective sample size is the number
    of distinct dates — see backtest/scoring.py:MIN_FORECAST_DATES.

    Returns the >=$1 headline cohort's metrics, matching production's headline
    tier, with the per-tier rows attached under "by_tier".
    """
    if not records:
        return None
    scored = score_by_tier(records)
    headline = next((m for tier, m, _ in scored if tier == HEADLINE_TIER), None)
    all_tiers = next((m for tier, m, _ in scored if tier is None), None)
    out = dict(headline or all_tiers or {})
    out["by_tier"] = {
        ("all" if tier is None else "headline" if tier == HEADLINE_TIER else f"tier_{tier}"):
            {"directional_accuracy": m["directional_accuracy"], "sample_count": n}
        for tier, m, n in scored
    }
    return out
```

- [x] **Step 4: Rewrite the fold loop to fit the classifier and collect records**

Replace lines 229-328 (from `fold_results = []` through the two `logger.info` summary lines) with:

```python
            clf_records = []
            median_records = []

            for window_end in range(split_idx + 1, len(dates), STEP_DAYS):
                train_dates = dates[:window_end]
                val_dates = dates[window_end:window_end + VAL_WINDOW_DAYS]

                if len(val_dates) < 7:
                    continue

                train_df = tdf[tdf["date"].isin(train_dates)]
                val_df = tdf[tdf["date"].isin(val_dates)]

                if len(val_df) < MIN_VAL_SAMPLES:
                    continue

                if len(train_df) > MAX_TRAIN_ROWS:
                    train_df = train_df.sort_values("date").tail(MAX_TRAIN_ROWS)

                feature_cols = [c for c in forecaster.feature_cols if c in tdf.columns]
                if not feature_cols:
                    exclude = {"item_id", "date", "timestamp", "price", "volume"}
                    exclude |= {f"target_{h}d" for h in forecaster.HORIZONS}
                    exclude |= {f"target_return_{h}d" for h in forecaster.HORIZONS}
                    feature_cols = [c for c in tdf.columns if c not in exclude
                                    and tdf[c].dtype in (np.float64, np.float32, np.int64, int, float)]

                if len(feature_cols) > 2:
                    corr = train_df[feature_cols].corr().abs()
                    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    to_drop = set()
                    for col in upper.columns:
                        if col in to_drop:
                            continue
                        highly_corr = upper[col][upper[col] > 0.95].index
                        to_drop.update(highly_corr)
                    feature_cols = [c for c in feature_cols if c not in to_drop]

                medians = train_df[feature_cols].median()
                X_train = train_df[feature_cols].fillna(medians)
                y_train = train_df[f"target_return_{horizon}d"]
                X_val = val_df[feature_cols].fillna(medians)
                y_val = val_df[f"target_return_{horizon}d"]

                preds = {}
                for q in QUANTILES:
                    params = _get_tuned_params(meta, horizon, q)
                    params.update({
                        "objective": "quantile",
                        "alpha": q,
                        "metric": "quantile",
                        "verbosity": -1,
                        "random_state": 42,
                        "n_jobs": -1,
                    })

                    dtrain = lgb.Dataset(X_train.values, y_train.values)
                    dval = lgb.Dataset(X_val.values, y_val.values, reference=dtrain)
                    model = lgb.train(
                        params, dtrain,
                        num_boost_round=200,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)]
                    )
                    preds[q] = model.predict(X_val.values)

                if len(preds) != 3:
                    continue

                low, high = ItemForecaster._fix_quantile_crossing(
                    preds[0.1], preds[0.5], preds[0.9]
                )

                # The estimator production actually serves. sigma_train/
                # sigma_val stay None so the fixed-band labels production
                # selects are used (forecaster.py:2984-2993).
                clf = forecaster._fit_direction_classifier(
                    X_train.values, y_train.values,
                    X_val.values, y_val.values,
                    _get_tuned_params(meta, horizon, 0.5).get("boosting_type", "gbdt"),
                    ItemForecaster._direction_tree_params(
                        {0.5: _get_tuned_params(meta, horizon, 0.5)}
                    ),
                    horizon=horizon,
                    sigma_train=None,
                    sigma_val=None,
                    num_boost_round=DIRECTION_NUM_ROUNDS,
                )
                predicted_classes = clf.predict(X_val.values).argmax(axis=1)

                fold_clf, fold_median = _score_fold(
                    item_ids=val_df["item_id"].to_numpy(),
                    forecast_dates=val_df["date"].to_numpy(),
                    base_prices=val_df["price"].to_numpy(dtype=float),
                    actual_returns_pct=y_val.to_numpy(dtype=float),
                    mid_returns_pct=preds[0.5],
                    low_returns_pct=low,
                    high_returns_pct=high,
                    predicted_classes=predicted_classes,
                )
                clf_records.extend(fold_clf)
                median_records.extend(fold_median)

            if not clf_records:
                logger.warning(f"    No folds completed for {horizon}d")
                continue

            agg_clf = _aggregate_records(clf_records)
            agg_median = _aggregate_records(median_records)
            entry = {
                "classifier": agg_clf,
                "median_sign": agg_median,
                "sample_count": len(clf_records),
            }
            if return_records:
                entry["records"] = clf_records
            results_by_horizon[horizon] = entry

            lo = agg_clf["directional_accuracy_ci_clustered_lower"]
            hi = agg_clf["directional_accuracy_ci_clustered_upper"]
            ci_txt = f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "[insufficient dates]"
            logger.info(
                f"    {horizon}d: {len(clf_records):,} records, "
                f"{agg_clf['distinct_forecast_dates']} dates "
                f"(sufficient={agg_clf['date_coverage_sufficient']})"
            )
            logger.info(
                f"      classifier DirAcc={agg_clf['directional_accuracy']:.1f}% "
                f"clustered95={ci_txt}   "
                f"median-sign DirAcc={agg_median['directional_accuracy']:.1f}%"
            )
            logger.info(
                f"      MAE=${agg_clf['mae']:.2f}  MAPE={agg_clf['mape']:.1f}%  "
                f"IntCov={agg_clf['interval_coverage']:.1f}%"
            )
```

- [x] **Step 5: Update the signature and the DB write**

Change the `run_walkforward` signature at line 178:

```python
def run_walkforward(max_items=500, horizons=None, skip_db=False, return_records=False):
```

Replace the DB-write block (old lines 335-348) with:

```python
        if not skip_db:
            today = date.today()
            for horizon, entry in results_by_horizon.items():
                clf = entry["classifier"]
                _upsert_accuracy(db, [{
                    "prediction_type": "walkforward_backtest",
                    "evaluation_date": today,
                    "horizon_days": horizon,
                    # Bumped: the metric definition changed (3-label with a
                    # flat band, classifier-sourced direction, clustered CI),
                    # so these rows are NOT continuous with lgbm-v3-tuned.
                    "model_version": "lgbm-v3-clustered",
                    "evaluation_window_days": None,
                    "sample_count": entry["sample_count"],
                    "metrics": {
                        k: clf[k] for k in [
                            "mae", "rmse", "mape", "wmape",
                            "directional_accuracy",
                            "directional_accuracy_ci_clustered_lower",
                            "directional_accuracy_ci_clustered_upper",
                            "distinct_forecast_dates",
                            "date_coverage_sufficient",
                            "interval_coverage",
                        ]
                    } | {
                        "median_sign_directional_accuracy":
                            entry["median_sign"]["directional_accuracy"],
                        "by_tier": clf["by_tier"],
                    },
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                }])
```

And in the report dict (old line 360), replace the `horizons` entry with:

```python
            "horizons": {
                str(h): {k: v for k, v in m.items()
                         if return_records or k != "records"}
                for h, m in results_by_horizon.items()
            },
```

⚠️ **The `return_records or` guard is essential.** Stripping `records`
unconditionally makes `return_records=True` a no-op, because `report` is the only
value `run_walkforward` returns — and Task 5's `compute_mde.py` reads
`runs[0]["horizons"][h]["records"]` to pair the two seed runs. Without the guard
it always gets nothing and the MDE can never be computed, silently. Records are
kept out of the report by default because they are large; the flag is what makes
them retrievable.

- [x] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_walkforward_gate.py tests/test_walkforward_records.py -v`

Expected: PASS, 15 tests.

- [x] **Step 7: Verify the module still compiles and the suite is green**

Run: `python3 -m py_compile scripts/walkforward_backtest.py && python3 -m pytest tests/ -q`

Expected: no compile error; suite passes.

- [x] **Step 8: Commit**

```bash
git add backend/scripts/walkforward_backtest.py backend/tests/test_walkforward_gate.py
git commit -m "fix: route the walkforward gate through the clustered scorer

Replaces sample_count-weighted per-fold averaging with pooled records
scored by score_by_tier, and fits the directional classifier per fold so
DA is measured on the estimator production serves rather than the sign of
the p50 regression. Reports both estimators. model_version bumped to
lgbm-v3-clustered because the metric definition changed."
```

---

### Task 4: Clean timing baseline

**Files:**
- Modify: `docs/architecture/model-optimization.md`
- No code change.

**Interfaces:**
- Consumes: nothing.
- Produces: a committed table of measured per-horizon, per-quantile training times. Task 12's speed claim is measured against it.

**Why:** spec #0 reports `465s` total retrain, `137s` fetch and `401s` for "14d + 30d DART" — figures that sum to more than the total. No speed claim can rest on them.

- [x] **Step 1: Run a full training pass with timing visible**

```bash
cd backend
SKIP_REGIMES=1 python3 scripts/forecast_prices.py --train-only 2>&1 | tee /tmp/baseline-train.log
```

- [x] **Step 2: Extract the phase timings**

```bash
grep -E "Training .*(ensemble|directional)|Optuna|Reusing cached HP|CV |fetch_price_history|elapsed|took|s\)" /tmp/baseline-train.log
```

Record, per horizon: HP-search seconds, per-quantile ensemble seconds, directional-classifier seconds, CV seconds.

If the log does not already emit per-phase seconds at INFO, add `time.time()` deltas around the three blocks in `forecaster.py` — the p50/p10/p90 ensemble loop at `:2947-2962`, the directional classifier at `:2987-2993`, and the Optuna block at `:2862-2937` — logging `f"  [timing] {label}: {elapsed:.1f}s"`. Keep these log lines; they are how Task 12 measures the after figure.

- [x] **Step 3: Run the predict path cold and time it**

```bash
VOTED_CACHE=0 python3 scripts/forecast_prices.py --predict-only 2>&1 | tee /tmp/baseline-predict.log
grep -E "\[timing\]|fetch|tail|engineer|Predict" /tmp/baseline-predict.log
```

- [x] **Step 4: Write the table into the docs**

Add a section to `docs/architecture/model-optimization.md`:

```markdown
### Measured baseline — 2026-08-04, pre-minimal-model

Local, 10-core Mac, `SKIP_REGIMES=1`. Supersedes the 465s/137s/401s figures
in spec #0, which came from different runs and do not sum.

| Phase | Horizon | Seconds |
|---|---|---|
| HP search (Optuna) | 3 / 7 / 14 / 30 | _fill from log_ |
| p10 ensemble (3 members) | 3 / 7 / 14 / 30 | _fill from log_ |
| p50 ensemble (3 members) | 3 / 7 / 14 / 30 | _fill from log_ |
| p90 ensemble (3 members) | 3 / 7 / 14 / 30 | _fill from log_ |
| Directional classifier | 3 / 7 / 14 / 30 | _fill from log_ |
| CV evaluate | 3 / 7 / 14 / 30 | _fill from log_ |
| **Total training** | — | _fill from log_ |
| Predict, cold (`VOTED_CACHE=0`) | — | _fill from log_ |
```

Replace every `_fill from log_` with the measured number. **A committed table still containing that placeholder is a failed task.**

- [x] **Step 5: Commit**

```bash
git add docs/architecture/model-optimization.md backend/models/forecaster.py
git commit -m "docs: measured training/predict timing baseline

Spec #0's 465s/137s/401s figures came from different runs and sum to more
than the total. This is one clean pass with per-phase timings, which is
what the minimal model's speed claim is measured against."
```

---

### Task 5: Compute the MDE and pre-register the acceptance bar

**Files:**
- Create: `backend/scripts/compute_mde.py`
- Modify: `docs/specs/2026-08-04-minimal-model-design.md`

**Interfaces:**
- Consumes: `run_walkforward(return_records=True)` (Task 3), `paired_da_difference` (Task 2)
- Produces: a committed per-horizon `MDE(horizon)` table and the bar text. Task 12 reads it.

**Method:** the MDE is estimated by pairing the current model against **itself under a different seed**. Two runs of the same design differ only by seed noise, so the width of that paired interval is the width the gate cannot see through — which is exactly the minimum detectable effect.

### The pinned measurement configuration (human decision, 2026-08-04)

All measurement in this plan — Task 5's two seed runs **and every arm in Task 12** — runs at:

```
--max-items 60  --step-days 120   (all 4 horizons)
```

**Measured cost basis.** At 60 items and the default `STEP_DAYS = 60` (27 folds), one horizon costs 402s and the two-seed four-horizon MDE run costs ~52 min. `--step-days 120` halves the folds to ~13, giving **~281 distinct dates** — still 14× the `MIN_FORECAST_DATES = 20` sufficiency floor — for **~25 min**. The human set this budget explicitly after the ~52 min option was measured.

**Why item count is not the lever.** Cutting items 500 → 60 (8.3×) cut wall time only 2.8× (1133s → 402s), because every fold trains on rows capped by `MAX_TRAIN_ROWS = 200_000` and the fold count is fixed by the date axis. Item count stops paying below ~100 items. Date count is *invariant* to it: both 60 and 500 items yielded exactly **562 dates**.

**The cost of this choice, stated honestly.** Fold count is the one remaining strong lever, and it is not free: dates scale with folds and CI width goes as 1/√dates, so ~281 dates instead of ~562 makes the MDE roughly **1.4× wider** — a more permissive bar. Record the measured MDE in the results document with this caveat attached, so nobody later reads the bar as tighter than it is.

### Amendment (human decision after a measured overrun): the MDE is split by horizon

The first attempt at a four-horizon two-seed run was killed at 16 minutes. Cause: the
controller's ~25 min estimate extrapolated from the 3d horizon and ignored the Task 4b
measurement showing **DART horizons cost 8.8× the GBDT ones** (297.1s vs 33.7s). Measured:
3d+7d = 7.4 min per seed; 14d+30d implied 15–65 min more, i.e. 45–145 min for both seeds.

Revised approach:

- **3d and 7d** — MDE measured now, on the current (arm A) design, via
  `compute_mde.py --max-items 60 --step-days 120 --horizons 3 7` (~15 min). This is the
  textbook construction: the noise floor of the design being compared against.
- **14d and 30d** — deferred until after Task 11 switches them to `gbdt`, at which point
  the same measurement is cheap. DART is precisely what the rewrite deletes, and the MDE
  needs *two* DART passes where the arm comparison needs only one.

⚠️ **The two halves are not the same construction, and the results document must not
present four uniform rows.** 3d/7d bars derive from the old design's noise floor; any
14d/30d bar measured post-rewrite derives from the *new* design's noise floor. That is
defensible for a paired comparison where one arm is the new design, but it is a different
quantity and must be labelled as such. If the 14d/30d MDE is never measured, those two
horizons ship with parity **explicitly untested** — and 14d currently has the best
production DA of the four (50.8%), making it the least comfortable one to leave unmeasured.

**Also fix `compute_mde.py`'s output behaviour.** It prints its JSON only after every
horizon and seed completes, so an interrupted run yields nothing at all — the 16 minutes
above were unrecoverable. It should emit each horizon's result as soon as that horizon is
paired.

⚠️ **Task 12's arms MUST use the identical `--max-items` and `--step-days`.** The bar is a paired quantity: it is only valid against arms measured on the same folds. An arm run at a different fold configuration is not comparable to this MDE, and pairing would silently drop to whatever `(item_id, forecast_date)` keys happen to overlap. `paired_da_difference` raises on zero overlap but will *not* warn about partial overlap.

- [x] **Step 1: Write the script**

Create `backend/scripts/compute_mde.py`:

```python
#!/usr/bin/env python3
"""Estimate the gate's minimum detectable effect, per horizon.

Runs the walkforward gate twice on the same folds, changing only the LightGBM
seed. Both runs are the same design, so the paired difference is pure noise
and the width of its date-clustered interval is the smallest real effect the
gate could distinguish from noise.

Usage:
    python3 scripts/compute_mde.py --max-items 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.paired_mde import paired_da_difference
from scripts import walkforward_backtest as wf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=500)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    args = parser.parse_args()

    out = {}
    runs = []
    for seed in (42, 7):
        # Pass the seed as an ARGUMENT. Do not rebind wf.FOLD_SEED: both runs
        # execute in this one process, so a global rebind leaks across them and
        # makes each run's configuration unreadable from its own call site.
        runs.append(wf.run_walkforward(
            max_items=args.max_items,
            horizons=args.horizons,
            skip_db=True,
            return_records=True,
            step_days=args.step_days,
            fold_seed=seed,
        ))

    for horizon_str in runs[0]["horizons"]:
        horizon = int(horizon_str)
        a = runs[0]["horizons"][horizon_str].get("records")
        b = runs[1]["horizons"][horizon_str].get("records")
        if not a or not b:
            out[horizon_str] = {"error": "no records returned"}
            continue
        out[horizon_str] = paired_da_difference(a, b)

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 1b: Make the fold step configurable**

`STEP_DAYS = 60` is a module constant. The pinned measurement configuration needs
`120` without editing source between runs, and Task 12 must be able to pass the
same value.

- Add a `step_days: int = STEP_DAYS` parameter to `run_walkforward`, and use it in
  place of the module constant in the `range(split_idx + 1, len(dates), ...)` fold
  loop. Keep the module constant as the default so nothing that omits the argument
  changes behaviour.
- Add `--step-days` to `main()`'s argument parser, defaulting to `STEP_DAYS`, and
  pass it through.
- `compute_mde.py` takes `--step-days` too and forwards it to both seed runs.

Do **not** mutate the module constant at runtime — a global rebind would leak
across the two seed runs inside one `compute_mde.py` process and make the
configuration of each run unclear from its own call site.

- [x] **Step 2: Make the seed configurable in the gate**

In `backend/scripts/walkforward_backtest.py`, add next to `QUANTILES` (line 45):

```python
# Seed for the per-fold boosters. scripts/compute_mde.py varies this to
# measure the gate's own noise floor: two runs of the same design differ only
# by seed, so the paired difference is the MDE.
FOLD_SEED = 42
```

Then replace both `"random_state": 42` occurrences in the fold loop — the quantile params dict and `_fit_direction_classifier`'s effect — by threading `FOLD_SEED` through. For the quantile params:

```python
                        "random_state": FOLD_SEED,
```

`_fit_direction_classifier` hardcodes `random_state=42` internally (`forecaster.py:3441`), so add a `random_state` parameter to it defaulting to 42, and pass `FOLD_SEED` from the gate:

```python
    def _fit_direction_classifier(self, X_train, y_train_ret, X_val, y_val_ret,
                                   boosting_type: str, tree_params: dict,
                                   horizon: Optional[int] = None,
                                   sigma_train=None, sigma_val=None,
                                   num_boost_round: int = 200,
                                   random_state: int = 42):
```

and at `:3439-3441`:

```python
        params.update(objective="multiclass", num_class=3, metric="multi_logloss",
                      boosting_type=boosting_type, verbosity=-1, n_jobs=-1,
                      random_state=random_state)
```

The default preserves production behaviour exactly.

- [x] **Step 3: Run it**

```bash
cd backend
python3 scripts/compute_mde.py --max-items 500 2>&1 | tee /tmp/mde.log
```

Expected: JSON with `mde_pp` and `n_dates` per horizon. If `n_dates < 2` for a horizon, that horizon cannot be gated — record it as unresolvable rather than inventing a bar.

- [x] **Step 4: Write the bar into the spec**

Add to `docs/specs/2026-08-04-minimal-model-design.md`, replacing the prose bar in Task 3 of Part 1:

```markdown
### The pre-registered acceptance bar (committed <date>, before any arm was run)

Measured by `scripts/compute_mde.py --max-items 500`: the same design under
two seeds, paired within (item, forecast_date), date-clustered bootstrap.

| Horizon | MDE (pp) | Dates | Bar: minimal model's paired DA difference vs arm A |
|---|---|---|---|
| 3 | _measured_ | _measured_ | `ci_lower_pp >= -MDE` |
| 7 | _measured_ | _measured_ | `ci_lower_pp >= -MDE` |
| 14 | _measured_ | _measured_ | `ci_lower_pp >= -MDE` |
| 30 | _measured_ | _measured_ | `ci_lower_pp >= -MDE` |

The bar is on the **classifier** DA figure, not the median's sign. A horizon
whose paired interval is unavailable (fewer than 2 shared dates) is recorded
as unresolvable; the design is not credited with passing it.
```

Replace every `_measured_` with the number from Step 3.

- [x] **Step 5: Commit**

```bash
git add backend/scripts/compute_mde.py backend/scripts/walkforward_backtest.py \
        backend/models/forecaster.py \
        docs/specs/2026-08-04-minimal-model-design.md
git commit -m "feat: measure the gate's MDE and pre-register the acceptance bar

Estimates the noise floor by running the same design under two seeds and
pairing within (item, forecast_date). The bar is committed before any arm
runs, on the classifier DA figure."
```

---

### Task 6: Add the Ridge and naive baseline arms

**Files:**
- Modify: `backend/scripts/walkforward_backtest.py`
- Test: `backend/tests/test_walkforward_arms.py`

**Interfaces:**
- Consumes: `_score_fold` (Task 3)
- Produces: `run_walkforward(arm=...)` accepting `"gbm" | "ridge" | "naive"`. Task 12 runs all of them.

**Why:** in M5, 92.5% of entrants failed to beat a simple off-the-shelf baseline. This repository has never measured its model against a naive arm on the same gate. If arm D is competitive, that outranks every other finding in this spec.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_walkforward_arms.py`:

```python
"""The gate must be able to score cheap baselines on identical folds."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import scripts.walkforward_backtest as wf


def test_run_walkforward_accepts_an_arm():
    assert "arm" in inspect.signature(wf.run_walkforward).parameters


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        wf.run_walkforward(arm="lstm", max_items=1, skip_db=True)


def test_naive_arm_predicts_the_trailing_return():
    X = np.zeros((3, 2))
    trailing = np.array([1.5, -2.0, 0.0])
    mid, low, high, classes = wf._naive_predict(trailing)
    assert mid == pytest.approx(trailing)
    # 0=down, 1=flat, 2=up with the 0.5% flat band.
    assert list(classes) == [2, 0, 1]
    assert np.all(low <= mid) and np.all(mid <= high)


def test_ridge_arm_returns_aligned_arrays():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 4))
    y_train = X_train[:, 0] * 2.0 + rng.normal(scale=0.1, size=200)
    X_val = rng.normal(size=(40, 4))
    mid, low, high, classes = wf._ridge_predict(X_train, y_train, X_val)
    assert mid.shape == (40,)
    assert low.shape == (40,) and high.shape == (40,)
    assert classes.shape == (40,)
    assert set(np.unique(classes)) <= {0, 1, 2}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_walkforward_arms.py -v`

Expected: FAIL — `arm` not in signature; `_naive_predict` / `_ridge_predict` missing.

- [x] **Step 3: Implement the two arms**

Add to `backend/scripts/walkforward_backtest.py` after `_score_fold`:

```python
ARMS = ("gbm", "ridge", "naive")

# Half-width of the baseline arms' interval, in percent return space. These
# arms exist to answer "what does the DA cost?", not to compete on interval
# coverage, so the band is a fixed placeholder and their interval_coverage
# figure must not be quoted.
BASELINE_BAND_PCT = 10.0


def _classes_from_returns(mid_returns_pct):
    """Direction classes from a point return, using production's flat band."""
    return ItemForecaster._direction_classes(
        np.asarray(mid_returns_pct, dtype=float),
        DIRECTION_FLAT_TOLERANCE_PCT,
    )


def _naive_predict(trailing_returns_pct):
    """Arm D: the forecast IS the trailing return over the same horizon."""
    mid = np.asarray(trailing_returns_pct, dtype=float)
    mid = np.nan_to_num(mid, nan=0.0, posinf=0.0, neginf=0.0)
    return (mid,
            mid - BASELINE_BAND_PCT,
            mid + BASELINE_BAND_PCT,
            _classes_from_returns(mid))


def _ridge_predict(X_train, y_train, X_val, alpha: float = 5.0):
    """Arm C: Ridge on the same features. Does the tree structure earn anything?"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # Ridge is scale-sensitive. Skipping this is the bug that produced a 100%
    # quantile-crossing rate in the shelved residual stacker
    # (forecaster.py:251-260) — do not remove the scaler.
    scaler = StandardScaler().fit(np.asarray(X_train, dtype=float))
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(scaler.transform(np.asarray(X_train, dtype=float)),
              np.asarray(y_train, dtype=float))
    mid = model.predict(scaler.transform(np.asarray(X_val, dtype=float)))
    return (mid,
            mid - BASELINE_BAND_PCT,
            mid + BASELINE_BAND_PCT,
            _classes_from_returns(mid))
```

Add the import near line 30:

```python
from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT, ItemForecaster
```

- [x] **Step 4: Branch the fold loop on the arm**

Change the signature:

```python
def run_walkforward(max_items=500, horizons=None, skip_db=False,
                    return_records=False, arm="gbm"):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
```

In the fold loop, replace the block from `preds = {}` down to `predicted_classes = clf.predict(...).argmax(axis=1)` with:

```python
                if arm == "naive":
                    trailing_col = f"return_{horizon}d"
                    if trailing_col not in val_df.columns:
                        logger.warning(f"    {trailing_col} absent; skipping naive fold")
                        continue
                    mid_ret, low_ret, high_ret, predicted_classes = _naive_predict(
                        val_df[trailing_col].to_numpy(dtype=float)
                    )
                elif arm == "ridge":
                    mid_ret, low_ret, high_ret, predicted_classes = _ridge_predict(
                        X_train.values, y_train.values, X_val.values
                    )
                else:
                    preds = {}
                    for q in QUANTILES:
                        params = _get_tuned_params(meta, horizon, q)
                        params.update({
                            "objective": "quantile",
                            "alpha": q,
                            "metric": "quantile",
                            "verbosity": -1,
                            "random_state": FOLD_SEED,
                            "n_jobs": -1,
                        })
                        dtrain = lgb.Dataset(X_train.values, y_train.values)
                        dval = lgb.Dataset(X_val.values, y_val.values, reference=dtrain)
                        model = lgb.train(
                            params, dtrain,
                            num_boost_round=200,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(20, verbose=False),
                                       lgb.log_evaluation(0)]
                        )
                        preds[q] = model.predict(X_val.values)

                    if len(preds) != 3:
                        continue

                    low_ret, high_ret = ItemForecaster._fix_quantile_crossing(
                        preds[0.1], preds[0.5], preds[0.9]
                    )
                    mid_ret = preds[0.5]

                    clf = forecaster._fit_direction_classifier(
                        X_train.values, y_train.values,
                        X_val.values, y_val.values,
                        _get_tuned_params(meta, horizon, 0.5).get("boosting_type", "gbdt"),
                        ItemForecaster._direction_tree_params(
                            {0.5: _get_tuned_params(meta, horizon, 0.5)}
                        ),
                        horizon=horizon,
                        sigma_train=None,
                        sigma_val=None,
                        num_boost_round=DIRECTION_NUM_ROUNDS,
                        random_state=FOLD_SEED,
                    )
                    predicted_classes = clf.predict(X_val.values).argmax(axis=1)
```

Then update the `_score_fold` call to use the arm-agnostic names:

```python
                fold_clf, fold_median = _score_fold(
                    item_ids=val_df["item_id"].to_numpy(),
                    forecast_dates=val_df["date"].to_numpy(),
                    base_prices=val_df["price"].to_numpy(dtype=float),
                    actual_returns_pct=y_val.to_numpy(dtype=float),
                    mid_returns_pct=mid_ret,
                    low_returns_pct=low_ret,
                    high_returns_pct=high_ret,
                    predicted_classes=predicted_classes,
                )
```

- [x] **Step 5: Add the CLI flag**

In `main()`:

```python
    parser.add_argument("--arm", choices=list(ARMS), default="gbm",
                        help="gbm (current design), ridge, or naive baseline")
```

and pass it:

```python
    report = run_walkforward(max_items=args.max_items, horizons=args.horizons,
                             skip_db=args.skip_db, arm=args.arm)
```

- [x] **Step 6: Run the tests**

Run: `python3 -m pytest tests/test_walkforward_arms.py tests/test_walkforward_gate.py -v`

Expected: PASS.

- [x] **Step 7: Verify the whole suite**

Run: `python3 -m py_compile scripts/walkforward_backtest.py && python3 -m pytest tests/ -q`

- [x] **Step 8: Commit**

```bash
git add backend/scripts/walkforward_backtest.py backend/tests/test_walkforward_arms.py
git commit -m "feat: add Ridge and naive baseline arms to the gate

In M5, 92.5% of entrants lost to a simple off-the-shelf baseline. This
repo has never measured its model against one on the same folds. Both
arms reuse the identical fold construction so the comparison is paired."
```

---

# PART 2 — The 8-model rewrite

Do not start until Part 1 is merged and Task 5's bar is committed.

---

### Task 7: The conformal module

**Files:**
- Create: `backend/models/conformal.py`
- Test: `backend/tests/test_conformal.py`

**Interfaces:**
- Consumes: numpy only.
- Produces: `NOMINAL_COVERAGE`, `ALPHA`, `sigma_bounds(sigma_raw) -> (float, float)`, `sigma_from_columns(price_std_60d, price, floor, cap) -> np.ndarray`, `calibrate(residuals_pct, sigma, alpha) -> float`, `band(mid_pct, sigma, q_hat) -> (np.ndarray, np.ndarray)`. Tasks 8 and 9 call these.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_conformal.py`:

```python
"""Locally-weighted split conformal band.

Replaces 24 p10/p90 quantile GBMs whose empirical coverage was 39-48% against
a nominal target the old code stated two ways (forecaster.py:3088 said 80%,
:3092 set alpha=0.10 for 90%). Nominal is pinned at 80% here.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.conformal import (
    ALPHA,
    NOMINAL_COVERAGE,
    band,
    calibrate,
    sigma_bounds,
    sigma_from_columns,
)


def test_nominal_coverage_is_pinned_at_eighty_percent():
    assert NOMINAL_COVERAGE == 0.80
    assert ALPHA == pytest.approx(0.20)


def test_sigma_is_a_coefficient_of_variation():
    # price_std_60d is in dollars; the band lives in return space, so sigma
    # must be scale-free or a $5000 knife and a $1 case get the same width.
    sigma = sigma_from_columns(
        price_std_60d=np.array([10.0, 0.10]),
        price=np.array([100.0, 1.0]),
        floor=0.001, cap=10.0,
    )
    assert sigma == pytest.approx([0.10, 0.10])


def test_sigma_is_clipped_to_the_bounds():
    sigma = sigma_from_columns(
        price_std_60d=np.array([0.0, 1000.0]),
        price=np.array([100.0, 1.0]),
        floor=0.01, cap=2.0,
    )
    assert sigma == pytest.approx([0.01, 2.0])


def test_sigma_falls_back_for_missing_or_zero_history():
    # PREDICT_MIN_HISTORY_DAYS = 14, so eligible items can have no 60d std.
    # A NaN reaching forecast_low would surface in the UI.
    sigma = sigma_from_columns(
        price_std_60d=np.array([np.nan, 0.0]),
        price=np.array([100.0, 50.0]),
        floor=0.02, cap=2.0,
        fallback=0.35,
    )
    assert sigma == pytest.approx([0.35, 0.35])
    assert np.all(np.isfinite(sigma))


def test_sigma_falls_back_when_price_is_nonpositive():
    sigma = sigma_from_columns(
        price_std_60d=np.array([1.0]),
        price=np.array([0.0]),
        floor=0.02, cap=2.0, fallback=0.35,
    )
    assert sigma == pytest.approx([0.35])


def test_sigma_bounds_are_the_first_and_ninety_ninth_percentiles():
    raw = np.concatenate([np.linspace(0.01, 1.0, 1000), [np.nan, np.inf]])
    floor, cap = sigma_bounds(raw)
    assert floor == pytest.approx(np.percentile(np.linspace(0.01, 1.0, 1000), 1))
    assert cap == pytest.approx(np.percentile(np.linspace(0.01, 1.0, 1000), 99))
    assert floor > 0


def test_calibrate_achieves_nominal_coverage_on_its_own_calibration_set():
    rng = np.random.default_rng(0)
    n = 5000
    sigma = rng.uniform(0.05, 0.5, size=n)
    # Heteroscedastic residuals: spread proportional to sigma.
    residuals = rng.normal(scale=sigma * 10.0, size=n)
    q_hat = calibrate(residuals, sigma, ALPHA)
    low, high = band(np.zeros(n), sigma, q_hat)
    covered = np.mean((residuals >= low) & (residuals <= high))
    assert covered == pytest.approx(NOMINAL_COVERAGE, abs=0.02)


def test_calibrate_is_scale_invariant_in_sigma():
    # Doubling sigma halves q_hat, leaving the band unchanged. This is the
    # property that makes the normalization meaningful rather than cosmetic.
    rng = np.random.default_rng(1)
    residuals = rng.normal(scale=1.0, size=2000)
    sigma = np.full(2000, 0.2)
    q1 = calibrate(residuals, sigma, ALPHA)
    q2 = calibrate(residuals, sigma * 2, ALPHA)
    assert q1 == pytest.approx(q2 * 2, rel=1e-9)


def test_band_width_varies_with_sigma():
    mid = np.zeros(3)
    sigma = np.array([0.1, 0.2, 0.4])
    low, high = band(mid, sigma, q_hat=5.0)
    widths = high - low
    assert widths[1] == pytest.approx(widths[0] * 2)
    assert widths[2] == pytest.approx(widths[0] * 4)


def test_band_is_ordered_and_centred_on_the_median():
    mid = np.array([3.0, -2.0])
    low, high = band(mid, np.array([0.2, 0.2]), q_hat=4.0)
    assert np.all(low <= mid) and np.all(mid <= high)
    assert (low + high) / 2 == pytest.approx(mid)


def test_calibrate_rejects_an_empty_calibration_set():
    with pytest.raises(ValueError, match="empty calibration set"):
        calibrate(np.array([]), np.array([]), ALPHA)


def test_calibrate_ignores_nonfinite_scores():
    residuals = np.array([1.0, 2.0, np.nan, 3.0])
    sigma = np.array([0.1, 0.1, 0.1, 0.0])
    q_hat = calibrate(residuals, sigma, ALPHA)
    assert np.isfinite(q_hat)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_conformal.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'models.conformal'`

- [x] **Step 3: Write the implementation**

Create `backend/models/conformal.py`:

```python
"""Locally-weighted split conformal prediction bands.

Replaces 24 p10/p90 quantile GBMs (303s of a 462s training budget) whose
empirical coverage was 39-48%. Their top feature was `price_std_60d` in 8 of
12 ensembles, i.e. they were learning "band width ~= recent volatility" — this
module states that relationship instead of fitting it.

The normalization is the point. A single global q_hat in return space would
give a $5,000 knife and a $1 case the same band width; dividing the
nonconformity score by a per-item sigma restores the item-level variation the
quantile models were supplying.

Pure: numpy only, no LightGBM, no I/O, no clock.
"""
from __future__ import annotations

import numpy as np

# Pinned. The old code stated two different targets: the comment at
# forecaster.py:3088 described (1-2*alpha) = 80% for a [p10, p90] base, while
# :3092 set alpha = 0.10 and logged "target coverage=90%". 80% is what the
# [p10, p90] band has always represented to the UI.
NOMINAL_COVERAGE = 0.80
ALPHA = 1.0 - NOMINAL_COVERAGE

# Percentiles of the cross-sectional sigma distribution used as clip bounds.
SIGMA_FLOOR_PCTL = 1.0
SIGMA_CAP_PCTL = 99.0


def sigma_bounds(sigma_raw) -> tuple[float, float]:
    """Clip bounds from the cross-sectional distribution of raw sigma.

    Computed on the training frame and persisted with the model: q_hat is
    calibrated against clipped sigmas, so serving must clip identically.
    """
    arr = np.asarray(sigma_raw, dtype=float)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        raise ValueError("cannot derive sigma bounds from an empty distribution")
    floor = float(np.percentile(finite, SIGMA_FLOOR_PCTL))
    cap = float(np.percentile(finite, SIGMA_CAP_PCTL))
    if floor <= 0:
        floor = float(finite.min())
    return floor, cap


def sigma_from_columns(price_std_60d, price, floor: float, cap: float,
                       fallback: float | None = None) -> np.ndarray:
    """Per-item volatility scale: the 60-day coefficient of variation.

    `price_std_60d` is in dollars, so it is divided by price to make the scale
    return-space and comparable across price tiers.

    PREDICT_MIN_HISTORY_DAYS = 14 means eligible items can carry a NaN or 0
    std (the rolling uses min_periods=1). Those rows take `fallback`, or the
    clip floor when no fallback is supplied. A NaN reaching forecast_low would
    surface in the UI.
    """
    std = np.asarray(price_std_60d, dtype=float)
    px = np.asarray(price, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = std / px

    default = float(fallback) if fallback is not None else float(floor)
    bad = ~np.isfinite(sigma) | (sigma <= 0) | ~np.isfinite(px) | (px <= 0)
    sigma = np.where(bad, default, sigma)
    return np.clip(sigma, floor, cap)


def calibrate(residuals_pct, sigma, alpha: float = ALPHA) -> float:
    """q_hat: the conformal quantile of normalized absolute residuals.

    `residuals_pct` are y - y_hat in percentage-return space, from
    out-of-fold predictions. Scores are |residual| / sigma, so q_hat is
    dimensionless and multiplies sigma at serve time.

    Uses the finite-sample corrected level ceil((n+1)(1-alpha))/n, which is
    what gives split conformal its distribution-free coverage guarantee.
    """
    res = np.asarray(residuals_pct, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    if res.size == 0:
        raise ValueError("empty calibration set: cannot compute q_hat")

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.abs(res) / sig
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("empty calibration set: no finite nonconformity scores")

    n = scores.size
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def band(mid_pct, sigma, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric band around the median, in percentage-return space.

    Cannot cross by construction, which is why predict() no longer needs
    _fix_quantile_crossing.
    """
    mid = np.asarray(mid_pct, dtype=float)
    half = float(q_hat) * np.asarray(sigma, dtype=float)
    return mid - half, mid + half
```

- [x] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_conformal.py -v`

Expected: PASS, 12 tests.

- [x] **Step 5: Commit**

```bash
git add backend/models/conformal.py backend/tests/test_conformal.py
git commit -m "feat: locally-weighted split conformal band module

Pure numpy. Replaces the 24 p10/p90 GBMs whose top feature was already
price_std_60d in 8 of 12 ensembles. Nominal coverage pinned at 80% —
the old code stated 80% in a comment and 90% in the alpha it used."
```

---

### Task 8: Calibrate at train time

**Files:**
- Modify: `backend/models/forecaster.py:317-320` (state), `:3085-3098` (the CQR block), `_cv_evaluate_horizon` (nonconformity scores)
- Test: `backend/tests/test_minimal_model_shape.py`

**Interfaces:**
- Consumes: `models.conformal.calibrate`, `sigma_bounds`, `sigma_from_columns`, `ALPHA`
- Produces: `self.conformal_calibration: Dict[int, float]` (unchanged type, new meaning — dimensionless multiplier of σ, not a percentage-point addend) and `self.sigma_clip: Dict[str, float]` with keys `floor`, `cap`, `fallback`. Task 9 reads both; Task 10 persists them.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_minimal_model_shape.py`:

```python
"""Shape guarantees for the 8-model forecaster.

The band now comes from conformal calibration around a single median model,
not from 24 p10/p90 GBMs. These tests guard the properties that made that
safe: item-varying width, a finite band for short-history items, ordering
without the crossing fix, and an artifact that cannot be loaded by the wrong
code version.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.conformal import ALPHA
from models.forecaster import ItemForecaster


def test_sigma_clip_defaults_are_present_and_finite():
    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    assert set(f.sigma_clip) == {"floor", "cap", "fallback"}
    assert 0 < f.sigma_clip["floor"] < f.sigma_clip["cap"]
    assert np.isfinite(f.sigma_clip["fallback"])


def test_alpha_matches_the_pinned_nominal_coverage():
    assert ALPHA == pytest.approx(0.20)
```

**Note on scope:** the config-constant assertions (`QUANTILES == [0.5]`, `N_ENSEMBLES == 1`, no DART, no residual stacking, the 8-model count) are **deliberately not in this file yet** — they are added by Task 11, in the same commit that makes them true. Writing them here would leave Tasks 8, 9 and 10 committing a red suite, which the Global Constraint forbids. Do not add them early.

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_minimal_model_shape.py -v`

Expected: FAIL — `_init_conformal_state` does not exist.

- [x] **Step 3: Add the conformal state initializer**

In `backend/models/forecaster.py`, replace the `conformal_calibration` state block at `:317-320`:

```python
        # Conformal calibration per horizon. NOTE: the meaning changed with the
        # minimal model — q_hat is now a DIMENSIONLESS multiplier of the
        # per-item sigma, not a percentage-point addend. Applied as:
        #   low = mid - q_hat * sigma_i,  high = mid + q_hat * sigma_i
        # The previous CQR scheme widened a [p10, p90] base interval by a
        # constant. Loading an old artifact into this code would silently
        # produce a band computed two different ways, which is why
        # MODEL_ARTIFACT_VERSION exists.
        self.conformal_calibration: Dict[int, float] = {}
        self._init_conformal_state()
```

And add the method near `_direction_tree_params`:

```python
    # Fallback sigma for items with no usable 60-day history, and the initial
    # clip bounds before a fit has measured the real distribution. Overwritten
    # by train() and restored by load_models().
    SIGMA_FALLBACK_DEFAULT = 0.15
    SIGMA_FLOOR_DEFAULT = 0.01
    SIGMA_CAP_DEFAULT = 2.0

    def _init_conformal_state(self) -> None:
        """Sigma clip bounds and fallback, persisted with the model.

        q_hat is calibrated against CLIPPED sigmas, so serving must clip
        identically or the coverage guarantee does not transfer.
        """
        self.sigma_clip: Dict[str, float] = {
            "floor": self.SIGMA_FLOOR_DEFAULT,
            "cap": self.SIGMA_CAP_DEFAULT,
            "fallback": self.SIGMA_FALLBACK_DEFAULT,
        }
```

- [x] **Step 4: Add a sigma helper that reads the feature frame**

Add next to `_init_conformal_state`:

```python
    def _sigma_for_rows(self, rows: pd.DataFrame) -> np.ndarray:
        """Per-item sigma for a feature frame, using the persisted clip bounds.

        Reads `price_std_60d` (engineered at :978) and `price`. Both are
        present on every frame that reaches training or prediction, so this
        adds no feature-engineering pass.
        """
        std = rows["price_std_60d"] if "price_std_60d" in rows.columns \
            else pd.Series(np.nan, index=rows.index)
        return conformal.sigma_from_columns(
            price_std_60d=std.to_numpy(dtype=float),
            price=rows["price"].to_numpy(dtype=float),
            floor=self.sigma_clip["floor"],
            cap=self.sigma_clip["cap"],
            fallback=self.sigma_clip["fallback"],
        )
```

Add the import near line 15:

```python
from models import conformal
```

- [x] **Step 5: Replace the CQR calibration block**

At `forecaster.py:3085-3098`, replace the whole `if nc_scores:` block with:

```python
                # Locally-weighted split conformal calibration. The old CQR
                # scheme derived nonconformity from the p10/p90 predictions;
                # those models no longer exist, so the score is the absolute
                # median residual normalized by the item's sigma.
                oof_resid = records_df["residual_pct"].to_numpy(dtype=float)
                oof_sigma = records_df["sigma"].to_numpy(dtype=float)
                q_hat = conformal.calibrate(oof_resid, oof_sigma, conformal.ALPHA)
                self.conformal_calibration[horizon] = q_hat
                logger.info(
                    f"  Conformal calibration: q_hat={q_hat:.4f} (dimensionless "
                    f"x sigma), n={len(oof_resid)}, alpha={conformal.ALPHA}, "
                    f"target coverage={conformal.NOMINAL_COVERAGE * 100:.0f}%"
                )
```

- [x] **Step 6: Unblock `_cv_evaluate_horizon` for a median-only fit**

⚠️ **This step is load-bearing. Without it, Task 11 breaks training silently-then-loudly:** `forecaster.py:4187` reads

```python
            if fold_p50 is None or fold_p10 is None or fold_p90 is None:
                continue
```

With `QUANTILES = [0.5]`, `fold_p10` and `fold_p90` are never assigned, so **every fold `continue`s**, `oof_records` comes back empty, the calibration block never runs, `conformal_calibration` stays `{}`, and `predict()` then raises the `RuntimeError` added in Task 9.

Replace that guard with:

```python
            if fold_p50 is None:
                continue
```

Then replace the CQR block at `:4190-4203`:

```python
            # Fix quantile crossing via isotonic regression (same as predict()).
            low_pred, high_pred = self._fix_quantile_crossing(
                fold_p10, fold_p50, fold_p90)
            current_prices = val_df["price"].values
            actual_returns = y_val.values

            # Conformal nonconformity scores: max(Q_low - y, y - Q_high) ...
            fold_scores = np.maximum(
                low_pred - actual_returns,
                actual_returns - high_pred,
            )
            fold_scores = np.clip(fold_scores, 0.0, None)
            nonconformity_scores.extend(fold_scores[~np.isnan(fold_scores)].tolist())
```

with:

```python
            current_prices = val_df["price"].values
            actual_returns = y_val.values

            # Locally-weighted split conformal: the score is the absolute
            # median residual normalized by the item's sigma. There is no
            # p10/p90 interval to measure exceedance against any more, and no
            # crossing to repair. `val_df` is the same row order as fold_p50,
            # so sigma aligns positionally.
            fold_sigma = self._sigma_for_rows(val_df)
            fold_residuals = actual_returns - fold_p50

            # The band the fold's own coverage is judged on, using the q_hat
            # from the PREVIOUS fit if one exists. On a first fit there is no
            # q_hat yet, so the band is omitted and coverage is computed after
            # calibration instead.
            prior_q_hat = self.conformal_calibration.get(horizon)
            if prior_q_hat is not None:
                low_pred, high_pred = conformal.band(
                    fold_p50, fold_sigma, prior_q_hat)
            else:
                low_pred = high_pred = None
```

- [x] **Step 7: Emit `residual_pct` and `sigma` on every OOF record**

In the per-row record loop at `:4263`, add the two fields the calibration block reads. `val_df` and the two fold arrays are already in scope:

```python
            for i in range(len(val_df)):
                mid_ret = float(fold_p50[i])
                curr = float(current_prices[i])
                actual_ret = float(actual_returns[i])
                # Consumed by conformal.calibrate via records_df.
                record_residual_pct = float(fold_residuals[i])
                record_sigma = float(fold_sigma[i])
```

and include in the appended dict:

```python
                    "residual_pct": record_residual_pct,
                    "sigma": record_sigma,
```

Every downstream read of `low_ret` / `high_ret` inside this loop must be guarded, because they are `None` on a first fit:

```python
                if low_pred is not None:
                    low_ret = float(low_pred[i])
                    high_ret = float(high_pred[i])
                else:
                    low_ret = high_ret = None
```

Any record field derived from them (a `range_pct`, an interval hit) must accept `None` — check what `_calibrate_confidence` reads before assuming it tolerates a missing key, and if it does not, compute those fields in a second pass after `q_hat` exists.

- [x] **Step 8: Drop `nonconformity_scores` from the return tuple**

The third return value is now unused. Change the return to `(oof_records, fold_metrics)`, delete `nonconformity_scores = []` at `:4120`, update the docstring at `:4099-4103`, and update the single call site at `:3075`:

```python
                oof_records, cv_metrics = self._cv_evaluate_horizon(
                    tdf, horizon, per_quantile_params)
```

Leaving it in place as a vestigial empty list would be the kind of dead field this codebase has been burned by.

- [x] **Step 5b: Set the clip bounds from the training frame**

Immediately before the per-horizon loop in `train()` (after `self.feature_cols` is finalized and the training frame exists), add:

```python
        # Sigma clip bounds come from the cross-sectional distribution of the
        # TRAINING frame, then are frozen into the artifact. q_hat is
        # calibrated against clipped sigmas.
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_raw = (df["price_std_60d"].to_numpy(dtype=float)
                         / df["price"].to_numpy(dtype=float))
        floor, cap = conformal.sigma_bounds(sigma_raw)
        finite = sigma_raw[np.isfinite(sigma_raw) & (sigma_raw > 0)]
        self.sigma_clip = {
            "floor": floor,
            "cap": cap,
            "fallback": float(np.median(finite)),
        }
        logger.info(
            f"Sigma clip: floor={floor:.5f} cap={cap:.5f} "
            f"fallback={self.sigma_clip['fallback']:.5f}"
        )
```

- [x] **Step 9: Run the shape tests that this task can satisfy**

Run: `python3 -m pytest tests/test_minimal_model_shape.py::test_sigma_clip_defaults_are_present_and_finite -v`

Expected: PASS. The constant tests still fail — Task 11 flips them.

- [x] **Step 10: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_minimal_model_shape.py
git commit -m "feat: calibrate the band with normalized split conformal at train time

q_hat becomes a dimensionless multiplier of a per-item sigma rather than a
percentage-point addend on a [p10, p90] base. Sigma clip bounds are
measured on the training frame and frozen into the artifact, because q_hat
is calibrated against clipped sigmas."
```

---

### Task 9: Build the band in predict()

**Files:**
- Modify: `backend/models/forecaster.py:3885-3922`
- Test: `backend/tests/test_minimal_model_shape.py` (extend)

**Interfaces:**
- Consumes: `self.conformal_calibration`, `self.sigma_clip`, `self._sigma_for_rows`, `conformal.band`
- Produces: `low_ret_arr`, `mid_ret_arr`, `high_ret_arr` with the same units and downstream contract as before, so blending, bias thresholds and the serving policy are untouched.

- [x] **Step 1: Write the failing test**

Append to `backend/tests/test_minimal_model_shape.py`:

```python
def test_predict_no_longer_calls_the_crossing_fix():
    import inspect

    src = inspect.getsource(ItemForecaster.predict)
    assert "_fix_quantile_crossing" not in src, (
        "a symmetric band around the median cannot cross; the crossing fix "
        "survives only for walkforward's baseline arm"
    )


def test_crossing_fix_still_exists_for_the_harness():
    # scripts/walkforward_backtest.py:294 uses it for arm A. Deleting it
    # breaks the baseline the minimal model is measured against.
    assert hasattr(ItemForecaster, "_fix_quantile_crossing")


def test_band_from_conformal_varies_by_item_and_is_finite():
    import pandas as pd

    f = ItemForecaster.__new__(ItemForecaster)
    ItemForecaster._init_conformal_state(f)
    f.sigma_clip = {"floor": 0.01, "cap": 2.0, "fallback": 0.25}

    rows = pd.DataFrame({
        "price": [100.0, 100.0, 50.0],
        # third row: no 60d history, the short-history case
        "price_std_60d": [5.0, 20.0, np.nan],
    })
    sigma = ItemForecaster._sigma_for_rows(f, rows)
    assert np.all(np.isfinite(sigma))
    assert sigma[1] > sigma[0]          # more volatile item, wider sigma
    assert sigma[2] == pytest.approx(0.25)   # fallback, not NaN

    from models.conformal import band
    low, high = band(np.zeros(3), sigma, q_hat=3.0)
    widths = high - low
    assert widths[1] > widths[0]
    assert np.all(np.isfinite(widths))
    assert np.all(low <= high)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_minimal_model_shape.py -k "crossing or band_from_conformal" -v`

Expected: FAIL — `predict` still contains `_fix_quantile_crossing`.

- [x] **Step 3: Replace the band construction**

In `forecaster.py`, replace the block from `:3890` (the `_fix_quantile_crossing` call) through `:3922` (the end of the old `q_hat` widening) with:

```python
            # Median from the single p50 model; band from locally-weighted
            # split conformal. There is no p10/p90 model to cross, so no
            # isotonic repair is needed.
            mid_ret_arr = p50_ret
            sigma_arr = self._sigma_for_rows(latest_rows)
            q_hat = self.conformal_calibration.get(horizon)
            if q_hat is None:
                raise RuntimeError(
                    f"no conformal calibration for horizon {horizon}d. The band "
                    f"cannot be constructed without q_hat; refusing to serve a "
                    f"forecast with a fabricated interval."
                )
            low_ret_arr, high_ret_arr = conformal.band(mid_ret_arr, sigma_arr, q_hat)

            # Momentum fallback for weak horizons: serve the trailing return as
            # the median, keeping the calibrated interval width.
            if horizon in self.MOMENTUM_FALLBACK_HORIZONS:
                mom_col = f"return_{horizon}d"
                if mom_col in latest_rows.columns:
                    momentum_ret = latest_rows[mom_col].to_numpy(dtype=float)
                    low_ret_arr, mid_ret_arr, high_ret_arr = self._recenter_on_momentum(
                        low_ret_arr, mid_ret_arr, high_ret_arr, momentum_ret)

            # Directional classifier: the served up/flat/down call + confidence.
            dir_class_arr = None
            dir_conf_arr = None
            clf = self.direction_models.get(horizon)
            if clf is not None:
                probs = clf.predict(X_horizon)
                dir_class_arr = probs.argmax(axis=1)
                dir_conf_arr = probs.max(axis=1)
```

Note the ordering: the classifier block moves below the band construction but stays above the blending at `:3928`, so every downstream correction still sees the same variables it did before.

- [x] **Step 4: Remove the p10/p90 prediction calls**

Above the replaced block, the code predicts each quantile. Restrict it to the median — with `QUANTILES == [0.5]` (Task 11) the loop naturally yields only `p50_ret`, but any code indexing `preds[0.1]` or `preds[0.9]` must be removed rather than left to `KeyError`. Search and fix:

```bash
grep -n "0\.1\]\|0\.9\]\|p10\|p90" models/forecaster.py
```

Every hit inside `predict()` and `train()` must either be removed or guarded by `if 0.1 in self.QUANTILES`. Prefer removal — a guard that is always false is dead code.

- [x] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_minimal_model_shape.py -v`

Expected: PASS for the crossing and band tests. Constant tests still fail until Task 11.

- [x] **Step 6: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_minimal_model_shape.py
git commit -m "feat: build the served band from conformal instead of p10/p90 models

predict() now centres a symmetric band on the median and scales its width
by the item's sigma, so the isotonic crossing repair is unnecessary. The
crossing fix survives as a static method for walkforward's baseline arm.
Refuses to serve rather than fabricating an interval when q_hat is absent."
```

---

### Task 10: Artifact version that fails loudly

**Files:**
- Modify: `backend/models/forecaster.py` (save/load metadata, `:4674-4680`, `:4750-4755`)
- Test: `backend/tests/test_minimal_model_shape.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ItemForecaster.MODEL_ARTIFACT_VERSION` and a `load_models` that raises `IncompatibleModelArtifact` on mismatch.

**Why this is not optional:** the recorded failure mode of this project is a green pipeline running something other than its design — the collectors, the CI outage, the drift retrain. A `.get("conformal_calibration", {})` default would let a pre-rewrite artifact load into post-rewrite code and serve a band computed two different ways, with no error anywhere.

- [x] **Step 1: Write the failing test**

Append to `backend/tests/test_minimal_model_shape.py`:

```python
def test_artifact_version_constant_exists():
    assert isinstance(ItemForecaster.MODEL_ARTIFACT_VERSION, int)
    assert ItemForecaster.MODEL_ARTIFACT_VERSION >= 2


def test_loading_a_pre_rewrite_artifact_raises(tmp_path, monkeypatch):
    import json

    from models.forecaster import IncompatibleModelArtifact

    model_dir = tmp_path / "saved_models"
    model_dir.mkdir()
    # A pre-rewrite meta.json: CQR floats, no artifact version, no sigma clip.
    (model_dir / "meta.json").write_text(json.dumps({
        "conformal_calibration": {"3": 4.21, "7": 6.02},
        "n_ensembles": 3,
        "quantiles": [0.1, 0.5, 0.9],
    }))

    f = ItemForecaster.__new__(ItemForecaster)
    with pytest.raises(IncompatibleModelArtifact, match="artifact version"):
        ItemForecaster._check_artifact_version(f, json.loads(
            (model_dir / "meta.json").read_text()
        ))


def test_current_artifact_version_is_accepted():
    f = ItemForecaster.__new__(ItemForecaster)
    meta = {"model_artifact_version": ItemForecaster.MODEL_ARTIFACT_VERSION}
    # Must not raise.
    ItemForecaster._check_artifact_version(f, meta)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_minimal_model_shape.py -k artifact -v`

Expected: FAIL — `MODEL_ARTIFACT_VERSION` and `IncompatibleModelArtifact` do not exist.

- [x] **Step 3: Add the exception and the version check**

Near the top of `forecaster.py`, after the imports:

```python
class IncompatibleModelArtifact(RuntimeError):
    """A saved model cache was written by an incompatible code version.

    Raised rather than defaulted around. Between the CQR scheme and the
    minimal model, `conformal_calibration` changed MEANING (percentage-point
    addend -> dimensionless sigma multiplier) without changing type, so a
    tolerant loader would silently serve a band computed two different ways.
    """
```

As a class constant on `ItemForecaster`:

```python
    # Bump when the MEANING of any persisted field changes, not just the set
    # of fields. v2: conformal_calibration became a dimensionless multiplier
    # of per-item sigma, p10/p90 models no longer exist, sigma_clip added.
    MODEL_ARTIFACT_VERSION = 2
```

And the check:

```python
    def _check_artifact_version(self, meta: dict) -> None:
        found = meta.get("model_artifact_version")
        if found == self.MODEL_ARTIFACT_VERSION:
            return
        raise IncompatibleModelArtifact(
            f"saved model artifact version {found!r} != expected "
            f"{self.MODEL_ARTIFACT_VERSION}. This cache predates the minimal "
            f"model, where conformal_calibration changed from a percentage-"
            f"point addend to a dimensionless sigma multiplier. Retrain "
            f"(mode=full) rather than loading it."
        )
```

- [x] **Step 4: Wire it into save and load**

In the metadata dict at `:4674`, add:

```python
            "model_artifact_version": self.MODEL_ARTIFACT_VERSION,
            "sigma_clip": dict(self.sigma_clip),
```

In `load_models`, immediately after `meta` is parsed and **before** any field is read:

```python
        self._check_artifact_version(meta)
```

Then replace the tolerant `conformal_calibration` restore at `:4750-4755` with a strict one, and restore `sigma_clip`:

```python
        # Strict: both fields are load-bearing for the band. A missing one
        # means the artifact is not what this code expects, which
        # _check_artifact_version should already have caught.
        self.conformal_calibration = {
            int(h): float(q) for h, q in meta["conformal_calibration"].items()
        }
        self.sigma_clip = {k: float(v) for k, v in meta["sigma_clip"].items()}
```

- [x] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_minimal_model_shape.py -k artifact -v`

Expected: PASS, 3 tests.

- [x] **Step 6: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_minimal_model_shape.py
git commit -m "feat: fail loudly on a pre-rewrite model artifact

conformal_calibration changed meaning without changing type, so a tolerant
loader would serve a band computed two different ways with no error. Adds
MODEL_ARTIFACT_VERSION = 2 and raises IncompatibleModelArtifact."
```

---

### Task 11: Collapse the grid and delete the dead code

**Files:**
- Modify: `backend/models/forecaster.py:148`, `:171-190`, `:261-262`, `:2966-2979`
- Test: `backend/tests/test_minimal_model_shape.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: the 8-model configuration. This is the commit that changes what gets trained.

- [x] **Step 1: Write the failing config tests**

These live here, not in Task 8, so that no intermediate commit ships a red suite. Append to `backend/tests/test_minimal_model_shape.py`:

```python
def test_quantiles_collapse_to_the_median_only():
    assert ItemForecaster.QUANTILES == [0.5]


def test_ensemble_is_a_single_member():
    assert ItemForecaster.N_ENSEMBLES == 1
    assert len(ItemForecaster.ENSEMBLE_SEEDS) == 1
    assert len(ItemForecaster.ENSEMBLE_FEATURE_FRACTIONS) == 1


def test_no_horizon_uses_dart():
    assert set(ItemForecaster.BOOSTING_TYPE_MAP.values()) == {"gbdt"}


def test_trained_model_count_is_eight():
    # 4 median GBMs + 4 directional classifiers. Guards accidental
    # re-expansion of the quantile/ensemble grid.
    expected = len(ItemForecaster.HORIZONS) * len(ItemForecaster.QUANTILES) \
        * ItemForecaster.N_ENSEMBLES
    assert expected == 4
    assert expected + len(ItemForecaster.HORIZONS) == 8


def test_residual_stacking_is_gone():
    assert not hasattr(ItemForecaster, "STACK_RESIDUALS")
    assert not hasattr(ItemForecaster, "RESIDUAL_ALPHA")


def test_dart_params_are_gone():
    assert not hasattr(ItemForecaster, "DART_PARAMS")
```

- [x] **Step 1b: Run them to verify they fail**

Run: `python3 -m pytest tests/test_minimal_model_shape.py -v`

Expected: FAIL on all six new tests — the constants still hold their old values.

- [x] **Step 2: Flip the constants**

In `backend/models/forecaster.py`:

```python
    HORIZONS = [3, 7, 14, 30]
    # The band no longer comes from quantile models — see models/conformal.py.
    # 24 p10/p90 GBMs cost 303s of a 462s budget for 39-48% coverage against
    # an 80% target, while their top feature was already price_std_60d.
    QUANTILES = [0.5]
```

```python
    # Single member. The 3-seed / 3-feature-fraction ensemble was estimated at
    # 0.3-0.5pp in docs/architecture/model-optimization.md, which is below the
    # MDE the gate reports, so it cannot be resolved in isolation. If the
    # minimal model misses its bar, restoring N_ENSEMBLES = 2 is the first
    # thing to try.
    N_ENSEMBLES = 1
    ENSEMBLE_SEEDS = [42]
    ENSEMBLE_FEATURE_FRACTIONS = [0.7]
```

```python
    # GBDT everywhere. DART's dropout was the single most expensive config
    # choice in this file and had never been measured against GBDT on a
    # trustworthy gate. Tested under the pre-registered bar; note 14d had the
    # best DA of the four horizons, so this is the change most likely to cost.
    WEAK_HORIZONS = [14, 30]
    BOOSTING_TYPE_MAP = {3: "gbdt", 7: "gbdt", 14: "gbdt", 30: "gbdt"}
```

- [x] **Step 3: Delete `DART_PARAMS` and its uses**

Delete the `DART_PARAMS` block at `:185-190`, then:

```bash
grep -n "DART_PARAMS\|drop_rate\|max_drop\|skip_drop\|xgboost_dart_mode\|uniform_drop" models/forecaster.py
```

Remove every hit. In `_direction_tree_params` (`:3458-3459`) the `keys` tuple lists `"drop_rate", "max_drop", "skip_drop"` — drop those three entries. In the Optuna search space, remove any DART-conditional suggestions.

Update the `SKIP_HP_HORIZONS` comment at `:181-184`, which currently justifies searching 14d/30d specifically so "DART's drop_rate/max_drop/skip_drop get tuned" — that reason is gone.

- [x] **Step 4: Delete residual stacking**

Delete `STACK_RESIDUALS` and `RESIDUAL_ALPHA` (`:261-262`), the `if self.STACK_RESIDUALS ...` / `elif` branch (`:2966-2979`), `self.residual_models` initialization, and its save/load handling.

```bash
grep -n "residual_model\|STACK_RESIDUALS\|RESIDUAL_ALPHA" models/forecaster.py
```

Expected after deletion: no hits.

- [x] **Step 5: Run the shape tests**

Run: `python3 -m pytest tests/test_minimal_model_shape.py -v`

Expected: PASS, all tests in the file — the two from Task 8, the three from Task 9, the three from Task 10, and the six added in Step 1 here.

- [x] **Step 6: Run the full suite**

Run: `python3 -m py_compile models/forecaster.py && python3 -m pytest tests/ -q`

Expected: green. Any test asserting three quantiles or DART now needs updating — update it to the new expectation rather than deleting it, and note in the commit message which ones moved.

- [x] **Step 7: Train once locally and confirm the model count**

```bash
SKIP_REGIMES=1 python3 scripts/forecast_prices.py --train-only 2>&1 | tee /tmp/minimal-train.log
ls models/saved_models/*.txt models/saved_models/*.pkl 2>/dev/null | wc -l
grep "\[timing\]" /tmp/minimal-train.log
```

Expected: 8 model files. Total training time in the tens of seconds against the Task 4 baseline.

- [x] **Step 8: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_minimal_model_shape.py
git commit -m "perf: collapse the model grid from 40 to 8

QUANTILES = [0.5], N_ENSEMBLES = 1, GBDT on every horizon. Deletes
DART_PARAMS and the dead residual-stacking Ridge branch (STACK_RESIDUALS
was already False)."
```

---

### Task 12: Run the arms and judge against the bar

**Files:**
- Create: `docs/changelog/2026-08-04-minimal-model-results.md`
- Modify: `docs/architecture/model-optimization.md`

**Interfaces:**
- Consumes: `run_walkforward(arm=..., return_records=True)`, `paired_da_difference`, the bar from Task 5.
- Produces: the pass/fail verdict.

- [x] **Step 1: Run all four arms on identical folds**

```bash
cd backend
for arm in gbm ridge naive; do
  python3 scripts/walkforward_backtest.py --arm $arm --max-items 500 --skip-db \
    2>&1 | tee /tmp/arm-$arm.log
done
```

Arm A (the current 40-model design) must be run from a checkout **before** Task 11 — its config constants no longer exist afterwards. Either run it first and save its records, or run it from `git stash` / a worktree at the Part 1 merge commit. Arm B is `--arm gbm` on the current checkout.

- [x] **Step 2: Pair each arm against arm A**

```bash
python3 - <<'PY'
import json, sys
sys.path.insert(0, ".")
from backtest.paired_mde import paired_da_difference
# Load the saved records for arm A and each candidate, then:
#   print(json.dumps(paired_da_difference(records_a, records_b), indent=2))
PY
```

Record, per horizon and per arm: `mean_diff_pp`, `ci_lower_pp`, `ci_upper_pp`, `n_dates`.

- [x] **Step 3: Apply the bar**

For each horizon, arm B passes if `ci_lower_pp >= -MDE(horizon)` from the Task 5 table. Do not adjust the bar now. If a horizon fails, the first remedy named in the spec is `N_ENSEMBLES = 2`.

- [x] **Step 4: Write the results document**

Create `docs/changelog/2026-08-04-minimal-model-results.md` containing:

- the Task 5 bar, quoted, with its commit hash as evidence it predates these numbers
- a table of per-horizon paired differences for arms B, C and D against arm A
- the verdict per horizon: pass / fail / unresolvable (fewer than 2 shared dates)
- measured training time before and after, from Task 4's table and `/tmp/minimal-train.log`
- measured band coverage before and after
- a sentence stating plainly what the MDE means, in the form: "at 30d the gate resolves 7pp, so the claim this supports is 'not worse by more than 7pp', not 'equivalent'"
- **if arm D (naive) is competitive, that finding goes at the top of the document**, ahead of the speed result

Do not write `interval_coverage` or any `conf_*` figure for arms C and D — their bands are a fixed `BASELINE_BAND_PCT` placeholder and their confidence is uniform `low`, so those fields are structurally meaningless.

- [x] **Step 5: Update the architecture doc**

In `docs/architecture/model-optimization.md`, update the **Models**, **Quantiles**, **Inference** and **Production DA** rows. Where a figure is retired rather than superseded, mark it with the ⚠️ convention already used in that file rather than deleting it.

- [x] **Step 6: Commit**

```bash
git add docs/changelog/2026-08-04-minimal-model-results.md \
        docs/architecture/model-optimization.md
git commit -m "docs: minimal-model results against the pre-registered bar"
```

---

# PART 3 — Item-coverage reinvestment

Do not merge with Part 2. Part 2 is expected to cost a little accuracy and Part 3 to buy some back; together, a null result is uninterpretable.

---

### Task 13: Propagate the row budget and raise it

**Files:**
- Modify: `backend/scripts/forecast_prices.py:209`, `backend/models/forecaster.py:2417`
- Test: `backend/tests/test_training_item_coverage.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `train(max_rows=...)` reaching `_subsample_for_features(max_feature_rows=...)`.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_training_item_coverage.py`:

```python
"""max_rows must reach the feature-row budget.

train(max_rows=700_000) was passed at forecast_prices.py:209 but never
reached max_feature_rows (default 100_000), so the model learned from 133 of
7,879 items and then served forecasts for 8,691.
"""
from __future__ import annotations

import inspect

from models.forecaster import ItemForecaster


def test_train_accepts_and_forwards_max_rows():
    sig = inspect.signature(ItemForecaster.train)
    assert "max_rows" in sig.parameters

    src = inspect.getsource(ItemForecaster.train)
    assert "max_feature_rows=" in src, (
        "train must forward its row budget to the feature subsample, or "
        "max_rows is silently ignored"
    )


def test_engineer_features_budget_is_not_hardcoded_to_100k():
    src = inspect.getsource(ItemForecaster.train)
    assert "max_feature_rows=100_000" not in src
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_training_item_coverage.py -v`

Expected: FAIL — `max_feature_rows=` absent from `train`'s source.

- [x] **Step 3: Thread the budget through**

In `train()`, forward its `max_rows` argument to the call that reaches `_subsample_for_features`, so the budget passed at `forecast_prices.py:209` is honoured. Keep `100_000` as the parameter default at `:2417` so nothing that omits the argument changes behaviour.

- [x] **Step 4: Measure the item count at each budget**

```bash
cd backend
for budget in 100000 400000 700000; do
  echo "=== budget $budget ==="
  SKIP_REGIMES=1 TRAIN_MAX_ROWS=$budget python3 scripts/forecast_prices.py \
    --train-only 2>&1 | grep -E "Stratified subsample|\[timing\] total"
done
```

Record items selected and total training seconds per budget. Pick the largest budget whose training time is acceptable.

- [x] **Step 5: Re-run the gate at the chosen budget**

Run arm B again with the raised budget and pair it against arm B at the 100k budget — same MDE, same bar, one variable moved.

- [x] **Step 6: Commit**

```bash
git add backend/scripts/forecast_prices.py backend/models/forecaster.py \
        backend/tests/test_training_item_coverage.py \
        docs/changelog/2026-08-04-minimal-model-results.md
git commit -m "fix: propagate the training row budget to the feature subsample

max_rows=700_000 at forecast_prices.py:209 never reached max_feature_rows,
so the model learned from 133 of 7,879 items while serving 8,691."
```

---

## Self-review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Part 1 Task 1 — route through `score_cohort` | 1, 3 |
| Part 1 Task 1b — score the classifier production serves | 1, 3 |
| Part 1 Task 2 — clean timing baseline | 4 |
| Part 1 Task 3 — MDE + pre-registered bar | 2, 5 |
| Part 1 Task 4 — four arms | 6, 12 |
| Part 2 — model inventory 40→8 | 11 |
| Part 2 — CQR → normalized split conformal | 7, 8, 9 |
| Part 2 — pin nominal coverage at 80% | 7 |
| Part 2 — deletions (crossing fix, Ridge, DART) | 9, 11 |
| Part 2 Risk 1 — artifact fails loudly | 10 |
| Part 2 Risk 2 — σ fallback for short history | 7, 9 |
| Part 2 Risk 3 — ensemble 3→1 bundled, remedy named | 11 |
| Part 3 — `max_feature_rows` propagation | 13 |
| All spec tests | 1, 3, 6, 7, 8, 9, 10, 11, 13 |

**Deviations from the spec, both deliberate:**

1. `_fix_quantile_crossing` is **not deleted** — `walkforward_backtest.py:294` needs it for the baseline arm. Removed from `predict()` only. Documented at the top of this plan and asserted by `test_crossing_fix_still_exists_for_the_harness`.
2. MDE is measured by a **paired same-design two-seed run** rather than from a single arm's CI width. Unpaired, the interval would be so wide no design could pass; pairing is valid only because the arms share folds, which they do.

**Type consistency:** `conformal_calibration` stays `Dict[int, float]` with changed meaning (hence Task 10). `sigma_clip` is `Dict[str, float]` with keys `floor`/`cap`/`fallback`, written by Task 8 and read by Tasks 9 and 10. `fold_records` returns the exact key set `score_cohort` reads, verified against `backtest_accuracy.py:546-562`. `_score_fold` returns `(classifier_records, median_sign_records)` in that order in Tasks 3, 6 and 12. `_cv_evaluate_horizon` returns `(oof_records, fold_metrics)` after Task 8 Step 8 — two values, not three.

**The hard blocker, found while planning and handled in Task 8 Step 6:** `forecaster.py:4187` skips any CV fold where `fold_p10` or `fold_p90` is `None`. Under `QUANTILES = [0.5]` that is *every* fold, which empties `oof_records`, which means no `q_hat`, which makes `predict()` raise. Flipping the constants in Task 11 without Task 8 Step 6 produces a forecaster that trains "successfully" and then cannot serve. The two tasks are ordered accordingly and Task 11 Step 7 trains locally to confirm.

**One residual risk the implementer must resolve in place:** Task 8 Step 7 guards `low_ret`/`high_ret` as `None` on a first fit, because the fold band needs a `q_hat` that does not exist yet. `_calibrate_confidence` consumes these records and was not read line-by-line. If it requires a numeric `range_pct`, the fix is a second pass over the pooled records after `q_hat` is computed — not a fabricated placeholder width.
