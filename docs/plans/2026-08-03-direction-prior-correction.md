# Directional Prior Correction Implementation Plan

> # 🛑 EXECUTED AND CLOSED (2026-08-03) — Tasks 5–7 were NOT built
>
> **Do not execute Tasks 5, 6 or 7.** Task 4's Step 0 gate ran and returned
> **STOP**: the weighted training class prior is balanced at every horizon
> (down/up 1.035, 1.040, 1.034, 1.023 against a 1.20 threshold), so the inherited
> prior does not explain the observed down-bias. The correction was deliberately
> never implemented.
>
> Tasks 1–4 shipped. The result is written up in
> `docs/changelog/2026-08-03-direction-prior-diagnostic.md`, which is the
> authoritative record.
>
> **Everything below describing the correction is a design refuted before
> implementation, kept as a record of what was tried.** Passages written before the
> gate fired — the zero-sum offset, `DIRECTION_PRIOR_TAU`,
> `_apply_direction_prior`, `direction_priors.json`, the `p_flat` and
> `dir_conf_arr` constraints — describe code that does not exist and must not be
> created on the strength of this document. Re-running
> `scripts/diagnose_direction_prior.py` after a future retrain is the way to
> revisit the question, since the prior is a property of each training window.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the directional classifier's inherited up/down class prior at serve time, and stop the outcome-fitted bias loop from fitting on a sample that spans two market days.

**Architecture:** Two independent components. Component 2 (Tasks 1–2) guards the existing outcome-fitted loop and ships unconditionally. Component 1 (Tasks 3–7) adds a zero-sum logit offset to the classifier's up/down probabilities before `argmax`, gated behind a diagnostic (Task 4) that can refute the hypothesis and stop the work. All new logic goes in pure static helpers on `ItemForecaster` so it is unit-testable without training a model or touching a DB.

**Tech Stack:** Python 3, LightGBM, pandas, numpy, DuckDB, pytest, SQLAlchemy.

## Global Constraints

- **No new training time.** The prior is a weighted count taken at fitting time; serving adds one vector operation. No task may add a training pass, an Optuna trial, or a feature rebuild to the training path.
- **No new features and no new model architecture.** `docs/research/accuracy-opportunities.md` is closed. This is serving-path work only.
- **τ = 1.0**, fixed, as `ItemForecaster.DIRECTION_PRIOR_TAU`. Do not add a per-horizon τ map and do not sweep it.
- **`p_flat` is never touched.** The offset is zero-sum over down/up only. The mover weighting (`DIRECTION_MOVER_WEIGHT_MAP = {3: 3.0, 7: 3.0, 14: 3.0, 30: 3.0}`) is deliberate and must survive.
- **`dir_conf_arr` stays `probs.max(axis=1)` on the *unadjusted* vector.** Confidence must be bit-identical before and after this change.
- **One threshold constant.** Reuse `MIN_FORECAST_DATES` from `backend/backtest/scoring.py:38`. Do not define a second date threshold.
- **Never point tests at the real `model_dir`.** Use the existing `forecaster` fixture in `tests/test_forecaster.py:25`, which uses `tmp_path_factory`. `models/saved_models/` holds gitignored, unrecoverable production artifacts.
- All commands run from the `backend/` directory.

## File Structure

| File | Responsibility |
|---|---|
| `backend/models/forecaster.py` (modify) | All four new pure helpers, the `predict()` wiring, the persistence round-trip, and the date guard. Follows the existing pattern of pure statics near `_direction_classes` (`:3230`). |
| `backend/tests/test_direction_prior.py` (create) | Component 1: prior computation, offset math, correction application, persistence. |
| `backend/tests/test_bias_fit_date_guard.py` (create) | Component 2: date coverage guard and provenance-versioned load. |
| `backend/scripts/diagnose_direction_prior.py` (create) | Task 4's read-only diagnostic. Standalone; imports nothing new into the serving path. |
| `.gitignore` (modify) | Ignore the new `direction_priors.json` production artifact. |

---

### Task 1: Date-coverage guard on the outcome fit

`update_bias_corrections_from_outcomes` (`forecaster.py:437`) fits per-tier thresholds from `forecast_outcomes`. Its only guards are row counts — `n < 20` (`:496`) and `MIN_THRESHOLD_SAMPLE = 100` (`:489`). No row count can distinguish 11,000 independent rows from 11,000 rows on two forecast dates, which is what the table actually holds.

**Files:**
- Modify: `backend/models/forecaster.py` — new static `_has_date_coverage`; add `forecast_date` to the SELECT at `:465-472`; guard inside the `groupby` loop at `:494`
- Test: `backend/tests/test_bias_fit_date_guard.py` (create)

**Interfaces:**
- Consumes: `MIN_FORECAST_DATES` from `backtest.scoring` (module is pure — `math`, `collections`, `numpy` only — so this import cannot create a cycle)
- Produces: `ItemForecaster._has_date_coverage(forecast_dates) -> bool`

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_bias_fit_date_guard.py`:

```python
"""The outcome-fitted bias loop must see date clustering, not just row counts.

update_bias_corrections_from_outcomes matches the predicted up/down/flat split
to the observed base rate. Every item sharing a forecast_date is exposed to the
same market-wide move, so 11,000 rows on two opposite-direction days describe
those two days, not the market. Row-count guards cannot detect this: the real
cohort is 11,000 rows and 2 dates.
"""
from __future__ import annotations

from datetime import date

from backtest.scoring import MIN_FORECAST_DATES
from models.forecaster import ItemForecaster


def _dates(n_distinct, rows_each=500):
    """rows_each rows on each of n_distinct consecutive days."""
    out = []
    for i in range(n_distinct):
        out.extend([date(2026, 1, 1 + i)] * rows_each)
    return out


def test_two_dates_lack_coverage_regardless_of_row_count():
    # The real cohort shape: five-figure row count, two market days.
    assert ItemForecaster._has_date_coverage(_dates(2, rows_each=5500)) is False


def test_coverage_is_met_at_the_minimum():
    assert ItemForecaster._has_date_coverage(_dates(MIN_FORECAST_DATES, 1)) is True


def test_one_below_the_minimum_lacks_coverage():
    assert ItemForecaster._has_date_coverage(_dates(MIN_FORECAST_DATES - 1, 1)) is False


def test_null_dates_never_count_toward_coverage():
    # Rows predating the forecast_date backfill must not manufacture coverage.
    dates = _dates(MIN_FORECAST_DATES, 1) + [None] * 5000
    assert ItemForecaster._has_date_coverage(dates) is True
    assert ItemForecaster._has_date_coverage([None] * 5000) is False


def test_empty_input_lacks_coverage():
    assert ItemForecaster._has_date_coverage([]) is False


def test_the_fit_and_the_headline_share_one_constant():
    """A second, drifting threshold is the failure this guards against.

    scoring.MIN_FORECAST_DATES gates what gets *reported*; the guard gates
    what gets *fitted*. If they ever diverge, production could fit
    thresholds on a cohort the same codebase refuses to quote.
    """
    from backtest import scoring
    from models import forecaster as fc

    assert fc.MIN_FORECAST_DATES is scoring.MIN_FORECAST_DATES
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_bias_fit_date_guard.py -v
```

Expected: FAIL — `AttributeError: type object 'ItemForecaster' has no attribute '_has_date_coverage'`.

- [x] **Step 3: Add the helper**

In `backend/models/forecaster.py`, add the module-level import alongside the existing imports:

```python
from backtest.scoring import MIN_FORECAST_DATES
```

Add this static method immediately after `_direction_classes` (`:3236`), keeping the pure helpers together:

```python
    @staticmethod
    def _has_date_coverage(forecast_dates) -> bool:
        """True when distinct non-null forecast dates reach MIN_FORECAST_DATES.

        Row count cannot substitute for this. Items sharing a forecast_date
        share one market-wide move, so a five-figure cohort on two dates is
        nearer two observations than 11,000 — see
        docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md.
        Reuses the reporting threshold so the value we fit on and the value we
        report cannot drift apart.
        """
        distinct = {d for d in forecast_dates if d is not None and not pd.isna(d)}
        return len(distinct) >= MIN_FORECAST_DATES
```

- [x] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_bias_fit_date_guard.py -v
```

Expected: 6 passed.

- [x] **Step 5: Wire the guard into the fit**

In `update_bias_corrections_from_outcomes`, add `forecast_date` to the SELECT (`:465-472`):

```python
            rows = self.db.execute(text("""
                SELECT fo.horizon_days, fo.current_price, fo.predicted_price_mid,
                       fo.direction_actual, fo.forecast_date
                FROM forecast_outcomes fo
                WHERE fo.model_version LIKE 'lgbm-v3%'
                  AND fo.current_price > 0
                  AND fo.direction_actual IS NOT NULL
            """)).fetchall()
```

Add the column to the DataFrame constructor (`:481-484`):

```python
        df = pd.DataFrame(rows, columns=[
            "horizon_days", "current_price", "predicted_price_mid",
            "direction_actual", "forecast_date",
        ])
```

Then guard each group, immediately after the `if n < 20: continue` check (`:496-497`):

```python
            if not self._has_date_coverage(g["forecast_date"]):
                n_dates = g["forecast_date"].nunique(dropna=True)
                logger.warning(
                    f"  Threshold[{horizon}d, {tier}]: refusing to fit — "
                    f"{n_dates} distinct forecast date(s) < {MIN_FORECAST_DATES} "
                    f"(n={n} rows). Leaving thresholds at defaults."
                )
                continue
```

`continue` leaves this tier at whatever `_set_default_bias_corrections` /
`_fill_missing_threshold_defaults` established, and leaves its `ewma_state`
counter untouched so a later well-covered fit starts cold.

- [x] **Step 6: Verify nothing else regressed**

```bash
python -m pytest tests/test_forecaster.py tests/test_backtest_scoring.py tests/test_accuracy_date_clustering.py -q
```

Expected: all pass. If a test asserted the old fitting behaviour on a low-date fixture, read it before changing it — if its scenario is now legitimately refused, update the test to assert the refusal and correct its docstring, as `test_gate_is_not_diluted_by_the_frozen_majority` was handled in `docs/changelog/2026-08-03-unresolvable-gate-denominator.md`. Do not weaken an assertion to make it pass.

- [x] **Step 7: Commit**

```bash
git add models/forecaster.py tests/test_bias_fit_date_guard.py
git commit -m "fix: refuse to fit bias thresholds without forecast-date coverage

update_bias_corrections_from_outcomes guarded only on row count, which
cannot see that the 11,000-row cohort spans two opposite-direction market
days. Reuses MIN_FORECAST_DATES so the fitted and reported thresholds
share one constant."
```

---

### Task 2: Discard pre-guard thresholds on load

`models/saved_models/bias_corrections.json` is **gitignored** (`.gitignore:70`), so the rail-clamped values in it cannot be fixed by a committed file edit — a hand-edit would not reproduce on CI, on prod, or on another machine. Every threshold currently in that file was fitted before Task 1's guard existed, on a two-date cohort, and several are pinned at the `±3.0` clamp rail (`:523-524`): horizon 30 holds `t_down = -3.00, t_up = -2.92` for `$1-5`.

The fix is a provenance version. Values written by guarded code are trusted; values with no version field predate the guard and are discarded on load. This is self-healing everywhere and needs no manual intervention.

This matters despite the thresholds being unreachable while classifiers load: `:4595` catches a corrupt classifier with a `logger.warning` and falls through to exactly these thresholds, so one bad file promotes a rail-clamped fit to the served path silently.

**Files:**
- Modify: `backend/models/forecaster.py` — `BIAS_FIT_SCHEMA_VERSION`; `_load_bias_corrections` (`:381-399`); `_save_bias_corrections` (`:401-410`)
- Test: `backend/tests/test_bias_fit_date_guard.py` (append)

**Interfaces:**
- Consumes: `ItemForecaster._has_date_coverage` (Task 1)
- Produces: `ItemForecaster.BIAS_FIT_SCHEMA_VERSION = 2`; `bias_corrections.json` gains a top-level `"schema_version"` key

- [x] **Step 1: Write the failing test**

Append to `backend/tests/test_bias_fit_date_guard.py`:

```python
import json

import pytest
from unittest.mock import MagicMock


DEFAULT_T = 0.5  # DIRECTION_FLAT_TOLERANCE_PCT


def _write_corrections(model_dir, payload):
    (model_dir / "bias_corrections.json").write_text(json.dumps(payload))


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "saved_models"
    d.mkdir()
    return d


def _load(model_dir):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(model_dir))


def test_unversioned_thresholds_are_discarded(model_dir):
    """The real production file: rail-clamped values, no provenance."""
    _write_corrections(model_dir, {
        "corrections": {},
        "thresholds": {"30": {"$1-5": {"t_down": -3.0, "t_up": -2.92}}},
        "ewma_state": {"30": {"$1-5": 2}},
    })
    f = _load(model_dir)
    assert f.bias_thresholds[30]["$1-5"] == {"t_down": -DEFAULT_T, "t_up": DEFAULT_T}
    assert f.bias_ewma_state.get(30, {}).get("$1-5", 0) == 0


def test_versioned_thresholds_are_kept(model_dir):
    _write_corrections(model_dir, {
        "schema_version": ItemForecaster.BIAS_FIT_SCHEMA_VERSION,
        "corrections": {},
        "thresholds": {"30": {"$1-5": {"t_down": -0.4, "t_up": 0.6}}},
        "ewma_state": {"30": {"$1-5": 2}},
    })
    f = _load(model_dir)
    assert f.bias_thresholds[30]["$1-5"] == {"t_down": -0.4, "t_up": 0.6}
    assert f.bias_ewma_state[30]["$1-5"] == 2


def test_save_stamps_the_schema_version(model_dir):
    f = _load(model_dir)
    f._save_bias_corrections()
    data = json.loads((model_dir / "bias_corrections.json").read_text())
    assert data["schema_version"] == ItemForecaster.BIAS_FIT_SCHEMA_VERSION


def test_a_discarded_load_survives_a_save_round_trip(model_dir):
    """Discard then save must not write the rails back out."""
    _write_corrections(model_dir, {
        "corrections": {},
        "thresholds": {"30": {"$1-5": {"t_down": -3.0, "t_up": -2.92}}},
        "ewma_state": {"30": {"$1-5": 2}},
    })
    f = _load(model_dir)
    f._save_bias_corrections()
    reloaded = _load(model_dir)
    assert reloaded.bias_thresholds[30]["$1-5"] == {"t_down": -DEFAULT_T, "t_up": DEFAULT_T}
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_bias_fit_date_guard.py -v -k "schema or discard or versioned"
```

Expected: FAIL — `AttributeError: ... has no attribute 'BIAS_FIT_SCHEMA_VERSION'`.

- [x] **Step 3: Implement the version check**

Add the class constant near `DIRECTION_MOVER_WEIGHT_MAP` (`:232`):

```python
    # Bumped when a fitting-logic change invalidates stored thresholds.
    # v2 (2026-08-03): thresholds must come from a date-coverage-guarded fit.
    # Unversioned files predate the guard, were fitted on a two-date cohort,
    # and are discarded on load.
    BIAS_FIT_SCHEMA_VERSION = 2
```

In `_load_bias_corrections` (`:381`), gate the threshold and EWMA restore on the version. Keep `corrections` loading as-is — it is the deprecated additive path (`:315`) and this version applies to thresholds:

```python
    def _load_bias_corrections(self):
        path = os.path.join(self.model_dir, "bias_corrections.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.bias_corrections = {int(k): v for k, v in data.get("corrections", {}).items()}
                version = int(data.get("schema_version", 0))
                if version < self.BIAS_FIT_SCHEMA_VERSION:
                    logger.warning(
                        f"  bias_corrections.json is schema v{version} < "
                        f"v{self.BIAS_FIT_SCHEMA_VERSION}; discarding stored "
                        f"thresholds. Pre-v2 files were fitted without "
                        f"forecast-date coverage and can sit at the ±3.0 clamp "
                        f"rail. Reverting to defaults until a guarded fit runs."
                    )
                    self.bias_thresholds = {}
                    self.bias_ewma_state = {}
                else:
                    raw_thresholds = data.get("thresholds", {})
                    self.bias_thresholds = {int(k): v for k, v in raw_thresholds.items()}
                    self.bias_ewma_state = {int(k): v for k, v in data.get("ewma_state", {}).items()}
                self._fill_missing_threshold_defaults()
                logger.info(f"  Loaded bias corrections for {len(self.bias_corrections)} horizons, "
                            f"thresholds for {len(self.bias_thresholds)} horizons")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"  Corrupt bias_corrections.json ({e}), using defaults")
                self._set_default_bias_corrections()
        else:
            logger.info("  No bias_corrections.json found, using defaults")
            self._set_default_bias_corrections()
```

Preserve the existing call order and the `_fill_missing_threshold_defaults()` call — it is what turns the emptied dict into full per-tier defaults. Do not remove the `logger.info` lines; the daily job's logs are read.

In `_save_bias_corrections` (`:401`), stamp the version:

```python
        data = {
            "schema_version": self.BIAS_FIT_SCHEMA_VERSION,
            "corrections": {str(k): v for k, v in self.bias_corrections.items()},
            "thresholds": {str(k): v for k, v in self.bias_thresholds.items()},
            "ewma_state": {str(k): v for k, v in self.bias_ewma_state.items()},
        }
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_bias_fit_date_guard.py -v
```

Expected: 10 passed.

- [x] **Step 5: Confirm the real production file is rejected**

```bash
python -c "
import json
d = json.load(open('models/saved_models/bias_corrections.json'))
print('schema_version:', d.get('schema_version', 0))
print('30d \$1-5:', d['thresholds']['30']['\$1-5'])
"
```

Expected: `schema_version: 0` and the rail-clamped pair — confirming the live file takes the discard path. Do not edit this file; the load path now handles it.

- [x] **Step 6: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass (355+ before these additions).

- [x] **Step 7: Commit**

```bash
git add models/forecaster.py tests/test_bias_fit_date_guard.py
git commit -m "fix: discard bias thresholds fitted before the date guard

bias_corrections.json is gitignored, so the rail-clamped 30d values could
not be corrected by a committed edit. A schema version discards
unversioned thresholds on load instead, which self-heals on every machine.
Matters because a corrupt classifier falls through to this path silently."
```

---

### Task 3: The weighted training class prior

Component 1 starts here. `_fit_direction_classifier` (`:3282`) trains with `objective="multiclass"` and no `class_weight` or `is_unbalance` (`:3308`), so the classifier inherits the up/down skew of its 1460-day training window and applies it unconditionally at serve time.

This task adds only the measurement. Task 4 decides whether the correction gets built.

The helper recomputes `_direction_classes` and `_direction_sample_weights` rather than reading them out of `_fit_direction_classifier`. That duplicates two cheap pure calls, and buys a pure function with no side effects — important because `_fit_direction_classifier` has a second caller in the CV loop (`:4038`) whose fold classifiers must not overwrite the production prior.

**Files:**
- Modify: `backend/models/forecaster.py` — new classmethod after `_direction_sample_weights` (`:3257`)
- Test: `backend/tests/test_direction_prior.py` (create)

**Interfaces:**
- Consumes: `ItemForecaster._direction_classes`, `ItemForecaster._direction_sample_weights`
- Produces: `ItemForecaster._direction_class_prior(returns, threshold, mover_weight) -> Dict[int, float]`, keyed `0=down, 1=flat, 2=up`, values summing to 1.0. Returns `{}` when the prior is not estimable.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_direction_prior.py`:

```python
"""The directional classifier's inherited up/down prior, and its correction.

The classifier trains multiclass with no class_weight, so whatever up/down
skew its 1460-day window carried is learned and then applied at serve time
regardless of current market state. That is the mechanism behind "predicts
down 57-87% of the time regardless of date" in
docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md.

Two asymmetries live in this classifier and only one is a bug: up-vs-down is
inherited and unwanted; mover-vs-flat is deliberate (DIRECTION_MOVER_WEIGHT_MAP
up-weights movers 3x). Every correction here is zero-sum over down/up and
leaves the flat mass alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT, ItemForecaster

THR = DIRECTION_FLAT_TOLERANCE_PCT  # 0.5
MOVER = 3.0


class TestDirectionClassPrior:
    def test_balanced_returns_give_a_balanced_prior(self):
        r = np.array([5.0, -5.0, 5.0, -5.0])
        prior = ItemForecaster._direction_class_prior(r, THR, MOVER)
        assert prior[0] == pytest.approx(0.5)
        assert prior[2] == pytest.approx(0.5)
        assert prior[1] == pytest.approx(0.0)

    def test_prior_is_weighted_by_mover_weight_not_raw_counts(self):
        # 1 up mover (w=3), 3 flat rows (w=1 each) -> total weight 6.
        r = np.array([5.0, 0.1, 0.1, 0.1])
        prior = ItemForecaster._direction_class_prior(r, THR, MOVER)
        assert prior[2] == pytest.approx(3.0 / 6.0)   # not 1/4
        assert prior[1] == pytest.approx(3.0 / 6.0)
        assert prior[0] == pytest.approx(0.0)

    def test_down_skew_is_reflected_in_the_prior(self):
        r = np.array([-5.0, -5.0, -5.0, 5.0])
        prior = ItemForecaster._direction_class_prior(r, THR, MOVER)
        assert prior[0] > prior[2]

    def test_prior_sums_to_one(self):
        rng = np.random.default_rng(0)
        r = rng.normal(-0.3, 4.0, size=5000)
        prior = ItemForecaster._direction_class_prior(r, THR, MOVER)
        assert sum(prior.values()) == pytest.approx(1.0)

    def test_non_finite_returns_are_dropped(self):
        r = np.array([5.0, -5.0, np.nan, np.inf, -np.inf])
        prior = ItemForecaster._direction_class_prior(r, THR, MOVER)
        assert sum(prior.values()) == pytest.approx(1.0)
        assert prior[0] == pytest.approx(0.5)

    def test_empty_input_is_not_estimable(self):
        assert ItemForecaster._direction_class_prior(np.array([]), THR, MOVER) == {}

    def test_all_non_finite_is_not_estimable(self):
        assert ItemForecaster._direction_class_prior(
            np.array([np.nan, np.nan]), THR, MOVER) == {}
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_direction_prior.py -v
```

Expected: FAIL — `AttributeError: ... has no attribute '_direction_class_prior'`.

- [x] **Step 3: Implement the helper**

Add after `_direction_sample_weights` (`:3257`) in `backend/models/forecaster.py`:

```python
    @classmethod
    def _direction_class_prior(cls, returns, threshold: float,
                               mover_weight: float) -> Dict[int, float]:
        """Weighted training class prior as {0: down, 1: flat, 2: up}.

        Weighted by _direction_sample_weights, because that is the
        distribution the classifier's multiclass objective actually sees —
        raw class counts would describe a model that was never trained.
        Returns {} when not estimable, which callers treat as "serve
        uncorrected".

        ``threshold`` must be a scalar. Production trains with
        sigma_train=None (:2902, :4042), so the band is the fixed scalar
        DIRECTION_FLAT_TOLERANCE_PCT; a per-row band would need the same
        rows dropped here as in the finite mask below.
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        if r.size == 0:
            return {}
        c = cls._direction_classes(r, float(threshold))
        w = cls._direction_sample_weights(r, float(threshold), mover_weight)
        total = float(w.sum())
        if total <= 0.0:
            return {}
        return {k: float(w[c == k].sum() / total) for k in (0, 1, 2)}
```

- [x] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_direction_prior.py -v
```

Expected: 7 passed.

- [x] **Step 5: Commit**

```bash
git add models/forecaster.py tests/test_direction_prior.py
git commit -m "feat: measure the directional classifier's weighted class prior

Pure helper, no call sites yet. Weighted by the mover sample weights
because that is the distribution the multiclass objective sees."
```

---

### Task 4: Step 0 diagnostic — GATE

**This task decides whether Tasks 5–7 happen.** The hypothesis is that the down-bias lives in the training class prior. It is falsifiable in one run, and the alternative is adding a branch to the serving path on an unverified assumption.

The diagnostic reads `models/saved_models/engineered_data.parquet` (the persisted engineered frame) and derives forward labels by reproducing `build_training_data`'s date-shifted merge (`:2159-2172`) — a date-based merge, never a row shift, because item series have gaps. This avoids a feature rebuild entirely; per `docs/research/accuracy-opportunities.md` a full frame build is 124.5s at 22 GB peak on 24 GB RAM.

**Files:**
- Create: `backend/scripts/diagnose_direction_prior.py`
- Uses: `ItemForecaster._direction_class_prior` (Task 3)

**Interfaces:**
- Consumes: `ItemForecaster._direction_class_prior`, `ItemForecaster.load_models`, `ItemForecaster.HORIZONS`, `ItemForecaster.DIRECTION_MOVER_WEIGHT_MAP`, `DIRECTION_FLAT_TOLERANCE_PCT`
- Produces: a printed table only. No files written, no DB writes.

- [x] **Step 1: Write the diagnostic**

Create `backend/scripts/diagnose_direction_prior.py`:

```python
#!/usr/bin/env python3
"""Step 0 gate for the directional prior correction.

Tests one falsifiable claim: that the classifier's unconditional "down" bias
is inherited from its training class prior.

  training prior up/down SKEWED   -> prior correction is the right fix
  training prior up/down BALANCED -> the bias is serve-time, not the prior.
                                     STOP. Do not build the correction.

Read-only. Derives forward labels from the persisted engineered frame by
reproducing build_training_data's date-shifted merge, so it needs no feature
rebuild.

Usage:
    python scripts/diagnose_direction_prior.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT, ItemForecaster

FRAME = Path(__file__).parent.parent / "models" / "saved_models" / "engineered_data.parquet"
# Skew large enough to be worth correcting. Below this the prior is not the
# explanation for a 57-87% down rate and the hypothesis is refuted.
SKEW_RATIO_THRESHOLD = 1.20


def forward_return(df, horizon):
    """target_return_{h}d, reproducing forecaster.py:2159-2172 exactly.

    Date-based merge, not a row shift: row-based shifts give wrong horizons
    wherever an item's series has gaps.
    """
    d = df[["item_id", "date", "price"]].copy()
    d["_dt"] = pd.to_datetime(d["date"])
    future = d[["item_id", "_dt", "price"]].copy()
    future.columns = ["item_id", "date", "target"]
    future["date"] = (future["date"] - pd.Timedelta(days=horizon)).dt.date
    merged = d.merge(future, on=["item_id", "date"], how="left")
    ret = (merged["target"] - merged["price"]) / merged["price"].replace(0, np.nan) * 100
    return ret.clip(-500.0, 500.0).to_numpy(dtype=float)


def main():
    if not FRAME.exists():
        print(f"FAIL: {FRAME} not found. Run a training pass first.")
        return 1

    df = pd.read_parquet(FRAME, columns=["item_id", "date", "price"])
    print(f"Engineered frame: {len(df):,} rows, "
          f"{df['item_id'].nunique():,} items, "
          f"{df['date'].min()} .. {df['date'].max()}\n")

    f = ItemForecaster(db_session=MagicMock())
    f.load_models()

    print(f"{'h':>3} {'pi_down':>8} {'pi_flat':>8} {'pi_up':>8} "
          f"{'down/up':>8} | {'served_down':>11} {'served_flat':>11} {'served_up':>10}")
    print("-" * 84)

    verdicts = {}
    for h in ItemForecaster.HORIZONS:
        y = forward_return(df, h)
        prior = f._direction_class_prior(
            y, DIRECTION_FLAT_TOLERANCE_PCT,
            ItemForecaster.DIRECTION_MOVER_WEIGHT_MAP.get(h, 3.0))
        if not prior:
            print(f"{h:>3} prior not estimable")
            continue

        ratio = prior[0] / prior[2] if prior[2] > 0 else float("inf")
        verdicts[h] = ratio

        served = "  (no classifier)"
        clf = f.direction_models.get(h)
        if clf is not None:
            # load_models restores feature_cols from meta.json (:4493). Require
            # ALL of them: predicting on a silently-truncated subset would
            # produce a served distribution that is not the served one.
            feats = list(f.feature_cols)
            full = pd.read_parquet(FRAME)
            missing = [c for c in feats if c not in full.columns]
            if not feats:
                served = "  (no feature_cols in meta.json)"
            elif missing:
                served = f"  (frame missing {len(missing)} feature cols)"
            else:
                cls = clf.predict(full[feats]).argmax(axis=1)
                n = len(cls)
                served = (f"{(cls == 0).mean() * 100:10.1f}% "
                          f"{(cls == 1).mean() * 100:10.1f}% "
                          f"{(cls == 2).mean() * 100:9.1f}%  (n={n:,})")

        print(f"{h:>3} {prior[0]:8.4f} {prior[1]:8.4f} {prior[2]:8.4f} "
              f"{ratio:8.3f} | {served}")

    print()
    skewed = {h: r for h, r in verdicts.items()
              if r > SKEW_RATIO_THRESHOLD or r < 1.0 / SKEW_RATIO_THRESHOLD}
    if skewed:
        print(f"VERDICT: PROCEED. Prior is up/down skewed beyond "
              f"{SKEW_RATIO_THRESHOLD:.2f}x at horizons "
              f"{sorted(skewed)} (down/up ratios "
              f"{ {h: round(r, 3) for h, r in sorted(skewed.items())} }).")
        print("         Build the correction (Tasks 5-7).")
    else:
        print(f"VERDICT: STOP. No horizon's prior is skewed beyond "
              f"{SKEW_RATIO_THRESHOLD:.2f}x, so the training prior does not "
              f"explain the observed down-bias.")
        print("         Do NOT build the correction. Ship Tasks 1-2, write the "
              "diagnostic up as a changelog entry, and stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 2: Run the diagnostic**

```bash
python scripts/diagnose_direction_prior.py
```

Expected: the table plus one `VERDICT:` line.

- [x] **Step 3: Act on the verdict — this is the gate**

- **`VERDICT: PROCEED`** → continue to Task 5.
- **`VERDICT: STOP`** → **do not implement Tasks 5–7.** Tasks 1–2 are already committed and stand on their own. Write `docs/changelog/2026-08-03-direction-prior-diagnostic.md` recording the measured priors, the served distributions, that the hypothesis was refuted, and that the down-bias is therefore serve-time in origin rather than a prior artifact. Report the numbers back and stop. A refutation here is a successful outcome of this task, not a failure.

- [x] **Step 4: Commit the diagnostic and its result**

```bash
git add scripts/diagnose_direction_prior.py
git commit -m "feat: add Step 0 diagnostic for the directional prior hypothesis

Read-only. Derives forward labels from the persisted engineered frame via
the same date-shifted merge build_training_data uses, so it needs no
feature rebuild. Prints PROCEED/STOP on the up/down skew ratio."
```

---

### Task 5: The offset and the correction — 🛑 **NEVER BUILT.** Task 4's gate returned STOP; this describes code that does not exist

Only proceed if Task 4 printed `VERDICT: PROCEED`.

Two pure helpers. The offset is **zero-sum** (`−b/2` to down, `+b/2` to up) rather than the textbook prior division (`log p_c − log π_c` for both). Prior division would boost up *and* down against flat, since both priors are below 1 — silently changing the flat share and undoing the deliberate mover weighting.

**Files:**
- Modify: `backend/models/forecaster.py` — `DIRECTION_PRIOR_TAU`; two statics after `_direction_class_prior`
- Test: `backend/tests/test_direction_prior.py` (append)

**Interfaces:**
- Consumes: `ItemForecaster._direction_class_prior` (Task 3)
- Produces:
  - `ItemForecaster.DIRECTION_PRIOR_TAU = 1.0`
  - `ItemForecaster._direction_prior_offset(prior, tau=None) -> float`
  - `ItemForecaster._apply_direction_prior(probs, offset) -> np.ndarray` of int class labels, shape `(n,)`

- [~] **Step 1: Write the failing test**

Append to `backend/tests/test_direction_prior.py`:

```python
class TestDirectionPriorOffset:
    def test_balanced_prior_gives_zero_offset(self):
        prior = {0: 0.3, 1: 0.4, 2: 0.3}
        assert ItemForecaster._direction_prior_offset(prior) == pytest.approx(0.0)

    def test_down_skewed_prior_gives_positive_offset(self):
        """Positive offset is what suppresses down and boosts up."""
        prior = {0: 0.5, 1: 0.2, 2: 0.3}
        assert ItemForecaster._direction_prior_offset(prior) > 0.0

    def test_up_skewed_prior_gives_negative_offset(self):
        prior = {0: 0.3, 1: 0.2, 2: 0.5}
        assert ItemForecaster._direction_prior_offset(prior) < 0.0

    def test_offset_is_log_of_the_down_up_ratio(self):
        prior = {0: 0.6, 1: 0.0, 2: 0.2}
        assert ItemForecaster._direction_prior_offset(prior) == pytest.approx(np.log(3.0))

    def test_tau_scales_the_offset(self):
        prior = {0: 0.6, 1: 0.0, 2: 0.2}
        full = ItemForecaster._direction_prior_offset(prior, tau=1.0)
        assert ItemForecaster._direction_prior_offset(prior, tau=0.5) == pytest.approx(full / 2)

    def test_production_tau_is_one(self):
        assert ItemForecaster.DIRECTION_PRIOR_TAU == 1.0

    def test_missing_prior_gives_zero_offset(self):
        assert ItemForecaster._direction_prior_offset({}) == 0.0
        assert ItemForecaster._direction_prior_offset(None) == 0.0

    def test_degenerate_prior_gives_zero_offset_not_infinity(self):
        assert ItemForecaster._direction_prior_offset({0: 0.5, 1: 0.5, 2: 0.0}) == 0.0
        assert ItemForecaster._direction_prior_offset({0: 0.0, 1: 0.5, 2: 0.5}) == 0.0

    def test_string_keys_from_json_round_trip_are_handled(self):
        prior = {"0": 0.6, "1": 0.0, "2": 0.2}
        assert ItemForecaster._direction_prior_offset(prior) == pytest.approx(np.log(3.0))


class TestApplyDirectionPrior:
    @staticmethod
    def _probs():
        # rows: clear down, marginal down, clear flat, marginal up, clear up
        return np.array([
            [0.80, 0.15, 0.05],
            [0.45, 0.20, 0.35],
            [0.10, 0.80, 0.10],
            [0.35, 0.20, 0.45],
            [0.05, 0.15, 0.80],
        ])

    def test_zero_offset_is_identical_to_plain_argmax(self):
        """The null-safety guarantee: no measured skew, no behaviour change."""
        p = self._probs()
        got = ItemForecaster._apply_direction_prior(p, 0.0)
        assert np.array_equal(got, p.argmax(axis=1))

    def test_positive_offset_moves_calls_from_down_to_up_not_the_reverse(self):
        """The sign test — the assertion most likely to catch an inversion."""
        p = self._probs()
        base = p.argmax(axis=1)
        got = ItemForecaster._apply_direction_prior(p, 1.0)
        assert (got == 0).sum() < (base == 0).sum()
        assert (got == 2).sum() > (base == 2).sum()
        # No row may move in the forbidden direction.
        assert not np.any((base == 2) & (got == 0))

    def test_negative_offset_moves_calls_from_up_to_down(self):
        p = self._probs()
        base = p.argmax(axis=1)
        got = ItemForecaster._apply_direction_prior(p, -1.0)
        assert (got == 2).sum() < (base == 2).sum()
        assert (got == 0).sum() > (base == 0).sum()

    def test_flat_share_is_preserved(self):
        """Zero-sum over down/up: the mover weighting must survive."""
        rng = np.random.default_rng(7)
        raw = rng.dirichlet([1.0, 1.0, 1.0], size=20000)
        base = raw.argmax(axis=1)
        got = ItemForecaster._apply_direction_prior(raw, 0.8)
        base_flat = (base == 1).mean()
        got_flat = (got == 1).mean()
        assert abs(got_flat - base_flat) < 0.02

    def test_zero_probabilities_do_not_produce_nan_or_inf(self):
        p = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        got = ItemForecaster._apply_direction_prior(p, 1.0)
        assert got.dtype.kind in "iu"
        assert set(np.unique(got)).issubset({0, 1, 2})

    def test_non_finite_offset_falls_back_to_plain_argmax(self):
        p = self._probs()
        for bad in (np.nan, np.inf, -np.inf):
            got = ItemForecaster._apply_direction_prior(p, bad)
            assert np.array_equal(got, p.argmax(axis=1))

    def test_is_deterministic(self):
        p = self._probs()
        a = ItemForecaster._apply_direction_prior(p, 0.6)
        b = ItemForecaster._apply_direction_prior(p, 0.6)
        assert np.array_equal(a, b)

    def test_does_not_mutate_the_input(self):
        p = self._probs()
        before = p.copy()
        ItemForecaster._apply_direction_prior(p, 1.0)
        assert np.array_equal(p, before)

    def test_rejects_a_non_three_class_matrix(self):
        with pytest.raises(ValueError):
            ItemForecaster._apply_direction_prior(np.array([[0.5, 0.5]]), 1.0)
```

- [~] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_direction_prior.py -v -k "Offset or Apply"
```

Expected: FAIL — `AttributeError: ... has no attribute 'DIRECTION_PRIOR_TAU'`.

- [~] **Step 3: Implement both helpers**

Add the constant next to `DIRECTION_CONFIDENCE_HIGH` (`:235`):

```python
    # Strength of the up/down prior correction. 1.0 fully neutralizes the
    # training prior's up/down skew. Fixed deliberately, not swept: a
    # per-horizon map against a 1.15pp harness noise floor is how noise gets
    # read as signal (docs/research/accuracy-opportunities.md).
    DIRECTION_PRIOR_TAU = 1.0
```

Add both statics after `_direction_class_prior` (Task 3):

```python
    @classmethod
    def _direction_prior_offset(cls, prior, tau: Optional[float] = None) -> float:
        """Zero-sum logit offset b = tau * log(pi_down / pi_up).

        Positive when down is over-represented in training, which is the
        case that needs suppressing. Returns 0.0 whenever the prior is
        missing or degenerate, so an unmeasurable prior serves uncorrected
        rather than raising or producing an infinite shift.
        """
        if not prior:
            return 0.0
        t = cls.DIRECTION_PRIOR_TAU if tau is None else float(tau)
        p = {int(k): float(v) for k, v in prior.items()}
        p_down, p_up = p.get(0, 0.0), p.get(2, 0.0)
        if p_down <= 0.0 or p_up <= 0.0:
            return 0.0
        return float(t * np.log(p_down / p_up))

    @staticmethod
    def _apply_direction_prior(probs, offset: float) -> np.ndarray:
        """Argmax over 3-class probabilities after a zero-sum down/up shift.

        s_down = log p_down - b/2
        s_flat = log p_flat            (anchor, unshifted)
        s_up   = log p_up   + b/2

        Zero-sum rather than dividing each class by its own prior: prior
        division lifts both down and up against flat (both priors are < 1),
        which would change the flat share and undo the deliberate mover
        weighting in DIRECTION_MOVER_WEIGHT_MAP. Only the up-vs-down
        asymmetry is the defect.
        """
        p = np.asarray(probs, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(
                f"expected an (n, 3) probability matrix, got {p.shape}")
        if not np.isfinite(offset) or offset == 0.0:
            return p.argmax(axis=1)
        s = np.log(np.clip(p, 1e-12, None))
        s[:, 0] -= offset / 2.0
        s[:, 2] += offset / 2.0
        return s.argmax(axis=1)
```

`np.log` returns a fresh array, so the in-place `s[:, 0] -=` cannot touch the caller's `probs`.

- [~] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_direction_prior.py -v
```

Expected: 25 passed.

- [~] **Step 5: Commit**

```bash
git add models/forecaster.py tests/test_direction_prior.py
git commit -m "feat: zero-sum up/down prior offset for the direction classifier

Pure helpers, not yet wired into predict(). Zero-sum over down/up so the
flat share -- and the deliberate mover weighting behind it -- survives."
```

---

### Task 6: Wire the correction into training, persistence, and serving — 🛑 **NEVER BUILT.** Task 4's gate returned STOP; this describes code that does not exist

**Files:**
- Modify: `backend/models/forecaster.py` — `self.direction_priors` init near `:289`; compute at `:2898`; save near `:4348`; load near `:4589`; apply at `:3710-3714`
- Modify: `.gitignore` — ignore `direction_priors.json`
- Test: `backend/tests/test_direction_prior.py` (append)

**Interfaces:**
- Consumes: `_direction_class_prior`, `_direction_prior_offset`, `_apply_direction_prior` (Tasks 3, 5)
- Produces: `self.direction_priors: Dict[int, Dict[int, float]]`; `models/saved_models/direction_priors.json`

- [~] **Step 1: Write the failing test**

Append to `backend/tests/test_direction_prior.py`:

```python
import json
from unittest.mock import MagicMock


class TestDirectionPriorPersistence:
    def test_priors_round_trip_through_json_with_int_keys(self, tmp_path):
        d = tmp_path / "saved_models"
        d.mkdir()
        f = ItemForecaster(db_session=MagicMock(), model_dir=str(d))
        f.direction_priors = {7: {0: 0.5, 1: 0.2, 2: 0.3}}
        f._save_direction_priors()

        g = ItemForecaster(db_session=MagicMock(), model_dir=str(d))
        g._load_direction_priors()
        # JSON stringifies keys; both levels must come back as ints.
        assert g.direction_priors[7][0] == pytest.approx(0.5)
        assert g._direction_prior_offset(g.direction_priors[7]) > 0.0

    def test_missing_file_leaves_priors_empty_not_raising(self, tmp_path):
        d = tmp_path / "saved_models"
        d.mkdir()
        f = ItemForecaster(db_session=MagicMock(), model_dir=str(d))
        f._load_direction_priors()
        assert f.direction_priors == {}

    def test_corrupt_file_leaves_priors_empty_not_raising(self, tmp_path):
        d = tmp_path / "saved_models"
        d.mkdir()
        (d / "direction_priors.json").write_text("{not json")
        f = ItemForecaster(db_session=MagicMock(), model_dir=str(d))
        f._load_direction_priors()
        assert f.direction_priors == {}

    def test_a_horizon_without_a_prior_serves_uncorrected(self, tmp_path):
        d = tmp_path / "saved_models"
        d.mkdir()
        f = ItemForecaster(db_session=MagicMock(), model_dir=str(d))
        f.direction_priors = {7: {0: 0.5, 1: 0.2, 2: 0.3}}
        assert f._direction_prior_offset(f.direction_priors.get(30, {})) == 0.0


class TestConfidenceIsUnaffected:
    """dir_conf_arr must read the UNADJUSTED probabilities.

    The correction changes which class is served; it must not change how
    confident we claim to be, because DIRECTION_CONFIDENCE_HIGH gates the
    served confidence string and the coverage machinery downstream of it.
    """

    def test_confidence_is_the_max_of_the_raw_probabilities(self):
        rng = np.random.default_rng(11)
        probs = rng.dirichlet([1.0, 1.0, 1.0], size=500)

        # What predict() computes for confidence, before and after the change.
        conf = probs.max(axis=1)

        # A large offset changes many served classes ...
        base_cls = ItemForecaster._apply_direction_prior(probs, 0.0)
        got_cls = ItemForecaster._apply_direction_prior(probs, 1.5)
        assert not np.array_equal(base_cls, got_cls)

        # ... while confidence, read off the raw matrix, is untouched.
        assert np.array_equal(probs.max(axis=1), conf)

    def test_the_correction_does_not_renormalize_the_input(self):
        """If it mutated probs, confidence would silently shift with it."""
        probs = np.array([[0.80, 0.15, 0.05], [0.35, 0.20, 0.45]])
        before = probs.max(axis=1).copy()
        ItemForecaster._apply_direction_prior(probs, 1.5)
        assert np.array_equal(probs.max(axis=1), before)
```

- [~] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_direction_prior.py -v -k Persistence
```

Expected: FAIL — `AttributeError: ... has no attribute '_save_direction_priors'`.

- [~] **Step 3: Add the attribute and the persistence pair**

In `__init__`, next to `self.direction_models` (`:289`):

```python
        # {horizon: {0: pi_down, 1: pi_flat, 2: pi_up}} — the weighted training
        # class prior, measured at fit time and divided out at serve time.
        self.direction_priors: Dict[int, Dict[int, float]] = {}
```

Add the pair next to the other persistence helpers (after `_save_bias_corrections`, `:410`):

```python
    def _save_direction_priors(self):
        path = os.path.join(self.model_dir, "direction_priors.json")
        payload = {str(h): {str(k): v for k, v in prior.items()}
                   for h, prior in self.direction_priors.items()}
        with open(path, "w") as f:
            json.dump({"priors": payload}, f, indent=2)
        logger.info(f"  Saved direction priors for {len(payload)} horizons")

    def _load_direction_priors(self):
        path = os.path.join(self.model_dir, "direction_priors.json")
        self.direction_priors = {}
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.direction_priors = {
                int(h): {int(k): float(v) for k, v in prior.items()}
                for h, prior in data.get("priors", {}).items()
            }
            logger.info(f"  Loaded direction priors for "
                        f"{len(self.direction_priors)} horizons")
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.warning(f"  Corrupt direction_priors.json ({e}), "
                           f"serving uncorrected")
            self.direction_priors = {}
```

- [~] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_direction_prior.py -v -k Persistence
```

Expected: 4 passed.

- [~] **Step 5: Compute the prior during training**

In the training path, immediately after the production classifier fit (`:2898-2904`):

```python
            self.direction_priors[horizon] = self._direction_class_prior(
                y_train, DIRECTION_FLAT_TOLERANCE_PCT,
                self.DIRECTION_MOVER_WEIGHT_MAP.get(horizon, 3.0))
            _b = self._direction_prior_offset(self.direction_priors[horizon])
            logger.info(f"  {horizon}d class prior: "
                        f"down={self.direction_priors[horizon].get(0, 0):.4f} "
                        f"flat={self.direction_priors[horizon].get(1, 0):.4f} "
                        f"up={self.direction_priors[horizon].get(2, 0):.4f} "
                        f"-> offset b={_b:+.4f}")
```

`DIRECTION_FLAT_TOLERANCE_PCT` matches the band the classifier trained with, because production passes `sigma_train=None` (`:2902`) and `_thr(None)` returns that scalar (`:3296-3298`).

**Do not** add this to the CV-loop fit at `:4038`. That fits per-fold classifiers for fold metrics; writing the prior there would overwrite the production value with a fold's.

- [~] **Step 6: Save and load alongside the classifiers**

In `save_models`, after the classifier loop (`:4348-4352`):

```python
        if self.direction_priors:
            self._save_direction_priors()
```

In `load_models`, after the classifier loop (`:4589-4598`):

```python
        self._load_direction_priors()
```

- [~] **Step 7: Ignore the new production artifact**

`models/saved_models/*.txt` and `bias_corrections.json` are already ignored (`.gitignore:68,70`) but a new `.json` is not, so it would otherwise be committed. Add next to line 70:

```
backend/models/saved_models/direction_priors.json
```

Verify:

```bash
git check-ignore -v models/saved_models/direction_priors.json
```

Expected: a match on the new line.

- [~] **Step 8: Apply the correction in `predict()`**

Replace the argmax block (`:3710-3714`):

```python
            clf = self.direction_models.get(horizon)
            if clf is not None:
                probs = clf.predict(X_horizon)
                # Divide out the training prior's up/down skew before the
                # argmax. The classifier trains multiclass with no
                # class_weight, so its inherited skew would otherwise be
                # applied unconditionally, irrespective of market state.
                _b = self._direction_prior_offset(
                    self.direction_priors.get(horizon, {}))
                dir_class_arr = self._apply_direction_prior(probs, _b)
                # Confidence stays on the UNADJUSTED vector, so the
                # correction cannot move DIRECTION_CONFIDENCE_HIGH
                # classifications or anything downstream of them.
                dir_conf_arr = probs.max(axis=1)
```

- [~] **Step 9: Verify end-to-end and check for regressions**

```bash
python -m pytest tests/test_direction_prior.py tests/test_forecaster.py -q
python -m pytest tests/ -q
```

Expected: all pass. With no `direction_priors.json` present, `_direction_prior_offset({})` is `0.0` and `_apply_direction_prior` returns plain `argmax` — so every existing prediction test must be unaffected. If any prediction test changes behaviour here, that is a bug in this task, not a stale test.

- [~] **Step 10: Commit**

```bash
git add models/forecaster.py tests/test_direction_prior.py ../.gitignore
git commit -m "feat: divide out the training class prior before the direction argmax

Served direction was an uncorrected argmax over a classifier trained
multiclass with no class_weight, so the training window's up/down skew was
applied unconditionally at serve time. Confidence still reads the
unadjusted probabilities. Absent a stored prior the offset is 0.0 and
serving is byte-identical."
```

---

### Task 7: Derive the MDE, then measure — 🛑 **NEVER BUILT.** Task 4's gate returned STOP; this describes code that does not exist

`backend/scripts/ab_test_direction_labels.py` is the right harness: walk-forward folds over the archive, so it does not inherit the two-date problem that makes the live series unreportable.

**The MDE comes first.** Per `docs/research/accuracy-opportunities.md`, six consecutive feature groups produced |effect| < 0.7pp against a floor of 1.15pp (3d) and 2.76–7.13pp (7d/14d/30d), and runs below the floor return coin-flip fold counts that get misread as "unproven" when they are unresolvable. A systematic prior bias is plausibly larger than a feature effect — but that gets derived, not assumed.

**Files:**
- Read: `backend/scripts/ab_test_direction_labels.py`
- Create: `docs/changelog/2026-08-03-direction-prior-correction.md`

- [~] **Step 1: Read the harness and find its arms switch**

```bash
sed -n '1,80p' scripts/ab_test_direction_labels.py
grep -n "argparse\|add_argument\|def main\|step=\|max_items" scripts/ab_test_direction_labels.py
```

Identify how to run a baseline arm (`DIRECTION_PRIOR_TAU = 0.0`) against a treatment arm (`= 1.0`), and note the fold `step` and item-count flags.

- [~] **Step 2: Run two arms on 7d only, and record per-fold deltas**

Switch arms with a one-line edit to the class constant — no new code, and it cannot be forgotten in a later run the way an env var can:

```bash
mkdir -p ../docs/research

# Baseline arm: correction off.
sed -i '' 's/    DIRECTION_PRIOR_TAU = 1.0/    DIRECTION_PRIOR_TAU = 0.0/' models/forecaster.py
grep -n "DIRECTION_PRIOR_TAU = " models/forecaster.py   # confirm it reads 0.0
python scripts/ab_test_direction_labels.py 2>&1 \
  | tee ../docs/research/2026-08-03-direction-prior-raw.txt

# Treatment arm: correction on. Restore the committed value.
sed -i '' 's/    DIRECTION_PRIOR_TAU = 0.0/    DIRECTION_PRIOR_TAU = 1.0/' models/forecaster.py
grep -n "DIRECTION_PRIOR_TAU = " models/forecaster.py   # confirm it reads 1.0
python scripts/ab_test_direction_labels.py 2>&1 \
  | tee -a ../docs/research/2026-08-03-direction-prior-raw.txt

git diff --stat models/forecaster.py   # MUST be empty: tau back at 1.0
```

Add the horizon and item-count flags found in Step 1 to restrict both runs to 7d and to keep the two arms identical in every respect but τ. The final `git diff --stat` is the guard against committing a `τ = 0.0` baseline by accident.

Save per-fold directional accuracy for both arms — the paired per-fold deltas are what Step 3 needs, not the pooled means.

- [~] **Step 3: Compute the MDE before reading the point estimate**

```bash
python -c "
import numpy as np
# Paste the paired per-fold deltas (treatment - baseline, in pp) from Step 2.
d = np.array([])
assert d.size, 'fill in the per-fold deltas first'
sd = d.std(ddof=1)
n = len(d)
mde = 2.8 * sd / np.sqrt(n)
print(f'n_folds={n}  mean={d.mean():+.3f}pp  paired sd={sd:.3f}pp  MDE(80%)={mde:.3f}pp')
print('DECIDABLE' if abs(d.mean()) >= mde else 'BELOW FLOOR - not resolvable by this design')
"
```

- [~] **Step 4: Decide, honestly**

- **|mean| ≥ MDE** → a resolvable result. Run the remaining horizons and record them.
- **|mean| < MDE** → **the run cannot produce a decision.** Say so plainly. Do not run more items to fix it: fold count is set by `step` and the archive date range, not by `--max-items` (26 folds at 40 items and 26 at 200). Do not reinterpret a below-floor delta as a positive result.

- [~] **Step 5: Write the changelog**

Create `docs/changelog/2026-08-03-direction-prior-correction.md` covering: the wiring gap (the bias corrector wrote `bias_thresholds`, which `predict()` reads only when no classifier is loaded, while all four `clf_*d.txt` exist); the Step 0 diagnostic result; the measured priors and offsets per horizon; the date guard and the schema-version discard; the MDE and whether the harness result was decidable; and what remains unmeasurable. State the MDE next to the point estimate so no future reader can quote one without the other.

Also state plainly that this does **not** make the live accuracy series reportable — that still needs 20 distinct forecast dates, which is calendar time.

- [~] **Step 6: Commit**

```bash
git add docs/changelog/2026-08-03-direction-prior-correction.md docs/research/2026-08-03-direction-prior-raw.txt
git commit -m "docs: record the direction prior correction and its measurement

Reports the MDE alongside the point estimate so the two cannot be quoted
apart."
```

---

## Notes for the implementer

**The absolute accuracy numbers from `walkforward_backtest.py` are not comparable to production.** Per `backend/AGENTS.md`, its `_load_all_prices` skips multi-source voting and collapses the archive's 1.37× duplicate item-days with a plain **mean**, where production serves an **outlier-voted median**. It also skips the `historical_fallback:` source filter, the dead-item filter, and `backfilled_only`. Only the A/B *delta* between two arms is meaningful.

**Do not quote any live accuracy figure.** Every stored cohort reports `date_coverage_sufficient = False`; the series spans two forecast dates that ran in opposite directions. See `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md`.

**Fitting the correction from realised outcomes is out of scope.** It needs the classifier margin per historical forecast, and `item_forecasts` stores only `direction` and a coarse `confidence` string (`database.py:143-158`). That is a later spec.
