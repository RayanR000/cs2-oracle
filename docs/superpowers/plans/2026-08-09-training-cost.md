# Training Cost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the weekly retrain from 872s to ≈600s and re-tune hyperparameters against the metric the project actually uses, without changing which features reach a booster or what the model serves.

**Architecture:** Six independent changes to `backend/models/forecaster.py` and one to `.github/workflows/price-forecast.yml`. The largest is a row cap inside `_cv_evaluate_horizon` mirroring the one `_build_production_split` already applies; the second is swapping the Optuna objective from early-stopped pinball loss to within-date rank IC. Nothing touches labels, the item universe, or the served signal.

**Tech Stack:** Python 3.13 local / 3.11 CI, LightGBM, Optuna, pandas, pytest. Run everything from `backend/` through `venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-09-training-cost-design.md`

## Global Constraints

- Run tests as `venv/bin/python -m pytest tests/... -q` from `backend/`. **Never bare `pytest`** — `scripts/test_social_signal.py` imports `thefuzz`, which is not in `requirements.txt`, and collection aborts.
- **`backend/.env` points at production Supabase.** Every command in this plan is either a test or `--train-only`, neither of which writes forecasts, but do not add a step that calls `predict()` or writes to the DB.
- **Never re-add training parallelism.** Horizons, quantiles and ensemble members train sequentially; the `spawn` Pool and `ThreadPoolExecutor` were deleted 2026-07-21 after OpenMP deadlocks.
- **Never set `SKIP_CV=1`.** `q_hat` and the confidence thresholds are fitted on CV out-of-fold predictions, and Task 1 changes exactly those rows.
- **Do not change `max_bin`, `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `feature_pre_filter`, or add `force_row_wise`.** All measured dead or deliberate on 2026-08-09.
- Only `train_set` may ever be thinned. Thinning a validation set moves the evaluation cohort.
- Commit after every task.

---

### Task 1: Cap CV fold training rows

**Files:**
- Modify: `backend/models/forecaster.py` — add `CV_MAX_TRAIN_ROWS` constant beside `CV_STEP_DAYS` (`:501`); add `_cv_max_train_rows()` beside `_cv_step_days()` (`:511`); apply in `_cv_evaluate_horizon` (`:5693`)
- Modify: `backend/models/forecaster.py:5650` — `_cv_evaluate_horizon` signature gains `per_item_row_sampling: bool = False`
- Modify: `backend/models/forecaster.py:3838` — the `_cv_evaluate_horizon` call site forwards `per_item_row_sampling`
- Test: `backend/tests/test_cv_row_cap.py` (create)

**Interfaces:**
- Consumes: `_per_item_row_sample(train_set, max_rows, seed=42)` (`:3363`), `_cv_step_days()` (`:511`)
- Produces: `ItemForecaster.CV_MAX_TRAIN_ROWS: int = 300_000`; `ItemForecaster._cv_max_train_rows(self) -> int` reading env `CV_MAX_TRAIN_ROWS`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_cv_row_cap.py`:

```python
"""CV folds must respect a row cap the way the production split does.

`max_rows` was applied only in `_build_production_split`; `_cv_evaluate_horizon`
took the whole expanding window every fold, so nine folds per horizon summed to
4.2x the training frame and the conformal CV phase was 50.4% of an 872s retrain.

These tests pin the cap and, more importantly, pin what it must NOT touch: the
fold count, the validation rows, and the OOF record count are the sample size of
q_hat, mean_rank_ic and the PT statistic.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


def _forecaster(tmp_path, feature_cols):
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.QUANTILES = [0.5]
    f.feature_cols = list(feature_cols)
    f.CV_MIN_TRAIN_DAYS = 40
    f.CV_STEP_DAYS = 15
    f.VALIDATION_WINDOW_DAYS = 10
    return f


def _frame(n_items=40, n_dates=120, horizon=3, seed=7):
    """A tdf in the shape _cv_evaluate_horizon consumes, big enough that a
    small cap actually binds on the later expanding-window folds."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2025-01-01")
    for item in range(n_items):
        for d in range(n_dates):
            rows.append({
                "item_id": f"item-{item}",
                "date": start + pd.Timedelta(days=d),
                "f0": rng.normal(),
                "f1": rng.normal(),
                "price_tier": 2,
                f"target_return_{horizon}d": rng.normal(scale=0.05),
            })
    return pd.DataFrame(rows)


def test_cap_default_is_300k():
    assert ItemForecaster.CV_MAX_TRAIN_ROWS == 300_000


def test_env_overrides_the_cap(tmp_path):
    f = _forecaster(tmp_path, ["f0", "f1"])
    with patch.dict(os.environ, {"CV_MAX_TRAIN_ROWS": "1234"}):
        assert f._cv_max_train_rows() == 1234


def test_fold_train_rows_are_capped(tmp_path):
    """Every fold's training frame is at or under the cap."""
    f = _forecaster(tmp_path, ["f0", "f1"])
    seen = []
    real_train = f._fit_cv_fold_model if hasattr(f, "_fit_cv_fold_model") else None
    with patch.dict(os.environ, {"CV_MAX_TRAIN_ROWS": "500"}):
        with patch.object(ItemForecaster, "_record_cv_fold_train_rows",
                          side_effect=seen.append, create=True):
            f._cv_evaluate_horizon(_frame(), 3, {0.5: {"objective": "quantile"}})
    assert seen, "no folds ran"
    assert max(seen) <= 500


def test_cap_does_not_change_fold_count_or_oof_rows(tmp_path):
    """The cap thins training rows only. Fold count and OOF record count are the
    sample size of q_hat, rank IC and PT — if either moves, the cap hit the
    wrong axis."""
    frame = _frame()
    params = {0.5: {"objective": "quantile", "num_leaves": 7, "verbosity": -1}}

    f_uncapped = _forecaster(tmp_path / "a", ["f0", "f1"])
    with patch.dict(os.environ, {"CV_MAX_TRAIN_ROWS": "100000000"}):
        oof_a, folds_a = f_uncapped._cv_evaluate_horizon(frame, 3, params)[:2]

    f_capped = _forecaster(tmp_path / "b", ["f0", "f1"])
    with patch.dict(os.environ, {"CV_MAX_TRAIN_ROWS": "500"}):
        oof_b, folds_b = f_capped._cv_evaluate_horizon(frame, 3, params)[:2]

    assert len(folds_a) == len(folds_b)
    assert len(oof_a) == len(oof_b)


def test_only_train_is_thinned_never_val(tmp_path):
    """Thinning val would move the evaluation cohort, which is the artifact
    pairing exists to remove."""
    frame = _frame()
    params = {0.5: {"objective": "quantile", "num_leaves": 7, "verbosity": -1}}
    f = _forecaster(tmp_path, ["f0", "f1"])
    with patch.dict(os.environ, {"CV_MAX_TRAIN_ROWS": "500"}):
        _, fold_metrics = f._cv_evaluate_horizon(frame, 3, params)[:2]
    # Every fold's val_size is the full 10-day window x 40 items.
    for m in fold_metrics:
        assert m["val_size"] == 400
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_cv_row_cap.py -q`
Expected: FAIL — `AttributeError: type object 'ItemForecaster' has no attribute 'CV_MAX_TRAIN_ROWS'`

- [ ] **Step 3: Add the constant and the reader**

Beside `CV_STEP_DAYS` at `backend/models/forecaster.py:501`:

```python
    # Cap on each CV fold's TRAINING rows. `max_rows` was applied only in
    # _build_production_split, so _cv_evaluate_horizon took the whole expanding
    # window every fold: nine folds per horizon summed to 4.2x the frame and the
    # conformal CV phase was 50.4% of an 872s retrain (measured 2026-08-09).
    #
    # Only train is thinned. Fold count, val rows and the OOF record count are
    # the sample size of q_hat, mean_rank_ic and the PT statistic, and none of
    # them moves. The fold model then fits on less data than the served one, so
    # q_hat comes out LARGER and the band wider -- over-coverage, which is the
    # safe direction. Verify against NOMINAL_COVERAGE before lowering it.
    CV_MAX_TRAIN_ROWS = 300_000
```

Beside `_cv_step_days` at `:511`, matching its instance-method rationale:

```python
    def _cv_max_train_rows(self) -> int:
        return int(os.environ.get("CV_MAX_TRAIN_ROWS", self.CV_MAX_TRAIN_ROWS))
```

- [ ] **Step 4: Apply the cap in the fold loop**

In `_cv_evaluate_horizon`, replace the fold-frame construction at `:5693-5694`:

```python
        for fold_id, (train_dates, val_dates) in enumerate(splits):
            train_df = tdf[tdf["date"].isin(train_dates)]
            val_df = tdf[tdf["date"].isin(val_dates)]
```

with:

```python
        cv_max_rows = self._cv_max_train_rows()
        for fold_id, (train_dates, val_dates) in enumerate(splits):
            train_df = tdf[tdf["date"].isin(train_dates)]
            val_df = tdf[tdf["date"].isin(val_dates)]

            # Same cap, same draw, same reasoning as _build_production_split:
            # sample randomly (never tail()) so the calendar window survives and
            # expanding-window CV is not silently disabled. val_df is never
            # thinned -- that would move the evaluation cohort.
            if len(train_df) > cv_max_rows:
                if per_item_row_sampling:
                    train_df = self._per_item_row_sample(train_df, cv_max_rows)
                else:
                    train_df = train_df.sample(
                        n=cv_max_rows, random_state=42).sort_values("date")
            self._record_cv_fold_train_rows(len(train_df))
```

Add the hook the test patches, near `_cv_max_train_rows`:

```python
    @staticmethod
    def _record_cv_fold_train_rows(n: int) -> None:
        """Seam for tests to observe the post-cap fold size. No-op in prod."""
```

- [ ] **Step 5: Thread `per_item_row_sampling` through**

Change the signature at `:5650`:

```python
    def _cv_evaluate_horizon(self, tdf, horizon, per_quantile_params,
                             per_item_row_sampling: bool = False):
```

and the call site at `:3838` so CV and the production split cannot diverge on which draw they use:

```python
                cv_out = self._cv_evaluate_horizon(
                    tdf, horizon, per_quantile_params,
                    per_item_row_sampling=per_item_row_sampling)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_cv_row_cap.py -q`
Expected: PASS, 5 tests

- [ ] **Step 7: Run the existing CV tests for regressions**

Run: `venv/bin/python -m pytest tests/test_cv_cohort_parity.py tests/test_forecaster.py tests/test_fixed_boost_rounds.py -q`
Expected: PASS. The synthetic frames in these files are far under 300,000 rows, so the cap must not bind and no expectation should move.

- [ ] **Step 8: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_cv_row_cap.py
git commit -m "perf: cap CV fold training rows, as the production split already does"
```

---

### Task 2: Score Optuna trials on within-date rank IC

**Files:**
- Modify: `backend/models/forecaster.py:2776-2853` — `_optuna_search_params` objective
- Modify: `backend/models/forecaster.py` — `MODEL_ARTIFACT_VERSION` 5 → 6
- Test: `backend/tests/test_optuna_objective.py` (create)

**Interfaces:**
- Consumes: `_within_date_rank_ic(pred, actual, dates, mask=None, min_rows=20)` (`:5958`), `_boost_rounds(horizon, cv=True)` (`:591`)
- Produces: `_optuna_search_params` gains a required `val_dates` parameter

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_optuna_objective.py`:

```python
"""Optuna must select on the metric the project uses, not the one it discarded.

FIXED_BOOST_ROUNDS replaced early stopping in training and CV on 2026-08-08
because the trailing validation window carries ~a dozen effective observations.
The Optuna objective was not touched: it still scored
`model.best_score["valid_0"]["quantile"]` under `lgb.early_stopping(20)`. The
FIXED_BOOST_ROUNDS comment records the conflict directly -- at 14d and 30d the
val-loss optimum is 25 rounds while rank IC peaks at 500-750.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


def test_optuna_does_not_early_stop():
    src = inspect.getsource(ItemForecaster._optuna_search_params)
    assert "early_stopping" not in src, (
        "Optuna still early-stops on the window it scores; that is the "
        "criterion FIXED_BOOST_ROUNDS replaced.")


def test_optuna_does_not_prune_on_the_old_metric():
    src = inspect.getsource(ItemForecaster._optuna_search_params)
    assert "LightGBMPruningCallback" not in src, (
        "the pruner prunes on LightGBM's reported `quantile` metric, not on "
        "the returned objective, so it re-introduces the replaced criterion.")


def test_optuna_scores_within_date_rank_ic():
    src = inspect.getsource(ItemForecaster._optuna_search_params)
    assert "_within_date_rank_ic" in src
    assert "best_score" not in src


def test_optuna_takes_val_dates():
    sig = inspect.signature(ItemForecaster._optuna_search_params)
    assert "val_dates" in sig.parameters, (
        "rank IC is within-date; a pooled Spearman would re-introduce the "
        "market factor the PT test exists to reject.")


def test_optuna_tunes_at_cv_rounds_not_production_rounds():
    src = inspect.getsource(ItemForecaster._optuna_search_params)
    assert "cv=True" in src, (
        "tuning at production rounds and evaluating at CV rounds selects "
        "params that only pay off at a depth CV never reaches.")


def test_artifact_version_bumped():
    assert ItemForecaster.MODEL_ARTIFACT_VERSION >= 6, (
        "a cached meta.json would otherwise supply params chosen under the "
        "old criterion.")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_optuna_objective.py -q`
Expected: FAIL — 6 failures; `early_stopping` present, `_within_date_rank_ic` absent, `val_dates` missing, version is 5

- [ ] **Step 3: Rewrite the objective**

In `_optuna_search_params` (`:2776`), add `val_dates` to the signature and replace the fit block. The parameter search space, the warm-start `enqueue_trial`, the sampler and the Dataset construction all stay exactly as they are.

```python
            # Fixed rounds, no early stopping, no pruner: the trailing val
            # window carries ~a dozen effective observations, so early stopping
            # trips on noise (FIXED_BOOST_ROUNDS, :551-560) and
            # LightGBMPruningCallback prunes on LightGBM's reported `quantile`
            # metric rather than on what this objective returns.
            model = lgb.train(
                params, dtrain,
                num_boost_round=self._boost_rounds(horizon, cv=True),
                callbacks=[lgb.log_evaluation(0)],
            )
            # Within-date, never pooled. A pooled Spearman re-introduces the
            # market factor and would select for the base-rate tracking the
            # Pesaran-Timmermann test exists to reject.
            ic = self._within_date_rank_ic(
                model.predict(X_val), y_val, val_dates)
            return -(ic if ic is not None else 0.0)
```

Remove the now-unused `LightGBMPruningCallback` import and the `_num_rounds` line. Change the study creation to drop the pruner:

```python
        study = optuna.create_study(direction="minimize", sampler=sampler)
```

- [ ] **Step 4: Pass `val_dates` at the call site**

Find the `_optuna_search_params(` call inside `_train_horizon_inline` and pass the validation split's date column:

```python
                    best_params = self._optuna_search_params(
                        X_train, y_train, X_val, y_val,
                        val_dates=val_set["date"],
                        quantile=q, horizon=horizon,
                        boosting_type=self.BOOSTING_TYPE, n_trials=n_trials)
```

- [ ] **Step 5: Bump the artifact version**

```python
    MODEL_ARTIFACT_VERSION = 6   # v6: Optuna selects on within-date rank IC
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_optuna_objective.py -q`
Expected: PASS, 6 tests

- [ ] **Step 7: Run the artifact-version and shape tests**

Run: `venv/bin/python -m pytest tests/test_minimal_model_shape.py tests/test_forecaster.py -q`
Expected: PASS. If a test asserts `MODEL_ARTIFACT_VERSION == 5`, update it to 6 — that assertion exists to force a retrain, which is the intent here.

- [ ] **Step 8: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_optuna_objective.py
git commit -m "fix: select hyperparameters on within-date rank IC, not early-stopped pinball loss"
```

---

### Task 3: Run the allowlist before the correlation prune

**Files:**
- Modify: `backend/models/forecaster.py:3475-3489` — reorder in `build_training_data`
- Test: `backend/tests/test_allowlist_before_prune.py` (create)

**Interfaces:**
- Consumes: `_prune_features(df) -> List[str]` (`:2392`), `_apply_feature_allowlist(feature_cols, allowlist)` (`:4471`)
- Produces: `ItemForecaster.ALLOWLIST_BEFORE_PRUNE: bool = True`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_allowlist_before_prune.py`:

```python
"""Prune 33 columns, not 123.

`df[self.feature_cols].corr()` is O(rows x p^2) single-threaded pandas: 25.2s
over the 123 selected candidates, 1.65s over the 33 the allowlist keeps
(measured 2026-08-09 on the production frame). On that frame the prune drops
zero price_technicals features, so the reorder is output-identical -- but
_prune_features keeps the LOWER-INDEXED member of each >0.95 pair and index
order does not follow group, so that is a property of the frame, not a theorem.
Hence the flag and this test.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from models.forecaster import ItemForecaster


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def test_flag_defaults_on():
    assert ItemForecaster.ALLOWLIST_BEFORE_PRUNE is True


def test_prune_sees_only_allowlisted_columns(tmp_path):
    """The expensive call must run on the post-allowlist set."""
    f = _f(tmp_path)
    f.feature_cols = ["return_1d", "rsi_14", "day_of_week", "rarity_ordinal"]
    seen = {}
    original = ItemForecaster._prune_features

    def spy(self, df):
        seen["n"] = len(self.feature_cols)
        return original(self, df)

    ItemForecaster._prune_features = spy
    try:
        rng = np.random.default_rng(3)
        df = pd.DataFrame({c: rng.normal(size=200) for c in f.feature_cols})
        f._reduce_feature_cols(df)
    finally:
        ItemForecaster._prune_features = original
    assert seen["n"] == 2, "prune ran on the pre-allowlist column set"


def test_reorder_is_output_identical_on_uncorrelated_features(tmp_path):
    """Both orderings agree when no >0.95 pair crosses the allowlist boundary."""
    rng = np.random.default_rng(11)
    cols = ["return_1d", "rsi_14", "day_of_week", "rarity_ordinal"]
    df = pd.DataFrame({c: rng.normal(size=500) for c in cols})

    a = _f(tmp_path / "a")
    a.feature_cols = list(cols)
    a.ALLOWLIST_BEFORE_PRUNE = True
    a._reduce_feature_cols(df)

    b = _f(tmp_path / "b")
    b.feature_cols = list(cols)
    b.ALLOWLIST_BEFORE_PRUNE = False
    b._reduce_feature_cols(df)

    assert a.feature_cols == b.feature_cols
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_allowlist_before_prune.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'ALLOWLIST_BEFORE_PRUNE'` and `_reduce_feature_cols` undefined

- [ ] **Step 3: Extract the reduction into one method**

Add the flag beside `FEATURE_GROUP_ALLOWLIST` (`:350`):

```python
    # Run the allowlist BEFORE _prune_features. The correlation matrix is
    # O(rows x p^2) single-threaded pandas: 25.2s over 123 candidate columns,
    # 1.65s over the 33 the allowlist keeps (measured 2026-08-09). Output was
    # identical on the production frame -- but _prune_features keeps the
    # lower-indexed member of a >0.95 pair and index order does not follow
    # group, so set this False to restore the old order if a feature count moves.
    ALLOWLIST_BEFORE_PRUNE = True
```

Add the method, lifting the existing body from `build_training_data:3475-3489`:

```python
    def _reduce_feature_cols(self, df: pd.DataFrame) -> None:
        """Apply the allowlist and the correlation prune, in the cheaper order."""
        allowlist = list(self.FEATURE_GROUP_ALLOWLIST or [])
        if allowlist and self.bymykel_metadata_enabled():
            allowlist.append(self.BYMYKEL_META_GROUP)

        def _allow():
            if not allowlist:
                return
            pre = len(self.feature_cols)
            self.feature_cols = self._apply_feature_allowlist(
                self.feature_cols, allowlist)
            logger.info(
                f"Feature allowlist {allowlist}: "
                f"{pre} -> {len(self.feature_cols)} features")

        if self.ALLOWLIST_BEFORE_PRUNE:
            _allow()
            self.feature_cols = self._prune_features(df)
        else:
            self.feature_cols = self._prune_features(df)
            _allow()
```

- [ ] **Step 4: Call it from `build_training_data`**

Replace `:3475-3489` with:

```python
        self._reduce_feature_cols(df)
        self._base_feature_cols = list(self.feature_cols)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_allowlist_before_prune.py -q`
Expected: PASS, 3 tests

- [ ] **Step 6: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_allowlist_before_prune.py
git commit -m "perf: apply the feature allowlist before the correlation prune"
```

---

### Task 4: Skip the feature blocks the allowlist discards

**Files:**
- Modify: `backend/models/forecaster.py:2891` — `engineer_features` gains `skip_unused_groups: bool = False`
- Modify: `backend/models/forecaster.py` — `build_training_data` passes it
- Test: `backend/tests/test_skip_unused_feature_groups.py` (create)

**Interfaces:**
- Consumes: `FEATURE_GROUP_ALLOWLIST`, `bymykel_metadata_enabled()`
- Produces: `ItemForecaster._skipped_feature_groups() -> set[str]`; `engineer_features(..., skip_unused_groups: bool = False)`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_skip_unused_feature_groups.py`:

```python
"""Don't engineer 8 blocks the allowlist throws away -- but keep the full path.

Measured 2026-08-09: engineer_features is 17.5s and 8.5s of that (48%) is
blocks whose columns FEATURE_GROUP_ALLOWLIST then drops.

The two constraints this file exists to pin:
1. Seven ab_test_* harnesses build their own frame and call
   _apply_feature_allowlist on it, so the full 123-column path must survive.
2. The skip set must be DERIVED from the allowlist. If it is hard-coded and
   `cross_sectional` is later re-admitted (Track C4), the frame would carry the
   group in the allowlist and its columns absent -- median-filled to zero,
   undetectably.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from models.forecaster import ItemForecaster


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def test_default_is_off_so_harnesses_keep_the_full_frame(tmp_path):
    sig = inspect.signature(ItemForecaster.engineer_features)
    assert sig.parameters["skip_unused_groups"].default is False


def test_skip_set_excludes_the_allowlisted_group(tmp_path):
    f = _f(tmp_path)
    assert "price_technicals" not in f._skipped_feature_groups()


def test_skip_set_covers_the_discarded_groups(tmp_path):
    f = _f(tmp_path)
    skipped = f._skipped_feature_groups()
    for group in ("temporal", "events", "cross_sectional", "social",
                  "item_identity", "item_metadata", "supply_depth"):
        assert group in skipped


def test_skip_set_follows_the_allowlist_not_a_literal(tmp_path):
    """Track C4 re-admits cross_sectional. The skip set must follow."""
    f = _f(tmp_path)
    f.FEATURE_GROUP_ALLOWLIST = ["price_technicals", "cross_sectional"]
    assert "cross_sectional" not in f._skipped_feature_groups()
    assert "temporal" in f._skipped_feature_groups()


def test_empty_allowlist_skips_nothing(tmp_path):
    """An empty allowlist means every group is kept, so nothing may be skipped."""
    f = _f(tmp_path)
    f.FEATURE_GROUP_ALLOWLIST = []
    assert f._skipped_feature_groups() == set()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_skip_unused_feature_groups.py -q`
Expected: FAIL — `skip_unused_groups` and `_skipped_feature_groups` do not exist

- [ ] **Step 3: Add the derived skip set**

```python
    ALL_FEATURE_GROUPS = frozenset({
        "price_technicals", "supply_depth", "item_identity", "item_metadata",
        "temporal", "events", "cross_sectional", "social", "other",
    })

    def _skipped_feature_groups(self) -> set:
        """Groups engineer_features may skip: everything the allowlist drops.

        Derived, never a literal. Track C4 proposes re-admitting
        `cross_sectional`; a hard-coded list would then produce a frame with the
        group allowlisted and its columns absent, median-filled to zero.

        `other` is never skipped: distance_to_support / distance_to_resistance /
        high_low_range_30d group as `other` because _feature_group matches the
        prefix `support_` while the columns are named `distance_to_*`, and they
        are computed inside _compute_price_features regardless.
        """
        allowlist = set(self.FEATURE_GROUP_ALLOWLIST or [])
        if not allowlist:
            return set()
        if self.bymykel_metadata_enabled():
            allowlist.add(self.BYMYKEL_META_GROUP)
        return set(self.ALL_FEATURE_GROUPS) - allowlist - {"other"}
```

- [ ] **Step 4: Gate the blocks in `engineer_features`**

Add the parameter at `:2891` and guard each of the eight calls. `_compute_price_features` is never guarded.

```python
    def engineer_features(self, price_df: pd.DataFrame, ...,
                          skip_unused_groups: bool = False):
        ...
        skip = self._skipped_feature_groups() if skip_unused_groups else set()
        ...
        if "temporal" not in skip:
            df = self._add_temporal_features(df)
        if "item_identity" not in skip:
            df = self._add_item_identity_features(df)
        if "events" not in skip:
            df = self._add_event_features(df)
        if "item_metadata" not in skip:
            df = self._add_item_metadata_features(df)
        if "item_identity" not in skip:
            df = self._add_supply_side_features(df)
        if "social" not in skip:
            df = self._add_social_features(df)
        if "cross_sectional" not in skip:
            df = self._add_cross_sectional_features(df)
        if "supply_depth" not in skip:
            df = self._add_supply_depth_features(df)
```

- [ ] **Step 5: Turn it on in `build_training_data` only**

At the `engineer_features` call inside `build_training_data`, pass `skip_unused_groups=True`. Leave every other call site — including all seven harnesses — untouched.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_skip_unused_feature_groups.py -q`
Expected: PASS, 5 tests

- [ ] **Step 7: Verify the harnesses still get the full frame**

Run: `venv/bin/python -m pytest tests/ -q -k "feature or allowlist or harness"`
Expected: PASS. `_add_supply_side_features` is guarded on `item_identity` because `_feature_group` assigns it there — confirm no test asserts a `supply_side` group name.

- [ ] **Step 8: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_skip_unused_feature_groups.py
git commit -m "perf: skip the feature blocks the allowlist discards, behind a flag"
```

---

### Task 5: Default the CV diagnostic classifier off

**Files:**
- Modify: `backend/models/forecaster.py:5927-5934` — `_cv_diagnostic_classifier_enabled`
- Modify: `docs/architecture/model-optimization.md` — the local-retrain figure
- Test: `backend/tests/test_fixed_boost_rounds.py` (extend)

**Interfaces:**
- Consumes: env `CV_DIAGNOSTIC_CLASSIFIER`
- Produces: unchanged signature, inverted default

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_fixed_boost_rounds.py`:

```python
def test_cv_diagnostic_classifier_defaults_off(monkeypatch):
    """932s / 52% of a classifier-on retrain, to populate a meta.json field no
    served artifact reads. Measured 2026-08-09: 872s off vs 1804s on. On by
    default put a local retrain at 30.1 min, over the project's own 30-minute
    run cap."""
    monkeypatch.delenv("CV_DIAGNOSTIC_CLASSIFIER", raising=False)
    assert ItemForecaster._cv_diagnostic_classifier_enabled() is False


def test_cv_diagnostic_classifier_can_be_re_enabled(monkeypatch):
    monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", "1")
    assert ItemForecaster._cv_diagnostic_classifier_enabled() is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_fixed_boost_rounds.py -q -k diagnostic`
Expected: FAIL — returns True with the variable unset

- [ ] **Step 3: Invert the default**

```python
    @staticmethod
    def _cv_diagnostic_classifier_enabled() -> bool:
        """Whether CV fits the per-fold directional classifier.

        Default OFF (2026-08-09). It feeds no served artifact -- only fold_p50
        reaches oof_records -- and costs 932s, 52% of a classifier-on retrain
        (872s off vs 1804s on), which put a local retrain over the project's own
        30-minute run cap. The replacement diagnostics are mean_rank_ic and the
        PT verdict, both computed from fold_p50 and unaffected.

        Set CV_DIAGNOSTIC_CLASSIFIER=1 to restore mean_classifier_acc_ge1.
        """
        return os.environ.get("CV_DIAGNOSTIC_CLASSIFIER", "0") != "0"
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python -m pytest tests/test_fixed_boost_rounds.py -q`
Expected: PASS. If a test asserts `mean_classifier_acc_ge1` is present in `cv_results` by default, set `CV_DIAGNOSTIC_CLASSIFIER=1` in that test rather than reverting the default.

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_fixed_boost_rounds.py
git commit -m "perf: default the CV diagnostic classifier off"
```

---

### Task 6: Make the voted cache reachable in CI

**Files:**
- Modify: `backend/models/forecaster.py:3510-3521` — `_archive_fingerprint`
- Modify: `.github/workflows/price-forecast.yml` — cache `backend/data`
- Test: `backend/tests/test_voted_cache_key.py` (create)

**Interfaces:**
- Consumes: `self.archive_dir`
- Produces: `_archive_fingerprint()` returns a content-derived string

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_voted_cache_key.py`:

```python
"""The voted cache key must not depend on mtime.

CI checks the archive out fresh every run, so st_mtime_ns is new on every run
and the key changed unconditionally -- the cache could never hit there. Cost of
the miss is ~48s per run (21.1s DuckDB read + 27.2s voting), paid daily by the
predict path too, not just by the Monday retrain.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

from models.forecaster import ItemForecaster


@pytest.fixture
def archive(tmp_path):
    d = tmp_path / "price-archive"
    d.mkdir()
    pd.DataFrame({
        "item_slug": ["a", "b"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "mean_price": [1.0, 2.0],
    }).to_parquet(d / "prices-2026-01.parquet")
    return d


def _f(tmp_path, archive):
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.archive_dir = archive
    return f


def test_fingerprint_is_stable_across_a_touch(tmp_path, archive):
    f = _f(tmp_path, archive)
    before = f._archive_fingerprint()
    path = archive / "prices-2026-01.parquet"
    os.utime(path, (1_600_000_000, 1_600_000_000))
    assert f._archive_fingerprint() == before


def test_fingerprint_changes_when_content_changes(tmp_path, archive):
    f = _f(tmp_path, archive)
    before = f._archive_fingerprint()
    pd.DataFrame({
        "item_slug": ["a", "b", "c"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "mean_price": [1.0, 2.0, 3.0],
    }).to_parquet(archive / "prices-2026-01.parquet")
    assert f._archive_fingerprint() != before


def test_fingerprint_does_not_read_mtime(tmp_path, archive):
    import inspect
    src = inspect.getsource(ItemForecaster._archive_fingerprint)
    assert "st_mtime" not in src
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_voted_cache_key.py -q`
Expected: FAIL — the touch changes the fingerprint, and `st_mtime` is in the source

- [ ] **Step 3: Replace mtime with content**

```python
    def _archive_fingerprint(self) -> str:
        """Identify the archive by each file's row count and max day.

        Was `name:st_size:st_mtime_ns`, which made the cache structurally
        CI-hostile: CI checks the archive out fresh every run, so every mtime
        was new and the key changed unconditionally. Row count and max day are
        content-derived and survive a checkout.

        Only prices-*.parquet feeds the voted frame -- ops/ artifacts are
        rewritten by the pipeline on every run and must not invalidate it.
        """
        import duckdb
        parts = []
        con = duckdb.connect()
        try:
            for path in sorted(self.archive_dir.glob("prices-*.parquet")):
                n, max_day = con.execute(
                    "SELECT COUNT(*), MAX(date) FROM read_parquet(?)",
                    [str(path)]).fetchone()
                parts.append(f"{path.name}:{n}:{max_day}")
        finally:
            con.close()
        return "|".join(parts)
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_voted_cache_key.py -q`
Expected: PASS, 3 tests

- [ ] **Step 5: Cache `backend/data` in the workflow**

In `.github/workflows/price-forecast.yml`, beside the existing `saved_models` cache step:

```yaml
      - name: Cache voted price frame
        uses: actions/cache@v4
        with:
          path: backend/data
          key: voted-v4-${{ hashFiles('backend/models/forecaster.py') }}
          restore-keys: voted-v4-
```

The `v4` matches `VOTED_CACHE_VERSION`. **Bump both together.** Note Task order: if G3 (6c) lands and bumps `VOTED_CACHE_VERSION` to 5, this key must move to `voted-v5-` in the same commit.

- [ ] **Step 6: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_voted_cache_key.py .github/workflows/price-forecast.yml
git commit -m "perf: content-hash the archive fingerprint so the voted cache can hit in CI"
```

---

### Task 7: Measure, and correct the docs

**Files:**
- Modify: `docs/architecture/model-optimization.md` — the lever table, the phase table, the row-budget section
- Modify: `docs/architecture/model.md:210,533` — the retired +3.50pp citation
- Create: `docs/changelog/2026-08-09-training-cost-levers.md`

- [ ] **Step 1: Run a full cold retrain and capture per-phase timing**

```bash
cd backend
FORCE_HP_SEARCH=1 venv/bin/python scripts/forecast_prices.py --train-only 2>&1 | tee /tmp/retrain.log
```

Expected: total ≈600s, from 872s.

- [ ] **Step 2: Verify nothing structural moved**

Read the fresh `backend/models/saved_models/meta.json` and check each row of the table in the spec's Verification section. Specifically confirm:
- `n_folds` per horizon is unchanged (9, 8 at 30d)
- `horizon_feature_cols` is an **identical set** to the previous artifact
- empirical band coverage is ≥ 0.80

**If coverage came in under 0.80, stop and raise `CV_MAX_TRAIN_ROWS`.** The band getting wider is expected; narrower means the cap hit the wrong axis.

- [ ] **Step 3: Strike the refuted lever table**

In `docs/architecture/model-optimization.md`, replace levers 3/4/5/6 with a single row recording that `max_bin` 63→31, `num_leaves` 47→31, `min_data_in_leaf` 15→100 and `feature_fraction` 0.7→0.4 were measured on the production frame on 2026-08-09 at 26.3 / 26.2 / 27.1 / 28.9 ms per round against a 25.5 baseline, i.e. dead, and must not be re-proposed. Re-size lever 2 from "~1% of the retrain" to ≈32s, and note that most of it is `_prune_features` (25.2s), which the doc did not mention.

- [ ] **Step 4: Correct the stale +3.50pp**

In `docs/architecture/model.md:210` and `:533`, replace the +3.50pp [+1.56, +5.98] @30d claim with the re-derivation: **+1.642pp [−0.809, +4.505], null**, citing `docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`, and note the ±3–4pp item-draw noise floor the `prod_pool_b` placebo measured.

- [ ] **Step 5: Write the changelog entry**

Create `docs/changelog/2026-08-09-training-cost-levers.md` recording: the measured before/after total, the per-phase table, the fold-geometry finding (4.2× the frame), the refuted lever table, the Optuna criterion mismatch and what changed, and the coverage check result. Follow the house convention — what was measured, what the measurement changed, and what is explicitly not established.

- [ ] **Step 6: Commit**

```bash
git add docs/
git commit -m "docs: record the training-cost levers and retire the refuted micro-lever table"
```

---

## Self-review notes

- **Task 1 and Task 2 both touch `_train_horizon_inline`'s call sites.** Land them in order; Task 2's diff assumes Task 1's `per_item_row_sampling` parameter already exists on `_cv_evaluate_horizon`.
- **Task 6 and G3 (6c) both concern `VOTED_CACHE_VERSION`.** G3 bumps it to 5. Whichever lands second must update the workflow cache key in the same commit.
- **Task 4 and Track C4 interact.** The skip set is derived from the allowlist specifically so re-admitting `cross_sectional` cannot silently produce a frame with the group allowlisted and its columns absent. `test_skip_set_follows_the_allowlist_not_a_literal` is the guard.
- **No task changes which features reach a booster.** Task 3 and Task 4 are pure reordering and short-circuiting; `horizon_feature_cols` must be set-identical before and after, and Step 2 of Task 7 checks it.
