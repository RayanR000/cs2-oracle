# Vol-scaled Direction Labels + Per-Horizon Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the directional classifier's fixed ±0.5% flat band with a per-item, per-horizon volatility-scaled band, make the flat multiplier and mover-weight per-horizon, and add a sweep script to tune them against a fixed accuracy yardstick.

**Architecture:** Add a trailing return-volatility column during feature engineering (used only for labeling, never as a model feature). A pure helper turns (σ, horizon, k) into a per-row threshold. `_direction_classes` / `_direction_sample_weights` accept a scalar-or-array threshold (scalar reproduces today's behavior exactly). `_fit_direction_classifier` gains a `horizon` param and optional `sigma_train`/`sigma_val` arrays; when supplied it labels with the vol-scaled band and uses the per-horizon mover-weight. Evaluation keeps calling with the scalar ±0.5% yardstick, so gated numbers stay comparable. A new `scripts/ab_test_direction_labels.py` sweeps `k` and `mover_weight` per horizon over the existing purge-gap CV.

**Tech Stack:** Python 3.13, LightGBM, pandas, numpy, pytest. All paths under `backend/`.

## Global Constraints

- Only the directional classifier changes. Quantile ensemble, features fed to the model, regime models, and serving/bias-threshold recalibration are untouched.
- Scalar-threshold code paths MUST remain byte-for-byte equivalent to today (fixed ±0.5%). Vol-scaling is opt-in via `sigma_*` arguments.
- The vol column is for labeling only — it must NOT enter `self.feature_cols` (add to the `exclude` set at `models/forecaster.py:2047`).
- No future information in the threshold: σ at row *t* uses only rows ≤ *t* (pandas `.rolling()` is trailing by default — do not center).
- Evaluation yardstick is fixed realized sign at ±0.5% (`DIRECTION_FLAT_TOLERANCE_PCT`), independent of the swept training threshold.
- Follow the existing `scripts/ab_test_*.py` pattern for the new sweep script.
- Run pytest from `backend/`: `cd backend && python -m pytest ...`.

---

### Task 1: Per-horizon constants + pure threshold helper

**Files:**
- Modify: `backend/models/forecaster.py` (constants near line 31 and class constants near line 172; add a static helper near `_direction_classes` at line 2781)
- Test: `backend/tests/test_direction_labels.py` (create)

**Interfaces:**
- Produces:
  - Module constants: `DIRECTION_VOL_WINDOW = 30`, `DIRECTION_THRESHOLD_FLOOR_PCT = 0.2`, `DIRECTION_THRESHOLD_CAP_PCT = 15.0`, `DIRECTION_LABEL_VOL_COL = "label_vol_30d"`.
  - Class dicts on `ItemForecaster`: `DIRECTION_MOVER_WEIGHT_MAP: Dict[int, float]` and `DIRECTION_VOL_MULTIPLIER_MAP: Dict[int, float]`.
  - `ItemForecaster._direction_threshold(sigma, horizon, k, floor, cap) -> np.ndarray`: given per-row daily-return volatility `sigma` (in percent), returns per-row flat-band threshold `clamp(k * sigma * sqrt(horizon), floor, cap)`, all in percent. Accepts scalar or array `sigma`; always returns a float ndarray.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_direction_labels.py
import numpy as np
from models.forecaster import ItemForecaster


def test_direction_threshold_scales_with_sqrt_horizon():
    # sigma = 1.0% daily, k = 1.0 -> threshold = sqrt(h)% (before clamp)
    sigma = np.array([1.0, 1.0])
    t3 = ItemForecaster._direction_threshold(sigma, horizon=3, k=1.0, floor=0.0, cap=100.0)
    np.testing.assert_allclose(t3, np.sqrt(3.0), rtol=1e-6)
    t30 = ItemForecaster._direction_threshold(sigma, horizon=30, k=1.0, floor=0.0, cap=100.0)
    np.testing.assert_allclose(t30, np.sqrt(30.0), rtol=1e-6)


def test_direction_threshold_applies_floor_and_cap():
    sigma = np.array([0.0, 1000.0])  # degenerate low and high vol
    t = ItemForecaster._direction_threshold(sigma, horizon=7, k=1.0, floor=0.2, cap=15.0)
    assert t[0] == 0.2   # floored
    assert t[1] == 15.0  # capped


def test_direction_threshold_accepts_scalar_sigma():
    t = ItemForecaster._direction_threshold(2.0, horizon=1, k=0.5, floor=0.0, cap=100.0)
    assert t.shape == (1,)
    np.testing.assert_allclose(t, 1.0, rtol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_direction_labels.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_direction_threshold'`

- [ ] **Step 3: Add constants and the helper**

Add module constants after line 31 (`DIRECTION_FLAT_TOLERANCE_PCT = 0.5`):

```python
# Vol-scaled directional labels (2026-07-27). The flat band is
# clamp(k_h * sigma_daily * sqrt(h), floor, cap) in percent, replacing the
# fixed ±DIRECTION_FLAT_TOLERANCE_PCT. sigma_daily is trailing std of
# log_return_1d over DIRECTION_VOL_WINDOW rows. Used for labeling only.
DIRECTION_VOL_WINDOW = 30
DIRECTION_THRESHOLD_FLOOR_PCT = 0.2
DIRECTION_THRESHOLD_CAP_PCT = 15.0
DIRECTION_LABEL_VOL_COL = "label_vol_30d"
```

Replace the class constant `DIRECTION_MOVER_WEIGHT = 3.0` at line 172 with per-horizon maps (defaults preserve today's mover-weight and give a neutral k that will be tuned by the sweep):

```python
    # Per-horizon directional-label knobs (2026-07-27). Defaults: mover-weight
    # keeps the prior global 3.0; vol multiplier k=1.0 is a starting point the
    # sweep (scripts/ab_test_direction_labels.py) tunes per horizon.
    DIRECTION_MOVER_WEIGHT_MAP = {3: 3.0, 7: 3.0, 14: 3.0, 30: 3.0}
    DIRECTION_VOL_MULTIPLIER_MAP = {3: 1.0, 7: 1.0, 14: 1.0, 30: 1.0}
```

Add the static helper next to `_direction_classes` (near line 2781):

```python
    @staticmethod
    def _direction_threshold(sigma, horizon: int, k: float,
                             floor: float, cap: float) -> np.ndarray:
        """Per-row flat-band threshold (percent) = clamp(k * sigma * sqrt(h),
        floor, cap). ``sigma`` is trailing daily-return std in percent; scalar
        or array. Always returns a float ndarray."""
        s = np.atleast_1d(np.asarray(sigma, dtype=float))
        raw = k * s * np.sqrt(float(horizon))
        return np.clip(raw, floor, cap)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_direction_labels.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_direction_labels.py
git commit -m "feat: per-horizon direction knobs + vol threshold helper"
```

---

### Task 2: Threshold-aware label & weight functions (backward compatible)

**Files:**
- Modify: `backend/models/forecaster.py` (`_direction_classes` line 2782, `_direction_sample_weights` line 2788)
- Test: `backend/tests/test_direction_labels.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `_direction_classes(returns, threshold=DIRECTION_FLAT_TOLERANCE_PCT) -> np.ndarray`: `threshold` is scalar or per-row array (percent). `up=2` if `return > +threshold`, `down=0` if `return < -threshold`, else `flat=1`. Called with a scalar it is identical to the old behavior.
  - `_direction_sample_weights(returns, threshold, mover_weight) -> np.ndarray`: rows with `|return| > threshold` get `mover_weight`, else `1.0`. `threshold` scalar or array.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_direction_labels.py
def test_direction_classes_scalar_matches_legacy():
    r = np.array([1.0, -1.0, 0.2, -0.2, 0.5, -0.5])
    got = ItemForecaster._direction_classes(r, 0.5)
    # >0.5 -> up(2), <-0.5 -> down(0), else flat(1); boundary 0.5 is flat
    assert got.tolist() == [2, 0, 1, 1, 1, 1]


def test_direction_classes_default_is_legacy_half_pct():
    r = np.array([0.6, -0.6, 0.0])
    assert ItemForecaster._direction_classes(r).tolist() == [2, 0, 1]


def test_direction_classes_per_row_threshold():
    r = np.array([1.0, 1.0])
    thr = np.array([0.5, 2.0])  # same return, different bands
    assert ItemForecaster._direction_classes(r, thr).tolist() == [2, 1]


def test_direction_sample_weights_per_row_threshold():
    r = np.array([1.0, 1.0])
    thr = np.array([0.5, 2.0])
    w = ItemForecaster._direction_sample_weights(r, thr, mover_weight=3.0)
    assert w.tolist() == [3.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_direction_labels.py -k "classes or sample_weights" -v`
Expected: FAIL — `test_direction_classes_per_row_threshold` / `_per_row_threshold` fail (current signature ignores a passed threshold) or TypeError.

- [ ] **Step 3: Generalize the two functions**

Replace `_direction_classes` (line 2782-2786):

```python
    @staticmethod
    def _direction_classes(returns, threshold=DIRECTION_FLAT_TOLERANCE_PCT) -> np.ndarray:
        """Bucket % returns into 0=down, 1=flat, 2=up using a flat band of
        ``threshold`` (scalar or per-row array, percent). Scalar reproduces the
        legacy fixed-±DIRECTION_FLAT_TOLERANCE_PCT behavior."""
        r = np.asarray(returns, dtype=float)
        thr = np.asarray(threshold, dtype=float)
        return np.where(r > thr, 2, np.where(r < -thr, 0, 1)).astype(int)
```

Replace `_direction_sample_weights` (line 2788-2795):

```python
    @staticmethod
    def _direction_sample_weights(returns, threshold, mover_weight: float) -> np.ndarray:
        """Up-weight clearly-moving rows (|return| > ``threshold``) by
        ``mover_weight``; flat rows keep weight 1.0. ``threshold`` scalar or
        per-row array (percent)."""
        r = np.asarray(returns, dtype=float)
        thr = np.asarray(threshold, dtype=float)
        w = np.ones(len(r))
        w[np.abs(r) > thr] = mover_weight
        return w
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_direction_labels.py -v`
Expected: PASS (all Task 1 + Task 2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_direction_labels.py
git commit -m "feat: threshold-aware direction classes and sample weights"
```

---

### Task 3: Compute trailing return-vol column (labeling only)

**Files:**
- Modify: `backend/models/forecaster.py` (`_compute_price_features`, add after `log_return_1d` at line 795; and the `exclude` set at line 2047)
- Test: `backend/tests/test_direction_labels.py` (append)

**Interfaces:**
- Produces: a `DIRECTION_LABEL_VOL_COL` (`"label_vol_30d"`) column on the feature frame = per-item trailing std of `log_return_1d` over `DIRECTION_VOL_WINDOW` rows, `min_periods=5`, in percent (log-returns here are already ×100? — verify below and match units). Excluded from `self.feature_cols`.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_direction_labels.py
import pandas as pd
from models.forecaster import DIRECTION_LABEL_VOL_COL


def test_label_vol_column_is_trailing_and_grouped():
    fc = ItemForecaster()
    # two items; constant-return item -> ~0 vol, noisy item -> >0 vol
    n = 60
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    calm = pd.DataFrame({"item_name": "calm", "date": dates,
                         "price": np.linspace(10.0, 12.0, n)})
    noisy_price = 10.0 * (1 + 0.1 * np.sin(np.arange(n)))
    noisy = pd.DataFrame({"item_name": "noisy", "date": dates, "price": noisy_price})
    df = pd.concat([calm, noisy], ignore_index=True)
    out = fc._compute_price_features(df)
    assert DIRECTION_LABEL_VOL_COL in out.columns
    calm_vol = out[out["item_name"] == "calm"][DIRECTION_LABEL_VOL_COL].iloc[-1]
    noisy_vol = out[out["item_name"] == "noisy"][DIRECTION_LABEL_VOL_COL].iloc[-1]
    assert noisy_vol > calm_vol
    # no leakage: first row per item has no trailing window -> NaN
    first_calm = out[out["item_name"] == "calm"][DIRECTION_LABEL_VOL_COL].iloc[0]
    assert np.isnan(first_calm)
```

> NOTE for implementer: before writing Step 3, read `_compute_price_features` around line 795 to confirm (a) the grouping variable name (`grouped = df.groupby("item_name")` or similar) and (b) whether `log_return_1d` is a fraction or already ×100. Match the column's units to the returns used for `target_return_{h}d` so the threshold (percent) and returns (percent) are comparable. Adjust the test's exact expectations only if units require it; the trailing/grouped/NaN-first assertions stand.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_direction_labels.py -k label_vol -v`
Expected: FAIL — column missing (`KeyError`/`assert ... in out.columns`).

- [ ] **Step 3: Add the column and exclude it from features**

In `_compute_price_features`, immediately after `log_return_1d` is created (line 795), using the same grouping object the function already uses (confirmed in Step 1 note):

```python
        # Trailing daily-return volatility for vol-scaled direction labels
        # (2026-07-27). Labeling only — excluded from model features.
        df[DIRECTION_LABEL_VOL_COL] = (
            df.groupby("item_name")["log_return_1d"]
              .transform(lambda s: s.rolling(DIRECTION_VOL_WINDOW, min_periods=5).std())
        )
```

At the `exclude` set (line 2047), add the label-vol column so it never becomes a feature. Find the set/list literal assigned to `exclude` and add `DIRECTION_LABEL_VOL_COL`:

```python
        exclude = {... existing entries ..., DIRECTION_LABEL_VOL_COL}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_direction_labels.py -k label_vol -v`
Expected: PASS

Then confirm it is not a model feature (guard against the allowlist being disabled later):

Run: `cd backend && python -c "from models.forecaster import ItemForecaster, DIRECTION_LABEL_VOL_COL; import inspect; src=inspect.getsource(ItemForecaster); assert 'exclude' in src; print('exclude wired:', DIRECTION_LABEL_VOL_COL)"`
Expected: prints `exclude wired: label_vol_30d`

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_direction_labels.py
git commit -m "feat: trailing return-vol column for direction labels"
```

---

### Task 4: Wire vol-scaled labels into the classifier fit

**Files:**
- Modify: `backend/models/forecaster.py` (`_fit_direction_classifier` line 2820; production call site line 2453; eval/CV call site line 3481)
- Test: `backend/tests/test_direction_labels.py` (append)

**Interfaces:**
- Consumes: `_direction_threshold`, `_direction_classes`, `_direction_sample_weights`, `DIRECTION_*_MAP`, clamp constants.
- Produces: new signature
  `_fit_direction_classifier(self, X_train, y_train_ret, X_val, y_val_ret, boosting_type, tree_params, horizon, sigma_train=None, sigma_val=None, num_boost_round=200)`.
  When `sigma_train` is provided, training labels/weights use the vol-scaled per-horizon band and the per-horizon mover-weight; when `None`, behavior is the legacy fixed-±0.5% band with `DIRECTION_MOVER_WEIGHT_MAP[horizon]`.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_direction_labels.py
def test_fit_classifier_signature_accepts_horizon_and_sigma():
    import inspect
    sig = inspect.signature(ItemForecaster._fit_direction_classifier)
    params = list(sig.parameters)
    assert "horizon" in params
    assert "sigma_train" in params and "sigma_val" in params


def test_fit_classifier_vol_scaling_changes_labels():
    # With a large per-row sigma, movers should collapse to flat, shrinking
    # the number of up/down training labels vs the fixed-0.5% baseline.
    fc = ItemForecaster()
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f0": rng.randn(400), "f1": rng.randn(400)})
    y = rng.randn(400) * 2.0  # returns in percent, spread around 0
    big_sigma = np.full(400, 50.0)  # huge vol -> band hits cap 15% -> most flat
    legacy = fc._direction_classes(y)  # fixed 0.5%
    thr = fc._direction_threshold(big_sigma, horizon=3,
                                  k=fc.DIRECTION_VOL_MULTIPLIER_MAP[3],
                                  floor=0.2, cap=15.0)
    scaled = fc._direction_classes(y, thr)
    assert (scaled == 1).sum() > (legacy == 1).sum()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_direction_labels.py -k "fit_classifier" -v`
Expected: FAIL — `horizon`/`sigma_*` not in signature.

- [ ] **Step 3: Update `_fit_direction_classifier` and both call sites**

Replace the body of `_fit_direction_classifier` (line 2820-2842) to thread horizon/sigma:

```python
    def _fit_direction_classifier(self, X_train, y_train_ret, X_val, y_val_ret,
                                   boosting_type: str, tree_params: dict,
                                   horizon: int, sigma_train=None, sigma_val=None,
                                   num_boost_round: int = 200):
        """Train a 3-class (down/flat/up) LightGBM classifier on returns,
        up-weighting movers. When ``sigma_train`` is given, the flat band is
        vol-scaled per row (k_h * sigma * sqrt(h), clamped); otherwise the
        legacy fixed ±DIRECTION_FLAT_TOLERANCE_PCT band is used. Early-stops on
        val multi-logloss for GBDT."""
        k = self.DIRECTION_VOL_MULTIPLIER_MAP.get(horizon, 1.0)
        mover_weight = self.DIRECTION_MOVER_WEIGHT_MAP.get(horizon, 3.0)
        floor, cap = DIRECTION_THRESHOLD_FLOOR_PCT, DIRECTION_THRESHOLD_CAP_PCT

        def _thr(sigma):
            if sigma is None:
                return DIRECTION_FLAT_TOLERANCE_PCT
            return self._direction_threshold(np.asarray(sigma, dtype=float),
                                              horizon, k, floor, cap)

        ds = {"max_bin": self.MAX_BIN, "feature_pre_filter": False}
        thr_train = _thr(sigma_train)
        c_train = self._direction_classes(y_train_ret, thr_train)
        w_train = self._direction_sample_weights(y_train_ret, thr_train, mover_weight)
        dtrain = lgb.Dataset(X_train, c_train, params=ds, weight=w_train)
        params = dict(tree_params)
        params.update(objective="multiclass", num_class=3, metric="multi_logloss",
                      boosting_type=boosting_type, verbosity=-1, n_jobs=-1,
                      random_state=42)
        callbacks = [lgb.log_evaluation(0)]
        valid_sets = None
        if X_val is not None and y_val_ret is not None and len(X_val):
            dval = lgb.Dataset(X_val, self._direction_classes(y_val_ret, _thr(sigma_val)),
                               reference=dtrain, params=ds)
            valid_sets = [dval]
            if boosting_type != "dart":
                callbacks.insert(0, lgb.early_stopping(20))
        return lgb.train(params, dtrain, num_boost_round=num_boost_round,
                         valid_sets=valid_sets, callbacks=callbacks)
```

Update the **production** call site (line 2453-2456). `train_set` / `val_set` are in scope (they build `X_train` at line 2263-2269); read the label-vol column from them:

```python
            self.direction_models[horizon] = self._fit_direction_classifier(
                X_train, y_train, X_val, y_val, boosting_type,
                self._direction_tree_params(per_quantile_params),
                horizon,
                sigma_train=train_set[DIRECTION_LABEL_VOL_COL].to_numpy(dtype=float),
                sigma_val=val_set[DIRECTION_LABEL_VOL_COL].to_numpy(dtype=float),
            )
```

> Implementer: confirm the loop-local names are `train_set` / `val_set` (line 2263-2269). If they differ, use the actual names.

Update the **eval/CV** call site (line 3481-3486): train vol-scaled, but keep the FIXED ±0.5% yardstick for `actual_cls` (it already passes no threshold → scalar default, so leave that line unchanged). `train_df` / `val_df` hold the rows here:

```python
            clf = self._fit_direction_classifier(
                X_train, y_train, X_val, y_val, boosting_type,
                self._direction_tree_params(per_quantile_params),
                horizon,
                sigma_train=train_df[DIRECTION_LABEL_VOL_COL].to_numpy(dtype=float),
                sigma_val=val_df[DIRECTION_LABEL_VOL_COL].to_numpy(dtype=float))
            pred_cls = clf.predict(X_val).argmax(axis=1)
            actual_cls = self._direction_classes(actual_returns)  # FIXED ±0.5% yardstick
            classifier_acc = round(float((pred_cls == actual_cls).mean()) * 100, 1)
```

> Implementer: confirm the row-frame names at line 3481 (`train_df`/`val_df` per line 3494-3495). Use the actual names.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_direction_labels.py -v`
Expected: PASS (all).

Import-sanity (no NameError from new symbols at call sites):
Run: `cd backend && python -c "import models.forecaster as m; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_direction_labels.py
git commit -m "feat: vol-scaled per-horizon labels in direction classifier fit"
```

---

### Task 5: Sweep script `ab_test_direction_labels.py`

**Files:**
- Create: `backend/scripts/ab_test_direction_labels.py`
- Read first: an existing `backend/scripts/ab_test_*.py` (e.g. `ab_test_hp_search.py`) to copy data-loading, CV-split, and CLI conventions.

**Interfaces:**
- Consumes: `ItemForecaster` (`build_training_frame`/feature pipeline, `_compute_cv_splits`, `_fit_direction_classifier`, `_direction_classes`, `_direction_threshold`, `DIRECTION_LABEL_VOL_COL`).
- Produces: a script that, per horizon, trains the classifier for each `(k, mover_weight)` combo over the purge-gap CV folds, evaluates every combo against the FIXED ±0.5% yardstick, and prints/writes a results table. No auto-adoption.

- [ ] **Step 1: Write a smoke test for the sweep entry point**

```python
# append to backend/tests/test_direction_labels.py
def test_sweep_grid_and_eval_are_importable():
    import importlib
    mod = importlib.import_module("scripts.ab_test_direction_labels")
    # grid constants exist and are non-empty
    assert len(mod.K_GRID) >= 2
    assert len(mod.MOVER_WEIGHT_GRID) >= 2
    # eval uses the fixed yardstick: a helper that scores preds vs fixed labels
    assert hasattr(mod, "score_fixed_yardstick")


def test_score_fixed_yardstick_ignores_training_threshold():
    from scripts.ab_test_direction_labels import score_fixed_yardstick
    import numpy as np
    actual_returns = np.array([1.0, -1.0, 0.1])  # up, down, flat @0.5%
    pred_cls = np.array([2, 0, 1])                # all correct vs fixed band
    acc, movers_acc = score_fixed_yardstick(pred_cls, actual_returns)
    assert acc == 1.0
    assert movers_acc == 1.0  # only the two movers, both correct
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_direction_labels.py -k "sweep or yardstick" -v`
Expected: FAIL — `ModuleNotFoundError: scripts.ab_test_direction_labels`.

- [ ] **Step 3: Write the script**

```python
# backend/scripts/ab_test_direction_labels.py
"""Sweep per-horizon vol multiplier k and mover-weight for the directional
classifier over the purge-gap CV, scored against a FIXED ±0.5% yardstick.

Prints a per-horizon table; adopt winners by editing
ItemForecaster.DIRECTION_VOL_MULTIPLIER_MAP / DIRECTION_MOVER_WEIGHT_MAP.
Nothing here writes model constants.
"""
import argparse
import itertools
import numpy as np

from models.forecaster import (
    ItemForecaster,
    DIRECTION_LABEL_VOL_COL,
    DIRECTION_FLAT_TOLERANCE_PCT,
    DIRECTION_THRESHOLD_FLOOR_PCT,
    DIRECTION_THRESHOLD_CAP_PCT,
)

K_GRID = [0.5, 1.0, 1.5, 2.0]
MOVER_WEIGHT_GRID = [1.5, 3.0, 5.0]
HORIZONS = [3, 7, 14, 30]


def score_fixed_yardstick(pred_cls, actual_returns):
    """Accuracy vs the fixed ±0.5% realized-sign labels, and accuracy over
    movers only (|actual| > 0.5%). Returns (overall_acc, movers_acc)."""
    actual = np.asarray(actual_returns, dtype=float)
    actual_cls = ItemForecaster._direction_classes(actual, DIRECTION_FLAT_TOLERANCE_PCT)
    pred = np.asarray(pred_cls, dtype=int)
    overall = float((pred == actual_cls).mean()) if len(pred) else float("nan")
    mover_mask = np.abs(actual) > DIRECTION_FLAT_TOLERANCE_PCT
    movers = (float((pred[mover_mask] == actual_cls[mover_mask]).mean())
              if mover_mask.any() else float("nan"))
    return overall, movers


def _run_horizon(fc, frame, horizon, purge_days, sample_frac):
    """Yield result dicts for every (k, mover_weight) combo at this horizon."""
    target_col = f"target_return_{horizon}d"
    feat = fc.feature_cols
    sorted_dates = np.array(sorted(frame["date"].unique()))
    splits = fc._compute_cv_splits(sorted_dates, purge_days=purge_days)
    boosting_type = fc.BOOSTING_TYPE_MAP.get(horizon, "gbdt")

    for k, mw in itertools.product(K_GRID, MOVER_WEIGHT_GRID):
        fc.DIRECTION_VOL_MULTIPLIER_MAP[horizon] = k
        fc.DIRECTION_MOVER_WEIGHT_MAP[horizon] = mw
        fold_overall, fold_movers = [], []
        for train_dates, val_dates in splits:
            tr = frame[frame["date"].isin(train_dates)].dropna(subset=[target_col])
            va = frame[frame["date"].isin(val_dates)].dropna(subset=[target_col])
            if len(tr) < 2000 or len(va) < 200:
                continue
            X_tr, X_va = tr[feat].fillna(0.0), va[feat].fillna(0.0)
            clf = fc._fit_direction_classifier(
                X_tr, tr[target_col].to_numpy(dtype=float),
                X_va, va[target_col].to_numpy(dtype=float),
                boosting_type,
                fc._direction_tree_params({}),
                horizon,
                sigma_train=tr[DIRECTION_LABEL_VOL_COL].to_numpy(dtype=float),
                sigma_val=va[DIRECTION_LABEL_VOL_COL].to_numpy(dtype=float))
            pred_cls = clf.predict(X_va).argmax(axis=1)
            ov, mv = score_fixed_yardstick(pred_cls, va[target_col].to_numpy(dtype=float))
            fold_overall.append(ov)
            fold_movers.append(mv)
        yield {
            "horizon": horizon, "k": k, "mover_weight": mw,
            "overall_acc": round(100 * np.nanmean(fold_overall), 2) if fold_overall else None,
            "movers_acc": round(100 * np.nanmean(fold_movers), 2) if fold_movers else None,
            "n_folds": len(fold_overall),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge-days", type=int, default=None,
                    help="CV purge/embargo gap; default = horizon per horizon")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    args = ap.parse_args()

    fc = ItemForecaster()
    frame = fc.build_training_frame()  # confirm exact builder name against ab_test_hp_search.py
    if args.sample_frac < 1.0:
        items = frame["item_name"].drop_duplicates().sample(frac=args.sample_frac, random_state=42)
        frame = frame[frame["item_name"].isin(items)]

    print(f"{'h':>3} {'k':>4} {'mw':>4} {'overall%':>9} {'movers%':>8} {'folds':>6}")
    results = []
    for horizon in HORIZONS:
        purge = args.purge_days if args.purge_days is not None else horizon
        best = None
        for row in _run_horizon(fc, frame, horizon, purge, args.sample_frac):
            results.append(row)
            print(f"{row['horizon']:>3} {row['k']:>4} {row['mover_weight']:>4} "
                  f"{str(row['overall_acc']):>9} {str(row['movers_acc']):>8} {row['n_folds']:>6}")
            if row["overall_acc"] is not None and (best is None or row["overall_acc"] > best["overall_acc"]):
                best = row
        if best:
            print(f"  -> best {horizon}d: k={best['k']} mover_weight={best['mover_weight']} "
                  f"overall={best['overall_acc']}% movers={best['movers_acc']}%")


if __name__ == "__main__":
    main()
```

> Implementer: `build_training_frame` and `feature_cols` population may require running the same setup `ab_test_hp_search.py` uses (it may call a `prepare`/`_engineer_features` step and set `fc.feature_cols`). Copy that setup exactly from the sibling script so `frame` has `feature_cols`, `target_return_{h}d`, `date`, `item_name`, and `DIRECTION_LABEL_VOL_COL`. Do not invent a new data path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_direction_labels.py -k "sweep or yardstick" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/ab_test_direction_labels.py backend/tests/test_direction_labels.py
git commit -m "feat: ab_test_direction_labels sweep with fixed-yardstick eval"
```

---

### Task 6: Run the sweep, record results, adopt winners

**Files:**
- Modify: `backend/models/forecaster.py` (`DIRECTION_VOL_MULTIPLIER_MAP`, `DIRECTION_MOVER_WEIGHT_MAP` — set winners)
- Create: `docs/research/2026-07-27-direction-label-sweep-results.md`

**Interfaces:** none (execution + config).

- [ ] **Step 1: Run the sweep**

Run: `cd backend && python -m scripts.ab_test_direction_labels 2>&1 | tee ../docs/research/2026-07-27-direction-label-sweep-raw.txt`
Expected: a per-horizon table plus a `-> best Hd:` line per horizon. If it errors on the data-loading setup, fix per the Task 5 implementer note (match the sibling script), not by changing the eval.

- [ ] **Step 2: Record results**

Create `docs/research/2026-07-27-direction-label-sweep-results.md` with: the per-horizon best `(k, mover_weight)`, its overall-vs-fixed-yardstick DA, movers-only DA, the prior baseline DA for that horizon (from the latest changelog: 3d 57.9 / 7d 57.1 / 14d 54.8 / 30d 54.9), and the delta. Note any horizon where no combo beats baseline (do NOT adopt those — keep the default there).

- [ ] **Step 3: Adopt winners**

Edit `DIRECTION_VOL_MULTIPLIER_MAP` and `DIRECTION_MOVER_WEIGHT_MAP` to the per-horizon winners. For any horizon with no improvement over baseline, leave `k=1.0` / `mover_weight=3.0` (or whichever default matched baseline) and say so in the results doc.

- [ ] **Step 4: Verify the model still trains and the label distribution is sane**

Run: `cd backend && python -m pytest tests/test_direction_labels.py tests/test_forecaster.py -v`
Expected: PASS.

Run a short training smoke (sample) to confirm no runtime error and non-degenerate labels — use the project's standard training entry point with a small sample flag (check `train_*.log` invocation / `scripts/forecast_prices.py`), and confirm logs show classifier training completing for all 4 horizons.

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py docs/research/2026-07-27-direction-label-sweep-results.md docs/research/2026-07-27-direction-label-sweep-raw.txt
git commit -m "chore: adopt per-horizon vol-scaled label winners from sweep"
```

---

## Self-Review

**Spec coverage:**
- Vol-scaled band `clamp(k·σ·sqrt(h), floor, cap)` → Task 1 (helper) + Task 3 (σ column) + Task 4 (wiring). ✔
- Per-horizon `k` and mover-weight dicts → Task 1 (dicts) + Task 6 (adopt). ✔
- Only classifier refits during sweep; quantile ensemble untouched → Task 5 trains only the classifier. ✔
- Fixed ±0.5% evaluation yardstick + movers-only secondary → Task 4 (eval site keeps scalar) + Task 5 (`score_fixed_yardstick`). ✔
- Serving unchanged → no serving files touched. ✔
- Tests: vol-band, clamp, penny degenerate, no-leakage, per-horizon fallback, label-distribution sanity → Tasks 1-3 tests + Task 6 Step 4. ✔
- 7-fold purge-gap CV gate → Task 5 uses `_compute_cv_splits(purge_days=horizon)`. ✔

**Placeholder scan:** No TBD/TODO. Two explicit implementer NOTES (units of `log_return_1d`; exact loop-local frame names and `build_training_frame` setup) are verification-of-existing-code steps, not deferred design — each says exactly what to confirm and how. Acceptable.

**Type consistency:** `_direction_threshold` returns ndarray everywhere; `_direction_classes(returns, threshold=scalar|array)` and `_direction_sample_weights(returns, threshold, mover_weight)` consistent across Tasks 2/4/5; `_fit_direction_classifier(..., horizon, sigma_train, sigma_val)` signature identical at both call sites and in the sweep; `DIRECTION_LABEL_VOL_COL` / clamp constants / `DIRECTION_*_MAP` names consistent throughout.
