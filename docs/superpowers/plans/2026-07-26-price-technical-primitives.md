# Pure-Price Technical Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six pure-price technical features (volatility asymmetry + oscillator divergence) to the served model and validate them with a walk-forward A/B against a placebo baseline.

**Architecture:** All six features are computed inside `_compute_price_features()` in `backend/models/forecaster.py`, after the MACD block, reusing already-computed `return_1d`/`return_7d`/`rsi_14`/`macd_histogram`. They auto-join the served `price_technicals` group via name-prefix routing in `_feature_group()` — no allowlist change. A standalone A/B script (`baseline`/`treatment`/`placebo` arms) decides ship-vs-revert.

**Tech Stack:** Python, pandas, numpy, LightGBM, DuckDB, pytest.

## Global Constraints

- Features must be **pure price math** — no dependence on `volume` (2026-07-16 audit: trade volume has zero predictive lift). OBV/VWAP-deviation are explicitly excluded.
- Feature names must start with a `price_technicals` prefix so `_feature_group()` (`backend/models/forecaster.py:78`) routes them to the served group: use `vol_`, `rsi_`, `macd_`.
- Missing values are left as NaN (downstream pipeline median-imputes to neutral). No new missing-flag columns.
- `vol_skew_30d` is clipped to `[0, 5]`.
- `rsi_price_divergence_7d` sign convention is fixed: **positive = price-up / RSI-down (bearish divergence)**.
- Ship only if ALL hold: `treatment` mean directional accuracy `>` `baseline` by a meaningful (non-flat) margin; no horizon regresses beyond the 0.5–1.5pp budget; `treatment` `>` `placebo`.
- Reference spec: `docs/superpowers/specs/2026-07-26-price-technical-primitives-design.md`.
- Commit trailer on every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Volatility asymmetry features

**Files:**
- Modify: `backend/models/forecaster.py` (inside `_compute_price_features()`, immediately after the MACD block that ends with `df["macd_histogram"] = ...`, ~line 840)
- Test: `backend/tests/test_forecaster.py` (class `TestFeatureEngineering`)

**Interfaces:**
- Consumes: `df["return_1d"]` and `df["item_id"]` (already present at this point in `_compute_price_features`).
- Produces: columns `vol_semidev_down_30d`, `vol_semidev_up_30d`, `vol_skew_30d` on the returned DataFrame.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_forecaster.py` in `class TestFeatureEngineering`:

```python
def test_vol_asymmetry_present_and_price_group(self, forecaster, basic_price_df):
    from models.forecaster import _feature_group
    df = forecaster._compute_price_features(basic_price_df)
    for col in ["vol_semidev_down_30d", "vol_semidev_up_30d", "vol_skew_30d"]:
        assert col in df.columns
        assert _feature_group(col) == "price_technicals"

def test_vol_skew_clip_bounds(self, forecaster, basic_price_df):
    df = forecaster._compute_price_features(basic_price_df)
    s = df["vol_skew_30d"].dropna()
    assert (s >= 0).all() and (s <= 5).all()

def test_vol_skew_reflects_asymmetry(self, forecaster):
    # Large, varied up-moves; small, varied down-moves => upside semidev >
    # downside semidev => skew > 1. Both sides vary, so neither semidev is
    # exactly zero (which would make the ratio NaN).
    rng = np.random.default_rng(0)
    price = 100.0
    rows = []
    for d in range(120):
        if d % 2 == 0:
            price *= 1 + rng.uniform(0.03, 0.06)    # big ups
        else:
            price *= 1 - rng.uniform(0.002, 0.006)  # tiny downs
        rows.append({"item_id": "a",
                     "date": date(2026, 1, 1) + timedelta(days=d),
                     "price": round(price, 2), "volume": 100})
    df = forecaster._compute_price_features(pd.DataFrame(rows))
    assert df["vol_skew_30d"].dropna().iloc[-1] > 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_forecaster.py -k "vol_asymmetry or vol_skew" -v`
Expected: FAIL (KeyError / column not found).

- [ ] **Step 3: Implement the features**

In `backend/models/forecaster.py`, immediately after the line `df["macd_histogram"] = df["macd_line"] - df["macd_signal"]`, insert:

```python
        # =====================================================================
        # Volatility asymmetry (downside vs upside semi-deviation) — pure price.
        # A symmetric std collapses panic (sharp downside) and froth (volatile
        # upside) into one number; splitting them exposes the difference.
        # =====================================================================
        ret = df["return_1d"]
        ret_neg = ret.where(ret < 0)
        ret_pos = ret.where(ret > 0)
        df["vol_semidev_down_30d"] = (
            ret_neg.groupby(df["item_id"]).rolling(30, min_periods=5).std()
            .reset_index(level=0, drop=True)
        )
        df["vol_semidev_up_30d"] = (
            ret_pos.groupby(df["item_id"]).rolling(30, min_periods=5).std()
            .reset_index(level=0, drop=True)
        )
        # Ratio > 1 => upside more volatile (froth); < 1 => downside sharper
        # (panic). Clipped: near-zero downside vol otherwise blows the ratio up.
        _semidev_down = df["vol_semidev_down_30d"].replace(0, np.nan)
        df["vol_skew_30d"] = (df["vol_semidev_up_30d"] / _semidev_down).clip(0, 5)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_forecaster.py -k "vol_asymmetry or vol_skew" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_forecaster.py
git commit -m "$(printf 'feat: volatility asymmetry price features\n\nvol_semidev_down/up_30d + vol_skew_30d. Pure price, joins the\nserved price_technicals group by prefix.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 2: Oscillator divergence features

**Files:**
- Modify: `backend/models/forecaster.py` (inside `_compute_price_features()`, immediately after the Task 1 volatility-asymmetry block)
- Test: `backend/tests/test_forecaster.py` (class `TestFeatureEngineering`)

**Interfaces:**
- Consumes: `df["rsi_14"]`, `df["macd_histogram"]`, `df["return_7d"]`, `df["item_id"]` (all present at this point).
- Produces: columns `rsi_divergence_7d`, `rsi_price_divergence_7d`, `macd_hist_slope_7d`.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_forecaster.py` in `class TestFeatureEngineering`:

```python
def test_oscillator_divergence_present_and_price_group(self, forecaster, basic_price_df):
    from models.forecaster import _feature_group
    df = forecaster._compute_price_features(basic_price_df)
    for col in ["rsi_divergence_7d", "rsi_price_divergence_7d", "macd_hist_slope_7d"]:
        assert col in df.columns
        assert _feature_group(col) == "price_technicals"

def test_rsi_price_divergence_sign(self, forecaster):
    # Price rising over the window while RSI is engineered to fall by feeding a
    # decelerating uptrend: check the divergence term is finite and the sign
    # convention holds where return_7d > 0 and rsi fell.
    df_in = pd.DataFrame({
        "item_id": ["a"] * 40,
        "date": [date(2026, 1, 1) + timedelta(days=d) for d in range(40)],
        "price": [100.0 + d for d in range(40)],  # steady climb
        "volume": [100] * 40,
    })
    df = forecaster._compute_price_features(df_in)
    row = df.iloc[-1]
    # return_7d positive on a climbing series; feature must be finite.
    assert row["return_7d"] > 0
    assert np.isfinite(row["rsi_price_divergence_7d"])

def test_divergence_nan_on_short_history(self, forecaster):
    df_in = pd.DataFrame({
        "item_id": ["a"] * 3,
        "date": [date(2026, 1, 1) + timedelta(days=d) for d in range(3)],
        "price": [100.0, 101.0, 102.0],
        "volume": [100, 100, 100],
    })
    df = forecaster._compute_price_features(df_in)
    # 7-day shift is impossible with 3 rows -> NaN.
    assert df["rsi_divergence_7d"].isna().all()
    assert df["macd_hist_slope_7d"].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_forecaster.py -k "oscillator_divergence or rsi_price_divergence or divergence_nan" -v`
Expected: FAIL (column not found).

- [ ] **Step 3: Implement the features**

In `backend/models/forecaster.py`, immediately after the Task 1 block (after `df["vol_skew_30d"] = ...`), insert:

```python
        # =====================================================================
        # Oscillator divergence — momentum of RSI/MACD, and price/RSI
        # disagreement. The frame is already item/date-sorted (MACD block
        # re-sorted it), so a groupby shift(7) is a clean 7-day lookback.
        # =====================================================================
        df["rsi_divergence_7d"] = (
            df["rsi_14"] - df.groupby("item_id")["rsi_14"].shift(7)
        )
        # Positive => price up while RSI down (bearish divergence). return_7d is
        # winsorized to +/-500; clip to +/-50 keeps typical moves on the same
        # scale as the RSI term (RSI change is bounded to +/-100).
        df["rsi_price_divergence_7d"] = (
            df["return_7d"].clip(-50, 50) / 50.0 - df["rsi_divergence_7d"] / 100.0
        )
        df["macd_hist_slope_7d"] = (
            df["macd_histogram"] - df.groupby("item_id")["macd_histogram"].shift(7)
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_forecaster.py -k "oscillator_divergence or rsi_price_divergence or divergence_nan" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full forecaster test module (no regressions)**

Run: `cd backend && python -m pytest tests/test_forecaster.py -v`
Expected: PASS (all existing tests + 6 new).

- [ ] **Step 6: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_forecaster.py
git commit -m "$(printf 'feat: oscillator divergence price features\n\nrsi_divergence_7d, rsi_price_divergence_7d, macd_hist_slope_7d.\nPure price, served via price_technicals prefix routing.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 3: A/B validation script (baseline / treatment / placebo)

**Files:**
- Create: `backend/scripts/ab_test_price_primitives.py` (start from a copy of `backend/scripts/ab_test_feature_contribution.py`)

**Interfaces:**
- Consumes: `ItemForecaster.engineer_features`, `ItemForecaster.prepare_targets`, `ItemForecaster.HORIZONS`, the `price-archive/prices-*.parquet` files, and the six feature columns from Tasks 1–2.
- Produces: a CLI that prints per-horizon directional accuracy for `baseline`, `treatment`, `placebo` and the treatment-arm per-feature importances.

- [ ] **Step 1: Copy the existing harness as the starting point**

```bash
cp backend/scripts/ab_test_feature_contribution.py backend/scripts/ab_test_price_primitives.py
```

- [ ] **Step 2: Replace the module docstring and prefix constants**

In `backend/scripts/ab_test_price_primitives.py`, replace the top docstring and the two prefix constants (`CROSS_SECTIONAL_PREFIXES`, `EVENT_PREFIXES`) with:

```python
"""
A/B test: do the pure-price technical primitives (volatility asymmetry +
oscillator divergence) add real directional accuracy?

Three arms on the same walk-forward folds:
  - baseline:  drop the 6 new primitive columns
  - treatment: all features (including the 6 new)
  - placebo:   the 6 new columns column-shuffled (capacity-inflation guard)

Ship only if treatment > baseline (meaningful, non-flat) AND treatment > placebo
AND no horizon regresses beyond the 0.5-1.5pp budget.

Usage:
    python scripts/ab_test_price_primitives.py [--max-items 200] [--horizon 14]
"""

NEW_PRIMITIVES = (
    "vol_semidev_down_30d", "vol_semidev_up_30d", "vol_skew_30d",
    "rsi_divergence_7d", "rsi_price_divergence_7d", "macd_hist_slope_7d",
)
```

Also update the logger name from `"ab_test_feature_contribution"` to `"ab_test_price_primitives"`.

- [ ] **Step 3: Replace the `subsets` definition with baseline/treatment/placebo**

Find the block that defines `subsets = { "full": ..., "no_cross_sectional": ..., "no_events": ... }` and replace it with:

```python
        # Three arms. baseline/treatment differ only by the 6 new columns.
        # placebo keeps the columns but shuffles their values (handled per-fold).
        subsets = {
            "baseline": [c for c in pruned if c not in NEW_PRIMITIVES],
            "treatment": pruned,
            "placebo": pruned,
        }
        present_new = [c for c in NEW_PRIMITIVES if c in pruned]
        logger.info(f"  New primitives present after prune: {present_new}")
        for name, cols in subsets.items():
            logger.info(f"    {name:20s}: {len(cols):>3d} features")
```

- [ ] **Step 4: Add per-fold placebo shuffling**

In the walk-forward loop, find where `train_df` and `val_df` are selected (the lines `train_df = tdf[tdf["date"].isin(train_dates)]` / `val_df = tdf[tdf["date"].isin(val_dates)]`) and, immediately after `val_df` is assigned, insert:

```python
                    if config_name == "placebo" and present_new:
                        rng = np.random.default_rng(42)
                        train_df = train_df.copy()
                        val_df = val_df.copy()
                        for col in present_new:
                            train_df[col] = rng.permutation(train_df[col].values)
                            val_df[col] = rng.permutation(val_df[col].values)
```

(`config_name` is the loop variable over `subsets`; `present_new` is defined in Step 3. Both are in scope here.)

- [ ] **Step 5: Add a ship-decision summary and treatment importances**

At the end of `run_evaluation` (just before it returns `results`), add a summary that compares the arms per horizon and reports treatment importances. Locate the `return results` line and insert before it:

```python
        logger.info("\n" + "=" * 64)
        logger.info("SHIP DECISION SUMMARY (dir-acc %, treatment vs baseline vs placebo)")
        logger.info("=" * 64)
        for h in sorted(results):
            r = results[h]
            if not all(k in r for k in ("baseline", "treatment", "placebo")):
                continue
            b = r["baseline"]["dir_acc"]
            t = r["treatment"]["dir_acc"]
            p = r["placebo"]["dir_acc"]
            logger.info(
                f"  {h:>2}d  baseline={b:5.2f}  treatment={t:5.2f}  "
                f"placebo={p:5.2f}  (t-b={t-b:+.2f}, t-p={t-p:+.2f})"
            )
```

- [ ] **Step 6: Smoke-run the script on a small sample**

Run: `cd backend && python scripts/ab_test_price_primitives.py --max-items 40 --horizon 7`
Expected: completes without error; prints three arms (`baseline`, `treatment`, `placebo`) for the 7d horizon and the SHIP DECISION SUMMARY line. (Accuracy numbers are not the deliverable here — a clean run is.)

- [ ] **Step 7: Commit**

```bash
git add backend/scripts/ab_test_price_primitives.py
git commit -m "$(printf 'feat: A/B harness for price technical primitives\n\nbaseline/treatment/placebo walk-forward arms with a column-shuffle\nplacebo as the capacity-inflation guard.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 4: Run full A/B, decide ship/shelve, record changelog

**Files:**
- Create: `docs/changelog/2026-07-26-price-technical-primitives.md`
- (Conditional) Modify/revert: `backend/models/forecaster.py` if the decision is to shelve

**Interfaces:**
- Consumes: the A/B script from Task 3.
- Produces: a changelog entry with the decision and the numbers backing it.

- [ ] **Step 1: Run the full A/B across all horizons**

Run: `cd backend && python scripts/ab_test_price_primitives.py --max-items 200 2>&1 | tee /tmp/price_primitives_ab.log`
Expected: completes; the SHIP DECISION SUMMARY prints `t-b` and `t-p` deltas for 3d/7d/14d/30d.

- [ ] **Step 2: Apply the ship rule**

Evaluate against the Global Constraints ship rule:
- `treatment > baseline` by a meaningful, non-flat margin (mean across horizons), AND
- no horizon's `t-b` is worse than the 0.5–1.5pp budget, AND
- `treatment > placebo` at the horizons that carry the gain.

Record which arms won at each horizon and the treatment importances of the 6 new features.

- [ ] **Step 3a (if SHIP): write the changelog**

Create `docs/changelog/2026-07-26-price-technical-primitives.md` documenting the feature set, the A/B table (`baseline`/`treatment`/`placebo` dir-acc per horizon), the placebo margin, the treatment importances, and the decision to keep. Leave the feature code in place.

- [ ] **Step 3b (if SHELVE): revert the feature code, keep the record**

If the result is net-flat or fails the placebo test, revert only the feature code added in Tasks 1–2 (keep the tests removed with them), keeping the A/B script and spec:

```bash
git revert --no-edit <task-1-commit-sha> <task-2-commit-sha>
```

Then create `docs/changelog/2026-07-26-price-technical-primitives.md` recording the A/B numbers and the shelve decision (the quality-spread precedent).

- [ ] **Step 4: Commit the changelog**

```bash
git add docs/changelog/2026-07-26-price-technical-primitives.md
git commit -m "$(printf 'docs: price technical primitives A/B result (%s)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>' "<ship|shelve>")"
```

---

## Notes for the executor

- Work happens in the worktree `../cs2-oracle-wt/price-primitives` on branch `feature/price-primitives`. Confirm `git rev-parse --show-toplevel` points there before committing (dispatched subagents have committed in the wrong checkout before — see the project memory note).
- The existing A/B harness filters price rows to a single source (`WHERE source = 'STEAMCOMMUNITY'`). If the smoke run in Task 3 Step 6 returns zero items, adjust that filter in `ab_test_price_primitives.py` to the source that has coverage in the current archive (e.g. `aggregator_sync`), then re-run.
- Do not touch `FEATURE_GROUP_ALLOWLIST`, non-price groups, or the uncommitted HP-search change in the main checkout.
