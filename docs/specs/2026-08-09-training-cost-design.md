# Training cost: the uncapped CV folds, and the selector that optimises a discarded criterion

**Track A** of `docs/research/2026-08-09-next-steps.md`. Source measurements:
`docs/research/2026-08-09-model-and-data-research.md` §2.

Six changes, none of which touches the served signal or the label. Five are cost; one (A2)
changes which hyperparameters are chosen and is the highest accuracy-per-second item in the
system. Together they take the weekly retrain from **872s to ≈600s** and remove the two reasons a
local retrain is unreproducible against a CI one.

**No retrain is required to land them.** A1, A3 and A4 change what a retrain does; A2 additionally
requires `FORCE_HP_SEARCH=1` and a `MODEL_ARTIFACT_VERSION` bump so the stale params cannot be
reused.

---

## What the measurement changed before anything was built

Three claims in `docs/architecture/model-optimization.md` do not survive re-measurement on the
production frame (918 items / 986,065 rows, the same frame the 872s run used, read from the
on-disk voted cache).

**1. The micro-lever table is dead.** Measured ms per boosting round, 3 reps each, against a 25.5
baseline:

| Lever | Doc claim | Measured |
|---|---|---:|
| `max_bin` 63 → 31 | "~10–15%" (`:163`) | **26.3** |
| `num_leaves` 47 → 31 | "~15–20%" (`:164`) | **26.2** |
| `min_data_in_leaf` 15 → 100 | "~5%" (`:166`) | **27.1** |
| `feature_fraction` 0.7 → 0.4 | — | **28.9** |

Every one is inside noise or slower. LightGBM is memory-bandwidth bound at this frame shape — 1
thread against all measured **1.5×**, matching the 1.55× already recorded at `:84`. `n_jobs = -1`
is already set at all four fit sites and float32 downcast is already done at `:3493-3495`. There
is nothing left in this family, and the table should be struck rather than re-costed.

**2. Feature engineering is 17.5s, not 7s** — and the waste is in the prune, not the engineering.
`_prune_features` (`:2392`) computes `df[self.feature_cols].corr()` over 123 columns in
single-threaded pandas for **25.2s**; the same call over the 33 allowlisted columns costs
**1.65s**. So lever 2 is worth ≈32s, not the ~1% the doc states — but most of that is the prune,
which the doc does not mention at all.

**3. The 84% conformal-CV share is the 100K config and is dead.** At the shipped config it is
**50.4%**, and the mechanism turns out to be arithmetic rather than anything intrinsic to
calibration. See below.

---

## A1. CV folds are uncapped

`max_rows` is applied only on the production split (`_build_production_split:2756-2761`).
`_cv_evaluate_horizon:5693` does:

```python
train_df = tdf[tdf["date"].isin(train_dates)]
```

— the entire expanding window, every fold, with no cap of any kind. Measured fold geometry on the
production frame:

| h | embargo | folds | Σ fold-train rows | ×frame | CV rounds | prod rounds |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 16 | 9 | 4,213,562 | 4.27× | 200 | 300 |
| 7 | 20 | 9 | 4,179,057 | 4.24× | 500 | 750 |
| 14 | 27 | 9 | 4,135,665 | 4.19× | 100 | 150 |
| 30 | 43 | 9 | 4,036,402 | 4.09× | 750 | 1000 |

(36 folds on the raw date list; the published 33 is after `prepare_targets` voids labels.)

A rows×rounds cost model gives CV / production-q50 = **2.94×**. The measured ratio is
439.3 / 158.4 = **2.77×**. The phase is therefore fully explained by *expanding windows summing to
4.2× the frame* × *CV rounds at 2/3 of production*, and there is no residual to look for.

### The fix, and why it is safe

Apply the same cap the production split already applies, in the same way — random sample, never
`tail()`, so the calendar window is preserved and expanding-window CV is not silently disabled.
`_per_item_row_sample` (`:3363`) is available for the breadth axis and should follow the same
`per_item_row_sampling` flag the production path uses, so the two cannot diverge.

Measured saving:

| Cap | Saving | Share of 872s |
|---|---:|---:|
| 600k | −56.9s | 6.5% |
| 400k | −137.6s | 15.8% |
| **300k (proposed default)** | **−194.6s** | **22.3%** |
| 200k | −263.6s | 30.2% |
| 100k | −345.0s | 39.6% |

**What is untouched:** fold count, validation rows, OOF record count, distinct forecast dates,
`mean_rank_ic`, the PT statistic. Only the model each fold fits sees less data.

**What changes, and why it is the safe direction:** the OOF residuals then describe a slightly
weaker model than the served one, so `q_hat` — a quantile of `|residual| / sigma` — comes out
larger. The band gets **wider**, i.e. over-coverage against the 80% nominal. A calibration set
that errs toward over-coverage is a conservative failure; the alternative (a band fitted on a
*stronger* model than the one served) is the unsafe one, and nothing here moves in that
direction.

**Only `train_df` is thinned.** Thinning `val_df` would move the evaluation cohort, which is the
artifact pairing exists to remove — the same reasoning `_build_production_split` records at
`:2754-2756`.

**Why this beats the documented lever 1.** `CV_STEP_DAYS` 150 → higher buys the same seconds by
destroying folds. Folds are the conformal calibration set, the confidence-threshold fit set, and —
since 2026-08-08 — the sample size of `mean_rank_ic` and the PT statistic, all of which are
clustered by date. The row cap costs none of that.

**Prior art in this repo:** `docs/changelog/2026-08-07-per-item-row-sampling.md` observed exactly
this asymmetry — *"the sampler thins the 36-second part of a 541-second run"* — and did not act
on it.

### Default choice

**300,000**, `CV_MAX_TRAIN_ROWS`, env-overridable. Rationale: it is the largest cap that clears
20% of the retrain, and it leaves each fold with ~30% of the production frame — comfortably above
the 100K config that trained the shipped model for months. A lower cap is available if the
coverage check comes back clean.

---

## A2. Optuna optimises a criterion the project discarded

`_optuna_search_params` (`:2776`) builds its Dataset once and reuses it across trials, which is
correct. The objective is not:

```python
_num_rounds = 100 if horizon == 7 else 200
opt_callbacks = [lgb.early_stopping(20), lgb.log_evaluation(0),
                 LightGBMPruningCallback(trial, "quantile")]
model = lgb.train(params, dtrain, num_boost_round=_num_rounds,
                  valid_sets=[dval], callbacks=opt_callbacks)
return model.best_score["valid_0"]["quantile"]
```

That is **early-stopped validation pinball loss on the trailing window** — the exact criterion
`FIXED_BOOST_ROUNDS` was introduced to stop using. The comment at `:551-560` states the conflict
in the code itself: at 14d and 30d the validation-loss optimum is **25 rounds** while rank IC
peaks at **500–750**. The two are anti-correlated, and the 2026-08-08 fix changed the *training*
and *CV* fits without touching the selector that chooses the parameters those fits use.

Consequence: the `num_leaves`, `learning_rate`, `lambda_*`, `max_depth`, `min_data_in_leaf` and
`subsample` values sitting in `meta.json` for 14d and 30d — the two noisiest horizons, and the two
that still search (`SKIP_HP_HORIZONS = [3]`) — were selected against a loss the project no longer
believes.

### The fix

Score each trial on **within-date rank IC** over the validation window, at
`_boost_rounds(horizon, cv=True)`, with no early stopping. `_within_date_rank_ic` (`:5958`)
already exists and already returns `None` on degenerate dates, so the objective is:

```python
ic = self._within_date_rank_ic(model.predict(X_val), y_val, val_dates)
return -(ic if ic is not None else 0.0)   # study minimises
```

Three details that decide whether this works:

1. **Within-date, never pooled.** A pooled Spearman reintroduces the market factor and would
   select for the base-rate tracking the PT test exists to reject. `_within_date_rank_ic` groups
   by date by construction; do not replace it with `scipy.stats.spearmanr` over the flat arrays.
2. **`LightGBMPruningCallback` must go.** It prunes on the metric LightGBM reports (`quantile`),
   not on the returned objective, so leaving it in prunes trials on the criterion being replaced.
   With early stopping removed there is no intermediate value to prune on anyway. Losing the
   pruner costs trials: budget the same 35s by keeping `N_TRIALS_MAP` where it is and accepting
   that each trial now runs to fixed rounds.
3. **`_num_rounds` must match CV, not production.** Tuning at 1000 rounds and evaluating at 750
   would select params that only pay off at a depth CV never reaches. `_boost_rounds(horizon,
   cv=True)` is the right call — 200/500/100/750.

**Cost:** Optuna is 35.0s / 4.0% of the retrain. Removing early stopping raises per-trial cost and
removing the pruner raises trial count actually run to completion; the two partly offset. Measure
it, and if it lands above ~60s cut `N_TRIALS_MAP[14]` and `[30]` from 15 to 10 rather than
reverting.

**Requires:** `FORCE_HP_SEARCH=1` on the first run, and a `MODEL_ARTIFACT_VERSION` bump (5 → 6) so
a cached `meta.json` cannot silently supply params chosen under the old criterion.

---

## A3 / A4. The allowlist, the prune, and the eight discarded blocks

Order in `build_training_data` today (`:3475-3489`):

```
_select_feature_cols  →  123 numeric candidates
_prune_features       →  118            (25.2s)
_apply_feature_allowlist → 33           (~0s)
```

`_prune_features` costs 25.2s over 123 columns and 1.65s over 33. On the production frame it drops
**zero** price_technicals features — 33 in, 33 out — so reordering is output-identical *here*.

**It is not provably identical in general.** `_prune_features` keeps the **lower-indexed** member
of each >0.95 pair (`:2413-2417`), and index order comes from `_select_feature_cols`, which does
not sort by group. So a price feature *could* be dropped in favour of a non-allowlisted partner
under the current order, and would survive under the new one. That is a difference in the safe
direction — it keeps a feature the model is allowed to use — but it is a behaviour change and must
be gated and asserted, not assumed.

Feature engineering itself, measured block by block:

| Block | s | group | survives? |
|---|---:|---|---|
| `_compute_price_features` | 9.06 | price_technicals | **yes** |
| `_add_cross_sectional_features` | 2.15 | cross_sectional | no |
| `_add_supply_depth_features` | 2.09 | supply_depth | no |
| `_add_item_identity_features` | 1.69 | item_identity | no |
| `_add_item_metadata_features` | 1.15 | item_metadata | no |
| `_add_supply_side_features` | 0.79 | item_identity | no |
| `_add_temporal_features` | 0.39 | temporal | no |
| `_add_event_features` | 0.01 | events | no |
| `_add_social_features` | 0.00 | social | no |
| **total** | **17.5** | | **8.5s discarded** |

### Two constraints that shape the implementation

**The skip must be a flag on `engineer_features`, never a deletion.** Seven harnesses build their
own frame and then call `_apply_feature_allowlist` on it, so they require the full 123-column
frame:

`ab_test_training_breadth.py:341` · `ab_test_train_universe.py:348` ·
`ab_test_item_metadata.py:306` · `ab_test_csfloat_basis.py:340` ·
`ab_test_interval_sampling.py:533` · `ab_test_q50_sampling.py:412` ·
`ab_test_direction_labels.py:246`

`HORIZON_EXCLUDED_GROUPS` (`:339`) and `_validate_feature_groups` (`:2427`) are also written in
terms of groups the allowlist has already removed, and would need the same frame to remain
meaningful.

**The skip must be derived from the allowlist, not hard-coded.** Track C4 proposes re-admitting
`cross_sectional`. If the skip list is a literal, that change would silently produce a frame with
`cross_sectional` in the allowlist and its columns absent — median-filled to zero, undetectably.
Derive the skip set as `{group for group in ALL_GROUPS if group not in effective_allowlist}` and
let `bymykel_metadata_enabled()` feed it the same way it feeds the allowlist at `:3479-3481`.

---

## A5. `CV_DIAGNOSTIC_CLASSIFIER` default

`:5934` returns `os.environ.get("CV_DIAGNOSTIC_CLASSIFIER", "1") != "0"` — default **on**. CI sets
it off. Measured 2026-08-09 by running the shipped config both ways: **872s off against 1804s on**,
so the diagnostic costs **932s, 52% of a classifier-on retrain**, to populate
`classifier_accuracy` and `classifier_accuracy_ge1` in `cv_results` → `meta.json`, read by nothing
in `models/`, `api/` or `scripts/`.

Flipping the default puts a local retrain at 872s (14.5 min), inside the project's 30-minute run
cap, which it currently exceeds at 30.1 min.

**This is a reporting change as much as a cost one.** `mean_classifier_acc_ge1` has been the
project's headline diagnostic for months and would vanish from local `meta.json`. The replacement
is `mean_rank_ic` plus the PT verdict, both computed from `fold_p50` (`:4227-4251`) and therefore
unaffected — which is the right substitution on the merits, since
`docs/research/2026-08-08-model-review.md` §2 measured that 51% of fold-to-fold variance in
`classifier_accuracy` is explained by the fold's realised direction mix.

Keep the env var. Research runs that want the old number set `CV_DIAGNOSTIC_CLASSIFIER=1`.

---

## A6. The voted cache cannot hit in CI

`_archive_fingerprint` (`:3510`):

```python
parts.append(f"{path.name}:{st.st_size}:{st.st_mtime_ns}")
```

Two independent failures:

1. **The directory does not survive the job.** `self.cache_dir` is `backend/data` (`:641`),
   gitignored, and `price-forecast.yml` caches only `backend/models/saved_models`
   (`:91-96`, `:216-219`).
2. **The key would change anyway.** CI checks the archive out fresh every run
   (`price-forecast.yml:58-68`), so every file's `st_mtime_ns` is new on every run and the key
   changes unconditionally. Fingerprinting on mtime is structurally CI-hostile.

Cost of the miss: **~48s per run** — 21.1s DuckDB read (16.5M raw rows → 7.57M after the
backfilled-slug join) plus 27.2s voting (→ 6,080,631 voted rows). This is paid **daily**, by the
predict path too, not weekly.

### The fix

Replace mtime with a content-derived component: per file, `(row_count, max_day)` read via DuckDB
metadata, or — cheaper and stronger — the archive repo's commit SHA when one is available. Add
`backend/data` to the workflow cache with a key that includes `VOTED_CACHE_VERSION`.

**`VOTED_CACHE_VERSION` must still be bumped by hand** whenever `_fetch_voted_price_history` or
`_apply_multi_source_voting` changes. The key cannot see code, and A6 does not change that — it is
the item-universe rule and it stays. Currently **4**. Note that G3 (6c) will bump it to 5, so
sequence A6 before or after that change deliberately, not concurrently.

`_apply_multi_source_voting` itself is **not** the problem it looks like: `:1206-1217` splits
single-source item-days into a vectorised `groupby().agg` and only routes multi-source days
through `groupby().apply(vote)`. On the unfiltered archive 16.0% of item-days are multi-source and
voting costs 242.5s; on the backfilled universe actually used it is **27.2s**. Vectorising it is
not worth doing.

---

## What is deliberately not in this spec

- **Split conformal / ACI** (Track C5). It is the bigger prize — one fit instead of 33 — but it is
  a scheme change and needs a per-regime-window coverage read that has no definition in the code.
  A1 gets 22% for a bounded change; do it first and measure coverage, then decide.
- **`CV_STEP_DAYS`.** Strictly dominated by A1.
- **`max_bin`, `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `force_row_wise`,
  `feature_pre_filter`.** Measured dead or deliberate. See "What the measurement changed".
- **Regime models** (95.4s / 10.9%). A decision, not a cost lever — the case rests on the
  degenerate 1-tree and 3-tree boosters in the deployed set, not on seconds. Track C6.
- **Booster reuse across CV folds.** Impossible: every fold's calibration rows must be unseen by
  the model producing them.
- **`reference=` on the CV and classifier Datasets.** Real but ≈6s (0.37s → 0.20s per construct
  over 36 CV folds plus ~8 classifier Datasets). Fold it into A1's diff if convenient; not worth
  its own task.

---

## Verification

After the full Track A lands:

```bash
cd backend
venv/bin/python -m pytest tests/test_forecaster.py tests/test_minimal_model_shape.py \
    tests/test_fixed_boost_rounds.py tests/test_cv_cohort_parity.py -q
FORCE_HP_SEARCH=1 venv/bin/python scripts/forecast_prices.py --train-only
```

Then read from the fresh `meta.json`:

| Check | Expectation |
|---|---|
| Total training seconds | ≈600s, from 872s |
| `cv_results[h]["n_folds"]` | **unchanged** at 9 (8 at 30d) — if this moved, A1 thinned dates instead of rows |
| OOF record count per horizon | **unchanged** — same guard, val side |
| Empirical band coverage | ≥ `NOMINAL_COVERAGE = 0.80`; over-coverage is expected and acceptable, under-coverage fails the change |
| `mean_rank_ic` per horizon | ≥ the pre-change value, or within its own fold-clustered interval |
| `horizon_feature_cols` | **identical set** to the pre-change artifact — A3/A4 must not change which features reach a booster |
| Optuna seconds | ≤ ~60s; above that, cut `N_TRIALS_MAP[14]`/`[30]` to 10 |

Run the suite as `pytest tests`, never bare `pytest` — `scripts/test_social_signal.py` imports
`thefuzz`, which is not in `requirements.txt`, and collection aborts.

**Do not use `SKIP_CV=1` for any of this.** `q_hat` and the confidence thresholds are both fitted
on the CV out-of-fold predictions, and A1 changes exactly those rows.
