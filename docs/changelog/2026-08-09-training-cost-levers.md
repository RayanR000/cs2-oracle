# Training cost levers, and what measuring them refuted

**Date:** 2026-08-09
**Plan:** `docs/superpowers/plans/2026-08-09-training-cost.md` (Phase 1 of
`docs/superpowers/plans/2026-08-09-master-execution-order.md`)
**Branch:** `training-cost`

Six changes to the retrain, plus two production bugs the verification retrain
exposed. The headline is not the speedup: it is that **three of the plan's own
numbers did not reproduce**, and that the retrain got *slower*, for a reason
worth paying.

---

## What shipped

| # | Change | Effect |
|---|---|---|
| 1.1 | `CV_MAX_TRAIN_ROWS = 300_000` on each CV fold's training rows | Binds at exactly 300,000 on all four horizons |
| 1.2 | Optuna selects on within-date rank IC, not early-stopped pinball loss | **Costs 392.3s.** `MODEL_ARTIFACT_VERSION` 5 → 6 |
| 1.3 | Feature allowlist runs before the correlation prune | −16.5s combined with 1.4 |
| 1.4 | `engineer_features` skips the blocks the allowlist discards | (see above) |
| 1.5 | `CV_DIAGNOSTIC_CLASSIFIER` defaults off | Local/research only; CI already set it to `0` |
| 1.6 | `_archive_fingerprint` is content-derived, and CI caches `backend/data` | The voted cache could never hit in CI before |

## Measured cold retrain: 1426.3s

`FORCE_HP_SEARCH=1 … --train-only`, local, `CV_DIAGNOSTIC_CLASSIFIER` unset
(so off), archive through 2026-08-08.

| Phase | 3d | 7d | 14d | 30d | Total |
|---|---|---|---|---|---|
| Optuna | 0.0 | 107.8 | 37.6 | **246.9** | **392.3** |
| q50 ensemble | 22.5 | 68.9 | 11.8 | 99.2 | 202.4 |
| Direction classifier | 13.6 | 37.4 | 11.5 | 59.9 | 122.4 |
| Conformal CV | 53.1 | 115.1 | 28.0 | **238.6** | **434.8** |
| Horizon total | 122 | 402 | 105 | — | **1426.3** |

3d's Optuna is 0.0s because `SKIP_HP_HORIZONS = [3]` freezes it on its
50-trial winner.

**The plan predicted ≈600s, from 872s. That is not what happened, and the
prediction was not sound to begin with.** Three confounds, none of them
separable after the fact:

1. **1.2 made HP search dramatically more expensive.** Removing early stopping
   means every trial trains the full `_boost_rounds(horizon, cv=True)` rather
   than stopping around 25 rounds. Optuna is now the single largest phase of a
   cold retrain at 392.3s, 27.5% of the run. This was a foreseeable consequence
   of the change and the plan did not budget for it.
2. **`FORCE_HP_SEARCH=1` forces a search a normal retrain reuses from cache.**
   The 872s baseline did not necessarily pay it.
3. **The 872s baseline was measured in CI, on different hardware, under early
   stopping, before `beec500` moved training to fixed boost rounds.** Comparing
   a local 1426.3s to it is not like-for-like, and no like-for-like control was
   run.

**Do not quote a speedup from this work.** What is established is the direction
of each individual lever and the new cost profile, not a net total.

## D2 — the coverage gate

The plan's gate was: read empirical coverage against `NOMINAL_COVERAGE = 0.80`
from the fresh `meta.json`; below 0.80, stop and raise `CV_MAX_TRAIN_ROWS`.

**That check is not executable, and could not have failed.** `meta.json` stores
no coverage field, and the run log reports only `target coverage=80%`. More
fundamentally, `q_hat` *is* the finite-sample conformal quantile of the pooled
OOF nonconformity scores, so coverage on that pool is ≥ 0.80 by construction at
n = 155,923–174,784. A conformal calibration cannot under-cover its own
calibration set.

The fallback the plan names is the informative one — did 1.1 thin **rows** or
**dates**?

| h | folds | max fold `n_train` | OOF rows | q_hat |
|---|---|---|---|---|
| 3 | 9 | 300,000 (capped) | 174,784 | 95.25 |
| 7 | 8 | 300,000 (capped) | 157,770 | 137.68 |
| 14 | 8 | 300,000 (capped) | 157,411 | 205.99 |
| 30 | 8 | 300,000 (capped) | 155,923 | 309.49 |

Fold counts are 9/8/8/8, matching the plan's expectation of "9, 8 at 30d". OOF
pools are six figures at every horizon. `q_hat` rose at all four horizons, so
the band got **wider** — the over-coverage direction the cap predicts, because
each fold model now fits on less data than the served one. **Verdict: proceed.**

## The plan's baselines were unusable

The plan asks three times to compare against "the previous artifact". On this
machine that artifact is **`model_artifact_version: 3`, trained 2026-08-06** —
the same stale cache CI was rejecting. It predates both the v5 dollar-scale
migration and the 2026-08-08 ≥$1 universe, so every comparison against it is
confounded:

- **`horizon_feature_cols` 36 → 33** decomposes exactly as the v5 change:
  `macd_line`/`macd_histogram` → `_rel` forms, `price_std_*` → `price_cv_*`,
  `price_log` and `price_lag_*` gone. Nothing to do with 1.3 or 1.4.
- **Fold `n_train` 109,639 → 300,000** — the folds got *bigger*, because the old
  artifact was trained in the 99-item subsample era.
- **Fold `n_val` ~22,000 → ~157,000**, same cause.

Taken at face value, the plan's check would have reported a false positive on
1.3/1.4 and a false "cap did nothing" on 1.1.

**The replacement check isolates the changes properly**: same code, same frame,
knobs flipped —

```
A (shipped)  ALLOWLIST_BEFORE_PRUNE=True,  skip_unused_groups=True   33 features, 8.0s
B (control)  ALLOWLIST_BEFORE_PRUNE=False, skip_unused_groups=False  33 features, 24.5s
```

Set-identical, and equal to the shipped `meta.json`'s `feature_cols`. The
invariant "no task changes which features reach a booster" holds, and the
combined saving is **16.5s** — not the 33.7s (25.2 + 8.5) the plan projected.

## The refuted micro-levers

`docs/architecture/model-optimization.md` listed four tuning levers as
"available". Measured per boosting round on the production frame, against a
**25.5 ms** baseline:

| Lever | ms/round | Verdict |
|---|---|---|
| `max_bin` 63 → 31 | 26.3 | slower |
| `num_leaves` 47 → 31 | 26.2 | slower |
| `min_data_in_leaf` 15 → 100 | 27.1 | slower |
| `feature_fraction` 0.7 → 0.4 | 28.9 | slower |

All four are slower than doing nothing. The "~10–20% of 28.1s" figures they
carried were estimates, never measurements. Struck from the doc; do not
re-propose.

## Two bugs the retrain caught that tests did not

Both are worth recording because in both cases a green suite meant nothing.

**1. `_archive_fingerprint` named a column that does not exist.** 1.6's first
implementation keyed on `MAX(date)`. The archive's date column is `day`, so the
first cold retrain died with `Binder Error: Referenced column "date" not
found`. All 1693 tests passed, because the plan's fixture and mine had both
invented a `date` column. The fingerprint now names no column at all — row
count plus byte size — which is the right shape independently: the archive is
**not schema-uniform** (`prices-2026-03` and `-04` carry `min_price`/`max_price`
the other 19 files do not). Dropping `st_mtime_ns` was always the entire fix;
`st_size` was already content-derived and survives a checkout.

**2. A NULL `volume` column crashed feature engineering in production.**
Unrelated to this branch. **2026-08-08 is the first archive day whose `volume`
is mostly NULL** — 33,613 non-null of 361,453 rows, against 100% populated on
every earlier day. DuckDB hands pandas a nullable `Int64`; the 7-day shift puts
`pd.NA` into the derived log-change; `(x > 0)` becomes `BooleanDtype` carrying
NA; `.astype(int)` raises `cannot convert NA to integer`.

This was latent behind the artifact-version failure: every CI run for weeks
died earlier, at the v3-vs-v5 check, so training had not actually run in CI
since before the feed died. Clearing that blocker exposed this one. Fixed with
`fillna(False)`, which is the pre-existing semantics rather than a new choice —
on the numpy path `NaN > 0` was already `False`, "no volume confirmation" — so
it is a no-op on all 13 years of prior data. Cherry-picked to `main` as
`6ddc268`, ahead of the branch, because `mode=full` cannot succeed without it.

## Not established

- **Any net speedup.** No like-for-like control was run. See the three confounds
  above.
- **That 1.2 improves accuracy.** It changes the *selection criterion* to the one
  the project uses. Whether the selected hyperparameters forecast better is
  unmeasured, and the fresh CV is not comparable to the v3 artifact's.
- **That 1.1 is safe at the served band.** Coverage is guaranteed on the
  calibration pool by construction; nothing here measures coverage on held-out
  data. If the served band is ever observed under-covering, `CV_MAX_TRAIN_ROWS`
  is the first knob to raise.
- **Anything about the 2026-08-08 volume outage beyond its shape.** The feed is
  still broken; only the crash is fixed. Track D2 (`prices.csgotrader.app`
  per-provider paths) is the actual repair.

## Sequencing note for Phase 2

1.6 added a CI cache key named `voted-v4-`, matching `VOTED_CACHE_VERSION = 4`.
Phase 2 bumps that to 5 (`n_ask_sources`) and then 6 (excluding the Steam
trailing-window feeds from voting). **Each bump must move the workflow key in
the same commit**, or CI restores a frame voted under the superseded rule and
trains on it silently. The obligation is written at the step itself in
`price-forecast.yml`, not only here.
