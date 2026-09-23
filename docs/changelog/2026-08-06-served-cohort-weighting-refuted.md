# Served-cohort weighting is refuted — reweighting the ≥$1 rows moves the classifier ~0.1pp

**Date:** 2026-08-06
**Spec:** `docs/specs/2026-08-06-served-cohort-weighting-design.md`
**Change:** a served-cohort training weight for the directional classifier, built,
measured on two paired cold retrains, and **left defaulted off**.
`DEFAULT_SERVED_COHORT_SHARE = None`, so production behaviour is byte-identical to
before. Incidentally: a `FORECAST_MODEL_DIR` override, which is the first way to
run a training arm without overwriting the deployed artifact.

## The hypothesis: a row-level train/serve population mismatch

Production serves only ≥$1 — `api/serving_policy.py` sets
`MIN_SERVED_PRICE_USD = 1.0`, deliberately equal to the lower bound of
`HEADLINE_MIN_TIER` (`backtest/scoring.py`, `HEADLINE_MIN_TIER = 1`).
`backend/models/forecaster.py` applies **no price filter to training**:
`price_tier` is a feature and a scoring partition, never a row filter and never a
weight, and `_stratified_item_subsample` stratifies on **rarity**, so the
subsample inherits the pool's tier mix. The classifier's only weighting,
`DIRECTION_MOVER_WEIGHT_MAP`, is mover-vs-flat and tier-blind.

Measured over the archive (1460-day window to 2026-08-04, `is_backfilled`
cohort, after voting/dead/corrupt filtering — 5,832,742 deduplicated item-days
over 5,378 items; numbers from the spec):

| tier | rule | item-days | share |
|---|---|---|---|
| 0 | < $1 | 4,780,367 | **81.96%** |
| 1–4 | ≥ $1 | 1,052,375 | **18.04%** |

So ~82% of the classifier's loss is computed on rows nobody is shown. The
hypothesis was that this costs served accuracy — the same class of defect as the
volume train/serve gap (`2026-08-06-volume-features-shelved.md`) but on rows
rather than columns.

## Why reopening the closed accuracy work was legitimate

`docs/research/accuracy-opportunities.md` carries a stop banner whose argument is
"six consecutive feature groups measured < 0.7pp against a 1.15pp floor". Every
one of those measurements, and every model-shrinking decision that followed, was
scored on the **pooled** metric: `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`
(2026-07-24), `HORIZON_EXCLUDED_GROUPS` dropping `cross_sectional` at 14d/30d
(2026-07-19), the six shelved price primitives (2026-07-31), `N_ENSEMBLES = 1`
and gbdt-only (2026-08-04). `classifier_accuracy_ge1` did not exist until
2026-08-05 (`2026-08-05-cv-cohort-parity-design.md`, landed in `d00e183` /
`080c1b1`).

That reasoning stands. It is the hypothesis that failed, not the licence to test
it — the served-vs-pooled cohort split remains a legitimate reason to re-examine
a decision taken on the pooled number.

## What was built

A served-cohort weight on the **directional classifier only**. The quantile
models and the split-conformal band are untouched (`_compute_sample_weights` is
unchanged), so they cannot confound the read: the served up/flat/down call comes
from the classifier via `_recenter_on_direction`, while the quantiles supply only
a width that conformal calibration then replaces.

In `backend/models/forecaster.py`:

* `_served_cohort_multiplier(base_weights, tiers, served_share)` — solves
  `m*W_s / (m*W_s + W_n) = share`, with `W_s`/`W_n` summed over the **mover**
  weights so the share is exact after both weightings compose. The knob is a
  target *share*, not a raw multiplier, because a multiplier's meaning drifts
  with per-fold tier composition. Returns `1.0` for both degenerate partitions.
* `_direction_sample_weights` gained optional `tiers` / `served_share`.
  `served_share=None` reproduces the old weights byte-identically, which is what
  makes the control arm provably the current model.
* `_direction_class_prior` gained the same two parameters. Its docstring commits
  it to reporting "the distribution the classifier's multiclass objective
  actually sees", so a weight it ignored would make
  `scripts/diagnose_direction_prior.py` silently describe a model that was never
  trained. `tiers` is filtered by the same finite mask as `returns`.
* `_fit_direction_classifier` gained `tier_train`, passed explicitly from both
  call sites — `_train_horizon_inline` uses `train_set["price_tier"]`,
  `_cv_evaluate_horizon` uses `train_df["price_tier"]` — rather than read off
  `X_train`, because the >0.95 correlation prune can drop `price_tier` from
  `feature_cols`, and a weight that switches itself off when a feature is pruned
  is exactly the failure mode `SHELVED_FEATURES` exists to avoid. Validation rows
  are deliberately left unweighted so early stopping still tracks the whole fold.
* `ItemForecaster.__init__` gained `served_cohort_share`, validated to `(0, 1)`
  or `None`.

In `backend/scripts/forecast_prices.py`:

* `TRAIN_SERVED_COHORT_SHARE` env knob, `DEFAULT_SERVED_COHORT_SHARE = None`.
* **`FORECAST_MODEL_DIR`** — independently useful and new. `train()` overwrites
  `meta.json` and the boosters in place, and there was previously no way to run a
  training arm without clobbering the deployed artifact. Env-configured, like
  `TRAIN_FEATURE_ROWS`, because the script parses argv as a plain set.

## How it was measured, and why the pairing is the whole point

Two full cold retrains into scratch directories via `FORECAST_MODEL_DIR` with
`--train-only`; the production artifact was untouched (`models/saved_models/`
still holds the `model_artifact_version` 3, 36-column model trained
2026-08-06 05:17 UTC). Both arms: 99 of 5,378 items, 115,763 subsampled feature
rows, 36 features after the allowlist.

Pairing was verified **before** any metric was read: **identical fold ids, row
counts and date bounds in both arms**, and **identical `tuned_params`**. Optuna
reran per arm — only 3d is in `SKIP_HP_HORIZONS` — but it optimises the quantile
p50 objective, which this change does not touch, and converged to the same
params. Only the training weight vector differed.

The weighting hit its target exactly: the logs report 22.2% of rows (the ≥$1
ones) carrying **50.0%** of training weight in the parity arm.

The control arm reproduced the deployed model closely, which validates the setup:

| horizon | control arm `classifier_accuracy_ge1` | deployed artifact (`meta.json`) |
|---|---|---|
| 3d | 50.10 | 49.9 |
| 7d | 48.74 | 49.0 |
| 14d | 50.81 | 51.1 |
| 30d | 55.02 | 53.4 |

## The result: no effect at any horizon

`classifier_accuracy_ge1`, paired per fold, control → parity at `share=0.50`
(percentage points; these are CV gate numbers, not production DA):

| horizon | folds | control | parity | paired diff | paired sd | t |
|---|---|---|---|---|---|---|
| 3d | 9 | 50.10 | 49.67 | **−0.43** | 2.59 | −0.50 |
| 7d | 9 | 48.74 | 49.92 | **+1.18** | 1.48 | 2.39 |
| 14d | 9 | 50.81 | 50.74 | **−0.07** | 1.74 | −0.11 |
| 30d | 8 | 55.02 | 54.05 | **−0.97** | 1.30 | −2.13 |

The pre-registered rule (set in the design spec before either arm ran) was:
adopt only on a paired mean difference **> +2pp at both 3d and 7d** and not worse
than −1pp at 14d/30d; **within ±1pp → the cohort mix is not the binding
constraint, leave the default at `None`, and do not proceed to the subsample
variant**. All four diffs sit within ±1.2pp with mixed signs. The one nominally
significant horizon (7d, t=2.39) is offset by 30d at t=−2.13, and one nominal hit
across four horizons is unremarkable. This is the "no effect" branch.

**The most informative row is the pooled one**, which the design predicted would
*fall* — the intended trade:

| horizon | control `classifier_accuracy` | parity | paired diff |
|---|---|---|---|
| 3d | 67.37 | 67.37 | −0.00 |
| 7d | 67.22 | 67.36 | +0.13 |
| 14d | 67.67 | 67.71 | +0.04 |
| 30d | 68.90 | 68.99 | +0.09 |

Moving the served cohort from the frame's own ~0.18–0.22 of the weight to 0.50
moved pooled accuracy by ~0.1pp at every horizon. The trade never happened
because the decision function barely moved: the sub-$1 and ≥$1 rows are teaching
the classifier essentially the same function. **Cohort mix is not the binding
constraint on served accuracy.**

Per-fold `classifier_accuracy_ge1`, control → parity:

* 3d: 53.1→47.1, 50.2→47.8, 47.7→49.9, 52.5→54.8, 50.9→51.8, 47.6→46.9,
  48.4→47.2, 50.6→51.4, 49.9→50.1
* 7d: 54.8→53.3, 45.3→44.7, 50.4→53.0, 47.7→49.0, 49.1→50.7, 47.9→48.6,
  46.9→48.1, 49.0→52.1, 47.6→49.8
* 14d: 55.9→55.7, 44.7→44.4, 48.9→52.0, 47.5→47.9, 56.7→56.1, 47.7→47.2,
  52.1→52.3, 51.8→52.7, 52.0→48.4
* 30d: 96.3→96.5, 20.2→21.1, 60.4→59.7, 47.9→46.3, 60.2→58.0, 52.0→49.6,
  47.4→47.6, 55.8→53.6

The 30d spread — fold 1 at 96.3%, fold 2 at 20.2%, both arms — is a reminder that
these folds remain dominated by the market direction of their window, consistent
with `2026-08-03-accuracy-is-clustered-by-forecast-date.md`.

## Decision

* `DEFAULT_SERVED_COHORT_SHARE` stays **`None`**. Production is unchanged,
  byte-identically.
* The code is **kept, defaulted off**, on the same reasoning that retained
  `scripts/ab_test_direction_labels.py` after its own negative result: it is the
  instrument that produced this measurement, and the next person to doubt the
  finding should be able to re-run it rather than rebuild it.

## What was deliberately not done

**The stronger variant — tier-stratifying `_stratified_item_subsample` — was not
run.** The pre-registered rule says not to on a within-±1pp read, and the flat
pooled result is the substantive reason: reweighting and resampling both act on
cohort mix, and reweighting to nearly 3x the served cohort's share of the loss
produced ~0.1pp.

**The conclusion is bounded, and the bound matters.** Weighting tests *capacity
allocation*, not *data quantity*. It reallocates the loss across rows already in
the frame; it does not add ≥$1 items. So this refutes "the classifier is spending
its capacity on the wrong cohort". It does **not** refute "more ≥$1 items would
help" — that is untested.

The archive can support that test if anyone revisits it: the median-≥$1 cohort
holds **925 items / 992,498 item-days**, ~9.9x the 100,000-row budget, with the
1460-day window intact (spec, §Not in scope). The reason not to run it is the
measurement, not availability — it changes the item universe, so the two arms
would score `classifier_accuracy_ge1` over *different* validation items, an
unpaired comparison at exactly the effect size this project cannot resolve
(`2026-07-31-price-primitives-decision-scale.md`).

**No harness was built.** The retrain's own expanding-window CV supplied the
folds, which is what made the comparison paired.

## Verification

`backend/tests/test_served_cohort_weighting.py` — **31 tests** (21 functions, 10
of them from three `parametrize` sets), all passing: control-arm byte-identity
three ways plus no in-place mutation of a caller-owned weight array, exact share
under arbitrary compositions, share exactness *after* composing with the mover
weight (a count-derived multiplier passes the naive version of this and fails
this one), `price_tier >= 2` counted as served, a share below the frame's current
one allowed to down-weight, both degenerate partitions, `(0, 1)` validation at
both the helper and the constructor, class-prior tracking, finite-mask alignment,
and classifier fits with and without `tier_train`.

Full `backend/tests/` suite: **801 passed**. A bare `pytest -q` from `backend/`
additionally collects `scripts/test_social_signal.py`, which fails to import on a
missing `thefuzz` — pre-existing and unrelated, that is the deleted social
collector.

## Provenance caveat I could not resolve

The working tree these arms ran from also carries an **unrelated** change to
`_compute_price_features`: lag lookups became as-of within a new
`LAG_TOLERANCE_DAYS = 3` instead of exact-date, with
`MODEL_ARTIFACT_VERSION` 3 → 4. That is the fix for the calendar-gap median-fill
left open in `2026-08-06-serving-down-skew-refuted.md`, and it gets its own
entry. Whether it was present when the two arms ran is **not determinable from
the diff**. It does not affect this read — both arms ran against the same tree,
so the paired contrast holds either way — but the absolute levels above should be
attributed to "the arms' tree", not specifically to the deployed v3 artifact.

## Still open

* Whether *more* ≥$1 training items would help. Untested, unpaired-by-nature,
  data available. See the bound above.
* The market-date domination of DA (`2026-08-03-accuracy-is-clustered-by-forecast-date.md`),
  visible again in the 30d per-fold spread here, remains the larger unexplained
  term in served accuracy.

## Related

* `docs/specs/2026-08-06-served-cohort-weighting-design.md` — full
  rationale, archive cohort measurement, pre-registered rule.
* `2026-08-06-volume-features-shelved.md` — the column-level version of the same
  train/serve mismatch, which *was* real.
* `2026-08-05-cv-cohort-parity-design.md` and `080c1b1` — where
  `classifier_accuracy_ge1` came from, without which this hypothesis could not
  have been stated.
* `docs/research/accuracy-opportunities.md` — the stop banner, now carrying a
  line recording this result.
