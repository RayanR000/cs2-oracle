# Served-Cohort Weighting for the Directional Classifier (2026-08-06)

## Problem

The 2026-08-06 artifact's CV directional accuracy, split by cohort:

| horizon | `mean_classifier_acc_ge1` | `mean_classifier_acc` (pooled) |
|---|---|---|
| 3d | **49.9%** | 67.3% |
| 7d | **49.0%** | 67.2% |
| 14d | **51.1%** | 67.8% |
| 30d | **53.4%** | 68.6% |

The left column is the population production serves: `api/serving_policy.py` sets
`MIN_SERVED_PRICE_USD = 1.0`, deliberately equal to the lower bound of
`HEADLINE_MIN_TIER`. The right column is roughly the penny-item score.

Measured over the archive, in the 1460-day window (2022-08-07 → 2026-08-04) and
the `is_backfilled` cohort, after voting/dead/corrupt filtering — 5,832,742
deduplicated item-days over 5,378 items:

| tier | rule | item-days | share |
|---|---|---|---|
| 0 | < $1 | 4,780,367 | **81.96%** |
| 1–4 | >= $1 | 1,052,375 | **18.04%** |

So the docs' "~83% tier-0" is right: the classifier's loss is ~82% sub-$1.

`forecaster.py` applies **no price filter to training**. `price_tier` is a
feature (line 1097) and a scoring partition (line 4698); it is never a row
filter and never a weight. `_stratified_item_subsample` stratifies on
**rarity**, so the subsample inherits the pool's tier mix. The directional
classifier's only weighting is `DIRECTION_MOVER_WEIGHT_MAP` — mover-vs-flat,
tier-blind.

So the model's capacity, its HP search, its correlation prune and its loss
weighting are ~83% spent on items nobody is shown. That is the same class of
defect as the volume train/serve gap
(`docs/changelog/2026-08-06-volume-features-shelved.md`) — a population
mismatch — but on rows rather than columns.

### Why this reopens work the stop banner closed

`docs/research/accuracy-opportunities.md` carries a stop banner whose argument
is "six consecutive feature groups measured < 0.7pp against a 1.15pp floor".
Every one of those measurements, and every model-shrinking decision that
followed, was scored on the **pooled** metric:

| decision | dated | scored on |
|---|---|---|
| `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` | 2026-07-24 | pooled |
| `HORIZON_EXCLUDED_GROUPS` drops `cross_sectional` at 14d/30d | 2026-07-19 | pooled |
| six shelved price primitives | 2026-07-31 | pooled |
| `N_ENSEMBLES = 1`, gbdt-only | 2026-08-04 | pooled |

`classifier_accuracy_ge1` did not exist until 2026-08-05
(`2026-08-05-cv-cohort-parity-design.md`). The banner's bar for reopening is "a
genuinely new data source". This change needs none: it is a correction to which
rows the existing loss is computed over.

## Change

Add an optional **served-cohort weight** to the directional classifier's
training weights. Rows with `price_tier >= HEADLINE_MIN_TIER` get their weight
scaled so the served cohort carries a configured share of total training weight.

The knob is a **target share**, not a raw multiplier. A raw multiplier's meaning
drifts with the frame's tier composition, which varies per fold; a share is the
quantity we actually mean, and the multiplier is derived per fit from the frame
in hand.

```
m = share * W_nonserved / ((1 - share) * W_served)
```

where `W_served` / `W_nonserved` are sums of the **mover weights** over each
partition — so the resulting share is exact after both weightings compose.

- `DEFAULT_SERVED_COHORT_SHARE = None` in `scripts/forecast_prices.py`.
  `None` means "no tier weighting", which is byte-identical to today's
  behaviour. The control arm is therefore provably the current model.
- `TRAIN_SERVED_COHORT_SHARE` env var overrides it, matching the
  `TRAIN_FEATURE_ROWS` / `SKIP_CV` convention (the script parses argv as a
  plain set, so a flag would not fit).
- Threaded into `ItemForecaster` as a constructor argument, then into
  `_direction_sample_weights(returns, threshold, mover_weight, tiers, served_share)`.

### Scope: the classifier only, not the quantile models

The served up/flat/down call comes from the classifier
(`_recenter_on_direction`); the quantile models supply only the interval width,
which is then replaced by split-conformal calibration. Weighting the quantile
models would move the band **and** the conformal calibration, confounding the
read on the metric we are targeting. `_compute_sample_weights` is untouched.

### Interaction with `_direction_class_prior`

That helper's docstring commits it to reporting "the distribution the
classifier's multiclass objective actually sees". It is only consumed by
`scripts/diagnose_direction_prior.py` and `tests/test_direction_prior.py`, but
if the classifier gains a weight term the helper does not model, the diagnostic
silently starts lying. It gets the same two optional parameters and passes them
through.

## Measurement

Target: `mean_classifier_acc_ge1` per horizon, from the retrain's own
expanding-window CV (8–9 folds). No new harness.

**This comparison is fully paired**, which is the reason for choosing weighting
over the stronger intervention. Only the training weight vector changes, so both
arms see identical folds, identical feature matrices, identical imputation
medians and identical validation rows. Per-fold differences are therefore
paired, and the fold-level sd that sets the A/B harness's 1.15–7.13pp MDE
(`2026-07-31-price-primitives-decision-scale.md`) is the *unpaired* quantity —
the paired sd is what applies here and is smaller.

Arms, one full retrain each (~15–18 min warm):

| arm | `TRAIN_SERVED_COHORT_SHARE` | served share of training weight |
|---|---|---|
| control | unset | ~0.18 before mover weighting (whatever the frame gives) |
| parity | `0.5` | 0.50 exactly |

Report per-horizon per-fold `classifier_accuracy_ge1` for both arms, the paired
mean difference, and the paired sd. Decide on the paired numbers, not the means.

`mean_classifier_acc` (pooled) is reported alongside and is **expected to fall**
— that is the intended trade, not a regression. It is not the served metric.

### Pre-registered decision rule

- Paired mean difference **> +2pp** at 3d and 7d (the two horizons below 50%),
  and not worse than −1pp at 14d/30d → adopt, set the default, run the stronger
  subsample variant next.
- Difference within ±1pp → the cohort mix is not the binding constraint. Record
  it, leave the default at `None`, and do **not** proceed to the subsample
  variant.
- Negative beyond −1pp at 3d/7d → the penny rows are carrying transferable
  signal. Record and stop.

The threshold is a judgement call, set above the 1.15pp unpaired floor so a
positive read is not a restatement of fold noise.

## Not in scope

**Tier-stratifying `_stratified_item_subsample`** is the stronger version of the
same idea: it would put ~5.5x more ≥$1 rows in the frame rather than merely
reweighting the ones already there.

The obvious objection — that there is not enough ≥$1 history to fill the budget
— was checked against the archive and **does not hold**:

| cohort | items | item-days | items a 100K budget draws | % of cohort sampled |
|---|---|---|---|---|
| all pool (today) | 5,378 | 5,832,742 | 92 | 1.7% |
| median >= $1 | 925 | 992,498 | 93 | **10.1%** |
| ever >= $1 | 2,546 | 2,801,878 | 90 | 3.5% |

The median≥$1 cohort holds ~9.9x the 100,000-row budget, so it fills with no
per-item repetition, and the 1460-day window survives intact — every quarter
carries 25K–96K ≥$1 item-days and the ≥$1 share is a stable 15–22%. It also
*raises* the sampling fraction from 1.7% to 10.1% of the cohort. (Item breadth
is thinner early: only 488 of the 925 items have 2022Q3 data, against all 925
from 2025Q4 on.)

It is deferred anyway, for the measurement reason rather than the data one: it
changes the item universe, so the two arms would score
`classifier_accuracy_ge1` over *different* validation items — an unpaired
comparison at exactly the effect size this project cannot resolve. It also moves
`market_return_30d`, and with it the bull/range/bear regime labels, since
`_add_cross_sectional_features` runs after the subsample. Run it only if the
weighting arm clears its bar.

## Testing

`backend/tests/test_served_cohort_weighting.py`:

- share `None` returns exactly the current mover weights (control is unchanged)
- the derived multiplier hits the requested share to floating tolerance
- the share is exact after composing with the mover weight, not before
- no served rows → weights unchanged rather than a divide-by-zero
- all served rows → weights unchanged (target already met, nothing to shift)
- `share` outside `(0, 1)` is rejected
- a share below the frame's current one down-weights, and is allowed
- `_direction_class_prior` moves when the tier weighting is applied
- `_fit_direction_classifier` accepts tiers and trains

Plus the existing `tests/test_direction_prior.py` and
`tests/test_forecaster.py` must stay green.
