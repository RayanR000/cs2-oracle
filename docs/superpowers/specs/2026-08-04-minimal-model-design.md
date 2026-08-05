# The Minimal Model: 40 LightGBM Models → 8

Date: 2026-08-04

## Summary

Training the forecaster costs ~462s across 40 models. Of that, **303s goes to 24
p10/p90 quantile GBMs** whose interval achieves 39–48% empirical coverage against
its nominal target, and which is corrected by a conformal adjustment at serve time
anyway. Their top feature is `price_std_60d` in 8 of 12 ensembles — they are
expensively rediscovering "band width ≈ recent volatility."

This spec replaces them with that statement, computed directly. The served band
survives; the models producing it do not. Combined with dropping the ensemble from
3 members to 1 and switching 14d/30d from DART to GBDT, the model count falls
**40 → 8** and training falls to an estimated **35–70s**.

Because "still fairly accurate" is the binding requirement, the spec **first**
repairs the only fresh-model gate in the repository — `walkforward_backtest.py`
currently weights folds by `sample_count`, reporting a tighter confidence interval
than it has earned — and pre-registers the acceptance bar before any arm is run.

This is spec #2 and #1 combined from the sequence recorded in
`docs/superpowers/specs/2026-08-04-remove-accidental-retrain-work-design.md`
("Findings recorded for later specs"). Merging them avoids building the measurement
rig twice.

**This spec does almost nothing for the daily prediction run.** That was spec #0's
job (835s → ~180–240s), and what remains there is data fetching and feature
engineering, not model scoring. The win here is local retrain iteration and
training cost.

## Background: where the 462s goes

| Component | Models | Cost |
|---|---|---|
| p10 + p90 quantile GBMs | 24 (4H × 2Q × 3E) | ~303s |
| p50 quantile GBMs | 12 (4H × 1Q × 3E) | ~138s |
| Directional classifiers | 4 (4H) | ~21s |
| **Total** | **40** | **~462s** |

The 4 classifiers that cost 21s produce the **served up/flat/down call and its
confidence** (`forecaster.py:2981-2982`: "the quantile models only supply the
interval"). The headline DA metric, the trust warning, and the opportunity ranking
are all judged on the classifier output. So 441 of the 462 seconds serve the band,
not the direction.

Production DA is **3d=48.4% / 7d=49.4% / 14d=50.8% / 30d=46.7%** against ~33%
chance with three labels (`docs/architecture/model-optimization.md`, re-resolved on
prod 2026-08-02). The edge is real but modest, which is the reason to ask whether
the complex model is earning its cost.

⚠️ **The cost breakdown above is not trustworthy to the second.** Spec #0 reports
`465s` total retrain, `137s` fetch, and `401s` for "14d + 30d DART" — figures that
sum to more than the total, so they came from different runs or overlap.
Task 2 below establishes a clean baseline before anything is compared.

## Part 1 — The measurement rig

Nothing in Part 2 may be merged before Part 1 is committed and its effect verified.

### Task 1: route `walkforward_backtest.py` through the clustered scorer

`scripts/walkforward_backtest.py:314-325` aggregates per-fold metrics like this:

```python
total_n = sum(f["sample_count"] for f in fold_results)
agg = {
    ...
    "directional_accuracy": sum(f["directional_accuracy"] * f["sample_count"]
                                for f in fold_results) / total_n,
```

Weighting by `sample_count` treats every item-row inside a fold as an independent
observation. It is not: directional accuracy is dominated by market-wide regime
moves, so the effective sample size is the number of independent validation
**dates**. This is the exact error documented in
`docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md`.

`backend/backtest/scoring.py` already carries the correct machinery — forecast-date
clustered bootstrap CIs (`directional_accuracy_ci_clustered_lower/upper`,
`scoring.py:199-200`), `distinct_forecast_dates`, and
`date_coverage_sufficient` against `MIN_FORECAST_DATES = 20` (`scoring.py:38`).
Route the walkforward aggregation through `score_cohort` so there is one definition
of how a cohort is scored.

**Verification that this landed:** re-score the *current, unmodified* model. The
reported confidence interval must come out **wider** than before. A rig fix that
does not reduce apparent precision did not do anything.

⚠️ **Pre-fix and post-fix walkforward DA numbers are not comparable**, for a reason
beyond the CI. `_compute_metrics` (`walkforward_backtest.py:118-124`) derives the
actual direction by exact float comparison against the current price, with no flat
band — so `flat` essentially never fires and it is a **2-label** problem where
chance is 50%. `scoring.py` uses `FLAT_TOLERANCE = 0.005` (a fraction, ≡ the
forecaster's `DIRECTION_FLAT_TOLERANCE_PCT = 0.5`), giving **3 labels** where chance
is ~33%. Routing through `score_cohort` adopts the 3-label definition, which is the
one production uses. Only arms measured *after* this task are comparable to one
another; the historical walkforward series is retired, not continued.

Unit trap for the implementer: `target_return_{h}d` is in **percent**, while
`direction_from_return` expects a **fraction**. Divide by 100 or the flat band
silently becomes 0.005%.

### Task 1b: score DA from the estimator production actually serves

`_compute_metrics:120-121` takes the **sign of the p50 regression** as the predicted
direction, and the harness never trains a directional classifier at all. Production
serves the **classifier's** argmax (`forecaster.py:2981-2982`: "the quantile models
only supply the interval"), and the headline DA, the trust warning, and the
opportunity ranking are all judged on it.

A DA bar measured on the median's sign therefore governs a signal that is not
served. Part 1 must fit the directional classifier inside each fold — via the same
`_fit_direction_classifier` production uses, with `sigma_train=None` /
`sigma_val=None` to select the fixed-band labels production selects — and derive
`predicted_direction` from its argmax (`0=down, 1=flat, 2=up`, per
`_direction_classes`).

Report DA from **both** estimators per arm. The two have never been compared on the
same folds, and whether they agree is itself worth knowing. The pre-registered bar
governs the **classifier** figure.

Note the consequence for Part 2: the minimal model leaves the classifier path
untouched, so its served DA is *expected* to move only through the shared
`per_quantile_params[0.5]` seeding and the feature frame. A large move in the
classifier figure is evidence of an unintended change, not of the design working.

### Task 2: establish a clean timing baseline

One local run on the current model, recording:

- training wall time per horizon, per quantile, per ensemble member
- HP-search time per horizon (`N_TRIALS_MAP = {3: 50, 7: 10, 14: 15, 30: 15}`,
  `SKIP_HP_HORIZONS = [3]`)
- predict wall time, phase by phase

Commit the numbers. Every speed claim in Part 2 is measured against this, not
against the inconsistent figures inherited from spec #0.

### Task 3: compute the MDE, then pre-register the bar

From the clustered CI width, compute the minimum detectable effect the gate can
resolve, per horizon. The prior for what to expect: the feature A/B harness noise
floor ran **1.15pp at 3d up to 7.13pp at longer horizons**
(`docs/architecture/model-optimization.md`).

### THE PRE-REGISTERED BAR — measured and committed 2026-08-04, before any arm was run

Measured by `scripts/compute_mde.py --max-items 60 --step-days 120 --horizons 3 7`:
the current design run twice, changing only the LightGBM seed (42 and 7), paired
within `(item_id, forecast_date)`, date-clustered bootstrap.

| Horizon | MDE (pp) | **Bar** | same-design mean diff | dates | paired records |
|---|---|---|---|---|---|
| 3d | **0.54** | paired `ci_lower_pp ≥ −0.54` | +0.152 pp, CI [−0.364, +0.711] | 289 | 16,464 |
| 7d | **0.74** | paired `ci_lower_pp ≥ −0.74` | +0.634 pp, CI [−0.080, +1.402] | 288 | 16,404 |
| 14d | not measured | **deferred** — see the amendment above | — | — | — |
| 30d | not measured | **deferred** | — | — | — |

The bar is on the **classifier** DA figure, not the median's sign.

**This is far tighter than this spec originally predicted.** The earlier text cited
the feature A/B harness noise floor of 1.15–7.13pp and warned that "parity" might
only mean "not worse by 7pp at 30d". The measured MDE is **sub-1pp**, because
pairing within `(item_id, forecast_date)` removes the between-date market variance
that dominates both arms equally — a power gain the original design of this spec did
not anticipate. **The consequence is a harder bar, not an easier one:** the minimal
model must land within 0.54pp at 3d, while simultaneously dropping from 3 ensemble
members to 1. It may well fail, and that would be a real result rather than a
measurement artifact.

Two caveats that must travel with these numbers:

1. **They are the `--step-days 120` figures.** At the full 27-fold configuration the
   MDE would be roughly 1.4× tighter still (~0.38pp / ~0.53pp), so this bar is the
   *more permissive* of the two available. It is not the tightest achievable.
2. **7d's same-design seed effect is the larger one (+0.634pp, CI barely excluding
   zero).** That is independent corroboration of a known defect rather than noise in
   this measurement: `docs/architecture/model-optimization.md` already records 7d q50
   early stopping as "noise-determined", with stopping-round sd 41 on a mean of 47.
   The gate rediscovered it. Any 7d comparison is partly measuring that instability.

The prose bar this replaces read as follows, and is retained for the record:

> For each horizon, the minimal model passes if the lower bound of its clustered
> DA confidence interval is not more than `MDE(horizon)` below the current model's
> DA point estimate on the same folds.

If the MDE at 30d turns out to be 7pp, then the honest claim this spec can support
is "not worse by more than 7pp at 30d" — not "equivalent." Say so in those words in
the results write-up. Do not run an arm before the bar is committed.

### What this gate can and cannot say

Per `backend/AGENTS.md`, `walkforward_backtest.py` does **not** use
`fetch_price_history`. It skips multi-source voting, collapses the archive's 1.37×
duplicate item-days with a plain mean where production serves an outlier-voted
median, and runs a different item universe.

Consequently:

- ✅ **Valid for old-vs-new comparison.** Both arms traverse the identical path, so
  the bias is common and cancels.
- ❌ **Not production DA.** Its absolute numbers must never be quoted as production
  accuracy. Conflating a fresh-model backtest figure with a resolved-forecast figure
  is what produced the retracted 61.5% claim (see the ⚠️ rows in
  `docs/architecture/model-optimization.md`).

### Task 4: the arms

Three of the four are nearly free once the harness exists, and the spec is weaker
without them.

| Arm | Purpose |
|---|---|
| **A** — current 40-model | baseline |
| **B** — minimal 8-model | the candidate |
| **C** — Ridge on the 44 served features | does the tree structure earn anything? |
| **D** — naive (trailing `return_{h}d` as the forecast) | what does zero modelling cost? |

Arm D is not ceremony. In the M5 competition **92.5% of entrants failed to beat a
simple off-the-shelf baseline**
([M5 lessons](https://medium.com/analytics-vidhya/predicting-the-future-with-learnings-from-the-m5-competition-d54e84ca3d0d)).
This repository has never measured its model against a naive arm on the same gate.
If arm D is competitive, that outranks every other finding in this spec.

Arms C and D are measurement-only. Neither is a candidate for serving.

## Part 2 — The minimal model

### Model inventory: 40 → 8

| Kept | Count | Deleted |
|---|---|---|
| Median GBM, `objective=quantile, alpha=0.5` | 4 (one per horizon) | 24 p10/p90 GBMs |
| Directional classifier | 4 (unchanged) | 8 of 12 q50 models (ensemble 3 → 1) |

Config changes, all class constants in `ItemForecaster`:

| Constant | From | To |
|---|---|---|
| `QUANTILES` (`:148`) | `[0.1, 0.5, 0.9]` | `[0.5]` |
| `N_ENSEMBLES` (`:171`) | `3` | `1` |
| `ENSEMBLE_SEEDS` (`:172`) | `[42, 73, 91]` | `[42]` |
| `ENSEMBLE_FEATURE_FRACTIONS` (`:173`) | `[0.6, 0.7, 0.8]` | `[0.7]` |
| `BOOSTING_TYPE_MAP` (`:179`) | `{3: gbdt, 7: gbdt, 14: dart, 30: dart}` | all `gbdt` |

Retaining the `alpha=0.5` quantile objective rather than switching to L2 keeps the
point target identical, removing one variable from the comparison.

`_direction_tree_params` (`:3454`) seeds the classifier from
`per_quantile_params[0.5]`, which still exists. **The directional classifier path
requires no change** — which matters, because it produces the served call.

The DART → GBDT switch is an explicit accuracy hypothesis, not a free win. DART's
dropout is the single most expensive configuration choice in the codebase and has
never been measured against GBDT on a trustworthy gate. It is tested under the same
pre-registered bar as everything else. Note that 14d currently has the **best** DA
of the four horizons (50.8%), so this is the change most likely to cost something.

### The band: CQR → locally-weighted split conformal

**Today.** The p10/p90 models produce a base interval; one global `q̂` per horizon
widens it symmetrically (`:3919-3922`):

```python
q_hat = self.conformal_calibration.get(horizon, 0.0)
if q_hat > 0:
    low_ret_arr = low_ret_arr - q_hat
    high_ret_arr = high_ret_arr + q_hat
```

**After.** With no base interval, the width must be constructed. Split conformal
prediction supplies distribution-free intervals around *any* point model without
training a model per quantile
([Conformalized Quantile Regression](https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf)),
and the normalized variant makes the width item-specific:

```
σᵢ    = clip(price_std_60d / price, SIGMA_FLOOR, SIGMA_CAP)   # per item
sᵢ    = |yᵢ − ŷᵢ| / σᵢ                                        # OOF nonconformity
q̂(h) = ⌈(n+1)(1−α)⌉ / n  quantile of {sᵢ}
low   = mid − q̂(h)·σᵢ
high  = mid + q̂(h)·σᵢ
```

The `/σᵢ` normalization is load-bearing. A single global `q̂` in return space would
give a $5,000 knife and a $1 case the same band width. Normalizing restores the
per-item variation the p10/p90 models were supplying — and `price_std_60d` was
already their top feature in 8 of 12 ensembles. The change replaces a learned
function of volatility with the stated one.

### What the normalization does and does not buy (measured 2026-08-04)

This distinction was missing from the first draft of this spec and it changes what
the band improvement can claim. Measured on n=8000 with a 50/50
calibration/test split and residual spread proportional to σ:

| | marginal coverage | low-vol half | high-vol half | gap |
|---|---|---|---|---|
| with σ-normalization | 0.815 | 0.810 | 0.820 | **0.009** |
| without (`σ = 1`) | 0.809 | 0.949 | 0.669 | **0.279** |

**Marginal coverage lands at nominal either way.** That is split conformal's
distribution-free guarantee doing its job, and it holds for *any* nonconformity
score — normalized or not. So "the band hits 80%" is a property of using conformal
at all, not evidence that the σ-weighting works.

What σ-normalization buys is **conditional** coverage. Unnormalized, the band
over-covers quiet items (94.9%) and under-covers volatile ones (66.9%) while still
averaging 80%. That average would look correct in any aggregate report while being
wrong for every individual item — which is precisely the failure the served band
must avoid, since a user reads one item's band, not the mean of 8,691 of them.

Consequences for the rest of this spec:

- The "Expected outcome" row reading *band coverage: 39–48% → at nominal by
  construction* is true but weaker than it sounds. It should be read as **marginal**
  coverage, and the results document must report **coverage stratified by volatility
  tier** as well, or it will be quoting the one number that cannot fail.
- A test asserting only marginal coverage cannot detect a broken normalization. The
  discriminating assertion is the stratified gap.

`σᵢ` is derived from `price_std_60d` (`:978`) and `price`, both already present on
`latest_rows` in `predict()`. No new feature-engineering pass is added.

`SIGMA_FLOOR` and `SIGMA_CAP` are set from the cross-sectional distribution of
`price_std_60d / price` over the training frame — the 1st and 99th percentiles — and
persisted with the model rather than hardcoded, since they must match the
distribution `q̂` was calibrated on. Their computed values are recorded in the
implementation plan's results.

### Pin the nominal coverage level

The current code disagrees with itself. At `:3088` the comment states
`[p10 - q_hat, p90 + q_hat]` "achieves ~(1-2α) empirical coverage" — 80% at
α=0.10 — while `:3092` and `:3098` set `alpha = 0.10` and log "target
coverage=90%". A `[p10, p90]` base interval is 80% nominal, which requires α=0.20
and the 0.80 quantile of scores; the code uses the 0.90 quantile.

**This spec pins nominal coverage at 80%**, matching what the `[p10, p90]` band has
always represented to the UI, with `α = 0.20`. Measured coverage is 39–48% today,
so the old interval misses either reading badly; the point of pinning is that the
new band's coverage becomes comparable to a stated target rather than to an
ambiguous one.

### Deletions this enables

| Code | Location | Why it goes |
|---|---|---|
| `_fix_quantile_crossing` + its call | `:3463`, `:3890` | A symmetric band around the median cannot cross |
| `STACK_RESIDUALS` / `RESIDUAL_ALPHA` and the Ridge branch | `:261-262`, `:2966-2979` | Already dead code — the flag is `False` |
| `DART_PARAMS` and the DART HP-search branch | `:185-190`, `:182-184` | Unreachable once `BOOSTING_TYPE_MAP` is all `gbdt` |

`self.residual_models` and its save/load handling go with the Ridge branch.

### Untouched

The 44-feature pipeline, `FEATURE_GROUP_ALLOWLIST`, correlation pruning, labels,
multi-source voting, `PREDICT_TAIL_ITEM_DAYS`, prior-forecast blending
(`:3928-3930`), per-tier bias thresholds (`:3937-3944`), the momentum-fallback hook,
and the serving policy.

### Risks

**1. Artifact compatibility — fail loudly.** `load_models` (`:4750-4755`) restores
`conformal_calibration` as one float per horizon. The new scheme needs `q̂` *plus*
the σ definition and its clip bounds. This project's recurring failure mode is a
green pipeline silently running something other than its design
(three recorded instances: the collectors, the CI outage, the drift retrain). So
this needs a **model metadata version bump that raises on an old cache**, not a
`.get()` with a default that would silently serve a band computed two different
ways. The retained Monday model cache is also the rollback, so it must be
recognizably incompatible rather than quietly loadable.

**2. σ for short-history items.** `PREDICT_MIN_HISTORY_DAYS = 14` (`:153`), so items
are eligible with far less than 60 days and `price_std_60d` will be `NaN` or `0`
(the rolling uses `min_periods=1`, so short items get a partial estimate that can be
`0` for a single row). Fallback: the cross-sectional median σ within the item's
price tier, applied before the clip. An implicit `NaN` propagating into
`forecast_low` would reach the UI.

**3. Ensemble 3 → 1 removes variance averaging.** `docs/architecture/model-optimization.md`
estimates ~0.3–0.5pp for this, but that estimate is not measured and sits below the
MDE the gate will report — so the spec cannot resolve it in isolation. It is bundled
into arm B deliberately; if arm B fails the bar, restoring `N_ENSEMBLES = 2` is the
first thing to try before abandoning the design.

## Part 3 — Item-coverage reinvestment

**Merged as its own commit, after Part 2 has been measured.**

`train()` subsamples to a 100,000-row budget: `max_feature_rows: int = 100_000`
(`:2417`). The log reads `Stratified subsample: 133/7,879 items, 95,721/5,891,875
rows (budget 100,000)`. The `max_rows=700_000` passed at
`scripts/forecast_prices.py:209` does not propagate to it.

So the model learns from **133 items** and then serves forecasts for **8,691**.
`docs/retrain-optimization-analysis.md` estimated "~450 items" at this budget; the
measured figure is 133.

Once training costs ~50s instead of ~462s, the budget can rise substantially inside
the original wall-clock. Fewer models over more items is plausibly *more* accurate,
not merely faster.

**Why this must not share a commit with Part 2.** Part 2 is expected to cost a
little accuracy; Part 3 is expected to buy some back. Merged together, a null result
is uninterpretable — it could be two real effects cancelling. Ship Part 2, measure
it against the bar, then ship Part 3 and measure again.

## Testing

| Test | Asserts |
|---|---|
| `test_band_coverage_matches_nominal` | On held-out OOF predictions, empirical coverage of the new band falls in **[75%, 85%]** against the pinned 80% nominal |
| `test_band_width_varies_by_item` | Two items with different σ receive different band widths — guards a silent regression to one global `q̂` |
| `test_sigma_fallback_short_history` | An item with <60 rows of history yields a finite `forecast_low`/`forecast_high`, never `NaN` |
| `test_band_ordering_without_crossing_fix` | `low ≤ mid ≤ high` holds for all items with `_fix_quantile_crossing` deleted |
| `test_stale_artifact_version_raises` | Loading a pre-rewrite model cache raises, rather than defaulting into a mismatched band |
| `test_trained_model_count_is_eight` | A trained forecaster holds exactly 4 quantile + 4 direction models — guards accidental re-expansion of the grid |
| `test_direction_output_unchanged` | Identical features produce identical direction calls and confidences; this path is supposed to be untouched |
| `test_walkforward_uses_clustered_scorer` | `walkforward_backtest` aggregation routes through `score_cohort` and reports `date_coverage_sufficient` |
| `test_walkforward_record_units` | A +0.3% return is labelled `flat` and a +2% return `up` — guards the percent/fraction conversion into `direction_from_return` |
| `test_walkforward_scores_classifier_direction` | The per-record `predicted_direction` comes from the classifier's argmax, mapped `0→down, 1→flat, 2→up`, not from the sign of the median |

Per `AGENTS.md`: `pytest backend/tests/ -q` and `python3 -m py_compile` must pass.
No frontend change is involved — the band keeps the same `forecast_low`/
`forecast_high` contract consumed at `frontend/app/items/[id]/page.tsx:250-252` —
so `npm` checks do not apply.

## Sequencing

Three commits, in order:

1. **Part 1** — rig + timing baseline + committed MDE and acceptance bar. No model
   change. Verified by the current model's CI coming out wider.
2. **Part 2** — the 8-model rewrite. Judged against the pre-registered bar.
3. **Part 3** — `max_feature_rows` propagation and a raised budget, measured
   separately.

**No feature flag.** Two model classes coexisting means two artifact formats, which
is the silent-substitution risk named in Risk 1. `git revert` plus the retained
model cache is the rollback.

## Expected outcome

| | Now | After |
|---|---|---|
| Models | 40 | 8 |
| Training | ~462s (breakdown unverified — see Task 2) | ~35–70s est. |
| Local retrain iteration | 12–16 min | ~2–4 min est. |
| Daily CI predict | ~87s local cold | **≈ unchanged** (data-bound, not model-bound) |
| Band coverage | 39–48% vs an ambiguous target | at 80% by construction |
| Training item coverage | 133 items | raised in Part 3 |
| DA | 46.7–50.8% | within `MDE(horizon)` of baseline, or the design is rejected |

Only the model count and the pinned coverage level are certain. Every timing figure
is an estimate until Task 2 lands, and the DA row is a hypothesis the spec exists to
test.

## Out of scope

- **Regime models.** `SKIP_REGIMES=1` is already the operating mode; the ≤108
  regime models are not currently trained or served. Removing the code path is a
  separate cleanup.
- **Reducing the 44-feature set.** Correlation pruning (`:2470`) runs *before* the
  group allowlist (`:2472-2477`), so which features survive depends on the discarded
  columns. Changing this changes the served feature set and belongs in its own spec.
  (Note: spec #0 cited these as `:2431-2436`; the file has shifted since.)
- **The daily predict path.** Its remaining cost is fetch and feature engineering.
  Caching the voted frame in the Actions cache is the next lever there, deferred by
  spec #0 as an infrastructure decision.
- **Anything from the closed FEATURE/ARCHITECTURE accuracy roadmap.** This is a
  model-*class* question, not a feature question.
