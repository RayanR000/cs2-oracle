# Directional Prior Correction and a Date Guard on the Bias Fit

Date: 2026-08-03

## Summary

The served up/down/flat call comes from an uncorrected classifier argmax. The only
feedback loop that corrects directional bias from realised outcomes writes to a
path production does not read. This spec closes that gap in the one place it can
be closed without new data: the classifier's inherited up/down class prior. It
also guards the outcome-fitted loop, which currently has no way to see that its
sample is two market days.

Neither change adds training time. Both are post-processing or fitting-time
arithmetic.

## Background

`docs/research/accuracy-opportunities.md` closed the feature and architecture
roadmap on 2026-07-31: six measured feature groups all landed at |effect| < 0.7pp
against an A/B harness floor of 1.15pp (3d) to 7.13pp (30d). **This spec does not
reopen it.** It proposes no new features and no new model architecture. It is
serving-path work, which `docs/changelog/2026-08-01-deterministic-backtest.md`
and its follow-ups establish as in scope.

`docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md` found the
whole reported accuracy series spans **two distinct forecast dates** (30d: one),
which ran in opposite market directions. Effective sample size is 2, not 60,737.
It also recorded, without an explanation, that the model predicts "down" 57–87% of
the time *regardless of date*. That unconditional bias is what this spec targets.

## The defect

Two facts that do not fit together.

**Served direction is an uncorrected argmax** — `models/forecaster.py:3710-3714`:

```python
clf = self.direction_models.get(horizon)
if clf is not None:
    probs = clf.predict(X_horizon)
    dir_class_arr = probs.argmax(axis=1)
    dir_conf_arr = probs.max(axis=1)
```

**The bias corrector targets a different path.**
`update_bias_corrections_from_outcomes` (`:437`) fits per-tier `t_down`/`t_up`
thresholds on `approx_mid_ret` to make the predicted up/down/flat split match the
observed base rate. `predict()` reads those thresholds only inside the
`dir_class_arr is None` branch at `:3784`. All four `clf_{3,7,14,30}d.txt` exist in
`backend/models/saved_models/`, so `dir_class_arr` is never `None` and the
thresholds are dead.

The system has a base-rate corrector. It corrects something that is not served.

Two consequences make it worse than an unused feature:

1. **The classifier inherits its prior and applies it unconditionally.**
   `_fit_direction_classifier` (`:3282`) trains with `objective="multiclass"` and
   no `class_weight` or `is_unbalance` (`:3308`), over a 1460-day window. Whatever
   up/down skew that window carried is learned and then applied at serve time
   irrespective of current market state — which is the shape of the
   "down 57–87% regardless of date" observation.

2. **The call drives the served price, not just the label.**
   `_recenter_on_direction` (`:3756`) maps `down → -|mid|`, `up → +|mid|`,
   `flat → 0`. The quantile model sets *how much*; the classifier sets *which
   way*. So an uncorrected prior moves prices, direction, and confidence together.

Separately, the thresholds the dead loop has been writing are pinned at their
clamp rail. From `backend/models/saved_models/bias_corrections.json`, horizon 30:

| tier | t_down | t_up |
|---|---|---|
| <$1 | -2.89 | 0.00 |
| $1-5 | **-3.00** | -2.92 |
| $5-20 | -0.98 | -0.77 |
| $20-100 | -2.10 | -1.91 |

`-3.00` is the `max(-3.0, min(3.0, ...))` clamp at `:523-524`, and
`BIAS_EWMA_ALPHA = 0.3` carries these forward across runs. This is latent rather
than live — but `:4595` catches a corrupt classifier with a `logger.warning` and
falls through to exactly these numbers, so a single corrupt file promotes a
rail-clamped fit to the served path silently.

## Step 0: a diagnostic gate

The hypothesis behind Component 1 — that the down-bias lives in the training class
prior — is falsifiable in one run. Test it before implementing.

Compute per horizon:

- the **weighted training class prior** `π_c = Σ w[c_train == c] / Σ w`, where
  `c_train` and `w_train` are both already in scope in
  `_fit_direction_classifier` (`:3304-3305`)
- the **serve-time predicted class distribution** from `probs.argmax`

Outcomes:

- **Training prior is up/down skewed** → the hypothesis holds. Proceed to
  Component 1.
- **Training prior is up/down balanced** → the bias originates in serve-time
  feature state, not the prior. **Stop.** Ship Component 2 on its own and write
  the diagnostic up as a changelog entry. Do not build a correction for a bias
  that is not in the prior.

This ordering is deliberate: the diagnostic costs one run, and the alternative is
adding a branch to the serving path on an unverified assumption.

## Component 1: up/down prior correction

Applied to `probs` before `argmax`. No training cost — the prior is a weighted
count taken at fitting time, and serving adds one vector operation.

### Only one of the two asymmetries is a bug

The classifier's class distribution is skewed along two independent axes, and they
must be treated differently:

| axis | origin | verdict |
|---|---|---|
| up vs down | inherited from the training window | **the defect** |
| mover vs flat | `DIRECTION_MOVER_WEIGHT_MAP = {3: 3.0, 7: 3.0, 14: 3.0, 30: 3.0}` (`:232`), applied via `_direction_sample_weights` (`:3249`) | **deliberate** |

Flat suppression is a design choice: movers are up-weighted 3× on purpose.
Restoring the "natural" 3-class prior would silently undo it.

So the correction is **up/down symmetric only**. `p_flat` is untouched.

### The rule

A **zero-sum** logit offset per horizon, applied to up and down only:

```
b_h = τ · log(π_down / π_up)      τ = 1.0

s_down = log p_down − b_h / 2
s_flat = log p_flat                  (anchor, unshifted)
s_up   = log p_up   + b_h / 2

direction = argmax(s_down, s_flat, s_up)
```

The sign follows from the defect: when down is over-represented in training,
`π_down > π_up`, so `b_h > 0`, which suppresses down and boosts up. That is the
intended direction of the fix.

The offset is **zero-sum by construction** (`−b/2`, `+b/2`) rather than the
textbook prior division (`log p_c − log π_c` for both classes). Prior division
would boost up *and* down against flat, because both priors are below 1, silently
changing the flat share and undoing the mover weighting. A zero-sum shift
rebalances up against down while leaving the up+down mass relative to flat intact.
This is what the "`p_flat` share unchanged" test checks.

One parameter per horizon, computed at train time and persisted with the existing
model artifacts.

**Confidence reads the original `probs`, not the adjusted scores.** `dir_conf_arr`
stays `probs.max(axis=1)` on the unadjusted vector, so the correction cannot move
`DIRECTION_CONFIDENCE_HIGH` classifications or anything downstream of them. Since
argmax is invariant to renormalization and confidence does not read the adjusted
scores, no renormalization step is needed at all.

`τ = 1.0` — full correction. Fixed, not swept. A per-horizon τ map would follow
the existing `DIRECTION_VOL_MULTIPLIER_MAP` pattern, but sweeping 4 horizons × N
values against a 1.15pp noise floor is how noise gets read as signal, which is
precisely what closed the accuracy roadmap. One constant yields a clean result in
either direction.

### Null safety

A balanced training prior gives `b_h = 0` and byte-identical predictions. This is
the primary regression test, and it means the change cannot alter serving output
except where a measured prior skew exists.

## Component 2: guard the outcome fit, reset the rails

### The guard

`update_bias_corrections_from_outcomes` gets a date-coverage guard. Its existing
guards are row counts only — `n < 20` (`:496`) and `MIN_THRESHOLD_SAMPLE = 100`
(`:489`) — and no row count can distinguish 11,000 independent rows from 11,000
rows on two days.

Reuse `MIN_FORECAST_DATES` from `backend/backtest/scoring.py:38` rather than
introducing a second threshold, so the value reporting is gated on and the value
fitting is gated on cannot drift apart. `forecast_outcomes.forecast_date` already
exists (`database.py:277`), so **no migration is required**.

Below coverage: refuse to fit, log at `WARNING` naming the date count, leave
thresholds at their defaults. This mirrors what `_score_groups` already does for
the headline — the same evidence standard applied to the fit, not just the report.

### The reset

Reset the values currently pinned at the clamp rail to
`±DIRECTION_FLAT_TOLERANCE_PCT` defaults, and clear the corresponding
`ewma_state` counters so a future well-covered fit starts cold rather than
EWMA-blending toward a rail.

A guard alone is insufficient: it prevents the *next* bad fit while leaving the
existing one in the file, and `:4595`'s corrupt-classifier fallthrough is what
makes that file reachable.

## Explicitly out of scope

**Fitting the correction from realised outcomes** instead of the training prior.
This would be the better estimator — it calibrates against what actually happened
rather than what the training window contained — but it needs the classifier
margin for each historical forecast, and `item_forecasts` stores only `direction`
(the argmax label) and a coarse `confidence` string (`database.py:143-158`). The
margin is not recoverable. It would require either a new column, which starts a
fresh 20-date clock from merge, or offline re-inference over archived features.
Worth a later spec. Component 1 does not depend on it.

**Any change to `dir_conf_arr`.** Confidence stays `probs.max(axis=1)`. Changing
it would entangle this work with the confidence-calibration and coverage
machinery.

**Repairs to the feature A/B harness.** Out of bounds per
`docs/research/accuracy-opportunities.md`.

## Measurement

`backend/scripts/ab_test_direction_labels.py` is the right harness: 26
walk-forward folds over the archive, so it does not inherit the two-date problem
that makes the live series unreportable.

Per the standing rule from the harness noise-floor work, **compute the MDE before
running it**: run two arms on one horizon, take the paired per-fold delta sd, and
derive `MDE = 2.8 · sd / sqrt(n_folds)`. If the expected effect is below MDE, say
so and skip the run rather than producing an undecidable result.

A systematic prior bias is the one class of effect in this project plausibly large
enough to clear 1.15pp, unlike the ~0.3pp feature effects. That is an argument for
measuring, not a prediction — it gets derived, not assumed.

Note also that `walkforward_backtest.py` scores a different price consensus than
production (plain mean over duplicate item-days vs production's outlier-voted
median; see `backend/AGENTS.md`), so its absolute accuracy is not comparable to
production. The A/B *delta* between two arms is what this spec reads.

## Testing

Component 1:

- a balanced training prior produces byte-identical predictions (`b_h == 0`)
- a down-skewed prior moves calls from down toward up, not the reverse — the sign
  test, and the one most likely to be inverted in implementation
- `p_flat` share is unchanged by the correction — the mover weighting survives
- prior computation weights by `w_train`, not raw class counts
- `dir_conf_arr` is identical with and without the correction
- a zero probability in any class does not produce `-inf` or `nan` (epsilon floor
  before `log`)
- a degenerate prior with `π_up == 0` or `π_down == 0` falls back to uncorrected
  rather than producing an infinite offset
- the offset is deterministic across runs
- a horizon with no stored prior serves uncorrected rather than raising

Component 2:

- a cohort of 11,000 rows on 2 dates does not fit, regardless of row count
- a cohort at exactly `MIN_FORECAST_DATES` does fit
- the fit and the headline read the same constant
- refusing to fit leaves thresholds at defaults, not at previous values
- reset clears `ewma_state` alongside the thresholds
- rows with a null `forecast_date` never count toward coverage

## Files

- `backend/models/forecaster.py` — prior computation in
  `_fit_direction_classifier`; correction at the `predict()` argmax; date guard in
  `update_bias_corrections_from_outcomes`; persistence in
  `_save_bias_corrections` / `_load_bias_corrections`
- `backend/models/saved_models/bias_corrections.json` — rail-clamped values reset
- `backend/tests/` — new test module per component

## Related

- `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md` — the
  down-bias observation this explains, and `MIN_FORECAST_DATES`
- `docs/changelog/2026-08-01-deterministic-backtest.md` — the serving-path work
  this continues
- `docs/research/accuracy-opportunities.md` — the closed roadmap this stays clear
  of
