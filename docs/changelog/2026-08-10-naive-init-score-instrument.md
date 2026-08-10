# The q50 can be boosted from `-return_1d`, gated off

**Date:** 2026-08-10
**Implements:** **N1** in `docs/research/2026-08-10-next-steps.md` — the top item of Track N.
**Follows:** `docs/changelog/2026-08-10-constant-call-is-hindsight-picked.md`, which established
that the *only* runnable baseline the model measurably loses to is `-return_1d` on rank IC.
**Status: shipped OFF and UNMEASURED.** `NAIVE_INIT_SCORE=1` enables it. No arm has been run.

## Why this one and not the DA work

The published "edge vs constant call" of −4.08 / −9.65 / −16.01pp is measured against a baseline
that picks its direction with hindsight per fold, so it is not a gap. The rank-IC gap against
`-return_1d` is: **−0.0159 / −0.0371 / −0.0433 / −0.0091** at 3/7/14/30d
(`2026-08-10-post-revote-retrain.md`), it uses no future information, and it survived the
2026-08-09 re-vote. It is the one baseline comparison in this repo that is both measured and
legitimate.

`init_score` is the standard boosting-from-an-offset answer: pass `-return_1d` as the starting
score on the training `Dataset`, add it back at predict time, and the booster fits the **residual**
to the naive predictor instead of competing with it.

## What changed

`models/forecaster.py`, ~90 lines, all gated:

- `naive_init_score_enabled()` (env, `NAIVE_INIT_SCORE=1`) governs **training**;
  `_naive_init_score_served()` (artifact, falling back to env) governs **prediction**. Same
  artifact-over-environment rule as `_tier_lead_served`, and it matters more here than for either
  earlier instrument — see the hazard below. Persisted as `naive_init_score` in `meta.json`.
- `_minus_return_1d` builds the offset: `-return_1d`, in **percent**, positionally aligned to the
  frame. `return_{lag}d` and `target_return_{h}d` are both `(a − b) / b * 100`, so no rescaling is
  needed; non-finite values become a **zero** offset ("no baseline view for this row") because a
  NaN `init_score` propagates into every prediction from that row. An **absent** `return_1d`
  raises rather than returning None — a silently missing offset removes the floor while leaving
  the metric that reads it looking normal.
- Five seams, because a missed one is not a crash but a booster fitted against a different
  target than its siblings: the global production `Dataset`, the **regime** `Dataset`s (which
  `predict` prefers over the global model), the CV fold `Dataset`, the Optuna search
  `Dataset` (or HP would be selected for a target the fit does not have), and
  `_holdout_conformal_records` (or the fallback `q_hat` would be fitted around a different mid).
- Every consumer of a q50 prediction adds the offset back: `predict`, the CV fold predictions
  (so `rank_ic`, the fold DA, the PT records and the conformal residuals all describe
  `model + baseline`, which is what is served), the Optuna objective, and
  `_validate_feature_groups`. That last one takes the offset as a parameter and applies it to
  both the base and the shuffled predictions — its metric is a **sign**, and the sign of a
  residual is not the sign of a forecast. The offset is deliberately *not* permuted along with
  the `price_technicals` group that owns `return_1d`: it is part of the predictor, not one of the
  features whose contribution is being measured.
- The **direction classifier is untouched.** It has its own objective and no offset, so the
  served up/flat/down call and every DA that reads it are unaffected by this instrument.

`.github/workflows/model-diagnostics.yml` gains a `naive_init_score` dispatch input beside
`tier_lead` and `cross_sectional_rank`. Default off, so the Sunday scheduled run stays the
control.

## The hazard this is built against

A booster fitted with `init_score` **emits a residual**. Serving it without adding the offset back
publishes a residual as a price forecast — silently, with no shape change in the output to give it
away. That is why the served flag follows the artifact rather than the environment: a warm daily
run that lost the env var would otherwise do exactly that.

`tests/test_naive_init_score.py` (18 tests) pins it from both ends. The load-bearing one is
`test_a_signal_free_model_reproduces_the_naive_baselines_ranking`: over a fold set whose features
are all constant, LightGBM cannot split, so each prediction is `offset + c` and the model's
`rank_ic` must equal `naive_rank_ic` **exactly**. It only can if the offset is applied on both the
training and the prediction side. Its control —
`test_without_the_instrument_the_same_model_has_no_ranking_at_all` — asserts `rank_ic is None`
under the same fixture with the flag off, so the first test cannot pass vacuously. A parametrised
real `train()` run covers the regime branch as well as the global one.

## How to read it

One `model-diagnostics.yml` dispatch per horizon with `naive_init_score=true`, against a control
dispatch on the same commit. **The read is `rank_ic_edge`, and the bar is `>= 0`** — it is
currently negative at all four horizons. Rank IC reproduced exactly across two runs at identical
cached HP and folds (`2026-08-10-served-classifier-scored.md` §1), which is what makes one
dispatch per arm readable.

Three caveats, stated before any number exists:

1. **`tuned_params` were selected against the un-offset target.** The first read reuses cached HP
   on purpose (HP held fixed across arms), and the run now logs a WARNING saying so. Confirm any
   positive with `FORCE_HP_SEARCH=1` before believing its size. Same caveat the tier-lead
   instrument carries.
2. **The floor is empirical, not algebraic.** N1's argument in the action list is that the worst
   case is "the model learns nothing and reproduces the baseline". That holds for a zero-tree
   model; a boosted model can fit its way back below the offset it started from. A negative read
   is a real outcome, not a wiring bug.
3. **On the serving path the offset is partly overwritten downstream.** `predict` recentres the
   mid on the classifier's directional call after all return-space corrections, and that
   recentring can translate the mid by twice its magnitude or pin it to zero
   (`2026-08-10-band-and-confidence-are-miscalibrated.md`). So a rank-IC gain measured in CV on
   `fold_p50` does not transfer one-for-one to the served mid until **F1** lands. The CV read is
   still the right first read — it is the metric the gap was measured on.

## Not done

- No arm run, so there is no result to cite.
- `-return_1d` is used at its raw scale for every horizon. It has the right *ordering* at all
  four (which is all rank IC sees) but a 1-day reversal is not scaled like a 30-day return, so
  the level it hands the booster is off by roughly √h. LightGBM's quantile objective renews leaf
  values as residual quantiles, which absorbs a scale mismatch within a few trees, so this is
  noted rather than fixed. If the read comes back negative at the long horizons only, a fitted
  scalar `beta * -return_1d` — identical ranks for any `beta > 0`, correct level — is the next
  variant to try, not a different feature.
