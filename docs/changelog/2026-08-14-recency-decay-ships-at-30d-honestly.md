# Recency decay ships at 30d under the honest trainer — item 4 (recency is NOT the leak)

**Date:** 2026-08-14
**Item:** 4 of `docs/research/2026-08-13-next-steps.md` — re-read the A/B arms that measured
positive under the old (early-stopping) trainer, since a positive there is the suspect case.
recency-weights 30d (+1.17pp, 6/8 folds, declined on a mechanism argument) was the only genuine
un-re-read positive.

## Result — it reproduced, and it was not the leak

`ab_test_recency_weights.py --horizon 30` on **today's production frame** (built via
`build_training_data(days_back=1460, backfilled_only=True, max_feature_rows=1_200_000,
min_median_price=1.0)` → 982,173 rows / 915 ≥$1 items / 33 allowlisted cols), decay
(`SAMPLE_WEIGHT_HALFLIFE_DAYS=365`) vs flat (control):

```
30d: q50 pinball +1.44% [PASS] | paired -0.108 [-0.192, -0.026] positive [PASS] | DA +0.36pp [PASS] | folds 6/8
VERDICT 30d: SHIP decay
```

All fits ran at fixed 750 rounds (`iters=750/750/750`) — **early stopping OFF, the honest
trainer**. It cleared all three pre-registered gates and reproduced the stored result almost
exactly (6/8 folds, ~+1.2–1.4% pinball). So the stored +1.17pp was **not** an early-stopping
artifact.

**Why the leak heuristic didn't apply here:** the 2026-08-13 leak pays *capacity*, and recency
decay only reweights samples — it adds no columns and no trees. So "positive arm = suspect" was
the wrong prior for this arm; the "declined on a mechanism argument" call looks wrong on the
evidence.

## Caveats — do not over-claim

- **8 folds**, near the fold-count floor; the paired interval excludes zero but the upper bound
  is only −0.026 (see `ab-fold-count-floor` — read a placebo before shipping).
- **30d only.** The stored effect was 30d-specific; 3d/7d/14d were not run here. Whether decay
  generalizes or is a long-horizon phenomenon is open.
- **It is a pinball / band-width gain, not a directional one.** DA is +0.36pp (flat). Decay
  improves the q50 quantile loss — i.e. the served band's calibration — more than the
  directional forecast. Read against the served band (over-coverage work), not DA.
- No shuffled placebo exists in this harness; lower concern than usual since decay adds no
  capacity, but a permuted-weight placebo is the honest next check.

## Harness re-runnability fixes (committed with this)

Two bit-rot defects blocked the run; both are `__new__`-bypass drift against a moved production
path, fixed in `ab_test_recency_weights.py`:

1. `prepare_targets` now records into `self.label_voiding`, unset on the `__new__`-built
   forecaster — seed `fc.label_voiding = {}` as `__init__` would.
2. Production's minimal model persists `tuned_params` for **q0.5 only** (the band is a conformal
   `q_hat*sigma`, not tuned q10/q90 boosters); `load_params` now falls back to the q0.5 params
   for the band quantiles. The ship gate is q50 pinball + DA either way.

## Next (not done here)

Run 3d/7d/14d on the same frame to see if 30d is special; add a permuted-weight placebo;
then decide whether decay is worth shipping (a band-calibration change, so read it through
`replay_serving.py`'s coverage table, not DA).
