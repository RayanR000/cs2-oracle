# The MDE gate failed its own placebo — `paired_mde` now clusters on folds

**Date:** 2026-08-07
**Change:** `backend/backtest/paired_mde.py`, `backend/backtest/walkforward_records.py`,
`backend/scripts/walkforward_backtest.py`, `backend/tests/test_paired_mde.py` (+6 cases).
**Suite:** 1,059 pass (`--ignore=scripts/test_social_signal.py`, which fails to collect on
a missing local `thefuzz` and is unrelated).

This entry records the **evidence and the caveats**. The fix's blast radius — the three
A/B harnesses that had to be updated, and the results it invalidates — is in
`2026-08-07-training-item-universe.md` and is not repeated here.

## The bug

`paired_da_difference` bootstrapped with `forecast_date` as the resampling unit. Dates are
not independent: every date inside one fold's 21-day validation window is scored by the
same fitted model, and at short horizons adjacent dates' forward-return windows overlap.
Resampling ~1,570 dates as if they were independent draws, when they came from 75 fitted
models, understates the variance.

The error direction is the dangerous one. Too-narrow intervals **overstate** significance.

## How it was caught

`scripts/compute_mde.py` compares two arms differing **only in the LightGBM seed** — pure
noise by construction, true difference zero. At h=3, `--max-items 300 --step-days 21`:

| | mean_diff_pp | n_paired | n_dates | n_clusters | 95% CI | mde_pp |
|---|---:|---:|---:|---:|---|---:|
| Date-clustered (before) | −0.1581 | 468,759 | 1,569 | — | **[−0.3099, −0.0140]** | 0.1479 |
| Fold-clustered (after) | −0.0026 | 469,359 | 1,571 | 75 | [−0.0115, +0.0045] | 0.0080 |

The before-row's interval **excludes zero**. A correctly-sized 95% interval rejects a true
null 5% of the time; this one rejected it on the first attempt, on the one comparison whose
answer is known a priori. After the fix the interval straddles zero.

This is the second time a placebo has caught this harness — see the ~7-fold permuted
placebo that read significantly positive. Read the placebo before the treatment, always.

## The fix

- `fold_records` stamps every record with `fold_id`; `walkforward_backtest` passes
  `int(window_end)`, which is unique per fold and identical across arms, so two arms pair
  and cluster on the same folds.
- `paired_da_difference` takes `cluster_key`, defaulting to `"fold_id"`, and reports
  `n_clusters` and `n_dates` side by side so a result is self-describing about its grain.
  `n_dates >> n_clusters` is the signature of the old interval.
- It **raises** on a missing or `None` cluster key rather than falling back to dates.
  Silent fallback would reinstate the bug on any caller that forgot to thread the fold
  through, which is exactly how it went unnoticed.

Pairing is still on `(item_id, forecast_date)`. Only the resampling grain moved; the point
estimate is unchanged by the fix, which `test_pairing_grain_is_unchanged_by_the_cluster_grain`
pins.

## Two things this measurement is not

**1. `0.008pp` is not a measurement floor.** It is the noise of *reseeding*. A seed placebo
perturbs only the RNG, so both arms are near-identical models and there is almost no
contrast to bound. A feature A/B changes the training data, a far larger perturbation. The
operative floor for gating an experiment is the **fold-clustered 2.21–3.69pp** measured on
the breadth A/B at 25–26 folds (`2026-08-07-training-item-universe.md`). Quoting 0.008pp as
"the MDE" would be a worse error than the one this entry fixes.

**2. The run is not reproducible, and that is unexplained.** The identical command run
twice moved `mean_diff_pp` from −0.1581 to −0.0026 — and `n_paired` (468,759 → 469,359) and
`n_dates` (1,569 → 1,571) moved too, so the two runs were not even scoring identical rows.
`mean_diff_pp` does not depend on clustering, so this is real divergence between runs, not
an artefact of the change. Run-to-run movement of ~0.155pp is ~19× the interval the second
run reports. **No cause was identified.** Until one is, treat any tight interval off this
harness with suspicion, and do not read a single run as definitive.

## Consequence for the date-level tables

`2026-08-06-date-level-exogenous-ingest.md` asked whether a date-level feature is
measurable here. Against a ~2–4pp fold-clustered floor, the answer is **no unless the
effect is large**. That entry's original claim — that the per-item floor was "the wrong
denominator entirely" because a date-level column's effective N is its date count — was
wrong on its premise and has been corrected there: the module already resampled coarser
than rows; the defect was the grain, not the denominator.

## Still open

- **The cause of the run-to-run divergence** above. This gates trusting any number here.
- **Re-deriving the CSFloat, ByMykel and breadth CIs** under fold clustering. Tracked in
  `2026-08-07-training-item-universe.md`; not done, and the re-runs cost full sweeps.
- **Horizons 7, 14 and 30 were not measured.** Only h=3 was run, and only as a placebo.
- **Whether folds are themselves independent.** Fold clustering is an improvement, not a
  proof. Training windows are nested and expanding, so adjacent folds share most of their
  training data; if the placebo had still failed, the next unit up would be non-overlapping
  blocks of folds.

## Related

- `docs/changelog/2026-08-07-training-item-universe.md` — the fix's blast radius, the three
  invalidated A/Bs, and the 2.21–3.69pp figure
- `docs/changelog/2026-08-06-date-level-exogenous-ingest.md` — the work that prompted this
- `docs/research/accuracy-opportunities.md` — the Measurement Floor section, whose
  1.15–7.13pp figures come from a different instrument (`ab_test_price_primitives.py`, a
  per-fold power calculation) and are not directly comparable to a bootstrap half-width
