# The first retrain on the post-exclusion consensus

**Date:** 2026-08-10
**Run:** `31356483719`, `price-forecast.yml` `mode=train-only` on `a0c215e`, 39m32s
(04:45:18 → 05:24:50 UTC). 12 boosters saved to `forecast-models-31356483719`.
**Why now:** `873148b` removed `aggregator_steam_7d/30d/90d` from the consensus vote, so every
label from 2026-03 onward moved. The shipped artifact was trained 2026-08-09 21:55 UTC, before
that commit, and no workflow had run on the merged tree. Until this run, production served a model
fitted to a vote that no longer existed.

**Dispatched as `train-only`, not `full`.** `full` runs `forecast_prices.py` with no flag, which
reaches the age gate at `:376-398`; the artifact was one day old, so it would have skipped training
and re-predicted from the stale model. `FORCE_RETRAIN` is an environment variable the workflow
neither sets nor exposes as a dispatch input. `--train-only` sets `do_train = True` unconditionally.
No forecasts were written by this run — the predict, publish and accuracy steps are all conditioned
on `mode != 'train-only'` — and the next daily chain serves from the new artifact.

## Frame

6,088,782 voted rows from 7,685,823 raw across 12 sources; 916 of 5,536 items and 986,451 rows
survive the ≥$1 median-price floor; feature matrix 985,000 rows × 33 features. The `voted-v6-`
cache key is now populated, so the next run votes warm.

## Result: the re-vote did not move the model-vs-baseline gap

> ⚠️ **Corrected 2026-08-10. The `edge` column against `constant-call` compares to an oracle.**
> `constant_call_baseline` picks its direction with hindsight, per fold — `up` on 4 of 8 folds at
> 30d — so the −9.10 / −11.37 / −15.90 / −20.61pp figures below are not losses to any runnable
> strategy. The runnable fixed call is always-down, whose accuracy is the `realised_down_rate`
> already stored on each fold. **The `rank IC` vs `naive −return_1d` columns are unaffected** — that
> baseline uses no hindsight, and it remains the one legitimate gap this entry identified.
> See `docs/changelog/2026-08-10-constant-call-is-hindsight-picked.md`.

| h | quantile-sign DA | constant-call | edge | PT excess | PT t | verdict | rank IC | naive `−return_1d` | edge |
|---|---|---|---|---|---|---|---|---|---|
| 3d | 39.6% (sd 4.6) | 48.68% | −9.10pp | 2.798pp | **12.14** | skill | 0.1774 | 0.1933 | **−0.0159** |
| 7d | 41.5% (sd 6.1) | 52.88% | −11.37pp | 2.266pp | **7.03** | skill | 0.1267 | 0.1638 | **−0.0371** |
| 14d | 43.0% (sd 12.5) | 58.95% | −15.90pp | 1.037pp | **3.08** | skill | 0.1023 | 0.1456 | **−0.0433** |
| 30d | 48.3% (sd 16.1) | 68.91% | −20.61pp | 1.744pp | **3.35** | skill | 0.0967 | 0.1058 | **−0.0091** |

Three things follow.

**Pesaran–Timmermann returns `skill` at all four horizons.** 30d came in at t=3.35, above the
t=3.06 the 750-round `CV_FIXED_BOOST_ROUNDS` entry was calibrated against, so the round count
still buys what it was chosen to buy. 14d at t=3.077 is now the marginal horizon.

**The model still loses to `−return_1d` on rank IC at every horizon**, by −0.009 to −0.043,
materially unchanged from the pre-re-vote artifact (−0.008 to −0.035). Excluding the
trailing-window feeds cleaned the label's measurement basis without closing the gap to the
one-line baseline. That gap is a modelling problem, not a data-quality one — which is what the
composition-stability result (`f833882`) predicted, and it is now confirmed on a second,
independent axis.

**The q50 sign is informative and inverted.** At 3d it calls direction correctly 39.6% of the
time against a 48.68% constant call, while PT reports t=12.14. A predictor that far below chance
carries real information pointed the wrong way, which is the shape you would expect if the true
signal is the reversal and the median model leans with momentum instead. Consistent with the rank
IC row above, and with `docs/research/2026-08-08-model-review.md` §4.

⚠️ **These DA numbers are the q50 sign, not the served signal.** `CV_DIAGNOSTIC_CLASSIFIER=0`
(`price-forecast.yml:156`), so `classifier_accuracy` and `classifier_accuracy_ge1` are `None` on
every fold and `served_acc` falls back to the quantile sign. **The classifier that actually serves
direction is unscored in CV.** Nothing in this table describes served accuracy.

## Cost: 2306.0s, worse than the 1884.2s reference, and Optuna is why

| Phase | run `31337078991` (2026-08-09) | this run | Δ |
|---|---|---|---|
| Optuna | 63.3s | **692.9s** | **+629.6s** |
| Conformal CV | 836.5s | 514.1s | −322.4s |
| q50 ensemble | 307.2s | 309.3s | +2.1s |
| Direction classifier | 300.6s | 268.2s | −32.4s |
| Remainder (data build, regime models, target prep) | ~376s | ~521s | — |
| **TOTAL training** | **1884.2s** | **2306.0s** | **+421.8s** |

Per-horizon Optuna this run: 0.0 (3d, `SKIP_HP_HORIZONS`) / 198.0 / 68.0 / **426.9**.

**Optuna's cost is bimodal across runs, not stable.** In `31337078991` the pruner killed most
trials at iteration 5–15 and the whole phase cost 63.3s; here it did not, and 30d alone spent
426.9s. Both runs were CI, same runner class, same `N_TRIALS_MAP`. Whether a trial prunes depends
on its intermediate values, so it moves with the data — and the data moved. **Do not budget from a
single run's Optuna figure**, and do not repeat the 2026-08-09 reading of "Optuna is cheap in CI";
it was one draw from a wide distribution.

Conformal CV moved the other way by a comparable amount, so the two largest phases are also the
two least predictable. Per-phase timings swing far more than the ±25% recorded in
`docs/research/2026-08-09-model-and-data-research.md` §4f.

## What this establishes, and what it does not

**Established:** the artifact now matches the shipped consensus; PT clears at all four horizons on
the new labels; the naive-baseline gap survives the re-vote; the `voted-v6-` and
`forecast-models-` caches are populated.

**Not established:**

- **Anything about served accuracy.** The classifier is unscored in CV, and no forecasts were
  written by this run. The first honest served read comes from the backtest once labels mature.
- **Any comparison to a stored A/B.** Every stored verdict predates the re-vote.
- **A cost trend.** Two runs, two very different Optuna draws, and no controlled comparison.
  Remaining levers and their measured basis: `docs/research/2026-08-10-training-cost-levers.md`.
