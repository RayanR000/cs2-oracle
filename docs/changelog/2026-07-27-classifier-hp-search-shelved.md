# Directional-classifier HP search — tested & shelved (2026-07-27)

## Hypothesis

The served up/flat/down signal (`direction_models`, added 2026-07-24) is trained by
`_fit_direction_classifier`, which borrows its tree params from the **q50 quantile config**
via `_direction_tree_params`. Those params are optimized for pinball loss on a GOSS median
model, not for 3-class classification. Hypothesis: giving the classifier its **own** Optuna
search (against `multi_logloss`) would improve the served directional accuracy.

## Experiment

`ab_test_classifier_hp.py` — same-fold walk-forward A/B, two arms on identical folds:

- **inherited** — classifier tree params from `_direction_tree_params(q50)` (current behavior).
- **tuned** — params from `_optuna_search_classifier_params` (15-trial `multi_logloss` Optuna).

Config: 100 items, 15 trials, step=120 (13 folds/horizon). Gate metric: plain 3-class accuracy,
matching production `classifier_accuracy` (`forecaster.py:3486`). Search on log-loss, judge on
accuracy (per the pre-registered objective choice).

**Pre-registered gate:** SHIP iff tuned beats inherited in ≥ half the paired folds AND mean
dir-acc delta ≥ +0.2pp. Otherwise keep inheriting.

## Result — KEEP INHERITING (0/4 horizons pass)

| Horizon | inherited dir-acc | tuned dir-acc | paired mean Δ | folds tuned better |
|---|---|---|---|---|
| 3d | 70.45% | 70.30% | −0.15pp | 6/13 |
| 7d | 71.90% | 70.93% | −0.97pp | 4/13 |
| 14d | 72.77% | 72.42% | −0.35pp | 6/13 |
| 30d | 77.11% | 72.62% | **−4.49pp** | 5/13 |

Tuning did not help on any horizon and hurt 30d materially.

## Why

Searched on `multi_logloss` but judged on accuracy, and the two diverged. The log-loss-optimal
configs came out shallow and lightly-regularized (14d/30d: `max_depth=3, min_data_in_leaf=5`),
which minimizes probabilistic loss on the search window but generalizes *worse* on argmax
accuracy over held-out folds — badly so on the noisiest horizon. The inherited q50 params are a
robust, well-regularized config that transfers to direction well.

## Decision

Reverted the feature code (`_optuna_search_classifier_params`, its unit tests, and
`ab_test_classifier_hp.py`) per the repo convention for shelved experiments (cf. the
quality-spread A/B, `b7188e3` / `8783c67`). The classifier continues to inherit q50 params.

## If revisited

- Optimize a **classification metric directly** (mover-weighted accuracy / balanced accuracy)
  rather than `multi_logloss` — the objective mismatch was the likely culprit.
- Or tune `DIRECTION_MOVER_WEIGHT` (the flat-vs-mover trade-off), held fixed here.
