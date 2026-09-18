# Per-horizon training objective and LambdaRank ranking head

**Date:** 2026-09-17

## Problem

The q50 centre model trains with quantile loss (α=0.5, ≡ MAE) but is evaluated
on within-date rank IC (Spearman correlation). This objective/metric misalignment
leaves IC at ~0.03 against a measured label noise ceiling of 0.53–0.80.

## Experiment

`scripts/objective_comparison_ab.py` compared five LightGBM objectives on the
production frame (977K rows, 897 items, 28 features) using production's temporal
split and evaluation. Results on the validation set:

| Arm        |  3d IC |  7d IC | 14d IC | 30d IC |
|------------|--------|--------|--------|--------|
| quantile   | 0.1688 | 0.1206 | 0.1331 | 0.1452 |
| huber      | 0.1655 | 0.1330 | 0.0793 | 0.1609 |
| mse        | 0.1175 | 0.0890 | 0.0668 | 0.1863 |
| lambdarank | 0.2024 | 0.1674 | 0.1396 | 0.1283 |
| binary     | 0.1556 | 0.1074 | 0.1252 | 0.1284 |

Paired delta vs quantile (significant where CI excludes zero):

- LambdaRank at 7d: **+0.047** [+0.011, +0.082] — significant
- MSE at 30d: **+0.041** [+0.020, +0.062] — significant
- LambdaRank at 3d: +0.034 [-0.003, +0.071] — nearly significant

LambdaRank outputs ranking scores (not return predictions), so it cannot
directly replace the q50 for the conformal band. MSE is a drop-in.

## Changes

### 1. Per-horizon centre objective (`CENTRE_OBJECTIVE`)

Module-level env var, default `"30:regression"`. Format: comma-separated
`horizon:objective` pairs. Unmentioned horizons keep `quantile`.

The MSE (regression) objective at 30d is a drop-in replacement for the band
path — same output semantics (percentage return prediction), same conformal
calibration. No predict-path changes needed.

Wired into three places: `_train_horizon_inline` base params, `_optuna_search_params`,
and the HP-reuse path. Recorded in `meta.json` as `centre_objective`.

### 2. LambdaRank ranking head (`RANKING_HEAD=1`)

A separate LambdaRank model per horizon, trained alongside the q50 model.
Converts returns to per-date decile relevance labels (0–9) and trains with
`lambdarank` objective (pairwise ranking). Outputs continuous ranking scores
served as `rank_score` in the forecast output.

Architecture matches the exceedance/anomaly heads: trained in
`_train_horizon_inline`, saved to `rank_{h}d.txt`, loaded in `load_models`,
scored in `predict`. Gated off by default.

## What this does NOT change

- The conformal band (still uses q50/regression return predictions)
- The data pipeline or label construction
- The feature set (price_technicals only)
- Any existing serving behaviour (ranking head is off by default, MSE at 30d is
  the only change to default training)
