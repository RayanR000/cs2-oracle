# Quality-spread / cross-wear features — experiment

**Date:** 2026-07-25
**Status:** Features implemented behind `QUALITY_SPREAD=1` (default off). A/B pending.
**Spec:** docs/superpowers/specs/2026-07-25-quality-spread-features-design.md

## What shipped (code)
- `_add_quality_spread_features` on `ItemForecaster`: wear-ladder ratio/z60/chg,
  StatTrak & Souvenir premiums, and `has_*` indicator flags (~13 columns).
- New `quality_spread` feature group in `_feature_group`; `QUALITY_SPREAD=1`
  computes the columns AND appends the group to the allowlist.

## How to run the A/B (from backend/)
Baseline (current production):
    DATABASE_URL="sqlite:///./cs2_market.db" python scripts/forecast_prices.py --train-only
    DATABASE_URL="sqlite:///./cs2_market.db" python scripts/backtest_accuracy.py

Treatment:
    QUALITY_SPREAD=1 DATABASE_URL="sqlite:///./cs2_market.db" python scripts/forecast_prices.py --train-only
    DATABASE_URL="sqlite:///./cs2_market.db" python scripts/backtest_accuracy.py

Record per-horizon CV directional accuracy (training log) and the backtest
table for both arms. The permutation gate is reported automatically by
`_validate_feature_groups` — look for the `quality_spread` group's `passed`
flag and `drop_pp` in the training log.

**Caveat — clear the predict cache between arms.** The engineered-feature
cache used by `predict()` is keyed by date, not by the `QUALITY_SPREAD`
flag. Run each arm via `--train-only` + `backtest_accuracy.py` (as above),
and if you ever invoke `predict()` during the experiment, delete the
engineered cache first so a flag-off cache is not reused under a flag-on
model (which would NaN-fill the quality_spread columns and silently
invalidate the treatment arm).

## Decision rule (pre-registered)
SHIP only if: `quality_spread` is permutation-retained on >=1 short horizon
(3d/7d) AND backtest delta >= +0.5pp there AND no horizon regresses beyond
-1.5pp. On ship: bump lgbm-v3 -> lgbm-v4 and set ENABLE_QUALITY_SPREAD=True.
Otherwise SHELVE: leave code, flag off, and add a null-result row to
docs/research/accuracy-opportunities.md's Reality Check table.

## Results
_(fill after the A/B run)_
