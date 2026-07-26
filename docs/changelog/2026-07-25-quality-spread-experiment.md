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

**Use the dedicated walk-forward ablation harness** (mirrors the existing
`ab_test_*.py` family — the method the project uses for every feature group):

    python scripts/ab_test_quality_spread.py --max-items 1500

It builds features once with `quality_spread` enabled, then walk-forward
evaluates two configs on identical folds — `full` (incl. quality_spread) vs
`no_quality_spread` — and prints the per-horizon directional-accuracy delta
(`full` minus `no_quality_spread`). It also logs per-axis coverage
(`has_wear_siblings` / `has_stattrak_pair` / `has_souvenir_pair`) and selects
the item universe from actual multi-variant groups so the features are
actually exercised.

> **Do NOT use `forecast_prices.py --train-only` + `backtest_accuracy.py` to
> measure this.** `backtest_accuracy.py` scores *stored historical forecasts*
> from the `item_forecasts` table against actuals — it never runs the freshly
> trained model, so both arms would report identical numbers. That flawed
> runbook was the original plan's mistake.

Optional confirmation of the permutation gate on the full production model: run
a single training pass with the flag on and read the `_validate_feature_groups`
log line for the `quality_spread` group's `passed` flag and `drop_pp`:

    QUALITY_SPREAD=1 DATABASE_URL="sqlite:///./cs2_market.db" python scripts/forecast_prices.py --train-only

## Decision rule (pre-registered)
SHIP only if: the `quality_spread` group shows a positive walk-forward delta
on >=1 short horizon (3d/7d) of >= +0.5pp (from `ab_test_quality_spread.py`)
AND no horizon regresses beyond -1.5pp. On ship: bump lgbm-v3 -> lgbm-v4 and
set ENABLE_QUALITY_SPREAD=True.
Otherwise SHELVE: leave code, flag off, and add a null-result row to
docs/research/accuracy-opportunities.md's Reality Check table.

## Results
_(fill after the A/B run)_
