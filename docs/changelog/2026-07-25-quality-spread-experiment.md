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

## Results (2026-07-26)

Walk-forward ablation, `ab_test_quality_spread.py --max-items 1500`. 1,500
items selected from multi-variant groups (of 2,620 eligible); 25 folds;
~777k samples/horizon. Coverage: `has_wear_siblings` 93.4%, `has_stattrak_pair`
23.3%, `has_souvenir_pair` 10.8%. 14 quality_spread features retained after
correlation pruning (141 vs 127 features).

| Horizon | full | no_quality_spread | delta |
|---------|:----:|:-----------------:|:-----:|
| 3d      | 53.78% | 54.56% | **−0.78pp** |
| 7d      | 55.89% | 54.61% | **+1.28pp** |
| 14d     | 56.50% | 55.74% | **+0.76pp** |
| 30d     | 59.03% | 59.66% | **−0.63pp** |

**Mixed:** helps the middle horizons (7d, 14d), slightly hurts the ends
(3d, 30d). Per the pre-registered gate this *passes* (7d +1.28pp ≥ +0.5pp on a
short horizon; worst regression −0.78pp is within the −1.5pp tolerance). The
sign flips across horizons argue against pure capacity inflation.

**Decision: qualified ship — enable quality_spread only where it helps.**
Rather than a global flip, enable the group for 7d/14d and exclude it from 3d/30d
via `HORIZON_EXCLUDED_GROUPS` (same mechanism already used to exclude
cross_sectional at 14d/30d). This captures the +1.28/+0.76pp gains and avoids
the −0.78/−0.63pp regressions. Net expected effect on served accuracy is
positive on 7d/14d, neutral on 3d/30d.
