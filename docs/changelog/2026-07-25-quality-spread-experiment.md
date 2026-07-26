# Quality-spread / cross-wear features — experiment

**Date:** 2026-07-25
**Status:** SHELVED (2026-07-26). Implemented behind `QUALITY_SPREAD=1` (default off); A/B ran, net-flat, feature disabled. See Results.
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
(3d, 30d). Averaged across horizons the effect is **~+0.16pp — flat**. It moves
accuracy between horizons rather than improving the model overall. The sign
flips argue against pure capacity inflation (which would show uniform small
positive deltas), so there is *some* real horizon-dependent signal — but small.

**Training-speed cost** (measured, 1,500-item / 5.5M-row frame): the feature
group adds **+77%** to the feature-engineering phase (66.4s → 117.8s) from the
several full-frame groupby + rolling-window passes. This is paid on every
training run.

## Decision: SHELVE (2026-07-26)

`ENABLE_QUALITY_SPREAD` stays **False** (default). The code remains in place,
disabled, behind the flag; the A/B harness (`scripts/ab_test_quality_spread.py`)
is kept for reproducibility.

Rationale: a **net-flat** accuracy result (+0.16pp mean) does not justify **+77%**
feature-build time plus the permanent complexity (14 features, item-metadata
dependency, heavy rolling passes). The ~1pp gains on 7d/14d sit in the magnitude
band that this project's calibration history repeatedly shows evaporating under
permutation testing, and no permutation confirmation was run — so the burden of
proof for adding permanent complexity is not met. If revisited, gate on a
permutation test on the full production model before enabling, and consider a
7d/14d-only enable via `HORIZON_EXCLUDED_GROUPS` (3d/30d excluded).
