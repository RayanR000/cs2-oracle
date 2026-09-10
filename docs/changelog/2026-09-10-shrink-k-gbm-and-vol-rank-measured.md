# 2026-09-10: SHRINK_K_GBM and VOLATILITY_RANK_GBM both fail their paired replay

## SHRINK_K_GBM: NULL at all four horizons

`scripts/shrink_k_vol_rank_ab.py` — paired per-fold deltas on log-width at matched 80%
coverage, 25–26 walk-forward folds, held-out items, production's own
`_build_climatology_table_adaptive` vs flat `CLIMATOLOGY_SHRINK_K=320`.

| h   | delta      | 95% CI              | narrower  | worst fold |
|-----|------------|---------------------|-----------|------------|
| 3d  | −0.0035    | [−0.008, +0.001]    | 16/26     | 1.035×     |
| 7d  | −0.0007    | [−0.005, +0.003]    | 15/26     | 1.022×     |
| 14d | −0.0005    | [−0.005, +0.004]    | 12/25     | 1.023×     |
| 30d | +0.0021    | [−0.004, +0.008]    | 11/25     | 1.065×     |

Every CI spans zero. Best case is −0.3% at h=3; at h=30 it is slightly *wider*.

**Stability analysis** (`scripts/shrink_k_stability.py`): the GBM's K predictions are
stable (median CV 0.06–0.10, pairwise Spearman ρ = 0.73–0.80 across 25 folds, 150
held-out items), so the null is not measurement noise — the GBM has learned a stable
function that consistently predicts K ≈ 900 (vs global 320), i.e. "pool harder". The
per-item variation adds nothing beyond the already-correct global constant.

This is the **5th GBM** on this panel to fail to beat a featureless constant on band
width (after the modelled sigma, climatology-beats-gbm, gbm-decorative-on-clean-label,
and the magnitude booster). The modelled-sigma family is dead.

`SHRINK_K_GBM: "0"` was already in `price-forecast.yml` since earlier today (gated off
pending this measurement). The gate is now confirmed correct.

## VOLATILITY_RANK_GBM: HARMFUL at all four horizons

| h   | delta      | 95% CI              | narrower | worst fold |
|-----|------------|---------------------|----------|------------|
| 3d  | +0.112     | [+0.099, +0.125]*   | 0/26     | 1.186×     |
| 7d  | +0.097     | [+0.080, +0.114]*   | 0/26     | 1.285×     |
| 14d | +0.062     | [+0.043, +0.080]*   | 0/25     | 1.235×     |
| 30d | +0.100     | [+0.071, +0.129]*   | 0/25     | 1.344×     |

0 winning folds at any horizon. The earlier commit-message claim of "6–8% narrower" was
wrong — the vol-rank multiplier is 6–12% **wider**. The discrepancy is likely the
difference between a single split and a walk-forward: the old claim had no harness, no
paired interval, and no changelog.

`VOLATILITY_RANK_GBM` was already off by default and not listed in price-forecast.yml.
Leave it off permanently.

## Label ceiling on 2025 panel (new)

`scripts/label_ceiling_2025.py` — split-half reliability of the voted label using the
newly-available 2025 multi-source panel (source=NULL vs source=buff_iflow, 360 days,
944 items at ≥$1).

| h   | R²(2025) | IC(2025) | R²(2026) | IC(2026) | Change |
|-----|----------|----------|----------|----------|--------|
| 1   | 0.37     | 0.61     | 0.28     | 0.53     | +32%   |
| 3   | 0.67     | 0.82     | 0.36     | 0.60     | +85%   |
| 7   | 0.79     | 0.89     | 0.43     | 0.66     | +84%   |
| 14  | 0.87     | 0.93     | 0.52     | 0.72     | +67%   |
| 30  | 0.92     | 0.96     | 0.64     | 0.80     | +44%   |

The 2025 clean-quote era has dramatically higher label reliability (32–85% improvement)
despite having only 2 sources vs 7. The 2025 era has 0.7–1.5% frozen quotes vs 18–33%
in 2026, which dominates the source count. The max achievable IC is 0.96 at h=30,
confirming large unexploited headroom.
