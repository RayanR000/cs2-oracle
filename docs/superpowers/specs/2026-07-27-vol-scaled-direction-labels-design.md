# Vol-scaled direction labels + per-horizon flat/mover sweep

**Date:** 2026-07-27
**Status:** Approved (design)
**Scope:** Directional classifier only (`models/forecaster.py`). Quantile ensemble, feature set, and serving path unchanged.

## Problem

The served direction and confidence come entirely from the 3-class directional
classifier. Its "flat/mover" boundary is a single global constant
`DIRECTION_FLAT_TOLERANCE_PCT = 0.5` applied to every item and every horizon, and
its mover up-weighting is a single global `DIRECTION_MOVER_WEIGHT = 3.0`. Neither
has ever been tuned per-horizon, and the fixed percent threshold mislabels
"movers" across wildly different volatility regimes (a $0.03 penny skin vs. a
$500 knife). Documented consequences: flat-bias on long horizons (43% of 7d
forecasts called flat when the item moved) and real-mover directional accuracy of
only ~55–57%.

A prior classifier HP search (2026-07-27, shelved) failed because it optimized
`multi_logloss` while success is judged on accuracy — an objective mismatch. This
work instead tunes the *label definition* and *mover weighting* against a
fixed accuracy yardstick.

## Goals

- Replace the fixed ±0.5% flat band with a per-item, per-horizon
  volatility-scaled band (#6).
- Make the flat-band multiplier and mover-weight per-horizon, and sweep them (#5b).
- Keep the change cheap: only the classifier refits during the sweep; the
  quantile ensemble is untouched. Inference is precomputed → zero latency impact.

## Non-goals

- No meta-labeling / secondary correctness model (separate future work).
- No live outcome tracking in this pass (acknowledged risk, see Evaluation).
- No change to features, quantile models, regime models, or serving/bias-threshold
  recalibration.

## Design

### 1. Volatility-scaled labels (`_direction_classes`, forecaster.py:2782)

Replace the fixed threshold with a per-item, per-horizon band:

```
threshold_{i,h} = clamp( k_h × σ_daily,i × sqrt(h),  FLOOR, CAP )
```

- `σ_daily,i` — trailing standard deviation of `log_return_1d` over a fixed
  30-trading-day window, computed point-in-time (trailing rows only). Uses returns
  already produced in `_compute_price_features`. No future information enters the
  threshold, so no label leakage (the target return is future; the threshold is not).
- `sqrt(h)` — scales the daily band to the horizon (standard vol-time scaling).
- `k_h` — per-horizon multiplier, the swept knob, replacing the fixed percent.
- `FLOOR` / `CAP` — clamp on the effective threshold (default 0.2% / 15%) so
  degenerate low- or high-vol items (penny floor items) can't produce absurd bands.
- Labels: `up=2` if `return > +threshold`, `down=0` if `return < −threshold`,
  else `flat=1`. Mover up-weighting mechanism unchanged, but "mover" is now
  vol-relative (a row is a mover iff `|return| > threshold_{i,h}`).

### 2. Per-horizon knobs

Convert globals to per-horizon dicts (keyed by horizon, with a scalar default used
as fallback for any missing horizon — backward compatible):

- `DIRECTION_MOVER_WEIGHT` (forecaster.py:172) → `{3, 7, 14, 30}` dict.
- New `DIRECTION_VOL_MULTIPLIER = {3, 7, 14, 30}` (the `k_h`), replacing
  `DIRECTION_FLAT_TOLERANCE_PCT`.
- New constants: `DIRECTION_VOL_WINDOW = 30`, `DIRECTION_THRESHOLD_FLOOR_PCT = 0.2`,
  `DIRECTION_THRESHOLD_CAP_PCT = 15.0`.

### 3. Sweep + honest evaluation

New script `scripts/ab_test_direction_labels.py`, following the existing
`ab_test_*.py` pattern.

- **Grid** (interpretable; only the classifier refits per combo):
  - `k_h ∈ {0.5, 1.0, 1.5, 2.0}`
  - `mover_weight ∈ {1.5, 3.0, 5.0}`
  - 12 combos × 4 horizons.
- **Harness:** the existing 7-fold expanding purge/embargo CV (`_compute_cv_splits`),
  the chosen acceptance gate.
- **Fixed evaluation yardstick (anti-gaming):** every candidate — regardless of its
  training `k_h` — is scored against ONE reference labeling: realized sign with a
  fixed ±0.5% band (the current production `DIRECTION_FLAT_TOLERANCE_PCT`, so
  gated numbers stay comparable to today's), independent of the swept threshold.
  This prevents a candidate from scoring well merely by relabeling more rows as flat.
  - Primary gate: directional accuracy + `edge_vs_best_baseline` on the fixed target.
  - Secondary (reported, not gating): movers-only DA per horizon (accuracy over rows
    where `|realized return| >` the fixed economic band), which targets the
    ~55–57% real-mover figure directly.
- **Output:** per-horizon results table (combo → gated DA, edge, movers-only DA).
  Nothing auto-adopts. Winning `k_h` / `mover_weight` are chosen by inspection and
  written into the per-horizon dicts.

### 4. Serving — unchanged

The classifier still emits down/flat/up plus confidence. Per-tier `bias_thresholds`
recalibration (`update_bias_corrections_from_outcomes`, forecaster.py:316) is
untouched. Only training-time labels and weights change.

## Testing (TDD)

- Vol-band computation: known σ input → known threshold output (incl. `sqrt(h)`
  scaling per horizon).
- Clamp behavior at FLOOR and CAP.
- Penny-item degenerate case (σ ≈ 0 and σ huge) → threshold stays within clamps.
- No-future-leakage check: σ at row t uses only rows ≤ t.
- Per-horizon dict lookup falls back to default when a horizon key is absent.
- Label-distribution sanity on a data sample: not ~100% flat and not ~0% flat.

## Speed

Sweep refits only 4 classifiers × 12 combos × 7 folds; classifiers are fast and the
quantile ensemble is not retrained during the sweep. Adopting winners changes only
constants and re-runs the normal ~10–16 min train. Inference is precomputed →
no serving-latency change.

## Risks / caveats

- **CV optimism (accepted):** the gate is CV, which has historically overstated live
  accuracy by ~10pp. The fixed-yardstick evaluation reduces self-gaming but does not
  substitute for live tracking. Treat adopted numbers as CV estimates, not live.
- Changing the label definition changes what "flat" means downstream; serving bias
  recalibration adapts from observed outcomes, so no code change is needed there, but
  the first post-adoption bias-threshold refresh should be watched.
