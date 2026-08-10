# Feature Contribution A/B Test — Cross-Sectional & Event Features by Horizon

> **Read this before citing anything below (added 2026-08-09).**
>
> **Its founding number has never been re-derived.** The `+3.5pp @30d` for removing
> cross-sectional features is the entire basis of
> `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`. The harness that produced it now
> carries its own warning (`backend/scripts/ab_test_feature_contribution.py:18-32`):
> it "had **no purge gap at all**" before 2026-08-08, so "**Every delta this harness
> printed before 2026-08-08 is un-embargoed.**" Re-derivation is tracked as **C4** in
> `docs/research/2026-08-09-next-steps.md:541`. The number is neither refuted nor
> validated — it is unmeasured.
>
> ⚠️ **Do not confuse it with the other `+3.50pp @30d` in this repo.** That one
> belongs to `TRAIN_MIN_MEDIAN_PRICE` and *has* been re-derived, to **+1.642pp
> [−0.809, +4.505], null** (`docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`).
> Two different results; only the price-floor one has a re-derivation.
>
> **`HORIZON_EXCLUDED_GROUPS` is now a no-op.** It still holds this doc's numbers
> (`backend/models/forecaster.py:342-345`), but `FEATURE_GROUP_ALLOWLIST` (`:353`)
> already removes `cross_sectional` and `events` at *every* horizon —
> `docs/architecture/model.md:142-144` states this. The "Immediate (implement in
> forecaster.py)" recommendations below are subsumed.
>
> **The cohort is the penny cohort and the benchmark is the refuted one.** Item
> selection is `ORDER BY row_count DESC LIMIT max_items` — longest history, no price
> floor (`ab_test_feature_contribution.py:108-118`). That exact selection was
> measured on a sibling harness
> (`docs/changelog/2026-08-06-volume-ab-and-harness-defects.md:74-97`) at **15 of 200
> items ≥$1**, median item price **$0.059**, **41.0% of 3d forward returns exactly
> zero**, and **31.27%** of the scored rows being free `sign(0) == sign(0)` hits. The
> query also filters to `source IS NULL OR source = 'STEAMCOMMUNITY'` (`:100`), so the
> study reads pre-2026 backfill rows only. That is why its 65.8–72.2% DA sits 15–20pp
> above production's ≥$1 CV.
>
> **"+X pp vs 50%" is the Pesaran–Timmermann null, not a skill measure.** A constant
> always-down call beats the model on every stored date, so §3 "Accuracy Ceiling Is
> Not Lower at Long Horizons" and "30d is the most predictable horizon" do not follow
> from these numbers. See `docs/research/2026-08-08-model-review.md` §2.

**Date:** 2026-07-19  
**Context:** Investigation into whether market-level cross-sectional features and event relevance weighting actually help 14d/30d horizons, given the hypothesis that skin-market prices past ~2 weeks are dominated by unforecastable one-off events.

---

## Methodology

Walk-forward ablation study on 100 backfilled items from the Parquet archive (entire date range, ~2000 days). Features engineered once, then evaluated with three column subsets on identical train/val splits:

| Config | Feature Set |
|--------|-------------|
| **Full** (baseline) | All 115 features after correlation pruning |
| **No Cross-Sec** | Remove `market_*`, `market_regime_*`, `item_return_vs_market_*`, `item_volume_vs_market_*` (18 removed → 97 features) |
| **No Events** | Remove `event_decay_*`, `events_next_*`, `event_density_*` (23 removed → 92 features) |

52K samples per horizon, 25 walk-forward folds. Positive Δ = removing the feature *improved* accuracy (the feature was adding noise).

---

## Results

| Horizon | Full | No Cross-Sec | Δ | No Events | Δ |
|---------|------|-------------|----|-----------|----|
| **3d**  | 65.8% | 65.6% | **−0.2pp** | 65.5% | **−0.3pp** |
| **7d**  | 65.7% | 65.7% | **0.0pp**  | 66.3% | **+0.6pp** |
| **14d** | 68.4% | 69.3% | **+0.9pp** | 66.4% | **−2.0pp** |
| **30d** | 68.7% | 72.2% | **+3.5pp** | 71.7% | **+3.0pp** |

### Directional Accuracy (%)

```
        3d        7d        14d       30d
Full    ████████████████████████████████████ 65.8%
No CS   ████████████████████████████████████ 65.6%  −0.2
No Ev   ████████████████████████████████████ 65.5%  −0.3

Full    ████████████████████████████████████ 65.7%
No CS   ████████████████████████████████████ 65.7%   0.0
No Ev   ████████████████████████████████████ 66.3%  +0.6

Full    ████████████████████████████████████ 68.4%
No CS   █████████████████████████████████████ 69.3%  +0.9
No Ev   ████████████████████████████████████ 66.4%  −2.0

Full    ████████████████████████████████████ 68.7%
No CS   ██████████████████████████████████████ 72.2%  +3.5
No Ev   ██████████████████████████████████████ 71.7%  +3.0
```

---

## Interpretation

### 1. Cross-Sectional Features: Harmful at 14d/30d

| Horizon | Δ (removing CS) | Verdict |
|---------|-----------------|---------|
| 3d  | −0.2pp | Tiny positive contribution |
| 7d  | 0.0pp  | Neutral |
| 14d | +0.9pp | **Actively harmful** |
| 30d | +3.5pp | **Strongly harmful** |

**Why?** Market-level features (`market_return_{1,7,14,30}d`, `market_regime_*`) capture how an item moved relative to the market *yesterday*. At 3d, this recent co-movement partially persists. At 30d, the noise from a single day's market regime has no causal relationship with an item's return a month later. The model learns spurious correlations.

The existing `_validate_feature_groups` permutation test doesn't catch this because it tests within the validation fold, where cross-sectional features still show recent co-movement that passes the permutation test (p < 0.05, drop > 0.5pp). But this signal doesn't generalize across the full walk-forward.

### 2. Event Features: Good at 14d, Bad at 30d

| Horizon | Δ (removing events) | Verdict |
|---------|---------------------|---------|
| 3d  | −0.3pp | Tiny positive contribution |
| 7d  | +0.6pp | Marginal (already neutral/slightly harmful) |
| 14d | −2.0pp | **Genuinely helpful** |
| 30d | +3.0pp | **Strongly harmful** |

**Why?** Event features use exponential decay (`exp(-days_since / tau)`) with tau from 7–60 days. At 14d (tau/2 to 2×tau), the decay signal is in its informative middle range for most event types — still strong enough to matter, past enough to separate signal from noise. At 30d, the decay has flattened to near-zero for most event types (tau=7), making the features nearly constant. The few events with long tau (majors, tau=60) still have some decay at 30d, but the remaining density/next-30d features become stale.

The 14d exception partially disproves the hypothesis: event features *can* help past 2 weeks, just not at 30d.

### 3. Accuracy Ceiling Is Not Lower at Long Horizons

Baseline accuracy is actually *higher* at 14d/30d (~68.4–68.7%) than at 3d/7d (~65.7–65.8%). But the optimal feature set differs:

| Horizon | Optimal Config | Accuracy | Improvement |
|---------|---------------|----------|-------------|
| 3d  | Full or No CS | 65.8% | +15.8pp vs 50% |
| 7d  | No Events     | 66.3% | +16.3pp vs 50% |
| 14d | No CS only    | 69.3% | +19.3pp vs 50% |
| 30d | No CS only — the combined **No CS + No Ev** arm was **never run** | 72.2% | ~~+22.2pp vs 50%~~ |

**Correction (2026-08-09):** the 30d row originally read `No CS + No Ev`, but 72.2% is
the **No-CS-only** number from the Results table above. No combined arm exists in the raw
data. Recommendation 2 below assumes the two removals stack for "~+6pp total", which would
be ~74.7% — that figure was never produced and is not supplied here. The "vs 50%" column
is the Pesaran–Timmermann null, not skill; see the banner.

30d is the most predictable horizon when using the right feature set — the opposite of the original hypothesis.

---

## Recommendations

### Immediate (implement in forecaster.py)
1. **Prune cross-sectional features at 14d/30d** — add to `horizon_feature_cols` exclusion. Expected gain: +0.9pp at 14d, +3.5pp at 30d.
2. **Prune event features at 30d** — expected gain: +3.0pp at 30d (stacks with CS removal for ~+6pp total).

### Structural
3. **Re-engineer cross-sectional features for long horizons:** Instead of `market_return_{1,7,14,30}d` (current-date returns against market), use lagged market returns relative to the *forecast date* (e.g., what was the market regime 30 days ago, predicting the next 30 days). A separate set of cross-sectional features with longer lookback for long horizons.
4. **Reconsider event decay constants by horizon:** At 7d, events already degrade accuracy — the decay tau of 7 (updates) may be too long. At 30d, only tau=60 majors hold signal; shorter-tau events should be zeroed out.
5. **Fix `_validate_feature_groups` for horizon-specific pruning:** The permutation test over-valids groups that have signal in-fold but don't generalize. Consider cross-validation of the permutation step, or use the walk-forward evaluation as the pruning validation signal.

### Periodic
6. **Run `ab_test_feature_contribution.py` monthly** as a diagnostic — feature group contributions can shift as market regimes change.

---

## Raw Data

```json
{
  "3d": {
    "full": {"dir_acc": 65.79, "mae": 0.1278, "int_cov": 86.61, "n": 52499},
    "no_cross_sectional": {"dir_acc": 65.62, "mae": 0.1279, "int_cov": 86.64, "n": 52499},
    "no_events": {"dir_acc": 65.50, "mae": 0.1272, "int_cov": 86.65, "n": 52499}
  },
  "7d": {
    "full": {"dir_acc": 65.65, "mae": 0.1594, "int_cov": 85.85, "n": 52499},
    "no_cross_sectional": {"dir_acc": 65.66, "mae": 0.1595, "int_cov": 85.73, "n": 52499},
    "no_events": {"dir_acc": 66.28, "mae": 0.1590, "int_cov": 85.65, "n": 52499}
  },
  "14d": {
    "full": {"dir_acc": 68.41, "mae": 0.1586, "int_cov": 87.45, "n": 52499},
    "no_cross_sectional": {"dir_acc": 69.29, "mae": 0.1582, "int_cov": 87.04, "n": 52499},
    "no_events": {"dir_acc": 66.44, "mae": 0.1588, "int_cov": 87.22, "n": 52499}
  },
  "30d": {
    "full": {"dir_acc": 68.71, "mae": 0.2480, "int_cov": 83.97, "n": 52498},
    "no_cross_sectional": {"dir_acc": 72.25, "mae": 0.2477, "int_cov": 84.42, "n": 52498},
    "no_events": {"dir_acc": 71.71, "mae": 0.2419, "int_cov": 83.93, "n": 52498}
  }
}
```
