# Scope: the `P(|return| > round-trip cost)` target

> **Status (as of 2026-08-21): ✅ THIS SHIPPED — and it is the project's one surviving signal.**
> The scope below was built and served. `P(|r_h| > c)` has **market-orthogonal, date-stable
> skill** (`changelog/2026-08-15-exceedance-target-phase1-has-market-orthogonal-skill.md`) and it
> is now exposed on the served surface as `move_odds` / `stability_label` volatility tags
> (`changelog/2026-08-20-exceedance-served-signal.md`,
> `changelog/2026-08-20-volatility-stability-tags.md`; PRs #28/#29). Only served-reliability
> maturation remains.
>
> **Two boundaries the scope did not anticipate.** (1) Using `p` to *scale the band*
> (`EXCEEDANCE_SCALE`) was built and then **refuted at serving** — the offline win does not
> transfer (`changelog/2026-08-16-exceedance-band-scale-refuted-at-serving.md`); it survives only
> as a band-**conditional-coverage** improvement
> (`changelog/2026-08-16-exceedance-probability-improves-band-conditional-coverage.md`).
> (2) `p` is **not a trade**. Extending it to h=90/180 — where the realised move finally clears
> the friction bar — produced **no selection skill** (AUC 0.51–0.53 over 3 OOS windows,
> top-decile lift ≤1.0×, unconditional bet negative in every window). The trade hunt is closed at
> every horizon; keep `p` as an annotation on the range, exactly as this doc proposed.


**Date:** 2026-08-15. A build scope, not a decision to ship. Grounds item 1 of
`2026-08-15-directional-accuracy-and-data-inventory.md` in the actual forecaster code
(`backend/models/forecaster.py`, `backend/backtest/friction.py`). Nothing built yet.

## What it is and why

Per item × horizon, predict `p = P(|r_h| > c)` — the probability the h-day move is large enough to
clear the round-trip cost `c`. This is the only target that makes a confident **3%** call
distinguishable from a confident **40%** call: the current q50+conformal range and the 3-class
direction head both treat "small move" and "big move" the same once the sign is fixed. `p` is an
**annotation on the range** ("this band is `p` likely to be worth acting on"), not a new directional
claim — it stays subject to `backend/AGENTS.md` invariant 4 (never quoted alone).

## Two facts that correct the source docs

1. **The label does not exist** (contra "labels already computed"). No `|return|`/exceedance column
   is persisted anywhere. But `target_return_{h}d` is (`forecaster.py:4260`), already winsorized
   ±500% and snapshot/collector-shift voided (`:4294-4396`), so the binary label is a one-line
   derivation off it — the "loss-function change, not a data problem" framing holds.
2. **The cost already exists.** `actionable_threshold(price_tier, venue) = ROUND_TRIP_COST[venue] +
   SPREAD_BY_TIER[price_tier]` (`friction.py:68`). `ROUND_TRIP_COST`: csfloat 0.020, skinport 0.087,
   **steam 0.161**; `SPREAD_BY_TIER`: 0.355 (sub-$1) → 0.052 ($1000+). `DEFAULT_VENUE = csfloat`.

## Label (Phase 1)

In `prepare_targets` (`forecaster.py:4225`), after `target_return_{h}d`:

```python
c = actionable_threshold(price_tier_of(row), VENUE)          # fraction, e.g. 0.072
df[f"target_exceed_{h}d"] = (df[f"target_return_{h}d"].abs() > 100 * c).astype(int)
```

Inherits the existing voiding/winsorization for free. `price_tier_of` = the same price band
`SPREAD_BY_TIER` is keyed on.

## Model head (Phase 1)

Mirror the existing per-horizon classifier exactly — `_fit_direction_classifier`
(`forecaster.py:6415`) → a new `_fit_exceedance_classifier`:
- `objective="binary"`, `metric="binary_logloss"` (vs the direction head's `multiclass`/`multi_logloss`
  at `:6472`).
- Same feature matrix, same served-cohort (≥$1) reweighting (`:6459-6469`), same
  `self.exceedance_models: Dict[int, lgb.Booster]` pattern as `self.direction_models` (`:873`).
- Output: `p_exceed` per item/horizon. No change to the q50 booster or conformal band.

## Serving (Phase 2 — only if Phase 1 clears)

- Add `p_exceed_cost` to the per-horizon forecast dict (`forecaster.py:7879-7885`), computed in the
  same loop as `direction`/`confidence`.
- Add a `p_exceed_cost` column to `ItemForecast` (`database.py:143`) and surface it in
  `api/routes/items.py` (~`:553`). There is currently **no** probability field in the served shape
  or DB schema.

## Validation — the honest part (all new; none of this exists today)

Grep confirms **no probabilistic scorer** anywhere (`brier|log_loss|roc_auc|calibration_curve` →
nothing; the head's only prob loss is LightGBM's training `binary_logloss`, never used to *score*).

- **Offline skill (Phase 1 go/no-go), on the existing CV fold grid:** Brier score + reliability
  curve + AUC vs the base-rate baseline `p = unconditional exceedance rate`. Plus the mandatory
  **shuffled-label placebo** and a **market-relative variant** (demean the return before thresholding)
  — because the most likely failure is that exceedance rides the same market-common factor that
  killed direction (`2026-08-06-market-relative-labels-refuted.md`). If it dies here, it dies cheap.
- **Economic (Phase 2):** selecting items with `p_exceed > τ` and acting must beat not-acting/random
  on **net** return after `c`. This is the real deliverable test, built into `backtest/actionable.py`
  (which already conditions on `|r_hat| > actionable_threshold`).
- **The bind is unchanged.** Served validation needs `MIN_FORECAST_DATES = 20` (`scoring.py:145`)
  distinct durable dates; the panel has ~0–4 per horizon. Offline CV can show skill now; **served**
  validation is calendar-blocked to 2026+, exactly as item 1 states.

## Open product decisions (pick before Phase 1; both change the base rate)

1. **Venue → `c`.** Recommend `csfloat` (matches `DEFAULT_VENUE` and the existing actionable scorer):
   `c ≈ 0.02 + spread`, i.e. ~7% for a $1000 covert, ~37% sub-$1. `steam` (16.1%) makes short-horizon
   exceedance rare → severe class imbalance; keep as a robustness, not the primary.
2. **Two-sided `|r|` vs one-sided `r > c`.** Skins can't be shorted, so only **upside** is tradeable
   — recommend the primary target be one-sided `P(r_h > c)`, with two-sided `|r|` as a diagnostic.

## Effort

- **Phase 1** (label + head + offline Brier/AUC/reliability + placebo + market-relative variant):
  ~0.5 day, buildable now, no prod write. **This is the whole go/no-go.**
- **Phase 2** (serving dict + DB column + API + economic backtest): ~0.5 day, only if Phase 1 clears.
- **Phase 3** (served validation): calendar-blocked; no work until durable dates > 20.

## Most likely outcome, stated up front

Phase 1's market-relative variant is the crux. If exceedance skill vanishes once the common factor is
demeaned — the pattern every cross-sectional arm has followed — this closes for the same reason
direction did, and Phase 2 never runs. The value of the scope is that Phase 1 answers that for ~0.5
day and changes the *deliverable* if it clears, rather than re-slicing sign for the Nth time.

## Phase 1 result (scored 2026-08-15) — GO on the signal, NOT a trade

The market-relative variant **survived** (AUC 0.68→0.68 at h=3, date-stable across all folds,
placebo-clean) — the first market-orthogonal, non-few-episode signal in the project. But the
top-probability decile is net-of-cost **negative** at every horizon: it predicts *magnitude*, not a
profitable *direction*. Highest-value use is a per-item band-width / confidence input to the range
product (a genuinely new, non-`sigma` width lever), or a magnitude-not-direction annotation — not a
buy signal. Full result + caveats:
`docs/changelog/2026-08-15-exceedance-target-phase1-has-market-orthogonal-skill.md`.
</content>
