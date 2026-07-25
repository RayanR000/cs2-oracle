# Quality-Spread / Cross-Wear Features — Design

**Date:** 2026-07-25
**Status:** Design approved, pending spec review
**Roadmap item:** `docs/research/accuracy-opportunities.md` §Remaining #1 — "Quality spread / cross-wear features"

## Goal

Test whether an item's directional forecast improves when the model can see how
that item's price sits *relative to its sibling variants* — the same skin in
other wears, StatTrak vs non-StatTrak, and Souvenir vs normal.

This is an **experiment-first** effort. Features are built behind a default-off
flag, validated with an A/B run plus a permutation test, and only wired into
production if they clear a pre-registered lift threshold. A measured 0pp is an
acceptable, valuable outcome and is written up either way.

## Motivation & the calibration lesson

`accuracy-opportunities.md` estimates this at 1–2pp but flags it as the only
remaining *genuinely novel* signal. The dominant prior from the supply-depth
finding (2026-07-16) is decisive for the design: the **level** of a spread was a
near-static liquidity characteristic that added 0pp, while only the
**change/velocity** variant was mechanistically predictive. This design mirrors
that split — level features are included only as controls so the permutation
test can confirm whether the signal (if any) lives in deviations/changes rather
than static levels.

## Feasibility (measured 2026-07-25)

Ran `parse_item_name` over the 8,691 items in `cs2_market.db`:

- 2,771 items parse into weapon + skin + wear.
- 693 wear-ladder groups with ≥2 wears, covering 2,522 items (~91% of skins),
  averaging 3.64 wears per group.
- 356 StatTrak pairs and 141 Souvenir pairs with both variants present.

Features are live for ~2,500 items and neutral for the rest (cases, stickers,
agents, charms — no wear ladder).

## Architecture

New method `_add_quality_spread_features(df)` on `PriceForecaster`, called in:

- `build_training_data` — after `_add_item_identity_features` (which supplies
  the parsed identity fields), alongside `_add_cross_sectional_features`.
- The predict path (near forecaster.py:2984, where `_add_cross_sectional_features`
  is called) — same call, so training and serving features match.

Gated behind a default-off flag `ENABLE_QUALITY_SPREAD` (env `QUALITY_SPREAD=1`).
With the flag off, no columns are added and behavior is byte-identical to
current production.

### Static per-item attributes

Computed once from `parse_item_name` and cached (like the existing identity
cache): `weapon`, `skin_name`, `quality_rank` (1–5), `is_stattrak`,
`is_souvenir`, plus three group keys:

- `wear_group` = (weapon, skin_name, is_stattrak, is_souvenir) — members differ by wear
- `st_group`   = (weapon, skin_name, quality, is_souvenir) — members differ by StatTrak
- `sv_group`   = (weapon, skin_name, quality, is_stattrak) — members differ by Souvenir

## Features (~13 total)

Anchor for the wear axis is the **same-date group mean** price (not pinned to FN).

| Feature | Type | Axis |
|---|---|---|
| `wear_spread_ratio` = price / same-date group-mean price | level (control) | wear |
| `wear_spread_ratio_z60` = rolling z-score of the ratio over trailing 60d, per item | **deviation (signal)** | wear |
| `wear_spread_ratio_chg_7d` / `_chg_14d` | **change (signal)** | wear |
| `wear_ladder_dispersion` = std of group ratios same date | group regime | wear |
| `stattrak_premium` = same-date ST price / non-ST price (assigned to both rows) | level (control) | ST |
| `stattrak_premium_z60`, `stattrak_premium_chg_7d` | **signal** | ST |
| `souvenir_premium`, `souvenir_premium_z60`, `souvenir_premium_chg_7d` | control + signal | SV |
| `has_wear_siblings`, `has_stattrak_pair`, `has_souvenir_pair` | indicator gates | all |

Items with no siblings on a given axis get `0.0` for that axis's features and the
corresponding indicator flag off, so the tree can gate them out cleanly.

## Leakage safety

- Every cross-item aggregate uses **same-date** sibling prices — observable at
  time `t`, no future information (identical to the existing
  `_add_cross_sectional_features` `groupby("date")` pattern).
- z-scores and change features use only the item's own **trailing** history.
- A unit test asserts a feature value at date `t` is unchanged when `t+1` data is
  perturbed.

## Experiment harness & gating

### Permutation gate

The 13 features register as a named group `quality_spread` in the existing
`_validate_feature_groups` permutation machinery. This is the causal gate the
calibration doc mandates ("always pair A/B tests with permutation tests") — it
distinguishes real signal from extra tree capacity.

### A/B protocol

Two runs on identical data and seeds:

1. Baseline — `QUALITY_SPREAD=0` (current production behavior).
2. Treatment — `QUALITY_SPREAD=1`.

Compared on both metrics, since CV and the production backtest disagree by design:

- CV directional accuracy per horizon (training-time diagnostic).
- `scripts/backtest_accuracy.py` production backtest (definitive benchmark).

### Decision rule (pre-registered)

- **Ship** only if: the `quality_spread` group is permutation-retained on ≥1
  short horizon (3d/7d) **AND** production backtest Δ ≥ **+0.5pp** on those
  horizons **AND** no horizon regresses beyond the −1.5pp budget. On ship, bump
  model version `lgbm-v3 → lgbm-v4` and flip the flag default to on.
- **Shelve** otherwise: leave the code in place, flag default off, and record the
  null result in `accuracy-opportunities.md`'s Reality Check table.

## Testing

- **Unit:** group-key construction from names; feature values on a synthetic
  3-wear ladder with hand-computed ratios/z-scores; StatTrak/Souvenir premium on
  a synthetic pair; neutral (0.0 + flag-off) handling for no-sibling items; the
  leakage-guard test.
- **Integration:** `build_training_data` with flag on adds exactly the 13
  columns and with flag off adds none; predict path stays consistent.
- **Regression:** full `pytest` (currently 179 passing) stays green with the flag
  off.

## Deliverables

1. `_add_quality_spread_features` + static-attribute caching on `PriceForecaster`.
2. `quality_spread` feature-group registration in `_validate_feature_groups`.
3. Test suite additions in `tests/test_forecaster.py`.
4. A short experiment-runner note: the two A/B commands and where results land.
5. Results written up either way (ship → changelog + version bump; shelve →
   Reality Check null-result entry).

## Out of scope (YAGNI)

- Ladder lead-lag / cointegration / cross-variant return spillover modeling
  (Approach C) — high effort and leakage risk for a feature capped at ~1–2pp.
- Pinning the wear anchor to a specific exterior (FN) — group mean is used.
- Any paid or new external data source — this uses only existing price archive +
  parsed item names.
