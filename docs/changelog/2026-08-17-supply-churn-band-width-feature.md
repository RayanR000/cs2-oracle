# Supply-churn band-width feature (`supply_churn_*`), gated off

**2026-08-17.** Added `SUPPLY_CHURN_FEATURES=1` — three features derived from the
day-over-day **magnitude** of BUFF listing change, as a volatility / band-width signal for
the range forecaster. Off by default; A/B-only until measured.

## Why

`docs/research/2026-08-17-supply-churn-volatility-signal.md` measured, on a 9.56M-row
2022–2026 iflow panel, that `|Δlog(listing_count)|` predicts forward `|return|` (corr +0.11)
— a **2nd-moment** signal (signed change and listing level are both null). On days a supply
move actually happens it separates forward volatility 1.49× (9.6%→14.3%) where the item's
trailing realized vol stays flat, i.e. it catches vol jumps the incumbent feature misses.
This is the first candidate this session that clears the recurring walls: it needs no trade
(no 15% fee) and targets band width, which is the actual product — not per-item direction.

## What changed (one file + test)

The plan was a new sidecar builder off the live `supply-*.parquet`. **Not needed** — the deep
`buff_listing_count` sidecar (`supply-history.parquet`, BUFF, 2021-07→2024-02, 15.4M rows,
median 831 obs/item) is **already joined** by `_attach_sidecars` and consumed by no feature
(only sat in the `feature_cols` exclude set). So the change consumes it directly, on exactly
the venue the signal was validated on.

- **`models/forecaster.py`**: `_supply_churn_features_enabled()` (`SUPPLY_CHURN_FEATURES`) +
  `_compute_supply_churn_features()`, called beside the bid/stattrak features in
  `_compute_price_features`. Emits:
  - `supply_churn` — per-item `Δlog(1 + buff_listing_count)`
  - `supply_churn_abs` — `|supply_churn|`, the validated predictor
  - `supply_churn_present` — 1 where a positive listing count was joined
  Log-difference ⇒ scale-free (passes `test_scale_free_features`). A 0 listing count is
  masked to NaN (no supply observed ≠ a real level), and the diff is taken within item on a
  date-sorted frame. Names auto-map to the existing `supply_depth` group. `buff_listing_count`
  stays excluded as a raw literal. `ENGINEERED_CACHE_VERSION` 3 → 4.
- **`tests/test_supply_churn_features.py`**: 6 tests (absent-sidecar safe; churn is the log
  diff; magnitude sign-free; diff never crosses an item boundary; 0-listing days aren't churn;
  names group to `supply_depth`). Neighbouring feature/sidecar/allowlist/scale-free suites (35
  tests) green.

## Scope / caveats

- **Trainable now, not servable yet.** `buff_listing_count` covers 2021–2024, so the feature
  is populated on those folds (where the A/B measures it) but **NaN on 2026 serving rows** —
  it can be learned but contributes nothing at serve time until the live `supply-*.parquet`
  (lis_skins / market_csgo / waxpeer / bitskins) is unioned into the sidecar. That union is
  the **follow-up**, and carries a venue change (live venues ≠ BUFF) whose churn→vol
  transfer is the open question.
- **Score on conditional coverage, not pooled R².** The value concentrates on supply-move
  days; full-panel incremental R² over trailing vol was only +0.0025. The A/B must read
  interval **width/coverage on supply-shock strata**, in the `supply_depth` group with the
  allowlist widened (bid/stattrak precedent), not a headline DA or a pooled fit.
- Per `AGENTS.md` this is a range forecaster and the width variable has been hard to move; a
  null here is the likely outcome and is fine. The change is off by default and inert until
  dispatched.
