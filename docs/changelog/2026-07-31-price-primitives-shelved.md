# Price technical primitives — tested & shelved (2026-07-31)

## Hypothesis

The served price_technicals group carries symmetric volatility (`vol_*_30d` std) and
level-only oscillators (`rsi_14`, `macd_histogram`). Two gaps follow from that:

- A symmetric std collapses **panic** (sharp downside) and **froth** (volatile upside)
  into one number.
- An oscillator *level* says nothing about whether momentum is building or fading, and
  nothing about price/RSI **disagreement** — the classic divergence setup.

Hypothesis: adding pure-price primitives for both would improve served directional
accuracy without touching any new data source.

Six features, all in `_compute_price_features` (`forecaster.py:947`):

| Feature | Reads as |
|---|---|
| `vol_semidev_down_30d` / `vol_semidev_up_30d` | downside / upside semi-deviation |
| `vol_skew_30d` | up-vol ÷ down-vol, clipped to [0, 5]; > 1 froth, < 1 panic |
| `rsi_divergence_7d` | 7-day change in RSI (momentum of momentum) |
| `rsi_price_divergence_7d` | price up while RSI down ⇒ bearish divergence |
| `macd_hist_slope_7d` | 7-day change in MACD histogram |

## Experiment

`ab_test_price_primitives.py` — three arms on identical walk-forward folds:

- **baseline** — the six columns dropped.
- **treatment** — all features, the six included.
- **placebo** — the six column-shuffled, as a capacity-inflation guard.

`merge_price_primitives_ab.py` unions sharded runs (one shard per horizon, all three
arms) and refuses to silently pick between two shards claiming the same (horizon, arm).
It reports **per-fold win counts** next to the pooled delta, because a pooled mean can
rest on a single lucky fold.

**Pre-registered gate:** SHIP iff treatment > baseline (meaningful, non-flat) AND
treatment > placebo AND no horizon regresses beyond the 0.5–1.5pp budget.

## Result — SHELVED (gate not cleared)

On the **40-item smoke**: pooled delta **+0.13pp**, and treatment beat baseline in
**12–14 of 26 folds at every horizon**. A coin flip in fold-win terms, where the pooled
mean merely looks small. Nothing here clears "meaningful, non-flat."

**This was never run at decision scale.** 40 items is a smoke config, so the honest
verdict is *unproven*, not *refuted* — the effect could be real and simply invisible at
this sample size. What the smoke does rule out is a large, obvious win.

## Disposition

- The six columns are **still engineered**, so the harness can re-run the arms unchanged.
- They are **withheld from training** via `ItemForecaster.SHELVED_FEATURES`
  (`forecaster.py:209`), which is unioned into the `exclude` set that builds
  `feature_cols`. The gate is needed because all six are price technicals *by name*
  (`_feature_group()` keys off the `vol_`/`rsi_`/`macd_` prefixes), so
  `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` would otherwise admit them straight
  into production.
- Regression cover: `test_build_training_data_*` asserts the shelved names stay out of
  `feature_cols` while remaining in the engineered frame.

**To revisit:** drop the names from `SHELVED_FEATURES` and re-run at decision scale
(≥200 items) before shipping — the smoke result is not evidence either way.
