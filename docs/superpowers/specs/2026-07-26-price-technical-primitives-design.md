# Design: Pure-Price Technical Primitives

Date: 2026-07-26
Status: Approved (pending spec review)
Branch: `feature/price-primitives`

## Goal

Add six new **pure-price** technical features to the served model to capture
volatility asymmetry and price/oscillator divergence — two documented price
dynamics (`docs/research/feature-engineering.md` §3.3, §5.1) that the current
feature set does not represent. Validate with a walk-forward A/B plus a
placebo (capacity-inflation) guard, and ship only on a real, non-flat gain.

## Background & rationale

- The served model is gated to a single feature group,
  `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` (`backend/models/forecaster.py:160`).
  The 2026-07-24 honest-serving ablation found the 85 non-price features add ~0
  directional accuracy. So the only low-friction place to add signal is *inside*
  `price_technicals`.
- `_feature_group()` (`forecaster.py:78`) routes features to groups by **name
  prefix**. Any feature named with `vol_`, `rsi_`, `macd_` (among others) joins
  `price_technicals` automatically and is served — **no allowlist change needed.**
- These primitives are all **pure price math**. They deliberately exclude
  volume-dependent primitives (OBV, VWAP-deviation): the 2026-07-16 audit found
  trade volume has zero predictive lift (`|r| < 0.002`,
  `docs/research/volume-data.md:25-29`). Including them would re-import known noise.
- The multi-source **cross-market divergence** feature (a separate idea) is
  deferred: a coverage check showed Buff163+Steam co-presence is erratic
  historically (0% most of 2026, ~57% only in July 2026) — too sparse to train
  on today. That is a "collect-now, revisit-later" bet, out of scope here.

## Feature specifications

All six are computed inside `_compute_price_features()`
(`backend/models/forecaster.py`), inserted **after the MACD block** (~line 840),
where `return_1d`, `return_7d`, `rsi_14`, and `macd_histogram` already exist.
Computation is per item on the daily-resampled, item/date-sorted frame.

### Volatility asymmetry (30d rolling over `return_1d`, `min_periods=5`)

| Feature | Definition | Interpretation |
|---|---|---|
| `vol_semidev_down_30d` | rolling std of `return_1d` where `return_1d < 0` | downside volatility |
| `vol_semidev_up_30d` | rolling std of `return_1d` where `return_1d > 0` | upside volatility |
| `vol_skew_30d` | `vol_semidev_up_30d / vol_semidev_down_30d` (denom `0→NaN`), `clip(0, 5)` | `>1` froth (upside more volatile), `<1` panic (downside sharper) |

Implementation note: build `ret_neg = return_1d.where(return_1d < 0)` and
`ret_pos = return_1d.where(return_1d > 0)`, then per-item rolling(30,
min_periods=5).std(). NaN where insufficient history.

### Oscillator divergence (per item, row-based `shift(7)` on daily series)

| Feature | Definition | Interpretation |
|---|---|---|
| `rsi_divergence_7d` | `rsi_14 − rsi_14.shift(7)` | RSI momentum (rising/falling oscillator) |
| `rsi_price_divergence_7d` | `return_7d.clip(-50, 50) / 50 − rsi_divergence_7d / 100` | large positive = price up while RSI down (bearish divergence); large negative = the reverse |
| `macd_hist_slope_7d` | `macd_histogram − macd_histogram.shift(7)` | MACD histogram momentum |

Implementation note: the exact scaling constants in `rsi_price_divergence_7d`
(the `/50` and `/100` normalizers) are a reasonable starting point and may be
tuned during implementation so the two terms are on comparable scales; the
*sign convention* (positive = price-up/RSI-down bearish divergence) is fixed.

### NaN handling

Features are left as NaN when history is insufficient, consistent with existing
price features — the downstream pipeline median-imputes them to a neutral value.
No new missing-flag columns (keeps the batch lean; the derived features inherit
missingness from `rsi_14` / `macd_histogram` / `return_*`).

### Group routing (verification)

Each name maps to `price_technicals` via `_feature_group()` prefixes:
`vol_semidev_down_30d`, `vol_semidev_up_30d`, `vol_skew_30d` → `vol_`;
`rsi_divergence_7d`, `rsi_price_divergence_7d` → `rsi_`;
`macd_hist_slope_7d` → `macd_`.

## Validation (Approach A: single batch A/B + placebo)

New script `backend/scripts/ab_test_price_primitives.py`, modeled on
`backend/scripts/ab_test_feature_contribution.py` (build features once, evaluate
subsets by column prefix) and reusing the walk-forward design from
`backend/scripts/walkforward_backtest.py` (expanding window, per-horizon
directional accuracy).

Three arms on identical walk-forward folds:

- **baseline** — drop the 6 new columns
- **treatment** — all features (including the 6 new)
- **placebo** — the 6 new columns column-shuffled (breaks any real signal while
  preserving added leaf capacity)

### Ship rule (ALL must hold)

1. `treatment` mean directional accuracy `>` `baseline` by a **meaningful**
   margin — not net-flat. (Precedent: quality-spread was +0.16pp mean → reverted.)
2. No individual horizon regresses beyond the **0.5–1.5pp** accuracy budget.
3. `treatment` `>` `placebo` — the gain is real signal, not LightGBM leaf-capacity
   inflation. (This is the increment-level analogue of the group permutation test
   in `_validate_feature_groups`, needed because all 6 features land in the
   already-served `price_technicals` group and cannot be isolated by that
   group-level test.)

Per-feature LightGBM importances from the treatment arm are reported for
attribution.

### Outcome handling

- **Pass** → keep the feature code; commit; write a `docs/changelog/` entry with
  the A/B + placebo numbers.
- **Fail** (net-flat or fails placebo) → revert the feature code; keep this spec
  and a changelog entry as a shelved-experiment record (the quality-spread
  pattern).

## Testing

Unit tests in `backend/tests/test_forecaster.py`:

- Each of the 6 features computes without error on a normal series.
- Correct NaN behavior on short history (< min_periods).
- `vol_skew_30d` respects the `[0, 5]` clip bounds.
- Monotonicity sanity: on a synthetic series with upside vol > downside vol,
  `vol_skew_30d > 1`; and the reverse case `< 1`.
- `rsi_price_divergence_7d` sign convention: a constructed price-up / RSI-down
  window yields a positive value.
- All 6 names route to `price_technicals` via `_feature_group()`.

## Workspace & rollout

- **Isolation:** git worktree `../cs2-oracle-wt/price-primitives` on branch
  `feature/price-primitives`, created off clean `main`. The uncommitted
  HP-search change (`forecaster.py` 14d/30d + `meta.json`) stays untouched in the
  main checkout and is out of scope here.
- **No production wiring changes** — once computed in `_compute_price_features()`,
  the features flow through the existing served path via prefix-based group
  routing.

## Deliverables

1. Feature code in `_compute_price_features()` (6 features).
2. A/B script `backend/scripts/ab_test_price_primitives.py` (3 arms).
3. Unit tests.
4. A `docs/changelog/` entry recording the A/B + placebo result (ship or shelve).

## Out of scope

- Volume-dependent primitives (OBV, VWAP-deviation) — excluded by decision.
- Cross-market / source divergence — deferred pending multi-source coverage.
- Any change to `FEATURE_GROUP_ALLOWLIST` or non-price feature groups.
- The uncommitted HP-search change.
