# The exceedance probability becomes a served signal (Phases A–D)

**2026-08-20**

## Why

Accuracy is not the product's binding constraint — the actionable strategy selects
almost no rows and friction dominates
(`2026-08-15-cs2-oracle-is-a-range-forecaster.md`, `2026-08-11-actionable-selection-is-the-base-wedge.md`),
and every band-**width** lever is measured dead (sigma-tilt / learned scale /
climatology all in `.claude/rules/`). The one market-orthogonal, date-stable
quantity is the exceedance probability `P(the h-day move clears the round-trip
cost)`. It already existed — but only as an internal band-width input under
`EXCEEDANCE_SCALE`, consumed as a conformal denominator and thrown away. And with
the winning `CLIMATOLOGY_SCALE` on by default (mutually exclusive with the
exceedance scale), the head was **not trained in production at all**, so no
`exceed_p` was produced anywhere.

These four phases promote it to a disclosed per-item output.

## What changed

**Phase A — decouple + emit** (`models/forecaster.py`)
- New flag `EXCEEDANCE_HEAD=1` (`exceedance_head_enabled()`, off by default). The
  production head now trains when EITHER the band uses it (`EXCEEDANCE_SCALE`) OR
  it is served (`EXCEEDANCE_HEAD`) — `_train_horizon_inline`. The CV out-of-fold
  head, which feeds `q_hat`, stays gated on `EXCEEDANCE_SCALE` alone: that is a
  matched pair (`q_hat` is only exceedance-scaled when the band is), and the served
  signal must not perturb calibration.
- New flag-independent accessor `exceedance_probability(horizon, rows)` — clipped
  to (1e-3, 1], `None` when the horizon has no head. `band_scale` refactored to
  reuse it, so there is one head-predict path.
- `predict()` emits `exceed_p` on every forecast record (`None` on a pre-Phase-2 /
  degenerate artifact). Save/load already carried the head unconditionally.

**Phase B — label decision (locked): one-sided.** The label is
`target_exceed_{h}d = target_return_{h}d > actionable_threshold(tier, csfloat)` —
P(**upside** move clears cost), not two-sided `|move| > cost`. It is what was
already trained and validated, and for a buy-side product "will it clear cost to
the upside" is the more actionable question. No refit. The served field carries
this meaning; `exceed_p` is a MAGNITUDE signal, never a directional call
(invariant 4).

**Phase C — persist** (`database.py`, `migrations/`, `scripts/forecast_prices.py`)
- `ItemForecast.exceed_p` — nullable `Float`; migration
  `0024_add_forecast_exceed_p` (mirrors the 0022 anchor disclosure, idempotent).
- `_write_forecasts_to_db` writes it, and `exceed_p` joins the missing-column
  guard: an unmigrated prod DB degrades to writing without it (NULL = "not
  recorded") rather than failing the daily batch. The Parquet mirror keeps it
  regardless (no schema to violate).

**Phase D — reliability** (`scripts/replay_serving.py`)
- `_reliability_rows` / `_reliability_ece` + an `EXCEEDANCE RELIABILITY` table:
  predicted `exceed_p` vs the realised one-sided exceedance rate
  (`actual_ret > actionable_threshold(tier, csfloat)`, the SAME bar the label uses,
  tier from the anchor quote via `scoring.price_tier`), in fixed-width probability
  bins, with a count-weighted ECE. Printed only when the artifact carries a head.
- It runs on **replayed** forecasts, because no served `exceed_p` exists in the
  outcomes store yet — the same pre-confirmation path band coverage used. Trust
  h3/h7 only: h14 is thin and there is no valid h30 outcome
  (`2026-08-19-drop-replayed-2025-12-01-cohort.md`).

## Status / not yet live

Off by default; every artifact today is byte-identical to before. Cutover is
three steps, in order: (1) `alembic upgrade head` lands `0024` in prod; (2) set
`EXCEEDANCE_HEAD=1` on the training workflow (costs ~4 boosters + 4 files per
retrain) and read the replay `EXCEEDANCE RELIABILITY` table before trusting it;
(3) served `exceed_p` then accrues in the outcomes store, at which point the
reliability read can move off replay onto served outcomes — the exceedance
analogue of the `q_hat` served-feedback gate.

**Caveat inherited from the threshold:** `SPREAD_BY_TIER` in
`backtest/friction.py` is a nearest-band approximation, not measured per tier, so
the cost bar — and every calibration figure resting on it — carries that
approximation.

## Tests

`tests/test_exceedance_head.py`, `tests/test_forecast_exceed_p_persistence.py`,
`tests/test_exceedance_reliability.py`. Also fixed two `test_exceedance_scale.py`
cases that broke when `CLIMATOLOGY_SCALE` went default-on (needed
`CLIMATOLOGY_SCALE=0` to reach the exceedance-scale assertion).
