# Volatility/stability tags — the range product's first discovery surface

The forecaster has no capturable trade edge at any horizon (short: friction
dominates; long: the move is big enough by h=90/180 but **unselectable** — AUC
0.51–0.53 across three OOS windows, and the unconditional bet lost money every
window). See the horizon screen + selection test run 2026-08-20. So the product
is a calibrated **range** forecaster, and its differentiator is calibrated
uncertainty, not a prediction to trade on.

This ships surface #1 of a range-product-first direction: **volatility/stability
tags**, built entirely on outputs the pipeline already emits (`price_low/mid/high`
and `exceed_p`). No new modelling, no change to how bands or `exceed_p` are
computed, no frontend.

## What it adds

- `api/volatility_tags.py` — pure, I/O-free derivation, fully unit-tested:
  - `swing_pct(low, high, mid)` = half-band / mid ("±X% over the horizon").
  - `compute_thresholds(swings)` = the 1/3 and 2/3 quantiles of a swing set.
  - `label_for(swing, thresholds)` = Stable (≤ low tertile) / Moderate / Volatile
    (> high tertile).
  - `tag_fields(...)` and `build_ranking(...)` compose these for the two surfaces.
- **Per-item** (`GET /items/{id}/prediction`) now carries `expected_swing_pct`,
  `move_odds` (`exceed_p` straight through, NULL on pre-exceedance artifacts) and
  `stability_label`, populated in **both** the parquet and DB paths. The
  no-forecast placeholder path stays untagged (no real band).
- **Discovery** (`GET /items/volatility?horizon=&sort=swing|move_odds&order=&min_price=&limit=`)
  ranks the served ≥$1 universe, each row carrying the three fields.

## Design notes

- **The label is relative-to-peers.** It comes from the within-horizon universe
  tertiles, so "Stable" means calm relative to the served set, not calm in
  absolute terms. Percentile-based because band scale differs by horizon.
- **The label rides on swing, not `exceed_p`**, because `exceed_p` is nullable;
  `move_odds` is surfaced alongside as the magnitude signal (never a trade call —
  consistent with the exceedance invariant).
- Per-item label needs the universe's tertiles; `_horizon_swing_thresholds`
  computes them once and memoises per `(horizon, day)` so a single lookup does
  not re-scan every served item.

## Explicitly out of scope

Band-breakout alerts (#2), portfolio aggregation (#3), any frontend, and any
change to band/`exceed_p` computation.

Tests: `tests/test_volatility_tags.py` (26), `tests/test_volatility_endpoint.py` (7).
