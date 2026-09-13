# Pre-registration: does the clean-era panel change the centre verdict?

**Date:** 2026-09-13, written and committed **before any centre number is computed on this panel.**
**Status:** PROPOSED — no measurement authorized beyond the coverage gate below.
**Context:** `2026-09-10-served-direction-withheld.md` marked the centre verdict CONVERGED
(`lambda* = 0.00` reproduced four times across three labels/harnesses). This is a single
falsification test of that verdict on new ground — not a reopened investigation.

## The one question

Achieved centre rank IC is order 0.03–0.04 (TFT 14d +0.043, commit `5c94230`; lambdarank
vs-q50 edge +0.039/+0.030 at 14/30d) against a measured label ceiling of IC 0.93/0.96 at
h=14/30 on the 2025 panel (`2026-09-10-shrink-k-gbm-and-vol-rank-measured.md`). Every prior
centre read ran on a panel where the ceiling was low (2026: 18–33% frozen quotes) or the
label was single-source by construction (the 08-22 within-source clean label). With the 2025
iflow promotion (`11eaf6c`, unblocked for the 2025 range by `1bbdb02`), the 2025 voted
composite is simultaneously high-ceiling *and* composite for the first time. The question is
whether the model captures any of that headroom — i.e. whether the CONVERGED verdict was a
property of the panel rather than of the centre.

## Coverage gate (measured pre-registration, read-only)

Local canonical archive, 2025-01-01..2025-12-31, through `prices_relation` +
`archive_universe_sql_filter`, `mean_price > 0`:

| check | result |
|---|---|
| days with both `source IS NULL` and `source='buff_iflow'` | **360/365 (98.6%)** |
| of those, days with ≥1,000 items per side | 358 |
| joint (item, day) pairs / items / days | 250,355 / 1,776 / 360 |

The 5 absent days are the known upstream iflow feed drops (2025-05-24, 06-05, 06-13, 06-14,
06-24). Gate (≥90% joint days) passes. The ≥$1 median filter applies at run time and will
reduce the item count toward the ~944 of the ceiling measurement; the day count is what the
gate constrains.

## Panel and arms (fixed now)

- **Panel:** 2025 voted composite forward returns at h=14/30 (primary; h=3/7 reported for
  context only), ≥$1 median items, walk-forward by forecast date with the `horizon + 13`
  embargo derived at call time (`forecaster.py::embargo_days`). Horizons never pooled.
- **Arms:** production q50 GBM centre (`r_hat`) fit walk-forward on pre-boundary rows only;
  last-price level null; `−return_1d` ranking null (the baseline the model has lost to at
  every prior read); capacity-matched shuffled-feature placebo.
- **Label basis** fixed before any read; primary read on the **tied cohort** per the 08-11
  denominator finding (`p/S` contaminates pooled rank IC ±0.14).

## Bars

- **Primary — rank IC.** Within-date Spearman of `r_hat` vs composite forward return,
  date-block bootstrap (never row bootstrap: one date's items share the market factor).
- **Secondary — level.** MAE skill vs last-price (`centre_vs_lastprice.py` logic, rebased
  off the model's own quote — never off `base_price`, which measures the anchor wedge).
- **(P) Placebo.** Shuffled-feature arm must read null (CI spans zero, |mean| ≤ 0.02);
  otherwise the harness is broken and the run is void, not a result.

**Kill criterion (either kills):** at *either* h=14 or h=30, GBM rank IC CI spans zero **or**
GBM fails to beat `−return_1d`. Then CONVERGED stands and the question is permanently
retired — no second panel, no second basis.

**Pass criterion (all required):** at *both* h=14 and h=30, GBM rank IC CI clears zero
**and** beats `−return_1d` **and** MAE skill vs last-price is positive. Ceiling headroom
alone is not a result: if the naive baselines also rise on this panel, the panel got
easier, not the model better.

## Void conditions

- Joint-source day share below 90% on the run-time panel (gate re-checked, not assumed).
- Any fold, embargo, universe, or anchor-selection change after a number is seen.
- `−return_1d` itself null on the panel — then the panel cannot referee ranking and the
  verdict is UNDERPOWERED, never a pass.
- Quoting pooled rank IC without the tied-cohort read, or any DA figure without its
  PT/realised-down-rate companions (invariant 4).

## Cost and what the outcome buys

Offline walk-forward retrains on 2025 — hours, one dispatch, no serving touch. A **fail**
retires the centre question permanently. A **pass** buys exactly one thing: a confirmation
run on served 2026 dates once h=30 matures there — never a serving change on its own.
