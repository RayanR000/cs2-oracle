# The centre shrinks to zero, and the dollar band's under-coverage is the anchor wedge

**Date:** 2026-08-25
**Scripts:** `backend/scripts/centre_vs_lastprice.py --shrink`,
`backend/scripts/dollar_band_wedge.py` — both read-only, prod Postgres, no replay, no retrain.
**Panel:** 45,285 resolved `>=$1` rows over 20 forecast dates, `excluded_forecast_date` applied.

Two measurements, one on each half of what production serves: where the interval
sits, and what the interval means in dollars.

## A note on the panel source

Both scripts read **prod Postgres by default**, not the Parquet mirror. Per the
`archive-reads` rule neither copy of `ops/forecast_outcomes.parquet` is the full
scored panel, and this was measured rather than assumed: the durable CI-written
archive holds 1–2 fewer clean dates per horizon than prod (h=3: 14 vs 16, h=7:
15 vs 17, h=14: 10 vs 11). Every figure below is the prod read. `--archive-dir`
remains for offline cross-checks.

## 1. The centre's optimal shrinkage is zero

`centre = lambda * r_hat` nests both arms already measured — `lambda=0` is
quoting the last price, `lambda=1` is what production serves — so the
keep/retire question becomes a dial, and a negative skill at `lambda=1` no
longer has to mean the centre is information-free. It does anyway.

| h | dates | lam* | 90% CI | MAE(lam*) | MAE(1) | gain vs served |
|---|---|---|---|---|---|---|
| 3  | 16 | **0.00** | [0.00, 0.00] | 0.03537 | 0.03742 | +5.5% |
| 7  | 17 | **0.00** | [0.00, 0.00] | 0.05025 | 0.05358 | +6.2% |
| 14 | 11 | **0.00** | [0.00, 0.10] | 0.08122 | 0.08853 | +8.3% |
| 30 | 2  | 1.00 | [0.55, 1.00] | 0.13686 | 0.13686 | +0.0% |

The date-bootstrap CI on `lam*` is **degenerate at zero** for h=3 and h=7 and
reaches only 0.10 at h=14: there is no shrinkage of the GBM centre that beats
predicting no move, and serving it unshrunk costs 5–8% of centre accuracy. The
centre is not over-expressed, it is empty.

h=30 is the one cell pointing the other way and has two clean dates inside a
single drawdown. It is not evidence.

## 2. The dollar band under-covers because of the anchor wedge, and that is fixable

`predict()` quotes the triple off `current_price` — the smoothed anchor,
substituted **unconditionally** at `forecaster.py:6443` and so a number no venue
quoted — while the outcome resolves off `base_price`. `_derive_verdict` already
reports both predicates; nothing had attributed the gap between them.

| h | rows | dollar cov | calibrated cov | gap | 90% CI | recoverable | genuine |
|---|---|---|---|---|---|---|---|
| 3  | 15,700 | 0.721 | 0.887 | +0.167 | [+0.029, +0.317] | 70.0% | 30.0% |
| 7  | 16,828 | 0.722 | 0.904 | +0.182 | [+0.088, +0.278] | 72.6% | 27.4% |
| 14 | 10,774 | 0.540 | 0.828 | +0.288 | [+0.182, +0.394] | 66.8% | 33.2% |
| 30 | 1,983  | 0.657 | 0.771 | +0.113 | [+0.093, +0.135] | 35.6% | 64.4% |

**A consumer reading the h=14 band in dollars gets 54% coverage from an interval
calibrated to 80%.** Two thirds of those misses are inside the rebased band —
the wedge alone, recoverable by re-anchoring the served quote.

The decile cut is the attribution, with the calibrated column as its own
control:

| decile | \|wedge\| <= | rows | dollar cov | calibrated cov |
|---|---|---|---|---|
| 1 | 0.0046 | 18,114 | 0.906 | 0.906 |
| 2 | 0.0218 | 4,530 | 0.877 | 0.888 |
| 3 | 0.0506 | 4,527 | 0.802 | 0.874 |
| 4 | 0.0928 | 4,531 | 0.610 | 0.838 |
| 5 | 0.1387 | 4,526 | 0.466 | 0.842 |
| 6 | 0.1918 | 4,530 | 0.280 | 0.835 |
| 7 | 0.5542 | 4,527 | 0.093 | 0.839 |

Dollar coverage falls **0.906 → 0.093** across the wedge while the calibrated
band stays flat at 0.84–0.91. A width or calibration defect would have moved
both columns; only the published one moves. On the 40% of rows where the two
anchors agree (decile 1) the predicates are identical, as they must be.

**Limit.** The stored quote is the SMOOTHED one, so this measures
quote-vs-resolved-anchor and bundles the unconditional smoothing substitution
(what Arm A gates) with ordinary drift between the two resolution times. It is a
CEILING on what any anchor arm can buy, not a simulation of one — that needs a
replay against the raw price frame.

## What follows

1. **The anchor fix is the highest-value change on the board.** Arm A of
   `2026-08-11-serving-anchor-freshness.md` passed its gate on dollar error and
   was left off because its band-coverage effect was unmeasured. It is measured
   now, from above: up to 17–29 coverage points on the product's main output.
2. **The centre should be shrunk to zero at h<=14**, which is the same change as
   retiring the GBM's centre, arrived at independently.
3. Neither is quotable yet under `MIN_FORECAST_DATES = 20` — h=3 is at 16 dates,
   h=7 at 17, h=14 at 11. Signs and the decile monotonicity are what these
   establish; re-read the magnitudes at 20.
