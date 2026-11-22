# Pre-registration: is served coverage uneven across price tiers?

**Date:** 2026-11-22, written and committed **before any confirmatory number is computed.**
**Status:** PROPOSED. Authorizes one read (below) and nothing else. No band, factor or serving change.
**Context:** `changelog/2026-09-09-conditional-coverage-by-stratum.md` found tier FLAT on a
20-date panel and skipped a per-tier arm. A descriptive look on 2026-11-22 suggests that has
changed. This is a falsification test of the new look, not a reopening of the old one.

## What was seen before writing this (disclosure)

On 2026-11-22 I ran one exploratory table on the served cohort (≥$1, `forecast_outcomes`
partitions, forecast dates 2026-10-01..2026-11-18, last 45 days) using ad hoc price bins. In-sample
coverage rose with price at every horizon. At h=3: $1–2 76.6%, $5–10 84.4%, $25–100 94.2%.
At h=7: 77.2 / 85.8 / 95.4. That table is **exploratory and excluded from the confirmatory read.**
It is also why this document exists, so the confirmatory data must be new.

## The one question

Is served interval coverage uneven across price tiers, beyond what the already-known
width tilt explains? (`changelog/2026-08-12-the-band-is-tilted-in-sigma`: low-sigma items
under-cover, high-sigma over-cover.) Price tier and sigma are correlated, so a tier gradient
could be the same tilt under another name. The width control below separates the two.

## Panel (fixed now)

- **Source:** `forecast_outcomes` partitions through `db/ops_store.py`, served cohort ≥$1,
  `in_interval` non-null, `excluded_forecast_date` applied.
- **Confirmatory window:** the **first 20 scoreable forecast dates on or after 2026-11-19**, per horizon.
  Earlier dates are excluded, including every date used in the exploratory table. Later dates never
  enter, so a re-run returns the same answer.
- **Tiers:** `backtest.scoring.price_tier`: tier 1 = $1–5, tier 2 = $5–20, tier 3 = $20–100.
  Tiers 4–5 (≥$100) are reported and excluded from every bar (thin: n ≈ 100 per horizon).
- **Horizons:** h=3 and h=7 are primary. h=14 is secondary. h=30 is reported descriptively
  only (too few matured dates inside the window to carry a CI).
- **Read timing:** one read per horizon, when that horizon has ≥ `MIN_HEADLINE_DATES = 20`
  scoreable forecast dates in the window. Expected: h=3 ~2026-12-11, h=7 ~12-15, h=14 ~12-22.
  No peeking before then; the instrument refuses to run early.

## Statistics

- **Coverage** per (horizon, tier) = share of rows with `in_interval`, pooled over dates.
- **Intervals:** moving-block bootstrap over forecast dates, block length = h dates
  (`research/2026-10-09-date-overlap-sensitivity`: lag-1 ACF of the h=7 series is +0.86, so an
  iid date bootstrap is too narrow). 2,000 resamples, 90% intervals.
- **Gradient G** = coverage(tier 3) − coverage(tier 1), in percentage points.
- **Width control G_w:** within each horizon, split rows into width terciles using
  `(price_high − price_low) / price_mid`, tertiles fixed on the window's own rows. G_w is the
  mean of tier-3 minus tier-1 coverage over terciles that contain ≥ 200 rows in both tiers.
- **Placebo:** permute tier labels within forecast date, 200 permutations. Must read null:
  the 90% interval for G spans zero and |mean G| ≤ 1pp, otherwise the harness is broken and the
  run is **void, not a result.**

## Bars

**PASS (per primary horizon):** G ≥ +8pp **and** its interval's lower bound > +3pp **and**
G_w ≥ +4pp with its lower bound > 0.

**Not-new (width tilt):** G passes but G_w fails. Then the gradient is the known sigma tilt,
and the result is recorded as a duplicate. No arm.

**KILL (either):** at h=3 **or** h=7, the interval for G spans zero, or G < +8pp.
Then tier-conditional coverage is retired, and the 2026-09-09 verdict stands.

**Overall:** PASS requires both primary horizons to pass. One passing and one null is
reported as UNDERPOWERED, not promoted. A failed bar does not license revising the bar.

## What a PASS authorizes

Only a second preregistration for an offline replay of a per-(horizon, tier) feedback factor in
`models/served_recalibration.py` (clamped, with a minimum-dates gate like the existing factor).
That one would bar on coverage within ±3pp in every tier **and** no wider overall band at
matched marginal coverage. Nothing here touches serving.

## Instrument

`backend/scripts/tier_coverage_read.py` (+ `tests/test_tier_coverage_read.py`), written and tested on
synthetic panels (known gradient, known null, width-tilt-only gradient, placebo) before the first read. It reads the partitions, refuses to run before 20
dates in the window, prints G, G_w, per-tier coverage and the placebo, and writes no artifact.
One run per horizon; a re-run after seeing the number is not part of this protocol.

## Standing constraints

This is a range forecaster. The design adds no directional claim, no ranker and no fourth
width denominator: a per-tier factor would be a multiplier on the existing band.
