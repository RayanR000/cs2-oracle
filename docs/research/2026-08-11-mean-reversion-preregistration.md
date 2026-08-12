# Pre-registered read: does index-level mean reversion replicate out of sample?

**Written 2026-08-11 before the test window was scored.** The hypothesis comes from
`2026-08-11-market-factor-is-not-forecastable-from-its-own-history.md`, where the trailing-180d
drift correlated **negatively** with the forward market factor at 4 of 4 horizons (−0.093 to
−0.162) on **2024-01-01 → 2026-08-08**. That is a post-hoc observation on a read built to test
the opposite sign, so on its own it is a candidate, not a finding.

## Why this is a real test and not a re-read

The test window is **2014-01-01 → 2023-12-31**, disjoint from the window that produced the
hypothesis. Coverage was checked first and is dense — 365 days in every year, 392→3,473 rows/day,
458→3,616 items — giving roughly 1,200 / 520 / 260 / 120 non-overlapping windows at 3/7/14/30d
against the 250 / 108 / 54 / 25 available in the discovery window.

Estimator and label are unchanged production code (`forecast_market_factor`,
`market_factor_for_horizon`). Nothing is re-fitted; only the window moves.

## The bar, fixed in advance

**Primary — does the sign replicate?** Pearson correlation between the trailing-drift forecast
and the realised factor, on non-overlapping windows, is **negative at ≥ 3 of 4 horizons**.

**Secondary — is it usable?** The contrarian estimator (the negated drift) beats the **constant
call** — `max(up-rate, down-rate)` on the same rows — on directional accuracy at **≥ 3 of 4
horizons**.

Both must hold to call this a signal. Primary alone means the sign is real and the magnitude is
not usable, which is recorded and stopped, not developed.

**Void conditions:**
- Fewer than 30 non-overlapping windows at a horizon → that horizon is unreadable, not null.
- Two or more horizons void → the whole read is void.
- If the index's early years fail `MIN_INDEX_ITEMS`, the affected dates drop out via `valid` and
  the surviving count is what the first bullet is applied to.

## Declared confounds

- **Composition drift.** The cohort grows 458 → 3,616 items across the window. An index whose
  membership changes can manufacture level moves; `build_market_index` pairs day-over-day per
  item, which is the defence, but it is not a proof.
- **Regime span.** Ten years covers multiple market regimes. A correlation pooled across all of
  them can hide sign flips, so per-horizon sign consistency is the statistic — not magnitude.
- **No confidence interval.** Overlapping windows autocorrelate and a naive CI would be too
  narrow (the same defect `paired_mde` had before 2026-08-07). Non-overlapping sampling is the
  mitigation; the readable statistic remains sign consistency across horizons and across two
  disjoint windows.

## Committed interpretation

- **Both rules pass** → mean reversion replicates across two disjoint windows and ~10 years. It
  becomes a candidate date-level feature, and the next step is a paired A/B — not a ship.
- **Primary passes, secondary fails** → the sign is real, the magnitude is unusable. Record and
  stop. This is the outcome the discovery window's magnitudes (median −0.1 to −0.9% against a
  realised sd of 2.8–6.2%) actually predict.
- **Primary fails** → the 4-of-4 negative correlation was window-specific. N2's own-history leg
  closes completely, and the exogenous leg is all that remains of the track.
