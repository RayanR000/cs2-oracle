# `log1p(listing_count)` band-width conditioner is refuted — do not re-run

**Date:** 2026-08-18

Closes the one lever `research/2026-08-16-listing-count-floor.md` left standing. That doc
killed the listing-count *floor* but recommended trying `log1p(listing_count)` as a band-**width**
conditioner (the gradient is a conditional-σ statement, not a gate). It was ranked as item 6 of
`research/2026-08-16-next-steps.md`. **Measured 2026-08-18; it fails.**

Prereg: `research/2026-08-18-listing-count-band-width-conditioner-preregistration.md`.

## Result — 0/3 horizons pass

Offline replay arm (reuses `scripts/replay_serving.py` internals in a standalone; audited anchor
2026-06-16, h=3/7/14; listing `L` = max over the 4 non-Steam venues from the single 2026-08-06
snapshot; slug-hash cal/score split; grid-fit `beta<=0`; ruler = conditional-coverage dispersion by
listing bucket):

- **h=3** cuts bucket-dispersion 27% but **worsens the worst-bucket miss** → guardrail fail.
- **h=7 / h=14** worsen dispersion (the cal-fit `beta` overfits).
- Width-neutral (~×0.99) as designed, so it is not even buying narrower bands.

## Why it can't work at the served cohort (the decisive, structural finding)

The thin buckets that are the entire hypothesis's target — **1–5 and 6–15 listings — hold <20
items each in the served ≥$1 cohort** and drop out. The surviving buckets are 16–30 / 31–50 /
51–100 / 101+, and **101+ alone holds 372 of 523 rows**, where control coverage is already flat
(0.77–0.88). The per-item `sigma` already absorbs the listing-count information for the population
actually served. There is no thin-tail miscalibration to fix at ≥$1 because the thin tail is barely
present. A live served-panel confirmation cannot repair an absent population.

## Scope / caveats

- One stale-snapshot anchor (`n ≈ 523`) — a first look, not a hard kill, but the direction is
  unambiguous across all three horizons.
- This conditioner targeted the **conditional** (per-bucket) miscalibration. That is a different
  axis from the marginal over-coverage, whose cause is **regime non-exchangeability** — *not* the
  expanding window (refuted, `2026-08-12-expanding-window-refuted-for-band-width.md`).

## Verdict

**CLOSED. Do not re-run.** The surviving band-calibration lever is the dormant served-outcome
`q_hat` feedback (`2026-08-16-served-outcome-feedback-calibration-built-dormant.md`; self-activates
~2026-09-06), not a listing term. Only reopen if the iflow backfill
(`superpowers/specs/2026-08-17-iflow-serve-universe-expansion-design.md`) materially grows the
thin-listing ≥$1 cohort — and re-check bucket `n` before spending any more time on it.
