# The band is sized on trailing volatility that the forward return does not repay

**Date:** 2026-08-13
**Answers:** the 1.19x left unaccounted in `2026-08-13-band-level-on-audited-anchors.md`
**Instrument:** `backend/scripts/attribute_band_level.py` (offline, ~90s per panel, writes nothing),
5 tests in `tests/test_band_level_attribution.py`
**Panels:** `voted_39810715096a47b1688e44ef` (731 dates, 2024-07-09 → 2026-07-09) and
`voted_13a571942605b81e37698369` (283 dates, 2025-10-26 → 2026-08-08)
**Status:** 🔑 the level defect is **located and quantitatively sufficient**. It is not the cohort and
not the denominator. No flag changes; the remedy overlaps a refuted class and needs its own
pre-registration.

## The question, and why it reduces to one ratio

`conformal.band` sets `high − low = 2·q_hat·sigma**beta` in return space, and both published
half-widths are half of that over the band's own centre. With one `q_hat` and `beta = 1`, **the
served/calibration width ratio IS the served/calibration `sigma` ratio** — there is no third term.
(`range_pct`'s `(1 + mid/100)` differs from 1 by the predicted return, median 0.95%.)

So the 1.529 / 1.548 / 1.553 / 1.520 measured on run `31657639707` is a statement about `sigma`, and
it decomposes exactly two ways: **which** items are served, or **when** they are served.

## The answer: when, not which — and the residual does not follow

On the two-year panel, served rows being the three audited anchors it covers
(`2026-04-22, 2026-05-16, 2026-06-16`):

| h | width ratio (measured) | same-items `sigma` | same-items `\|resid\|` | **score = `\|resid\|/sigma`** |
|---|---:|---:|---:|---:|
| 3 | 1.529 | **1.488** | 1.123 | **0.755** |
| 7 | 1.548 | **1.502** | 0.984 | **0.655** |
| 14 | 1.553 | **1.448** | 1.027 | **0.709** |
| 30 | 1.520 | **1.699** | 1.106 | **0.651** |

**The same-items `sigma` ratio reproduces the width ratio** (1.49 / 1.50 / 1.45 / 1.70 against
1.53 / 1.55 / 1.55 / 1.52). Composition adds only ~10% on top of it — the pooled-basis figures are
1.637 / 1.667 / 1.621 / 1.862 — so **the cohort is not the story and the earlier "1.19x unaccounted"
is closed.** The 1.28-1.29x on record was measured against a pooled window containing the anchors'
own era; against each item's own two-year history the shift is 1.45-1.70x.

**And the residual the `sigma` normalises does not move: 0.98-1.19x.** That is the whole defect.
The conformal score is `|resid| / sigma`, so a served `sigma` 1.5x the item's norm costs nothing if
the item's forward move is also 1.5x — the band is wider because the item is genuinely wilder.
Here the trailing volatility rises by half and the realised return does not, so the served score sits
at **0.65-0.76x** the pooled score and `q_hat` — the pooled 80th percentile — lands at a higher
quantile of it. That is arithmetically sufficient for the observed 86-89%: `p80 · 1/0.7` sits near the
88th-92nd percentile of a distribution this heavy-tailed.

Corroboration on the shorter panel (2025-10 → 2026-08, so a pool much closer to the anchors' own
era): the same-items `sigma` ratio falls to 1.33 / 1.32 / 1.26 / 1.41 and the score ratio rises to
0.87 / 0.76 / 0.82 / 0.77 — same sign, same mechanism, smaller because the comparison window is
itself half-volatile.

## `sigma` is a quarterly regime, and the audited anchors sit in a high one

Median `sigma` on the ≥$1 cohort, by quarter, two-year panel:

| 2025Q3 | 2025Q4 | 2026Q1 | 2026Q2 | 2026Q3 |
|---:|---:|---:|---:|---:|
| 0.0663 | 0.1148 | 0.0655 | **0.1151** | 0.0693 |

**Trailing volatility nearly doubles between quarters**, and three of the four audited anchors sit in
2026Q2.

⚠️ **CORRECTION, same day, before this entry was acted on.** This section first said the audited set is
"concentrated in one volatility regime" and named a fifth anchor-selection criterion on that basis.
**That is wrong.** It rested on the three anchors the panel could measure — `prepare_targets` voids
every `2026-07-06` label as spanning a collector cutover, so the instrument silently dropped the one
anchor outside 2026Q2. Measured directly on the panel's `sigma` (which needs no label), the set spans
a **2.2× range** and the dose-response is *within* it:

| anchor | `sigma` vs own history | cov% h=3 | h=7 | h=14 | h=30 |
|---|---:|---:|---:|---:|---:|
| 2026-04-22 | **1.888×** | 89.85 | 94.04 | 95.90 | 93.76 |
| 2026-05-16 | 1.528× | 85.58 | 91.52 | 82.38 | 88.12 |
| 2026-06-16 | 0.994× | 85.99 | 89.45 | 85.99 | 84.92 |
| 2026-07-06 | 0.857× | 88.90 | 80.86 | 80.08 | 77.94 |

Spearman correlation between the ratio and served coverage: **0.2 / 1.0 / 0.8 / 1.0** at 3/7/14/30d —
monotone at 7d and 30d, near-monotone at 14d, and h=3 the exception. n = 4, so this is a pattern to
test, not a result. **This is stronger evidence for the mechanism than the withdrawn claim was**: the
level moves with trailing volatility *across dates within one audited set*, and the across-anchor
coverage spread (4.27 / 13.18 / 15.82 / 15.82pp) is the quantity a remedy has to shrink.

The genuine anchor-selection lesson survives in weaker form: **an anchor's trailing-vol level belongs
beside its collection audit when a set is chosen**, because a set drawn only from high-`sigma` dates
would report a coverage mean that is not the year's. This set is not that.

## What this does and does not license

**Does:** it names the level defect as a **trailing-vs-forward volatility gap** — `price_std_60d` is
a backward-looking window, it doubles between quarters, and the forward return does not double with
it. This is coherent with the three-scale failure: `sigma`, `sigma**beta` and a learned scale all
re-weight items *within* a date, and none of them can fix a term that moves the whole cross-section
between quarters.

**Does not:** license the obvious remedy. "Divide `sigma` by its own cross-sectional median on the
date" is a date-level rescaling of the band, and the date-level conditioning class was **refuted**
(`2026-08-12-the-band-is-tilted-in-sigma.md`: a 60-day trailing `q_hat` and three date-level
volatility states are all worse than pooled, placebos ≤0.11pp). Two reasons that refutation may not
bind here, and both are arguments rather than results:

1. It was scored **level-matched to 80% marginal**, which removes exactly the quantity this entry is
   about before the comparison starts.
2. It conditioned `q_hat` on a date-level *state*; this is a rescaling of `sigma` itself, so the
   within-date ordering is untouched.

**Anything built on this needs a pre-registration naming the bar, the placebo, and — per
`2026-08-12-sigma-exponent-paired-read.md` — whether the width and coverage bars are calibration-set
or served.** The instrument to read it on is a forecast-date panel, not a CV fold.

## Reading the instrument

- `decompose(frame, anchors)` is the whole arithmetic and is unit-tested against constructed panels:
  a pure date effect, a pure composition effect (which must vanish on the same-items basis), a
  **matched** residual (which must leave the score at exactly 1.0 — the null the entry turns on), and
  an item whose only row is the anchor.
- Ratios only. The panel derives its own clip bounds and cohort, so no absolute number here is
  production's, and the residual is the realised return rather than a booster's error — the same
  standing assumption as `attribute_marginal_coverage`, justified by predicted `|return|` being median
  0.95% against half-widths of 10-31%.
- ⚠️ **`2026-07-06` is absent from every per-anchor row.** `prepare_targets` voids labels spanning a
  **collector cutover**, and its own detector names `2026-03-22, 2026-07-09, 2026-07-10, 2026-07-11,
  2026-07-12` — so at h=3 the 07-06 label (target 07-09) is voided, and at longer horizons too. Worth
  noting for its own sake: **the label path already had a cutover detector that flagged 2026-07-09,
  and anchor selection never consulted it.** It does *not* flag 2026-04-15/16, which the new
  collection audit does catch, so the two checks are complementary rather than redundant.
