# Pre-registration: does h=14 tied ranking survive on served dates?

**Date:** 2026-09-13, written and committed **before any rank IC is computed on served rows.**
**Status:** PROPOSED — runnable now (maturity gate already clears; see below).
**Follows:** `2026-09-13-clean-era-centre-preregistration.md` (the 2025 read) and its verdict
note `changelog/2026-09-13-clean-era-centre-measured.md`. This confirms the ranking
observation only. Level is settled dead and is not re-tested here.

## Why this leg exists, and what it is not

The 2025 read found tied-cohort h=14 rank IC +0.094 (paired vs naive significant, placebo
null) with level skill negative — "ranks without level accuracy". Two precedents collide:

- FOR transfer: the 08-11 replay measured served tied rank IC +0.10/+0.15/+0.12 at 3/7/14d
  (`2026-08-11-smoothed-anchor-label-measured.md`) — same ~0.1 magnitude, different era,
  different trainer. The 2025 number may be a replication, not a novelty.
- AGAINST naively translating it: that 08-11 read had 4 anchors, no interval, no naive
  baseline — "a replication, not power". And the realised follow-up to a replay signal
  attenuated 4× (+0.13–0.17 → +0.037,
  `2026-08-11-clean-anchor-gate-measured-on-realised-outcomes.md`).

So the open question is narrow: with production models on realised 2026 outcomes, a naive
baseline, and a date-block interval, is h=14 wedge-free ranking positive? This leg answers
only that. A null does not refute the 2025 read (different trainers, different regime);
a positive does not ship anything (range stance and withheld direction stand —
`2026-09-10-served-direction-withheld.md` is untouched by any outcome here).

## Amendment (2026-09-13, before any rank number is computed)

While scoping the instrument, `item_forecasts.anchor_clean` was found: the serve-time
tied mask, recorded per forecast row (the exact 08-11 cohort definition — "the cohort
split every 2026-08-11 served-signal figure was measured on"). It dominates the
wedge-negligible proxy (serve-time truth vs stored-legs reconstruction under archive
revision), so the primary cohort is decided mechanically at run time, before any rank
statistic is read:

- If `anchor_clean` is non-null on ≥70% of h=14 panel rows → primary is
  `anchor_clean == True`; wedge-negligible becomes a reported sensitivity read.
- Else → primary stays wedge-negligible as originally written; `anchor_clean == True`
  is reported iff it spans ≥10 qualifying dates, never barred on.

Either way both cohorts are reported and the CONFIRMED conjunction reads the primary.
Reason and rule are fixed here, ahead of data; the coverage share itself is a panel
count, not a result.

## Maturity gate (measured pre-registration; a date count is not a result)

`centre_vs_lastprice.py --gate`, prod Postgres read-only, 2026-09-13: h=14 holds
**24 clean dates / 22,705 rows** (was 13 on 09-08; the gap-fix accruals landed), skill
−0.069, CI clear of zero. Level reconfirms dead on served — consistent with 2025, and
out of scope here. h=14 clears MIN_FORECAST_DATES=20 today; h=30 sits at 10 and is
excluded from this leg entirely.

## Panel and arms (fixed now)

- **Panel:** served `forecast_outcomes`, h=14 only, `base_price >= $1`, realised outcomes,
  `excluded_forecast_date` applied — the `centre_vs_lastprice` panel byte-for-byte.
- **Prediction:** `r_hat = mid/current_price − 1`, the script's own rebasing (charges the
  model for its centre, never for the anchor wedge).
- **Outcome:** `actual_price/current_price − 1`, the quote basis. On the primary cohort
  (below) this equals the scored basis by construction, which is what dissolves the
  anchor question instead of answering it.
- **Naive:** `−return_1d`, log trailing-1d return on the voted composite at the forecast
  date, calendar-exact. Rows missing it drop from BOTH arms — pairing on identical rows,
  pre-registered. (The 2025 read's referee, same definition.)
- **Primary cohort — wedge-negligible:** `|current_price/base_price − 1| ≤ 0.001`, on
  stored legs only (arm-invariant). Rationale: exact float-equality with a later-resolved
  leg would understate the cohort after archive revisions; 0.1% against a ~7% return
  scale contributes ≤2e-4 of rank variance — bounded negligible. Conservative by design:
  every included row is wedge-free whichever side moved. An exact-tied (`== 0`) sensitivity
  read is reported iff it spans ≥10 dates, never barred on.
- **Secondary (report only):** all-cohort read. Expect wedge contamination (naive ≥ model,
  per 2025 and 08-11); it is the contamination demo, not evidence.

## Bars

Statistic: per-forecast-date Spearman, date-block bootstrap 95% CI (helpers imported
from `clean_era_centre_ab` — no second implementation), minimum 10 cohort rows per date.

- **CONFIRMED:** wedge-negligible h=14 model IC CI entirely positive **and** paired
  model−naive CI entirely positive.
- **NOT CONFIRMED:** anything else. The 2025 read stands as a single-regime observation;
  no further action, no re-litigation either way.
- Magnitude is explicitly NOT compared to +0.094 (different trainers, eras, bases, and
  the 4× replay→realised precedent). The bar is sign + superiority, nothing more.

## Void conditions

- Fewer than 20 qualifying dates (≥10 cohort rows each) at run time.
- Cohort share below 15% of h=14 rows (mask degenerate).
- Any anchor, universe, pairing, or date-selection change after a rank number is seen.
- `LABEL_SMOOTHED_ANCHOR` or any anchor arm non-default at run time.
- Quoting the all-cohort cell as the read, or any DA figure without PT companions.

## Cost and instrument

Minutes, fully read-only (prod Postgres read + local archive read for the naive leg), no
retrain, no dispatch: a rank overlay reusing the `centre_vs_lastprice` panel query and the
`clean_era_centre_ab` statistics. Firewall stated plainly: the level table above comes
from prior published machinery — no rank IC has been computed on served rows as of this
writing, and the author of the instrument has not seen one.
