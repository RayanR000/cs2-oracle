# Pre-registration: does h=14 wedge-free ranking survive on served rows, scored on the ARCHIVE basis?

**Date:** 2026-09-13, written and committed **before any rank IC is computed on the archive basis.**
**Status:** PROPOSED — **NOT runnable today.** The power gate is measured below and fails
(13 qualifying dates vs a required 20). Expected to clear ~2026-09-27; see "Power gate".
**Follows:** `2026-09-13-clean-era-centre-preregistration.md` (the 2025 read,
`changelog/2026-09-13-clean-era-centre-measured.md`) and the VOIDed served leg
(`2026-09-13-served-centre-rank-preregistration.md`,
`changelog/2026-09-13-served-centre-rank-void.md`).

## Why this leg exists

The 2025 clean-era read found tied-cohort h=14 rank IC **+0.094 [+0.067, +0.123]**, paired
vs naive significant, placebo null, level skill negative — "ranks without level accuracy".
The changelog named exactly one follow-up: *does h=14 tied ranking survive on served 2026
dates?*, and required a new prereg because that run's PASS bar failed.

The first attempt at that follow-up VOIDed, and the reason is a design constraint this leg
must obey: **quote-basis outcomes cannot referee ranking.** Measured on 2026-08-11 (942
paired rows, same rows, swapped outcome legs): `-r1` IC is **+0.043** against archive
forwards and **-0.138** against the served quote. The serve-time quote is a lagging
median-of-3 while the voted archive price is contemporaneous, and their gap is
**+0.60 [+0.46, +0.73]** rank-correlated with trailing 1d returns — so quote-basis outcomes
hand momentum a mechanical boost and reversal a mechanical penalty. The 2025 edge is driven
by *reversal* features (`return_60d` 10/10 folds, `return_3d` 10/10,
`trend_divergence_30_60` 8/10), i.e. exactly the side that basis penalises.

So this leg rescores the same question on a contemporaneous anchor. It confirms the
**ranking** observation only. Level is settled dead (`lambda* = 0`, reproduced three times)
and is not re-tested.

## What this leg cannot do

A null here does **not** refute the 2025 read — different trainer, different regime, and
the 14–70-independent-episodes constraint binds both. A positive **ships nothing**: the
range stance and the withheld direction (`2026-09-10-served-direction-withheld.md`) are
untouched by any outcome here.

## Basis (the whole point of this leg) — fixed now

All three legs are anchored on the **contemporaneous archive voted composite**, never on the
served quote. Let `A_i(d)` be the voted price for item `i` on archive day `d`
(`prices_relation` + `archive_universe_sql_filter` + production's
`_apply_multi_source_voting`).

- **Outcome:** `A_i(d+14*) / A_i(d) - 1`.
- **Prediction:** `r_hat = predicted_price_mid / A_i(d) - 1`. Archive-anchored, so the model
  is **charged for the anchor wedge**. That is deliberate and it is why the primary cohort
  is anchor-clean; it is also what makes prediction and outcome share one basis, which is
  what the VOID note demanded.
- **Naive:** `-return_1d`, the trailing-1d log return on the same archive voted composite,
  calendar-exact consecutive days. Same definition as the 2025 read's referee.
- Rows missing any leg drop from **both** arms — pairing is on identical rows.

## Anchor rules — fixed now, before any statistic

These two rules exist because the archive has real holes and real dead days. Both were
decided from panel counts alone (a date count is not a result).

1. **Forward anchor with tolerance.** `d+14*` is the nearest **available, non-frozen**
   archive day to `d+14` within **±2 calendar days**, ties broken toward the **earlier**
   day (the shorter, more conservative horizon). The realised offset is recorded per row.
   Rationale: the working-copy archive is missing 2026-08-28, 08-30, 08-31, 09-01, 09-03,
   09-04 and 09-05, which alone costs **8 of 24** panel dates their exact `d+14` anchor.
   A pre-specified sensitivity read at **exact `d+14` only** is reported alongside.

2. **Frozen-date exclusion — whole days, never rows.** Any archive day whose
   exactly-unchanged share on the served cohort is **≥ 0.70** is barred from anchoring
   either leg. Measured on the served cohort (1,114 slugs, consecutive days only),
   2026-09-13: the baseline runs **0.08–0.19** and exactly three days blow out —
   **2026-07-16 (1.000), 2026-07-22 (1.000), 2026-08-23 (0.995)**. The next highest is
   2026-08-19 at 0.578, so the 0.70 threshold (inherited from the 2026-08-29 measurement,
   not chosen here) separates cleanly. A sensitivity read at **0.50** — which additionally
   bars 08-19 — is reported.
   **Row-level frozen drops are forbidden:** on a frozen row the forward return is measured
   off a dead base and last-price is right by construction, but dropping such rows is a
   *cohort* change that selects volatile items, and it flipped the h=3 climatology verdict
   spuriously. Whole-day exclusion is cohort-neutral; row-level is not.

## Cohort — fixed now, and why it is not a post-hoc selection

**Primary: `item_forecasts.anchor_clean == True`** (the serve-time tied mask, recorded per
forecast row).

This leg does **not** inherit the previous leg's mechanical 70%-coverage rule, and the
reason is a measurement, not a preference. The archive-vs-quote anchor gap `A_i(d)/current_price - 1`
on the h=14 panel (21,688 rows carrying an archive price) has median |gap| **0.051** overall,
and splits sharply by era:

| era | per-date median &#124;gap&#124; |
|---|---|
| 2026-07-17 .. 2026-08-09 | 0.089 – 0.240 |
| 2026-08-11 .. 2026-08-27 | 0.013 – 0.029 |

An archive-anchored prediction is only meaningful where the archive anchor and the serve
anchor agree to within roughly the return scale (~7% at h=14). Before 2026-08-11 the gap is
**larger than the signal being measured**, so those dates cannot support this basis at all —
regardless of what any stored wedge column says. `anchor_clean` coverage begins on 2026-08-11
for the same engineering reason. Restricting to it is therefore forced by the basis, and it
is fixed here, ahead of every rank statistic.

Reported as secondary, never barred on: the wedge-negligible cohort
(`|current_price/base_price - 1| <= 0.001`) and the all-cohort read (the contamination demo).

## Statistic

Per-forecast-date Spearman, minimum **10** cohort rows per date; date-block bootstrap 95% CI
(1,000 resamples, seed 42) — never a row bootstrap, since one date's items share the market
factor. Helpers are **imported** from `clean_era_centre_ab` (`per_date_ic`, `bootstrap_ci`,
`paired_delta_ci`); no second implementation.

## Bars

- **CONFIRMED:** primary-cohort model IC CI entirely positive **and** paired
  model-minus-naive CI entirely positive.
- **KILL:** at ≥ 20 qualifying dates, model IC CI spans zero **or** the paired CI is not
  entirely positive.
- **VOID:** fewer than 20 qualifying dates. **No statistic is read at all** — not reported
  as underpowered, not reported as suggestive. The previous leg VOIDed and its exploratory
  cell (13 dates, ±0.10 CIs) is the reason this bar is absolute.

No outcome changes serving.

## Power gate (measured pre-registration; a date count is not a result)

Measured 2026-09-13 against prod Postgres read-only and the working-copy archive:

- h=14 served panel after `excluded_forecast_date`: **24 dates / 22,705 rows**, spanning
  2026-07-17 .. 2026-08-27.
- `anchor_clean` is non-null on **48.2%** of h=14 panel rows and `== True` on
  **13 dates** (2026-08-11 onward).
- **13 < 20 → this leg is VOID today and must not be run.**

**When it clears.** The core chain is active again (Aggregator / Price Forecast / Backtest
Accuracy), and `item_forecasts` carries 22,144 rows/day for 2026-09-06 .. 09-13 after a
2026-08-28 .. 09-05 outage. Those resolve at h=14 on **2026-09-20 .. 09-27**, which brings
the count to **20 on or about 2026-09-27** if the chain stays green. Re-check with the gate
command below; run the leg only when it prints ≥ 20.

## Reproduce

```
# gate only — safe to run any time, prints the date count and nothing else
venv/bin/python -m scripts.archive_basis_centre_rank --gate

# the leg itself; refuses to run and exits VOID below 20 qualifying dates
venv/bin/python -m scripts.archive_basis_centre_rank \
    --archive-dir ../price-archive --out /tmp/abcr_h14.json
venv/bin/python -m pytest tests/test_archive_basis_centre_rank.py -q
```

The durable checkout (`cs2-oracle-data/price-archive`) is at 2026-08-25 locally; the working
copy carries 2026-09 through 09-08. Pass `--archive-dir` deliberately and record which was
used — the forward anchor depends on it.
