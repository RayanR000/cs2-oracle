# 2026-08-07 — Friction-conditioned tier scoring

**Step 3 of `docs/research/2026-08-07-next-steps.md`.** Spec:
`docs/specs/2026-08-07-friction-conditioned-tier-scoring-design.md`. Plan:
`docs/plans/2026-08-07-friction-conditioned-tier-scoring.md`.

The scorer now says whether a forecast implies a trade, and it no longer reports one accuracy
number across populations whose bid–ask spreads differ by a factor of seven. Full suite green
at **1,298 tests**.

## What landed

**Six price bands.** `price_tier` gained a cut at $1000. Tier 4 used to merge the
10.8%-spread (`$50–500`) and 5.2%-spread (`$1000+`) cohorts — the two most different liquidity
populations in the market, on the n = 22,449 BUFF163 measurement.

**This is a series discontinuity, not a refinement.** A row in `prediction_accuracy` with
`price_tier == 4` written before 2026-08-07 means `≥$100`; after it, `$100–1000`. Nothing
migrates the old rows and nothing can — the tier is all that was stored. Tiers 0–3 are
unchanged, so `HEADLINE_MIN_TIER`, `MIN_SERVED_PRICE_USD` and every
`>= HEADLINE_MIN_TIER` comparison in `models/forecaster.py` are untouched, and
`tests/test_serving_policy.py` passed without edits — which is the guard that the served
population is still the population the headline describes.

**`backtest/friction.py`** — round-trip costs (CSFloat/DMarket 2.0%, Skinport 8.7%, Steam
16.1%) and a tier→spread table. `actionable_threshold(tier, venue)` is their sum: **37.5% at
tier 0 down to 7.2% at tier 5** at CSFloat.

**`backtest/actionable.py`** — `ActionableDA(v,h) = P(sign(r_act) = sign(r̂) | |r̂| > RT_v + s_i)`,
scoped to h ∈ {14, 30}. Four numbers together on every scored row, flat keys:
`actionable_share_pct`, `actionable_da`, `actionable_e_net_pct`, and PT on the subset under an
`actionable_pt_*` prefix.

**Floor sweep.** `FLOOR_SWEEP = {-1: $1, -2: $5, -3: $20}` as stored `prediction_accuracy`
rows, for the reason `HEADLINE_TIER` is stored: a headline that exists only in a run's console
output cannot be audited or recomputed. `HEADLINE_TIER` stays `-1` and stays `≥$1`, so
`/accuracy/headline` and the homepage placard are unchanged. The `PRICE_TIER_QUERY` bound
widened from `ge=-1, le=4` to `ge=-3, le=5` — the old bound would have returned 422 on rows
that exist.

## Two decisions worth recording

**The spread table is measured at the wrong cuts, and says so.** The bands are `<$1`,
`$1–10`, `$10–50`, `$50–500`, `$1000+`; `price_tier`'s cuts are 1/5/20/100/1000. They do not
align. Each tier borrows the band whose geometric midpoint is nearest its own in log price,
and `SPREAD_SOURCE_BAND` records which one — so tiers 2 and 3 share `$10–50`'s 17.3%. A test
reproduces the mapping rule independently of the values, because a re-pointed band would move
every actionable number without failing anything. **Do not cite these as measured per tier.**

**`floor_records` filters in tier space, not on `base_price`.** Every floor in the sweep is a
`price_tier` cut, so the two are equivalent — but filtering on tiers keeps each floor cohort
exactly a *union of bands*, which is what the partition tests use to catch double-counting,
and it keeps `headline_records` selecting the identical population it always has. A floor that
is not a band edge would silently round down to the band below;
`test_every_sweep_floor_is_a_band_edge` is what stops one being added.

## Deliberately not done

- **The staleness axis is 2 buckets, not the review's 4.** Every row already carries the
  `actual_price == base_price` split (`directional_accuracy_unchanged` / `_moved` /
  `unchanged_pct`), which is a real staleness partition, and with six bands that is a 6 × 2
  grid. The quartiles need `stale_run_days` — a run-length scan of the archive, which is
  **step 6's** deliverable. No quartile was invented here.
- **`s_i` is a tier median, not the per-item BUFF spread the review specifies.** The honest
  version needs a `buff_spread_rel` frozen onto `forecast_outcomes` at resolution time; the
  bid feed starts 2026-07-11, so it would be NULL on almost every stored outcome. Deferred
  rather than approximated per item.
- **No new frozen column, migration or `--reresolve`.** The whole change reads only columns
  already on `forecast_outcomes`, which is what keeps `backtest/scoring.py` pure and
  `--rescore` archive-free.
- **`MIN_SERVED_PRICE_USD` did not move.** The sweep measures where the headline stabilises;
  raising the serving floor is a decision that follows the measurement, and the spread
  evidence (35.5% sub-$1) already argues the honest floor is above $1.
- **Nothing was re-scored.** A `--rescore` populates the new keys on existing rows at no
  archive cost, but running it is an operational step, not part of this change.

## What to expect from the first production run

**Failure, on `n_actionable` first.** The review says so and calls it the publishable internal
result. At CSFloat the model must predict a **>7.2% move at tier 5 and >23.1% at tier 1** for
a call to be actionable at all, against a model whose predictions are mostly single-digit
percentages.

There is also **no production verdict yet**: every live cohort still spans 1–2 forecast dates,
so `actionable_pt_verdict` will read `insufficient_dates` exactly as `pt_verdict` does, and
`MIN_FORECAST_DATES = 20` is unchanged. The deliverable of this change is the instrument, not
a number.

One consequence to watch: the walkforward gate now threads its horizon into `fold_records`, so
`actionable_*` is live there at h=14 and h=30. That harness still has **no purge and no
embargo** (step 5), so its actionable numbers inherit the same boundary-overlap inflation as
its DA.
