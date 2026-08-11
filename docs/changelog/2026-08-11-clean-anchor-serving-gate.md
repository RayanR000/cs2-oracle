# The ranked surfaces show only clean-anchor items, 2026-08-11

`/opportunities` no longer ranks items whose anchor quote deviates from its own local
median. Per-item lookups still serve those forecasts and now say which cohort they came
from.

## The evidence, and its exact scope

On a fresh artifact trained in CI and replayed at four non-overlapping anchors, served
rank IC on the **tied** cohort is **+0.1321 / +0.1562 / +0.1747** at 3/7/14d, positive at
4 anchors of 4, and **+0.0536** at 30d (3 of 4, unconfirmed). On the **deviating** cohort
it is **−0.2014 at 0 of 4** for h=3, and indistinguishable from zero at 7/14/30 once the
2026-07-09 anchor is excluded (`2026-08-11-clean-anchor-confirmed-in-ci.md`). Ten local
anchors on a different archive and a different artifact agree
(`2026-08-11-clean-anchor-signal-replicates.md`).

This is the project's **first measured servable signal at a useful size**, and it had no
consumer.

**Rank IC is an ordering statistic.** `/opportunities` ranks by predicted return, so it is
the surface the evidence covers — hence a gate there and disclosure everywhere else.
Refusing a forecast someone asked for by name would go past what was measured.

**This is not an accuracy improvement.** Nothing about the model changed. It stops
publishing a ranking over a cohort whose measured ordering skill is zero or negative.

## What shipped

- `ItemForecaster._anchor_disclosure` → `anchor_clean` (bool) and `anchor_wedge_pct`
  (signed %) per item, on the `predict` payload.
- Migration `0022`, both columns nullable, **applied to prod 2026-08-11**. All 310,304
  existing rows are NULL.
- `api/serving_policy.py`: `meets_anchor_gate` (Python) and `anchor_clean_clause` (SQL),
  applied in `_latest_forecasts` and again in `select_opportunities` — the same
  belt-and-braces the price floor uses, and the gate runs **before** the sort and the
  `[:limit]`.
- `PredictionOut` carries both fields, so `/items/{id}/prediction` discloses the cohort.

## Three things that were easy to get wrong

**The disclosure is computed BEFORE `_serving_base_price`.** That method overwrites
`latest_rows["price"]` with the served base, and under the shipped arm the served base
*is* the smoothed median — so a mask computed one line later finds `p == S` for every item
and publishes the whole catalogue as clean. A mutation test confirms three tests fail if
the two lines are swapped.

**The gate is exact equality, not the 10% outlier test.** `predict` already computes a
deviation mask at `ANCHOR_OUTLIER_TOLERANCE`, and reusing it would have been free. It is a
different population — ~5% of items against ~2/3 — and nothing was measured on it. The
gate uses `isclose(rtol=0, atol=1e-9)`, matching `replay_serving._tied_mask`, which is
where every published figure came from. `anchor_wedge_pct` is stored beside the flag
because whether the effect is a cliff at zero or monotone in `|p/S − 1|` is **not**
measured, and the size is what lets that be read later without another serving change.

**NULL passes the gate.** A bare `column.is_(True)` would have dropped all 310,304
pre-existing rows the moment the migration landed and refilled the surface only after the
next forecast run — an outage produced by adding a column.

## The un-migrated-schema hazard

No workflow runs Alembic, and this repo's prod schema is known to run behind its
migrations — `daily_analysis` was dropped by `0015` and is still in prod, while
`alembic_version` reads `0021`. An INSERT naming a column the table lacks fails the whole
batch, so shipping this naively would have taken down the daily forecast run.

`_write_forecasts_to_db` inspects the table and drops both columns from the DB payload if
they are absent, with a WARNING naming the fix. The Parquet mirror is **not** narrowed —
`_append_parquet` widens on write, so the disclosure lands there either way.

The warning is loud on purpose. A silent skip is how this project has more than once
ended up with a green run and no data.

## What it does not do

- **It does not backfill.** Nothing stored on a historical row can reconstruct the flag:
  `current_price` is the *served* base, which under the shipped arm is already the
  smoothed median, so a backfill from the table would call every historical row clean and
  be exactly wrong. NULL is the honest value.
- **It does not touch the backtest.** Scored cohorts are unchanged, so the gate cannot
  flatter a published accuracy figure by construction. Whether the backtest should report
  the two cohorts separately is a real question and is not answered here.
- **30d is unconfirmed.** +0.0536 at 3 of 4 anchors, against a local +0.1423. The gate
  applies at every horizon anyway, because the deviating cohort has no measured signal at
  any of them — the weak leg is the *tied* claim at 30d, not the case for excluding the
  rest.
- **No interval.** Four anchors and ten anchors are replications, not power. The readable
  statistic is sign consistency.
