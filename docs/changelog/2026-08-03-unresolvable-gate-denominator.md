# The Unresolvable Gate Fired Every Day on 0.3% of the Cohort (2026-08-03)

The daily backtest failed at 11:21Z with:

```
Archive covers through 2026-08-01; evaluating forecasts matured on or before that date
Found 75195 mature forecasts to evaluate
Unresolvable: 225/225 (100.0% of forecasts requiring resolution)
RuntimeError: 100.0% of mature forecasts could not be resolved (cap 10.0%)
```

225 of 75,195 is **0.3%**. It was reported as 100%, and would have been reported
that way every day.

## Root cause: the denominator shrank to the failures themselves

`unresolvable_pct = n_unresolvable / n_considered`, where `n_considered` counted
only forecasts *requiring* resolution — frozen rows were excluded by design
(`already frozen` is logged separately).

That works while the cohort is fresh and breaks completely once it is frozen.
After the 2026-08-01 backfill froze the cohort, the only rows still requiring
resolution were the **same 225 chronically-unresolvable stragglers that backfill
had itself logged as "unresolvable 225 (0.4%)"**. They are by definition 100% of
themselves.

The gate's intent was right and its threshold was right. Its denominator became
degenerate the moment the freeze landed — structurally the same shape as the
maturity bug the 08-01 work caught, whose changelog observed "the daily CI
backtest would have failed this gate every single day after the branch merged."
The same sentence applied here, to the gate itself.

## Why a one-line denominator swap was wrong

The replaced comment made a legitimate argument for the old choice:

> counting the frozen majority would dilute a genuinely broken resolution rate
> to nothing

It is correct. 500 consecutive fresh failures against 75,000 frozen rows is
0.7%, under any sane cap. So the two failure modes pull opposite ways:

| | symptom | what a single ratio does |
|---|---|---|
| hypersensitivity | 225 chronic rows read as 100% | fires daily on nothing |
| dilution | 500 fresh failures read as 0.7% | misses a real regression |

**Count cannot separate them** — 225 chronic and 500 fresh failures are
indistinguishable by magnitude. A first attempt at an attempt-count floor
(`MIN_ATTEMPTS_FOR_RATE_GATE = 1000`) was written, then discarded: any threshold
between 225 and 500 is arbitrary and fragile against a drifting straggler
population, and it would have excused a genuine 500-row breakage.

## The discriminator is chronicity, not count

`backtest/resolution_gate.py` — pure, no DB/archive/clock — splits failures by
whether the archive has already moved past the target date:

- **chronic** — `coverage_end - target_date > MAX_WINDOW_SPAN_DAYS`. The archive
  covers well beyond the target and the forecast still will not resolve, so the
  data is never arriving. These re-enter `to_resolve` every run forever, because
  a forecast that cannot resolve never earns a frozen outcome. A fixed tax.
- **fresh** — target date near the coverage edge. A cluster of these is what a
  broken resolver or stalled collector looks like.

The grace window is `MAX_WINDOW_SPAN_DAYS` (7, from
`collectors.pipeline.FALLBACK_MAX_AGE_DAYS`), so "the archive moved past it"
means the same span here as everywhere else in the codebase.

Two ratios, neither needing an arbitrary floor:

| ratio | denominator | catches |
|---|---|---|
| coverage | whole mature cohort, frozen included | mass cohort shrinkage |
| fresh rate | attempts minus chronic | resolver / archive regressions |

Chronic rows are excused from the **rate** but still counted in **coverage**, so
a chronic population that grows large enough to matter fails the run rather than
being excused indefinitely (`test_a_chronic_population_large_enough_to_matter_still_fails`).

Verified against the real numbers:

| scenario | mature | attempted | fresh | chronic | result |
|---|---|---|---|---|---|
| today's run | 75,195 | 225 | 0 | 225 | **pass** (0.3% coverage, warn) |
| same day, resolver broken | 75,195 | 5,000 | 4,775 | 225 | **fail** (100% fresh rate) |

## One deliberate asymmetry

Forecasts with **no slug mapping** count as fresh regardless of target date.
That is a referential-integrity break between `item_forecasts` and `items`, not
the archive lagging — no future collection fixes it, and it must never be
excused as a chronic archive gap.

## Tests

`tests/test_unresolvable_gate_denominator.py`, 19 new, organised around the two
opposing failure modes. Load-bearing ones:

- `test_frozen_cohort_with_chronic_stragglers_passes` — the exact prod numbers
- `test_chronic_rows_do_not_create_a_fresh_rate_at_all` — 225 chronic attempts
  leave zero informative attempts, not a 100% rate
- `test_a_broken_resolver_on_real_volume_still_fails` — the dilution guard
- `test_a_broken_resolver_is_caught_even_alongside_chronic_rows`
- `test_a_small_fresh_breakage_is_not_excused_by_being_small` — 20 of 20 fresh
  failures must fail; count is explicitly not the test
- `test_coverage_uses_the_mature_cohort_not_the_attempts` — mutation guard
- `classify_chronic` boundary tests, including that missing dates are never
  chronic

One existing test was **renamed, not weakened**:
`test_gate_denominator_counts_only_forecasts_requiring_resolution` →
`test_gate_is_not_diluted_by_the_frozen_majority`. Its scenario still fails the
gate — via the fresh-rate ratio rather than the denominator it was named for —
so its docstring was corrected to describe the mechanism that now catches it.
It is the dilution half of the contract; the new file is the hypersensitivity
half.

Full suite: 355 pass.

## Related — the other incident found in the same audit

The 2026-08-02 23:58Z **aggregator** run failed with `Can't locate revision
identified by '0020_tier_prediction_accuracy_unique'`. Prod's `alembic_version`
was stamped at 0020 by the local backfill, but `origin/main` was still
`e11be1c`, whose migrations stop at 0018 — the deterministic-backtest merge
existed locally at 23:19Z and was not pushed until after 02:07Z. **The database
ran ahead of the deployed code.**

`price-forecast.yml` skips when the upstream aggregator does not succeed, so the
forecast run was *skipped* rather than failed, which is why it looks benign in
the run list. Cost: one day of price collection and one forecast batch.

Self-healed once main was pushed; no code change. Recorded because the failure
mode — migrating prod from a workstation before the migration is on the default
branch — leaves no trace in the repo and is invisible to CI until the next
scheduled run.

## Files changed

- `backend/backtest/resolution_gate.py` — new, pure
- `backend/scripts/backtest_accuracy.py` — split counters by chronicity; gate
  call site; `MAX_UNRESOLVABLE_PCT` re-exported so the threshold has one home
- `backend/tests/test_unresolvable_gate_denominator.py` — new
- `backend/tests/test_backtest_scoring.py` — one test renamed, docstring
  corrected

## Related

- `docs/changelog/2026-08-01-deterministic-backtest.md` — the freeze that made
  this denominator degenerate, and the 225 it logged as 0.4%
- `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md` — why
  the metric this gate protects still cannot be quoted
