# 2026-09-28 — Multi-source voting moved into DuckDB

## What broke

The Monday full retrain died twice on 2026-09-28: the scheduled run `36366406155` and the
redispatch `36462984374`. Both were killed with exit 143 about 10 minutes into
"Applying multi-source voting" on the train-universe read, and the first logged "The runner
has received a shutdown signal". It was the same step both times, and the input was 12.39M
parsed rows against 11.92M on the last green retrain (`35549467741`, 2026-09-21).

## Cause

The step no longer fit the runner's memory. The voted cache misses on every run by design
(its key holds the cutoff date and the archive fingerprint), so every retrain re-votes the
full 1,460-day frame. The old path materialised every raw row in pandas and then ran
`groupby.apply` returning one `pd.Series` per multi-source group. The code comment said that
was "typically <2%" of groups. On the 2026 archive it is **1.21M of 6.58M groups (18%),
holding 52% of rows**, because ~13 aggregator feeds vote daily. Locally, on 11.1M rows, the
process peaked at 8.9 GB with DuckDB's default memory limit (80% of RAM).

Deadline this set: the models date from 09-21, and at 14 days (10-05) the nightly
predict-only runs retrain too (`forecast_prices.py`, age gate), so every run would fail.

## Change

- `_fetch_voted_price_history` keeps raw rows inside DuckDB (`TRY_CAST` replaces
  `pd.to_numeric`) and votes with `_multi_source_voting_sql`. pandas receives only the voted
  frame.
- The connection is capped at `VOTED_DUCKDB_MEMORY_LIMIT` (default `2GB`), so it spills to
  disk instead of scaling with the machine.
- `_apply_multi_source_voting` is unchanged and remains the reference: the DB path and every
  voting test use it. `tests/test_sql_voting.py` holds the two equal over randomised frames
  covering every branch, and it fails when the 2σ threshold or the sync stand-down is mutated.
- `VOTED_CACHE_VERSION` 9 → 10.

## Measured (local archive, train universe, 1,460 days, 11.1M rows → 6,584,167 voted)

| Path | DuckDB cap | Process peak RSS | Wall time |
|---|---|---|---|
| old (pandas vote) | 4 GB | 7.38 GB | 177 s |
| new | 4 GB | 6.40 GB | 5 s |
| new | 2 GB (default) | 4.60 GB | 8 s |
| new | 1 GB | 3.55 GB | 11 s |

**Equivalence: byte-identical on all 6,584,167 rows** (keys, price, volume, `n_ask_sources`).
The first SQL draft used `stddev_pop` and differed on exactly one row: Souvenir MP9 | Sand
Dashed (FT), 2026-08-10, where the outlier sits exactly 0.28 from the median and 2σ is
0.28 under `np.std` and 0.27999999999999997 under `stddev_pop`. σ is now two-pass, as numpy
computes it.

Two differences remain: the voted frame's `volume` is `float64` where the old path produced
pandas' nullable `Float64` (no missing values either way), and rows come back ordered by
`(item_id, date)`. The full suite passed (2,671).

## Not verified

The CI memory numbers are inferred, not measured. The runner doesn't log memory, and the
local figures are macOS RSS on an archive ~10% smaller than CI's. The proof is the next
`mode=full` run getting past voting.
