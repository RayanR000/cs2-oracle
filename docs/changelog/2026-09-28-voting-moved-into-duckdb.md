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

## Follow-up (same day): the first version failed in CI

The post-merge `mode=full` dispatch (`36470822456`, on `3cc38cd`) failed in 1m39s with
DuckDB's own `Out of Memory Error: could not allocate block of size 256.0 KiB (1.8 GiB/1.8
GiB used)` inside the vote. Locally the same query finished even at a 1GB cap, on CI's exact
input (12,389,757 rows from a clone of the durable archive) with 2 threads. **It reproduced
exactly only with spilling disabled (`temp_directory=''`)**, so the runner was not spilling.
Why is not established; the working directory is writable. The fix removes the dependency
instead of explaining it:

- **Chunked by item hash.** One query per `hash(item_id) % VOTED_CHUNKS` (default 8),
  streamed from Parquet with no temp table. A group never spans chunks, so the frame cannot
  depend on the count. With spilling disabled, 2 threads and CI's input, it runs under a
  **1GB** cap and peaks at **2.8GB process-wide** (old pandas path: 7.4GB), in ~15s.
- **Exact 2σ ties are kept** (`VOTE_TIE_RTOL = 1e-9`, both implementations). Chunking
  exposed that the SQL vote was **non-deterministic**: the 2GB and 1GB runs disagreed on a
  row, because the std's last ulp depends on summation order and cent-rounded prints sit
  exactly on 2σ. The two-pass σ above did not fix that (it only matched numpy by one
  ordering's luck), so it was reverted to `stddev_pop`. Output is now identical across chunk
  counts 1/3/8 and caps 1/2/8GB. Against the old pandas frame, **17 of 6,584,167 rows**
  change: all ties numpy's rounding dropped, all **sub-$1** penny items ($0.08–0.35), below
  the $1 training and serving floor.
- `VOTED_CACHE_VERSION` 10 → 11.

The "byte-identical" claim above is superseded by this: identical except those 17 ties.
