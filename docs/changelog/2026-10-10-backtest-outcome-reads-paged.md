# 2026-10-10 — The backtest reads forecast_outcomes in pages, not IN lists

Performance review 2026-10-08, §1 (after the DuckDB vote, #101).

## Cost

After #101, Backtest Accuracy run `38016621198` (day 10-10) took 12m50s. The script itself
ran 02:21:26–02:32:35, and most of that was four reads of `forecast_outcomes` restricted to
the ~1.0M mature forecast ids, each sent as 900-id `IN` lists, ~1,000 round trips per read:

| Step | Time | What it read |
|---|---|---|
| frozen check | 97 s | every frozen outcome as an ORM object |
| store | 89 s | an existence re-check over ~930K ids, all of them already frozen |
| verdict refresh | 112 s | every mature outcome, 900 ids at a time |
| scoring records | 125 s | every mature outcome joined to `item_forecasts`, 900 ids at a time |

The vote passes took ~25 s each, and candidates 65 s.

## Change

- **Frozen check:** one `SELECT forecast_id FROM forecast_outcomes`, intersected with the
  mature ids in Python. The ORM rows it used to load existed only to be handed to the freeze.
- **Store:** the freeze gets only the new outcomes. It still re-checks each id for an existing
  row and writes only those with none, so it is still the single gate on the table; it no
  longer re-checks ~930K ids that `to_resolve` had already excluded.
- **Verdict refresh and scoring records:** a keyset walk over `forecast_outcomes.id` in pages
  of `OUTCOME_PAGE` (50,000), with the id restriction applied to each page in Python. Every
  outcome has an `item_forecasts` row (foreign key), so this keeps exactly the rows the `IN`
  lists did.
- **`update_bias` is off by default.** CI never restores `saved_models`, so the fit wrote
  `bias_corrections.json` to a runner that was then discarded ("No saved models found" in the
  same run). `--update-bias` still runs it, and `run_forecast_local.sh` now passes it, so local
  runs are unchanged.

## Order

Scoring records now come back in `forecast_outcomes.id` order. The `IN`-list read had no
`ORDER BY`, so its order was whatever Postgres returned per chunk. `bootstrap_ci` (the iid
`dir_ci_*` / `mae_ci_*` bounds) indexes records by position, so its draws depend on that
order; the date-block bootstrap and the PT test sort by date and do not.

## Measured against prod (read-only transaction, cutoff 2026-10-09, from a laptop)

Old (`main`) and new readers on the same 1,015,850 mature forecasts:

| Read | old | new |
|---|---|---|
| frozen ids (old path timed on ids only, not the ORM rows CI loaded) | 31.5–43.3 s | 6.5–6.8 s |
| scoring records | 80.8–91.2 s | 33.3–46.8 s |

- Frozen ids: the same 932,560 on both paths.
- Scoring records: the same multiset in all 4 `(horizon, identity)` groups; only the order
  differs.
- `_score_groups` output, all 37 result rows: every point metric, the PT test, the
  date-clustered CI and every count are identical. **Only the four iid bootstrap bounds move**
  — `directional_accuracy_ci_lower/upper` (max 0.89 pp) and `mae_ci_lower/upper` (max $1.32,
  on the high tiers), in 30–33 rows. That is the order dependence above, not a data change:
  the old bounds were already a function of an unspecified row order. Expect those four stored
  columns in `prediction_accuracy` to step once on the first run after merge; nothing reads
  them as a headline (invariant 4 quotes PT and the clustered interval).

Expected in CI: the frozen check, store and the two outcome reads fall from ~7 min to ~1.5 min,
putting the daily Backtest at ~6 min. Read the next run's log timestamps to confirm.

## Not done

From §1: candidate resolution still loads its three tables whole as ORM objects (65 s and
growing ~44K candidates a day), and the smaller items (`penny_metrics` recompute, chunked
`bootstrap_ci` draws, row-by-row `_upsert_accuracy` / candidate writes, the serial evidence
reports) are untouched.
