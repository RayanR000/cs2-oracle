# Drop the case panel, sticker panel and Reddit event detector

2026-09-30. Item 13 of `research/2026-09-28-next-steps.md` ("wire them or drop them"): dropped.
All three were built on 09-08 (`changelog/2026-09-08-accuracy-panels-and-orderbook-accumulation.md`)
and never put in a workflow.

## Why not wire them

The item's premise was that nothing accumulates while they sit unwired. For the two panels that is
false. Both are derived views over `supply-YYYY-MM.parquet`, which the Aggregator already archives
daily, so any past date can be rebuilt on demand. Run against the live archive on 2026-09-30
(`supply-2026-08/09` and `event-calendar.parquet` from the data repo):

| Panel | Rows | What was populated |
|---|---|---|
| case | 18,885 (409 slugs, 47 dates, 08-07..09-29) | `visible_supply` 100%, 7d depletion 72%, 30d 46%. `drop_pool_status` 100% `unknown`. `case_ev`, `case_roi_vs_key`, `estimated_openings` 0% |
| sticker | **0** | nothing |

Neither builder had ever run on real data:

- **The sticker selector matched nothing.** It keys on `sticker-…` / `steam_sticker…` slugs, but
  the archive uses market hash names (`Sticker | jkaem (Gold) | Paris 2023`), and 12,185 of
  32,834 supply names in September are stickers.
- **The event join was dead in both panels.** `index_events_by_date` reads a `date` column, but
  `event-calendar.parquet` keys on `day`. It also has none of the `crate_discontinued` /
  `capsule_removed` style columns the status functions read. Every status reads `unknown`.
- **The case selector counted sticker capsules and souvenir packages as cases.** It matches
  `capsule` / `package` by name.
- **Both scripts failed to import from the CLI** after the move into `scripts/archive/`. Their
  `sys.path` insert resolved to `scripts/`, not `backend/`. The tests still passed because pytest
  sets the path.

The panels' only real signal is visible supply, which is already archived. Every other column needs a
feed that doesn't exist for free (openings, key-price/content EV, CSFloat application velocity). A
case-only or sticker-only study should build its panel from the archive when it is preregistered,
against the real name format.

**Reddit event detector.** It needs `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET` or
`REDDIT_BEARER_TOKEN`, and the repo has neither secret (`gh secret list`, 09-30). The 09-08 note
sequenced it after the market-flow panels, which are now dropped. Generic sentiment is already on
the do-not-run list, and the previous Reddit collector was deleted on 08-01 after runner IPs got
403s and it stored 0 rows. Unlike the panels, its engagement snapshots can't be backfilled. That
loss is accepted: with no credentials it would collect nothing anyway.

## Deleted

- `scripts/archive/build_case_panel.py`, `scripts/archive/build_sticker_panel.py`,
  `scripts/_panel_common.py`, `tests/test_case_sticker_panels.py`.
- `collectors/reddit_events.py`, `scripts/archive/run_reddit_events.py`, `tests/test_reddit_events.py`.
- `run_task.ROW_COUNT_FIELDS`: `reddit_event_rows`, `case_panel_rows`, `sticker_panel_rows`.
- `check_sidecar_continuity.WATCHED["reddit-events"]` (its missing-table test now uses `volume`).
- `docs/operations.md`: the paragraph telling you to run `scripts/run_supply_scraper.py` and
  `run_task.py reddit_social` locally. Neither exists.

Kept: the `social_mentions` table and migration `0018`. Dropping a table needs its own migration,
and the table already holds 0 rows.

Recover any of it with `git show afc0aa6:backend/<path>`.
