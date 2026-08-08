# The event calendar is scheduled, and the correlation job has events again

**Date:** 2026-08-08
**Context:** `docs/changelog/2026-08-06-date-level-exogenous-ingest.md`
**Consumer:** `.github/workflows/event-correlation-analysis.yml` (Sun 04:00 UTC)

## The gap

`ingest_steam_news.py` was a one-shot. **No `ingest_*` script was scheduled at
all.** Meanwhile `event_correlation_analysis.py` runs weekly against the `events`
DB table, loaded from `data/cs2_events.json`, whose newest entry is
**2026-05-10** — so from roughly 2026-08-08 its 90-day window is empty and every
Sunday returns `no_events_in_window`. A correct status, and a permanently empty
report.

## Why "point the analyser at `event-calendar.parquet`" is not the fix

The two are different shapes, and the mismatch is deliberate on both sides.

`event-calendar.parquet` is **date-level**: one row per day, counts and
`days_since_*`, *no event identity and no titles*. It is a feature table for a
date-level model, and its docstring is explicit that no classification is applied
because neither candidate classifier validated.

`event_correlation_analysis.py` is **per-event**: it measures a price window
around one announcement and writes `event_impacts` rows whose `event_id` is a
real FK to `events.id`. Reading a Parquet file would either break that FK or
require synthesising ids nothing else agrees on.

So the calendar cannot feed the analyser — the identity it needs was never in
that table.

## What landed

| Path | Change |
|---|---|
| `scripts/ingest_steam_news.py` | `news_rows()` + `event-news.parquet` — the per-event companion (`gid`, `day`, `published_at`, `feed_type`, `is_valve`, `title`, `url`) |
| `scripts/sync_events_from_news.py` | **new** — upserts Valve announcements into `events` |
| `tests/test_sync_events_from_news.py` | **new** — 14 tests |
| `scripts/event_correlation_analysis.py` | the `no_events_in_window` warning named a JSON file that is no longer the source |
| `.github/workflows/event-correlation-analysis.yml` | `Refresh the CS2 event calendar`, before the analysis and inside the publish window |

`fetch_news` already cached the full items and `news_events()` discarded
everything but day and `feed_type`, so the per-event rows cost one extra
`to_parquet` and no extra request.

## Decisions

**Weekly, not daily.** The analysis is the only consumer and it runs weekly. The
feed is ~2s for four pages, so daily would be affordable but would refresh a
table nothing reads in between. Move it into the aggregator when something reads
the calendar daily.

**Valve announcements only** (`feed_type == 1`). Syndicated press stays out for
the same reason it has its own column upstream: a games-press article is a
reaction to the market as often as a cause, and mixing a reactive series into a
causal analysis manufactures correlations.

**One `type`, no classification.** Every synced row is `type = "update"`. Both
candidate case-release classifiers were measured on 2026-08-06 and neither
survived (33% recall from text; 14/14 disagreement against ByMykel
`first_sale_date`, median 38 days apart). A wrong `type` would partition
`event_patterns` by a meaningless label.

**Idempotent on `gid`, carried in the description suffix.** `events` has no gid
column and adding one is a migration this does not need. Matching on
`(timestamp, description)` would insert a duplicate every time Steam edits a
headline — there is a test for exactly that.

**`continue-on-error`.** A Steam blip must not fail the analysis; it would run
against the calendar it already has. The cost is that an empty window now has two
causes, so the warning names the step to check.

## Verified

Against the real cached feed: **1,678 events, 397 from Valve, 2012-03-16 →
2026-08-03**, and **51 Valve announcements in 2026** — comfortably inside the
90-day window, so the Sunday job has something to correlate. Sync exercised
against SQLite, never prod. 1,563 tests pass.

## Not claimed

No lift. `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` is unchanged and
nothing here touches the served model. The research is explicit that **every
large move was unannounced** — the Oct-2025 trade-up, the Dec-2024 rare-pool
removal, Trade Protection, the bot purge — so this is *contemporaneous detection*
and regime-break dating, never a forward-looking feature. The calendar measured
null at item level precisely because it is a clock there; the date-level frame
(MDE ~0.3pp) is where it was ever going to be measurable, and that experiment has
not been run.

## Open

- `event-calendar.parquet` still has **no consumer**. This wires the per-event
  table; the date panel remains unread.
- The MDE gate for a date-level design has not been re-derived. Run
  `scripts/compute_mde.py` before believing anything off either table.
