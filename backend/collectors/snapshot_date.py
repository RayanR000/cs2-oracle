"""Which calendar day a collected snapshot belongs to.

Pure: no network, no DB. The single source of truth for the aggregator's date,
shared by the pipeline (CSV filenames) and by `append_to_parquet` (the `day`
column), because those used to read the wall clock independently and could
disagree across a midnight boundary.

WHY THIS EXISTS. The archive lost 2026-07-27, 2026-07-30 and 2026-08-03
outright, and none of those days had a failing run. The cron is `0 23 * * *`,
but Actions scheduling delay was observed pushing actual starts from 23:52 to
00:08 — so two consecutive firings could both land on the same UTC date and the
day between them got nothing. Green badge, no warning, missing day. That hole
then broke the backtest: see backtest.resolution_gate's GAP category.

The second, latent bug was worse. `pipeline.py` stamped CSV names from its own
`datetime.utcnow()` and the workflow passed `--date $(date -u +%F)` from a
*separate* clock read one step later. A pipeline finishing at 23:59:58 writes
`aggregator-snapshots-D.csv` while the append step starting at 00:00:02 looks
for `aggregator-snapshots-D+1.csv`. That had simply not rolled yet.

THE RULE. CSGOTrader regenerates its price dumps once a day at ~21:40 UTC
(verified via Last-Modified headers over 5 weeks; extra same-day runs returned
identical prices). So a snapshot's day is decided by *which dump it came from*,
not by when the runner woke up: the snapshot date is the date of the most recent
dump boundary at or before now. A delay of up to
`24 - (24 - DUMP_PUBLISHED_HOUR_UTC)` hours can no longer relabel the data.

This also fixes manual runs, which were never right either: a `workflow_dispatch`
at 14:00 is looking at yesterday's ~21:40 dump and now stamps it accordingly.

KNOWN EDGE. A run between the real dump time (~21:40) and the boundary hour
below stamps the previous day while fetching the current dump. No scheduled run
falls in that window, and the direction is safe — it never invents a day or
double-stamps one. Widening the gap between ~21:40 and the boundary trades that
risk against upstream drift; the boundary is deliberately kept a comfortable
margin after 21:40 rather than tight against it.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta

# The hour (UTC) by which the day's CSGOTrader dump is reliably published.
# Upstream regenerates at ~21:40; 22:00 leaves margin for that drift while
# staying two hours clear of midnight. Not the same number as the cron hour
# (23), and deliberately so — the cron says when we *look*, this says which
# dump we are looking at.
DUMP_PUBLISHED_HOUR_UTC = 22

# Set to a YYYY-MM-DD string to force a specific day. The escape hatch for
# re-running a specific date; also how the workflow pins one value across steps.
ENV_OVERRIDE = "AGGREGATOR_SNAPSHOT_DATE"


def resolve_snapshot_date(
    now: datetime | None = None,
    boundary_hour: int = DUMP_PUBLISHED_HOUR_UTC,
    override: str | None = None,
) -> date:
    """Return the calendar day the current snapshot belongs to.

    *override* takes precedence over the clock; when it is None the
    ``AGGREGATOR_SNAPSHOT_DATE`` environment variable is consulted. An empty or
    whitespace-only value means "not set" and falls through to the clock, since
    that is how an unset variable arrives from a shell. A malformed value raises
    rather than silently falling back — a typo in a backfill invocation must not
    quietly write to today.
    """
    if override is None:
        override = os.environ.get(ENV_OVERRIDE, "")
    if override and override.strip():
        # date.fromisoformat raises ValueError on anything malformed, which is
        # the behaviour we want: refuse loudly.
        return date.fromisoformat(override.strip())

    if now is None:
        now = datetime.utcnow()
    if now.hour >= boundary_hour:
        return now.date()
    return now.date() - timedelta(days=1)
