"""Shared price estimator for the forecast backtest.

Both legs of ``actual_ret`` — the forecast-date base and the target-date
actual — go through :func:`resolve_anchors`. That is the entire determinism
guarantee: the same function, the same window, the same source on both sides.

Before 2026-08-01 the base leg was ``item_forecasts.current_price`` (a
3-observation median written at serving time) and the actual leg was a raw
single-day voted price read fresh from the archive on every run. Differencing
two different estimators against a 0.5% flat band is what let the same 5,512
forecasts score 61.76% one day and 33.74% the next.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from collectors.pipeline import FALLBACK_MAX_AGE_DAYS
from db.archive import price_files, prices_relation

# Mirrors ItemForecaster.predict()'s tail(3) median (forecaster.py:3513).
SMOOTH_WINDOW = 3

# Every selected observation must lie within this many calendar days of the
# ANCHOR — not merely within this many days of each other. Measuring the span
# between the selected observations alone leaves the gap from the newest
# observation to the anchor unbounded, so an anchor arbitrarily far past the end
# of the archive resolves to a carried-forward price stamped as the anchor's own
# value. That is the price-laundering shape commit db5bddb removed from the
# historical collector fallback; it must not live in the scorer either.
#
# Derived from collectors.pipeline.FALLBACK_MAX_AGE_DAYS to ensure a single
# staleness convention across the codebase. If an operator overrides
# FALLBACK_MAX_AGE_DAYS via environment variable, both sides (backtest and
# production) will use the same value.
MAX_WINDOW_SPAN_DAYS = FALLBACK_MAX_AGE_DAYS


def archive_max_day(archive_dir: Path) -> date:
    """Return the newest day present in the archive's ``prices-*.parquet``.

    This is the resolvability horizon. A forecast whose target date falls past
    it cannot be scored, no matter how long ago it matured by the calendar, so
    callers use this rather than ``date.today()`` to decide which forecasts are
    evaluable. Read from Parquet column statistics, so the cost is metadata
    only — no row scan.

    Raises FileNotFoundError on a missing or empty archive, matching
    :func:`load_voted_prices`: an absent archive must never read as "nothing is
    evaluable yet", which is a green run that scores zero forecasts.
    """
    import duckdb

    con = duckdb.connect()
    try:
        relation = prices_relation(con, archive_dir, columns=["day"])
        row = con.sql(f"SELECT max(day) FROM {relation}").fetchone()
    finally:
        con.close()

    # An aggregate query always returns exactly one row (NULL when the archive
    # has no dated rows); the None check below is the empty-archive guard.
    assert row is not None
    newest = row[0]

    if newest is None:
        raise FileNotFoundError(f"price archive at {archive_dir} contains no dated price rows")

    return newest if isinstance(newest, date) else pd.Timestamp(newest).date()


def archive_covered_days(archive_dir: Path) -> set[date]:
    """Return every distinct day present in the archive's ``prices-*.parquet``.

    :func:`archive_max_day` gives the coverage *edge*, which bounds maturity but
    is blind to holes inside the range. The 2026-08-02/03 collection outage is
    exactly that shape: the max day was 08-04 and looked healthy while two days
    in the middle held nothing. Interior coverage is what
    :func:`backtest.resolution_gate.classify_archive_gap` needs to tell "the
    collector missed a day" apart from "this forecast failed for some other
    reason".

    Distinct days only, so the cost is one grouped scan of a single column rather
    than a row scan. Raises FileNotFoundError on a missing or empty archive for
    the same reason ``archive_max_day`` does: an absent archive must never read
    as "no gaps", which would silently excuse every unresolvable forecast.
    """
    import duckdb

    con = duckdb.connect()
    try:
        relation = prices_relation(con, archive_dir, columns=["day"])
        rows = con.sql(f"SELECT DISTINCT day FROM {relation} WHERE day IS NOT NULL").fetchall()
    finally:
        con.close()

    return {r[0] if isinstance(r[0], date) else pd.Timestamp(r[0]).date() for r in rows}


@dataclass(frozen=True)
class Resolution:
    """A resolved anchor plus the window that produced it.

    The observation dates are what let a *caller* — the only layer that knows
    two anchors form a leg pair — reject a pair whose actual leg carries no
    information the base leg did not already have.
    ``oldest_observation`` is the one that check must use: a pair is safe only
    when the two windows are DISJOINT. Testing ``newest_observation`` admits a
    pair whose windows overlap in 2 of 3 slots, and a median decided by the
    shared observations scores an exact 0.0 return no matter what the one
    unshared observation did.

    This class stays leg-agnostic: it reports what supported the estimate, and
    makes no judgement about which leg it is.
    """

    price: float
    oldest_observation: date
    newest_observation: date


def resolve_anchors(
    voted: pd.DataFrame,
    anchors: set[tuple[str, date]],
    window: int = SMOOTH_WINDOW,
    max_span_days: int = MAX_WINDOW_SPAN_DAYS,
) -> dict[tuple[str, date], Resolution]:
    """Median of the last ``window`` observed prices at or before each anchor.

    ``voted`` must already be voted to one row per item-day, with columns
    ``item_id`` (slug), ``date``, ``price``.

    Anchors that cannot be resolved — no observation at or before the anchor, or
    any selected observation more than ``max_span_days`` before the anchor — are
    omitted from the result. Callers must treat a missing key as a dropped
    forecast rather than substituting a fallback, which would reintroduce the
    asymmetry this function exists to remove.

    The anchor-relative bound subsumes a between-observations bound: every
    selected observation is at or before the anchor, so the newest is too, and
    ``selected[-1] - selected[0] <= anchor - selected[0]``. If the oldest
    selected observation is within ``max_span_days`` of the anchor, the
    observations are necessarily within ``max_span_days`` of each other.
    """
    if voted.empty or not anchors:
        return {}

    by_item: dict[str, list[tuple[date, float]]] = {}
    for slug, group in voted.groupby("item_id", sort=False):
        ordered = group.sort_values("date")
        by_item[slug] = list(zip(ordered["date"], ordered["price"]))

    resolved: dict[tuple[str, date], Resolution] = {}
    for slug, anchor in anchors:
        observations = by_item.get(slug)
        if not observations:
            continue

        # Last `window` observations at or before the anchor.
        selected = [(d, p) for d, p in observations if d <= anchor][-window:]
        if not selected:
            continue

        # Measured from the ANCHOR, not across the selected window. This bounds
        # both the scatter of the observations and their staleness relative to
        # the date being resolved.
        if (anchor - selected[0][0]).days > max_span_days:
            continue

        prices = sorted(p for _, p in selected)
        mid = len(prices) // 2
        if len(prices) % 2:
            price = float(prices[mid])
        else:
            price = float((prices[mid - 1] + prices[mid]) / 2)

        resolved[(slug, anchor)] = Resolution(
            price=price,
            oldest_observation=selected[0][0],
            newest_observation=selected[-1][0],
        )

    return resolved


def load_voted_prices(
    archive_dir: Path,
    slugs: list[str],
    min_date: date,
    max_date: date,
    max_span_days: int = MAX_WINDOW_SPAN_DAYS,
) -> pd.DataFrame:
    """Load voted daily prices from the Parquet archive.

    Reaches back ``max_span_days`` before ``min_date``: resolving an anchor
    needs the observations preceding it, not just the anchor's own day.

    Raises FileNotFoundError when the archive is absent. The previous
    behaviour — warn and return {} — produced a green run that evaluated zero
    forecasts, which is exactly the silent-success shape commit 324cfff was
    written to eliminate.

    Votes in DuckDB (``ItemForecaster._multi_source_voting_sql``), one query
    per item-hash chunk, as the forecaster's archive read has since
    2026-09-28. The pandas vote cost 96% of this function (36.7s of 38.4s on
    2.4M raw rows) and ran once per (horizon, model) group, ~5.7 min each in
    run 37877462648. A group never spans chunks, so the frame does not depend
    on the chunk count.
    """
    import os

    import duckdb
    from models.forecaster import ItemForecaster

    if not slugs:
        # Checked after price_files() below would have run, so a caller passing
        # no slugs against a missing archive still gets the FileNotFoundError
        # rather than an innocuous empty frame.
        price_files(archive_dir)
        return pd.DataFrame(columns=["item_id", "date", "price"])

    lookback_start = min_date - pd.to_timedelta(int(max_span_days), unit="D")

    con = duckdb.connect(config={"memory_limit": os.environ.get("VOTED_DUCKDB_MEMORY_LIMIT", "2GB")})
    try:
        relation = prices_relation(con, archive_dir, columns=["item_slug", "day", "mean_price", "source", "volume"])

        con.register("wanted_slugs", pd.DataFrame({"item_slug": slugs}))
        # TRY_CAST and the NULL/NaN drop are pd.to_numeric(errors="coerce") +
        # dropna: some Parquet years store mean_price/volume as VARCHAR.
        rows_sql = f"""
            SELECT * FROM (
                SELECT s.item_slug AS item_id, CAST(s.day AS DATE) AS date,
                       TRY_CAST(s.mean_price AS DOUBLE) AS price,
                       TRY_CAST(s.volume AS DOUBLE) AS volume, s.source
                FROM {relation} s
                JOIN wanted_slugs w ON w.item_slug = s.item_slug
                WHERE s.day BETWEEN DATE '{lookback_start}' AND DATE '{max_date}'
            ) WHERE price IS NOT NULL AND NOT isnan(price)
        """
        n_chunks = max(1, int(os.environ.get("VOTED_CHUNKS", "8")))
        parts = []
        for k in range(n_chunks):
            chunk_sql = f"SELECT * FROM ({rows_sql}) WHERE hash(item_id) % {n_chunks} = {k}"
            parts.append(
                con.sql(f"SELECT item_id, date, price FROM ({ItemForecaster._multi_source_voting_sql(chunk_sql)})").df()
            )
    finally:
        con.close()

    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=["item_id", "date", "price"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values(["item_id", "date"]).reset_index(drop=True)
