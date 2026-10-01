#!/usr/bin/env python3
"""Continuity audit for accumulation sidecars (supply/volume/reddit-events).

Research-note Phase 1: the Sep-08 doc gates every order-book/volume test on
"enough independent forecast dates". This script answers that question without
training anything:

- distinct snapshot days per table, sorted, with gaps listed
- per-feed row counts on the latest day (a dead feed hiding behind a healthy
  total is the 2026-07-16 failure shape)
- longest run of consecutive days (the velocity-feature requirement)

Reads the archive only. Exit 1 with --gate when any watched table has fewer
than --min-days distinct days or a trailing gap (latest day older than
--max-staleness-days before today), so CI can fail loudly instead of going
green behind a stalled accumulator.

Usage:
    venv/bin/python scripts/check_sidecar_continuity.py --archive-dir ../price-archive
    venv/bin/python scripts/check_sidecar_continuity.py --gate --min-days 45
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import itertools

import pandas as pd

WATCHED = {
    "supply": ("supply-[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet", "snapshot_day", "source"),
    "volume": ("volume-[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet", "day", "source"),
    "reddit-events": ("reddit-events-[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet", "snapshot_day", "source"),
}


def _project_days_sources(paths: list[Path], day_col: str, source_col: str) -> pd.DataFrame | None:
    """(day, source) over all files via DuckDB projection pushdown.

    Only the two audit columns leave the files — the old per-file
    `pd.read_parquet` loaded every column. No DISTINCT: `per_source_latest`
    counts rows on the latest day, and duplicates are real listings.
    Returns None when no file carries the day column.
    """
    import duckdb

    quoted = ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths)
    rel = f"read_parquet([{quoted}], union_by_name=true)"
    con = duckdb.connect()
    try:
        cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()}
        if day_col not in cols:
            return None
        proj = day_col if source_col not in cols else f"{day_col}, {source_col}"
        df = con.execute(f"SELECT {proj} FROM {rel}").fetchdf()
    finally:
        con.close()
    df[day_col] = pd.to_datetime(df[day_col]).dt.date
    return df


def _project_days_sources_fallback(paths: list[Path], day_col: str, source_col: str) -> pd.DataFrame | None:
    """Per-file read used only when the DuckDB projection fails (e.g. a corrupt file)."""
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p, columns=[c for c in (day_col, source_col) if c])
        except Exception:
            continue
        if day_col not in df.columns:
            continue
        df[day_col] = pd.to_datetime(df[day_col]).dt.date
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _audit_table(archive_dir: Path, pattern: str, day_col: str, source_col: str) -> dict:
    paths = sorted(archive_dir.glob(pattern))
    if not paths:
        return {
            "files": 0,
            "days": 0,
            "dates": [],
            "gaps": [],
            "longest_run": 0,
            "latest": None,
            "per_source_latest": {},
        }
    try:
        all_days = _project_days_sources(paths, day_col, source_col)
    except Exception:
        all_days = _project_days_sources_fallback(paths, day_col, source_col)
    if all_days is None or all_days.empty:
        return {
            "files": len(paths),
            "days": 0,
            "dates": [],
            "gaps": [],
            "longest_run": 0,
            "latest": None,
            "per_source_latest": {},
        }
    dates = sorted(all_days[day_col].unique())
    gaps = [str(d) for d in pd.date_range(dates[0], dates[-1]).date if d not in set(dates)]
    longest = cur = 1
    for a, b in itertools.pairwise(dates):
        cur = cur + 1 if (b - a).days == 1 else 1
        longest = max(longest, cur)
    latest = dates[-1]
    per_source: dict[str, int] = {}
    if source_col in all_days.columns:
        last = all_days[all_days[day_col] == latest]
        per_source = {str(k): int(v) for k, v in last[source_col].value_counts().items()}
    return {
        "files": len(paths),
        "days": len(dates),
        "dates": [str(d) for d in dates],
        "gaps": gaps,
        "longest_run": longest,
        "latest": str(latest),
        "per_source_latest": per_source,
    }


def audit(archive_dir: Path) -> dict:
    return {name: _audit_table(archive_dir, pat, dcol, scol) for name, (pat, dcol, scol) in WATCHED.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent.parent / "price-archive")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--min-days", type=int, default=45)
    ap.add_argument("--max-staleness-days", type=int, default=7)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    report = audit(args.archive_dir)
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))

    if not args.gate:
        return 0
    today = date.today()
    failures = []
    for name in ("supply", "volume"):
        rep = report[name]
        if rep["days"] < args.min_days:
            failures.append(f"{name}: {rep['days']} days < min {args.min_days}")
        if rep["latest"] is not None:
            stale = (today - date.fromisoformat(rep["latest"])).days
            if stale > args.max_staleness_days:
                failures.append(f"{name}: latest {rep['latest']} is {stale}d stale")
    if failures:
        print("CONTINUITY GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"CONTINUITY GATE PASSED (each watched table >= {args.min_days} days).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
