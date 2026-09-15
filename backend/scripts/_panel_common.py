#!/usr/bin/env python3
"""Shared loader/orchestrator for the case-only and sticker-only research panels.

`build_case_panel.py` and `build_sticker_panel.py` read the same two inputs
(the monthly supply files plus the event calendar), collapse listings to a
visible-supply frame the same way, index events by date the same way, and emit
their result (stdout summary, `--write`, `--json-out`) the same way. Only the
slug selection and the panel-specific columns differ — those stay in the
scripts. Anything extracted here is byte-identical in both callers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SUPPLY_GLOB = "supply-[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"
SUPPLY_COLUMNS = ["item_slug", "snapshot_day", "listing_count"]
EVENT_CALENDAR_NAME = "event-calendar.parquet"
NO_SUPPLY_ERROR = "no supply-*.parquet in archive"


def panel_arg_parser(description: str | None, write_help: str | None = None) -> argparse.ArgumentParser:
    """The CLI both panel builders share: archive dir, --write, --json-out."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--archive-dir", type=Path, default=Path(__file__).parent.parent.parent / "price-archive")
    ap.add_argument("--write", action="store_true", help=write_help)
    ap.add_argument("--json-out", type=Path, default=None)
    return ap


def load_supply(archive_dir: Path) -> pd.DataFrame | None:
    """Concat of the monthly supply files, or None when there are none."""
    supply_files = sorted(archive_dir.glob(SUPPLY_GLOB))
    if not supply_files:
        return None
    return pd.concat(
        [pd.read_parquet(p, columns=SUPPLY_COLUMNS) for p in supply_files],
        ignore_index=True,
    )


def load_events(archive_dir: Path) -> pd.DataFrame | None:
    """The event calendar, or None when it has not been built."""
    ev_path = archive_dir / EVENT_CALENDAR_NAME
    if ev_path.exists():
        return pd.read_parquet(ev_path)
    return None


def fail_no_supply() -> int:
    """The stdout failure both builders emit when the archive has no supply files."""
    print(json.dumps({"status": "failed", "error": NO_SUPPLY_ERROR}, indent=2))
    return 1


def normalise_supply_frame(supply: pd.DataFrame) -> pd.DataFrame:
    """Copy with `snapshot_day` as dates — the shared panel input frame."""
    frame = supply.copy()
    frame["snapshot_day"] = pd.to_datetime(frame["snapshot_day"]).dt.date
    return frame


def collapse_visible_supply(frame: pd.DataFrame) -> pd.DataFrame:
    """Max listing count per (item, day) — the shared `visible_supply` basis.

    Max-collapse, so a feed outage reads as a small dip, not a crash.
    """
    return (
        frame.groupby(["item_slug", "snapshot_day"], as_index=False)["listing_count"]
        .max()
        .rename(columns={"snapshot_day": "date", "listing_count": "visible_supply"})
    )


def index_events_by_date(events: pd.DataFrame | None) -> dict:
    """First event row per date, keyed by date — `{}` when there are no events."""
    if events is not None and not events.empty and "date" in events.columns:
        ev = events.copy()
        ev["date"] = pd.to_datetime(ev["date"]).dt.date
        return {d: g.iloc[0].to_dict() for d, g in ev.groupby("date")}
    return {}


def emit_panel_result(
    panel: pd.DataFrame,
    summary: dict,
    *,
    archive_dir: Path,
    sidecar_name: str,
    write: bool,
    json_out: Path | None,
) -> int:
    """Stdout summary, `--write` sidecar, `--json-out` — the shared epilogue."""
    print(json.dumps(summary, indent=2, default=str))
    if write:
        panel.to_parquet(archive_dir / sidecar_name, index=False, compression="zstd")
    if json_out:
        json_out.write_text(json.dumps(summary, indent=2, default=str))
    return 0
