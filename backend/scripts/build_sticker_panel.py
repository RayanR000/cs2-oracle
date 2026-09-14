#!/usr/bin/env python3
"""Sticker-only research panel (application velocity shell, substitutes, capsule state).

Research-note implementation for
``docs/research/2026-09-08-next-accuracy-indicators.md`` priority 5: sticker
consumption (applications/crafts vs visible supply, capsule sale status,
substitutes, tournament exposure) is undefined for ordinary skins, so it gets a
sticker-only panel — research output, NOT a forecaster input. Nothing trains on
``sticker-panel.parquet`` until a preregistered sticker-only A/B clears.

Application velocity comes from CSFloat applied-sticker/craft queries
(observational coverage, 500 req/day rate limit). Until that feed is wired,
``application_velocity_*`` columns are NaN — never zero-filled, since zero
applications is a real observation and absence of data is not it.

Rankings must be collection-relative, never global: ``substitute_group`` keys
the relevant collection/event so downstream scoring stays within substitutes.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logger = logging.getLogger("build_sticker_panel")

STICKER_PANEL_COLUMNS = [
    "item_slug",
    "date",
    "visible_supply",
    "application_velocity_7d",
    "application_velocity_30d",
    "craft_velocity_7d",
    "substitute_group",
    "capsule_status",
]

SIDECAR_NAME = "sticker-panel.parquet"

_EVENT_PREFIX = re.compile(
    r"^(berlin-2019|stockholm-2021|antwerp-2022|rio-2022|paris-2023|copenhagen-2024|"
    r"shanghai-2024|austin-2025|budapest-2025|colonge-2026|london-2018|boston-2018|"
    r"katowice-2019|dreamhack-[a-z0-9-]+)"
)


def _is_sticker_slug(slug: str) -> bool:
    s = slug.lower()
    if s.startswith("sticker-") or s.startswith("steam_sticker"):
        return True
    if _EVENT_PREFIX.match(s) and "souvenir-package" not in s:
        return True
    return False


def substitute_group(slug: str) -> str:
    """Collection/event scope for relative-value scoring (never global)."""
    s = slug.lower()
    for prefix in ("sticker-", "steam_sticker_", "steam_sticker-"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    m = _EVENT_PREFIX.match(s) or _EVENT_PREFIX.search(slug.lower())
    if m:
        return m.group(1)
    if "capsule" in s:
        return s.split("capsule")[0].rstrip("-_ ")
    return "other"


def _capsule_status(slug: str, event_row: dict | None) -> str:
    if event_row:
        for key in ("capsule_removed", "sale_ended", "discontinued"):
            if event_row.get(key):
                return "removed"
        for key in ("capsule_sale", "on_sale"):
            if event_row.get(key):
                return "on_sale"
    return "unknown"


def build_sticker_panel(
    supply: pd.DataFrame,
    events: pd.DataFrame | None = None,
    universe: list[str] | None = None,
) -> pd.DataFrame:
    """Build the sticker-only panel from a supply frame."""
    if supply.empty:
        return pd.DataFrame(columns=STICKER_PANEL_COLUMNS)
    frame = supply.copy()
    frame["snapshot_day"] = pd.to_datetime(frame["snapshot_day"]).dt.date
    slugs = (
        universe
        if universe is not None
        else sorted({s for s in frame["item_slug"].unique() if _is_sticker_slug(str(s))})
    )
    frame = frame[frame["item_slug"].isin(slugs)]
    if frame.empty:
        return pd.DataFrame(columns=STICKER_PANEL_COLUMNS)
    vis = (
        frame.groupby(["item_slug", "snapshot_day"], as_index=False)["listing_count"]
        .max()
        .rename(columns={"snapshot_day": "date", "listing_count": "visible_supply"})
    )
    event_by_date: dict = {}
    if events is not None and not events.empty and "date" in events.columns:
        ev = events.copy()
        ev["date"] = pd.to_datetime(ev["date"]).dt.date
        event_by_date = {d: g.iloc[0].to_dict() for d, g in ev.groupby("date")}
    vis["application_velocity_7d"] = float("nan")
    vis["application_velocity_30d"] = float("nan")
    vis["craft_velocity_7d"] = float("nan")
    vis["substitute_group"] = vis["item_slug"].map(substitute_group)
    vis["capsule_status"] = [
        _capsule_status(slug, event_by_date.get(d)) for slug, d in zip(vis["item_slug"], vis["date"])
    ]
    return vis[STICKER_PANEL_COLUMNS].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-dir", type=Path, default=Path(__file__).parent.parent.parent / "price-archive")
    ap.add_argument("--write", action="store_true", help="Write sticker-panel.parquet.")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    supply_files = sorted(args.archive_dir.glob("supply-[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"))
    if not supply_files:
        print(json.dumps({"status": "failed", "error": "no supply-*.parquet in archive"}, indent=2))
        return 1
    supply = pd.concat(
        [pd.read_parquet(p, columns=["item_slug", "snapshot_day", "listing_count"]) for p in supply_files],
        ignore_index=True,
    )
    events = None
    ev_path = args.archive_dir / "event-calendar.parquet"
    if ev_path.exists():
        events = pd.read_parquet(ev_path)
    panel = build_sticker_panel(supply, events)
    summary = {
        "status": "success",
        "sticker_panel_rows": len(panel),
        "distinct_stickers": int(panel["item_slug"].nunique()) if len(panel) else 0,
    }
    print(json.dumps(summary, indent=2, default=str))
    if args.write:
        panel.to_parquet(args.archive_dir / SIDECAR_NAME, index=False, compression="zstd")
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
