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

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from scripts._panel_common import (
    collapse_visible_supply,
    emit_panel_result,
    fail_no_supply,
    index_events_by_date,
    load_events,
    load_supply,
    normalise_supply_frame,
    panel_arg_parser,
)

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
    frame = normalise_supply_frame(supply)
    slugs = (
        universe
        if universe is not None
        else sorted({s for s in frame["item_slug"].unique() if _is_sticker_slug(str(s))})
    )
    frame = frame[frame["item_slug"].isin(slugs)]
    if frame.empty:
        return pd.DataFrame(columns=STICKER_PANEL_COLUMNS)
    vis = collapse_visible_supply(frame)
    event_by_date = index_events_by_date(events)
    vis["application_velocity_7d"] = float("nan")
    vis["application_velocity_30d"] = float("nan")
    vis["craft_velocity_7d"] = float("nan")
    vis["substitute_group"] = vis["item_slug"].map(substitute_group)
    vis["capsule_status"] = [
        _capsule_status(slug, event_by_date.get(d)) for slug, d in zip(vis["item_slug"], vis["date"])
    ]
    return vis[STICKER_PANEL_COLUMNS].reset_index(drop=True)


def main() -> int:
    ap = panel_arg_parser(__doc__, write_help="Write sticker-panel.parquet.")
    args = ap.parse_args()

    supply = load_supply(args.archive_dir)
    if supply is None:
        return fail_no_supply()
    events = load_events(args.archive_dir)
    panel = build_sticker_panel(supply, events)
    summary = {
        "status": "success",
        "sticker_panel_rows": len(panel),
        "distinct_stickers": int(panel["item_slug"].nunique()) if len(panel) else 0,
    }
    return emit_panel_result(
        panel,
        summary,
        archive_dir=args.archive_dir,
        sidecar_name=SIDECAR_NAME,
        write=args.write,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
