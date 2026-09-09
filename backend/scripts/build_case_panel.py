#!/usr/bin/env python3
"""Case-only research panel (supply consumption, drop-pool state, EV/ROI shell).

Research-note implementation for
``docs/research/2026-09-08-next-accuracy-indicators.md`` priority 4: cases move
on supply consumption (openings vs visible supply, drop-pool status, EV/ROI),
not on generic price technicals. Those drivers are undefined for ordinary
skins, so this builds a case-only panel — research output, NOT a forecaster
input. Nothing trains on ``case-panel.parquet`` until a preregistered case-only
A/B clears its gates.

Columns per (item_slug, date):
- visible_supply: max listing count across supply feeds (same max-collapse as
  the forecaster, so a feed outage reads as a small dip, not a crash)
- supply_depletion_7d/30d: fractional change vs 7/30 days ago (negative = sink)
- net_depletion_7d/30d: absolute unit change over the same windows
- drop_pool_status: active | rare | discontinued | unknown (from event-calendar
  crate columns + name heuristics; unknown — never guessed — when silent)
- case_ev / case_roi_vs_key: NaN until a key-price + content-EV feed exists.
  The columns exist so the panel schema is stable; filling them with price
  proxies would fabricate the signal under test.
- estimated_openings: NaN until an openings feed exists (same reason).

Missing supply is missing: depletion windows with no anchor read NaN, not 0.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logger = logging.getLogger("build_case_panel")

CASE_PANEL_COLUMNS = [
    "item_slug",
    "date",
    "visible_supply",
    "supply_depletion_7d",
    "supply_depletion_30d",
    "net_depletion_7d",
    "net_depletion_30d",
    "drop_pool_status",
    "case_ev",
    "case_roi_vs_key",
    "estimated_openings",
]

SIDECAR_NAME = "case-panel.parquet"


def _is_case_slug(slug: str) -> bool:
    s = slug.lower()
    if "souvenir-package" in s or "souvenir package" in s:
        return True
    if "capsule" in s or "package" in s:
        return True
    if s.endswith("-case") or s.endswith(" case") or "weapon case" in s:
        return True
    return False


def _drop_pool_status(slug: str, event_row: dict | None) -> str:
    """active | rare | discontinued | unknown. Unknown on silence — never guessed."""
    if event_row:
        for key in ("crate_discontinued", "drop_pool_removed", "discontinued"):
            if event_row.get(key):
                return "discontinued"
        for key in ("crate_rare", "rare_drop"):
            if event_row.get(key):
                return "rare"
        for key in ("crate_active", "crate_first_sale"):
            if event_row.get(key):
                return "active"
    s = slug.lower()
    if "discontinued" in s or "retired" in s:
        return "discontinued"
    return "unknown"


def build_case_panel(
    supply: pd.DataFrame,
    events: pd.DataFrame | None = None,
    universe: list[str] | None = None,
) -> pd.DataFrame:
    """Build the case-only panel from a supply frame.

    ``supply``: (item_slug, snapshot_day, listing_count). ``events``: optional
    (date, ...) crate/drop-pool columns keyed by date. ``universe``: optional
    slug allowlist; defaults to case-shaped slugs in ``supply``.
    """
    if supply.empty:
        return pd.DataFrame(columns=CASE_PANEL_COLUMNS)
    frame = supply.copy()
    frame["snapshot_day"] = pd.to_datetime(frame["snapshot_day"]).dt.date
    slugs = universe if universe is not None else sorted({s for s in frame["item_slug"].unique() if _is_case_slug(str(s))})
    frame = frame[frame["item_slug"].isin(slugs)]
    if frame.empty:
        return pd.DataFrame(columns=CASE_PANEL_COLUMNS)
    vis = (frame.groupby(["item_slug", "snapshot_day"], as_index=False)["listing_count"]
           .max().rename(columns={"snapshot_day": "date", "listing_count": "visible_supply"}))
    vis = vis.sort_values(["item_slug", "date"])
    # Date-anchored depletion: shift(window) would shift ROWS, which is wrong
    # on a sparse panel (a 7-day-apart pair is 1 row apart, not 7). Look up the
    # anchor date explicitly; a missing anchor reads NaN, never 0.
    level = {(s, d): v for s, d, v in
             zip(vis["item_slug"], vis["date"], vis["visible_supply"])}
    for window, stem in ((7, "7d"), (30, "30d")):
        anchors, nets, fracs = [], [], []
        for slug, day, cur in zip(vis["item_slug"], vis["date"], vis["visible_supply"]):
            anchor = level.get((slug, day - timedelta(days=window)))
            anchors.append(anchor)
            if anchor is None or anchor == 0:
                nets.append(float("nan"))
                fracs.append(float("nan"))
            else:
                nets.append(float(anchor - cur))
                fracs.append(float((anchor - cur) / anchor))
        vis[f"net_depletion_{stem}"] = nets
        vis[f"supply_depletion_{stem}"] = fracs
    event_by_date: dict = {}
    if events is not None and not events.empty and "date" in events.columns:
        ev = events.copy()
        ev["date"] = pd.to_datetime(ev["date"]).dt.date
        event_by_date = {d: g.iloc[0].to_dict() for d, g in ev.groupby("date")}
    vis["drop_pool_status"] = [
        _drop_pool_status(slug, event_by_date.get(d)) for slug, d in zip(vis["item_slug"], vis["date"])
    ]
    vis["case_ev"] = float("nan")
    vis["case_roi_vs_key"] = float("nan")
    vis["estimated_openings"] = float("nan")
    return vis[CASE_PANEL_COLUMNS].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-dir", type=Path,
                    default=Path(__file__).parent.parent.parent / "price-archive")
    ap.add_argument("--write", action="store_true", help="Write case-panel.parquet.")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    supply_files = sorted(args.archive_dir.glob("supply-[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"))
    if not supply_files:
        print(json.dumps({"status": "failed", "error": "no supply-*.parquet in archive"}, indent=2))
        return 1
    supply = pd.concat([pd.read_parquet(p, columns=["item_slug", "snapshot_day", "listing_count"])
                        for p in supply_files], ignore_index=True)
    events = None
    ev_path = args.archive_dir / "event-calendar.parquet"
    if ev_path.exists():
        events = pd.read_parquet(ev_path)
    panel = build_case_panel(supply, events)
    summary = {"status": "success", "case_panel_rows": len(panel),
               "distinct_cases": int(panel["item_slug"].nunique()) if len(panel) else 0,
               "drop_pool_status": panel["drop_pool_status"].value_counts().to_dict() if len(panel) else {}}
    print(json.dumps(summary, indent=2, default=str))
    if args.write:
        panel.to_parquet(args.archive_dir / SIDECAR_NAME, index=False, compression="zstd")
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
