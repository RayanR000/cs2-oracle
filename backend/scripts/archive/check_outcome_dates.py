#!/usr/bin/env python
"""Verify the durable forecast-outcomes store keeps accumulating distinct
forecast dates toward MIN_HEADLINE_DATES=20 and is NOT being pruned.

Pulls the authoritative copy from the remote data repo
(RayanR000/cs2-oracle-data :: price-archive/ops/forecast_outcomes.parquet),
reports distinct forecast dates per horizon, and diffs against the last
snapshot so a rolling-window prune (oldest date advancing, or count
shrinking) shows up immediately.

Run it now for a baseline, then again in a few days:
    backend/venv/bin/python backend/scripts/check_outcome_dates.py

Snapshots are appended to backend/data/outcome_date_snapshots.jsonl.
Read-only against the repo; writes only the local snapshot log.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from backtest.scoring import MIN_HEADLINE_DATES

REPO = "RayanR000/cs2-oracle-data"
REMOTE_PATH = "price-archive/ops/forecast_outcomes.parquet"
TARGET_DATES = MIN_HEADLINE_DATES  # backend/backtest/scoring.py
HORIZONS = [3, 7, 14, 30]
SNAPSHOT_LOG = Path(__file__).resolve().parents[2] / "data" / "outcome_date_snapshots.jsonl"


def download_remote() -> Path:
    """Fetch the current durable parquet via the GitHub raw media type."""
    tmp = Path(tempfile.mkstemp(suffix=".parquet", prefix="outcomes_")[1])
    with tmp.open("wb") as fh:
        proc = subprocess.run(
            ["gh", "api", f"repos/{REPO}/contents/{REMOTE_PATH}?ref=main", "-H", "Accept: application/vnd.github.raw"],
            stdout=fh,
            stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        sys.exit(f"gh download failed:\n{proc.stderr.decode().strip()}")
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        sys.exit("Downloaded file is empty.")
    return tmp


def summarize(df: pd.DataFrame) -> dict:
    """Per-horizon distinct forecast dates (all + usable-label)."""
    df = df.copy()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    df["horizon_days"] = df["horizon_days"].astype(int)
    out = {}
    for h in HORIZONS:
        sub = df[df["horizon_days"] == h]
        all_dates = sorted({d.isoformat() for d in sub["forecast_date"]})
        usable = sub[sub["base_price"].notna()]
        usable_dates = sorted({d.isoformat() for d in usable["forecast_date"]})
        out[str(h)] = {
            "distinct_dates": len(all_dates),
            "usable_dates": len(usable_dates),
            "min_date": all_dates[0] if all_dates else None,
            "max_date": all_dates[-1] if all_dates else None,
            "all_dates": all_dates,
        }
    return out


def load_prev() -> dict | None:
    if not SNAPSHOT_LOG.exists():
        return None
    lines = [l for l in SNAPSHOT_LOG.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def eta(remaining_needed: int) -> str:
    """Calendar-date estimate at ~1 new date/day."""
    if remaining_needed <= 0:
        return "reached"
    return date.fromordinal(date.today().toordinal() + remaining_needed).isoformat()


def main() -> None:
    parquet = download_remote()
    try:
        df = pd.read_parquet(parquet)
    finally:
        parquet.unlink(missing_ok=True)

    summary = summarize(df)
    prev = load_prev()
    now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    print(f"Durable outcomes @ {REPO}:{REMOTE_PATH}")
    print(f"rows={len(df)}  checked_at={now} (UTC)\n")
    print(f"{'h':>4} {'dates':>6} {'usable':>7} {'need→20':>8} {'oldest':>12} {'newest':>12}  {'ETA to 20':>11}")
    for h in HORIZONS:
        s = summary[str(h)]
        need = max(0, TARGET_DATES - s["distinct_dates"])
        est = eta(need)
        print(
            f"{h:>4} {s['distinct_dates']:>6} {s['usable_dates']:>7} "
            f"{need:>8} {s['min_date']!s:>12} {s['max_date']!s:>12}  {est:>11}"
        )

    # Prune / stall detection vs previous snapshot.
    if prev:
        print(f"\nvs previous snapshot ({prev['checked_at']}):")
        alarm = False
        for h in HORIZONS:
            cur, old = summary[str(h)], prev["summary"][str(h)]
            d_dates = cur["distinct_dates"] - old["distinct_dates"]
            note = []
            if cur["min_date"] and old["min_date"] and cur["min_date"] > old["min_date"]:
                note.append(f"⚠ OLDEST ADVANCED {old['min_date']}→{cur['min_date']} (PRUNE?)")
                alarm = True
            if d_dates < 0:
                note.append(f"⚠ COUNT DROPPED by {-d_dates}")
                alarm = True
            print(
                f"  h={h:<3} distinct {old['distinct_dates']}→{cur['distinct_dates']} ({d_dates:+d})  {'  '.join(note)}"
            )
        if not alarm:
            print("  ✓ no pruning: oldest dates held, counts non-decreasing.")
    else:
        print("\n(no previous snapshot — this is the baseline; re-run in a few days.)")

    with SNAPSHOT_LOG.open("a") as fh:
        fh.write(json.dumps({"checked_at": now, "rows": len(df), "summary": summary}) + "\n")
    print(f"\nsnapshot appended → {SNAPSHOT_LOG}")


if __name__ == "__main__":
    main()
