#!/usr/bin/env python3
"""Phase 4 PR 1 gate: the partitions-only forecast write equals today's dual-write.

Docs: docs/plans/2026-11-09-delete-postgres.md. Takes one forecast date from the real
`item_forecasts` partitions, rebuilds the `results` frame the forecast step hands its writer, and
writes it twice into empty temp roots:

  old  `_write_forecasts_to_db` against a SQLite stand-in for Postgres (ids read back from it),
       then the partition upsert it does after the commit.
  new  `_write_forecasts_partitions` (ids from `assign_ids`).

Every column except `id` and `created_at` must match row for row, and the new ids must be unique.
Scores nothing, writes nothing outside its temp dirs, never touches Postgres.

    venv/bin/python -m scripts.forecast_write_parity --ops-dir <price-archive/ops> --date 2026-11-05
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

IGNORED = {"id", "created_at"}
FORECAST_KEY = ["item_id", "forecast_date", "horizon_days"]


def load_night(ops_dir: Path, forecast_date: date) -> pd.DataFrame:
    """The real partition rows for one forecast date, every model_config together."""
    import duckdb

    month = ops_dir / "item_forecasts" / f"{forecast_date:%Y-%m}.parquet"
    return duckdb.sql(
        f"SELECT * FROM read_parquet('{month}') WHERE forecast_date = DATE '{forecast_date.isoformat()}' "
        "AND pg_deleted IS NOT TRUE"
    ).fetchdf()


def results_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The frame the forecast step passes its writer: one row per item with a `forecasts` dict."""
    out = []
    for slug, g in rows.groupby("item_slug", sort=True):
        first = g.iloc[0]
        forecasts = {
            int(r.horizon_days): {
                "low": r.price_low,
                "mid": r.price_mid,
                "high": r.price_high,
                "direction": r.direction,
                "confidence": r.confidence,
                "exceed_p": r.exceed_p,
                "anomaly_p": r.anomaly_p,
                "band_multiplier": r.band_multiplier,
            }
            for r in g.itertuples()
        }
        out.append(
            {
                "item_id": slug,
                "current_price": first.current_price,
                "anchor_clean": first.anchor_clean,
                "anchor_wedge_pct": first.anchor_wedge_pct,
                "anchor_date": first.forecast_date.date()
                if hasattr(first.forecast_date, "date")
                else first.forecast_date,
                "forecasts": forecasts,
            }
        )
    return pd.DataFrame(out)


def _read_partitions(root: Path) -> pd.DataFrame:
    import duckdb

    files = sorted((root / "item_forecasts").glob("*.parquet"))
    lst = ", ".join(f"'{f}'" for f in files)
    return duckdb.sql(f"SELECT * FROM read_parquet([{lst}], union_by_name = true)").fetchdf()


def run(ops_dir: Path, forecast_date: date) -> dict:
    from database import Base, Item
    from db import parquet
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from scripts import forecast_prices as fp

    rows = load_night(ops_dir, forecast_date)
    if rows.empty:
        raise SystemExit(f"no item_forecasts rows for {forecast_date} under {ops_dir}")
    # One model_config per call, as the forecast step writes it; the night may hold several.
    slug_to_id = {str(s): int(i) for s, i in rows[["item_slug", "item_id"]].drop_duplicates().itertuples(index=False)}
    id_to_slug = {v: k for k, v in slug_to_id.items()}
    version = str(rows["model_version"].iloc[0])
    configs = [c for c in rows["model_config"].dropna().unique()] or [None]

    old_root, new_root = Path(tempfile.mkdtemp(prefix="fw-old-")), Path(tempfile.mkdtemp(prefix="fw-new-"))
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    for slug, item_id in slug_to_id.items():
        db.add(Item(id=item_id, item_id=slug, name=slug, type="skin"))
    db.commit()

    saved = parquet.OPS_DIR
    try:
        for cfg in configs:
            sub = rows if cfg is None else rows[rows["model_config"] == cfg]
            results = results_from_rows(sub)
            parquet.OPS_DIR = old_root
            fp._write_forecasts_to_db(db, results, version, slug_to_id, forecast_date, model_config=cfg)
            parquet.OPS_DIR = new_root
            frows = fp._build_forecast_rows(results, version, slug_to_id, forecast_date)
            fp._write_forecasts_partitions(frows, id_to_slug, cfg)
        old, new = _read_partitions(old_root), _read_partitions(new_root)
    finally:
        parquet.OPS_DIR = saved
        engine.dispose()
    return compare(old, new)


def compare(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    cols = [c for c in old.columns if c not in IGNORED]
    a = old.sort_values(FORECAST_KEY).reset_index(drop=True)
    b = new.sort_values(FORECAST_KEY).reset_index(drop=True)
    report = {"old_rows": len(a), "new_rows": len(b), "ids_unique": bool(new["id"].is_unique), "diffs": {}}
    if list(a[FORECAST_KEY].itertuples(index=False)) != list(b[FORECAST_KEY].itertuples(index=False)):
        report["diffs"]["_keys"] = "row sets differ"
        return report
    for c in cols:
        if c not in b.columns:
            report["diffs"][c] = "missing in new"
            continue
        x, y = a[c], b[c]
        if x.dtype.kind in "fi" and y.dtype.kind in "fi":
            bad = ~(((x.astype(float) - y.astype(float)).abs() <= 1e-12) | (x.isna() & y.isna()))
        else:
            bad = ~((x == y) | (x.isna() & y.isna()))
        if bad.any():
            report["diffs"][c] = int(bad.sum())
    report["identical"] = not report["diffs"] and report["old_rows"] == report["new_rows"] and report["ids_unique"]
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ops-dir", required=True, type=Path)
    ap.add_argument("--date", required=True, type=date.fromisoformat, action="append")
    args = ap.parse_args(argv)
    ok = True
    for d in args.date:
        r = run(args.ops_dir, d)
        print(f"{d}: {r}")
        ok &= bool(r.get("identical"))
    print("IDENTICAL" if ok else "DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
