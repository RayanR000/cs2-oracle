#!/usr/bin/env python3
"""
One-time backfill: export player counts from csmarketapi_reference.db SQLite
into player-counts-YYYY.parquet files in price-archive/, matching the schema
that append_to_parquet.py produces.

Usage:
    python scripts/backfill_player_counts_to_parquet.py
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
import numpy as np
from database import engine as pg_engine
from sqlalchemy import text


ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"
REF_DB_PATH = Path(__file__).parent.parent / "runtime" / "csmarketapi_reference.db"


def backfill_from_reference_db():
    """Read player_counts from SQLite reference DB and write Parquet files."""
    if not REF_DB_PATH.exists():
        print(f"Reference DB not found at {REF_DB_PATH}")
        return False

    import sqlite3
    conn = sqlite3.connect(str(REF_DB_PATH))
    df = pd.read_sql("SELECT timestamp, players FROM player_counts ORDER BY timestamp", conn)
    conn.close()

    print(f"Read {len(df):,} player count rows from {REF_DB_PATH.name}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    df["day"] = pd.to_datetime(df["timestamp"]).dt.date
    df["day_dt"] = pd.to_datetime(df["day"])

    daily = df.groupby("day").agg(
        mean_players=("players", "mean"),
        peak_players=("players", "max"),
        min_players=("players", "min"),
        reading_count=("players", "count"),
        last_players=("players", "last"),
    ).reset_index()

    daily["mean_players"] = daily["mean_players"].round(0).astype(np.int64)
    daily["peak_players"] = daily["peak_players"].astype(np.int64)
    daily["min_players"] = daily["min_players"].astype(np.int64)
    daily["last_players"] = daily["last_players"].astype(np.int64)

    daily["day"] = pd.to_datetime(daily["day"])

    print(f"  {len(daily)} unique days after grouping")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(daily["day"].dt.year.min(), daily["day"].dt.year.max() + 1):
        year_df = daily[daily["day"].dt.year == year].copy()
        if year_df.empty:
            continue

        out_path = ARCHIVE_DIR / f"player-counts-{year}.parquet"

        con = duckdb.connect()
        try:
            if out_path.exists():
                existing = con.sql(f"SELECT * FROM read_parquet('{out_path}')").fetchdf()
                combined = pd.concat([existing, year_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["day"], keep="last")
                combined = combined.sort_values("day")
                combined.to_parquet(out_path, index=False)
                print(f"  Appended to {out_path.name}: {len(year_df)} days, {len(combined)} total")
            else:
                year_df.to_parquet(out_path, index=False)
                print(f"  Wrote {out_path.name}: {len(year_df)} days")
        finally:
            con.close()

    print(f"\nDone. Parquet files written to {ARCHIVE_DIR}/")
    return True


def backfill_from_postgres():
    """Fallback: try reading from Postgres price_history if it has player counts."""
    try:
        with pg_engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT DATE(timestamp) AS day,
                       AVG(CAST(volume AS FLOAT)) AS mean_players
                FROM price_history
                WHERE volume IS NOT NULL AND volume > 0
                  AND source = 'steam_batch'
                GROUP BY DATE(timestamp)
                HAVING COUNT(*) > 10
                ORDER BY day
            """)).fetchall()

        if not rows:
            print("No player count data found in Postgres either.")
            return False

        df = pd.DataFrame(rows, columns=["day", "mean_players"])
        df["day"] = pd.to_datetime(df["day"])
        df["peak_players"] = df["mean_players"]
        df["min_players"] = df["mean_players"]
        df["reading_count"] = 1
        df["last_players"] = df["mean_players"]
        df["mean_players"] = df["mean_players"].round(0).astype(np.int64)

        for year in range(df["day"].dt.year.min(), df["day"].dt.year.max() + 1):
            year_df = df[df["day"].dt.year == year].copy()
            if year_df.empty:
                continue
            out_path = ARCHIVE_DIR / f"player-counts-{year}.parquet"
            year_df.to_parquet(out_path, index=False)
            print(f"  Wrote {out_path.name}: {len(year_df)} days (from Postgres)")

        return True
    except Exception as e:
        print(f"Postgres fallback failed: {e}")
        return False


def verify():
    """Quick sanity check that the forecaster can load the data."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.forecaster import ItemForecaster
    from database import SessionLocal

    db = SessionLocal()
    try:
        fc = ItemForecaster(db_session=db)
        pc = fc._fetch_player_counts()
        print(f"\nVerification: forecaster loaded {len(pc)} days of player counts")
        if not pc.empty:
            print(f"  Columns: {list(pc.columns)}")
            print(f"  Range: {pc['day'].min()} to {pc['day'].max()}")
            print(f"  Mean players: {pc['mean_players'].mean():.0f}")
        else:
            print("  WARNING: empty DataFrame — forecaster will zero-fill features!")
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("PLAYER COUNT BACKFILL — SQLite → Parquet")
    print("=" * 60)
    success = backfill_from_reference_db()
    if not success:
        print("Trying Postgres fallback...")
        backfill_from_postgres()
    verify()
