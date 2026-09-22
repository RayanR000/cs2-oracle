#!/usr/bin/env python3
"""
Publish CSMarketAPI parquet files from the local price-archive into
the cs2-oracle-data repo, then push with the same orphan-commit
pattern CI uses.

Two modes:
  --dry-run   (default) Show what would be copied/merged, don't touch anything.
  --apply              Do it: copy, merge, orphan-commit, force-push.

Steps:
  1. git -C <data-repo> pull to get the latest.
  2. For each local prices-*.parquet that has csmarketapi rows:
     - If the file doesn't exist in the data repo → copy it.
     - If it does exist → merge via DuckDB (dedup on item_slug+day+source,
       first ingested_at wins).
  3. Orphan-commit + force-push to main.

Usage:
    python scripts/publish_csmarketapi_to_archive.py --dry-run
    python scripts/publish_csmarketapi_to_archive.py --apply
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

LOCAL_ARCHIVE = Path(__file__).resolve().parent.parent.parent / "price-archive"
DATA_REPO = Path(__file__).resolve().parent.parent.parent.parent / "cs2-oracle-data"
PROD_ARCHIVE = DATA_REPO / "price-archive"

DEDUP_KEYS = ["item_slug", "day", "source"]


def has_csmarketapi_rows(path: Path) -> bool:
    try:
        df = pq.read_table(str(path), columns=["source"]).to_pandas()
        return df["source"].str.contains("csmarket", na=False).any()
    except Exception:
        return False


def merge_parquet(src: Path, dst: Path):
    """Merge src into dst, deduplicating on DEDUP_KEYS. First ingested_at wins."""
    con = duckdb.connect()
    try:
        src_df = pq.read_table(str(src)).to_pandas()
        dst_df = pq.read_table(str(dst)).to_pandas()

        for col in ("day",):
            if col in src_df.columns:
                src_df[col] = pd.to_datetime(src_df[col])
            if col in dst_df.columns:
                dst_df[col] = pd.to_datetime(dst_df[col])

        combined = pd.concat([dst_df, src_df], ignore_index=True)

        if "ingested_at" in combined.columns:
            combined["ingested_at"] = (
                combined.groupby(DEDUP_KEYS, dropna=False)["ingested_at"]
                .transform("min")
            )

        combined = combined.drop_duplicates(subset=DEDUP_KEYS, keep="last")

        cols = [c for c in ["item_slug", "day", "source", "mean_price", "volume", "ingested_at"] if c in combined.columns]
        extra = [c for c in combined.columns if c not in cols]
        ordered = cols + extra
        combined = combined[ordered]

        projection = ", ".join(
            f'CAST("{c}" AS DATE) AS "{c}"' if c == "day" else f'"{c}"'
            for c in ordered
        )
        tmp = dst.with_suffix(".parquet.tmp")
        con.register("_merged", combined)
        con.sql(f"COPY (SELECT {projection} FROM _merged) TO '{tmp}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
        os.replace(tmp, dst)
        con.unregister("_merged")
        tmp_path = Path(tmp)
        if tmp_path.exists():
            tmp_path.unlink()

        return len(src_df), len(dst_df), len(combined)
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually copy/merge and push")
    parser.add_argument("--no-push", action="store_true", help="Copy/merge but don't push")
    args = parser.parse_args()
    dry_run = not args.apply

    if not LOCAL_ARCHIVE.exists():
        print(f"ERROR: local archive not found at {LOCAL_ARCHIVE}")
        sys.exit(1)
    if not DATA_REPO.exists():
        print(f"ERROR: data repo not found at {DATA_REPO}")
        sys.exit(1)

    mode = "DRY RUN" if dry_run else "APPLY"
    print(f"=== Publish CSMarketAPI to archive ({mode}) ===")
    print(f"  Local: {LOCAL_ARCHIVE}")
    print(f"  Prod:  {PROD_ARCHIVE}")
    print()

    if not dry_run:
        print("Syncing data repo to remote main...")
        subprocess.run(["git", "-C", str(DATA_REPO), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(DATA_REPO), "checkout", "main"], check=True)
        subprocess.run(
            ["git", "-C", str(DATA_REPO), "reset", "--hard", "origin/main"],
            check=True,
        )
        print()

    local_files = sorted(f for f in os.listdir(LOCAL_ARCHIVE)
                         if f.startswith("prices-") and f.endswith(".parquet"))

    to_copy = []
    to_merge = []

    print("Scanning local parquets for CSMarketAPI data...")
    for f in local_files:
        local_path = LOCAL_ARCHIVE / f
        prod_path = PROD_ARCHIVE / f
        if not has_csmarketapi_rows(local_path):
            continue
        if prod_path.exists():
            local_size = local_path.stat().st_size
            prod_size = prod_path.stat().st_size
            if local_size == prod_size:
                continue
            to_merge.append(f)
        else:
            to_copy.append(f)

    print(f"\n  {len(to_copy)} files to COPY (new)")
    print(f"  {len(to_merge)} files to MERGE (exists in prod)")
    print()

    if to_copy:
        print("── COPY ──")
        for f in to_copy:
            src = LOCAL_ARCHIVE / f
            dst = PROD_ARCHIVE / f
            size = src.stat().st_size
            print(f"  {f:40s}  {size:>10,} bytes")
            if not dry_run:
                shutil.copy2(src, dst)
        print()

    if to_merge:
        print("── MERGE ──")
        for f in to_merge:
            src = LOCAL_ARCHIVE / f
            dst = PROD_ARCHIVE / f
            ls = src.stat().st_size
            ps = dst.stat().st_size
            print(f"  {f:40s}  local={ls:>10,}  prod={ps:>10,}")
            if not dry_run:
                n_src, n_dst, n_out = merge_parquet(src, dst)
                print(f"    → merged: {n_src} local + {n_dst} prod = {n_out} combined")
        print()

    if dry_run:
        print("Dry run complete. Re-run with --apply to execute.")
        return

    if not to_copy and not to_merge:
        print("Nothing to publish.")
        return

    if args.no_push:
        print("Files written. Skipping push (--no-push).")
        return

    print("── PUBLISH (orphan commit + force-push) ──")
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    # Delete stale "flat" branch if it exists from a prior run
    subprocess.run(
        ["git", "-C", str(DATA_REPO), "branch", "-D", "flat"],
        capture_output=True,
    )

    steps = [
        ("checkout --orphan flat",
         ["git", "-C", str(DATA_REPO), "checkout", "--orphan", "flat"]),
        ("add -A",
         ["git", "-C", str(DATA_REPO), "add", "-A"]),
        ("commit",
         ["git", "-C", str(DATA_REPO), "commit", "-m",
          f"archive: add csmarketapi backfill {today}"]),
        ("push --force origin HEAD:main",
         ["git", "-C", str(DATA_REPO), "push", "--force", "origin", "HEAD:main"]),
    ]
    for label, cmd in steps:
        print(f"  $ git {label}")
        subprocess.run(cmd, check=True)

    # Return to main so the local checkout tracks remote
    subprocess.run(
        ["git", "-C", str(DATA_REPO), "checkout", "main"],
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(DATA_REPO), "reset", "--hard", "origin/main"],
        capture_output=True,
    )

    print(f"\nDone. {len(to_copy)} copied, {len(to_merge)} merged, pushed to main.")


if __name__ == "__main__":
    main()
