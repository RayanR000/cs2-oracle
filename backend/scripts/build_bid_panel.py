"""Build a bid sidecar from the archive's own BUFF bid source.

`aggregator_buff163_buy` is BUFF's highest_order (a bid). The universe filter
excludes it from PRICE voting, so it is never a feature today -- this surfaces
it as a (item_id, date) panel for a bid/spread feature. Reads the local archive
directly via DuckDB; writes locally only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb
import pandas as pd

from db.archive import prices_relation
from models.item_parser import archive_universe_sql_filter

BID_SOURCE = "aggregator_buff163_buy"
SIDECAR_NAME = "bid-panel.parquet"


def build_bid_panel(bid_rows: pd.DataFrame) -> pd.DataFrame:
    df = bid_rows.rename(columns={"item_slug": "item_id", "day": "date",
                                  "mean_price": "buff_bid"})
    df = df[df["buff_bid"] > 0]
    out = (df.groupby(["item_id", "date"], as_index=False)["buff_bid"].mean())
    return out[["item_id", "date", "buff_bid"]]


def main(archive_dir: Path) -> Path:
    con = duckdb.connect()
    # Invariant #1: typed reader, not a bare glob. Invariant #2: phase/phantom
    # slug filters via archive_universe_sql_filter, but source_column=None so the
    # bid-source EXCLUSION is skipped -- we want exactly the bid rows.
    rel = prices_relation(con, archive_dir=archive_dir,
                          columns=["item_slug", "day", "mean_price", "source"])
    uni = archive_universe_sql_filter(source_column=None)
    rows = con.sql(
        f"select item_slug, day, mean_price from {rel} "
        f"where source = '{BID_SOURCE}' and {uni}"
    ).df()
    panel = build_bid_panel(rows)
    out = archive_dir / SIDECAR_NAME
    panel.to_parquet(out, index=False)
    print(f"wrote {len(panel):,} rows, {panel['item_id'].nunique():,} items -> {out}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]))
