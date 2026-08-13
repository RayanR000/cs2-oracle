import sys
from pathlib import Path
from datetime import date
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.build_bid_panel import build_bid_panel
from scripts.build_stattrak_panel import build_stattrak_panel
from scripts.build_all_sidecars import _count_sidecars, SIDECAR_NAMES


def test_every_sidecar_has_the_join_key():
    # Contract: every sidecar exposes exactly (item_id, date) + its feed cols.
    bid = build_bid_panel(pd.DataFrame({
        "item_slug": ["AK | X"], "day": [date(2026, 8, 1)], "mean_price": [8.0]}))
    assert {"item_id", "date"} <= set(bid.columns)
    st = build_stattrak_panel(pd.DataFrame({
        "item_id": ["StatTrak™ AK | X", "AK | X"],
        "date": [date(2026, 8, 1), date(2026, 8, 1)], "price": [16.0, 8.0]}))
    assert {"item_id", "date"} <= set(st.columns)


def test_count_sidecars_reports_all_four_by_known_filename(tmp_path):
    # Regression: the old `*-panel.parquet` glob silently dropped
    # supply-history.parquet (no "-panel" suffix) from the returned dict even
    # though the file wrote correctly. Counting by the four known SIDECAR_NAME
    # constants must include it.
    assert "supply-history.parquet" in SIDECAR_NAMES
    for name in SIDECAR_NAMES:
        pd.DataFrame({"item_id": ["AK | X"], "date": [date(2026, 8, 1)],
                      "value": [1.0]}).to_parquet(tmp_path / name, index=False)

    counts = _count_sidecars(tmp_path)

    assert set(counts) == set(SIDECAR_NAMES)
    assert counts["supply-history.parquet"] == 1
    assert all(v == 1 for v in counts.values())
