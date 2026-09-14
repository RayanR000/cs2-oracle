"""The set of days the archive actually holds.

`archive_max_day` answers "how far does coverage reach", which is enough to bound
maturity but blind to holes *inside* the range. The 2026-08-02/03 outage is
exactly such a hole: max day was 08-04 and looked healthy, while two days in the
middle held nothing. `classify_archive_gap` needs the interior, not the edge.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from backtest.price_resolution import archive_covered_days


def _write_archive(tmp_path, days):
    """Write a minimal prices-*.parquet holding one row per day given."""
    rows = [{"item_slug": "ak47", "day": pd.Timestamp(d), "source": "steam", "price": 1.0} for d in days]
    df = pd.DataFrame(rows)
    for ym, group in df.groupby(df["day"].dt.strftime("%Y-%m")):
        group.to_parquet(tmp_path / f"prices-{ym}.parquet", index=False)
    return tmp_path


class TestCoveredDays:
    def test_returns_every_day_present(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            d = _write_archive(Path(td), [date(2026, 8, 1), date(2026, 8, 2), date(2026, 8, 3)])
            assert archive_covered_days(d) == {
                date(2026, 8, 1),
                date(2026, 8, 2),
                date(2026, 8, 3),
            }

    def test_the_live_hole_is_visible_as_absence(self):
        """The real 2026-08 shape: 08-01 and 08-04, nothing between."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            d = _write_archive(Path(td), [date(2026, 8, 1), date(2026, 8, 4)])
            covered = archive_covered_days(d)
            assert covered == {date(2026, 8, 1), date(2026, 8, 4)}
            assert date(2026, 8, 2) not in covered
            assert date(2026, 8, 3) not in covered

    def test_spans_multiple_monthly_files(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            d = _write_archive(Path(td), [date(2026, 7, 31), date(2026, 8, 1)])
            assert archive_covered_days(d) == {date(2026, 7, 31), date(2026, 8, 1)}

    def test_duplicate_rows_collapse_to_one_day(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            d = _write_archive(Path(td), [date(2026, 8, 1), date(2026, 8, 1), date(2026, 8, 1)])
            assert archive_covered_days(d) == {date(2026, 8, 1)}


class TestMissingArchiveIsLoud:
    def test_absent_directory_raises(self):
        """Mirrors archive_max_day: an absent archive must not read as "no gaps"."""
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            archive_covered_days(Path("/nonexistent/price-archive"))

    def test_directory_without_price_files_raises(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td, pytest.raises(FileNotFoundError):
            archive_covered_days(Path(td))
