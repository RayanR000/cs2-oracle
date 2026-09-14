"""Tests for the case-only and sticker-only research panels.

Panels are research outputs, not forecaster inputs. Missing data stays NaN:
a zero depletion or zero application velocity is a real observation, absence
of data is not it.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from scripts.build_case_panel import build_case_panel  # noqa: E402
from scripts.build_sticker_panel import build_sticker_panel, substitute_group  # noqa: E402


def _supply(slugs_days: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"item_slug": s, "snapshot_day": pd.to_datetime(d).date(), "listing_count": c} for s, d, c in slugs_days]
    )


class TestCasePanel:
    def test_selects_case_slugs_and_measures_depletion(self):
        supply = _supply(
            [
                ("kilowatt-case", "2026-09-01", 1000),
                ("kilowatt-case", "2026-09-08", 900),
                ("ak-47-asiimov-minimal-wear", "2026-09-08", 50),
            ]
        )
        panel = build_case_panel(supply)
        assert panel["item_slug"].unique().tolist() == ["kilowatt-case"]
        row = panel[panel["date"] == date(2026, 9, 8)].iloc[0]
        assert row["visible_supply"] == 900
        assert row["net_depletion_7d"] == 100.0
        assert abs(row["supply_depletion_7d"] - 0.1) < 1e-9

    def test_drop_pool_unknown_on_silence(self):
        panel = build_case_panel(_supply([("kilowatt-case", "2026-09-08", 5)]))
        assert panel.iloc[0]["drop_pool_status"] == "unknown"

    def test_ev_and_openings_stay_nan(self):
        panel = build_case_panel(_supply([("kilowatt-case", "2026-09-08", 5)]))
        assert panel["case_ev"].isna().all()
        assert panel["estimated_openings"].isna().all()


class TestStickerPanel:
    def test_selects_stickers_and_scopes_substitutes(self):
        supply = _supply(
            [
                ("sticker-berlin-2019-legends-holo", "2026-09-08", 40),
                ("ak-47-asiimov-minimal-wear", "2026-09-08", 50),
            ]
        )
        panel = build_sticker_panel(supply)
        assert panel["item_slug"].unique().tolist() == ["sticker-berlin-2019-legends-holo"]
        assert panel.iloc[0]["substitute_group"] == "berlin-2019"

    def test_application_velocity_stays_nan_until_csfloat(self):
        panel = build_sticker_panel(_supply([("sticker-foo", "2026-09-08", 7)]))
        assert panel["application_velocity_7d"].isna().all()
        assert panel["craft_velocity_7d"].isna().all()

    def test_substitute_group_falls_back(self):
        assert substitute_group("sticker-foo") == "other"
