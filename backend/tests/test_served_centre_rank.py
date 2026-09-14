"""The served centre-rank overlay's pure helpers.

Derivation, pairing, qualification and bars are the referee — pinned here on
synthetic frames. No DB, no archive.
"""

import numpy as np
import pandas as pd
import pytest
from scripts.served_centre_rank import (
    attach_naive,
    derive_panel,
    evaluate_confirmation,
    qualify,
    select_primary,
    summarize,
)


def _served(n=60, seed=0, clean=True, wedge=0.0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(2, 50, size=n)
    cur = base * (1 + wedge)
    return pd.DataFrame(
        {
            "forecast_date": ["2026-08-01"] * n,
            "base_price": base,
            "actual_price": base * (1 + rng.normal(0, 0.05, size=n)),
            "current_price": cur,
            "predicted_price_mid": cur * (1 + rng.normal(0, 0.02, size=n)),
            "slug": [f"item-{i % 20}" for i in range(n)],
            "anchor_clean": [clean] * n,
        }
    )


class TestDerivePanel:
    def test_r_hat_and_realized_share_the_quote(self):
        df = derive_panel(_served())
        r = df.iloc[0]
        assert r["r_hat"] == (r["predicted_price_mid"] / r["current_price"] - 1.0)
        assert r["realized"] == (r["actual_price"] / r["current_price"] - 1.0)

    def test_wedge_cohorts(self):
        df = derive_panel(_served(wedge=0.0005))
        assert df["coh_wedge"].all() and not df["coh_exact"].any()
        df = derive_panel(_served(wedge=0.05))
        assert not df["coh_wedge"].any()

    def test_missing_current_is_cohort_unknown(self):
        sdf = _served()
        sdf.loc[0, "current_price"] = np.nan
        df = derive_panel(sdf)
        assert not df.loc[0, "coh_wedge"] and not df.loc[0, "coh_exact"]
        # Falls back to base for the centre (centre_vs_lastprice rule) —
        # still scored, never in a wedge cohort.
        assert np.isfinite(df.loc[0, "r_hat"])

    def test_null_anchor_clean_is_unknown_not_clean(self):
        sdf = _served(clean=None)
        df = derive_panel(sdf)
        assert not df["coh_clean"].any()
        assert not df["clean_known"].any()


class TestSelectPrimary:
    def test_clean_wins_at_gate(self):
        assert select_primary(0.70) == "coh_clean"
        assert select_primary(0.699) == "coh_wedge"


class TestAttachNaive:
    def test_missing_naive_drops_the_row_from_both_arms(self):
        df = derive_panel(_served(n=40))
        m = {(s, pd.Timestamp("2026-08-01").date()): 0.01 for s in df["slug"].unique()[:10]}
        paired, dropped = attach_naive(df, m)
        assert dropped == 20 and len(paired) == 20
        assert np.isfinite(paired["naive"]).all()
        assert (paired["naive"] == -0.01).all()


class TestQualify:
    def test_min_rows_gate(self):
        df = derive_panel(_served(n=60))
        df["coh_wedge"] = True
        assert qualify(df, "coh_wedge") == ["2026-08-01"]
        df["coh_wedge"] = False
        assert qualify(df, "coh_wedge") == []


class TestSummarizeAndBars:
    def test_perfect_model_reads_one(self):
        df = derive_panel(_served(n=60))
        df["coh_wedge"] = True
        df["r_hat"] = df["realized"]
        out = summarize(df, qualify(df, "coh_wedge"))
        assert out["2026-08-01"] == pytest.approx(1.0)

    def test_bars(self):
        good = {"mean": 0.05, "ci_low": 0.01, "ci_high": 0.09}
        null = {"mean": 0.01, "ci_low": -0.02, "ci_high": 0.04}
        assert evaluate_confirmation(good, good) == "CONFIRMED"
        assert evaluate_confirmation(good, null) == "NOT CONFIRMED"
        assert evaluate_confirmation(null, good) == "NOT CONFIRMED"
        assert evaluate_confirmation(None, good) == "VOID"
