"""The clean-era centre A/B's pure helpers.

Rank IC, the date-block bootstrap, the paired delta and the prereg bars are
the whole referee — if any drifts, the KILL/PASS read is meaningless, so all
are pinned here. No DB, no archive: everything under test is a pure function
of small arrays.
"""

import numpy as np
import pandas as pd
import pytest
from scripts.archive.clean_era_centre_ab import (
    _spearman,
    bootstrap_ci,
    evaluate_bars,
    fold_boundaries,
    paired_delta_ci,
    per_date_ic,
)


class TestSpearman:
    def test_perfect_rank_agreement_is_one(self):
        x = np.arange(50, dtype=float)
        assert _spearman(x, x) == pytest.approx(1.0)

    def test_perfect_inversion_is_minus_one(self):
        x = np.arange(50, dtype=float)
        assert _spearman(x, -x) == pytest.approx(-1.0)

    def test_constant_prediction_is_unscorable(self):
        rng = np.random.default_rng(0)
        assert _spearman(np.ones(30), rng.normal(size=30)) is None

    def test_too_few_rows_is_unscorable(self):
        assert _spearman(np.arange(9, dtype=float), np.arange(9, dtype=float)) is None

    def test_nans_are_dropped_not_fatal(self):
        x = np.arange(50, dtype=float)
        y = x.copy()
        y[::7] = np.nan
        assert _spearman(x, y) == 1.0


class TestPerDateIc:
    def test_groups_by_date_and_skips_thin_dates(self):
        dates = ["2025-01-01"] * 20 + ["2025-01-02"] * 5 + ["2025-01-03"] * 20
        x = np.arange(45, dtype=float)
        out = per_date_ic(dates, x, x)
        assert set(out) == {"2025-01-01", "2025-01-03"}
        assert out["2025-01-01"] == pytest.approx(1.0)


class TestBootstrapCi:
    def test_degenerate_input_gives_degenerate_interval(self):
        vals = {f"d{i}": 0.05 for i in range(30)}
        c = bootstrap_ci(vals)
        assert c["n_dates"] == 30
        assert c["ci_low"] <= 0.05 <= c["ci_high"]
        assert c["mean"] == 0.05

    def test_seeded_determinism(self):
        rng = np.random.default_rng(3)
        vals = {f"d{i}": float(v) for i, v in enumerate(rng.normal(0.04, 0.02, size=60))}
        assert bootstrap_ci(vals) == bootstrap_ci(vals)

    def test_single_date_is_unscorable(self):
        assert bootstrap_ci({"only": 0.1}) is None

    def test_clear_signal_excludes_zero(self):
        rng = np.random.default_rng(4)
        vals = {f"d{i}": float(v) for i, v in enumerate(rng.normal(0.10, 0.02, size=100))}
        c = bootstrap_ci(vals)
        assert c["ci_low"] > 0

    def test_paired_delta_of_identical_series_spans_zero(self):
        rng = np.random.default_rng(5)
        a = {f"d{i}": float(v) for i, v in enumerate(rng.normal(size=40))}
        c = paired_delta_ci(a, dict(a))
        assert c["mean"] == 0.0
        assert c["ci_low"] <= 0 <= c["ci_high"]

    def test_paired_delta_uses_shared_dates_only(self):
        a = {f"d{i}": 0.1 for i in range(10)}
        b = {f"d{i}": 0.0 for i in range(5, 15)}
        c = paired_delta_ci(a, b)
        assert c["n_dates"] == 5
        assert c["mean"] == 0.1


class TestFoldBoundaries:
    def test_full_year_grid(self):
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        bounds = fold_boundaries(dates)
        assert bounds[0] == pd.Timestamp("2025-04-15")
        assert bounds[-1] == pd.Timestamp("2025-12-23")
        assert len(bounds) == 13
        # Run-time target availability (d+h in-panel) trims the tail per
        # horizon; the splitter itself stays horizon-blind.

    def test_short_windows_skipped_not_fatal(self):
        dates = list(pd.date_range("2025-01-01", "2025-05-01", freq="D"))
        bounds = fold_boundaries(dates)
        assert bounds and bounds[0] == pd.Timestamp("2025-04-15")


def _ci(mean, lo, hi):
    return {"n_dates": 50, "mean": mean, "ci_low": lo, "ci_high": hi}


class TestEvaluateBars:
    def _summ(self, m, nv, p, pb, skill):
        return {"model": _ci(*m), "naive": _ci(*nv), "paired": _ci(*p), "placebo": _ci(*pb), "mae_skill": skill}

    def test_pass_needs_everything(self):
        s = self._summ((0.06, 0.02, 0.10), (0.03, 0.01, 0.05), (0.03, 0.01, 0.05), (0.001, -0.01, 0.012), 0.02)
        assert evaluate_bars({14: s}) == {14: "PASS"}

    def test_model_null_is_kill(self):
        s = self._summ((0.01, -0.02, 0.04), (0.03, 0.01, 0.05), (-0.02, -0.05, 0.01), (0.0, -0.01, 0.01), 0.01)
        assert evaluate_bars({14: s}) == {14: "KILL"}

    def test_failing_to_beat_naive_is_kill(self):
        s = self._summ((0.05, 0.02, 0.08), (0.04, 0.02, 0.06), (0.005, -0.01, 0.02), (0.0, -0.01, 0.01), 0.01)
        assert evaluate_bars({14: s}) == {14: "KILL"}

    def test_negative_level_is_kill(self):
        s = self._summ((0.06, 0.02, 0.10), (0.03, 0.01, 0.05), (0.03, 0.01, 0.05), (0.0, -0.01, 0.01), -0.01)
        assert evaluate_bars({14: s}) == {14: "KILL"}

    def test_hot_placebo_is_void_not_pass(self):
        s = self._summ((0.06, 0.02, 0.10), (0.03, 0.01, 0.05), (0.03, 0.01, 0.05), (0.05, 0.02, 0.08), 0.02)
        assert evaluate_bars({14: s}) == {14: "VOID"}

    def test_null_naive_is_underpowered_not_kill(self):
        s = self._summ((0.01, -0.02, 0.04), (0.005, -0.02, 0.03), (0.005, -0.02, 0.03), (0.0, -0.01, 0.01), 0.0)
        assert evaluate_bars({14: s}) == {14: "UNDERPOWERED"}

    def test_missing_ci_is_underpowered(self):
        s = {"model": None, "naive": None, "paired": None, "placebo": None, "mae_skill": None}
        assert evaluate_bars({30: s}) == {30: "UNDERPOWERED"}


class TestLabelBasisGuard:
    def test_smoothed_label_refuses_to_run(self, monkeypatch):
        import scripts.archive.clean_era_centre_ab as cec

        monkeypatch.setenv("LABEL_SMOOTHED_ANCHOR", "1")
        try:
            cec.run(pd.DataFrame(), [])
        except SystemExit as exc:
            assert "LABEL_SMOOTHED_ANCHOR" in str(exc)
        else:
            raise AssertionError("run() must refuse the smoothed label basis")
