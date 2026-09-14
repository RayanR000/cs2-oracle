"""`q_hat` must be calibrated in the basis the band is served and scored in.

The defect this pins, measured 2026-08-12 on 19,917 prod outcomes (>=$1,
``lgbm-v3%``, 2026-07-19 excluded): the served band covered **87.2 / 91.8 /
90.6 / 89.0%** at 3/7/14/30d against ``NOMINAL_COVERAGE = 0.80``. To land on
nominal the half-width had to shrink to **0.72 / 0.61 / 0.73 / 0.77**.

Root cause. ``prepare_targets`` builds the training label as
``(P[d+h] - p_raw[d]) / p_raw[d]``, but ``predict`` quotes every item from
``_smoothed_anchor_prices``' median and ``backtest/price_resolution.resolve_anchors``
scores against the same statistic. So the residual production is judged on is
``P/S - 1 - r_hat`` while ``q_hat`` was fitted on ``P/p_raw - 1 - r_hat``. The
anchor deviation ``p[d]/S[d]`` inflates the second, so ``q_hat`` comes out too
wide for the basis it is served in.

It is not a quiet-dates artifact. Measured against the dispersion of the
calibration set's own 9 out-of-fold validation windows (2023-02-26 → 2026-07-21),
the scored forecast dates have median ``rel_cal`` of **1.01 / 0.96 / 1.07 / 0.97**
at 3/7/14/30d — ordinary — and at h=14 they are *more* dispersed than the
calibration basis, which predicts under-coverage where over-coverage is observed.

This is the DENOMINATOR half of what
``docs/changelog/2026-08-11-conformal-centre-follows-serving.md`` fixed for the
CENTRE, and it is NOT the ``LABEL_SMOOTHED_ANCHOR`` arm, which moves the
training label and was refuted.

See docs/changelog/2026-08-12-conformal-basis-follows-serving.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from models.conformal import ALPHA, NOMINAL_COVERAGE, band, calibrate
from models.forecaster import ANCHOR_TIED_COL, ItemForecaster, calibration_target_col


def _forecaster(tmp_path):
    """Never the default model_dir — save paths there clobber production
    artifacts, which are gitignored and unrecoverable."""
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


class TestTheMechanism:
    """Pure: a calibration/serving basis mismatch over-covers, on its own."""

    @staticmethod
    def _coverage(anchor_deviation_sd: float, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        n = 60_000
        sigma = rng.uniform(0.05, 0.5, size=n)
        # True return off the SMOOTHED anchor, heteroscedastic in sigma.
        r = rng.normal(scale=sigma * 0.18, size=n)
        # The raw quote sits `k` away from its own local median.
        k = np.clip(1.0 + rng.normal(scale=anchor_deviation_sd, size=n), 0.2, 5.0)

        raw_basis = ((1.0 + r) / k - 1.0) * 100  # what q_hat is fitted on
        served_basis = r * 100  # what the band is scored on

        q_hat = calibrate(raw_basis, sigma, ALPHA)
        low, high = band(np.zeros(n), sigma, q_hat)
        return float(((served_basis >= low) & (served_basis <= high)).mean())

    def test_matched_bases_hit_nominal(self, tmp_path):
        """The control. With no anchor deviation the two bases are the same
        column and split conformal delivers its guarantee."""
        assert self._coverage(0.0) == pytest.approx(NOMINAL_COVERAGE, abs=0.01)

    def test_a_basis_mismatch_over_covers(self, tmp_path):
        """And it over-covers monotonically in the size of the deviation, which
        is why this failure mode is invisible to a test that draws calibration
        and evaluation residuals from one distribution — as both coverage tests
        in test_conformal.py do."""
        mild = self._coverage(0.05)
        severe = self._coverage(0.10)
        assert mild > NOMINAL_COVERAGE + 0.05
        assert severe > mild


def _frame():
    """Three random-walk series, long enough for a 3-day target.

    A walk rather than anything hand-built: a flat or repeating series is
    voided outright by the frozen-price-run rule (`LABEL_MAX_STALE_RUN_DAYS`),
    and a monotone one is never tied. This yields both cohorts with usable
    labels — 14 rows whose raw quote equals its own 3-observation median and
    37 where it does not.
    """
    rng = np.random.default_rng(7)
    days = pd.date_range("2026-01-01", periods=20, freq="D").date
    rows = []
    for item in ("a", "b", "c"):
        price = 10.0
        for d in days:
            price = round(price * float(np.exp(rng.normal(0, 0.05))), 4)
            rows.append({"item_id": item, "date": d, "price": price})
    return pd.DataFrame(rows)


class TestPrepareTargets:
    def test_emits_a_calibration_column(self, tmp_path):
        df = _forecaster(tmp_path).prepare_targets(_frame(), horizon=3)
        assert calibration_target_col(3) in df.columns

    def test_the_two_bases_agree_exactly_on_a_tied_anchor(self, tmp_path):
        """`anchor_is_tied` is the cohort where `p[d] == S[d]`, so the two
        denominators are the same number and the columns must be identical.
        This is the negative control for the test below."""
        df = _forecaster(tmp_path).prepare_targets(_frame(), horizon=3)
        tied = df[df[ANCHOR_TIED_COL] & df["target_return_3d"].notna()]
        assert len(tied) >= 10
        assert np.allclose(tied["target_return_3d"], tied[calibration_target_col(3)])

    def test_the_two_bases_differ_on_every_deviating_anchor(self, tmp_path):
        """The whole point: wherever the raw quote is off its own median, a
        return divided by it is not the return the band is scored on."""
        df = _forecaster(tmp_path).prepare_targets(_frame(), horizon=3)
        dev = df[~df[ANCHOR_TIED_COL] & df["target_return_3d"].notna()]
        assert len(dev) >= 10
        assert not np.any(np.isclose(dev["target_return_3d"], dev[calibration_target_col(3)]))

    def test_a_voided_label_voids_the_calibration_column_too(self, tmp_path):
        """A void means the return is fabricated, which is a property of the
        series and not of the denominator. Leaving these in would fit q_hat on
        exactly the artifacts the label rules exist to remove."""
        df = _forecaster(tmp_path).prepare_targets(_frame(), horizon=3)
        voided = df["target_return_3d"].isna()
        assert voided.any()
        assert df.loc[voided, calibration_target_col(3)].isna().all()


class TestConformalRecords:
    def test_residual_is_measured_against_the_served_basis(self, tmp_path):
        fc = _forecaster(tmp_path)
        records = fc._conformal_records(
            mid_ret=[1.0],
            actual_ret=[9.0],
            sigma=[0.1],
            current_price=[10.0],
            residual_actual_ret=[4.0],
        )
        assert records[0]["residual_pct"] == pytest.approx(4.0 - 1.0)

    def test_the_training_label_still_drives_the_confidence_inputs(self, tmp_path):
        """`hit` and `change_pct` feed `_calibrate_confidence`, and moving them
        under cover of this change would alter a second thing silently."""
        fc = _forecaster(tmp_path)
        served = fc._conformal_records(
            mid_ret=[1.0],
            actual_ret=[9.0],
            sigma=[0.1],
            current_price=[10.0],
            residual_actual_ret=[-9.0],
        )
        control = fc._conformal_records(
            mid_ret=[1.0],
            actual_ret=[9.0],
            sigma=[0.1],
            current_price=[10.0],
        )
        assert served[0]["hit"] == control[0]["hit"]
        assert served[0]["change_pct"] == pytest.approx(control[0]["change_pct"])

    def test_the_row_index_survives_the_keep_filter(self, tmp_path):
        """The passthrough is positional and the function drops rows. If the
        index were appended instead of indexed, every record after the first
        dropped row would carry the wrong item's identity -- and a learned scale
        built on it would look perfectly reasonable and be wrong.

        Row 1 has a zero current price and is dropped, so the surviving records
        must carry labels 500 and 502, not 500 and 501.
        """
        fc = _forecaster(tmp_path)
        records = fc._conformal_records(
            mid_ret=[1.0, 1.0, 1.0],
            actual_ret=[9.0, 9.0, 9.0],
            sigma=[0.1, 0.1, 0.1],
            current_price=[10.0, 0.0, 10.0],
            row_index=[500, 501, 502],
        )
        assert [r["row_index"] for r in records] == [500, 502]

    def test_a_misaligned_row_index_raises_rather_than_mislabelling(self, tmp_path):
        fc = _forecaster(tmp_path)
        with pytest.raises(ValueError, match="positional"):
            fc._conformal_records(
                mid_ret=[1.0, 1.0],
                actual_ret=[9.0, 9.0],
                sigma=[0.1, 0.1],
                current_price=[10.0, 10.0],
                row_index=[7],
            )

    def test_the_row_index_is_absent_unless_asked_for(self, tmp_path):
        """Off by default: the records feed q_hat, and a stray column would
        reach every consumer of the calibration frame."""
        fc = _forecaster(tmp_path)
        records = fc._conformal_records(
            mid_ret=[1.0],
            actual_ret=[9.0],
            sigma=[0.1],
            current_price=[10.0],
        )
        assert "row_index" not in records[0]

    def test_the_arm_is_off_by_default(self, tmp_path, monkeypatch):
        """Shipped default. The coherence argument is not the measurement, and
        the one read available (run 31563209228) moved q_hat the wrong way."""
        monkeypatch.delenv("CONFORMAL_SERVED_BASIS", raising=False)
        fc = _forecaster(tmp_path)
        frame = pd.DataFrame({calibration_target_col(3): [1.0, 2.0]})
        out = fc._calibration_returns(frame, 3, np.array([9.0, 9.0]))
        assert np.allclose(out, [9.0, 9.0])
        assert fc.conformal_basis[3] == "raw_anchor"

    def test_the_arm_reads_the_served_column_when_on(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CONFORMAL_SERVED_BASIS", "1")
        fc = _forecaster(tmp_path)
        frame = pd.DataFrame({calibration_target_col(7): [1.0, 2.0]})
        out = fc._calibration_returns(frame, 7, np.array([9.0, 9.0]))
        assert np.allclose(out, [1.0, 2.0])
        assert fc.conformal_basis[7] == "served"

    def test_the_arm_warns_when_it_cannot_take_effect(self, tmp_path, monkeypatch, caplog):
        """On, but the frame predates the column — the arm silently is not the
        arm, which is the one state a paired read must not mistake for a
        result."""
        monkeypatch.setenv("CONFORMAL_SERVED_BASIS", "1")
        fc = _forecaster(tmp_path)
        frame = pd.DataFrame({"price": [10.0, 11.0]})
        with caplog.at_level("WARNING"):
            out = fc._calibration_returns(frame, 3, np.array([1.0, 2.0]))
        assert np.allclose(out, [1.0, 2.0])
        assert fc.conformal_basis[3] == "raw_anchor"
        assert "NOT in effect" in caplog.text
