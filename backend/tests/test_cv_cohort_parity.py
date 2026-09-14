"""CV must score the >=$1 cohort separately, not only all tiers.

`cv_results.mean_classifier_acc` pools every price tier, and the offline frame
is 82.6% tier-0, so it is approximately the penny-item score. The production
headline scores `price_tier >= HEADLINE_MIN_TIER` only. Comparing the two was
comparing different populations, which is most of the "~20pp train/serve gap"
that five hypotheses failed to explain.

These tests pin the fix: a second, additive accuracy restricted to the
production cohort, without disturbing the all-tiers series or the trust gate
that reads it.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from backtest.scoring import HEADLINE_MIN_TIER
from models.forecaster import ItemForecaster


@pytest.fixture(autouse=True)
def _diagnostic_classifier_on(monkeypatch):
    """Every test here scores the per-fold directional classifier, which has
    been off by default since 2026-08-09 (932s, 52% of a retrain, feeding no
    served artifact). The cohort split these tests pin is still correct and
    still runs whenever the diagnostic is asked for, so they opt in rather than
    the default reverting."""
    monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", "1")


def _tier(price: float) -> int:
    """The production banding from engineer_features, so the fixtures cannot
    describe a frame `price_tier` would never actually produce."""
    for bound, tier in ((100, 4), (20, 3), (5, 2), (1, 1)):
        if price >= bound:
            return tier
    return 0


def _cv_forecaster(tmp_path, feature_cols):
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.QUANTILES = [0.5]
    f.feature_cols = list(feature_cols)
    # Shrink the expanding window so a synthetic frame yields >= 2 folds.
    f.CV_MIN_TRAIN_DAYS = 40
    f.CV_STEP_DAYS = 15
    f.VALIDATION_WINDOW_DAYS = 10
    return f


def _frame(prices, n_dates=80, horizon=3, seed=11, with_tier=True):
    """Synthetic tdf in the shape _cv_evaluate_horizon consumes.

    The two cohorts get deliberately separable label distributions so their
    accuracies cannot coincide by luck: penny rows never move (every row is
    `flat` under the fixed +/-0.5% band) and dollar rows always move (+/-5%,
    never `flat`). Nothing in the feature matrix predicts *which* way a dollar
    row moved, so the >=$1 cohort sits near 50% while the pooled figure is
    dragged up by the perfectly-predictable penny half.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for item, base in enumerate(prices):
        penny = base < 1
        for d in range(n_dates):
            price = base * (1.0 + 0.002 * d)
            row = {
                "item_id": f"item_{item}",
                "date": date(2025, 1, 1) + timedelta(days=d),
                "price": price,
                "price_std_60d": abs(rng.normal(base * 0.05, base * 0.01)) + 1e-9,
                "feat_a": rng.normal(),
                "feat_b": rng.normal(),
                f"target_return_{horizon}d": 0.0 if penny else (5.0 if d % 2 else -5.0),
            }
            if with_tier:
                row["price_tier"] = _tier(price)
            rows.append(row)
    return pd.DataFrame(rows)


MIXED_PRICES = [0.30, 0.45, 0.60, 0.85, 2.00, 3.00, 4.00, 4.50]
PENNY_PRICES = [0.30, 0.45, 0.60, 0.85, 0.25, 0.35, 0.55, 0.75]


def _run_cv(tmp_path, prices, with_tier=True, feature_cols=("feat_a", "feat_b", "price_tier")):
    f = _cv_forecaster(tmp_path, feature_cols)
    tdf = _frame(prices, with_tier=with_tier)
    _, fold_metrics = f._cv_evaluate_horizon(tdf, 3, {0.5: {}})[:2]
    assert len(fold_metrics) >= 2
    return fold_metrics


def test_fold_metrics_score_the_dollar_cohort_separately(tmp_path):
    """The load-bearing assertion: the two keys are different numbers.

    If the tier filter silently did not apply, `classifier_accuracy_ge1` would
    equal `classifier_accuracy` on this frame instead of sitting ~25pp below it.
    """
    fold_metrics = _run_cv(tmp_path, MIXED_PRICES)

    for m in fold_metrics:
        assert m["classifier_accuracy_ge1"] is not None, (
            "the mixed frame has >=$1 rows in every fold, so the cohort accuracy must be a number"
        )
        assert m["classifier_accuracy_ge1"] < m["classifier_accuracy"] - 5, (
            f"pooled={m['classifier_accuracy']} ge1={m['classifier_accuracy_ge1']} — the >=$1 filter did not apply"
        )


def test_dollar_cohort_accuracy_is_none_not_zero_when_a_fold_is_all_penny(tmp_path):
    """An empty partition has no accuracy. A 0.0 would be read as the model
    scoring nothing on dollar items, which is a different — and false — claim.
    Same rule `score_cohort` follows for its own empty partitions."""
    fold_metrics = _run_cv(tmp_path, PENNY_PRICES)

    for m in fold_metrics:
        assert m["classifier_accuracy_ge1"] is None
        assert m["classifier_accuracy"] is not None, "the all-tiers figure is unconditional and must still be scored"


def test_dollar_cohort_accuracy_is_none_when_the_frame_has_no_price_tier(tmp_path):
    """`price_tier` is a real column in production, but CV is also driven over
    frames that predate it. A missing column is 'cannot tell', not zero."""
    fold_metrics = _run_cv(tmp_path, MIXED_PRICES, with_tier=False, feature_cols=("feat_a", "feat_b"))

    for m in fold_metrics:
        assert m["classifier_accuracy_ge1"] is None


# ---------------------------------------------------------------------------
# The aggregation seam: fold_metrics -> cv_results, through the real train()
# ---------------------------------------------------------------------------


def _train_frame(n_dates=140, seed=5):
    """Minimal frame in the shape build_training_data returns, spanning tiers.

    `price_tier` is recomputed per row from the drifting price, as
    engineer_features does — the cohort a row belongs to is a property of that
    day's price, not of the item.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for item, base in enumerate([0.30, 0.60, 0.90, 2.0, 4.0, 8.0, 15.0, 25.0, 40.0, 60.0]):
        price = base
        for d in range(n_dates):
            price = max(price * (1.0 + rng.normal(0.001, 0.02)), 0.05)
            rows.append(
                {
                    "item_id": f"item_{item}",
                    "date": date(2025, 1, 1) + timedelta(days=d),
                    "price": price,
                    "price_tier": _tier(price),
                    "price_std_60d": abs(rng.normal(base * 0.05, base * 0.01)) + 1e-9,
                    "feat_a": rng.normal(),
                    "feat_b": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained():
    """One real train() over a mixed-tier frame, shared by the seam tests."""
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        f = ItemForecaster(db_session=MagicMock(), model_dir=tmp)
        f.N_ENSEMBLES = 1
        f.SKIP_HP_HORIZONS = list(f.HORIZONS)  # no Optuna
        f.CV_MIN_TRAIN_DAYS = 40
        f.CV_STEP_DAYS = 25
        f.VALIDATION_WINDOW_DAYS = 10
        df = _train_frame()

        def fake_build(*a, **kw):
            f.feature_cols = ["feat_a", "feat_b", "price_tier"]
            f._base_feature_cols = list(f.feature_cols)
            return df.copy()

        # This fixture is module-scoped, so it is built before the
        # function-scoped _diagnostic_classifier_on fixture can apply. It has
        # to opt in itself or train() runs with the diagnostic off and
        # mean_classifier_acc comes back None.
        with (
            patch.object(f, "build_training_data", fake_build),
            patch.object(f, "save_models", lambda *a, **kw: None),
            patch.dict("os.environ", {"SKIP_REGIMES": "1", "CV_DIAGNOSTIC_CLASSIFIER": "1"}),
        ):
            f.train()
        yield f


def test_cv_results_carry_the_dollar_cohort_alongside_the_all_tiers_series(trained):
    """Both keys reach cv_results, and the new one is the mean of the folds
    that actually had a >=$1 cohort."""
    for h in trained.HORIZONS:
        cv = trained.cv_results[h]
        assert cv["mean_classifier_acc"] is not None, (
            "the all-tiers series must survive — every historical meta.json holds it and the trust gate reads it"
        )

        per_fold = [
            m["classifier_accuracy_ge1"] for m in cv["per_fold"] if m.get("classifier_accuracy_ge1") is not None
        ]
        assert per_fold, f"{h}d frame has >=$1 rows, so folds must score them"
        expected = round(float(np.mean(per_fold)), 1)
        assert cv["mean_classifier_acc_ge1"] == pytest.approx(expected), (
            f"{h}d: cv_results does not aggregate the per-fold >=$1 accuracies"
        )


def test_the_trust_gate_still_reads_the_all_tiers_classifier_accuracy(trained):
    """`edge_vs_best_baseline` gates whether directional forecasts are called
    trustworthy, and it also drives the confidence-threshold calibration.
    Repointing it at the >=$1 cohort would move both silently, so the cohort
    number is additive only."""
    for h in trained.HORIZONS:
        cv = trained.cv_results[h]
        baselines = [b for b in (cv["mean_persistence_acc"], cv["mean_momentum_acc"]) if b is not None]
        if not baselines or cv["edge_vs_best_baseline"] is None:
            continue
        expected = round(cv["mean_classifier_acc"] - max(baselines), 1)
        assert cv["edge_vs_best_baseline"] == pytest.approx(expected), (
            f"{h}d: the edge is no longer computed from the all-tiers classifier accuracy"
        )


def test_the_cohort_split_uses_the_same_floor_the_headline_does(tmp_path):
    """The whole point is parity with production. A CV cohort cut at a
    different tier would reintroduce the mismatch this change exists to remove.

    Priced so the frame straddles the floor exactly: tier 0 vs tier 1.
    """
    assert HEADLINE_MIN_TIER == 1
    tiers = {_tier(p) for p in MIXED_PRICES}
    assert tiers == {0, 1}, f"fixture must straddle the headline floor, got {tiers}"
