"""Rank IC on the cohort where neither label basis is contaminated.

`prepare_targets` divides by the raw quote at the anchor; `predict` divides by
its local median. The 2026-08-11 replay measured the gap between the two at
+0.1464 rank IC and attributed +0.1398 of it to that denominator alone, in 16
CI cells of 16 (`docs/changelog/2026-08-11-the-gap-is-the-anchor-denominator.md`,
`2026-08-11-clean-anchor-confirmed-in-ci.md`).

Moving the label to the served denominator was measured and REFUTED: it swings
pooled served rank IC to +0.17-0.31, and every point of that is `p[d]/S[d]`
re-entering as a factor the model can read at the anchor
(`2026-08-11-smoothed-anchor-label-measured.md`). Both bases carry the wedge,
with opposite signs.

The **tied** cohort -- items whose anchor quote already equals its own local
median -- is the one place where `p[d]/S[d]` is identically 1 and neither basis
can operate. That is the cohort an arm has to be ranked on, and these tests pin
the metric that makes it readable: `rank_ic_tied` per fold and
`rank_ic_edge_vs_naive_tied` in `cv_results`.

The mask is deliberately NOT a second implementation. `replay_serving._tied_mask`
defines the population every 2026-08-11 result was measured on; a copy that drifted
from it would let CV and the replay print the same word for different cohorts, so
one test here holds the two against each other on the same panel.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from models.forecaster import ANCHOR_TIED_COL, ItemForecaster


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path_factory.mktemp("saved_models")))


def _row(out, item, day):
    r = out[(out["item_id"] == item) & (out["date"] == day)]
    assert len(r) == 1, f"{item} {day}: {len(r)} rows"
    return r.iloc[0]


# ------------------------------------------------------------- the definition


def test_the_anchor_quote_that_is_its_own_median_is_tied(forecaster):
    """Tied means `p[d] == S[d]`, so the wedge the label carries is exactly 1.

    A zig-zag whose latest quote lands BETWEEN its two predecessors is the
    ordinary way this happens on a real series -- it is not the same as a frozen
    price, and it must not need one.
    """
    day = date(2026, 2, 10)
    df = pd.DataFrame(
        [
            {"item_id": "a", "date": day - timedelta(days=2), "price": 10.0},
            {"item_id": "a", "date": day - timedelta(days=1), "price": 14.0},
            {"item_id": "a", "date": day, "price": 12.0},
            {"item_id": "a", "date": day + timedelta(days=3), "price": 13.0},
        ]
    )
    assert bool(_row(forecaster.prepare_targets(df, 3), "a", day)[ANCHOR_TIED_COL])


def test_a_trending_quote_is_not_tied(forecaster):
    """The newest observation of a monotone series is its window's MAXIMUM, not
    its median -- so a trending item is deviating, and the cohort is not simply
    'items that did not move'."""
    day = date(2026, 2, 10)
    df = pd.DataFrame(
        [
            {"item_id": "a", "date": day - timedelta(days=2), "price": 10.0},
            {"item_id": "a", "date": day - timedelta(days=1), "price": 11.0},
            {"item_id": "a", "date": day, "price": 12.0},
            {"item_id": "a", "date": day + timedelta(days=3), "price": 13.0},
        ]
    )
    assert not bool(_row(forecaster.prepare_targets(df, 3), "a", day)[ANCHOR_TIED_COL])


def test_a_spike_at_the_anchor_is_deviating(forecaster):
    """The row the whole finding is about: one noisy-high print inflates
    `return_1d` and deflates the label at once."""
    day = date(2026, 2, 10)
    df = pd.DataFrame(
        [
            {"item_id": "a", "date": day - timedelta(days=2), "price": 10.0},
            {"item_id": "a", "date": day - timedelta(days=1), "price": 10.5},
            {"item_id": "a", "date": day, "price": 40.0},
            {"item_id": "a", "date": day + timedelta(days=3), "price": 10.2},
        ]
    )
    assert not bool(_row(forecaster.prepare_targets(df, 3), "a", day)[ANCHOR_TIED_COL])


def test_an_items_first_observation_is_tied(forecaster):
    """A one-point window is its own median, so the wedge is 1 and the row
    belongs in the clean cohort. `replay_serving._pinned_anchor` falls back to
    the latest observation for exactly this case; the two must agree."""
    day = date(2026, 2, 10)
    df = pd.DataFrame(
        [
            {"item_id": "a", "date": day, "price": 10.0},
            {"item_id": "a", "date": day + timedelta(days=3), "price": 11.0},
        ]
    )
    assert bool(_row(forecaster.prepare_targets(df, 3), "a", day)[ANCHOR_TIED_COL])


def test_the_mask_does_not_move_with_the_label_arm(forecaster, monkeypatch):
    """`anchor_is_tied` is a property of the PRICE SERIES, not of the label.

    If it followed `LABEL_SMOOTHED_ANCHOR` the control and the arm would be
    ranked on different populations, which is the failure the cohort exists to
    prevent.
    """
    rng = np.random.default_rng(5)
    rows = []
    for i in range(6):
        price = 10.0 + i
        for d in range(40):
            price *= 1.0 + rng.normal(0.0, 0.02)
            rows.append({"item_id": f"i{i}", "date": date(2026, 1, 1) + timedelta(days=d), "price": price})
    df = pd.DataFrame(rows)

    monkeypatch.delenv("LABEL_SMOOTHED_ANCHOR", raising=False)
    off = forecaster.prepare_targets(df, 3)[ANCHOR_TIED_COL].to_numpy()
    monkeypatch.setenv("LABEL_SMOOTHED_ANCHOR", "1")
    on = forecaster.prepare_targets(df, 3)[ANCHOR_TIED_COL].to_numpy()

    assert off.any() and not off.all(), "fixture must contain both cohorts"
    assert np.array_equal(off, on)


def test_it_agrees_with_the_replays_definition(forecaster):
    """One population, two readers. `scripts/replay_serving.py` is where every
    2026-08-11 tied/deviating number came from; this is the test that keeps CV's
    copy from drifting away from it while printing the same word."""
    from scripts.replay_serving import _pin_matches_production, _tied_mask

    if not _pin_matches_production():
        pytest.skip("replay's pinned window no longer matches production's")

    rng = np.random.default_rng(11)
    rows = []
    for i in range(12):
        price = 5.0 + i
        for d in range(30):
            price *= 1.0 + rng.normal(0.0, 0.03)
            rows.append({"item_id": f"i{i}", "date": date(2026, 1, 1) + timedelta(days=d), "price": price})
    df = pd.DataFrame(rows)

    anchor = date(2026, 1, 25)
    outcomes = df.rename(columns={"date": "day"}).copy()
    outcomes["day"] = pd.to_datetime(outcomes["day"])
    theirs = _tied_mask(outcomes, anchor)

    ours = forecaster.prepare_targets(df, 3)
    ours = ours[ours["date"] == anchor].set_index("item_id")[ANCHOR_TIED_COL]

    assert len(ours) == 12
    assert theirs.reindex(ours.index).eq(True).equals(ours.eq(True))
    assert ours.any() and not ours.all(), "fixture must contain both cohorts"


# --------------------------------------------------------------- the CV metric


def _tier(price: float) -> int:
    """The production banding from `engineer_features`, so a fixture cannot
    describe a `price_tier` the real frame would never produce."""
    for bound, tier in ((100, 4), (20, 3), (5, 2), (1, 1)):
        if price >= bound:
            return tier
    return 0


def _cv_forecaster(tmp_path):
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.QUANTILES = [0.5]
    f.feature_cols = ["feat_a"]
    f.CV_MIN_TRAIN_DAYS = 40
    f.CV_STEP_DAYS = 15
    f.VALIDATION_WINDOW_DAYS = 10
    return f


def _tdf(n_items=60, n_dates=80, horizon=3, seed=3, with_tied=True, tied_price=5.0, deviating_price=5.0):
    """A tdf in the shape `_cv_evaluate_horizon` consumes, carrying the mask.

    Half the items are tied and their label is a clean monotone function of
    `feat_a`; the other half are deviating and their label is CONSTANT, so they
    contribute no within-date ordering at all. A rank IC that pooled the two is
    therefore strictly worse than one restricted to the tied half -- which is
    the plumbing these tests pin, not a claim about the real cohort.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        tied = i % 2 == 0
        for d in range(n_dates):
            feat = rng.normal()
            row = {
                "item_id": f"i{i}",
                "date": date(2025, 1, 1) + timedelta(days=d),
                "price": tied_price if tied else deviating_price,
                "price_tier": _tier(tied_price if tied else deviating_price),
                "price_std_60d": 1.0,
                "feat_a": feat,
                "return_1d": rng.normal(),
                f"target_return_{horizon}d": 5.0 * feat if tied else 1.0,
            }
            if with_tied:
                row[ANCHOR_TIED_COL] = tied
            rows.append(row)
    return pd.DataFrame(rows)


def _run_cv(tmp_path, **kw):
    f = _cv_forecaster(tmp_path)
    _, fold_metrics = f._cv_evaluate_horizon(_tdf(**kw), 3, {0.5: {}})[:2]
    assert len(fold_metrics) >= 2
    return fold_metrics


def test_folds_carry_the_tied_cohort_beside_the_pooled_one(tmp_path):
    """The load-bearing assertion: two different numbers from one fold.

    If the mask silently did not apply, `rank_ic_tied` would equal `rank_ic`.
    """
    for m in _run_cv(tmp_path):
        assert m["rank_ic"] is not None and m["rank_ic_tied"] is not None
        assert m["rank_ic_tied"] > m["rank_ic"] + 0.05, (
            f"pooled={m['rank_ic']} tied={m['rank_ic_tied']} — the mask did not apply"
        )


def test_the_naive_baseline_is_scored_on_the_same_cohort(tmp_path):
    """`rank_ic_tied` is only rankable against a bar measured on the same rows.
    A tied model figure differenced against a pooled `-return_1d` would compare
    two populations, which is the error the cohort exists to remove."""
    for m in _run_cv(tmp_path):
        assert m["naive_rank_ic_tied"] is not None
        assert m["naive_rank_ic_tied"] != m["naive_rank_ic"]


def test_the_tied_cohort_is_intersected_with_the_served_one(tmp_path):
    """A penny item is not served, so a tied penny row cannot enter the number
    the product is ranked on. `n_tied` counts the intersection."""
    served = _run_cv(tmp_path, tied_price=5.0)
    penny = _run_cv(tmp_path, tied_price=0.30)

    assert all(m["n_tied"] > 0 for m in served)
    for m in penny:
        assert m["n_tied"] == 0
        assert m["rank_ic_tied"] is None, "no served tied rows is 'cannot tell', never a number"


def test_a_frame_without_the_mask_reports_none_not_a_fallback(tmp_path):
    """CV is also driven over frames built before the column existed -- every
    `ab_test_*` harness among them. A missing mask is 'cannot tell'; falling
    back to the pooled rows would publish a contaminated number under the clean
    key, which is the one outcome worse than no number."""
    for m in _run_cv(tmp_path, with_tied=False):
        assert m["rank_ic"] is not None
        assert m["rank_ic_tied"] is None
        assert m["naive_rank_ic_tied"] is None
        assert m["n_tied"] == 0


def test_a_date_with_too_few_tied_rows_is_dropped_not_padded(tmp_path):
    """`_within_date_rank_ic` needs `min_rows` per date, and the tied cohort is
    a third of the panel -- so a date can clear the bar pooled and fail it tied.
    The fold has to say how many dates it actually read, or the two columns look
    paired when they are not."""
    for m in _run_cv(tmp_path):
        assert m["rank_ic_tied_dates"] >= 1
        assert m["rank_ic_tied_dates"] <= m["rank_ic_dates"]


# ------------------------------------------------------------- what gets saved


def test_cv_results_publish_the_tied_edge(tmp_path):
    """`rank_ic_edge_vs_naive` is the bar every Track N arm is read on, and it
    is measured on the contaminated basis. The tied edge is the one that means
    what it says, and it has to reach `meta.json` to be readable at all."""
    f = _cv_forecaster(tmp_path)
    fold_metrics = f._cv_evaluate_horizon(_tdf(), 3, {0.5: {}})[1]
    summary = f._summarise_rank_ic(fold_metrics)

    assert summary["mean_rank_ic_tied"] is not None
    assert summary["mean_naive_rank_ic_tied"] is not None
    assert summary["rank_ic_edge_vs_naive_tied"] == pytest.approx(
        round(summary["mean_rank_ic_tied"] - summary["mean_naive_rank_ic_tied"], 4)
    )
    assert summary["rank_ic_edge_vs_naive_tied"] != summary["rank_ic_edge_vs_naive"]


def test_the_diagnostics_summary_greps_the_line_that_carries_it(tmp_path):
    """The number has to reach the only vehicle that reports it.

    `model-diagnostics.yml` filters the run log into the step summary with one
    `grep -E`. Its `rank IC` alternative matched only the WARNING form -- the
    headline line reads `rank_ic=` -- so the pair was absent from the summary
    whenever the edge was positive, i.e. exactly when an arm had worked. A
    metric nobody can read is not a metric.
    """
    import re
    from pathlib import Path

    wf = (Path(__file__).resolve().parents[2] / ".github/workflows/model-diagnostics.yml").read_text()
    pattern = re.search(r'grep -E "([^"]+)" \\\n\s+diagnostics-', wf)
    assert pattern, "the diagnostics grep moved — re-point this test"

    emitted = (
        "  Cross-sectional (>=$1, CLEAN ANCHOR): rank_ic=0.1321 vs "
        "naive=0.0842 → edge=0.0479 | 12,004 rows, 31 of 44 date-folds. "
        "← RANK ARMS ON THIS LINE."
    )
    assert re.search(pattern.group(1), emitted)


def test_the_pooled_series_is_unchanged(tmp_path):
    """Additive, never a replacement. `mean_rank_ic` is the series every
    historical `meta.json` holds and the trust warning reads; the tied cohort
    sits beside it."""
    f = _cv_forecaster(tmp_path)
    fold_metrics = f._cv_evaluate_horizon(_tdf(), 3, {0.5: {}})[1]
    summary = f._summarise_rank_ic(fold_metrics)

    pooled = [m["rank_ic"] for m in fold_metrics if m["rank_ic"] is not None]
    assert summary["mean_rank_ic"] == pytest.approx(round(float(np.mean(pooled)), 4))
