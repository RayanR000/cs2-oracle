"""Tests for the two gated accuracy instruments added 2026-08-10.

- `tier_lead_return_1d`: the expensive->cheap lead-lag feature (z = 9.1 in
  docs/research/2026-08-07-cs2-forecasting-research.md).
- The within-date rank transform of the feature matrix (Gu/Kelly/Xiu fn 29).

Both are off by default. The tests that matter most here are the ones pinning
that: a gated instrument that silently turns itself on is worse than one that
does not exist, because every stored A/B becomes uninterpretable.
"""

import numpy as np
import pandas as pd
import pytest
from models.forecaster import ItemForecaster, _feature_group

# ----------------------------------------------------------------------
# Gating
# ----------------------------------------------------------------------


def test_both_instruments_are_off_by_default(monkeypatch):
    monkeypatch.delenv("TIER_LEAD_FEATURE", raising=False)
    monkeypatch.delenv("CROSS_SECTIONAL_RANK", raising=False)
    assert ItemForecaster.tier_lead_enabled() is False
    assert ItemForecaster.cross_sectional_rank_enabled() is False


def test_only_the_exact_flag_value_enables(monkeypatch):
    for value in ("0", "", "true", "yes", "TRUE"):
        monkeypatch.setenv("TIER_LEAD_FEATURE", value)
        monkeypatch.setenv("CROSS_SECTIONAL_RANK", value)
        assert ItemForecaster.tier_lead_enabled() is False
        assert ItemForecaster.cross_sectional_rank_enabled() is False
    monkeypatch.setenv("TIER_LEAD_FEATURE", "1")
    monkeypatch.setenv("CROSS_SECTIONAL_RANK", "1")
    assert ItemForecaster.tier_lead_enabled() is True
    assert ItemForecaster.cross_sectional_rank_enabled() is True


def test_tier_lead_column_groups_to_its_own_group():
    assert _feature_group("tier_lead_return_1d") == "tier_lead"
    # Must NOT land in price_technicals: that group is allowlisted, so a
    # misgrouped column would reach the booster with the instrument off.
    assert _feature_group("tier_lead_return_1d") != "price_technicals"


def test_tier_lead_is_not_in_cross_sectional():
    """It must not inherit cross_sectional's PREDICT_TAIL_ITEM_DAYS problem.

    market_return_30d_percentile uses rolling(365) against a 240-item-day
    predict tail; PREDICT_TAIL_ITEM_DAYS documents that admitting
    `cross_sectional` requires raising the constant past 368. The tier lead is
    lag-1 and carries no such dependency, which is why it gets its own group
    rather than being folded into the existing one.
    """
    assert _feature_group("tier_lead_return_1d") != "cross_sectional"
    assert "tier_lead" in ItemForecaster.ALL_FEATURE_GROUPS


def test_allowlist_admits_tier_lead_only_when_enabled(monkeypatch):
    fc = ItemForecaster.__new__(ItemForecaster)
    cols = ["return_1d", "tier_lead_return_1d"]

    monkeypatch.delenv("TIER_LEAD_FEATURE", raising=False)
    assert fc._apply_feature_allowlist(cols, list(ItemForecaster.FEATURE_GROUP_ALLOWLIST)) == ["return_1d"]

    monkeypatch.setenv("TIER_LEAD_FEATURE", "1")
    allowlist = list(ItemForecaster.FEATURE_GROUP_ALLOWLIST) + [ItemForecaster.TIER_LEAD_GROUP]
    assert fc._apply_feature_allowlist(cols, allowlist) == cols


def test_skipped_groups_tracks_the_flag(monkeypatch):
    fc = ItemForecaster.__new__(ItemForecaster)

    monkeypatch.delenv("TIER_LEAD_FEATURE", raising=False)
    monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
    assert ItemForecaster.TIER_LEAD_GROUP in fc._skipped_feature_groups()

    monkeypatch.setenv("TIER_LEAD_FEATURE", "1")
    assert ItemForecaster.TIER_LEAD_GROUP not in fc._skipped_feature_groups()


def test_engineering_and_allowlist_cannot_disagree(monkeypatch):
    """The hazard _skipped_feature_groups documents, pinned for this group.

    A group that is allowlisted but skipped during engineering produces columns
    that are absent and median-filled to zero -- undetectably. Whatever the
    flag, the group must be on the same side of both decisions.
    """
    fc = ItemForecaster.__new__(ItemForecaster)
    for value in (None, "1"):
        if value is None:
            monkeypatch.delenv("TIER_LEAD_FEATURE", raising=False)
        else:
            monkeypatch.setenv("TIER_LEAD_FEATURE", value)
        skipped = ItemForecaster.TIER_LEAD_GROUP in fc._skipped_feature_groups()
        admitted = "tier_lead_return_1d" in fc._apply_feature_allowlist(
            ["tier_lead_return_1d"],
            list(ItemForecaster.FEATURE_GROUP_ALLOWLIST)
            + ([ItemForecaster.TIER_LEAD_GROUP] if ItemForecaster.tier_lead_enabled() else []),
        )
        assert skipped != admitted


# ----------------------------------------------------------------------
# Tier lead-lag mechanics
# ----------------------------------------------------------------------


def _panel(rows):
    """rows: (item_id, 'YYYY-MM-DD', price_tier, return_1d)."""
    return pd.DataFrame(rows, columns=["item_id", "date", "price_tier", "return_1d"]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def _fc():
    return ItemForecaster.__new__(ItemForecaster)


def test_cheap_item_receives_the_tier_above_lagged_one_day():
    df = _panel(
        [
            # tier 2 moved +6% on d1, tier 1 moved +1%.
            ("a", "2026-01-01", 1, 1.0),
            ("b", "2026-01-01", 2, 6.0),
            ("a", "2026-01-02", 1, 0.0),
            ("b", "2026-01-02", 2, 0.0),
        ]
    )
    fc = _fc()
    out = fc._add_tier_lead_features(df.copy())

    d2 = out[out["date"] == "2026-01-02"].set_index("item_id")
    # The tier-1 item on d2 gets tier 2's d1 return.
    assert d2.loc["a", "tier_lead_return_1d"] == pytest.approx(6.0)
    # The tier-2 item has no tier 3 in the frame, so NaN.
    assert np.isnan(d2.loc["b", "tier_lead_return_1d"])
    # First date has no predecessor.
    d1 = out[out["date"] == "2026-01-01"]
    assert d1["tier_lead_return_1d"].isna().all()


def test_direction_is_expensive_to_cheap_not_the_reverse():
    """The folk direction is backwards and the sign of this test is the point.

    Cheap -> expensive measured +0.043, inside the noise band; expensive ->
    cheap measured +0.213 at z = 9.1. If this feature ever starts handing the
    cheap tier's return to expensive items, the measured effect it was built on
    no longer applies.
    """
    df = _panel(
        [
            ("cheap", "2026-01-01", 0, 99.0),
            ("rich", "2026-01-01", 4, 1.0),
            ("cheap", "2026-01-02", 0, 0.0),
            ("rich", "2026-01-02", 4, 0.0),
        ]
    )
    out = _fc()._add_tier_lead_features(df.copy())
    d2 = out[out["date"] == "2026-01-02"].set_index("item_id")
    # Tier 4 must NOT receive tier 0's +99%.
    assert np.isnan(d2.loc["rich", "tier_lead_return_1d"])


def test_lag_is_over_observed_dates_not_the_calendar():
    """The archive is missing whole days (aggregator-archive-day-gaps).

    A calendar shift would emit NaN for the day after a gap; a positional shift
    over observed dates carries the last observation across it, which is the
    convention every other lag in the forecaster uses.
    """
    df = _panel(
        [
            ("a", "2026-01-01", 1, 0.0),
            ("b", "2026-01-01", 2, 5.0),
            # 2026-01-02 .. 01-09 missing entirely.
            ("a", "2026-01-10", 1, 0.0),
            ("b", "2026-01-10", 2, 0.0),
        ]
    )
    out = _fc()._add_tier_lead_features(df.copy())
    d10 = out[out["date"] == "2026-01-10"].set_index("item_id")
    assert d10.loc["a", "tier_lead_return_1d"] == pytest.approx(5.0)


def test_chunked_partials_reproduce_the_whole_frame_result():
    """The property _engineer_features_chunked's docstring promises.

    Predict runs in item chunks of PREDICT_CHUNK_ITEMS (1000 by default, against
    ~5,600 eligible items), so this is the production path -- a tier index over
    one chunk would be an index over the wrong cross-section.
    """
    rows = []
    for i in range(12):
        for d in ("2026-01-01", "2026-01-02"):
            rows.append((f"i{i}", d, i % 4, float(i) + (0.5 if d.endswith("2") else 0.0)))
    df = _panel(rows)
    fc = _fc()

    whole = fc._tier_lead_from_partials([fc._accumulate_tier_partials(df)])

    items = sorted(df["item_id"].unique())
    chunks = [items[:5], items[5:9], items[9:]]
    chunked = fc._tier_lead_from_partials([fc._accumulate_tier_partials(df[df["item_id"].isin(c)]) for c in chunks])

    pd.testing.assert_frame_equal(whole, chunked)


def test_missing_columns_yield_nan_not_a_crash():
    df = pd.DataFrame(
        {
            "item_id": ["a"],
            "date": pd.to_datetime(["2026-01-01"]),
        }
    )
    out = _fc()._add_tier_lead_features(df.copy())
    assert out["tier_lead_return_1d"].isna().all()


def test_row_count_and_order_are_preserved():
    """_apply_tier_lead merges, and a merge can duplicate or reorder rows.

    Every caller downstream of the feature builders assumes positional
    alignment with the frame it passed in.
    """
    df = _panel(
        [
            ("a", "2026-01-01", 1, 1.0),
            ("b", "2026-01-01", 2, 2.0),
            ("a", "2026-01-02", 1, 3.0),
            ("b", "2026-01-02", 2, 4.0),
        ]
    )
    out = _fc()._add_tier_lead_features(df.copy())
    assert len(out) == len(df)
    assert list(out["item_id"]) == list(df["item_id"])
    assert list(out["return_1d"]) == list(df["return_1d"])


# ----------------------------------------------------------------------
# Cross-sectional rank transform
# ----------------------------------------------------------------------


def test_rank_transform_maps_into_minus_one_to_one():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 4),
            "f": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"])
    assert out["f"].min() >= -1.0 and out["f"].max() <= 1.0
    # Evenly spaced input -> evenly spaced percentiles, centred on 0.
    assert out["f"].tolist() == pytest.approx([-0.75, -0.25, 0.25, 0.75])


def test_rank_transform_is_within_date_not_global():
    """The whole point: a date's level must not survive the transform.

    Date 2 is uniformly 100x date 1. If ranks leaked across dates, date 2's
    values would all sit above date 1's.
    """
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-02"] * 3),
            "f": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
        }
    )
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"])
    first = out["f"].iloc[:3].tolist()
    second = out["f"].iloc[3:].tolist()
    assert first == pytest.approx(second)


def test_rank_transform_is_monotone_within_a_date():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=50)
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"] * 50), "f": raw})
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"])
    assert (np.argsort(raw) == np.argsort(out["f"].to_numpy())).all()


def test_nan_is_preserved_not_ranked():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "f": [1.0, np.nan, 3.0],
        }
    )
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"])
    assert np.isnan(out["f"].iloc[1])
    assert not out["f"].iloc[[0, 2]].isna().any()


def test_date_constant_columns_are_skipped_not_zeroed():
    """Ranking a date-constant column deletes it: all ties -> 0.5 -> 0.

    Nothing in the current allowlist is date-constant, but a date-level feature
    added later would be erased without a word, so the guard is pinned.
    """
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-02"] * 3),
            "varies": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "date_level": [7.0, 7.0, 7.0, 9.0, 9.0, 9.0],
        }
    )
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["varies", "date_level"])
    assert out["date_level"].tolist() == [7.0, 7.0, 7.0, 9.0, 9.0, 9.0]
    assert out["varies"].tolist() != [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_transform_only_touches_named_columns():
    """Targets must never be rank-transformed."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "f": [1.0, 2.0, 3.0],
            "target_7d": [0.1, 0.2, 0.3],
        }
    )
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"])
    assert out["target_7d"].tolist() == [0.1, 0.2, 0.3]


def test_price_tier_is_never_rank_transformed():
    """Regression for a void A/B arm — run 31424689196, 2026-08-10.

    price_tier is a feature column AND the cohort gate: _cv_evaluate_horizon
    selects the >=$1 rows with `val_df["price_tier"] >= HEADLINE_MIN_TIER`, and
    _fit_direction_classifier takes it as `tier_train`. Ranked into [-1, 1] the
    maximum is 1 - 1/n, so the mask matched nothing: rank IC and
    classifier_accuracy_ge1 came back None while the POOLED metrics populated and
    looked 3-5pp better at every horizon. The run was green.
    """
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 5),
            "price_tier": [0, 1, 2, 3, 4],
            "return_1d": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["price_tier", "return_1d"])

    assert out["price_tier"].tolist() == [0, 1, 2, 3, 4]
    # The >=$1 gate must still select the four rows it selected before.
    assert (out["price_tier"].to_numpy() >= 1).sum() == 4
    # And the real feature must still have been transformed.
    assert out["return_1d"].tolist() != [1.0, 2.0, 3.0, 4.0, 5.0]


def test_the_exclusion_set_is_consulted_not_hardcoded():
    """Adding a name to RANK_TRANSFORM_EXCLUDED must be enough to protect it."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "guard": [1.0, 2.0, 3.0],
        }
    )
    original = ItemForecaster.RANK_TRANSFORM_EXCLUDED
    try:
        ItemForecaster.RANK_TRANSFORM_EXCLUDED = frozenset({"guard"})
        out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["guard"])
        assert out["guard"].tolist() == [1.0, 2.0, 3.0]
    finally:
        ItemForecaster.RANK_TRANSFORM_EXCLUDED = original


def test_absent_columns_are_tolerated():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 2),
            "f": [1.0, 2.0],
        }
    )
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f", "nope"])
    assert "nope" not in out.columns


# ----------------------------------------------------------------------
# The wiring seam: allowlist -> prune -> transform, in that order
# ----------------------------------------------------------------------


def _realistic_frame(n_items=8, dates=("2026-01-01", "2026-01-02")):
    """A frame shaped like the one _build_feature_frame hands to the reducer."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n_items):
        for d in dates:
            rows.append(
                {
                    "item_id": f"i{i}",
                    "date": pd.Timestamp(d),
                    "price": 5.0 + i,
                    "volume": 10.0,
                    "price_tier": i % 4,
                    "return_1d": float(rng.normal()),
                    "rsi_14": float(rng.uniform(20, 80)),
                    # A cross_sectional column, which the allowlist must still drop.
                    "market_return_1d": 0.5,
                    "target_7d": float(rng.normal()),
                }
            )
    return pd.DataFrame(rows)


def test_tier_lead_survives_selection_and_the_allowlist(monkeypatch):
    """End-to-end through the reducer, which is where a misgrouped name dies."""
    monkeypatch.setenv("TIER_LEAD_FEATURE", "1")
    monkeypatch.delenv("BYMYKEL_METADATA", raising=False)

    fc = ItemForecaster.__new__(ItemForecaster)
    fc.prune_failed_groups = False
    df = _realistic_frame()
    df = fc._add_tier_lead_features(df)

    fc.feature_cols = fc._select_feature_cols(df, ItemForecaster.HORIZONS, ItemForecaster.SHELVED_FEATURES)
    assert "tier_lead_return_1d" in fc.feature_cols

    fc._reduce_feature_cols(df)
    assert "tier_lead_return_1d" in fc.feature_cols, (
        "the tier-lead column was dropped by the allowlist even with the flag "
        "on — check _feature_group and _reduce_feature_cols agree on the group"
    )
    # The allowlist must still be doing its job on everything else.
    assert "market_return_1d" not in fc.feature_cols
    assert "target_7d" not in fc.feature_cols


def test_tier_lead_is_dropped_by_the_allowlist_when_the_flag_is_off(monkeypatch):
    monkeypatch.delenv("TIER_LEAD_FEATURE", raising=False)
    monkeypatch.delenv("BYMYKEL_METADATA", raising=False)

    fc = ItemForecaster.__new__(ItemForecaster)
    fc.prune_failed_groups = False
    df = _realistic_frame()
    df = fc._add_tier_lead_features(df)
    fc.feature_cols = fc._select_feature_cols(df, ItemForecaster.HORIZONS, ItemForecaster.SHELVED_FEATURES)
    fc._reduce_feature_cols(df)
    assert "tier_lead_return_1d" not in fc.feature_cols


def test_rank_transform_covers_the_reduced_feature_set(monkeypatch):
    """The transform runs after the reducer, over the surviving columns only.

    Order matters for cost (33 columns, not 123) and it is the order
    _build_feature_frame uses; this pins that the two agree.
    """
    monkeypatch.setenv("TIER_LEAD_FEATURE", "1")
    monkeypatch.delenv("BYMYKEL_METADATA", raising=False)

    fc = ItemForecaster.__new__(ItemForecaster)
    fc.prune_failed_groups = False
    df = _realistic_frame(n_items=12)
    df = fc._add_tier_lead_features(df)
    fc.feature_cols = fc._select_feature_cols(df, ItemForecaster.HORIZONS, ItemForecaster.SHELVED_FEATURES)
    fc._reduce_feature_cols(df)

    before = df["rsi_14"].copy()
    out = ItemForecaster._apply_cross_sectional_ranks(df, fc.feature_cols)

    assert out["rsi_14"].between(-1, 1).all()
    assert not np.allclose(out["rsi_14"], before)
    # Untouched: not a feature column.
    assert out["target_7d"].equals(_realistic_frame(n_items=12)["target_7d"])


# ----------------------------------------------------------------------
# Artifact-over-environment on the serving path
# ----------------------------------------------------------------------


def test_served_flags_follow_the_artifact_over_the_environment(monkeypatch):
    """A booster fitted on ranks must not be served raw values, whatever the env.

    This is the divergence that makes an env-only gate unsafe: the training run
    records what it did, and predict obeys the record.
    """
    fc = ItemForecaster.__new__(ItemForecaster)
    monkeypatch.setenv("TIER_LEAD_FEATURE", "1")
    monkeypatch.setenv("CROSS_SECTIONAL_RANK", "1")

    fc._artifact_tier_lead = False
    fc._artifact_xs_rank = False
    assert fc._tier_lead_served() is False
    assert fc._cross_sectional_rank_served() is False

    fc._artifact_tier_lead = True
    fc._artifact_xs_rank = True
    monkeypatch.delenv("TIER_LEAD_FEATURE", raising=False)
    monkeypatch.delenv("CROSS_SECTIONAL_RANK", raising=False)
    assert fc._tier_lead_served() is True
    assert fc._cross_sectional_rank_served() is True


def test_no_artifact_falls_back_to_the_environment(monkeypatch):
    """None means "this artifact does not say", which is not the same as False.

    Artifacts written before these flags existed carry no key, and a bare
    forecaster with no model loaded has to read something.
    """
    fc = ItemForecaster.__new__(ItemForecaster)
    fc._artifact_tier_lead = None
    fc._artifact_xs_rank = None

    monkeypatch.delenv("TIER_LEAD_FEATURE", raising=False)
    monkeypatch.delenv("CROSS_SECTIONAL_RANK", raising=False)
    assert fc._tier_lead_served() is False
    assert fc._cross_sectional_rank_served() is False

    monkeypatch.setenv("TIER_LEAD_FEATURE", "1")
    monkeypatch.setenv("CROSS_SECTIONAL_RANK", "1")
    assert fc._tier_lead_served() is True
    assert fc._cross_sectional_rank_served() is True


# ----------------------------------------------------------------------
# The reference cohort: training ranks 916 items, predict sees 5,536
#
# `_filter_by_median_price` runs before the transform on the training path, so
# the booster is fitted on percentiles within the >= $1 cohort. `predict`
# applies no floor. Ranking the served frame over all 5,536 items would hand
# the booster percentiles from a 6x larger, ~72% sub-$1 population -- silently,
# because every column still exists and still lies in [-1, 1].
# ----------------------------------------------------------------------


def _cohort_panel():
    """Four >= $1 items and four penny items, two dates.

    The penny values are interleaved with the cohort's on purpose: if they
    voted, they would move every cohort percentile rather than merely extend
    the tails.
    """
    rows = []
    for date in ("2026-01-01", "2026-01-02"):
        for i, (item, price, f) in enumerate(
            [
                ("rich_a", 10.0, 1.0),
                ("rich_b", 20.0, 2.0),
                ("rich_c", 30.0, 3.0),
                ("rich_d", 40.0, 4.0),
                ("penny_a", 0.03, 1.5),
                ("penny_b", 0.04, 2.5),
                ("penny_c", 0.05, 3.5),
                ("penny_d", 0.06, 0.5),
            ]
        ):
            rows.append({"item_id": item, "date": pd.Timestamp(date), "price": price, "f": f})
    return pd.DataFrame(rows)


def test_reference_rows_get_exactly_the_training_transform():
    """The load-bearing property: a served >= $1 row must receive the same
    number training computed for it, to the bit. Anything else means the
    booster is scoring a feature it was not fitted on."""
    df = _cohort_panel()
    mask = df["price"] >= 1.0

    served = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"], reference_mask=mask)

    # What training saw: the cohort alone, no mask, no penny rows in the frame.
    trained = ItemForecaster._apply_cross_sectional_ranks(df[mask].copy().reset_index(drop=True), ["f"])

    assert served.loc[mask.values, "f"].tolist() == pytest.approx(trained["f"].tolist())
    # And that is the evenly-spaced four-item answer, not a de-facto eight.
    assert served.loc[mask.values, "f"].iloc[:4].tolist() == pytest.approx([-0.75, -0.25, 0.25, 0.75])


def test_out_of_cohort_rows_are_placed_in_the_reference_distribution():
    """Sub-$1 items keep getting a forecast row, so they need a value on the
    same scale -- their position within the cohort's distribution, not their
    position within a population the booster never saw."""
    df = _cohort_panel()
    mask = df["price"] >= 1.0
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"], reference_mask=mask)

    first = out[out["date"] == pd.Timestamp("2026-01-01")].set_index("item_id")["f"]
    # f=1.5 sits between the cohort's 1.0 and 2.0 -> between their percentiles.
    assert -0.75 < first["penny_a"] < -0.25
    # f=0.5 is below every cohort value -> pinned at the floor.
    assert first["penny_d"] == pytest.approx(-1.0)
    # Monotone across the out-of-cohort rows too.
    assert first["penny_a"] < first["penny_b"] < first["penny_c"]


def test_out_of_cohort_rows_do_not_move_cohort_percentiles():
    """The failure this whole change exists to prevent."""
    df = _cohort_panel()
    mask = df["price"] >= 1.0

    with_mask = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"], reference_mask=mask)
    without = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"])

    cohort_masked = with_mask.loc[mask.values, "f"].tolist()
    cohort_pooled = without.loc[mask.values, "f"].tolist()
    assert cohort_masked != pytest.approx(cohort_pooled), (
        "pooling the penny items left the cohort percentiles unchanged; the fixture no longer exercises the defect"
    )


def test_no_mask_is_the_training_path_unchanged():
    """Training must not change behaviour: it passes no mask and every row is
    in the cohort by construction."""
    df = _cohort_panel()
    plain = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"])
    all_true = ItemForecaster._apply_cross_sectional_ranks(
        df.copy(), ["f"], reference_mask=pd.Series(True, index=df.index)
    )
    pd.testing.assert_frame_equal(plain, all_true)


def test_an_empty_reference_cohort_raises_rather_than_ranking_on_nothing():
    """A frame with no cohort rows cannot be transformed into the fitted scale.
    Median-filling the lot to zero would serve a constant and log nothing."""
    df = _cohort_panel()
    with pytest.raises(ValueError, match="reference cohort"):
        ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["f"], reference_mask=pd.Series(False, index=df.index))


def test_served_reference_mask_uses_the_artifact_floor():
    """The floor is a property of the artifact, not of today's environment: a
    model trained at >= $1 must be served at >= $1 even if the default moves."""
    fc = ItemForecaster.__new__(ItemForecaster)
    fc._artifact_min_median_price = 5.0
    df = _cohort_panel()
    mask = fc._reference_cohort_mask(df)
    assert set(df.loc[mask, "item_id"]) == {"rich_a", "rich_b", "rich_c", "rich_d"}

    fc._artifact_min_median_price = 25.0
    mask = fc._reference_cohort_mask(df)
    assert set(df.loc[mask, "item_id"]) == {"rich_c", "rich_d"}


def test_reference_mask_is_by_item_median_not_by_row():
    """`_filter_by_median_price` selects items on their median over the window.
    A row-wise price test would let one spike promote a penny item for a day,
    and the cohort would then differ from the trained one on that date alone."""
    fc = ItemForecaster.__new__(ItemForecaster)
    fc._artifact_min_median_price = 1.0
    df = pd.DataFrame(
        {
            "item_id": ["spiky"] * 3 + ["steady"] * 3,
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"] * 2),
            "price": [0.02, 50.0, 0.02, 5.0, 5.0, 5.0],
            "f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    mask = fc._reference_cohort_mask(df)
    assert not mask[df["item_id"] == "spiky"].any()
    assert mask[df["item_id"] == "steady"].all()


def test_predict_refuses_a_rank_artifact_that_records_no_cohort():
    """An artifact written before the floor was recorded cannot be served with
    the transform on: there is no way to know which items its percentiles were
    relative to, and guessing silently is the whole defect.

    Asserted against `predict`'s source rather than by calling it -- predict
    needs a loaded booster and a price archive. A source test is weak, but it
    fails if the guard is deleted, which is the regression that matters.
    """
    import inspect

    src = inspect.getsource(ItemForecaster.predict)
    guard = src.split("_cross_sectional_rank_served()")[1]
    assert "_artifact_min_median_price is None" in guard
    assert "raise RuntimeError" in guard
    # And the guard must precede the transform, not follow it.
    assert guard.index("raise RuntimeError") < guard.index("_apply_cross_sectional_ranks")
    assert "reference_mask=cohort" in guard, (
        "predict calls the transform without a reference cohort; it would rank "
        "the whole 5,536-item frame against itself"
    )


def test_the_cohort_floor_round_trips_through_meta():
    """save_models records it, load_models reads it back. Without the pair the
    predict path has nothing to rebuild the cohort from."""
    import inspect

    src = inspect.getsource(ItemForecaster)
    assert '"train_min_median_price": self._train_min_median_price' in src
    assert 'meta.get("train_min_median_price")' in src
    assert "self._train_min_median_price = min_median_price" in src


def test_training_publishes_its_cohort_to_the_predict_path():
    """train-then-predict in one process must read THIS run's floor.

    `_artifact_min_median_price` is otherwise whatever load_models() read from
    the cache before training -- absent on any artifact older than this change,
    which would make predict refuse to serve a model it had just trained.
    """
    import inspect

    src = inspect.getsource(ItemForecaster.train)
    assert "self._artifact_min_median_price = min_median_price" in src
    assert src.index("self._artifact_min_median_price") < src.index("self.horizon_feature_cols = {}"), (
        "the cohort must be published before training proceeds, not after"
    )


# ----------------------------------------------------------------------
# The skip set must follow the artifact, not the served frame
#
# Training ranked 32/32 columns; serving ranked 31/32, because macd_missing is
# within-date constant on the predict frame and the date-constant skip fired
# there only (run 31440424106). The column then reaches the booster raw where
# it was fitted ranked.
# ----------------------------------------------------------------------


def _two_date_frame(values):
    return pd.DataFrame(
        {
            "item_id": ["a", "b", "c"] * 2,
            "date": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-02"] * 3),
            "price": [10.0, 20.0, 30.0] * 2,
            "flag": values,
            "f": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )


def test_training_reports_which_columns_it_skipped():
    """The artifact cannot record a decision the transform does not surface."""
    df = _two_date_frame([0.0] * 6)  # constant on every date
    skipped = []
    ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["flag", "f"], skipped_out=skipped)
    assert skipped == ["flag"]


def test_serving_ranks_a_column_training_ranked_even_if_now_constant():
    """The macd_missing case. Training ranked it, so serving must too.

    Ranking an all-ties date yields exactly 0.0, which is what training
    produced on its own constant dates -- so following the artifact reproduces
    training rather than leaking a raw value in.
    """
    df = _two_date_frame([1.0] * 6)  # constant NOW, ranked at training
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["flag", "f"], skip_cols=[])
    assert out["flag"].tolist() == pytest.approx([0.0] * 6)
    # Not the raw 1.0 the served frame would otherwise have handed the booster.
    assert out["flag"].tolist() != pytest.approx([1.0] * 6)


def test_serving_leaves_raw_a_column_training_skipped():
    """The other direction: training fitted on raw, so serving must not rank."""
    df = _two_date_frame([0.0, 1.0, 2.0] * 2)  # varies now
    out = ItemForecaster._apply_cross_sectional_ranks(df.copy(), ["flag", "f"], skip_cols=["flag"])
    assert out["flag"].tolist() == pytest.approx([0.0, 1.0, 2.0] * 2)


def test_predict_refuses_when_the_artifact_records_no_skip_set():
    import inspect

    src = inspect.getsource(ItemForecaster.predict)
    guard = src.split("_cross_sectional_rank_served()")[1]
    assert "_artifact_xs_rank_skipped is None" in guard
    assert "skip_cols=" in guard
