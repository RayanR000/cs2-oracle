"""The ByMykel bundle's wiring into the feature pipeline.

Five things here are silent when they break, which is why each has a test:
the flag defaulting on, the nullable-Int64 columns being dropped by
`_select_feature_cols`, the allowlist admitting neighbouring refuted groups
along with the nine measured columns, `item_age_meta_days` colliding with
the unrelated `item_age_days` that `_add_temporal_features` already produces,
and `EXCEEDANCE_META` widening a group it is not supposed to widen.
"""

import sys
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.forecaster import (
    IncompatibleModelArtifact,
    ItemForecaster,
    _feature_group,
)

META_COLS = sorted(ItemForecaster.BYMYKEL_META_FEATURES)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("BYMYKEL_METADATA", "1")


@pytest.fixture
def meta_frame():
    return pd.DataFrame(
        {
            "item_slug": ["A", "B", "C"],
            "item_age_first_sale_date": pd.to_datetime(["2013-09-20", "2020-01-01", None]),
            "item_age_ambiguous": pd.array([1, 0, 0], dtype="int8"),
            "rarity_meta_rank": pd.array([6, 3, None], dtype="Int64"),
            "is_meta_stattrak": pd.array([1, 0, 0], dtype="int8"),
            "is_meta_souvenir": pd.array([0, 0, 0], dtype="int8"),
            "float_meta_min": [0.0, 0.1, None],
            "float_meta_max": [0.08, 0.7, None],
            "type_meta_crate_id": pd.array([12, 40, None], dtype="Int64"),
            "type_meta_collection_id": pd.array([3, None, None], dtype="Int64"),
        }
    )


@pytest.fixture
def price_frame():
    return pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "C", "D"],
            "date": pd.to_datetime(["2014-01-01", "2015-01-01", "2021-01-01", "2021-01-01", "2021-01-01"]),
        }
    )


@pytest.fixture
def forecaster(meta_frame, monkeypatch):
    fc = ItemForecaster.__new__(ItemForecaster)
    fc._bymykel_meta_cache = meta_frame.rename(columns={"item_slug": "item_id"})
    return fc


class TestFlag:
    def test_defaults_off(self, monkeypatch):
        # The effect is measured only on a held-out CV instrument. Nothing about
        # it has been read through the production retrain path.
        monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
        assert ItemForecaster.bymykel_metadata_enabled() is False

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", True),
            ("0", False),
            ("", False),
            ("true", False),
        ],
    )
    def test_only_the_literal_one_enables_it(self, monkeypatch, value, expected):
        monkeypatch.setenv("BYMYKEL_METADATA", value)
        assert ItemForecaster.bymykel_metadata_enabled() is expected

    def test_disabled_adds_no_columns(self, forecaster, price_frame, monkeypatch):
        monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
        out = forecaster._add_bymykel_metadata_features(price_frame.copy())
        assert not set(META_COLS) & set(out.columns)


class TestFeatureGroup:
    def test_every_bundle_column_gets_its_own_group(self):
        for col in META_COLS:
            assert _feature_group(col) == "bymykel_metadata"

    def test_the_bundle_does_not_leak_into_neighbouring_groups(self):
        # Each of these would be claimed by a prefix rule if the exact-name
        # lookup were removed, dragging the 2026-07-24 ablation's losers into
        # the allowlist alongside the nine measured columns.
        assert _feature_group("item_age_days") == "temporal"
        assert _feature_group("rarity_ordinal") == "item_identity"
        assert _feature_group("type_rifle") == "item_metadata"

    def test_allowlist_admits_the_bundle_and_nothing_else_new(self):
        cols = [*META_COLS, "price_mean_7d", "item_age_days", "rarity_ordinal", "type_rifle", "supply_listings"]
        kept = ItemForecaster._apply_feature_allowlist(cols, ["price_technicals", "bymykel_metadata"])
        assert sorted(kept) == sorted([*META_COLS, "price_mean_7d"])


class TestJoin:
    def test_adds_every_column(self, forecaster, price_frame, enabled):
        out = forecaster._add_bymykel_metadata_features(price_frame.copy())
        assert set(META_COLS).issubset(out.columns)

    def test_age_is_per_row_not_per_item(self, forecaster, price_frame, enabled):
        # Item A appears on two dates; its age must differ between them. A
        # stored "age as of today" column would give one value for both.
        out = forecaster._add_bymykel_metadata_features(price_frame.copy())
        a = out[out["item_id"] == "A"]["item_age_meta_days"].tolist()
        assert a == [103, 468]

    def test_does_not_touch_the_unrelated_item_age_days(self, forecaster, price_frame, enabled):
        df = price_frame.copy()
        df["item_age_days"] = 7.0
        out = forecaster._add_bymykel_metadata_features(df)
        assert out["item_age_days"].tolist() == [7.0] * 5
        assert "item_age_meta_days" in out.columns

    def test_unmatched_item_is_null_not_zero(self, forecaster, price_frame, enabled):
        out = forecaster._add_bymykel_metadata_features(price_frame.copy())
        row = out[out["item_id"] == "D"]
        assert pd.isna(row["item_age_meta_days"].iloc[0])
        assert pd.isna(row["rarity_meta_rank"].iloc[0])

    def test_missing_first_sale_date_yields_null_age(self, forecaster, price_frame, enabled):
        out = forecaster._add_bymykel_metadata_features(price_frame.copy())
        assert pd.isna(out[out["item_id"] == "C"]["item_age_meta_days"].iloc[0])

    def test_negative_age_is_nulled_not_clipped(self, forecaster, enabled):
        # Trading before the catalogue's first-sale date means the metadata is
        # wrong for that item, not that the item is brand new. Clipping to 0
        # would fabricate a real-looking value at a meaningful boundary.
        df = pd.DataFrame({"item_id": ["A"], "date": pd.to_datetime(["2010-01-01"])})
        out = forecaster._add_bymykel_metadata_features(df)
        assert pd.isna(out["item_age_meta_days"].iloc[0])

    def test_columns_survive_select_feature_cols(self, forecaster, price_frame, enabled):
        # The regression this exists for: the source columns are pandas nullable
        # Int64, and _select_feature_cols keeps a column only if its dtype is in
        # (float64, float32, int64, int, float). Int64 is none of those, so an
        # uncast column is dropped silently and the arm measures a clean null.
        out = forecaster._add_bymykel_metadata_features(price_frame.copy())
        kept = ItemForecaster._select_feature_cols(out, ItemForecaster.HORIZONS, frozenset())
        assert set(META_COLS).issubset(kept)

    def test_empty_metadata_is_a_no_op(self, price_frame, enabled):
        fc = ItemForecaster.__new__(ItemForecaster)
        fc._bymykel_meta_cache = pd.DataFrame()
        out = fc._add_bymykel_metadata_features(price_frame.copy())
        assert not set(META_COLS) & set(out.columns)


class TestArtifactGuard:
    def _forecaster(self):
        return ItemForecaster.__new__(ItemForecaster)

    def test_flag_mismatch_refuses_to_load(self, monkeypatch):
        monkeypatch.setenv("BYMYKEL_METADATA", "1")
        meta = {"model_artifact_version": ItemForecaster.MODEL_ARTIFACT_VERSION, "bymykel_metadata": False}
        with pytest.raises(IncompatibleModelArtifact, match="BYMYKEL_METADATA"):
            self._forecaster()._check_artifact_version(meta)

    def test_flag_mismatch_the_other_way_also_refuses(self, monkeypatch):
        monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
        meta = {"model_artifact_version": ItemForecaster.MODEL_ARTIFACT_VERSION, "bymykel_metadata": True}
        with pytest.raises(IncompatibleModelArtifact, match="BYMYKEL_METADATA"):
            self._forecaster()._check_artifact_version(meta)

    def test_matching_flag_loads(self, monkeypatch):
        monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
        meta = {"model_artifact_version": ItemForecaster.MODEL_ARTIFACT_VERSION, "bymykel_metadata": False}
        self._forecaster()._check_artifact_version(meta)

    def test_a_pre_flag_artifact_reads_as_disabled(self, monkeypatch):
        # Every existing artifact predates the key. Absent must mean "trained
        # without the bundle", which is true, rather than raising for everyone.
        monkeypatch.delenv("BYMYKEL_METADATA", raising=False)
        meta = {"model_artifact_version": ItemForecaster.MODEL_ARTIFACT_VERSION}
        self._forecaster()._check_artifact_version(meta)


class TestExceedanceMetaGate:
    """`EXCEEDANCE_META` widens only the exceedance head's matrix.

    Silent-when-broken in both directions: a gate that leaks would put nine
    refuted columns into the main model behind a flag that claims not to, and a
    gate that widens without the columns present would KeyError a whole retrain
    on any date the ByMykel join came back empty.
    """

    @staticmethod
    def _train_set():
        return pd.DataFrame(
            {
                "return_1d": [0.1, -0.2, 0.0],
                "price_zscore_30d": [1.0, 0.0, -1.0],
                "rarity_meta_rank": [6, 3, 1],
                "is_meta_stattrak": [1, 0, 0],
                "float_meta_min": [0.0, 0.07, 0.15],
                "target_exceed_7d": [1, 0, 1],
            }
        )

    FEATURE_COLS: ClassVar[list] = ["return_1d", "price_zscore_30d"]

    def test_defaults_off(self, monkeypatch):
        monkeypatch.delenv("EXCEEDANCE_META", raising=False)
        assert ItemForecaster.exceedance_meta_enabled() is False

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", True),
            ("0", False),
            ("", False),
            ("true", False),
        ],
    )
    def test_only_the_literal_one_enables_it(self, monkeypatch, value, expected):
        monkeypatch.setenv("EXCEEDANCE_META", value)
        assert ItemForecaster.exceedance_meta_enabled() is expected

    def test_disabled_matrix_is_exactly_the_allowlist(self, monkeypatch):
        monkeypatch.delenv("EXCEEDANCE_META", raising=False)
        fc = ItemForecaster.__new__(ItemForecaster)
        X = fc._exceedance_feature_matrix(self._train_set(), self.FEATURE_COLS)
        assert list(X.columns) == self.FEATURE_COLS

    def test_enabled_appends_present_meta_columns_sorted(self, monkeypatch):
        monkeypatch.setenv("EXCEEDANCE_META", "1")
        fc = ItemForecaster.__new__(ItemForecaster)
        X = fc._exceedance_feature_matrix(self._train_set(), self.FEATURE_COLS)
        # Allowlist order is preserved; the widening is appended, sorted, so the
        # column order is a function of the flag alone and not of frame order.
        assert list(X.columns) == [*self.FEATURE_COLS, "float_meta_min", "is_meta_stattrak", "rarity_meta_rank"]

    def test_enabled_with_no_meta_columns_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("EXCEEDANCE_META", "1")
        fc = ItemForecaster.__new__(ItemForecaster)
        bare = self._train_set().drop(columns=["rarity_meta_rank", "is_meta_stattrak", "float_meta_min"])
        X = fc._exceedance_feature_matrix(bare, self.FEATURE_COLS)
        assert list(X.columns) == self.FEATURE_COLS

    def test_a_meta_column_already_allowlisted_is_not_duplicated(self, monkeypatch):
        monkeypatch.setenv("EXCEEDANCE_META", "1")
        fc = ItemForecaster.__new__(ItemForecaster)
        cols = [*self.FEATURE_COLS, "rarity_meta_rank"]
        X = fc._exceedance_feature_matrix(self._train_set(), cols)
        assert list(X.columns) == [*cols, "float_meta_min", "is_meta_stattrak"]
