"""The artifact must record WHICH shrink constant built it.

`CLIMATOLOGY_SHRINK_K` moved 20 -> 320 on 2026-08-26, and that is a
band-geometry change: `served_recalibration.SHRINK_K_SERVING_START` has to be
set to the date the first K=320 artifact SERVED. Nothing recorded K, so
identifying that deploy meant remembering which run retrained -- and a
predict-only run on a K=20 cache serves the old geometry while looking
identical. Persisting K makes the cutover derivable from the artifact.

It is deliberately NOT part of MODEL_ARTIFACT_VERSION, for the reason the
neighbouring flags are not: an artifact written before the key existed is
otherwise byte-identical, so bumping would force every checkout into a
needless retrain.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.forecaster import ItemForecaster  # noqa: E402


@pytest.fixture
def fc(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(),
                          model_dir=str(tmp_path_factory.mktemp("saved_models")))


class TestTheArtifactField:
    def test_a_fresh_forecaster_has_no_artifact_k(self, fc):
        assert fc._artifact_climatology_shrink_k is None

    def test_served_k_falls_back_to_the_code_constant(self, fc):
        assert fc.climatology_shrink_k_served() == ItemForecaster.CLIMATOLOGY_SHRINK_K

    def test_served_k_follows_the_artifact_when_it_records_one(self, fc):
        fc._artifact_climatology_shrink_k = 20
        assert fc.climatology_shrink_k_served() == 20


class TestGeometryMismatchIsDetectable:
    def test_a_matching_artifact_reports_no_mismatch(self, fc):
        fc._artifact_climatology_shrink_k = ItemForecaster.CLIMATOLOGY_SHRINK_K
        assert fc.climatology_geometry_matches_code() is True

    def test_an_artifact_built_at_the_old_k_reports_a_mismatch(self, fc):
        fc._artifact_climatology_shrink_k = 20
        assert fc.climatology_geometry_matches_code() is False

    def test_a_legacy_artifact_is_not_flagged(self, fc):
        """Pre-key artifacts record nothing. Their K is unknowable, so claiming
        a mismatch would fire on every old cache and train nobody to trust it."""
        fc._artifact_climatology_shrink_k = None
        assert fc.climatology_geometry_matches_code() is True


class TestItIsWiredIntoTheArtifact:
    def test_save_models_writes_the_key(self):
        src = inspect.getsource(ItemForecaster.save_models)
        assert '"climatology_shrink_k"' in src

    def test_the_load_path_reads_the_key(self):
        src = inspect.getsource(ItemForecaster.load_models)
        assert "climatology_shrink_k" in src
