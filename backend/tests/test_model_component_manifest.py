"""Component manifest: descriptive, never authoritative.

save_models() records which components an artifact carries and which centre
is champion per horizon. Code configuration selects champions; on load a
GBM champion requires its q50 artifact while featureless last_price
requires none. Missing ranking artifacts disable only the ranking shadow.
"""

import json
from unittest.mock import MagicMock


def _forecaster(tmp_path):
    from models.forecaster import ItemForecaster

    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def test_manifest_present_after_save(tmp_path):
    f = _forecaster(tmp_path)
    f.save_models()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert set(meta["components"]) == {"centre", "interval", "anomaly", "exceedance", "ranking", "direction"}
    assert meta["components"]["centre"] == {"gbm_q50": f.MODEL_ARTIFACT_VERSION}
    assert meta["components"]["direction"] == {}
    assert meta["centre_champions"] == {"3": "gbm_q50", "7": "gbm_q50", "14": "gbm_q50", "30": "gbm_q50"}


def test_manifest_reflects_trained_heads(tmp_path):
    f = _forecaster(tmp_path)
    f.anomaly_models = {3: MagicMock()}
    f.exceedance_models = {3: MagicMock()}
    f.ranking_models = {7: MagicMock()}
    f.save_models()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert "anomaly_gbm_v1" in meta["components"]["anomaly"]
    assert "exceedance_gbm_v1" in meta["components"]["exceedance"]
    assert "lambdarank_v1" in meta["components"]["ranking"]


def test_gbm_champion_reports_missing_artifacts(tmp_path):
    from models.forecaster import ItemForecaster

    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.models = {(3, 0.5): object()}
    assert f._require_centre_artifacts() == [7, 14, 30]


def test_last_price_champion_requires_no_artifact(tmp_path, monkeypatch):
    from models.candidate_predictions import CENTRE_CHAMPIONS
    from models.forecaster import ItemForecaster

    monkeypatch.setitem(CENTRE_CHAMPIONS, 7, "last_price")
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.models = {(h, 0.5): object() for h in (3, 14, 30)}
    assert f._require_centre_artifacts() == []


def test_empty_model_dir_requires_nothing(tmp_path):
    from models.forecaster import ItemForecaster

    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    assert f._require_centre_artifacts() == []  # no artifact yet: retrain path, not an error
