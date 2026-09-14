"""VOLUME_FEATURES gates whether the shelved volume columns reach training.

Task 5 joins a recovered ``volume`` column into feature engineering; this
flag is what lets the volume-derived columns (still all in
``SHELVED_FEATURES`` per ``test_volume_features_shelved.py``) actually reach
``_select_feature_cols`` when explicitly turned on. Off by default so
production behaviour is unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.forecaster import ItemForecaster


def test_volume_names_shelved_by_default(monkeypatch):
    monkeypatch.delenv("VOLUME_FEATURES", raising=False)
    f = ItemForecaster.__new__(ItemForecaster)
    active = f._active_shelved_features()
    assert active >= ItemForecaster.VOLUME_FEATURE_NAMES  # still shelved


def test_flag_unshelves_volume_names(monkeypatch):
    monkeypatch.setenv("VOLUME_FEATURES", "1")
    f = ItemForecaster.__new__(ItemForecaster)
    active = f._active_shelved_features()
    assert not (ItemForecaster.VOLUME_FEATURE_NAMES & active)  # none still shelved


def test_flag_off_for_other_values(monkeypatch):
    monkeypatch.setenv("VOLUME_FEATURES", "0")
    assert ItemForecaster._volume_features_enabled() is False


def test_non_volume_shelved_features_are_unaffected_by_the_flag(monkeypatch):
    monkeypatch.setenv("VOLUME_FEATURES", "1")
    f = ItemForecaster.__new__(ItemForecaster)
    active = f._active_shelved_features()
    non_volume_shelved = ItemForecaster.SHELVED_FEATURES - ItemForecaster.VOLUME_FEATURE_NAMES
    assert non_volume_shelved <= active
