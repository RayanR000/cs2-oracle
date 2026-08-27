"""`replay_serving` must be pointable at a scratch artifact.

The replay serves whatever artifact it loads, and it hardcoded the deployed
directory -- so comparing two artifacts (a band-geometry arm against its
control) meant swapping `models/saved_models/` in place, which is exactly the
clobber `forecast_prices._model_dir` exists to prevent. This gives the replay the
same `FORECAST_MODEL_DIR` override, so an arm and its control can be replayed at
one anchor without touching the deployed model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import replay_serving as rs  # noqa: E402


class TestModelDirOverride:
    def test_unset_keeps_the_deployed_default(self, monkeypatch):
        monkeypatch.delenv("FORECAST_MODEL_DIR", raising=False)
        assert rs._model_dir() is None

    def test_empty_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("FORECAST_MODEL_DIR", "")
        assert rs._model_dir() is None

    def test_a_path_is_returned_and_created(self, monkeypatch, tmp_path):
        target = tmp_path / "arm" / "artifact"
        monkeypatch.setenv("FORECAST_MODEL_DIR", str(target))
        got = rs._model_dir()
        assert got == str(target)
        assert target.is_dir()

    def test_a_tilde_path_is_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FORECAST_MODEL_DIR", "~")
        assert rs._model_dir() == str(Path("~").expanduser())


class TestItIsTheSameContractAsTheTrainer:
    def test_both_scripts_read_the_same_variable(self, monkeypatch, tmp_path):
        """A different env name would silently replay the deployed artifact
        while the trainer wrote the arm somewhere else -- the arm and control
        would then be the same numbers."""
        import forecast_prices as fp

        monkeypatch.setenv("FORECAST_MODEL_DIR", str(tmp_path / "shared"))
        assert rs._model_dir() == fp._model_dir()
