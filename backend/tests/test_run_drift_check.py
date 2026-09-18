"""Tests for the drift check CI script."""

import pytest


class TestDriftCheckScript:
    def test_module_imports(self):
        """The drift check script should be importable."""
        from scripts import run_drift_check
        assert hasattr(run_drift_check, "main")

    def test_reference_snapshot_roundtrip(self, tmp_path):
        """Saving and loading a reference snapshot should be lossless."""
        import pandas as pd
        import numpy as np
        from scripts.run_drift_check import save_reference, load_reference

        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "price": rng.normal(100, 10, 100),
            "return_1d": rng.normal(0, 0.02, 100),
        })
        path = str(tmp_path / "ref.parquet")

        save_reference(ref, path)
        loaded = load_reference(path)

        pd.testing.assert_frame_equal(ref, loaded)
