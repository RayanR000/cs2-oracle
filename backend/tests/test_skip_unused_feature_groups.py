"""Don't engineer 8 blocks the allowlist throws away -- but keep the full path.

Measured 2026-08-09: engineer_features is 17.5s and 8.5s of that (48%) is
blocks whose columns FEATURE_GROUP_ALLOWLIST then drops.

The two constraints this file exists to pin:
1. Seven ab_test_* harnesses build their own frame and call
   _apply_feature_allowlist on it, so the full 123-column path must survive.
2. The skip set must be DERIVED from the allowlist. If it is hard-coded and
   `cross_sectional` is later re-admitted (Track C4), the frame would carry the
   group in the allowlist and its columns absent -- median-filled to zero,
   undetectably.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from models.forecaster import ItemForecaster


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def test_default_is_off_so_harnesses_keep_the_full_frame(tmp_path):
    sig = inspect.signature(ItemForecaster.engineer_features)
    assert sig.parameters["skip_unused_groups"].default is False


def test_skip_set_excludes_the_allowlisted_group(tmp_path):
    f = _f(tmp_path)
    assert "price_technicals" not in f._skipped_feature_groups()


def test_skip_set_covers_the_discarded_groups(tmp_path):
    f = _f(tmp_path)
    skipped = f._skipped_feature_groups()
    for group in ("temporal", "events", "cross_sectional", "social", "item_identity", "item_metadata", "supply_depth"):
        assert group in skipped


def test_skip_set_follows_the_allowlist_not_a_literal(tmp_path):
    """Track C4 re-admits cross_sectional. The skip set must follow."""
    f = _f(tmp_path)
    f.FEATURE_GROUP_ALLOWLIST = ["price_technicals", "cross_sectional"]
    assert "cross_sectional" not in f._skipped_feature_groups()
    assert "temporal" in f._skipped_feature_groups()


def test_empty_allowlist_skips_nothing(tmp_path):
    """An empty allowlist means every group is kept, so nothing may be skipped."""
    f = _f(tmp_path)
    f.FEATURE_GROUP_ALLOWLIST = []
    assert f._skipped_feature_groups() == set()
