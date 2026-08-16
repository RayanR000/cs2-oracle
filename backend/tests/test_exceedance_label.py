"""`prepare_targets` must emit a one-sided exceedance label `target_exceed_{h}d`.

`target_exceed_{h}d = 1[ target_return_{h}d > 100 * actionable_threshold(tier, csfloat) ]`,
NaN wherever `target_return_{h}d` is NaN (missing target or voided by the snapshot /
collector-cutover / frozen-run rules), so the exceedance head trains on exactly the rows
the range model does. Scope: docs/superpowers/plans/2026-08-16-exceedance-band-scale-phase2-plan.md.

The production change that makes this fail if reverted: the label block appended to
`prepare_targets` after label voiding.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from models.forecaster import ItemForecaster


def _forecaster(tmp_path):
    # Never the default model_dir — it clobbers gitignored production artifacts.
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _frame():
    # One item, strictly-varying prices (so the frozen-run rule voids nothing),
    # tier 2 throughout ($5-20). 3d-forward returns are hand-chosen around the
    # csfloat tier-2 threshold (2.0% + 17.3% = 19.3%).
    prices = {
        "2026-01-01": 10.00,   # ->01-04 13.00 : +30.0%  > 19.3%  => exceed 1
        "2026-01-02": 10.10,   # ->01-05 10.35 : +2.48%           => exceed 0
        "2026-01-03": 10.20,
        "2026-01-04": 13.00,
        "2026-01-05": 10.35,
        "2026-01-06": 10.40,
        "2026-01-07": 10.50,
        "2026-01-08": 10.60,
        "2026-01-09": 10.70,
        "2026-01-10": 10.80,   # ->01-13 absent => target NaN => exceed NaN
    }
    return pd.DataFrame({
        "item_id": "A",
        "date": [pd.to_datetime(d).date() for d in prices],
        "price": list(prices.values()),
    })


def test_exceedance_label_is_one_sided_and_inherits_voiding(tmp_path):
    f = _forecaster(tmp_path)
    out = f.prepare_targets(_frame(), horizon=3).set_index("date")
    col = "target_exceed_3d"

    assert col in out.columns
    assert out.loc[pd.to_datetime("2026-01-01").date(), col] == 1.0   # +30% clears 19.3%
    assert out.loc[pd.to_datetime("2026-01-02").date(), col] == 0.0   # +2.5% does not
    # Missing target -> return is NaN -> exceedance label must be NaN, not 0.
    assert np.isnan(out.loc[pd.to_datetime("2026-01-10").date(), col])


def test_exceedance_label_is_never_selected_as_a_feature(tmp_path):
    # target_exceed_{h}d is a target-like float column prepare_targets adds; it must
    # be excluded from feature selection like target_return_{h}d, so a post-
    # prepare_targets frame can never leak the answer into the feature matrix.
    f = _forecaster(tmp_path)
    out = f.prepare_targets(_frame(), horizon=3)
    cols = ItemForecaster._select_feature_cols(out, [3], shelved=[])
    assert "target_exceed_3d" not in cols
    assert "target_return_3d" not in cols   # guards the existing invariant too
