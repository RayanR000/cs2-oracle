"""The label path must say which dates it voided.

`_collection_shift_dates` fires 12 times in 4,735 days and the list of dates has
never been written down anywhere -- it exists only as a count in a code comment.
The measured cost of that: a 2026-08-09 audit re-reported the already-handled
2026-07-09/10 cutover as a new, undocumented consensus break, and put a no-op
task at the top of a roadmap.

These tests pin the audit trail. They do NOT touch either detector's logic --
the universe-size rule is what makes it impossible for the detector to mask a
real crash.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from models.forecaster import ItemForecaster


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _frame_with_a_cutover(n_items=60, n_dates=40, cutover_at=20):
    """The item universe halves on one day -- a collector cutover by the
    universe-size rule, with no price move that would itself fire a detector.

    A per-item price held EXACTLY flat across all `n_dates` days would make
    every day but the first match `_snapshot_dates`' "100% of the
    cross-section repeats yesterday" rule too, since nothing here ever varies
    it -- that would flag ~19 of the 40 days as re-published snapshots and
    swamp the cutover-specific signal this fixture exists to isolate. The
    `d * 1e-6` term breaks the exact-float-equality both `_snapshot_dates` and
    the frozen-run rule test for, while staying too small to move any
    threshold or the item ordering by price.
    """
    rows = []
    start = pd.Timestamp("2026-01-01")
    for d in range(n_dates):
        live = n_items if d < cutover_at else n_items // 3
        for item in range(live):
            rows.append({
                "item_id": f"item-{item}",
                # prepare_targets merges this against a future-frame date column
                # normalised to python `date` (see its `.dt.date` cast); a
                # datetime64 column here trips a pandas dtype-mismatch merge
                # error, not label-voiding logic, so match that convention.
                "date": (start + pd.Timedelta(days=d)).date(),
                "price": 10.0 + item * 0.01 + d * 1e-6,
            })
    return pd.DataFrame(rows)


def _frame_with_flat_prices(n_items=60, n_dates=40):
    """Same population as `_frame_with_a_cutover`, minus the `d * 1e-6` drift
    and the cutover -- every item's price is bit-identical on every day it
    appears, so this is what `_frame_with_a_cutover` would have been before
    that fix, and it exists specifically to fire `_snapshot_dates`: >=99% of a
    >=25-item cross-section repeating the previous day's price exactly.
    `_frame_with_a_cutover` can no longer exercise `snapshot_dates` at all
    (that is the point of its drift), so this fixture is the only coverage
    for the ISO-string/sorted contract on that key.
    """
    rows = []
    start = pd.Timestamp("2026-01-01")
    for d in range(n_dates):
        for item in range(n_items):
            rows.append({
                "item_id": f"item-{item}",
                "date": (start + pd.Timedelta(days=d)).date(),
                "price": 10.0 + item * 0.01,
            })
    return pd.DataFrame(rows)


def test_audit_is_empty_before_prepare_targets(tmp_path):
    assert _f(tmp_path).label_voiding == {}


def test_audit_records_the_shift_date(tmp_path):
    f = _f(tmp_path)
    f.prepare_targets(_frame_with_a_cutover(), horizon=3)
    shifts = f.label_voiding["collection_shift_dates"]
    assert "2026-01-21" in shifts, f"cutover day not recorded: {shifts}"


def test_audit_records_the_frame_window(tmp_path):
    """Absence from the list must be distinguishable from 'outside the window'.
    The training window is days_back=1460, so the 2013 and 2016 cutovers fall
    outside it on most runs."""
    f = _f(tmp_path)
    f.prepare_targets(_frame_with_a_cutover(), horizon=3)
    lo, hi = f.label_voiding["frame_date_range"]
    assert lo == "2026-01-01"
    assert hi == "2026-02-09"


def test_audit_counts_voided_labels_per_horizon(tmp_path):
    f = _f(tmp_path)
    f.prepare_targets(_frame_with_a_cutover(), horizon=3)
    f.prepare_targets(_frame_with_a_cutover(), horizon=7)
    counts = f.label_voiding["voided_labels_by_horizon"]
    assert set(counts) == {3, 7}
    assert counts[7] >= counts[3], (
        "a longer horizon spans the cutover from more anchor dates, so it "
        "cannot void fewer labels")


def test_dates_are_sorted_iso_strings(tmp_path):
    """meta.json is JSON; a frozenset of datetime.date is not serialisable, and
    an unsorted list is not diffable across runs."""
    f = _f(tmp_path)
    f.prepare_targets(_frame_with_a_cutover(), horizon=3)
    for key in ("snapshot_dates", "collection_shift_dates"):
        got = f.label_voiding[key]
        assert isinstance(got, list)
        assert all(isinstance(d, str) for d in got)
        assert got == sorted(got)


def _frame_with_one_frozen_item(n_items=60, n_dates=40, frozen_item=0):
    """One item's price never moves; every other item drifts by its own daily
    increment, so neither the cross-section-wide `_snapshot_dates` rule (needs
    >=99% of a >=25-item cross-section to repeat the previous day) nor the
    universe-size `_collection_shift_dates` rule fires. Only the per-item
    frozen-run rule should touch this frame, isolating it from the other two
    voiding rules `bad` unions together."""
    rows = []
    start = pd.Timestamp("2026-01-01")
    for d in range(n_dates):
        for item in range(n_items):
            price = 10.0 if item == frozen_item else (
                10.0 + item * 0.01 + d * (item + 1) * 1e-3)
            rows.append({
                "item_id": f"item-{item}",
                "date": (start + pd.Timedelta(days=d)).date(),
                "price": price,
            })
    return pd.DataFrame(rows)


def test_frozen_run_labels_published_separately_from_voided_total(tmp_path):
    """`voided_labels_by_horizon` is the union of snapshot days, collector
    cutovers, AND the frozen-price-run rule -- the audit must not let a
    reader attribute the whole count to cutovers/snapshots when the frozen-run
    rule is what actually did it, so `frozen_run_labels` has to exist as its
    own key and be no larger than the union it is a part of."""
    f = _f(tmp_path)
    f.prepare_targets(_frame_with_one_frozen_item(), horizon=3)
    assert f.label_voiding["collection_shift_dates"] == [], (
        "fixture must not also trip the cutover rule, or it isn't isolating "
        "the frozen-run rule")
    assert f.label_voiding["snapshot_dates"] == [], (
        "fixture must not also trip the snapshot rule, or it isn't isolating "
        "the frozen-run rule")
    frozen = f.label_voiding["frozen_run_labels"]
    assert frozen[3] > 0, "the frozen item's own labels must be counted"
    assert frozen[3] <= f.label_voiding["voided_labels_by_horizon"][3]


def test_dates_are_sorted_iso_strings_for_a_populated_snapshot_list(tmp_path):
    """The loop above never exercises a non-empty `snapshot_dates`:
    `_frame_with_a_cutover`'s drift (needed to keep the cutover assertions
    from being swamped by a spurious snapshot-day confound -- see its
    docstring) means that fixture always yields `snapshot_dates == []`, and
    the ISO-string/sorted checks pass vacuously on an empty list. The real
    archive does populate this key (Step 8 measurement: 2026-07-16,
    2026-07-22), so pin the contract against a fixture that actually fires
    `_snapshot_dates`."""
    f = _f(tmp_path)
    f.prepare_targets(_frame_with_flat_prices(), horizon=3)
    got = f.label_voiding["snapshot_dates"]
    assert got, "flat prices across a >=25-item cross-section must fire _snapshot_dates"
    assert isinstance(got, list)
    assert all(isinstance(d, str) for d in got)
    assert got == sorted(got)
