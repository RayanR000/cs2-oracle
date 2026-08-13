"""The band-level decomposition's arithmetic, pinned on constructed panels.

`scripts/attribute_band_level.py` asks why the served band is 1.52-1.55x as wide as
the same `q_hat`'s calibration band. Its whole answer is one split — WHICH items are
served versus WHEN they are served — plus the observation that the band's coverage
responds to `|resid| / sigma` rather than to `sigma` alone. Both are asserted here
against panels whose answer is known by construction, never against the archive:
the archive read is the *input* to that arithmetic and moves week to week.

See `docs/changelog/2026-08-13-the-band-is-sized-on-trailing-volatility.md`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.attribute_band_level import decompose

ANCHOR = "2026-06-16"
OTHER = ["2026-06-09", "2026-06-10", "2026-06-11"]


def _panel(rows):
    """rows: (item_id, 'YYYY-MM-DD', sigma, |resid|)."""
    return pd.DataFrame(
        [{"item_id": i, "date": pd.Timestamp(d).date(), "sigma": s, "absr": r}
         for i, d, s, r in rows])


def _uniform(items, dates, sigma, absr):
    return [(i, d, sigma, absr) for i in items for d in dates]


def test_a_date_effect_is_reported_on_both_bases():
    """Every item's sigma is 1.5x on the anchor and its residual unchanged. The
    cohort is identical on every date, so WHICH cannot contribute and both bases
    must report the same thing -- a sigma ratio of 1.5 and a score ratio of 1/1.5.
    """
    items = [f"item-{n}" for n in range(20)]
    panel = _panel(_uniform(items, OTHER, 0.08, 4.0)
                   + _uniform(items, [ANCHOR], 0.12, 4.0))
    d = decompose(panel, [pd.Timestamp(ANCHOR).date()])
    assert d["same_sigma"] == pytest.approx(1.5)
    assert d["same_absr"] == pytest.approx(1.0)
    assert d["score_same"] == pytest.approx(1 / 1.5)
    assert d["pooled_sigma"] == pytest.approx(1.5)


def test_a_pure_composition_effect_vanishes_on_the_same_items_basis():
    """The distinguishing case, and the reason the second basis exists. Sigma is
    constant in time per item; the anchor simply serves the volatile half of the
    cohort. The pooled ratio must rise and the same-items ratio must be exactly 1
    -- otherwise 'which items' and 'when' are not separated and the instrument
    cannot tell a cohort artifact from a regime shift.
    """
    # The calm group is the majority on purpose: at a 50/50 split the pooled
    # median itself sits in the volatile half and the composition effect the test
    # is constructing would not show up in a median ratio at all.
    calm = [f"calm-{n}" for n in range(30)]
    wild = [f"wild-{n}" for n in range(10)]
    panel = _panel(_uniform(calm, OTHER + [ANCHOR], 0.05, 2.0)
                   + _uniform(wild, OTHER + [ANCHOR], 0.20, 8.0))
    # Only the volatile items are quoted on the anchor.
    panel = panel[(panel["date"] != pd.Timestamp(ANCHOR).date())
                  | (panel["item_id"].str.startswith("wild"))]
    d = decompose(panel, [pd.Timestamp(ANCHOR).date()])
    assert d["same_sigma"] == pytest.approx(1.0)
    assert d["same_absr"] == pytest.approx(1.0)
    assert d["score_same"] == pytest.approx(1.0)
    assert d["pooled_sigma"] > 1.5


def test_a_matched_residual_leaves_the_score_alone():
    """The null the whole entry turns on. An anchor twice as volatile in BOTH legs
    is not a calibration defect at all -- the band is wider because the item is
    genuinely wilder, and coverage is unchanged. If `score_*` moved here, the
    measured 0.65-0.76 could not be read as over-coverage.
    """
    items = [f"item-{n}" for n in range(20)]
    panel = _panel(_uniform(items, OTHER, 0.08, 4.0)
                   + _uniform(items, [ANCHOR], 0.16, 8.0))
    d = decompose(panel, [pd.Timestamp(ANCHOR).date()])
    assert d["same_sigma"] == pytest.approx(2.0)
    assert d["same_absr"] == pytest.approx(2.0)
    assert d["score_same"] == pytest.approx(1.0)


def test_an_item_served_but_never_pooled_elsewhere_does_not_inflate_the_ratio():
    """An item whose only row IS the anchor has itself as its own median, so its
    relative sigma is exactly 1 and it can neither create nor hide a date effect.
    Worth pinning: the served cohort at a real anchor contains such items, and a
    NaN or a division by a foreign median would silently bias the median ratio.
    """
    items = [f"item-{n}" for n in range(9)]
    panel = _panel(_uniform(items, OTHER, 0.08, 4.0)
                   + _uniform(items, [ANCHOR], 0.16, 4.0)
                   + [("newcomer", ANCHOR, 0.40, 20.0)])
    d = decompose(panel, [pd.Timestamp(ANCHOR).date()])
    assert d["n_served"] == 10
    assert d["same_sigma"] == pytest.approx(2.0)   # the newcomer's own is 1.0
    assert np.isfinite(d["score_same"])


def test_no_served_row_returns_empty_rather_than_a_ratio_of_nothing():
    panel = _panel(_uniform(["a", "b"], OTHER, 0.08, 4.0))
    assert decompose(panel, [pd.Timestamp(ANCHOR).date()]) == {}
