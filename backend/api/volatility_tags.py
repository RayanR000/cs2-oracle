"""Volatility/stability tag derivation - pure functions, no I/O.

The API surface turns a forecast band into a per-item volatility tag:
  - expected swing  = half-band / mid
  - a Stable/Moderate/Volatile label from the within-horizon swing tertiles

Kept separate from the routes so the arithmetic and the labelling rule are
unit-testable in isolation and the endpoints stay thin.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

Thresholds = tuple[float, float]

#: Horizons at which the exceedance head's `exceed_p` (served as `move_odds`) is
#: calibrated enough to publish. Replay reliability is ECE <1.3pp at h3/h7,
#: ~1.7pp (borderline) at h14 and ~3.8pp at h30, so only 3 and 7 are trusted.
#: Outside this set `move_odds` is suppressed (None) rather than published as a
#: number the data does not support — the same "never fabricate" rule as
#: `swing_pct`.
CALIBRATED_MOVE_ODDS_HORIZONS: tuple[int, ...] = (3, 7)


def move_odds_calibrated(horizon: int) -> bool:
    """Whether `move_odds` may be published at this horizon (h3/h7 only)."""
    return horizon in CALIBRATED_MOVE_ODDS_HORIZONS


def swing_pct(low: float | None, high: float | None, mid: float | None) -> float | None:
    """Half-band as a fraction of the mid: ((high - low) / 2) / mid.

    Returns None when any input is missing or the mid is non-positive, so a
    degenerate forecast never fabricates a swing.
    """
    if low is None or high is None or mid is None:
        return None
    if mid <= 0:
        return None
    return ((high - low) / 2.0) / mid


def compute_thresholds(swings: Iterable[float | None]) -> Thresholds:
    """The 1/3 and 2/3 quantile cut points of a swing distribution.

    None values are dropped. Raises ValueError on an empty distribution rather
    than inventing thresholds.
    """
    vals = [s for s in swings if s is not None]
    if not vals:
        raise ValueError("cannot compute thresholds from an empty distribution")
    lo, hi = np.quantile(vals, [1.0 / 3.0, 2.0 / 3.0])
    return float(lo), float(hi)


def label_for(swing: float | None, thresholds: Thresholds) -> str | None:
    """Stable (<= low tertile), Volatile (> high tertile), else Moderate.

    None swing has no label.
    """
    if swing is None:
        return None
    lo, hi = thresholds
    if swing <= lo:
        return "Stable"
    if swing > hi:
        return "Volatile"
    return "Moderate"


def tag_fields(
    low: float | None,
    high: float | None,
    mid: float | None,
    exceed_p: float | None,
    thresholds: Thresholds | None,
    calibrated_move_odds: bool = True,
) -> dict:
    """The three per-item tag fields from a band + exceed_p + universe thresholds.

    Swing is always derivable from the row alone; the label needs the
    within-horizon universe thresholds and is None when they are unavailable
    (empty universe) or the band is degenerate. `move_odds` is published only
    when `calibrated_move_odds` (h3/h7); otherwise None, since the probability
    is not trustworthy at that horizon (see `move_odds_calibrated`).
    """
    swing = swing_pct(low, high, mid)
    label = label_for(swing, thresholds) if thresholds is not None else None
    move_odds = exceed_p if calibrated_move_odds else None
    return {"expected_swing_pct": swing, "move_odds": move_odds, "stability_label": label}


_SORT_FIELDS = {"swing": "expected_swing_pct", "move_odds": "move_odds"}


def build_ranking(
    rows: Iterable[Mapping],
    sort: str = "swing",
    order: str = "desc",
    limit: int | None = None,
    calibrated_move_odds: bool = True,
    label: str | None = None,
) -> list[dict]:
    """Rank a universe of forecast rows by volatility.

    Each input row carries ``item_id, name, current_price, low, high, mid,
    exceed_p``. Rows with no computable swing (no band) are dropped. The
    Stable/Moderate/Volatile label is assigned from the tertiles of THIS call's
    swing distribution, so it is relative-to-peers, not an absolute cutoff.

    ``move_odds`` (from ``exceed_p``) is published only when
    ``calibrated_move_odds`` (h3/h7); otherwise it is None on every row, since
    the probability is not trustworthy at that horizon (see
    ``move_odds_calibrated``).

    ``label`` (Stable/Moderate/Volatile), when given, keeps only rows carrying
    it. The label is computed over the WHOLE input first, so the filter selects
    a tertile of the full universe -- it never re-tertiles the filtered subset.

    Sort key is ``swing`` (expected_swing_pct) or ``move_odds``; rows missing the
    sort value sort last in either direction. ``limit`` is applied after sorting.
    """
    tagged = []
    for r in rows:
        s = swing_pct(r.get("low"), r.get("high"), r.get("mid"))
        if s is None:
            continue  # no band -> not rankable
        tagged.append(
            {
                "item_id": r.get("item_id"),
                "name": r.get("name"),
                "current_price": r.get("current_price"),
                "expected_swing_pct": s,
                "move_odds": r.get("exceed_p") if calibrated_move_odds else None,
                "stability_label": None,  # filled once thresholds are known
            }
        )
    if not tagged:
        return []

    thresholds = compute_thresholds(t["expected_swing_pct"] for t in tagged)
    for t in tagged:
        t["stability_label"] = label_for(t["expected_swing_pct"], thresholds)

    if label is not None:
        tagged = [t for t in tagged if t["stability_label"] == label]

    field = _SORT_FIELDS.get(sort, "expected_swing_pct")
    have = [t for t in tagged if t[field] is not None]
    missing = [t for t in tagged if t[field] is None]
    have.sort(key=lambda t: t[field], reverse=(order != "asc"))
    ordered = have + missing  # missing sort last regardless of direction
    return ordered[:limit] if limit is not None else ordered
