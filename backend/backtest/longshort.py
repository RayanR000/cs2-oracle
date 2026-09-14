"""Cross-sectional long-short read-outs for the C2 serving-transfer diagnostic.

Pure functions over (score, realised, dates). The ranker emits an ordinal
score with no return scale, so its only honest read-outs are the ORDERING it
implies: the within-date rank IC (in `forecaster._within_date_rank_ic`), the
top-minus-bottom decile spread here, and a within-date above/below-median
directional call for the Pesaran-Timmermann test. No level metric belongs here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _grouped(score, realised, dates, mask):
    p = np.asarray(score, dtype=float)
    a = np.asarray(realised, dtype=float)
    d = pd.to_datetime(pd.Series(dates).to_numpy())
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        p, a, d = p[m], a[m], d[m]
    fin = np.isfinite(p) & np.isfinite(a)
    return pd.DataFrame({"d": d[fin], "p": p[fin], "a": a[fin]})


def decile_longshort_by_date(score, realised, dates, mask=None, decile: float = 0.1, min_rows: int = 20) -> dict:
    """Per date: mean realised of the top-`decile` by score minus the bottom.

    A date with fewer than `min_rows` scorable rows contributes nothing (the
    same floor `_within_date_rank_ic` uses), so the two reads describe one
    calendar. `mean` is None when no date qualifies.
    """
    frame = _grouped(score, realised, dates, mask)
    per_date: dict = {}
    for day, g in frame.groupby("d"):
        if len(g) < min_rows:
            continue
        k = max(1, int(round(len(g) * decile)))
        order = g.sort_values("p")
        bottom = order["a"].iloc[:k].mean()
        top = order["a"].iloc[-k:].mean()
        per_date[str(pd.Timestamp(day).date())] = float(top - bottom)
    mean = float(np.mean(list(per_date.values()))) if per_date else None
    return {"per_date": per_date, "mean": mean, "n_dates": len(per_date)}


def net_of_cost(spread: float, roundtrip: float) -> float:
    """The decile spread after the microstructure haircut.

    `roundtrip` is the SUMMED cost the long and short legs pay (15% Steam fee +
    tier spread, doubled), supplied by the caller — this function does not know
    the venue. A relative edge is only actionable if it clears this; but it is a
    tradeability conditioner, not the pass/fail bar.
    """
    return spread - roundtrip


def direction_records(score, realised, dates, mask=None, min_rows: int = 20) -> list[dict]:
    """Within-date above/below-median directional call, for `pesaran_timmermann`.

    Maps the ranker's ordinal score to a served direction the only way that
    respects its scale-freedom: above its own within-date median → "up". The
    actual leg is the same split on realised return. This is the ranker's
    directional skill on the cross-section, which is what invariant 4's PT test
    scores. Thin dates (< `min_rows`) are dropped, matching the rank-IC floor.
    """
    frame = _grouped(score, realised, dates, mask)
    out: list[dict] = []
    for day, g in frame.groupby("d"):
        if len(g) < min_rows:
            continue
        pmed, amed = g["p"].median(), g["a"].median()
        fd = str(pd.Timestamp(day).date())
        for p, a in zip(g["p"], g["a"]):
            pdir = "up" if p > pmed else "down"
            adir = "up" if a > amed else "down"
            out.append(
                {
                    "predicted_direction": pdir,
                    "actual_direction": adir,
                    "direction_correct": pdir == adir,
                    "forecast_date": fd,
                }
            )
    return out
