"""Served-outcome feedback calibration.

The served conformal band over-covers (prod panel 87-92% against an 80% target) because a
scalar q_hat, pooled over a calibration window that mixes volatility regimes, is served on
individual dates. The offline conditioning remedies are exhausted; the one that directly
targets SERVED coverage is to recalibrate against the served outcomes themselves.

This module re-runs split conformal on the served `forecast_outcomes` panel: per horizon, the
factor that would have made the served (asymmetric, mid-anchored) band cover at the 80%
nominal. It multiplies q_hat at serve time (`ItemForecaster.served_qhat_multiplier`). It is a
LEVEL correction only — no conditional band — and it replaces the calibration POPULATION
(OOF -> served) rather than transferring a shape between them, which is why it sidesteps the
prior "different populations" refutations (docs/changelog/2026-08-12-served-sigma-profile.md).

Gated on MIN_FORECAST_DATES distinct served dates per horizon (data-blocked today: ~0-4 exist),
so it ships dormant and self-activates. Clamped so a contaminated panel cannot wreck the band.
Design: docs/superpowers/specs/2026-08-16-served-outcome-feedback-calibration-design.md.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from backtest.scoring import HEADLINE_MIN_TIER, MIN_FORECAST_DATES
from models.conformal import ALPHA

logger = logging.getLogger(__name__)

# A single application cannot more than double or halve the band, whatever the panel says.
# A clamp hit is a data-quality signal (a bad cutover, a resolver regression), not a tuning
# outcome — mirrors conformal.fit_beta's clamp philosophy.
FACTOR_MIN = 0.5
FACTOR_MAX = 2.0

# The signed conformal band (asymmetric offsets about the q50 mid) changed the served band's
# GEOMETRY. Rows served under the OLD geometry carry a symmetric band whose mid was moved by
# `_recenter_on_direction`; the factor's nonconformity score `r` reads the STORED band shape,
# so a factor pooled across the cutover calibrates a shape that matches neither band. The
# `forecast_outcomes` panel has no geometry marker (no model_version column, and the served
# `MODEL_VERSION` string deliberately does not encode config), so the only available separator
# is the forecast_date. Set this to the date the signed band served its FIRST prod forecast.
# Until then every stored row is old-geometry, so the multiplier stays dormant (the caller gets
# {}) regardless of the date count — an unfiltered factor would calibrate the wrong band.
SIGNED_BAND_SERVING_START: Optional[str] = None  # ISO date, e.g. "2026-08-25"; set at deploy.

# The columns the estimator needs from forecast_outcomes.
PANEL_COLUMNS = (
    "forecast_date", "horizon_days", "price_tier",
    "predicted_price_low", "predicted_price_mid", "predicted_price_high",
    "actual_price",
)


def _conformal_level(n: int, alpha: float) -> float:
    """The finite-sample corrected split-conformal level, identical to conformal.calibrate."""
    return min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)


def factors_from_panel(panel: pd.DataFrame, horizons: Iterable[int], *,
                       min_dates: int = MIN_FORECAST_DATES, alpha: float = ALPHA,
                       min_tier: int = HEADLINE_MIN_TIER,
                       since: Optional[str] = None) -> Dict[int, float]:
    """Per-horizon served-coverage q_hat multiplier from a forecast_outcomes frame.

    For each horizon, over the >= $1 served rows with a usable band, the nonconformity score is
    the ratio to the relevant HALF about the served mid (the band is asymmetric):

        r = (actual - mid) / (high - mid)   if actual >= mid
          = (mid - actual) / (mid - low)    otherwise

    and the factor is the split-conformal quantile of `r` — the scale `s` that covers a row iff
    `s >= r`, so `P80(r)` lands 80% coverage on the panel. Horizons with fewer than `min_dates`
    distinct served dates are OMITTED from the result (the caller falls back to 1.0). The factor
    is clamped to [FACTOR_MIN, FACTOR_MAX].

    `since` (ISO date) is the signed-band geometry floor: rows with `forecast_date < since` are
    dropped before anything else, because `r` reads the stored band shape and pre-cutover rows
    carry the old symmetric geometry (see SIGNED_BAND_SERVING_START). None keeps every row.
    """
    out: Dict[int, float] = {}
    if panel is None or panel.empty:
        return out

    df = panel[panel["price_tier"].to_numpy() >= min_tier]
    if since is not None:
        df = df[pd.to_datetime(df["forecast_date"]).to_numpy() >= np.datetime64(since)]
    for h in horizons:
        rows = df[df["horizon_days"].to_numpy() == h]
        if rows.empty:
            continue

        mid = rows["predicted_price_mid"].to_numpy(dtype=float)
        low = rows["predicted_price_low"].to_numpy(dtype=float)
        high = rows["predicted_price_high"].to_numpy(dtype=float)
        actual = rows["actual_price"].to_numpy(dtype=float)

        above = actual >= mid
        half = np.where(above, high - mid, mid - low)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(above, actual - mid, mid - actual) / half
        # Drop rows with a degenerate half (<= 0) or any non-finite input.
        keep = np.isfinite(r) & (half > 0)
        r = r[keep]
        dates = rows["forecast_date"].to_numpy()[keep]

        n_dates = pd.unique(dates).size
        if n_dates < min_dates or r.size == 0:
            continue

        raw = float(np.quantile(r, _conformal_level(r.size, alpha)))
        factor = float(np.clip(raw, FACTOR_MIN, FACTOR_MAX))
        if factor != raw:
            logger.warning(
                f"  {h}d served-coverage factor clamped {raw:.4f} -> {factor:.4f} "
                f"(outside [{FACTOR_MIN}, {FACTOR_MAX}], {r.size:,} rows over {n_dates} dates) "
                f"— treat the served panel as suspect for this horizon."
            )
        else:
            logger.info(
                f"  {h}d served-coverage factor {factor:.4f} on {r.size:,} served rows "
                f"over {n_dates} dates (>= {min_dates}); q_hat is scaled by it at serve."
            )
        out[h] = factor
    return out


def _load_panel(session, horizons: Iterable[int], *,
                since: Optional[str] = None) -> pd.DataFrame:
    """Read the scored forecast_outcomes panel from Postgres (the ops parquet mirrors are
    stale/selected — see backend/AGENTS.md). Read-only, columns narrowed to the estimator's.
    `since` (ISO date), when set, floors forecast_date to the signed-band geometry cutover."""
    from sqlalchemy import bindparam, text

    hs = [int(h) for h in horizons]
    where = ("WHERE horizon_days IN :horizons AND actual_price IS NOT NULL "
             "AND predicted_price_mid IS NOT NULL")
    params: dict = {"horizons": hs}
    if since is not None:
        where += " AND forecast_date >= :since"
        params["since"] = since
    sql = text(
        "SELECT forecast_date, horizon_days, price_tier, "
        "predicted_price_low, predicted_price_mid, predicted_price_high, actual_price "
        f"FROM forecast_outcomes {where}"
    ).bindparams(bindparam("horizons", expanding=True))
    rows = session.execute(sql, params).mappings().all()
    return pd.DataFrame(rows, columns=list(PANEL_COLUMNS))


def served_coverage_factors(session, horizons: Iterable[int], *,
                            min_dates: int = MIN_FORECAST_DATES,
                            alpha: float = ALPHA,
                            since: Optional[str] = SIGNED_BAND_SERVING_START) -> Dict[int, float]:
    """Per-horizon served-coverage q_hat multipliers, or an empty map.

    Returns {} on any read failure or when no horizon clears the gate, so a missing/empty panel
    (the data-blocked default today) leaves the band byte-identical to a no-feedback artifact.

    Returns {} outright when `since` is None (SIGNED_BAND_SERVING_START unset): the whole panel is
    then pre-signed-band geometry, and a factor fit on it would calibrate the wrong band shape and
    apply it to the signed offsets. The feedback stays dormant until the cutover date is set.
    """
    if session is None:
        return {}
    if since is None:
        logger.info(
            "  served-coverage: SIGNED_BAND_SERVING_START unset; feedback dormant — the panel is "
            "pre-signed-band geometry, so no q_hat correction is applied.")
        return {}
    try:
        panel = _load_panel(session, horizons, since=since)
    except Exception as e:                       # a panel read must never fail a retrain
        logger.warning(f"  served-coverage panel read failed ({e}); no q_hat correction applied.")
        return {}
    return factors_from_panel(panel, horizons, min_dates=min_dates, alpha=alpha, since=since)
