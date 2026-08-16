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
                       min_tier: int = HEADLINE_MIN_TIER) -> Dict[int, float]:
    """Per-horizon served-coverage q_hat multiplier from a forecast_outcomes frame.

    For each horizon, over the >= $1 served rows with a usable band, the nonconformity score is
    the ratio to the relevant HALF about the served mid (the band is asymmetric):

        r = (actual - mid) / (high - mid)   if actual >= mid
          = (mid - actual) / (mid - low)    otherwise

    and the factor is the split-conformal quantile of `r` — the scale `s` that covers a row iff
    `s >= r`, so `P80(r)` lands 80% coverage on the panel. Horizons with fewer than `min_dates`
    distinct served dates are OMITTED from the result (the caller falls back to 1.0). The factor
    is clamped to [FACTOR_MIN, FACTOR_MAX].
    """
    out: Dict[int, float] = {}
    if panel is None or panel.empty:
        return out

    df = panel[panel["price_tier"].to_numpy() >= min_tier]
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


def _load_panel(session, horizons: Iterable[int]) -> pd.DataFrame:
    """Read the scored forecast_outcomes panel from Postgres (the ops parquet mirrors are
    stale/selected — see backend/AGENTS.md). Read-only, columns narrowed to the estimator's."""
    from sqlalchemy import bindparam, text

    hs = [int(h) for h in horizons]
    sql = text(
        "SELECT forecast_date, horizon_days, price_tier, "
        "predicted_price_low, predicted_price_mid, predicted_price_high, actual_price "
        "FROM forecast_outcomes "
        "WHERE horizon_days IN :horizons AND actual_price IS NOT NULL "
        "AND predicted_price_mid IS NOT NULL"
    ).bindparams(bindparam("horizons", expanding=True))
    rows = session.execute(sql, {"horizons": hs}).mappings().all()
    return pd.DataFrame(rows, columns=list(PANEL_COLUMNS))


def served_coverage_factors(session, horizons: Iterable[int], *,
                            min_dates: int = MIN_FORECAST_DATES,
                            alpha: float = ALPHA) -> Dict[int, float]:
    """Per-horizon served-coverage q_hat multipliers, or an empty map.

    Returns {} on any read failure or when no horizon clears the gate, so a missing/empty panel
    (the data-blocked default today) leaves the band byte-identical to a no-feedback artifact.
    """
    if session is None:
        return {}
    try:
        panel = _load_panel(session, horizons)
    except Exception as e:                       # a panel read must never fail a retrain
        logger.warning(f"  served-coverage panel read failed ({e}); no q_hat correction applied.")
        return {}
    return factors_from_panel(panel, horizons, min_dates=min_dates, alpha=alpha)
