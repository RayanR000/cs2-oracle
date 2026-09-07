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

from backtest.scoring import HEADLINE_MIN_TIER, MIN_FORECAST_DATES, price_tier
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
SIGNED_BAND_SERVING_START: Optional[str] = "2026-08-19"  # first clean signed-band prod serve
# (deployed via full-retrain run 32293991440; 2026-08-18 excluded — a symmetric predict-only serve
# preceded the retrain overwrite on that date, so its geometry is ambiguous).

# The climatology band scale (CLIMATOLOGY_SCALE, on by default since 2026-08-19) is a SECOND band-
# geometry change on the same panel: it replaces the sigma denominator with a per-item climatology
# scale, so the served HALF-WIDTHS `r` reads differ from a sigma-scaled row's. A factor pooled
# across this cutover blends two miscoverage regimes (sigma over-covers, climatology is near-
# calibrated) and applies the blend to a climatology artifact, over-shrinking it. Set this to the
# date the climatology band served its FIRST prod forecast (the first full-retrain deploy carrying
# `climatology_scale: true` in meta.json). Left None until then: prod is still sigma-scaled, so the
# signed-band floor alone is correct and the feedback need not restart its date count early.
CLIMATOLOGY_SERVING_START: Optional[str] = "2026-08-20"  # first clean climatology prod serve
# (climatology artifact cached by full-retrain run 32323684984; that run's predict re-wrote 2026-08-19
# on top of an earlier sigma predict-only serve, so 08-19's geometry is mixed and excluded — the first
# aggregator-chained predict-only serve on the climatology cache is 2026-08-20).

# Raising CLIMATOLOGY_SHRINK_K from 20 to 320 (2026-08-26) is a THIRD band-geometry change on the
# same panel. It does not swap the denominator the way the climatology cutover did, but it pools a
# well-observed item's dispersion far harder toward its tier -- 6.5-15.4% narrower at matched 80%
# coverage across h=3..30 -- so the served half-widths, and therefore the `r` this estimator reads,
# are a different band shape. A factor pooled across the cutover would fit the K=20 miscoverage and
# apply it to a K=320 artifact, over-shrinking a band that is already narrower. Set this to the date
# the K=320 band serves its FIRST prod forecast (the first deploy whose artifact was BUILT at
# K=320 -- the tables are persisted in meta.json, so a predict-only run on a K=20 cache still serves
# the old geometry). That deploy is now IDENTIFIABLE rather than remembered: `meta.json` records
# `climatology_shrink_k`, and `ItemForecaster.climatology_geometry_matches_code()` is False whenever
# the loaded cache predates a K change. Take the date from the first run whose artifact reports
# K=320, not from whichever run happened to be green.
#
# SET 2026-09-07 to the anchor date of Price Forecast run 34080103996, the first prod run since the
# billing pause. It is `mode=full`, so it BUILT the climatology tables at the code's K=320 rather
# than loading a K=20 cache, and it served forecast_date 2026-09-06. Nothing earlier qualifies:
# K=320 landed on main in f02320b at 2026-08-27 19:47 -0400, and no Price Forecast run executed
# between 2026-08-22 and 2026-09-03 (the pause), so the 2026-08-25..27 rows in `item_forecasts`
# predate the merge and carry K=20.
#
# This floor SUPERSEDES the 2026-08-20 climatology floor, which drops the post-floor panel to a
# single date. That is the correct cost: the 2026-08-20..27 served rows carry the K=20 band shape,
# and a factor fitted on them would over-shrink an already-narrower K=320 band.
SHRINK_K_SERVING_START: Optional[str] = "2026-09-06"
# docs/changelog/2026-08-26-climatology-shrink-k-re-swept.md

# A sentinel distinguishing "caller did not pass since" from an explicit since=None (dormant).
_UNSET = object()


def _geometry_floor() -> Optional[str]:
    """The forecast_date floor that isolates the CURRENT served band geometry.

    Each configured start is a date on which a band-geometry change first served prod. The panel
    must be floored to the LATEST of them: the multiplier is applied to an artifact carrying EVERY
    change, and `r` reads the stored band shape, so any row served under an earlier geometry
    calibrates a shape the current band no longer has. Returns None only when no cutover is set
    (SIGNED_BAND_SERVING_START unset), which drives the dormancy in `served_coverage_factors`.
    Read at call time so the constants can be monkeypatched in tests."""
    starts = [s for s in (SIGNED_BAND_SERVING_START, CLIMATOLOGY_SERVING_START,
                          SHRINK_K_SERVING_START)
              if s is not None]
    return max(starts) if starts else None  # ISO dates order lexically

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
    # `forecast_outcomes` has no `price_tier` column -- that lives on `prediction_accuracy`
    # (added there by migration 0019). Selecting it here raised UndefinedColumn on every prod
    # run, and `served_coverage_factors` swallows the read failure, so the feedback was silently
    # dormant for reasons that had nothing to do with panel depth. The tier is a pure function of
    # the serve-time price, so derive it with the canonical banding rather than storing it: that
    # keeps the >= $1 cohort here identical to the one the headline scores, and works on the rows
    # already in the table instead of only on ones written after a migration.
    sql = text(
        "SELECT forecast_date, horizon_days, "
        "COALESCE(base_price, current_price) AS tier_price, "
        "predicted_price_low, predicted_price_mid, predicted_price_high, actual_price "
        f"FROM forecast_outcomes {where}"
    ).bindparams(bindparam("horizons", expanding=True))
    rows = session.execute(sql, params).mappings().all()
    cols = ["tier_price" if c == "price_tier" else c for c in PANEL_COLUMNS]
    df = pd.DataFrame(rows, columns=cols)
    # A NULL serve-time price tiers to 0, which is below HEADLINE_MIN_TIER and so drops the row --
    # the same treatment a sub-$1 item gets, which is the conservative side for a band factor.
    px = pd.to_numeric(df["tier_price"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    df["price_tier"] = [price_tier(float(v)) for v in px]
    return df[list(PANEL_COLUMNS)]


def served_coverage_factors(session, horizons: Iterable[int], *,
                            min_dates: int = MIN_FORECAST_DATES,
                            alpha: float = ALPHA,
                            since: Optional[str] = _UNSET) -> Dict[int, float]:
    """Per-horizon served-coverage q_hat multipliers, or an empty map.

    Returns {} on any read failure or when no horizon clears the gate, so a missing/empty panel
    (the data-blocked default today) leaves the band byte-identical to a no-feedback artifact.

    `since` defaults to `_geometry_floor()` — the latest of the configured band-geometry cutovers
    (signed band, climatology scale) — so the factor is fit only on rows served under the band
    shape it will be applied to. Returns {} outright when that floor is None (no cutover set): the
    whole panel is then pre-signed-band geometry, and a factor fit on it would calibrate the wrong
    band shape and apply it to the signed offsets. The feedback stays dormant until a cutover is set.
    """
    if since is _UNSET:
        since = _geometry_floor()
    if session is None:
        return {}
    if since is None:
        logger.info(
            "  served-coverage: no band-geometry cutover set (SIGNED_BAND_SERVING_START and "
            "CLIMATOLOGY_SERVING_START both unset); feedback dormant — the panel is pre-signed-band "
            "geometry, so no q_hat correction is applied.")
        return {}
    try:
        panel = _load_panel(session, horizons, since=since)
    except Exception as e:                       # a panel read must never fail a retrain
        logger.warning(f"  served-coverage panel read failed ({e}); no q_hat correction applied.")
        return {}
    return factors_from_panel(panel, horizons, min_dates=min_dates, alpha=alpha, since=since)
