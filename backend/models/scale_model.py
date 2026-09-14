"""The learned band scale — what `sigma` is a hand-picked stand-in for.

`models/conformal.py` divides the nonconformity score by a per-item scale, and
that scale has always been `sigma = price_std_60d / price`, optionally damped by
an exponent. Two measurements say that variable is the wrong one:

- `sigma` spans **11x** across its deciles while the `|residual|` it normalises
  spans **2.3-2.6x** (`2026-08-12-the-band-is-tilted-in-sigma.md`), so it
  over-reacts by roughly four, which is the entire tilt.
- The fitted exponent that repairs that (~0.33-0.42) is a damping constant, not
  an estimate of anything: it drifts 0.20-0.61 between folds, and applied at
  serving it *flipped* the tilt's sign at three of four horizons
  (`2026-08-12-served-sigma-profile.md`).

This module replaces the guess with an estimate: a small model predicting how
large the error usually is for an item like this one, fitted on the out-of-fold
residuals `q_hat` is already calibrated on. `sigma` is one of its inputs, so it
**nests** the current behaviour — the worst case is that it learns `s ~= sigma`
and nothing changes, which makes a null result informative rather than ambiguous.

**This is not the deleted p10/p90 design.** Those were 24 quantile boosters
predicting the band's endpoints, they cost 223.2s of a 381.2s budget, and they
covered 39-48% against 80% (`2026-08-04-minimal-model-results.md`). This fits one
small model per horizon on records that **already exist**, and it does not
produce the interval — split conformal still does, so the coverage guarantee is
unchanged. Conformalized quantile regression would restore that guarantee too,
but it needs p10/p90 *out of fold*, which triples the CV phase that is already
63.5% of training and puts a retrain at ~41-46 minutes against a 30-minute cap.

## Three properties worth knowing before changing anything here

**The target is `log |residual|`, and only its RELATIVE variation matters.**
`q_hat` is the (1-alpha) quantile of `|residual| / s`, so multiplying every `s`
by a constant divides `q_hat` by the same constant and leaves the band exactly
where it was. A constant offset in log space is therefore free, and the model
only has to get the *ordering and spread* right. That is a much easier fit than
predicting error magnitude outright, and it is why a deliberately small model is
the right choice rather than a concession.

**The objective is L1, so the model predicts the conditional MEDIAN of
`log |residual|`.** Squared loss on a heavy-tailed target chases the outliers,
and an item whose band is set by its worst historical day is exactly the
over-reaction being fixed.

**The scale must be cross-fitted or the calibration is a lie.** A scale fitted on
the same residuals `q_hat` is then calibrated on is in-sample: it matches those
residuals better than it will match a served item's, so `q_hat` comes out too
small and the band under-covers in production while looking perfect offline.
`cross_fit` is the honest path and `fit` is only for the final serving model.
This repo has been caught by precisely this before — see the note in
`2026-08-04-minimal-model-results.md` that coverage measured on the OOF pool
`q_hat` was fitted on is ">= 80% by construction".
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# A residual of exactly zero is a real outcome (an unchanged price), and
# log(0) is not. Floored rather than dropped: those rows are the calmest items,
# which is the half of the sigma range the band currently gets wrong, so
# dropping them would remove the evidence for the thing being fixed.
LOG_RESID_FLOOR_PCT = 1e-3

# Percentile clip on the SERVED scale, mirroring `conformal.sigma_bounds`. A
# model is free to predict a near-zero scale for some corner of feature space it
# saw twice; without this the band there collapses to a point and coverage for
# those items goes to zero. Wide enough not to bind on ordinary rows.
SCALE_FLOOR_PCTL = 1.0
SCALE_CAP_PCTL = 99.0

# Below this the fit is not worth trusting and the caller falls back to `sigma`.
# The conformal pool carries ~155-176K rows per horizon, so this cannot bind in
# production; it exists for the single-holdout path and for tests.
MIN_FIT_ROWS = 5_000

# Deliberately small. The target is a smooth, low-signal quantity and the model
# only needs relative variation (see the module docstring), so capacity here buys
# overfitting rather than accuracy -- and overfitting is not a quality problem
# but a CORRECTNESS one, because it re-creates the in-sample scale that
# `cross_fit` exists to avoid.
PARAMS = {
    "objective": "regression_l1",
    "num_leaves": 15,
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 150,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbosity": -1,
    "n_jobs": -1,
}


def enabled() -> bool:
    """Whether the learned scale is switched on.

    Default OFF. When off, `conformal` divides by `sigma ** beta` exactly as it
    did before this module existed. Set LEARNED_SCALE=1.
    """
    return os.environ.get("LEARNED_SCALE") == "1"


def _target(residual_pct) -> np.ndarray:
    """`log |residual|`, floored. See LOG_RESID_FLOOR_PCT."""
    r = np.abs(np.asarray(residual_pct, dtype=float))
    return np.log(np.maximum(r, LOG_RESID_FLOOR_PCT))


def fit(X: pd.DataFrame, residual_pct, params: dict | None = None) -> lgb.Booster | None:
    """One scale model. Returns None when the fit is not worth trusting.

    None is a supported outcome, not an error: the caller falls back to `sigma`,
    which is the pre-existing behaviour. Raising here would take out a retrain
    over a diagnostic.
    """
    y = _target(residual_pct)
    ok = np.isfinite(y) & np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    if int(ok.sum()) < MIN_FIT_ROWS:
        logger.warning(
            f"  learned scale: {int(ok.sum()):,} usable rows < {MIN_FIT_ROWS:,}, falling back to sigma for this horizon"
        )
        return None

    cfg = dict(PARAMS)
    if params:
        cfg.update(params)
    model = lgb.LGBMRegressor(**cfg)
    model.fit(X[ok], y[ok])
    return model.booster_


def predict_scale(
    booster: lgb.Booster | None, X: pd.DataFrame, clip: tuple[float, float] | None = None, fallback=None
) -> np.ndarray:
    """The scale itself: `exp(model)`, clipped.

    `fallback` (normally `sigma`) fills any row the model cannot score. A NaN
    scale reaching `conformal.band` produces a NaN half-width, which surfaces in
    the API as a missing interval rather than a loud failure -- so every
    non-finite value is replaced here, at the source.
    """
    n = len(X)
    fb = np.full(n, np.nan) if fallback is None else np.asarray(fallback, dtype=float)
    if booster is None:
        return fb

    with np.errstate(over="ignore"):
        s = np.exp(booster.predict(X))
    s = np.asarray(s, dtype=float)

    bad = ~np.isfinite(s) | (s <= 0)
    if bad.any():
        s = np.where(bad, fb, s)

    if clip is not None:
        s = np.clip(s, float(clip[0]), float(clip[1]))
    return s


def clip_bounds(scale_values) -> tuple[float, float]:
    """Percentile bounds to persist beside the model.

    Derived from the CALIBRATION scale distribution, so the bounds a served row
    is clipped into are the ones `q_hat` was fitted inside. Deriving them at
    serve time from the served batch instead would move the clip with the daily
    cohort and break that correspondence.
    """
    s = np.asarray(scale_values, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        raise ValueError("no finite scale values: cannot derive clip bounds")
    return (float(np.percentile(s, SCALE_FLOOR_PCTL)), float(np.percentile(s, SCALE_CAP_PCTL)))


def cross_fit(
    X: pd.DataFrame, residual_pct, folds: Sequence, params: dict | None = None, fallback=None
) -> tuple[np.ndarray, int]:
    """Out-of-sample scale for every calibration row, by leave-one-fold-out.

    Returns `(scale, n_models)`. Each row is scored by a model that never saw
    that row's fold, so `q_hat` calibrated on `|residual| / scale` is honest --
    which is the whole reason this function exists instead of a single `fit`.

    A fold whose complement is too small to fit falls back to `fallback` for its
    rows rather than borrowing another fold's model, because a model that saw
    the row is worse than no model here.

    ⚠️ **Cost.** One small model per fold per horizon, ~8-9 folds. Deliberately
    the cheap half of the design: the expensive alternative (out-of-fold p10/p90
    boosters, i.e. conformalized quantile regression) triples the CV phase that
    is already 63.5% of training. Time it before assuming it is free.
    """
    f = np.asarray(folds)
    n = len(X)
    fb = np.full(n, np.nan) if fallback is None else np.asarray(fallback, dtype=float)
    out = np.array(fb, dtype=float, copy=True)

    n_models = 0
    for fold in np.unique(f):
        held = f == fold
        booster = fit(X[~held], np.asarray(residual_pct)[~held], params)
        if booster is None:
            continue
        n_models += 1
        # No clip on the calibration pass: the bounds are derived FROM this
        # distribution afterwards, so clipping first would be circular.
        out[held] = predict_scale(booster, X[held], clip=None, fallback=fb[held])
    return out, n_models
