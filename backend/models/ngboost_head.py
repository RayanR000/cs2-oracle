"""
NGBoost distributional head: per-prediction mean + std via Normal distribution.

Shadow-only — never drives serving or the band. Trained on h=7 and h=14 only
(where the IC ceiling headroom is largest per label-noise-ceiling measurement).

Pure module: takes training data in, returns predictions out. No DB, no I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Only these horizons are supported — h=7 and h=14 have the largest IC
# ceiling headroom and keep training within the 30-minute budget.
SUPPORTED_HORIZONS: frozenset[int] = frozenset({7, 14})


@dataclass
class NGBoostPrediction:
    """Container for NGBoost predictions: mean and std per row."""

    mean: np.ndarray
    std: np.ndarray
    horizon: int

    def __post_init__(self) -> None:
        assert len(self.mean) == len(self.std), "mean and std must have the same length"
        assert self.horizon in SUPPORTED_HORIZONS, f"horizon {self.horizon} not in {SUPPORTED_HORIZONS}"


@dataclass
class NGBoostTrainResult:
    """Result of training an NGBoost model for one horizon."""

    model: Any  # ngboost.NGBRegressor
    horizon: int
    train_nll: float
    val_nll: float
    n_train: int
    n_val: int


def train_ngboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: np.ndarray,
    horizon: int,
    *,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    minibatch_frac: float = 0.8,
    random_state: int = 42,
) -> NGBoostTrainResult:
    """Train an NGBoost model with Normal distribution for one horizon.

    Parameters
    ----------
    X_train, y_train : training features and target (return_hd).
    X_val, y_val : validation features and target.
    horizon : forecast horizon in days (must be 7 or 14).
    n_estimators : number of boosting stages.
    learning_rate : step size shrinkage.
    minibatch_frac : fraction of training data per boosting iteration.
    random_state : random seed for reproducibility.

    Returns
    -------
    NGBoostTrainResult with the fitted model and diagnostics.
    """
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"NGBoost head only supports horizons {sorted(SUPPORTED_HORIZONS)}, got {horizon}")

    from ngboost import NGBRegressor
    from ngboost.distns import Normal

    # Clean inputs: drop rows with NaN targets
    train_mask = np.isfinite(y_train)
    val_mask = np.isfinite(y_val)

    X_tr = X_train[train_mask]
    y_tr = y_train[train_mask]
    X_v = X_val[val_mask]
    y_v = y_val[val_mask]

    logger.info(f"  NGBoost {horizon}d: {len(y_tr)} train / {len(y_v)} val rows")

    model = NGBRegressor(
        Dist=Normal,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        minibatch_frac=minibatch_frac,
        random_state=random_state,
        verbose=False,
    )

    model.fit(
        X_tr,
        y_tr,
        X_val=X_v,
        Y_val=y_v,
        early_stopping_rounds=20,
    )

    # Compute NLL on train and val
    train_nll = -model.score(X_tr, y_tr)
    val_nll = -model.score(X_v, y_v)

    logger.info(f"  NGBoost {horizon}d: train NLL={train_nll:.4f}, val NLL={val_nll:.4f}")

    return NGBoostTrainResult(
        model=model,
        horizon=horizon,
        train_nll=train_nll,
        val_nll=val_nll,
        n_train=len(y_tr),
        n_val=len(y_v),
    )


def predict_ngboost(
    model: Any,  # ngboost.NGBRegressor
    X: pd.DataFrame | np.ndarray,
    horizon: int,
) -> NGBoostPrediction:
    """Generate mean + std predictions from a trained NGBoost model.

    Parameters
    ----------
    model : trained NGBRegressor.
    X : feature matrix for prediction.
    horizon : the horizon this model was trained for.

    Returns
    -------
    NGBoostPrediction with mean and std arrays.
    """
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"NGBoost head only supports horizons {sorted(SUPPORTED_HORIZONS)}, got {horizon}")

    dist = model.pred_dist(X)
    mean = dist.loc  # location parameter = mean for Normal
    std = dist.scale  # scale parameter = std for Normal

    return NGBoostPrediction(
        mean=np.asarray(mean),
        std=np.asarray(std),
        horizon=horizon,
    )


def save_ngboost_model(model: Any, path: str) -> None:
    """Persist an NGBoost model to disk via joblib."""
    import joblib

    joblib.dump(model, path)
    logger.info(f"  Saved NGBoost model to {path}")


def load_ngboost_model(path: str) -> Any:
    """Load an NGBoost model from disk via joblib."""
    import joblib

    model = joblib.load(path)
    logger.info(f"  Loaded NGBoost model from {path}")
    return model
