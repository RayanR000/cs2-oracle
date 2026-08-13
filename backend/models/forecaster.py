"""
LightGBM-based price forecaster for CS2 items.
Trains quantile regression models for 3d, 7d, 14d and 30d horizons,
using price history, technical indicators, events, and item metadata.
"""

import os
import gc
import sys
import json
import time
import hashlib
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from sqlalchemy import text
from models import conformal
from models import scale_model
from models.item_parser import (
    BID_SOURCES,
    PHASE_COLLAPSED_EXEMPT_PATTERNS,
    PHASE_COLLAPSED_SLUG_PATTERNS,
    TRAILING_WINDOW_SOURCES,
    archive_universe_sql_filter,
    bid_sources_sql_filter,
    is_phantom_slug,
    is_phase_collapsed,
    parse_item_name,
    phantom_slug_sql_filter,
    phase_collapsed_sql_filter,
)
from models.staleness import STALE_RUN_GAP_BREAK_DAYS, stale_run_days
from backtest.scoring import HEADLINE_MIN_TIER, MIN_FORECAST_DATES
from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS, SMOOTH_WINDOW
# Invariant #4 (backend/AGENTS.md): a directional accuracy is quotable only
# beside the constant call and the realised down rate, with Pesaran-Timmermann
# as the headline. These lived in backtest/ and were applied only to production
# scoring; the offline CV quoted DA against persistence and momentum alone.
from backtest.directional_test import (
    constant_call_baseline,
    pesaran_timmermann,
    realised_down_rate,
)

logger = logging.getLogger(__name__)

RNG = np.random.RandomState(42)
DIRECTION_FLAT_TOLERANCE_PCT = 0.5

# Vol-scaled directional labels (2026-07-27). The flat band is
# clamp(k_h * sigma_daily * sqrt(h), floor, cap) in percent, replacing the
# fixed ±DIRECTION_FLAT_TOLERANCE_PCT. sigma_daily is trailing std of
# log_return_1d over DIRECTION_VOL_WINDOW rows. Used for labeling only.
DIRECTION_VOL_WINDOW = 30
DIRECTION_THRESHOLD_FLOOR_PCT = 0.2
DIRECTION_THRESHOLD_CAP_PCT = 15.0
DIRECTION_LABEL_VOL_COL = "label_vol_30d"

# The cohort on which neither label basis is contaminated (2026-08-11).
# `prepare_targets` divides by the raw quote at the anchor and `predict` divides
# by its local median, so both carry the wedge `p[d]/S[d]` -- with opposite
# signs, which is why moving the label to the served denominator was refuted
# rather than fixing it. Where the two are equal the wedge is identically 1 and
# neither can operate, and that is the subset an arm has to be ranked on.
# Same name as `replay_serving`'s column, deliberately: one population, two
# readers. See `docs/changelog/2026-08-11-the-gap-is-the-anchor-denominator.md`.
ANCHOR_TIED_COL = "anchor_is_tied"


def calibration_target_col(horizon: int) -> str:
    """The per-horizon return column `q_hat` is calibrated on.

    Distinct from `target_return_{h}d`, which is what the boosters are TRAINED
    on. The two differ in their denominator: the training label divides by the
    raw anchor quote `p[d]`, this one by the served smoothed anchor `S[d]`,
    which is the basis `predict` quotes from and the backtest scores in. A
    function rather than an f-string at each site because a conformal set built
    from the wrong column is silent -- it produces a plausible q_hat that is
    simply the wrong width.
    """
    return f"target_return_{horizon}d_cal"


# Cached result of GPU availability check (avoids repeated subprocess probes)
_GPU_AVAILABLE_CACHE: Optional[bool] = None

# `BID_SOURCES`, `PHASE_COLLAPSED_SLUG_PATTERNS` and the SQL filters over them
# are imported from `models/item_parser.py` above and re-exported here. The
# rules are about names and sources, not about models, and `api/` needs them
# without importing LightGBM — but every archive reader already takes its
# universe from this module, so the re-export is what keeps that true.


def embargo_days(horizon: int) -> int:
    """How many days a train/validation split must embargo at *horizon*.

    A row dated ``d`` is labelled with the price at ``d + horizon``, so a purge
    of exactly ``horizon`` looks sufficient. It is not, because that label is
    not a point observation. Both legs resolve through
    `backtest/price_resolution.py::resolve_anchors`, which takes the median of
    the last `SMOOTH_WINDOW` (3) observations, admits an observation up to
    `MAX_WINDOW_SPAN_DAYS` (7) before the anchor, and — on the feature side —
    `LAG_TOLERANCE_DAYS` (3) reaches a lag lookup back past its exact date when
    the archive dropped that calendar day. The label's support therefore runs
    ``horizon + 13`` days past ``d``, and an ``horizon``-day purge leaves the
    last 13 days of it drawn from inside the validation window.

    Reads `ItemForecaster.LAG_TOLERANCE_DAYS` at call time, so the constant has
    one definition even though the class is declared below this function.

    At h=30 the embargo (43d) exceeds `VALIDATION_WINDOW_DAYS`. That is the
    correct cost of the overlap, not a bug: it means a 30d fold cannot be built
    from 30 days of history, which was always true and was previously hidden.
    """
    carry = (ItemForecaster.LAG_TOLERANCE_DAYS
             + SMOOTH_WINDOW + MAX_WINDOW_SPAN_DAYS)
    return int(horizon) + carry


# Price tier boundaries for per-tier bias correction
PRICE_TIER_BOUNDARIES = [(0, 1, "<$1"), (1, 5, "$1-5"), (5, 20, "$5-20"),
                         (20, 100, "$20-100"), (100, float("inf"), ">$100")]
BIAS_EWMA_ALPHA = 0.3

# Direction upweight: multiplier for positive-return samples during training.
# Counteracts the model's conservative bias (underpredicts "up" by ~2×).
# Applied in _compute_sample_weights before normalization.
DIRECTION_UPWEIGHT = 1.5

# Recency half-life, in days, for time-decayed sample weights: a row `h` days
# older than the newest row in the frame carries 0.5× the gradient weight.
# Set to 0 to disable decay entirely.
#
# Why (diagnosed 2026-07-29, see docs/changelog/2026-07-29-7d-q50-early-stop.md):
# training spans 1460 days but only ~14.5% of rows fall in the last 180 days,
# while the early-stopping validation window (most recent 30 days) has ~2× the
# return spread of the training set overall (std 35.1 vs 21.9). The model was
# therefore fit mostly on a calmer historical regime and validated against the
# current volatile one, leaving the 7d q50 validation curve nearly flat (0.28%
# total improvement) and its early-stopping round noise-determined.
#
# DISABLED (0.0) — A/B'd 2026-07-29 and it did not clear the gate.
# `scripts/ab_test_recency_weights.py`, 8 purge-gap folds on the real 141-item
# production frame, half-life 365d vs flat. q50 pinball: 3d −0.01% (5/8 folds),
# 7d −0.63% (2/8), 14d −0.22% (5/8), 30d +1.23% (6/8, DA +1.17pp). Only 30d
# passed, and 14d — long-horizon too, and DART-configured at the time, as 30d
# was — moved the other way, so the 30d win reads as noise, not mechanism.
#
# It also failed at the thing it was built for: the 7d q50 stopping-round sd
# rose from 81 to 102. Recency weighting does NOT stabilise early stopping.
#
# Kept because the knob is cheap (a weight multiplier, zero training cost) and
# the finding is worth preserving. Set to e.g. 365.0 to re-enable; a 30d-only
# ship would need confirmation with the full 3-member ensemble first.
#
# 365d was deliberately mild: a 4-year-old row still carries 0.0625. The
# roadmap's α^days_ago with α=0.99 would leave a 1460-day-old row at ~6e-7,
# effectively truncating training to ~200 days.
SAMPLE_WEIGHT_HALFLIFE_DAYS = 0.0

# The longest bit-identical price run a label may sit on. A row whose anchor
# day OR target day carries a `stale_run_days` above this has its label voided
# in `prepare_targets`, exactly as a snapshot day or a collector cutover does.
#
# 0 means "the day must be a fresh price level on both legs"; None disables the
# rule and reproduces the pre-2026-08-08 label set. Named rather than written as
# a literal so `scripts/ab_test_frozen_runs.py` can contrast None/0/1/2 without
# editing this file. The review states the rule as "run >= 2", which is
# 1-indexed and selects the same rows.
#
# Measured 2026-08-08, 2024+, on the >=$1 cohort: this voids 19.2/18.2/16.0/13.8%
# of labels at h=3/7/14/30 (39.8/39.5/38.8/38.8% on the whole unfiltered
# universe). Two things that number hides, both in
# `docs/superpowers/specs/2026-08-08-frozen-price-runs-design.md`:
#
#   1. It is a 2026 filter. The >=$1 stale rate runs 0.5-0.8% through 2025 and
#      6-33% across 2026, because no Steam-derived series in the archive is a
#      point observation. So this removes ~30% of 2026 >=$1 labels and ~0.6% of
#      everything before 2024 — defensible, since 2026 is what production
#      trains and serves on, but it is not an even 13-year cleaning.
#   2. It is not surgical. 13.7-15.9% of the >=$1 labels it takes carry a
#      NON-zero return. The rule can therefore be net-harmful, which is why it
#      is verified on paired interval WIDTH and not on a point estimate.
LABEL_MAX_STALE_RUN_DAYS = 0


def _gpu_available() -> bool:
    global _GPU_AVAILABLE_CACHE
    if _GPU_AVAILABLE_CACHE is not None:
        return _GPU_AVAILABLE_CACHE
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            _GPU_AVAILABLE_CACHE = False
            return False
        # Verify LightGBM was compiled with CUDA by probing in a subprocess.
        # Directly calling lgb.train with device="cuda" can segfault if the
        # pip wheel is CPU-only, and that crash can't be caught in-process.
        _probe_code = (
            "import numpy as np; import lightgbm as lgb;"
            "ds=lgb.Dataset(np.array([[0.0]]), label=np.array([0.0]));"
            "lgb.train({'device':'cuda','num_threads':1,'verbosity':-1},ds,num_boost_round=1)"
        )
        probe = subprocess.run(
            [sys.executable, "-c", _probe_code],
            capture_output=True, text=True, timeout=30,
        )
        _GPU_AVAILABLE_CACHE = probe.returncode == 0
        return _GPU_AVAILABLE_CACHE
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        _GPU_AVAILABLE_CACHE = False
        return False


# The ByMykel bundle, routed by exact name before any prefix rule can claim it:
# `item_age_meta_days` would match the "item_age" temporal prefix, `is_meta_*` and
# `rarity_meta_rank` the item_identity prefixes, and `type_meta_*` the item_metadata
# one. Declared here rather than on the class because _feature_group is a module
# function; ItemForecaster.BYMYKEL_META_FEATURES is the same frozenset.
_BYMYKEL_META_FEATURES = frozenset({
    "item_age_meta_days", "item_age_ambiguous", "rarity_meta_rank",
    "is_meta_stattrak", "is_meta_souvenir", "float_meta_min", "float_meta_max",
    "type_meta_crate_id", "type_meta_collection_id",
})


def _feature_group(name: str) -> str:
    if name in _BYMYKEL_META_FEATURES:
        return "bymykel_metadata"
    if any(name.startswith(p) for p in ("price_", "return_", "log_return_", "bb_",
                                         "rsi_", "macd_", "vol_", "trend_",
                                         "price_accel_", "autocorr_", "support_",
                                         "volume_", "vol_price_")):
        return "price_technicals"
    if name.startswith("supply_"):
        return "supply_depth"
    # Before the item_identity / temporal prefixes below, none of which can
    # claim a `tier_lead_` name today -- but the group is small and explicit,
    # so keep it adjacent to supply_depth rather than relying on that.
    if name.startswith("tier_lead_"):
        return "tier_lead"
    if any(name.startswith(p) for p in ("is_", "quality_rank", "rarity_")):
        return "item_identity"
    if name.startswith("type_"):
        return "item_metadata"
    if any(name.startswith(p) for p in ("day_", "month_", "quarter_", "week_",
                                         "item_age", "weekend")):
        return "temporal"
    if name.startswith("event_"):
        return "events"
    if any(name.startswith(p) for p in ("market_", "market_regime_")):
        return "cross_sectional"
    if name.startswith("social_"):
        return "social"
    return "other"


class IncompatibleModelArtifact(RuntimeError):
    """A saved model cache was written by an incompatible code version.

    Raised rather than defaulted around. Between the CQR scheme and the
    minimal model, `conformal_calibration` changed MEANING (percentage-point
    addend -> dimensionless sigma multiplier) without changing type, so a
    tolerant loader would silently serve a band computed two different ways.
    """


class ItemForecaster:
    # Re-exposed as class attributes -- not just module names -- because
    # `_apply_multi_source_voting` is the method that applies them and is
    # itself a staticmethod called unbound (`ItemForecaster._apply_...`,
    # `backtest/price_resolution.py:246`), so the class that applies a source
    # rule is the natural place for a reader or a test to find it, without
    # also importing `models.item_parser` to look up what it means.
    BID_SOURCES = BID_SOURCES
    TRAILING_WINDOW_SOURCES = TRAILING_WINDOW_SOURCES

    HORIZONS = [3, 7, 14, 30]
    # The band no longer comes from quantile models — see models/conformal.py.
    # Measured (2026-08-04 warm baseline): 24 p10/p90 GBMs cost 223.2s of a
    # 381.2s budget for 39-48% empirical coverage against an 80% target, while
    # their top feature was already price_std_60d. The 303s/462s figures this
    # comment first carried were the spec's pre-measurement estimate.
    QUANTILES = [0.5]
    # Bump when the MEANING of any persisted field changes, not just the set
    # of fields. v2: conformal_calibration became a dimensionless multiplier
    # of per-item sigma, p10/p90 models no longer exist, sigma_clip added.
    # v3 (2026-08-06): the eleven volume features left feature_cols. This IS a
    # set change, which the rule above would not normally bump for — but a v2
    # artifact carries its own 47-column feature_cols and the columns are still
    # engineered, so predict() would keep serving the dead features until the
    # 14-day age trigger fired. Bumping forces the retrain that realises the fix.
    # v4 (2026-08-06): lag lookups became as-of within LAG_TOLERANCE_DAYS instead
    # of exact-date. The feature *set* is unchanged, but the VALUES and the
    # persisted feature_medians now mean something different, so a v3 booster
    # scored against v4 features would itself be a train/serve mismatch — the
    # exact defect the change removes. The bump forces the retrain.
    # v5 (2026-08-06): the dollar-denominated columns left the feature set for
    # their scale-free forms (_DOLLAR_SCALE_FEATURES, price_cv_{w}d,
    # macd_*_rel). A v4 booster's splits are thresholds in dollars and are
    # meaningless against v5 features, so this must not load across the bump.
    # v6 (2026-08-09): Optuna selects on within-date rank IC instead of
    # early-stopped pinball loss. A cached meta.json would otherwise supply
    # hyperparameters chosen under the criterion this replaced.
    MODEL_ARTIFACT_VERSION = 6
    MIN_HISTORY_DAYS = 30
    # Prediction eligibility is looser than training: the live aggregator
    # series is still young, and 14 daily points is enough for the lag/rolling
    # features to be non-degenerate.
    PREDICT_MIN_HISTORY_DAYS = 14
    # How far a lag lookup may reach back past its exact target date when that
    # date is absent from the archive. The aggregator drops whole calendar days
    # (August 2026 held only 08-01 and 08-04), and a strict exact-date lookup
    # then NaNs every lag-day feature for every item simultaneously -> median
    # fill on 100% of served rows. 3 days covers the observed gap widths while
    # staying far short of the multi-week holes that must remain NaN, because a
    # value reached across those would fabricate a jump.
    LAG_TOLERANCE_DAYS = 3
    # How far the latest quote may sit from its own local median before serving
    # calls it an outlier. It has always been the threshold of predict()'s
    # `using smoothed price` warning; it only became a *decision* threshold with
    # SERVE_OUTLIER_GATED_ANCHOR, and `_serving_base_price` reads it under both
    # arms so the deviating cohort is the same population either way.
    ANCHOR_OUTLIER_TOLERANCE = 0.10
    # Directional-accuracy floor for the drift *alert*. This no longer gates a
    # retrain: the model's measured production DA is 46.7-50.8%
    # (docs/architecture/model-optimization.md), so a 60% floor was never
    # attainable and fired on every run. Kept as an alert threshold only.
    DRIFT_DA_THRESHOLD = 60.0
    # Walk-forward validation split: most recent N days are held out.
    # A relative split stays valid as data accumulates (a fixed date would
    # eventually leave the validation set covering all new data).
    # Also the per-fold width for expanding-window CV: directional accuracy is
    # dominated by market-wide regime moves, so the effective sample size is
    # the number of independent validation DATES, not rows. A wider window
    # (was 21) buys more market episodes per fold and more OOF points for
    # conformal calibration.
    VALIDATION_WINDOW_DAYS = 30
    # Floors a validation window must clear to be worth stopping and tuning on.
    # Below either one, early stopping and the Optuna objective are reading
    # noise, and the feature-group permutation test false-positives hard enough
    # to prune 14d/30d down to ~4 features.
    MIN_VAL_ROWS = 2000
    MIN_VAL_DATES = 7
    # How far back the trailing window may be widened to reach those floors.
    # Voided labels (prepare_targets) thin the default window, and the old
    # response was a positional 80/20 split whose validation window is ~10
    # months. Widening keeps validation a recent contiguous window; this bound
    # keeps "recent" meaningful and keeps the window off the CV folds' turf.
    MAX_VALIDATION_WINDOW_DAYS = 90
    REGIMES = ["bear", "range", "bull"]
    REGIME_RETURN_THRESHOLD_BEAR = -3.0   # market_return_30d < -3% → bear
    REGIME_RETURN_THRESHOLD_BULL = 3.0    # market_return_30d > 3% → bull
    # Single member. The 3-seed / 3-feature-fraction ensemble was estimated at
    # 0.3-0.5pp in docs/architecture/model-optimization.md, which is below the
    # MDE the gate reports, so it cannot be resolved in isolation. If the
    # minimal model misses its bar, restoring N_ENSEMBLES = 2 is the first
    # thing to try.
    N_ENSEMBLES = 1
    ENSEMBLE_SEEDS = [42]
    ENSEMBLE_FEATURE_FRACTIONS = [0.7]
    MAX_BIN = 63
    # GBDT is the only boosting type. DART was the single most expensive config
    # choice in this file and had never been measured against GBDT on a
    # trustworthy gate; when it finally was, under the pre-registered bar, 14d
    # improved +3.14pp (CI [+1.955, +4.373]) and 30d was unchanged — and 14d was
    # the horizon DART was supposedly earning its cost on. So there is no
    # per-horizon BOOSTING_TYPE_MAP and no DART_NUM_BOOST_ROUND any more; the
    # dropout branches they selected are gone with them.
    # See docs/changelog/2026-08-04-minimal-model-results.md.
    BOOSTING_TYPE = "gbdt"
    N_TRIALS_MAP = {3: 50, 7: 10, 14: 15, 30: 15}
    # 3d is frozen (50-trial search, winner warm-started in _optuna_search_params).
    # 14d/30d still search because they are the noisiest horizons; the original
    # reason (tuning DART's drop_rate/max_drop/skip_drop) went away with DART.
    SKIP_HP_HORIZONS = [3]
    # Horizon-specific feature exclusions based on ablation study
    # (2026-07-19-feature-contribution-by-horizon.md):
    # - Cross-sectional features actively harm 14d (−0.9pp) and 30d (−3.5pp)
    # - Event features harm 30d (−3.0pp) but help 14d (+2.0pp)
    HORIZON_EXCLUDED_GROUPS = {
        14: ["cross_sectional"],
        30: ["cross_sectional", "events"],
    }
    # Global feature-group allowlist. Set to a list of _feature_group() names to
    # restrict the model to those groups; None uses every group.
    # Ablation (2026-07-24, 7-fold purge-gap CV): the 85 non-price features add
    # no measurable directional accuracy over price/technical features alone
    # (full−price = −0.6/+0.1/+1.6/−1.3pp across 3/7/14/30d, all within fold
    # noise) and hurt at 3d/30d. Restrict to price technicals; the momentum
    # (return_Nd) features live in this group, so trend signal is retained.
    FEATURE_GROUP_ALLOWLIST = ["price_technicals"]
    # Run the allowlist BEFORE _prune_features. The correlation matrix is
    # O(rows x p^2) single-threaded pandas: 25.2s over 123 candidate columns,
    # 1.65s over the 33 the allowlist keeps (measured 2026-08-09). Output was
    # identical on the production frame -- but _prune_features keeps the
    # lower-indexed member of a >0.95 pair and index order does not follow
    # group, so set this False to restore the old order if a feature count moves.
    ALLOWLIST_BEFORE_PRUNE = True
    # Every name _feature_group() can return, minus bymykel_metadata, which is
    # gated by bymykel_metadata_enabled() rather than by the allowlist alone.
    ALL_FEATURE_GROUPS = frozenset({
        "price_technicals", "supply_depth", "item_identity", "item_metadata",
        "temporal", "events", "cross_sectional", "social", "other",
        "tier_lead",
    })
    # The tier lead-lag group, gated by tier_lead_enabled() the way
    # bymykel_metadata is -- the allowlist alone cannot admit it, because a group
    # that is allowlisted but never engineered yields columns absent and
    # median-filled to zero (the hazard _skipped_feature_groups documents).
    TIER_LEAD_GROUP = "tier_lead"
    TIER_LEAD_FEATURES = ("tier_lead_return_1d",)
    # The naive predictor N1 boosts from, negated: the one runnable baseline the
    # model measurably loses to on rank IC. See naive_init_score_enabled().
    NAIVE_OFFSET_COL = "return_1d"
    # Columns the within-date rank transform must not touch, because something
    # other than a booster reads their VALUES.
    #
    # `price_tier` is the whole set and it cost a void A/B arm to find. It is a
    # feature column, so a transform over feature_cols catches it -- but it is
    # also the cohort gate: _cv_evaluate_horizon does
    # `val_df["price_tier"] >= HEADLINE_MIN_TIER` (:6514, :6537) to select the
    # >=$1 rows, and _fit_direction_classifier takes it as `tier_train` (:4490,
    # :6490). Ranked into [-1, 1] the mid-rank maximum is 1 - 1/n, so the mask
    # matches NOTHING: run 31424689196 returned mean_rank_ic,
    # mean_naive_rank_ic and mean_classifier_acc_ge1 all None while
    # mean_dir_acc and edge_vs_constant_call -- which are pooled over every tier
    # -- populated and looked 3-5pp BETTER at all four horizons. A green run
    # publishing the penny metric as if it were the served one.
    #
    # It is also the wrong transform for the column on its own terms: price_tier
    # is a bounded categorical, deliberately kept as one (see the pruning note
    # at :477).
    RANK_TRANSFORM_EXCLUDED = frozenset({"price_tier"})
    # The ByMykel item-metadata bundle, joined from
    # price-archive/item-metadata-bymykel.parquet by
    # scripts/ingest_bymykel_metadata.py. Off by default: the effect is measured
    # only on a held-out-item CV instrument over an 870-item deep >=$1 universe
    # (+1.26pp at 7d, +3.72pp at 30d raw labels; null at 3d and 14d), and this
    # project's record is that such gains need not survive the production retrain
    # path -- market-relative labels cleared the same harness and were refuted on
    # their own pre-registered rule the same day. Set BYMYKEL_METADATA=1 to enable.
    # See docs/changelog/2026-08-06-bymykel-metadata-ingest.md.
    #
    # These columns get their OWN group rather than reusing item_identity /
    # item_metadata / temporal. Those three carry other, separately-refuted
    # features, so admitting one of them to the allowlist would admit the 2026-07-24
    # ablation's losers alongside the nine measured columns and make any result
    # unattributable.
    # Note `item_age_meta_days` is NOT `item_age_days`. That name is already taken
    # by a DIFFERENT quantity -- observation date minus the first date in the price
    # frame (_add_temporal_features) -- where this one is observation date minus the
    # catalogue first-sale date. Same units, different meaning; sharing the name
    # would silently overwrite one with the other.
    BYMYKEL_META_GROUP = "bymykel_metadata"
    BYMYKEL_META_FEATURES = _BYMYKEL_META_FEATURES
    # Features computed but withheld from training. These are price technicals by
    # name, so the allowlist above would otherwise pull them straight into prod.
    # Volatility-asymmetry and oscillator-divergence primitives (2026-07-26) are
    # shelved: the A/B cleared no gate (see
    # docs/changelog/2026-07-31-price-primitives-shelved.md). The columns are
    # still engineered so ab_test_price_primitives.py -- which builds its own
    # feature list from the frame -- can re-run the arms unchanged.
    # The eleven volume features (2026-08-06) are shelved because the archive's
    # volume column has been identically 0 since 2026-05 — stored as 0, never
    # NULL, which defeats every guard in _compute_volume_features: `has_volume`
    # tests notna() so it stays True, `volume_missing` reports 0 ("present"),
    # and the raw levels are served a real-looking 0 against training medians of
    # 98.0 / 99.0 / 115.7 / 43.9 / 124.2. They carry real signal on the
    # pre-2026-05 training rows and are dead on every served row, which is a
    # pure train/serve gap on 100% of items. Reinstate them only with a repaired
    # feed. See docs/changelog/2026-08-06-volume-features-shelved.md.
    SHELVED_FEATURES = frozenset({
        "vol_semidev_down_30d",
        "vol_semidev_up_30d",
        "vol_skew_30d",
        "rsi_divergence_7d",
        "rsi_price_divergence_7d",
        "macd_hist_slope_7d",
        "volume_missing",
        "volume_lag_1d",
        "volume_lag_7d",
        "volume_mean_30d",
        "volume_std_30d",
        "volume_mean_60d",
        "volume_log_change_1d",
        "volume_log_change_7d",
        "volume_zscore_30d",
        "volume_price_conf_7d",
        "volume_price_conf_1d",
        # These two were never in the 47-column feature_cols: the >0.95
        # correlation prune dropped them in favour of their 30d partners.
        # Shelving those partners removes what they correlated against, so
        # without this they SURVIVE the prune and re-enter production — the one
        # failure mode a name-list assertion cannot see. See
        # test_no_volume_feature_survives_the_real_selection_and_prune.
        "volume_mean_7d",
        "volume_std_60d",
    })

    # The volume-derived subset of SHELVED_FEATURES (see the comment above and
    # tests/test_volume_features_shelved.py) -- every column
    # _compute_volume_features adds. Task 5 recovered a real `volume` column;
    # VOLUME_FEATURES=1 lets these back into training. Listed exhaustively,
    # not by prefix, for the same reason as SHELVED_FEATURES itself: the
    # >0.95 correlation prune is data-dependent, so volume_mean_7d and
    # volume_std_60d (which the prune would otherwise drop in favour of their
    # 30d partners) must be un-shelved together with those partners or they
    # silently survive the prune and re-enter production alone.
    VOLUME_FEATURE_NAMES = frozenset({
        "volume_missing",
        "volume_lag_1d",
        "volume_lag_7d",
        "volume_mean_7d",
        "volume_mean_30d",
        "volume_std_30d",
        "volume_mean_60d",
        "volume_std_60d",
        "volume_log_change_1d",
        "volume_log_change_7d",
        "volume_zscore_30d",
        "volume_price_conf_7d",
        "volume_price_conf_1d",
    })

    @staticmethod
    def _volume_features_enabled() -> bool:
        """Whether the recovered volume columns reach the trained feature set.

        Default OFF: production keeps shelving them per SHELVED_FEATURES'
        volume comment (the archive's volume was identically 0 from 2026-05
        through the recovery in Task 5). Set VOLUME_FEATURES=1 to un-shelve.
        """
        return os.environ.get("VOLUME_FEATURES") == "1"

    def _active_shelved_features(self) -> frozenset:
        """SHELVED_FEATURES, minus the volume names when the flag is on."""
        if self._volume_features_enabled():
            return self.SHELVED_FEATURES - self.VOLUME_FEATURE_NAMES
        return self.SHELVED_FEATURES

    # Dollar-denominated columns. The target is a PERCENTAGE return, so a
    # feature measured in dollars cannot be a price signal — it can only encode
    # which item this is. On the 2026-08-06 artifact these carried 55.6 / 70.2 /
    # 77.5 / 86.6% of total gain at 3/7/14/30d, rising with horizon in step with
    # the served-cohort accuracy gap, while the training median price was $0.086
    # against served items reaching $639. With max_bin = 63 that puts nearly
    # everything above ~$1 into one saturated terminal bin, which is a mechanism
    # for the pooled-67% / >=$1-50% split.
    #
    # Every one of these is still COMPUTED — the conformal band reads
    # price_std_60d, _apply_market_aggregates reads price_std_30d, and the
    # z-score / Bollinger / MA-distance / support-resistance / log-return
    # features are all derived from the means, mins, maxes and lags. They are
    # shelved from the FEATURE SET only, and replaced by the scale-free forms
    # that already existed or were added alongside them:
    #   price_std_{w}d  -> price_cv_{w}d          (new)
    #   macd_line       -> macd_line_rel          (new)
    #   macd_histogram  -> macd_histogram_rel     (new)
    #   price_lag_{n}d  -> return_{n}d / log_return_{n}d   (already features)
    #   price_mean_{w}d -> price_zscore_30d, price_dist_ma{100,200}
    #   price_min/max   -> distance_to_support/resistance, high_low_range_30d
    #   bb_upper/lower  -> bb_pct_b, bb_width
    #   price_log       -> price_tier (bounded categorical, deliberately kept)
    #
    # Listed exhaustively rather than by prefix because the >0.95 correlation
    # prune is data-dependent: a column pruned today can survive tomorrow, which
    # is the failure mode documented for the volume shelf above.
    # test_model_features_are_invariant_to_price_scale asserts the property
    # instead of this list, so a new dollar-scale feature fails without anyone
    # remembering to add it here.
    _DOLLAR_SCALE_FEATURES = frozenset(
        {f"price_std_{w}d" for w in (7, 14, 20, 30, 60)}
        | {f"price_mean_{w}d" for w in (7, 14, 20, 30, 60, 100, 200)}
        | {f"price_min_{w}d" for w in (7, 14, 20, 30, 60)}
        | {f"price_max_{w}d" for w in (7, 14, 20, 30, 60)}
        | {f"price_lag_{n}d" for n in (1, 3, 7, 14, 30, 60, 90, 120, 180)}
        | {"price_log", "macd_line", "macd_signal", "macd_histogram",
           "bb_upper", "bb_lower"}
    )
    SHELVED_FEATURES = SHELVED_FEATURES | _DOLLAR_SCALE_FEATURES
    # Horizons served as momentum (trailing return_Nd) instead of the ML median.
    # Superseded by the directional classifier (2026-07-24), which beats
    # momentum at every horizon including 30d — so this is now empty. Kept as a
    # knob; _recenter_on_momentum still exists for it.
    MOMENTUM_FALLBACK_HORIZONS = []
    # Directional classifier (2026-07-24): a 3-class (down/flat/up) LightGBM
    # trained on multiclass log-loss — optimizing the served metric directly —
    # supplies the direction + confidence. Quantile models still supply the
    # price interval. Training up-weights movers (|return| > flat tolerance) so
    # the classifier spends capacity on the hard up/down calls rather than the
    # easily-predicted flat mass.
    # Per-horizon directional-label knobs (2026-07-27). Defaults: mover-weight
    # keeps the prior global 3.0; vol multiplier k=1.0 is a starting point the
    # sweep (scripts/ab_test_direction_labels.py) tunes per horizon.
    DIRECTION_MOVER_WEIGHT_MAP = {3: 3.0, 7: 3.0, 14: 3.0, 30: 3.0}
    DIRECTION_VOL_MULTIPLIER_MAP = {3: 1.0, 7: 1.0, 14: 1.0, 30: 1.0}
    # Bumped when a fitting-logic change invalidates stored thresholds.
    # v2 (2026-08-03): thresholds must come from a date-coverage-guarded fit.
    # Unversioned files predate the guard, were fitted on a two-date cohort,
    # and are discarded on load.
    BIAS_FIT_SCHEMA_VERSION = 2
    # Max class probability at/above which a directional call is "high" confidence.
    DIRECTION_CONFIDENCE_HIGH = 0.5
    # Residual stacking (a Ridge on LightGBM residuals, applied to 14d/30d) was
    # disabled on 2026-07-25 and is DELETED here. It was fit on RAW, unscaled
    # feature values, so it extrapolated without bound at serving time: a penny
    # item whose return_Nd feature is legitimately +900% got a linear correction
    # in the +100,000% range. On the 2026-inclusive retrain the 14d residual
    # over-corrected 99% of items across every price tier (median +184% for
    # penny, -480% for mid-price) and inverted quantile ordering on 100% of
    # predictions. The served direction comes from the classifier and the band
    # from conformal calibration, so nothing replaces it.
    # Weight given to the previous day's forecast when smoothing/blending
    # current predictions to reduce daily direction flip-flopping.
    FORECAST_BLEND_WEIGHT = 0.15
    PRUNE_CORRELATION_THRESHOLD = 0.95
    # Expanding-window cross-validation.
    # The archive spans ~1,256 distinct trading dates, so we can afford more,
    # wider folds than the old 21d/200-step config (which gave only ~3-6 folds
    # of 21 days). Directional accuracy is dominated by market-wide regime
    # moves, so the effective sample size is the number of independent
    # validation DATES, not rows — widening the window and adding folds is what
    # actually tightens the CI on mean directional accuracy (and gives the
    # conformal calibration more OOF points).
    CV_STEP_DAYS = 150            # was 200; ~7 non-overlapping folds on real data
    CV_MIN_TRAIN_DAYS = 200       # Minimum unique dates before first validation fold
    # Overridable because this is the dominant training cost. Post-minimal-model
    # the out-of-fold conformal CV is ~84% of a warm retrain (~148.6s of 176.7s),
    # since it refits a median model per fold per horizon purely to generate the
    # residuals q_hat is calibrated on. Raising the stride cuts folds ~linearly.
    # Read here, not at class-definition time, so a sweep does not need an edit.
    # Instance method, not classmethod: tests override CV_STEP_DAYS on the
    # INSTANCE to force extra folds out of a small synthetic frame, and a
    # classmethod reading cls.CV_STEP_DAYS would silently ignore them.
    def _cv_step_days(self) -> int:
        return int(os.environ.get("CV_STEP_DAYS", self.CV_STEP_DAYS))

    # Cap on each CV fold's TRAINING rows. `max_rows` was applied only in
    # _build_production_split, so _cv_evaluate_horizon took the whole expanding
    # window every fold: nine folds per horizon summed to 4.2x the frame and the
    # conformal CV phase was 50.4% of an 872s retrain (measured 2026-08-09).
    #
    # Only train is thinned. Fold count, val rows and the OOF record count are
    # the sample size of q_hat, mean_rank_ic and the PT statistic, and none of
    # them moves. The fold model then fits on less data than the served one, so
    # q_hat comes out LARGER and the band wider -- over-coverage, which is the
    # safe direction. Verify against NOMINAL_COVERAGE before lowering it.
    #
    # ✅ VERIFIED 2026-08-12 (run 31611508808), and this cap is NOT why the band
    # over-covers at 87.2/91.8/90.6/89.0% against 80%. Per-fold q_hat says the
    # pooled value sits 0.94/0.91/0.92/0.84x the p80 of the folds already at this
    # cap -- below 1, where the mechanism above needs above -- and dropping the
    # one fold materially under it WIDENS the calibration to 1.06-1.13x. At
    # identical n_train the fold spread is still 1.56-2.25x, so training size
    # explains none of it; the folds track their validation window's volatility
    # regime instead. Raising this cap is therefore a COST decision only, and
    # lowering it still needs the coverage check -- the finding is that the
    # effect is small, not that it is absent.
    # docs/changelog/2026-08-12-expanding-window-refuted-for-band-width.md
    CV_MAX_TRAIN_ROWS = 300_000

    def _cv_max_train_rows(self) -> int:
        return int(os.environ.get("CV_MAX_TRAIN_ROWS", self.CV_MAX_TRAIN_ROWS))

    @staticmethod
    def _record_cv_fold_train_rows(n: int) -> None:
        """Seam for tests to observe the post-cap fold size. No-op in prod."""

    # ------------------------------------------------------------------
    # Boost rounds: fixed, not early-stopped
    # ------------------------------------------------------------------
    # Early stopping was scored against `_build_production_split`'s trailing
    # 30-day window. That window is not a sample of 15,000 rows — at h=14 the
    # last 14 days carry no label and frozen-run voiding removes ~30% more,
    # leaving ~a dozen distinct dates, and every item moves with the market
    # inside a date (docs/changelog/2026-08-03-accuracy-is-clustered-by-
    # forecast-date.md). Its EFFECTIVE sample size is that dozen. Any real
    # fitting therefore reads as a val-loss regression against one different
    # market period, and training halts almost immediately.
    #
    # Measured 2026-08-08 (docs/research/2026-08-08-model-review.md): it left
    # 9 of 33 CV folds fitting a SINGLE tree, the served clf_14d at 3 trees and
    # lgb_14d_q50 at 4, and cost 23-88% of out-of-fold rank IC at 7/14/30d.
    #
    # Replaced by a fixed round count per horizon, calibrated on pooled
    # out-of-fold rank IC using production's tuned params (the calibration
    # sweep is scripts/calibrate_boost_rounds.py). The counts are horizon-
    # specific because the tuned learning rates are not comparable across
    # horizons — 3d is 0.01 and 7d is 0.0053, so one shared count would badly
    # undertrain the slow ones.
    #
    # EARLY_STOPPING=1 restores the old behaviour for a like-for-like read.
    # Set it to reproduce any artifact or measurement from before this change.
    # Calibrated 2026-08-08 by scripts/calibrate_boost_rounds.py: each CV fold
    # trained once to 1500 rounds with production's tuned params, then scored at
    # checkpoints via predict(num_iteration=k). The counts below are the KNEE —
    # the smallest count within 1% of peak pooled out-of-fold rank IC — not the
    # peak, because the curve is flat past it and rounds are the dominant cost.
    #
    # What the sweep showed, and why early stopping had to go: the validation
    # pinball loss and the cross-sectional rank IC point in OPPOSITE directions
    # at the long horizons.
    #
    #   h    val-loss optimum   rank-IC peak    IC @ 25 rounds -> IC at peak
    #   3d      300 rounds        300 rounds       0.1666 -> 0.1893
    #   7d      200 rounds       1500 rounds       0.1003 -> 0.1287
    #  14d       25 rounds        500 rounds       0.1104 -> 0.1204
    #  30d       25 rounds        750 rounds       0.0267 -> 0.0961  (3.6x)
    #
    # At 14d and 30d the val loss is minimised at 25 rounds and rises
    # monotonically after it, so early stopping was not merely noisy — it was
    # optimising a metric anti-correlated with the thing the product needs. 3d
    # is the one horizon where the two agree, which is why it was the only
    # healthy one.
    #
    # Rounds are now the dominant training cost — they have to be, because the
    # old configuration was cheap precisely by not training (1-90 trees). So
    # these are set at the VALUE point, not the peak: the measured CV cost per
    # horizon against the share of peak IC it buys.
    #
    #   h    rounds   share of peak IC   CV phase
    #   3d      200        99.7%            94s     (300 buys +0.3pp for +47s)
    #   7d      500        96.7%           132s     (750 buys +2.6pp for +66s)
    #  14d      100        99.6%            29s     (curve flat 50-750)
    #  30d      750       100.0%           266s     (NOT the value point - see below)
    #
    # 30d is deliberately at the peak rather than the 500-round value point.
    # Measured on two full retrains: at 750 rounds its Pesaran-Timmermann
    # statistic is t=3.06, verdict "skill"; at 500 rounds it is t=2.81, verdict
    # "no_skill". The 4% of rank IC that 500 gives up is what carries the
    # horizon across the project's own PT hurdle, so the cheaper point would
    # buy 89s by making the horizon unpublishable under invariant #4.
    #
    # Production trains on a larger frame than the average expanding-window CV
    # fold, so its counts are ~1.5x the CV count. The curve is very forgiving
    # upward (3d loses 1.5% of peak IC going 300 -> 1500 rounds), so erring
    # high costs wall-clock rather than accuracy.
    FIXED_BOOST_ROUNDS = {3: 300, 7: 750, 14: 150, 30: 1000}
    CV_FIXED_BOOST_ROUNDS = {3: 200, 7: 500, 14: 100, 30: 750}

    @staticmethod
    def _early_stopping_enabled() -> bool:
        return os.environ.get("EARLY_STOPPING") == "1"

    @classmethod
    def _boost_rounds(cls, horizon: Optional[int], cv: bool = False) -> int:
        """Fixed round count for a horizon, or the legacy cap under EARLY_STOPPING=1.

        Falls back to the legacy caps for an unknown horizon so a HORIZONS
        change cannot silently train a 0-round model.
        """
        if cls._early_stopping_enabled():
            return 200 if cv else 1000
        table = cls.CV_FIXED_BOOST_ROUNDS if cv else cls.FIXED_BOOST_ROUNDS
        return table.get(horizon, 300 if cv else 500)

    ENGINEERED_CACHE_NAME = "engineered_data.parquet"
    # Bump when the *shape* of the cached frame changes, not just its contents.
    # v2: the predict path now truncates to PREDICT_TAIL_ITEM_DAYS item-days per
    # item, so a v1 cache holds full history the loader would misread as a tail.
    # Carried in DataFrame.attrs alongside _cache_date (verified to survive the
    # pyarrow round-trip on pandas 2.3.3) rather than as a column: the predict
    # frame reaches ~2M rows and copying it to append a constant would double
    # peak memory on the path that already OOMs in CI.
    ENGINEERED_CACHE_VERSION = 3   # v3: sidecar columns join into the daily frame

    # --- Voted price frame cache ---
    # Bump VOTED_CACHE_VERSION whenever _fetch_voted_price_history or
    # _apply_multi_source_voting changes shape or semantics. The key covers the
    # archive contents and the query window, but it cannot see the code — this
    # constant is the only thing standing between a logic change and a stale
    # frame silently training the next model.
    # v2: BID_SOURCES is excluded from voting, so every cached v1 frame holds a
    # consensus displaced by a median -8.0% (-10.8% on the >=$1 served cohort).
    # v3: PHASE_COLLAPSED_SLUG_PATTERNS leaves the universe, so a v2 frame still
    # carries the Doppler names whose returns are phase-composition artifacts.
    # v4: the phantom slug keys leave the universe. A v3 frame carries both
    # copies of 3,149 items, which is what let a fold score an item it had
    # already trained on.
    # v5: n_ask_sources on the voted frame. A v4 frame lacks the column
    # entirely, so a stale cache would train the next model without it.
    # v6: TRAILING_WINDOW_SOURCES (aggregator_steam_7d/30d/90d) leaves the
    # voting pool. A v5 frame still lets a trailing-window mean vote on equal
    # terms against point-in-time asks, damping the consensus and
    # manufacturing mean-reversion in every return computed across it.
    VOTED_CACHE_VERSION = 6
    VOTED_CACHE_PREFIX = "voted_"
    VOTED_CACHE_MAX_ENTRIES = 3

    def __init__(self, db_session, model_dir: str = None, prune_failed_groups: bool = True,
                 served_cohort_share: Optional[float] = None):
        self.db = db_session
        # Share of the directional classifier's training weight to place on the
        # >= $1 cohort production serves. None = no tier weighting, which is
        # byte-identical to the pre-2026-08-06 model. Set from
        # TRAIN_SERVED_COHORT_SHARE by scripts/forecast_prices.py. See
        # docs/superpowers/specs/2026-08-06-served-cohort-weighting-design.md.
        if served_cohort_share is not None and not 0.0 < served_cohort_share < 1.0:
            raise ValueError(
                f"served_cohort_share must be in (0, 1) or None, "
                f"got {served_cohort_share!r}")
        self.served_cohort_share = served_cohort_share
        self.model_dir = model_dir or str(Path(__file__).parent / "saved_models")
        # Kept off model_dir: that holds gitignored production model artifacts,
        # and tests assert nothing else lands there.
        self.cache_dir = str(Path(__file__).parent.parent / "data")
        self.archive_dir = Path(__file__).parent.parent.parent / "price-archive"
        self.models: Dict[Tuple[int, float], lgb.Booster] = {}
        self.regime_models: Dict[Tuple[str, int, float], list] = {}
        # Per-horizon 3-class directional classifier (down/flat/up).
        self.direction_models: Dict[int, lgb.Booster] = {}
        self.regime_feature_cols: Dict[Tuple[int, str], List[str]] = {}
        self.feature_cols: List[str] = []
        self.prune_failed_groups = prune_failed_groups
        self.tuned_params: Dict[int, Dict[float, Dict[str, Any]]] = {}
        # Per-horizon confidence thresholds: {horizon: {"high_range": ..., "high_change": ..., "high_accuracy": ...}}
        self.confidence_thresholds: Dict[int, Dict[str, float]] = {}
        self.feature_medians: pd.Series = pd.Series(dtype=np.float64)
        # What the LOADED artifact was trained with, or None when nothing has
        # been loaded. Read on the predict path via _tier_lead_served /
        # _cross_sectional_rank_served so serving follows the artifact rather
        # than whatever the environment happens to hold; training reads the
        # environment directly. None, not False, so "absent from an older
        # meta.json" is distinguishable from "trained with it off".
        self._artifact_tier_lead: Optional[bool] = None
        self._artifact_xs_rank: Optional[bool] = None
        self._artifact_naive_init: Optional[bool] = None
        # The median-price floor the artifact was TRAINED under. Load-bearing
        # only when the rank transform is on, and then it is load-bearing
        # absolutely: the transform's output depends on which items are in the
        # cross-section, and training's is the >= $1 cohort while predict's
        # frame is every backfilled item. See _reference_cohort_mask.
        self._artifact_min_median_price: Optional[float] = None
        self._artifact_cohort_items: Optional[int] = None
        # Set by train() so save_models can record what the frame was built
        # under. Not read on the predict path -- that reads the artifact.
        self._train_min_median_price: Optional[float] = None
        self._train_cohort_items: Optional[int] = None
        # Which columns the date-constant skip removed at training. None means
        # "the artifact does not say"; [] means "nothing was skipped", and the
        # two are not the same -- see the predict guard.
        self._artifact_xs_rank_skipped: Optional[List[str]] = None
        self._train_xs_rank_skipped: Optional[List[str]] = None
        # Conformal calibration per horizon. NOTE: the meaning changed with the
        # minimal model — q_hat is now a DIMENSIONLESS multiplier of the
        # per-item sigma, not a percentage-point addend. Applied as:
        #   low = mid - q_hat * sigma_i,  high = mid + q_hat * sigma_i
        # The previous CQR scheme widened a [p10, p90] base interval by a
        # constant. Loading an old artifact into this code would silently
        # produce a band computed two different ways, which is why
        # MODEL_ARTIFACT_VERSION exists.
        self.conformal_calibration: Dict[int, float] = {}
        self._init_conformal_state()
        # Expanding-window CV results per horizon: {horizon: {fold_count, fold_accs, per_fold, ...}}
        self.cv_results: Dict[int, Dict] = {}
        # Event decay constants (grid-searchable per event type)
        self.horizon_feature_cols: Dict[int, List[str]] = {}
        self.event_decay_constants: Dict[str, float] = {
            "major": 60,
            "operation": 21,
            "case_drop": 14,
            "update": 7,
            "game_update": 7,
        }
        # Per-tier bias corrections: {horizon: {tier_label: correction_pct}}
        # Correction is ADDED to mid_ret (positive shifts predictions upward)
        # DEPRECATED in favor of bias_thresholds below.
        self.bias_corrections: Dict[int, Dict[str, float]] = {}
        # Threshold-based corrections: {horizon: {tier_label: {"t_down": x, "t_up": y}}}
        # Recalibrates classification boundaries to match the true outcome base rate
        # instead of shifting mid_ret (which pushes predictions into the flat dead-zone).
        self.bias_thresholds: Dict[int, Dict[str, dict]] = {}
        # EWMA state for tracking which tiers have been seen
        self.bias_ewma_state: Dict[int, Dict[str, int]] = {}
        # What the label path voided, recorded so a reader can tell a handled
        # cutover from an unhandled one. `_collection_shift_dates` fires 12
        # times in 4,735 days and that list has never been written down, which
        # is how a 2026-08-09 audit re-reported the handled 2026-07-09/10
        # cutover as a new finding.
        self.label_voiding: dict = {}

    @staticmethod
    def _smoothed_anchor_prices(df: "pd.DataFrame", anchor) -> Dict[Any, float]:
        """Per-item base price for the dollar conversion: a span-bounded median.

        The median of the most recent SMOOTH_WINDOW observations that lie
        within MAX_WINDOW_SPAN_DAYS of *anchor*.

        The bound is the point. This previously took `tail(3)` — the last three
        *rows* — with no calendar constraint, so a sparsely observed item could
        anchor on prices months apart and serve their median as today's value.
        That is the price-laundering shape `db5bddb` removed from the
        collector's historical fallback, and it is why serving's
        `current_price` diverges from the backtest's archive-resolved
        `base_price` by a median 3.6% (90th percentile 35%) despite both being
        documented as "the 3-day median". Both sides now apply one staleness
        convention, ultimately derived from
        `collectors.pipeline.FALLBACK_MAX_AGE_DAYS`.

        Unlike the backtest, serving may not drop an item: an unresolvable
        anchor there is one fewer scored row, but here it is a missing product.
        An item with nothing inside the window therefore falls back to its
        latest observation — no smoothing, but never a manufactured price, and
        never a silent disappearance.
        """
        from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS, SMOOTH_WINDOW

        if df.empty:
            return {}

        anchor = pd.Timestamp(anchor)
        cutoff = anchor - pd.Timedelta(days=MAX_WINDOW_SPAN_DAYS)
        ordered = df.sort_values(["item_id", "date"])
        # The predict frame carries datetime.date; the tests and the archive
        # path carry Timestamps. Compare in one type rather than assuming.
        dates = pd.to_datetime(ordered["date"])

        in_window = ordered[(dates >= cutoff) & (dates <= anchor)]
        smoothed = (
            in_window.groupby("item_id").tail(SMOOTH_WINDOW)
            .groupby("item_id")["price"].median()
        )

        # Items with no in-window observation keep their latest known price.
        latest = ordered.groupby("item_id")["price"].last()
        return latest.to_dict() | smoothed.to_dict()

    @staticmethod
    def outlier_gated_anchor_enabled() -> bool:
        """Whether serving quotes the RAW price unless it looks like a spike.

        Off (shipped): `predict` substitutes the smoothed anchor for every item,
        outlier or not, so every served `current_price` carries the wedge
        `p[d]/S[d]` -- the last raw quote over the median production quotes from
        -- and is a number no venue published.

        On (arm A): the substitution is conditional on the >10% deviation test
        the code's own warning already describes, so the wedge is confined to the
        items the smoothing was motivated by.

        **This is a SERVING change, not a label change**, which is the whole
        point: `2026-08-11-smoothed-anchor-label-measured.md` measured moving the
        label's denominator and refuted it -- the gain was `p/S` re-entering as a
        factor readable at the anchor, worth -0.02 to -0.03 on the tied cohort
        where the arithmetic cannot operate. Narrowing the wedge at source is the
        question that experiment raised and did not answer.

        **Read it on dollar error, never on rank IC.** The arm moves
        `current_price`, which the replay's headline divides both the prediction
        and the outcome by -- so a rank IC across arms measures a redefinition.
        `scripts/replay_serving.py` prints a basis-free dollar table and a
        `pinnedIC` for exactly this. The gate is the deviating cohort, because on
        the tied cohort both arms serve an identical price.

        Set SERVE_OUTLIER_GATED_ANCHOR=1. Off by default and **unmeasured**. See
        `docs/superpowers/plans/2026-08-11-serving-anchor-freshness.md`.
        """
        return os.environ.get("SERVE_OUTLIER_GATED_ANCHOR") == "1"

    @classmethod
    def _serving_base_price(cls, price: "pd.Series",
                            smoothed: "pd.Series") -> "tuple[pd.Series, pd.Series]":
        """The base price the dollar conversion uses, and the deviation mask.

        Returns `(base, deviates)`. `deviates` is **arm-invariant**: it is what
        the `using smoothed price` warning counts and what the replay's gate is
        read on, so both arms have to describe the same cohort.

        `smoothed > 0` guards the division, and it also decides the degenerate
        cases: an item with no smoothed value (NaN) or a zero one is not a
        deviation, and under either arm it keeps its raw quote. Serving degrades,
        it never drops -- a NaN base price would void the item's whole forecast.
        """
        deviates = (smoothed > 0) & (
            (price - smoothed).abs() / smoothed > cls.ANCHOR_OUTLIER_TOLERANCE
        )
        if cls.outlier_gated_anchor_enabled():
            # Raw unless it spikes. `where` keeps `price` where the mask is
            # False, so a NaN or zero smoothed value falls through to the quote.
            base = price.where(~deviates, smoothed)
        else:
            base = smoothed.fillna(price)
        return base, deviates

    @staticmethod
    def _anchor_disclosure(price: "pd.Series",
                           smoothed: "pd.Series") -> "tuple[pd.Series, pd.Series]":
        """`(anchor_clean, anchor_wedge_pct)` for the served rows.

        `anchor_clean` is the cohort split every 2026-08-11 served-signal figure
        was measured on: the model reaches rank IC +0.13 to +0.17 at 3/7/14d
        where the anchor quote equals its own local median and ~0 or negative
        where it does not (`2026-08-11-clean-anchor-confirmed-in-ci.md`, 4 CI
        anchors of 4). `api/serving_policy.py::meets_anchor_gate` keeps the
        deviating cohort off the ranked surfaces.

        **`price` must be the RAW quote.** `_serving_base_price` overwrites it
        with the served base, and under the shipped arm that base *is*
        `smoothed` -- so this called after that line reports a clean catalogue.

        **Exact equality, matching `replay_serving._tied_mask`** (`rtol=0,
        atol=1e-9`), NOT `ANCHOR_OUTLIER_TOLERANCE`. The 10% test describes a
        different and far smaller population, and nothing has been measured on
        it. `anchor_wedge_pct` is published beside the flag because the split at
        exact equality is what was measured and whether the effect is a cliff
        there or monotone in `|p/S - 1|` is not.

        **Arm-invariant.** Reads neither `outlier_gated_anchor_enabled` nor
        `label_smoothed_anchor_enabled`; two arms must describe one cohort.

        A non-positive or missing median yields `clean=False` and a NaN wedge:
        unknown is not clean, the same rule the replay's mask follows.
        """
        p = price.to_numpy(dtype=float)
        s = smoothed.to_numpy(dtype=float)
        usable = np.isfinite(s) & (s > 0) & np.isfinite(p)
        clean = pd.Series(np.isclose(p, s, rtol=0, atol=1e-9) & usable,
                          index=price.index, name=ANCHOR_TIED_COL)
        wedge = pd.Series(
            np.where(usable, (p / np.where(usable, s, 1.0) - 1.0) * 100.0,
                     np.nan),
            index=price.index, name="anchor_wedge_pct")
        return clean, wedge

    @staticmethod
    def label_smoothed_anchor_enabled() -> bool:
        """Whether the LABEL divides by the price `predict` quotes from.

        Off: `prepare_targets` divides by the raw quote observed on the anchor
        day. That quote is also what `return_1d` and every level feature are
        built from, so one noisy observation deflates the label and inflates the
        feature together, and a model that reads the noise scores as if it had
        read the market.

        On: the denominator is `_smoothed_anchor_prices`' span-bounded median,
        computed per anchor day -- the same statistic the serving path converts
        a return-space forecast to dollars with, and one no single quote can
        move.

        Measured, not hypothesised. The 2026-08-11 serving replay crossed the
        two axes between the CV label and the served one and found the
        denominator carries **+0.1398 of the +0.1464 rank IC gap**; the outcome
        leg carries +0.0251 and the serving transforms none. In CI on a fresh
        artifact at four non-overlapping anchors the gap is zero (within
        +/-0.021) on items whose anchor quote already equals its local median
        and positive in all sixteen deviating cells.

        **A rank IC under this flag is not comparable to one without it.** The
        two arms are scored against different targets, so the CV number moves
        for reasons that have nothing to do with forecast quality. Read the arms
        through `scripts/replay_serving.py`, which scores both on one basis.

        Set LABEL_SMOOTHED_ANCHOR=1. See
        `docs/changelog/2026-08-11-the-gap-is-the-anchor-denominator.md`.
        """
        return os.environ.get("LABEL_SMOOTHED_ANCHOR") == "1"

    @staticmethod
    def sigma_exponent_enabled() -> bool:
        """Whether the band divides by `sigma ** beta` instead of `sigma`.

        Off: `sigma ** 1.0`, which is what locally-weighted split conformal
        assumes and what this repo served until 2026-08-12. The assumption is
        that a twice-as-volatile item's residual is twice as large.

        On: `beta` is fitted per horizon on the same OOF records `q_hat` is,
        persisted as `conformal_beta` in `meta.json`, and applied in BOTH
        `conformal.calibrate` and `conformal.band`.

        Measured, not hypothesised. The elasticity is **0.429 / 0.369 / 0.350 /
        0.313** on ~157K real OOF records (run `31619383780`, within 0.03 of a
        model-free instrument over 731 dates), against the 1.000 assumed — so
        `sigma`'s cross-sectional range is ~4x too wide for the dispersion it
        normalises and level-matched coverage ramps **62->93 / 60->95 / 57->96 /
        52->97%** across `sigma` deciles. Walk-forward over 507-588 dates at this
        repo's 14-day retrain cadence, one fitted exponent per horizon cuts that
        tilt **-89% / -94% / -84% / -73%** held out and narrows the median band to
        **0.87 / 0.86 / 0.84 / 0.77x**. Shrinking `beta` toward 1 is a **no-op**
        (its departure from 1 is 6-14x the standard error of its own estimate) and
        a non-parametric scale buys nothing outside 30d — both were measured and
        refuted, so this is deliberately ONE float per horizon.

        ⚠️ **`q_hat` and `beta` are a matched pair.** `sigma` is ~0.07, so
        `sigma ** 0.4` is ~5x larger and `q_hat` absorbs the difference. An
        artifact's `q_hat` served at the wrong exponent is wrong by ~5x, not
        partially corrected, which is why they are written together, why a missing
        `conformal_beta` defaults to 1.0, and why no `q_hat` may be differenced
        across this flag.

        ⚠️ **It does NOT fix the marginal over-coverage** (87.2/91.8/90.6/89.0%
        vs 80%). It closes the `sigma`-mix channel, which is 36-68% of that
        defect; the rest is a residual-law shift. On a calm period the corrected
        band covers **74-77%**, i.e. this leaves the level unresolved in BOTH
        directions and no 80% claim rests on it.

        Set SIGMA_EXPONENT=1. See
        `docs/superpowers/specs/2026-08-12-sigma-exponent-design.md` and
        `docs/changelog/2026-08-12-the-sigma-scale-is-one-exponent-per-horizon.md`.
        """
        return os.environ.get("SIGMA_EXPONENT") == "1"

    @staticmethod
    def conformal_served_basis_enabled() -> bool:
        """Whether `q_hat` is calibrated on the denominator serving quotes from.

        Off (default): the residual is measured against the training label,
        whose denominator is the raw anchor quote `p[d]`. That is incoherent
        with serving -- `predict` quotes from the smoothed anchor `S[d]` and
        `resolve_anchors` scores against it -- and the served band over-covers
        at 87.2 / 91.8 / 90.6 / 89.0% against an 80% target.

        On: the residual is measured against `calibration_target_col(h)`, which
        divides by `S[d]`. The training label is untouched either way, so unlike
        `LABEL_SMOOTHED_ANCHOR` this cannot hand the model a factor -- `q_hat`
        is post-hoc and changes only a band width.

        **Off by default because it is REFUTED as a remedy.** Paired on one
        commit -- arm `31564924194` against control `31564943172`, same fold
        counts, identical n per horizon -- it moves q_hat the WRONG way at all
        four horizons: 94.72 -> 103.70, 141.77 -> 148.80, 204.34 -> 205.80,
        312.05 -> 316.96 (+9.5 / +5.0 / +0.7 / +1.6%). The over-coverage needs
        q_hat 25-39% SMALLER.

        **The sign is itself the finding.** With r_hat ~ 0 the raw-basis
        residual would be the more dispersed of the two (`R/k - 1` against
        `R - 1`, with k = p[d]/S[d] scattered about 1), so this arm should have
        shrunk q_hat. It widened it. The only way the raw-basis residual comes
        out smaller is if r_hat already contains the k term and cancels part of
        it -- the booster is trained on the raw-basis label and `return_1d` is
        built from the same raw quote, so it can read the anchor deviation and
        evidently does. Against a smoothed-basis outcome that component becomes
        added error instead of cancelled error.

        So: the calibration basis is NOT the cause of the over-coverage, fixing
        the incoherence makes the symptom worse, and the coherent fix has to
        move the prediction and the residual together. Kept as an instrument
        because the incoherence is real and the flag is how it is measured.

        Set CONFORMAL_SERVED_BASIS=1. See
        `docs/changelog/2026-08-12-conformal-basis-follows-serving.md`.
        """
        return os.environ.get("CONFORMAL_SERVED_BASIS") == "1"

    @staticmethod
    def _rolling_anchor_prices(df: "pd.DataFrame") -> "pd.Series":
        """`_smoothed_anchor_prices`, evaluated at every row's own date.

        The serving version resolves ONE anchor across items and returns a dict;
        a label needs the same statistic at ~900 items x ~4,700 dates, so this
        is the panel form: the median of the most recent SMOOTH_WINDOW
        observations lying within MAX_WINDOW_SPAN_DAYS *before* the row's date.

        Built from k-step shifts rather than a rolling window because the two
        bounds are of different kinds -- a count (SMOOTH_WINDOW rows) and a span
        (MAX_WINDOW_SPAN_DAYS days) -- and pandas' rolling takes one or the
        other. Shifting gives both: take the k most recent rows, then blank the
        ones that fall outside the span.

        The row's own observation is always in its own window (k=0, zero days
        old), so the result is never NaN and the serving fallback to
        `latest known price` has nothing to fall back from. Returned on the
        caller's index, unsorted.
        """
        ordered = df.sort_values(["item_id", "date"])
        day = pd.to_datetime(ordered["date"])
        span = pd.to_timedelta(int(MAX_WINDOW_SPAN_DAYS), unit="D")
        by_item = ordered.groupby("item_id", sort=False)

        legs = []
        for k in range(SMOOTH_WINDOW):
            price = by_item["price"].shift(k)
            age = day - day.groupby(ordered["item_id"]).shift(k)
            legs.append(price.where(age <= span))
        med = pd.concat(legs, axis=1).median(axis=1, skipna=True)
        return med.reindex(df.index)

    @classmethod
    def _anchor_is_tied(cls, df: "pd.DataFrame") -> "pd.Series":
        """Per row: does the anchor's raw quote equal its own local median?

        The clean cohort. `prepare_targets` divides the label by `p[d]` and
        `predict` divides the served forecast by `S[d]`, so a rank IC on either
        basis is partly a measurement of the wedge `p[d]/S[d]` -- positive on
        the raw basis, negative on the smoothed one
        (`2026-08-11-smoothed-anchor-label-measured.md`). Where `p[d] == S[d]`
        the wedge is 1 and neither term exists.

        **Arm-invariant, and it has to stay that way.** This reads the price
        series only -- never `label_smoothed_anchor_enabled()` or
        `outlier_gated_anchor_enabled()`. A mask that followed either flag would
        score the control and the arm on different populations, which is the
        exact failure the cohort was introduced to remove.

        `atol=1e-9, rtol=0` matches `replay_serving._tied_mask` exactly, because
        every published tied/deviating number came from that function and the
        two must not describe different cohorts under one word. A NaN median
        yields False for the same reason it does there: an item with nothing to
        compare is unknown, not clean.
        """
        smoothed = cls._rolling_anchor_prices(df)
        return pd.Series(
            np.isclose(df["price"].to_numpy(dtype=float),
                       smoothed.to_numpy(dtype=float),
                       rtol=0, atol=1e-9),
            index=df.index, name=ANCHOR_TIED_COL)

    @staticmethod
    def _get_price_tier(price: float) -> str:
        for lo, hi, label in PRICE_TIER_BOUNDARIES:
            if lo <= price < hi:
                return label
        return ">$100"

    # ------------------------------------------------------------------
    # Bias correction
    # ------------------------------------------------------------------

    def _load_bias_corrections(self):
        path = os.path.join(self.model_dir, "bias_corrections.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.bias_corrections = {int(k): v for k, v in data.get("corrections", {}).items()}
                try:
                    version = int(data.get("schema_version", 0))
                except (TypeError, ValueError):
                    # Malformed schema_version (null, a string, a list/dict, ...)
                    # is provenance we can't trust either -- treat it the same
                    # as "unversioned" rather than nuking the whole file
                    # (which would also drop the still-valid `corrections`
                    # dict already parsed above).
                    version = 0
                if version < self.BIAS_FIT_SCHEMA_VERSION:
                    logger.warning(
                        f"  bias_corrections.json is schema v{version} < "
                        f"v{self.BIAS_FIT_SCHEMA_VERSION}; discarding stored "
                        f"thresholds. Pre-v2 files were fitted without "
                        f"forecast-date coverage and can sit at the ±3.0 clamp "
                        f"rail. Reverting to defaults until a guarded fit runs."
                    )
                    self.bias_thresholds = {}
                    self.bias_ewma_state = {}
                else:
                    raw_thresholds = data.get("thresholds", {})
                    self.bias_thresholds = {int(k): v for k, v in raw_thresholds.items()}
                    self.bias_ewma_state = {int(k): v for k, v in data.get("ewma_state", {}).items()}
                self._fill_missing_threshold_defaults()
                logger.info(f"  Loaded bias corrections for {len(self.bias_corrections)} horizons, "
                            f"thresholds for {len(self.bias_thresholds)} horizons")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"  Corrupt bias_corrections.json ({e}), using defaults")
                self._set_default_bias_corrections()
        else:
            logger.info("  No bias_corrections.json found, using defaults")
            self._set_default_bias_corrections()

    def _save_bias_corrections(self):
        data = {
            "schema_version": self.BIAS_FIT_SCHEMA_VERSION,
            "corrections": {str(k): v for k, v in self.bias_corrections.items()},
            "thresholds": {str(k): v for k, v in self.bias_thresholds.items()},
            "ewma_state": {str(k): v for k, v in self.bias_ewma_state.items()},
        }
        path = os.path.join(self.model_dir, "bias_corrections.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("  Saved bias corrections")

    def _set_default_bias_corrections(self):
        self.bias_corrections = {h: {} for h in self.HORIZONS}
        self.bias_thresholds = {}
        for h in self.HORIZONS:
            self.bias_thresholds[h] = {}
            for _, _, label in PRICE_TIER_BOUNDARIES:
                self.bias_thresholds[h][label] = {
                    "t_down": -DIRECTION_FLAT_TOLERANCE_PCT,
                    "t_up": DIRECTION_FLAT_TOLERANCE_PCT,
                }
        self.bias_ewma_state = {h: {} for h in self.HORIZONS}

    def _fill_missing_threshold_defaults(self):
        defaults = {"t_down": -DIRECTION_FLAT_TOLERANCE_PCT, "t_up": DIRECTION_FLAT_TOLERANCE_PCT}
        filled = 0
        for h in self.HORIZONS:
            if h not in self.bias_thresholds:
                self.bias_thresholds[h] = {}
            for _, _, label in PRICE_TIER_BOUNDARIES:
                if label not in self.bias_thresholds[h]:
                    self.bias_thresholds[h][label] = dict(defaults)
                    filled += 1
        if filled:
            logger.info(f"  Filled {filled} missing tier defaults in bias_thresholds")

    def update_bias_corrections_from_outcomes(self):
        """Query forecast_outcomes, compute per-tier threshold-based bias
        correction from the actual outcome distribution, and update.

        For each tier/horizon:
          1. Compute the true base rate: what % of items actually go up/down.
          2. Find threshold pair (t_down, t_up) on the predicted mid_ret
             distribution that makes the predicted up/down/flat split match
             the actual observed split.
          3. Store thresholds for use in predict()'s direction classification.

        This replaces the old additive-shift/50-50 logic which mostly dumped
        predictions into the flat dead-zone.
        Called after a backtest run.
        """
        try:
            # NOTE: fo.direction_actual is a derived verdict, not an
            # observation — the reported accuracy re-derives direction from
            # base_price/actual_price on every run (see
            # backtest_accuracy._records_from_frozen_outcomes) and never reads
            # this column. It is safe to fit production predict() thresholds on
            # because every scoring run also refreshes the stored column to
            # match that derivation
            # (backtest_accuracy._refresh_verdict_columns). In the daily job
            # there is no staleness window at all: run_backtest() calls
            # backtest_forecasts() — which refreshes — and only then calls this,
            # in the same process, so the labels fitted here are the ones the
            # headline just reported.
            rows = self.db.execute(text("""
                SELECT fo.horizon_days, fo.current_price, fo.predicted_price_mid,
                       fo.direction_actual, fo.forecast_date
                FROM forecast_outcomes fo
                WHERE fo.model_version LIKE 'lgbm-v3%'
                  AND fo.current_price > 0
                  AND fo.direction_actual IS NOT NULL
            """)).fetchall()
        except Exception as e:
            logger.warning(f"  Could not query outcomes for bias update: {e}")
            return

        if not rows:
            logger.warning("  No outcome rows found for bias update")
            return

        df = pd.DataFrame(rows, columns=[
            "horizon_days", "current_price", "predicted_price_mid",
            "direction_actual", "forecast_date",
        ])
        df["tier"] = df["current_price"].apply(self._get_price_tier)
        df["approx_mid_ret"] = (df["predicted_price_mid"] / df["current_price"] - 1) * 100

        # Safety constants for threshold fitting
        MIN_THRESHOLD_SAMPLE = 100       # require this many rows to trust percentile fit
        MIN_FLAT_MARGIN = 0.15           # minimum flat-zone width in percentage points
        DFLT_T_DOWN = -DIRECTION_FLAT_TOLERANCE_PCT
        DFLT_T_UP = DIRECTION_FLAT_TOLERANCE_PCT

        for (horizon, tier), g in df.groupby(["horizon_days", "tier"]):
            n = len(g)
            if n < 20:
                continue

            if not self._has_date_coverage(g["forecast_date"]):
                n_dates = g["forecast_date"].nunique(dropna=True)
                logger.warning(
                    f"  Threshold[{horizon}d, {tier}]: refusing to fit — "
                    f"{n_dates} distinct forecast date(s) < {MIN_FORECAST_DATES} "
                    f"(n={n} rows). Leaving this tier's stored thresholds unchanged."
                )
                continue

            mid_rets = g["approx_mid_ret"].values
            if len(mid_rets) == 0:
                continue

            # Skip if predicted mid_ret distribution is too narrow — the
            # model produces no usable directional signal for this tier/horizon.
            iqr = float(np.percentile(mid_rets, 75) - np.percentile(mid_rets, 25))
            if iqr < 0.25:
                logger.info(f"  Threshold[{horizon}d, {tier}]: skipping (mid_ret IQR={iqr:.2f} < 0.25)")
                continue

            actual_up = (g["direction_actual"] == "up").mean()
            actual_down = (g["direction_actual"] == "down").mean()

            # t_up: threshold above which pred is "up"
            # Want P(mid_ret > t_up) = actual_up → t_up = (1-actual_up) quantile
            t_up_est = float(np.percentile(
                mid_rets, max(0, min(100, (1 - actual_up) * 100))))
            # t_down: threshold below which pred is "down"
            # Want P(mid_ret < t_down) = actual_down → t_down = actual_down quantile
            t_down_est = float(np.percentile(
                mid_rets, max(0, min(100, actual_down * 100))))

            # Clamp to [-3, 3]
            t_up_est = max(-3.0, min(3.0, t_up_est))
            t_down_est = max(-3.0, min(3.0, t_down_est))

            # Enforce minimum flat-zone width
            if t_up_est - t_down_est < MIN_FLAT_MARGIN:
                center = (t_up_est + t_down_est) / 2.0
                half = MIN_FLAT_MARGIN / 2.0
                t_down_est = center - half
                t_up_est = center + half
                t_up_est = max(-3.0, min(3.0, t_up_est))
                t_down_est = max(-3.0, min(3.0, t_down_est))

            # If order flipped, fall back to defaults
            if t_down_est >= t_up_est:
                t_down_est, t_up_est = DFLT_T_DOWN, DFLT_T_UP

            # Conservative shrinkage toward ±0.5 for small samples
            if n < MIN_THRESHOLD_SAMPLE:
                shrink = n / float(MIN_THRESHOLD_SAMPLE)
                t_down_est = shrink * t_down_est + (1 - shrink) * DFLT_T_DOWN
                t_up_est = shrink * t_up_est + (1 - shrink) * DFLT_T_UP

            current = self.bias_thresholds.get(horizon, {}).get(tier, {})
            curr_t_down = current.get("t_down", DFLT_T_DOWN)
            curr_t_up = current.get("t_up", DFLT_T_UP)

            n_seen = self.bias_ewma_state.get(horizon, {}).get(tier, 0)
            if n_seen == 0:
                new_t_down = t_down_est
                new_t_up = t_up_est
            else:
                new_t_down = BIAS_EWMA_ALPHA * t_down_est + (1 - BIAS_EWMA_ALPHA) * curr_t_down
                new_t_up = BIAS_EWMA_ALPHA * t_up_est + (1 - BIAS_EWMA_ALPHA) * curr_t_up

            if horizon not in self.bias_thresholds:
                self.bias_thresholds[horizon] = {}
            self.bias_thresholds[horizon][tier] = {
                "t_down": round(new_t_down, 2),
                "t_up": round(new_t_up, 2),
            }

            if horizon not in self.bias_ewma_state:
                self.bias_ewma_state[horizon] = {}
            self.bias_ewma_state[horizon][tier] = n_seen + 1

            logger.info(f"  Threshold[{horizon}d, {tier}]: "
                        f"t_down={curr_t_down:+.2f}→{new_t_down:+.2f}, "
                        f"t_up={curr_t_up:+.2f}→{new_t_up:+.2f} "
                        f"(actual_up={actual_up*100:.1f}% actual_down={actual_down*100:.1f}%, n={n})")

        self._save_bias_corrections()

    # ------------------------------------------------------------------
    # Regime switching
    # ------------------------------------------------------------------

    def _assign_regime_label(self, market_return_30d: float) -> str:
        """Assign a regime label based on market_return_30d.

        Thresholds:
          bear:  market_return_30d < -3%
          range: -3% <= market_return_30d <= 3%
          bull:  market_return_30d > 3%
        """
        if market_return_30d < self.REGIME_RETURN_THRESHOLD_BEAR:
            return "bear"
        elif market_return_30d <= self.REGIME_RETURN_THRESHOLD_BULL:
            return "range"
        else:
            return "bull"

    def _assign_regime_labels(self, df: pd.DataFrame) -> pd.Series:
        """Assign a regime label to each row based on market_return_30d.

        Returns a Series of regime strings ("bear", "range", "bull")
        indexed like df. Rows without market_return_30d get "range".
        """
        if "market_return_30d" not in df.columns:
            return pd.Series("range", index=df.index)
        labels = df["market_return_30d"].apply(
            lambda x: self._assign_regime_label(x) if pd.notna(x) else "range"
        )
        return labels

    def _detect_current_regime(self, df: pd.DataFrame) -> str:
        """Detect the current market regime from the latest engineered data.

        Uses the most recent row's market_return_30d (if available).
        Falls back to 'range' when undetermined.
        """
        if "market_return_30d" not in df.columns:
            return "range"
        latest_rets = df["market_return_30d"].dropna()
        if latest_rets.empty:
            return "range"
        return self._assign_regime_label(latest_rets.iloc[-1])

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    @staticmethod
    def replay_anchor() -> Optional[date]:
        """`REPLAY_ANCHOR=YYYY-MM-DD` rewinds the serving clock.

        Set, every window `predict` derives -- the fetch cutoff, the voted
        cache key, the model age gate, the prior-day blend -- resolves as if
        today were that date, and the archive read gains an UPPER bound so the
        replay cannot see past its own anchor.

        This exists because the serving path is the one thing walk-forward CV
        does not score. `_recenter_on_direction`, the tier bias, the prior-day
        blend and the conformal band all run inside `predict()` and nowhere
        else, so the only way to measure them was to publish a forecast and
        wait for it to mature. A replay scores them against outcomes the
        archive already holds.

        Research and diagnostics only. It must never be set on the daily path:
        every forecast it produces is dated in the past.
        """
        raw = os.environ.get("REPLAY_ANCHOR", "").strip()
        if not raw:
            return None
        try:
            return date.fromisoformat(raw)
        except ValueError:
            raise ValueError(
                f"REPLAY_ANCHOR={raw!r} is not an ISO date (YYYY-MM-DD). "
                f"Refusing to guess -- an unparsed anchor would silently "
                f"replay against today and score a forecast on its own answer."
            )

    # The serving transforms that move the median, in the order `predict`
    # applies them. The conformal band is deliberately absent: it sets `low` and
    # `high` around the mid and cannot move the mid's cross-sectional ranking.
    REPLAY_DISABLABLE = frozenset({"blend", "bias", "recenter"})

    @classmethod
    def replay_disabled(cls) -> frozenset:
        """`REPLAY_DISABLE=blend,bias,recenter` drops serving transforms.

        Exists to answer one question: CV reports rank IC 0.09-0.18 while the
        serving replay of the same artifact reads near zero
        (`docs/changelog/2026-08-11-rank-transform-does-not-transfer-to-serving.md`).
        Three transforms sit between those two measurements and none has ever
        been scored. Turning them off one at a time at a fixed anchor attributes
        the loss.

        **Honoured only under `REPLAY_ANCHOR`.** Set without one, it raises
        rather than being ignored: a knob that silently changes what production
        serves is worse than no knob, and "ignored" is indistinguishable from
        "applied" in a log. An unknown name also raises -- a typo would
        otherwise read as a clean control and quietly measure nothing.
        """
        raw = os.environ.get("REPLAY_DISABLE", "").strip()
        if not raw:
            return frozenset()
        names = frozenset(n.strip() for n in raw.split(",") if n.strip())
        unknown = names - cls.REPLAY_DISABLABLE
        if unknown:
            raise ValueError(
                f"REPLAY_DISABLE={raw!r} names {sorted(unknown)}, which is not "
                f"a serving transform. Known: {sorted(cls.REPLAY_DISABLABLE)}. "
                f"A typo here reads as a control run and measures nothing.")
        if cls.replay_anchor() is None:
            raise ValueError(
                f"REPLAY_DISABLE={raw!r} is set without REPLAY_ANCHOR. This "
                f"knob is for research replays only; honouring it on the live "
                f"path would change what production serves.")
        return names

    def _now(self):
        anchor = self.replay_anchor()
        if anchor is not None:
            return datetime(anchor.year, anchor.month, anchor.day,
                            tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    def fetch_price_history(self, days_back: int = 365,
                            backfilled_only: bool = False) -> pd.DataFrame:
        logger.info(f"Fetching price history (last {days_back}d)...")

        if self.archive_dir.exists() and days_back > 14:
            backfilled_slugs = (self._resolve_backfilled_slugs()
                                if backfilled_only else None)

            # The voted frame is a pure function of the archive contents plus
            # the query window, so repeated runs over an unchanged archive —
            # walk-forward folds, A/B harnesses, retrain iteration — can skip
            # the rebuild. Measured on the full archive: 35.5s cold, 0.3s
            # cached. That is the whole win; it is not a retrain-time lever
            # (a cold retrain is ~12m30s and almost entirely model fitting).
            cache_key = self._voted_cache_key(days_back, backfilled_only,
                                              backfilled_slugs)
            cached = self._load_voted_cache(cache_key)
            if cached is not None:
                return cached

            df = self._fetch_voted_price_history(
                days_back=days_back,
                backfilled_only=backfilled_only,
                backfilled_slugs=backfilled_slugs,
            )
            self._save_voted_cache(cache_key, df)
            return df

        cutoff = self._now() - timedelta(days=days_back)
        rows = self.db.execute(text("""
            SELECT item_id, date(timestamp) AS day, source, AVG(price) AS price, SUM(volume) AS volume
            FROM price_history
            WHERE timestamp >= :cutoff
              AND source NOT LIKE 'synthetic_demo'
              AND source NOT LIKE 'historical_fallback:%'
            GROUP BY item_id, date(timestamp), source
            ORDER BY item_id, day
        """), {"cutoff": cutoff}).fetchall()
        if not rows:
            logger.info("  No DB price history rows found")
            return pd.DataFrame(columns=["item_id", "timestamp", "price", "volume", "date"])
        df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume", "source"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        n_before = len(df)
        n_sources_before = df["source"].nunique() if "source" in df.columns else 1
        df = self._apply_multi_source_voting(df)
        n_after = len(df)
        logger.info(f"  {n_after:,} rows (DB, voted from {n_before:,} rows "
                    f"across {n_sources_before} sources), "
                    f"{df.item_id.nunique():,} items")
        return df

    def _resolve_backfilled_slugs(self) -> set:
        """The set of STEAMCOMMUNITY-backfilled item slugs to train on.

        Split out of ``fetch_price_history`` because the voted-frame cache key
        has to include it — two runs with different backfill sets produce
        different frames from an identical archive.
        """
        try:
            slug_rows = self.db.execute(text("""
                SELECT item_id FROM items WHERE is_backfilled = 1
            """)).fetchall()
            slugs = {r[0] for r in slug_rows}
            logger.info(f"  Backfilled items filter: {len(slugs)} items from DB")
            return slugs
        except Exception as e:
            logger.warning(f"  Could not fetch backfilled items from DB, using all: {e}")
            import duckdb
            with duckdb.connect() as con:
                return {
                    r[0] for r in con.sql("""
                        SELECT DISTINCT item_slug
                        FROM read_parquet(?)
                    """, params=[str(self.archive_dir / "prices-*.parquet")]).fetchall()
                }

    def _fetch_voted_price_history(self, days_back: int,
                                   backfilled_only: bool,
                                   backfilled_slugs: set = None) -> pd.DataFrame:
        """Read the Parquet archive and collapse it to one consensus price per
        item per day. The expensive half of ``fetch_price_history``."""
        archive_dir = self.archive_dir
        import duckdb
        cutoff = (self._now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        con = duckdb.connect()
        try:
            # One reader for the whole archive; it handles the pre-2026 files
            # lacking `source` and the TIMESTAMP/TIMESTAMP_NS split. See
            # db/archive.py for why a plain glob read is not enough.
            from db.archive import prices_relation
            relation = prices_relation(
                con, archive_dir,
                columns=["item_slug", "day", "mean_price", "volume", "source"])

            # Filter to backfilled slugs via a temp table JOIN (handles special chars safely)
            if backfilled_slugs is not None:
                con.sql("CREATE TEMP TABLE _backfilled (slug VARCHAR)")
                con.executemany("INSERT INTO _backfilled VALUES (?)",
                                [(s,) for s in backfilled_slugs])
                logger.info(f"  Filtering to {len(backfilled_slugs)} backfilled items via temp table")

            slug_join = "JOIN _backfilled b ON sub.item_slug = b.slug" if backfilled_slugs is not None else ""

            # Interpolated rather than bound: a bound NULL has no type for
            # DuckDB to infer, and `replay_anchor()` has already parsed this
            # through `date.fromisoformat`, so it cannot carry SQL.
            _anchor = self.replay_anchor()
            _upper_bound = (
                f"AND day <= '{_anchor.strftime('%Y-%m-%d')}'" if _anchor else "")
            if _anchor:
                logger.warning(
                    f"  REPLAY_ANCHOR={_anchor}: archive read bounded at that "
                    f"date. This is a backdated replay, not a live forecast.")
            df = con.sql(f"""
                SELECT item_slug, day, mean_price AS price, volume, source
                FROM {relation} sub
                {slug_join}
                WHERE day >= ?
                  {_upper_bound}
                  AND (source IS NULL OR source NOT LIKE 'historical_fallback:%')
                  AND {phase_collapsed_sql_filter("sub.item_slug")}
                  AND {phantom_slug_sql_filter("sub.item_slug")}
                ORDER BY item_slug, day, source
            """, params=[cutoff]).fetchdf()
            logger.info(f"  DuckDB query returned {len(df):,} rows "
                        f"(phase-collapsed names excluded: "
                        f"{', '.join(PHASE_COLLAPSED_SLUG_PATTERNS)}; "
                        f"phantom duplicate keys excluded)")
            df = df.rename(columns={"item_slug": "item_id", "day": "timestamp"})
            logger.info(f"  DataFrame created, converting types...")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date
            # Some Parquet years store mean_price/volume as VARCHAR; the
            # glob union then coerces the whole column to string. Force
            # numeric so multi-source voting (np.median) works.
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            df = df.dropna(subset=["price"])
            logger.info(f"  After parsing: {len(df):,} rows, {df.item_id.nunique():,} items")
            n_before = len(df)
            n_sources_before = df["source"].nunique() if "source" in df.columns else 1
            logger.info(f"  Applying multi-source voting ({n_sources_before} sources)...")
            df = self._apply_multi_source_voting(df)
            n_after = len(df)
            logger.info(f"  {n_after:,} rows (Parquet, voted from {n_before:,} rows "
                        f"across {n_sources_before} sources), "
                        f"{df.item_id.nunique():,} items")
            if backfilled_only:
                logger.info(f"  Filtered to STEAMCOMMUNITY-backfilled items only")
            return df
        finally:
            con.close()

    @staticmethod
    def _apply_multi_source_voting(df: pd.DataFrame) -> pd.DataFrame:
        """Apply multi-source outlier voting to get a single consensus price per item per day.

        For each (item_id, date) with >= 3 sources:
        - Compute median and std of source prices
        - Reject sources > 2 std from the median consensus
        - Use median of remaining sources
        For < 3 sources: use simple median.

        This is a data quality improvement — not a new feature dimension.
        Reducing noise in the target price improves ALL downstream features
        (lags, returns, rolling stats, Bollinger, RSI, MACD, volume features).
        """
        if "source" not in df.columns:
            # No source information at all here -- both production callers
            # (_fetch_price_history_from_db, _fetch_voted_price_history) select
            # `source` explicitly. This is a legacy/test path where the count
            # is genuinely unknowable, so it does not carry n_ask_sources.
            unique_rows = df.groupby(["item_id", "date"]).size()
            already_single = not (unique_rows > 1).any()
            if already_single:
                return df
            return df.groupby(["item_id", "date"], as_index=False).agg(
                price=("price", "mean"),
                volume=("volume", "sum"),
            )

        # Drop bids and Steam's trailing-window means before anything else
        # reads the group, so neither counts toward the median or the
        # >=3-source gate that enables the outlier mask. `isin` is False for
        # NaN, which is what keeps the pre-2026 `source IS NULL` series — 13
        # years of the archive — voting. `frozenset` membership only, never a
        # prefix match: `aggregator_steam_17mafo` is a distinct source (the
        # only ask for 2026-04-16 -> 07-10) and must keep voting.
        excluded = BID_SOURCES | TRAILING_WINDOW_SOURCES
        df = df[~df["source"].isin(excluded)]
        if df.empty:
            # Every input row was a bid or a trailing-window mean. 2,338
            # item-days across 217 items have no ask at all from the bid drop
            # alone (a further 650 from the trailing-window drop, 2026 >=$1);
            # they drop out rather than falling back to a bid or a stale
            # window, because a series whose basis alternates fabricates the
            # wedge, or the time-basis change, as a return.
            return pd.DataFrame(
                columns=["item_id", "date", "price", "volume", "n_ask_sources"]
            ).astype({"n_ask_sources": "int64"})

        # Speedup: split into single-source (≤1 row per item/date) and multi-source groups.
        # Single-source rows use fast groupby agg; multi-source uses the vote function.
        # This avoids calling a Python function millions of times.
        item_date_counts = df.groupby(["item_id", "date"], as_index=False).size()
        multi_groups = item_date_counts[item_date_counts["size"] > 1]
        if multi_groups.empty:
            # Every item-day has exactly one row left after the bid drop, i.e.
            # exactly one ask source voted, by construction.
            out = df.drop(columns=["source"], errors="ignore")
            out["n_ask_sources"] = 1
            return out

        multi_keys = multi_groups[["item_id", "date"]].drop_duplicates()
        is_multi = df.set_index(["item_id", "date"]).index.isin(
            multi_keys.set_index(["item_id", "date"]).index
        )

        single_df = df[~is_multi].copy()
        multi_df = df[is_multi].copy()

        def vote(group):
            prices = group["price"].values
            n_sources = len(prices)
            # nunique of the source column, not the row count: a group can
            # hold duplicate rows from the same source. NaN sources collapse
            # to one bucket via fillna so a NULL-source group still counts 1.
            n_ask_sources = int(group["source"].fillna("__null__").nunique())

            if n_sources >= 3:
                consensus = np.median(prices)
            else:
                consensus = np.median(prices)
                return pd.Series({
                    "price": consensus,
                    "volume": group["volume"].sum() if "volume" in group.columns else 0,
                    "n_ask_sources": n_ask_sources,
                })

            median = consensus
            std = np.std(prices, ddof=0)
            if std > 0:
                mask = np.abs(prices - median) <= 2.0 * std
                kept = mask.sum()
                if kept >= 1:
                    consensus = np.median(prices[mask])
                else:
                    consensus = median
            else:
                consensus = median

            return pd.Series({
                "price": consensus,
                "volume": group["volume"].sum() if "volume" in group.columns else 0,
                "n_ask_sources": n_ask_sources,
            })

        # Fast path: single-source rows -- exactly one row per item-day, so
        # exactly one ask source voted, by construction.
        if len(single_df) > 0:
            result_single = single_df.groupby(["item_id", "date"], as_index=False).agg(
                price=("price", "mean"),
                volume=("volume", "sum"),
            )
            result_single["n_ask_sources"] = 1
        else:
            result_single = pd.DataFrame(columns=["item_id", "date", "price", "volume", "n_ask_sources"])

        # Slow path: multi-source rows (small subset, typically <2% of groups)
        if len(multi_df) > 0:
            result_multi = multi_df.groupby(["item_id", "date"], as_index=False).apply(
                vote
            ).reset_index(drop=True)
        else:
            result_multi = pd.DataFrame(columns=["item_id", "date", "price", "volume", "n_ask_sources"])

        result = pd.concat([result_single, result_multi], ignore_index=True)
        # Concatenating single- and multi-source frames can upcast to float
        # when one side is empty; n_ask_sources must stay a true integer.
        result["n_ask_sources"] = result["n_ask_sources"].astype("int64")
        return result

    def fetch_events(self) -> pd.DataFrame:
        if hasattr(self, "_events_cache") and self._events_cache is not None:
            return self._events_cache
        try:
            rows = self.db.execute(text("""
                SELECT id, type, timestamp, description
                FROM events
                ORDER BY timestamp
            """)).fetchall()
        except Exception:
            logger.warning("  DB connection lost, reconnecting...")
            from database import SessionLocal
            self.db = SessionLocal()
            rows = self.db.execute(text("""
                SELECT id, type, timestamp, description
                FROM events
                ORDER BY timestamp
            """)).fetchall()
        df = pd.DataFrame(rows, columns=["id", "type", "timestamp", "description"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        self._events_cache = df
        logger.info(f"  events: {len(df)}")
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering price features...")
        df = df.sort_values(["item_id", "date"]).copy()

        # Lag prices — DATE-based, not row-based. A row-based shift(lag) spans
        # any gap in an item's daily series (e.g. a multi-week ingestion hole),
        # turning "return_14d" into a multi-month return and blowing the feature
        # out of distribution at the serving edge. This matches prepare_targets,
        # which already uses a date-based lookup for the same reason. Longer lags
        # (90/120/180) feed the trend features.
        #
        # The lookup is as-of rather than exact: take the most recent price at or
        # before `date - lag`, but only within LAG_TOLERANCE_DAYS. An exact-date
        # lookup was correct about multi-week holes and wrong about the 1-2 day
        # ones the aggregator produces daily — those NaN'd the freshest features
        # for the entire served universe at once. Anything beyond the tolerance
        # still yields NaN (later imputed to the median → neutral).
        LAGS = [1, 3, 7, 14, 30, 60, 90, 120, 180]
        df["_date_dt"] = pd.to_datetime(df["date"])
        df = df.reset_index(drop=True)
        # One row per (item, date) keeps the lookup unambiguous even if the
        # caller passed un-resampled (intraday-duplicate) rows. merge_asof needs
        # both sides sorted on the join key.
        observed = (
            df[["item_id", "_date_dt", "price"]]
            .drop_duplicates(subset=["item_id", "_date_dt"])
            .sort_values("_date_dt")
        )
        tolerance = pd.Timedelta(self.LAG_TOLERANCE_DAYS, unit="D")
        for lag in LAGS:
            col = f"price_lag_{lag}d"
            targets = pd.DataFrame({
                "_row": df.index,
                "item_id": df["item_id"],
                "_target": df["_date_dt"] - pd.Timedelta(lag, unit="D"),
            }).sort_values("_target")
            resolved = pd.merge_asof(
                targets,
                observed.rename(columns={"price": col}),
                left_on="_target",
                right_on="_date_dt",
                by="item_id",
                direction="backward",
                tolerance=tolerance,
            )
            df[col] = resolved.set_index("_row")[col].reindex(df.index)

        # Returns (winsorized at ±500% against residual data artifacts)
        for lag in LAGS:
            col = f"price_lag_{lag}d"
            df[f"return_{lag}d"] = (
                (df["price"] - df[col]) / df[col].replace(0, np.nan) * 100
            ).clip(-500, 500)

        df = df.drop(columns=["_date_dt"])
        # Re-bind groupby: the merges above returned new frames, so any earlier
        # groupby handle is stale. All remaining row-based rolling ops use this.
        grouped = df.groupby("item_id")

        # Rolling statistics (min_periods=1 so items with short history get partial estimates)
        for window in [7, 14, 20, 30, 60]:
            roll = grouped["price"].rolling(window, min_periods=1)
            roll_agg = roll.agg(["mean", "std", "min", "max"])
            df[f"price_mean_{window}d"] = roll_agg["mean"].values
            df[f"price_std_{window}d"] = roll_agg["std"].values
            df[f"price_min_{window}d"] = roll_agg["min"].values
            df[f"price_max_{window}d"] = roll_agg["max"].values

        # Coefficient of variation — the return-space form of the rolling std.
        # The raw price_std_{w}d are DOLLARS while target_return_{h}d is a
        # PERCENT, which made them item-identity proxies rather than volatility
        # signals (they carried 55.6-86.6% of gain on the 2026-08-06 artifact).
        # Dividing by price is exactly what conformal.sigma_from_columns already
        # does for price_std_60d; price_cv_60d and that sigma are the same
        # quantity, pinned by test_cv_60d_matches_what_conformal_computes_for_sigma.
        # The dollar columns stay computed — the band, the market aggregates and
        # the z-score/Bollinger derivations all read them — but they are shelved
        # out of the feature set. See _DOLLAR_SCALE_FEATURES.
        _px = df["price"].replace(0, np.nan)
        for window in [7, 14, 20, 30, 60]:
            df[f"price_cv_{window}d"] = df[f"price_std_{window}d"] / _px

        # Z-score vs 30d rolling
        mean_30 = df["price_mean_30d"]
        std_30 = df["price_std_30d"].replace(0, np.nan)
        df["price_zscore_30d"] = (df["price"] - mean_30) / std_30

        # Long-term volatility regime: ratio of 60d to 30d volatility
        # Values > 1 mean volatility is rising, < 1 mean it's falling.
        vol_30 = df["price_std_30d"].replace(0, np.nan)
        vol_60 = df["price_std_60d"].replace(0, np.nan)
        df["vol_regime_60_30"] = vol_60 / vol_30

        # Trend divergence: ratio of 30d return to 60d return.
        # Shows whether short-term momentum agrees with long-term trend.
        df["trend_divergence_30_60"] = (df.get("return_30d", pd.Series(np.nan, index=df.index)) /
                                        df.get("return_60d", pd.Series(np.nan, index=df.index)).replace(0, np.nan))

        # Price acceleration (2nd derivative)
        df["price_accel_7d"] = df["return_7d"] - df["return_7d"].groupby(df["item_id"]).shift(7)

        # Log price (scale-invariant)
        df["price_log"] = np.log(df["price"].clip(lower=0.01))

        # Price tier (categorical price level buckets)
        price = df["price"]
        df["price_tier"] = 0
        df.loc[price >= 1, "price_tier"] = 1
        df.loc[price >= 5, "price_tier"] = 2
        df.loc[price >= 20, "price_tier"] = 3
        df.loc[price >= 100, "price_tier"] = 4
        df["price_tier"] = df["price_tier"].astype(int)

        # Log returns (stationary, scale-invariant)
        df["log_return_1d"] = np.log(df["price"] / df["price_lag_1d"].replace(0, np.nan))
        df["log_return_7d"] = np.log(df["price"] / df["price_lag_7d"].replace(0, np.nan))

        # Trailing daily-return volatility for vol-scaled direction labels
        # (2026-07-27). Labeling only — excluded from model features (see
        # `exclude` below). log_return_1d is a raw fractional log return
        # (~0.01 for a 1% move); scale by 100 so this column is in PERCENT
        # units, matching target_return_{h}d and the flat-band threshold
        # (k * sigma * sqrt(h)) it feeds. Trailing only (no centering) —
        # no future information leaks into the window.
        df[DIRECTION_LABEL_VOL_COL] = (
            df.groupby("item_id")["log_return_1d"]
              .transform(lambda s: (s * 100.0).rolling(DIRECTION_VOL_WINDOW, min_periods=5).std())
        )

        # Price autocorrelation proxy (direction agreement between lag-1 and lag-7 returns)
        df["autocorr_1d"] = df["return_1d"] * df["return_1d"].groupby(df["item_id"]).shift(1)
        df["autocorr_7d"] = df["return_7d"] * df["return_7d"].groupby(df["item_id"]).shift(7)

        # =====================================================================
        # Bollinger Bands (20-day)
        # =====================================================================
        bb_mid = df["price_mean_20d"]
        bb_std = df["price_std_20d"].replace(0, np.nan)
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_pct_b"] = ((df["price"] - df["bb_lower"]) / bb_range).clip(-2, 2)
        df["bb_width"] = (bb_range / bb_mid.replace(0, np.nan))

        # =====================================================================
        # RSI (14-day)
        # =====================================================================
        price_change = grouped["price"].diff()
        gain = price_change.clip(lower=0)
        loss = (-price_change).clip(lower=0)
        avg_gain = gain.groupby(df["item_id"]).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
        avg_loss = loss.groupby(df["item_id"]).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].clip(0, 100)

        # =====================================================================
        # MACD (vectorized — no lambda transforms, matches original adjust=True)
        # =====================================================================
        df = df.sort_values(["item_id", "date"])
        # Compute EMAs per group using EWMA on sorted data (adjust=True matches
        # the original lambda-based ewm(adjust=True).mean() behavior).
        ewm12 = df.groupby("item_id")["price"].ewm(span=12, min_periods=12, adjust=True).mean()
        ewm26 = df.groupby("item_id")["price"].ewm(span=26, min_periods=26, adjust=True).mean()
        df["macd_line"] = ewm12.reset_index(level=0, drop=True) - ewm26.reset_index(level=0, drop=True)
        df["macd_signal"] = (
            df.groupby("item_id")["macd_line"]
            .ewm(span=9, min_periods=9, adjust=True)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

        # MACD is a difference of price EMAs, so it is in DOLLARS and scales
        # with the item. Normalising by price turns it into the oscillator it is
        # meant to be. The raw columns stay computed for macd_hist_slope_7d and
        # the macd_missing flag; only these relative forms reach a booster.
        _macd_px = df["price"].replace(0, np.nan)
        df["macd_line_rel"] = df["macd_line"] / _macd_px
        df["macd_histogram_rel"] = df["macd_histogram"] / _macd_px

        # =====================================================================
        # Volatility asymmetry (downside vs upside semi-deviation) — pure price.
        # A symmetric std collapses panic (sharp downside) and froth (volatile
        # upside) into one number; splitting them exposes the difference.
        # =====================================================================
        ret = df["return_1d"]
        ret_neg = ret.where(ret < 0)
        ret_pos = ret.where(ret > 0)
        df["vol_semidev_down_30d"] = (
            ret_neg.groupby(df["item_id"]).rolling(30, min_periods=5).std()
            .reset_index(level=0, drop=True)
        )
        df["vol_semidev_up_30d"] = (
            ret_pos.groupby(df["item_id"]).rolling(30, min_periods=5).std()
            .reset_index(level=0, drop=True)
        )
        # Ratio > 1 => upside more volatile (froth); < 1 => downside sharper
        # (panic). Clipped: near-zero downside vol otherwise blows the ratio up.
        _semidev_down = df["vol_semidev_down_30d"].replace(0, np.nan)
        df["vol_skew_30d"] = (df["vol_semidev_up_30d"] / _semidev_down).clip(0, 5)

        # =====================================================================
        # Oscillator divergence — momentum of RSI/MACD, and price/RSI
        # disagreement. The frame is already item/date-sorted (MACD block
        # re-sorted it), so a groupby shift(7) is a clean 7-day lookback.
        # =====================================================================
        df["rsi_divergence_7d"] = (
            df["rsi_14"] - df.groupby("item_id")["rsi_14"].shift(7)
        )
        # Positive => price up while RSI down (bearish divergence). return_7d is
        # winsorized to +/-500; clip to +/-50 keeps typical moves on the same
        # scale as the RSI term (RSI change is bounded to +/-100).
        df["rsi_price_divergence_7d"] = (
            df["return_7d"].clip(-50, 50) / 50.0 - df["rsi_divergence_7d"] / 100.0
        )
        df["macd_hist_slope_7d"] = (
            df["macd_histogram"] - df.groupby("item_id")["macd_histogram"].shift(7)
        )

        # =====================================================================
        # Support / Resistance distances
        # =====================================================================
        df["distance_to_support"] = ((df["price"] - df["price_min_30d"]).replace(0, np.nan) /
                                      df["price_min_30d"].replace(0, np.nan) * 100)
        df["distance_to_resistance"] = ((df["price_max_30d"] - df["price"]).replace(0, np.nan) /
                                         df["price"].replace(0, np.nan) * 100)
        df["high_low_range_30d"] = ((df["price_max_30d"] - df["price_min_30d"]).replace(0, np.nan) /
                                     df["price_min_30d"].replace(0, np.nan) * 100)

        # =====================================================================
        # Longer-horizon trend features (added 2026-07-24)
        # Momentum out to ~30d beat the model at the 30d horizon, and the
        # existing lookback topped out at 60d. Give the model the longer-trend
        # signal it was losing to. All price-derived, so they stay inside the
        # price_technicals allowlist group. (return_90/120/180d are computed
        # date-based with the other lags above.)
        # =====================================================================
        # Distance from long-run moving averages (position within the long trend).
        for window in [100, 200]:
            ma = grouped["price"].rolling(window, min_periods=30).mean()
            df[f"price_mean_{window}d"] = ma.values
            ma_col = df[f"price_mean_{window}d"].replace(0, np.nan)
            df[f"price_dist_ma{window}"] = (df["price"] - ma_col) / ma_col * 100

        # Trend consistency: fraction of up-days over the last 30 sessions.
        # High values = a persistent uptrend (what momentum exploits); ~0.5 = chop.
        up_day = (df["return_1d"] > 0).astype(float)
        df["trend_up_fraction_30d"] = (
            up_day.groupby(df["item_id"]).rolling(30, min_periods=5).mean()
            .reset_index(level=0, drop=True)
        )

        # =====================================================================
        # Volume features
        # =====================================================================
        df = self._compute_volume_features(df, grouped)

        # Boolean indicators for features with frequent missingness
        df["rsi_missing"] = df["rsi_14"].isna().astype(int)
        df["macd_missing"] = df["macd_line"].isna().astype(int)

        return df

    @staticmethod
    def _compute_volume_features(df: pd.DataFrame, grouped=None) -> pd.DataFrame:
        """Engineer the volume-derived columns.

        ALL of these are in SHELVED_FEATURES, so none of them reaches training —
        see the comment there for why (the archive's volume has been identically
        0 since 2026-05, stored as 0 rather than NULL). They are still computed
        because two features that are NOT shelved read them:
        ``supply_to_volume_ratio`` reads ``volume_mean_30d`` and
        ``item_volume_vs_market_30d`` reads the raw ``volume`` column.

        Note ``has_volume`` is deliberately left testing ``notna()``. Making it
        treat all-zero as absent would only swap a served 0 for a median-filled
        ~98, which is no more truthful; the shelving is the fix.
        """
        if grouped is None:
            grouped = df.groupby("item_id")

        has_volume = "volume" in df.columns and df["volume"].notna().any()
        df["volume_missing"] = (1 if not has_volume else
                                df["volume"].isna().astype(int))

        if has_volume:
            df["volume_lag_1d"] = grouped["volume"].shift(1)
            df["volume_lag_7d"] = grouped["volume"].shift(7)
            df["volume_mean_7d"] = grouped["volume"].rolling(
                7, min_periods=1
            ).mean().reset_index(level=0, drop=True)
            df["volume_mean_30d"] = grouped["volume"].rolling(
                30, min_periods=1
            ).mean().reset_index(level=0, drop=True)
            df["volume_std_30d"] = grouped["volume"].rolling(
                30, min_periods=1
            ).std().reset_index(level=0, drop=True)
            df["volume_mean_60d"] = grouped["volume"].rolling(
                60, min_periods=1
            ).mean().reset_index(level=0, drop=True)
            df["volume_std_60d"] = grouped["volume"].rolling(
                60, min_periods=1
            ).std().reset_index(level=0, drop=True)

            # Log-ratio volume change (avoids division-by-zero issues)
            vol_lag_1 = df["volume_lag_1d"].replace(0, np.nan)
            vol_lag_7 = df["volume_lag_7d"].replace(0, np.nan)
            df["volume_log_change_1d"] = np.log(df["volume"] / vol_lag_1)
            df["volume_log_change_7d"] = np.log(df["volume"] / vol_lag_7)

            # Volume z-score vs 30d
            vol_std_30 = df["volume_std_30d"].replace(0, np.nan)
            df["volume_zscore_30d"] = ((df["volume"] - df["volume_mean_30d"]) / vol_std_30)

            # Volume-price confirmation.
            #
            # fillna(False) before astype(int) because `volume` is nullable
            # from 2026-08-08 on: that is the first day the upstream feed
            # returned no volume (33,613 non-null of 361,453 rows; every
            # earlier day is 100% populated). DuckDB hands a column with NULLs
            # to pandas as a nullable dtype, so `> 0` yields BooleanDtype
            # carrying pd.NA and astype(int) raises "cannot convert NA to
            # integer" -- which is exactly how the first mode=full run in
            # weeks died, in CI, on data this machine's archive did not yet
            # have.
            #
            # False is the pre-existing semantics, not a new choice: on the
            # numpy path `NaN > 0` was already False, i.e. "no confirmation".
            # This is a no-op on every day before 2026-08-08.
            df["volume_price_conf_7d"] = (
                df["return_7d"]
                * (df["volume_log_change_7d"] > 0).fillna(False).astype(int))
            df["volume_price_conf_1d"] = (
                df["return_1d"]
                * (df["volume_log_change_1d"] > 0).fillna(False).astype(int))
        else:
            for col in ["volume_lag_1d", "volume_lag_7d", "volume_mean_7d",
                        "volume_mean_30d", "volume_std_30d",
                        "volume_mean_60d", "volume_std_60d",
                        "volume_log_change_1d", "volume_log_change_7d",
                        "volume_zscore_30d", "volume_price_conf_7d",
                        "volume_price_conf_1d"]:
                df[col] = np.nan
        return df

    def _fetch_item_metadata(self) -> pd.DataFrame:
        if hasattr(self, "_item_meta_cache") and self._item_meta_cache is not None:
            return self._item_meta_cache
        try:
            rows = self.db.execute(text("""
                SELECT item_id, name, type FROM items
            """)).fetchall()
            df = pd.DataFrame(rows, columns=["item_id", "name", "type"])
        except Exception:
            self._item_meta_cache = pd.DataFrame(columns=["item_id", "name", "type"])
            return self._item_meta_cache
        self._item_meta_cache = df
        logger.info(f"  item metadata: {len(df)} items loaded")
        return df

    @staticmethod
    def bymykel_metadata_enabled() -> bool:
        """Whether the ByMykel bundle joins into the feature frame.

        Off by default. See BYMYKEL_META_FEATURES for why.
        """
        return os.environ.get("BYMYKEL_METADATA") == "1"

    @staticmethod
    def tier_lead_enabled() -> bool:
        """Whether `tier_lead_return_1d` is engineered and allowlisted.

        Off by default, so this is an instrument to A/B and not a shipped
        feature. The prior is the one positive cross-sectional structure measured
        in this archive: expensive tiers lead cheap tiers by a day at lag-1 corr
        +0.213 (z = 9.1), Granger incremental R^2 9.0%, stable in 4 of 5 years,
        and it survives removing the market factor (0.122, R^2 4.5%) -- which is
        what distinguishes it from every refuted feature here, all of which died
        with the common factor. See
        docs/research/2026-08-07-cs2-forecasting-research.md.

        Set TIER_LEAD_FEATURE=1. **Read the staleness caveat before adopting:**
        cheap skins have the highest zero-change rate (1.33% vs 0.16%), so a
        partially-updating cheap index could fake this signature. The decisive
        test is dropping high-`stale_run_days` item-days and re-measuring.
        """
        return os.environ.get("TIER_LEAD_FEATURE") == "1"

    @staticmethod
    def cross_sectional_rank_enabled() -> bool:
        """Whether features are rank-transformed within each forecast date.

        Off by default. `2 * (rank(pct=True) - 0.5)` per date, i.e. Gu, Kelly &
        Xiu's footnote-29 transform. The features here are scale-free *per item*
        (pinned by tests/test_scale_free_features.py), which is not the same as
        cross-sectionally normalised: every column stays loaded on the common
        market factor on every date, which is the diagnosed mechanism behind both
        "DA is dominated by the forecast date" and the market-relative label
        refutation.

        This is NOT that refuted experiment.
        docs/changelog/2026-08-06-market-relative-labels-refuted.md changed the
        **label** and left a pointwise loss fighting a noisy residual; this
        changes the **features** and leaves the label alone.

        Set CROSS_SECTIONAL_RANK=1.
        """
        return os.environ.get("CROSS_SECTIONAL_RANK") == "1"

    @staticmethod
    def naive_init_score_enabled() -> bool:
        """Whether the quantile models are boosted from `-return_1d`.

        Off by default. With it on, `-return_1d` is passed as `init_score` on
        every quantile `lgb.Dataset` and added back to the model's output at
        predict time, so the booster fits the RESIDUAL to the naive predictor
        instead of competing with it.

        The measured gap is rank IC, not DA: ranking by minus yesterday's return
        beats the model at all four horizons (-0.0159 / -0.0371 / -0.0433 /
        -0.0091, `docs/changelog/2026-08-10-post-revote-retrain.md`), it uses no
        hindsight, and it survived the 2026-08-09 re-vote. That makes it the one
        legitimate baseline gap in this repo -- unlike `constant_call_accuracy`,
        which is hindsight-selected per fold
        (`docs/changelog/2026-08-10-constant-call-is-hindsight-picked.md`).

        Read `rank_ic_edge` per horizon; the bar is `>= 0`.

        Two cautions. `tuned_params` were selected against the un-offset target,
        so the first read should reuse cached HP and any positive should be
        confirmed with `FORCE_HP_SEARCH=1` before its size is believed. And the
        floor is empirical, not algebraic: a boosted model can still fit its way
        below the offset it started from, so a negative read is a real outcome
        rather than a bug.

        Set NAIVE_INIT_SCORE=1. Tracked as N1 in
        `docs/research/2026-08-10-next-steps.md`.
        """
        return os.environ.get("NAIVE_INIT_SCORE") == "1"

    def _tier_lead_served(self) -> bool:
        """Whether the loaded artifact was trained with the tier-lead feature.

        The artifact wins over the environment on the predict path, and the
        environment is only consulted when no artifact has been loaded. A model
        trained without the column must not be served a frame that has it (the
        column would be dropped by feature alignment, harmlessly) and a model
        trained *with* it must never be served a frame without it -- alignment
        would add the column as NaN and fill it with the persisted median,
        feeding the booster a constant where it expects a signal.

        Training deliberately reads `tier_lead_enabled()` instead: a warm retrain
        restores the previous artifact's meta, and inheriting its flag would make
        the arm untestable.
        """
        if self._artifact_tier_lead is not None:
            return self._artifact_tier_lead
        return self.tier_lead_enabled()

    def _cross_sectional_rank_served(self) -> bool:
        """Whether the loaded artifact was trained on rank-transformed features.

        Same artifact-over-environment rule as _tier_lead_served, and it matters
        more here: serving raw feature values to a booster fitted on within-date
        ranks is not a degradation, it is a different input space.
        """
        if self._artifact_xs_rank is not None:
            return self._artifact_xs_rank
        return self.cross_sectional_rank_enabled()

    def _naive_init_score_served(self) -> bool:
        """Whether the loaded artifact was fitted on top of `-return_1d`.

        Same artifact-over-environment rule as _tier_lead_served, and the
        divergence it prevents is the worst of the three: a booster fitted with
        the offset emits a RESIDUAL, so serving it without adding the offset back
        publishes a residual as a price forecast, silently and with no shape
        change to give it away.
        """
        if self._artifact_naive_init is not None:
            return self._artifact_naive_init
        return self.naive_init_score_enabled()

    @classmethod
    def _minus_return_1d(cls, frame) -> np.ndarray:
        """`-return_1d` for each row of `frame`, in PERCENT, positionally.

        Percent because `return_{lag}d` and `target_return_{h}d` are both
        `(a - b) / b * 100` -- the offset needs no rescaling to sit in the
        target's units, and an offset in fractions would be 100x too small and
        would read as a null result rather than as a units bug.

        Non-finite values become a zero offset ("no baseline view for this row"),
        never NaN: a NaN init_score propagates into every prediction LightGBM
        makes from that row.

        Raises when the column is absent rather than returning None. N1 exists to
        put a floor under the model, and a silently absent offset removes the
        floor while leaving the metric that reads it looking normal.
        """
        col = cls.NAIVE_OFFSET_COL
        if col not in getattr(frame, "columns", ()):
            raise RuntimeError(
                f"NAIVE_INIT_SCORE is on but the frame has no `{col}` column, "
                f"so the naive offset cannot be built. Every allowlisted feature "
                f"set contains it; a frame without it is wrong, and training or "
                f"serving without the offset would quietly void the arm."
            )
        v = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        return -np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def _naive_offset(self, frame) -> Optional[np.ndarray]:
        """The training-side offset: reads the environment, per N1's gate.

        None when the instrument is off, which is what every call site checks
        before touching `init_score`. Training reads the environment rather than
        the restored artifact's flag for the reason _tier_lead_served documents:
        a warm retrain restores the previous meta, and inheriting its flag would
        make the arm untestable.
        """
        if not self.naive_init_score_enabled():
            return None
        return self._minus_return_1d(frame)

    def _naive_offset_served(self, frame) -> Optional[np.ndarray]:
        """The predict-side offset: follows the artifact, per _naive_init_score_served."""
        if not self._naive_init_score_served():
            return None
        return self._minus_return_1d(frame)

    def _fetch_bymykel_metadata(self) -> pd.DataFrame:
        """Load the ByMykel item-metadata bundle, or an empty frame if absent.

        Absent is not an error. `price-archive/` is a gitignored local directory
        (the durable archive is the separate cs2-oracle-data repo), so a checkout
        that has never run scripts/ingest_bymykel_metadata.py legitimately has no
        such file. The feature columns are then simply not added, and the
        allowlist finds nothing to admit.
        """
        if getattr(self, "_bymykel_meta_cache", None) is not None:
            return self._bymykel_meta_cache

        # Overridable so scripts/paired_retrain_bymykel.py can point a placebo
        # arm at a permuted copy without a second code path. Unset in production.
        override = os.environ.get("BYMYKEL_METADATA_PATH")
        path = (Path(override) if override else
                Path(__file__).parent.parent.parent / "price-archive"
                / "item-metadata-bymykel.parquet")
        df = pd.DataFrame()
        if path.exists():
            try:
                df = pd.read_parquet(path).rename(columns={"item_slug": "item_id"})
                logger.info(f"  ByMykel metadata: {len(df):,} items loaded")
            except Exception as e:
                logger.warning(f"  Failed to load ByMykel metadata: {e}")
                df = pd.DataFrame()
        else:
            logger.warning(
                f"  BYMYKEL_METADATA=1 but {path.name} is absent — the nine "
                f"metadata features will not be added. Run "
                f"scripts/ingest_bymykel_metadata.py.")
        self._bymykel_meta_cache = df
        return df

    def _add_bymykel_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join the ByMykel bundle and derive per-row catalogue age.

        Age is computed HERE rather than stored, because it is (observation date
        - first sale date) and so belongs to a row, not an item. A stored "days
        since sale as of today" column would be the calendar frozen into a
        per-item constant.
        """
        if not self.bymykel_metadata_enabled():
            return df

        meta = self._fetch_bymykel_metadata()
        if meta.empty or "item_id" not in meta.columns:
            return df

        meta = meta.drop_duplicates(subset=["item_id"]).set_index("item_id")
        aligned = meta.reindex(df["item_id"].to_numpy())

        if "item_age_first_sale_date" in aligned.columns:
            obs = pd.to_datetime(pd.Series(df["date"].to_numpy()))
            first_sale = pd.to_datetime(
                pd.Series(aligned["item_age_first_sale_date"].to_numpy()),
                errors="coerce")
            age = (obs - first_sale).dt.days.to_numpy(dtype=float)
            # A negative age means the item traded before the catalogue says it
            # first went on sale: the metadata is wrong for that item, not that
            # the item is "very new". Null it rather than clip to 0, which would
            # fabricate a real-looking value at a meaningful boundary.
            age[age < 0] = np.nan
            df["item_age_meta_days"] = age

        for col in self.BYMYKEL_META_FEATURES:
            if col == "item_age_meta_days" or col not in aligned.columns:
                continue
            # float64, not the source Int64. _select_feature_cols keeps a column
            # only if its dtype is in (float64, float32, int64, int, float), and
            # pandas' nullable Int64 is none of those — leaving it would drop the
            # column silently, which reads exactly like a null result.
            df[col] = pd.to_numeric(
                aligned[col].to_numpy(), errors="coerce").astype(np.float64)

        present = sorted(c for c in self.BYMYKEL_META_FEATURES if c in df.columns)
        logger.info(f"  ByMykel metadata features added: {len(present)} "
                    f"({', '.join(present)})")
        return df

    def _fetch_supply_metadata(self) -> pd.DataFrame:
        """Load supply-side metadata (rarity, weapon_type) from Parquet or DB.

        Tries price-archive/item-metadata.parquet first, then falls back
        to the items table in the database.
        """
        if hasattr(self, "_supply_meta_cache") and self._supply_meta_cache is not None:
            return self._supply_meta_cache

        archive_dir = Path(__file__).parent.parent.parent / "price-archive"
        meta_path = archive_dir / "item-metadata.parquet"

        if meta_path.exists():
            try:
                df = pd.read_parquet(meta_path)
                df = df.rename(columns={"item_slug": "item_id"})
                logger.info(f"  supply metadata: {len(df)} items loaded from Parquet")
                self._supply_meta_cache = df
                return df
            except Exception as e:
                logger.warning(f"  Failed to load supply metadata from Parquet: {e}")

        try:
            rows = self.db.execute(text("""
                SELECT item_id, rarity, rarity_rank, weapon_type FROM items
            """)).fetchall()
            df = pd.DataFrame(rows, columns=["item_id", "rarity", "rarity_rank", "weapon_type"])
            logger.info(f"  supply metadata: {len(df)} items loaded from DB")
        except Exception:
            logger.warning("  Could not fetch supply metadata from DB; using empty DataFrame")
            df = pd.DataFrame(columns=["item_id", "rarity", "rarity_rank", "weapon_type"])

        self._supply_meta_cache = df
        return df

    def _add_supply_side_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add supply-side features: rarity ordinal and one-hot dummies.

        Permutation test confirmed strong causal signal (+10-12pp across all
        horizons) from rarity features. Weapon-type one-hot and cross-sectional
        features were removed — they showed zero causal signal.
        """
        logger.info("Adding supply-side features...")

        meta = self._fetch_supply_metadata()
        if meta.empty:
            return df

        df = df.merge(meta, on="item_id", how="left")

        # Rarity ordinal (NaN → 0 for missing)
        df["rarity_ordinal"] = df["rarity_rank"].fillna(0).astype(int)

        # Rarity one-hot dummies
        rarity_cats = ["base", "consumer", "industrial", "milspec",
                       "restricted", "classified", "covert",
                       "high_grade", "remarkable", "exotic", "extraordinary"]
        for cat in rarity_cats:
            col = f"rarity_{cat}"
            df[col] = ((df["rarity"] == cat).astype(int))

        return df



    # ── Supply-depth features (sell_listings, skinport_quantity) ──────

    def _fetch_supply_snapshots(self) -> pd.DataFrame:
        """Load daily supply depth from `price-archive/supply-*.parquet`.

        Returns DataFrame with columns:
          item_id, date, sell_listings, skinport_quantity

        Reads the archive, NOT the `supply_snapshots` Postgres table this method
        used to query. That table holds one stale day (2026-07-15) from the Steam
        burst scraper, which cannot run from CI at all — Steam 429s runner IPs —
        and it was never published to the data repo, so it was invisible to
        everything that reads the archive. `collectors/supply_depth.py` replaced
        it: slug-keyed, published inside the aggregator's own commit, and joined
        on name like every other archive table.

        `sell_listings` is the max listing count across marketplaces for the day
        rather than a sum. The feeds overlap heavily (Spearman 0.65–0.82, and
        market.csgo.com is close to a superset of Waxpeer), so a sum would double
        count the same inventory, and — worse — would make the series lurch
        whenever a feed drops out. A max degrades gracefully: losing one
        marketplace lowers the level a little instead of creating a phantom
        supply crash, which matters because the predictive variant is the
        *change*, and a feed outage would otherwise read as a real move.
        """
        if hasattr(self, "_supply_snap_cache") and self._supply_snap_cache is not None:
            return self._supply_snap_cache

        empty = pd.DataFrame(columns=["item_id", "date", "sell_listings", "skinport_quantity"])
        archive_dir = Path(__file__).parent.parent.parent / "price-archive"
        paths = sorted(archive_dir.glob("supply-*.parquet"))
        if not paths:
            logger.info("  supply snapshots: no supply-*.parquet in the archive")
            self._supply_snap_cache = empty
            return empty

        try:
            frames = [pd.read_parquet(p, columns=["item_slug", "snapshot_day",
                                                  "source", "listing_count"])
                      for p in paths]
            raw = pd.concat(frames, ignore_index=True)
            raw = raw[raw["listing_count"].notna()]
            if raw.empty:
                logger.info("  supply snapshots: archive files hold no usable rows")
                self._supply_snap_cache = empty
                return empty

            df = (
                raw.groupby(["item_slug", "snapshot_day"], as_index=False)["listing_count"]
                .max()
                .rename(columns={"item_slug": "item_id",
                                 "snapshot_day": "date",
                                 "listing_count": "sell_listings"})
            )
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["sell_listings"] = df["sell_listings"].astype(int)
            # Retained so `_add_supply_depth_features` keeps its column contract.
            # The old Steam/Skinport split no longer exists — one consolidated
            # depth series replaces it — so this stays 0 rather than being
            # dropped, which would change the feature set as a side effect of a
            # data-source change.
            df["skinport_quantity"] = 0
            df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

            logger.info(
                f"  supply snapshots: {len(df):,} item-days, "
                f"{df.item_id.nunique():,} items, "
                f"{df.date.nunique()} days from {len(paths)} file(s)"
            )
            self._supply_snap_cache = df
            return df
        except Exception as e:
            logger.warning(f"  Failed to load supply depth from archive: {e}")
            self._supply_snap_cache = empty
            return empty

    def _add_supply_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add supply-depth features: listing count, change, ratio.

        Uses steam sell_listings as the primary signal, with
        skinport_quantity as a secondary source where available.

        Features:
          supply_listings_log        — log(1 + sell_listings)
          supply_listings_zscore     — z-score vs item's own history
          supply_change_7d           — % change in sell_listings (7d)
          supply_skinport_qty_log    — log(1 + skinport_quantity)
          supply_to_volume_ratio     — sell_listings / (volume_30d + 1)
        """
        snap = self._fetch_supply_snapshots()
        if snap.empty:
            for col in ["supply_listings_log", "supply_listings_zscore",
                        "supply_change_7d", "supply_skinport_qty_log",
                        "supply_to_volume_ratio"]:
                df[col] = 0.0
            return df

        df = df.merge(snap, on=["item_id", "date"], how="left")
        df["sell_listings"] = df["sell_listings"].fillna(0).astype(int)
        df["skinport_quantity"] = df["skinport_quantity"].fillna(0).astype(int)

        # Log transform (scale-invariant, handles right skew)
        df["supply_listings_log"] = np.log1p(df["sell_listings"]).astype(np.float32)
        df["supply_skinport_qty_log"] = np.log1p(df["skinport_quantity"]).astype(np.float32)

        # Z-score vs item's own history (30d rolling)
        df = df.sort_values(["item_id", "date"])
        grouped = df.groupby("item_id")["sell_listings"]
        rolling_mean = grouped.transform(lambda x: x.rolling(30, min_periods=1).mean())
        rolling_std = grouped.transform(lambda x: x.rolling(30, min_periods=1).std().replace(0, np.nan))
        df["supply_listings_zscore"] = (
            (df["sell_listings"] - rolling_mean) / rolling_std
        ).fillna(0).astype(np.float32)

        # 7-day change in listing count
        df["supply_change_7d"] = (
            (df["sell_listings"] - grouped.shift(7))
            / grouped.shift(7).replace(0, np.nan) * 100
        ).fillna(0).astype(np.float32)

        # Supply-to-volume ratio: listings / trailing 30d volume
        vol_col = None
        for candidate in ["volume_30d", "volume_mean_30d", "volume"]:
            if candidate in df.columns:
                vol_col = candidate
                break
        if vol_col:
            df["supply_to_volume_ratio"] = (
                df["sell_listings"] / (df[vol_col].fillna(0).replace(0, 1) + 1)
            ).astype(np.float32)
        else:
            df["supply_to_volume_ratio"] = 0.0

        # Drop intermediate raw columns
        df = df.drop(columns=["sell_listings", "skinport_quantity"], errors="ignore")

        logger.info("  supply depth features added")
        return df

    # ── Social sentiment features (Reddit mentions, VADER scores) ──────

    def _fetch_social_mentions(self) -> pd.DataFrame:
        """Load social mentions from the DB.

        Returns DataFrame with columns:
          item_id, date, mention_count, avg_sentiment, avg_score
        One row per item per day with at least one mention.
        """
        if hasattr(self, "_social_cache") and self._social_cache is not None:
            return self._social_cache

        try:
            rows = self.db.execute(text("""
                SELECT
                    i.item_id,
                    DATE(sm.mentioned_at) AS date,
                    COUNT(*) AS mention_count,
                    AVG(sm.sentiment_score) AS avg_sentiment,
                    AVG(sm.post_score) AS avg_score
                FROM social_mentions sm
                JOIN items i ON i.id = sm.item_id
                GROUP BY i.item_id, DATE(sm.mentioned_at)
                ORDER BY i.item_id, date
            """)).fetchall()
            df = pd.DataFrame(rows, columns=[
                "item_id", "date", "mention_count",
                "avg_sentiment", "avg_score"
            ])
            if df.empty:
                logger.info("  social mentions: empty")
                self._social_cache = df
                return df
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["mention_count"] = df["mention_count"].fillna(0).astype(int)
            df["avg_sentiment"] = df["avg_sentiment"].fillna(0.0).astype(float)
            df["avg_score"] = df["avg_score"].fillna(0.0).astype(float)
            logger.info(f"  social mentions: {len(df):,} rows, {df.item_id.nunique():,} items")
            self._social_cache = df
            return df
        except Exception as e:
            logger.warning(f"  Failed to load social mentions: {e}")
            self._social_cache = pd.DataFrame()
            return self._social_cache

    def _add_social_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add social sentiment features: mention counts, velocity, sentiment.

        Features:
          social_mentions_1d       — Reddit mention count in last 24h
          social_mentions_7d       — Reddit mention count in last 7 days
          social_mention_velocity  — mentions_1d / max(mentions_7d, 1)
          social_sentiment_7d      — Rolling 7d avg VADER compound score
          social_score_7d          — Rolling 7d avg Reddit post score
        """
        social = self._fetch_social_mentions()
        if social.empty:
            for col in ["social_mentions_1d", "social_mentions_7d",
                        "social_mention_velocity", "social_sentiment_7d",
                        "social_score_7d"]:
                df[col] = 0.0
            return df

        df = df.merge(social, on=["item_id", "date"], how="left")
        df["mention_count"] = df["mention_count"].fillna(0).astype(int)
        df["avg_sentiment"] = df["avg_sentiment"].fillna(0.0).astype(float)
        df["avg_score"] = df["avg_score"].fillna(0.0).astype(float)

        # Rolling mention counts per item
        df = df.sort_values(["item_id", "date"])
        grouped = df.groupby("item_id")["mention_count"]
        df["social_mentions_1d"] = grouped.transform(
            lambda x: x.rolling(1, min_periods=1).sum()
        ).fillna(0).astype(np.float32)
        df["social_mentions_7d"] = grouped.transform(
            lambda x: x.rolling(7, min_periods=1).sum()
        ).fillna(0).astype(np.float32)

        # Mention velocity: acceleration signal
        mentions_7d = df["social_mentions_7d"].replace(0, 1)
        df["social_mention_velocity"] = (
            df["social_mentions_1d"] / mentions_7d
        ).fillna(0).astype(np.float32)

        # Rolling 7d avg sentiment and score
        grouped_sent = df.groupby("item_id")["avg_sentiment"]
        df["social_sentiment_7d"] = grouped_sent.transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        ).fillna(0).astype(np.float32)

        grouped_score = df.groupby("item_id")["avg_score"]
        df["social_score_7d"] = grouped_score.transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        ).fillna(0).astype(np.float32)

        df = df.drop(columns=["mention_count", "avg_sentiment", "avg_score"], errors="ignore")

        logger.info("  social sentiment features added")
        return df

    def _add_item_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        meta = self._fetch_item_metadata()
        if meta.empty:
            return df
        df = df.merge(meta, on="item_id", how="left")
        df["type"] = df["type"].fillna("unknown")
        type_dummies = pd.get_dummies(df["type"], prefix="item_type").astype(int)
        for t in ["skin", "sticker", "case", "graffiti", "musickit", "unknown"]:
            col = f"item_type_{t}"
            if col not in type_dummies.columns:
                type_dummies[col] = 0
        df = pd.concat([df, type_dummies], axis=1)
        df = df.drop(columns=["type"])
        return df

    def _add_temporal_features(self, df: pd.DataFrame,
                               item_first_dates=None) -> pd.DataFrame:
        dates = pd.to_datetime(df["date"])
        dow = dates.dt.dayofweek
        month = dates.dt.month
        doy = dates.dt.dayofyear
        df["day_of_week"] = dow
        df["month"] = month
        df["quarter"] = dates.dt.quarter
        df["day_of_year"] = doy
        df["is_weekend"] = (dow >= 5).astype(int)

        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 366)
        if "item_id" in df.columns:
            if item_first_dates is not None:
                # The frame's own min date is wrong whenever the caller has
                # truncated history (the predict path keeps only
                # PREDICT_TAIL_ITEM_DAYS item-days), which would report every
                # item as exactly that many days old. Prefer the true
                # first-seen date when the caller can supply it.
                item_first_date = df["item_id"].map(item_first_dates)
                item_first_date = item_first_date.fillna(
                    df.groupby("item_id")["date"].transform("min")
                )
            else:
                item_first_date = df.groupby("item_id")["date"].transform("min")
            df["item_age_days"] = (pd.to_datetime(df["date"]) - pd.to_datetime(item_first_date)).dt.days
        else:
            df["item_age_days"] = 0
        return df

    def _add_event_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        event_types = ["major", "operation", "case_drop", "update", "game_update"]

        decay_constants = self.event_decay_constants

        if events_df.empty:
            for event_type in event_types:
                df[f"event_decay_{event_type}"] = 0.0
                df[f"events_next_30d_{event_type}"] = 0
                df[f"event_density_30d_{event_type}"] = 0
                df[f"event_density_90d_{event_type}"] = 0
            return df

        for event_type in event_types:
            type_events = events_df[events_df["type"] == event_type].sort_values("date")
            decay_tau = decay_constants.get(event_type, 30)

            if type_events.empty:
                df[f"event_decay_{event_type}"] = 0.0
                df[f"events_next_30d_{event_type}"] = 0
                df[f"event_density_30d_{event_type}"] = 0
                df[f"event_density_90d_{event_type}"] = 0
                continue

            dates = pd.to_datetime(df["date"])
            event_dates = pd.to_datetime(type_events["date"].unique())
            sorted_events = np.sort(event_dates)
            all_dates = dates.values

            # Exponential decay of most recent event: exp(-days_since / tau)
            indices = np.searchsorted(sorted_events, all_dates) - 1
            valid = indices >= 0
            decay_val = np.zeros(len(dates), dtype=float)
            if valid.any():
                last_event_dates = sorted_events[indices[valid]]
                days_since = (all_dates[valid] - last_event_dates).astype('timedelta64[D]').astype(float)
                decay_val[valid] = np.exp(-days_since / decay_tau)
            df[f"event_decay_{event_type}"] = decay_val

            # Count events in next 30 days (vectorized)
            left = np.searchsorted(sorted_events, all_dates, side="right")
            right = np.searchsorted(sorted_events, all_dates + np.timedelta64(30, "D"), side="right")
            df[f"events_next_30d_{event_type}"] = right - left

            # Event density: number of events in recent windows
            past_30 = np.searchsorted(sorted_events, all_dates, side="right") - np.searchsorted(
                sorted_events, all_dates - np.timedelta64(30, "D"), side="right"
            )
            past_90 = np.searchsorted(sorted_events, all_dates, side="right") - np.searchsorted(
                sorted_events, all_dates - np.timedelta64(90, "D"), side="right"
            )
            df[f"event_density_30d_{event_type}"] = past_30
            df[f"event_density_90d_{event_type}"] = past_90

        # Add relevance-weighted event signals
        # Different item types respond differently to each event type.
        has_identity = all(c in df.columns for c in
                           ["is_sticker", "is_case", "is_glove", "is_knife"])
        if has_identity:
            is_skin = (
                1 - df["is_sticker"] - df["is_case"] - df["is_music_kit"]
                - df["is_graffiti"] - df["is_charm"] - df["is_patch"]
                - df["is_capsule"]
            ).clip(lower=0).astype(float)

            relevance_map = {
                "major": (
                    df["is_sticker"].astype(float) * 1.0
                    + df["is_case"].astype(float) * 0.3
                    + df["is_capsule"].astype(float) * 0.6
                ),
                "operation": (
                    df["is_case"].astype(float) * 1.0
                    + is_skin * 0.3
                ),
                "case_drop": (
                    df["is_case"].astype(float) * 1.0
                    + is_skin * 0.5
                ),
                "update": is_skin * 0.5,
                "game_update": is_skin * 0.3,
            }
            for et in event_types:
                raw_col = f"event_decay_{et}"
                if raw_col in df.columns:
                    weight = relevance_map.get(et, pd.Series(1.0, index=df.index))
                    df[f"event_decay_{et}_weighted"] = df[raw_col] * weight

        return df

    # Every per-date market quantity below is a plain mean over all items on a
    # date. That is what makes chunked prediction exact: a mean is recoverable
    # from (sum, count) partials, so the identical table can be built one item
    # chunk at a time without ever holding the whole frame. See
    # _accumulate_market_partials / _market_from_partials.
    MARKET_MEAN_COLS = ("return_1d", "return_7d", "return_14d", "return_30d",
                        "price_std_30d", "volume")

    @staticmethod
    def _accumulate_market_partials(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Per-date (sum, count) for each market column present in this frame.

        `count` excludes NaN, matching pandas' skipna mean, so combining
        partials reproduces a global groupby(date).mean() exactly.
        """
        partials = {}
        for col in ItemForecaster.MARKET_MEAN_COLS:
            if col in df.columns:
                partials[col] = df.groupby("date")[col].agg(["sum", "count"])
        return partials

    @staticmethod
    def _market_from_partials(partial_list: List[Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """Combine per-chunk partials into the per-date market table."""
        present = set()
        for p in partial_list:
            present |= set(p)

        market = pd.DataFrame()
        for col in ItemForecaster.MARKET_MEAN_COLS:
            if col not in present:
                continue
            frames = [p[col] for p in partial_list if col in p]
            totals = pd.concat(frames).groupby(level=0).sum()
            # count == 0 means every value for that date was NaN; pandas' mean
            # yields NaN there, and 0/NaN does too.
            mean = totals["sum"] / totals["count"].replace(0, np.nan)

            if col == "volume":
                # A date with no volume at all must not contribute to the
                # rolling window, so keep NaN rather than filling.
                market["daily_market_vol"] = mean
                market["_volume_observed"] = totals["count"].sum() > 0
            elif col == "price_std_30d":
                market["market_volatility_30d"] = mean
            else:
                market[f"market_{col}"] = mean

        if not market.empty:
            market = market.sort_index()
            if "daily_market_vol" in market.columns:
                market["market_volume_mean_30d"] = market["daily_market_vol"].rolling(
                    30, min_periods=1
                ).mean()
        return market

    # ------------------------------------------------------------------
    # Tier lead-lag (gated by tier_lead_enabled)
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate_tier_partials(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Per-(date, price_tier) (sum, count) of `return_1d`.

        The same (sum, count) shape _accumulate_market_partials uses, and for the
        same reason: the predict path engineers features in item chunks, and a
        tier index computed over one chunk is an index over the wrong
        cross-section. Combining partials reproduces a global
        groupby([date, price_tier]).mean() exactly.

        **Mean, not median, and that is a real choice.** A median does not
        compose from partials, so a chunked path could not reproduce a
        whole-frame median without holding every chunk in memory -- which is the
        thing chunking exists to avoid. The exposure this accepts is outlier
        sensitivity in the index. It lands mostly where it does least harm: the
        feature is the tier *above* each item, so the indices actually consumed
        are the more liquid, less noisy ones, and the noisiest tier (0, sub-$1)
        is never read by anything because no tier leads it.
        """
        if "return_1d" not in df.columns or "price_tier" not in df.columns:
            return None
        return df.groupby(["date", "price_tier"])["return_1d"].agg(["sum", "count"])

    @classmethod
    def _tier_lead_from_partials(
        cls, partial_list: List[Optional[pd.DataFrame]]
    ) -> pd.DataFrame:
        """Combine tier partials into a per-date table of each tier's LAGGED mean.

        Returns a frame indexed by date with one column per tier, holding that
        tier's mean `return_1d` on the previous **available** date.

        The shift is positional over sorted observed dates, not calendar. The
        archive is missing whole days (docs: aggregator-archive-day-gaps), and
        every other lag in this file resolves against observed item-days for the
        same reason -- a calendar shift would emit NaN across a collection outage
        and a `.reindex(full_calendar)` would fabricate an index for a day
        nothing was collected on.
        """
        frames = [p for p in partial_list if p is not None and not p.empty]
        if not frames:
            return pd.DataFrame()

        totals = pd.concat(frames).groupby(level=[0, 1]).sum()
        # count == 0 means every value for that (date, tier) was NaN; pandas'
        # mean yields NaN there and so does 0/NaN.
        mean = totals["sum"] / totals["count"].replace(0, np.nan)
        table = mean.unstack(level=1).sort_index()
        # One row per observed date; shift by one row = one observed date back.
        return table.shift(1)

    def _apply_tier_lead(self, df: pd.DataFrame, lagged: pd.DataFrame) -> pd.DataFrame:
        """Attach `tier_lead_return_1d` given a precomputed lagged tier table.

        For an item in tier t on date d, the value is tier **t+1**'s mean
        `return_1d` on the previous observed date -- the return of the next more
        liquid tier, one day earlier. Direction matters and the folk version is
        backwards: cheap does NOT lead expensive (+0.043, inside the noise band).

        The top tier has no tier above it and gets NaN, as does any (date, tier)
        the archive did not observe. NaN is left for the existing median fill
        rather than zero-filled here: a zero would assert "the tier above was
        flat", which is a different claim from "unobserved".
        """
        col = self.TIER_LEAD_FEATURES[0]
        if lagged.empty or "price_tier" not in df.columns:
            df[col] = np.nan
            return df

        # Long form (date, tier_below) -> value, so the join is a plain merge on
        # the item's own tier. `lead_tier` is the tier whose lagged return the
        # item receives; `tier_below` is the tier that receives it.
        #
        # melt rather than stack: stack's dropna parameter is deprecated in
        # pandas 2.1 and its replacement (future_stack=True) does not accept one,
        # so the NaN-preserving spelling differs across versions. melt keeps NaN
        # unconditionally and is stable API.
        long = lagged.rename_axis(columns="lead_tier").reset_index().melt(
            id_vars="date", value_name=col)
        long["tier_below"] = long["lead_tier"] - 1
        long = long[long["tier_below"] >= 0][["date", "tier_below", col]]

        out = df.merge(
            long,
            left_on=["date", "price_tier"],
            right_on=["date", "tier_below"],
            how="left",
        ).drop(columns=["tier_below"])
        # merge() returns a fresh RangeIndex; every caller downstream of the
        # feature builders assumes positional alignment with `df`, so make the
        # reset explicit rather than incidental (the same note
        # _apply_market_aggregates carries for its volume branch).
        return out.reset_index(drop=True)

    def _add_tier_lead_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Whole-frame path: accumulate the tier table and apply it."""
        logger.info("Adding tier lead-lag feature...")
        lagged = self._tier_lead_from_partials([self._accumulate_tier_partials(df)])
        return self._apply_tier_lead(df, lagged)

    # ------------------------------------------------------------------
    # Cross-sectional rank transform (gated by cross_sectional_rank_enabled)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_cross_sectional_ranks(
        df: pd.DataFrame, cols: List[str],
        reference_mask: Optional[pd.Series] = None,
        skip_cols: Optional[List[str]] = None,
        skipped_out: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Map each column in *cols* to its centred within-date percentile.

        **`reference_mask` is what makes the served frame comparable to the
        trained one, and it is not optional on the predict path.** Training
        applies the median-price floor before this runs, so the booster is
        fitted on percentiles within the **916-item >= $1 cohort**; `predict`'s
        frame is every backfilled item with 14 days of history, **5,536 of
        them, ~72% sub-$1**. Ranking that pooled frame would feed the booster a
        column it never saw, in range and without an error. Pass the cohort as
        `reference_mask` and every row -- cohort or not -- is positioned in the
        cohort's distribution instead. Cohort rows then receive exactly the
        number training computed; out-of-cohort rows are placed by
        interpolation and pin to +/-1 outside the cohort's range, which is the
        honest answer for an item outside the fitted support.

        `None` means "every row is in the cohort", which is the training path.

        **`skip_cols` is the same argument for the date-constant skip.** Pass
        the set training recorded and serving reproduces its decision; leave it
        `None` and the skip is re-derived from the frame in hand, which is how
        the served frame came to rank 31 columns against training's 32.
        `skipped_out`, if given, receives whatever was skipped -- that is how
        training learns what to record.

        Range [-1, 1], NaN preserved (pandas' rank skips NaN, so a missing
        characteristic stays missing and is median-filled downstream -- which now
        resolves to the *cross-sectional* median by construction rather than to
        the persisted feature_medians, removing the calendar-gap lag-fill
        artifact as a side effect).

        **This deliberately deviates from the formula C1 specifies.** Track C1
        (docs/research/2026-08-09-next-steps.md) writes it as
        `2 * (rank(pct=True) - 0.5)`, but pandas' `pct=True` divides by n, so
        that maps the top item to exactly +1.0 and the bottom to `-1 + 2/n`. The
        cross-section width varies by date here -- items enter and exit, which is
        the same fact that makes `lambdarank_norm` matter -- so an endpoint that
        moves with n injects a date-varying artifact into the one transform whose
        entire purpose is removing date effects. The mid-rank form
        `(rank - 0.5) / n` is symmetric at every n and has no mass at the
        endpoints. At n ~ 900 the difference is ~0.1%; it is fixed here because
        it is free to fix, not because it was going to dominate.

        **Date-constant columns are skipped, and skipping them is required, not
        an optimisation.** Ranking a column that is identical across the
        cross-section yields all-ties -> pct 0.5 everywhere -> exactly 0 after
        centring, i.e. the transform would silently delete the column. Nothing in
        the current allowlist is date-constant, but `tier_lead_return_1d`'s
        upstream index is, and a date-level feature added later would be erased
        without a word.
        """
        if not cols:
            return df
        excluded = sorted(set(cols) & ItemForecaster.RANK_TRANSFORM_EXCLUDED)
        if excluded:
            logger.info(
                f"  cross-sectional rank: NOT transforming {excluded} — "
                f"read as a gate, not only as a feature (RANK_TRANSFORM_EXCLUDED)")
        present = [c for c in cols if c in df.columns
                   and c not in ItemForecaster.RANK_TRANSFORM_EXCLUDED]
        if not present:
            return df

        if reference_mask is None:
            ref = df
        else:
            ref = df[np.asarray(reference_mask, dtype=bool)]
            if ref.empty:
                raise ValueError(
                    "cross-sectional rank: the reference cohort is empty, so "
                    "there is no distribution to rank against. Ranking the "
                    "whole frame instead would hand the booster percentiles "
                    "from a population it was never fitted on."
                )

        grouped = ref.groupby("date")
        skipped = []
        follow = None if skip_cols is None else set(skip_cols)
        for col in present:
            if follow is not None:
                # Serving: obey the artifact. Re-deriving here is what made the
                # served frame disagree with the fitted one -- 32/32 columns
                # ranked at training against 31/32 at serving, because
                # macd_missing is date-constant on the predict frame only. A
                # column training ranked must be ranked now even if it is
                # constant today (all-ties -> exactly 0.0, which is what
                # training produced on ITS constant dates), and a column
                # training skipped must stay raw even if it varies today.
                if col in follow:
                    skipped.append(col)
                    continue
            else:
                # nunique over the whole frame would be O(rows); a per-date std
                # of 0 on every date is the property that matters and
                # transform() is one pass. Constant-within-date includes the
                # all-NaN case, where std is NaN and the column carries nothing
                # anyway.
                within_date_spread = grouped[col].transform("std")
                if not (within_date_spread.fillna(0) > 0).any():
                    skipped.append(col)
                    continue
            # Mid-rank, not rank(pct=True) — see the docstring. `count` excludes
            # NaN, matching rank()'s own treatment, so an item-day missing this
            # characteristic does not inflate the denominator for the rest.
            n = grouped[col].transform("count")
            ranked = 2.0 * ((grouped[col].rank() - 0.5) / n - 0.5)
            if ref is df:
                df[col] = ranked
            else:
                df[col] = ItemForecaster._place_in_reference(
                    df, ref, col, ranked)

        if skipped:
            logger.info(
                f"  cross-sectional rank: skipped {len(skipped)} date-constant "
                f"column(s) {skipped} — ranking them would zero them out"
            )
        logger.info(
            f"  cross-sectional rank transform applied to "
            f"{len(present) - len(skipped)}/{len(present)} features"
        )
        if skipped_out is not None:
            skipped_out.extend(skipped)
        return df

    @staticmethod
    def _place_in_reference(df: pd.DataFrame, ref: pd.DataFrame, col: str,
                            ranked: pd.Series) -> pd.Series:
        """Cohort rows keep *ranked*; the rest get their position within it.

        Two-sided `searchsorted` so a value tying with k cohort values lands at
        the middle of their block rather than at either edge -- the same
        mid-rank convention the cohort rows themselves get, so the two are on
        one scale. A value below the whole cohort maps to -1 and one above it to
        +1: the booster has no fitted support out there and the endpoint says so
        rather than pretending to interpolate.
        """
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        out.loc[ranked.index] = ranked.to_numpy()

        outside = df.index.difference(ref.index)
        if len(outside) == 0:
            return out

        others = df.loc[outside, ["date", col]]
        for date, block in others.groupby("date", sort=False):
            ref_vals = ref.loc[ref["date"] == date, col].dropna().to_numpy()
            if ref_vals.size == 0:
                # No cohort observation of this characteristic on this date, so
                # there is no distribution to place anything in. NaN, which the
                # median fill downstream resolves -- unlike a 0.0, which would
                # read as "exactly median" and be indistinguishable from a real
                # mid-ranked value.
                continue
            ref_vals.sort()
            values = block[col].to_numpy(dtype="float64")
            known = ~np.isnan(values)
            lo = np.searchsorted(ref_vals, values[known], side="left")
            hi = np.searchsorted(ref_vals, values[known], side="right")
            pct = ((lo + hi) / 2.0) / ref_vals.size
            placed = np.full(values.shape, np.nan)
            placed[known] = 2.0 * (pct - 0.5)
            out.loc[block.index] = placed
        return out

    def _reference_cohort_mask(self, df: pd.DataFrame) -> pd.Series:
        """Rows belonging to the cohort the loaded artifact was trained on.

        By **item median over the frame**, which is `_filter_by_median_price`'s
        statistic. A row-wise price test would be a different cohort: one spike
        would promote a penny item for a single date, and that date's
        percentiles would then be computed over a population training never
        used.

        ⚠️ **The statistic matches; the SUPPORT does not.** Training takes the
        median over `build_training_data`'s 1460-day window. This frame is the
        predict frame -- fetched at `PREDICT_FETCH_DAYS` (730) and already cut to
        `PREDICT_TAIL_ITEM_DAYS` (240 observed item-days) by the time this runs.
        Same rule, ~6x different support, so the two cohorts are not identical
        sets. Measured 2026-08-13 at six anchors: **0.7-0.9% disagreement**
        (Jaccard 0.991-0.993), and **directional** -- serving admits 140-173
        items training excludes against 30-33 the other way, because a shorter
        recent window catches items that have risen through the floor lately.
        Too small to have been C1's CV->serving gap, which is what it was
        measured to test, but it is the term that would grow in a rising market
        and nothing tracks it. The >25% guard in `predict` is a frame-sanity
        check and would not see this.
        `docs/changelog/2026-08-13-cohort-geometry-is-not-c1s-gap.md`.
        """
        floor = self._artifact_min_median_price
        if not floor:
            return pd.Series(True, index=df.index)
        item_median = df.groupby("item_id")["price"].median()
        keep = set(item_median[item_median >= floor].index)
        return df["item_id"].isin(keep)

    def _add_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market-level and category-level context features."""
        market = self._market_from_partials([self._accumulate_market_partials(df)])
        return self._apply_market_aggregates(df, market)

    def _apply_market_aggregates(self, df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
        """Attach cross-sectional features given a precomputed per-date table.

        Split out of _add_cross_sectional_features so the whole-frame path and
        the chunked path share one implementation — the only difference is
        where `market` came from.
        """
        logger.info("Adding cross-sectional features...")

        # Market return: mean return across all items per date
        for lag in [1, 7, 14, 30]:
            ret_col = f"return_{lag}d"
            market_col = f"market_return_{lag}d"
            if ret_col not in df.columns or market_col not in market.columns:
                continue
            df[market_col] = df["date"].map(market[market_col])
            df[f"item_return_vs_market_{lag}d"] = df[ret_col] - df[market_col]

        # Market volatility: mean of individual item volatilities per date
        if "market_volatility_30d" in market.columns:
            df["market_volatility_30d"] = df["date"].map(market["market_volatility_30d"])

        # Market volume: daily market mean, then rolling 30d of that. Gated on
        # whether volume was observed ACROSS ALL items, not just this chunk —
        # a chunk with no volume must still get the global columns.
        if "market_volume_mean_30d" in market.columns and bool(market["_volume_observed"].iloc[0]):
            df["market_volume_mean_30d"] = df["date"].map(market["market_volume_mean_30d"])
            # This branch used to be a df.merge(), which returns a fresh
            # RangeIndex. map() preserves the index, so reset explicitly to keep
            # the frame downstream of this function byte-for-byte as before.
            df = df.reset_index(drop=True)
            df["item_volume_vs_market_30d"] = (
                df["volume"] / df["market_volume_mean_30d"].replace(0, np.nan)
            )

        # Market regime: refined 5-state regime with duration tracking
        if "market_return_30d" in df.columns:
            market_ret_median = df.groupby("date")["market_return_30d"].transform("median")
            df["market_regime_crash"] = (market_ret_median < -10).astype(int)
            df["market_regime_bear"] = ((market_ret_median >= -10) & (market_ret_median < -3)).astype(int)
            df["market_regime_range"] = ((market_ret_median >= -3) & (market_ret_median <= 3)).astype(int)
            df["market_regime_bull"] = ((market_ret_median > 3) & (market_ret_median <= 10)).astype(int)
            df["market_regime_mania"] = (market_ret_median > 10).astype(int)

            # Market regime duration: consecutive days in same regime
            regime_cols = ["market_regime_crash", "market_regime_bear",
                           "market_regime_range", "market_regime_bull",
                           "market_regime_mania"]
            combined = pd.DataFrame(index=df.index, dtype=int)
            combined["regime_id"] = 0
            for i, col in enumerate(regime_cols):
                if col in df.columns:
                    combined.loc[df[col] == 1, "regime_id"] = i + 1
            # Count consecutive same-regime days per item
            regime_changes = (combined["regime_id"] != combined["regime_id"].groupby(df["item_id"]).shift(1)).astype(int)
            df["market_regime_duration_days"] = regime_changes.groupby(df["item_id"]).cumsum().groupby(
                [df["item_id"], regime_changes.cumsum()]
            ).cumcount() + 1

        # Market return percentile vs rolling 365-day history
        # Uses rolling rank (fast Cython) instead of rolling+apply (slow Python loop).
        if "market_return_30d" in df.columns:
            df["market_return_30d_percentile"] = df.groupby("item_id")["market_return_30d"].transform(
                lambda x: x.rolling(365, min_periods=30).rank(pct=True)
            )

        return df

    def _add_item_identity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add item identity features (is_stattrak, is_knife, quality_rank, etc.).

        Fetches item names from the DB and parses them into structured identity
        features. Items not found in the DB get default (0) values.
        """
        logger.info("Adding item identity features...")

        meta_df = self._fetch_item_metadata()
        if meta_df.empty:
            logger.warning("  No item metadata available; using default identity features")
            identity_cols = [
                "is_stattrak", "is_souvenir", "is_knife", "is_glove",
                "is_sticker", "is_case", "is_capsule", "is_agent",
                "is_music_kit", "is_graffiti", "is_charm", "is_patch",
                "quality_rank",
            ]
            for col in identity_cols:
                df[col] = 0
            return df

        item_map = {}
        for _, r in meta_df.iterrows():
            item_map[str(r["item_id"])] = r

        # Build identity features for each unique item
        identity_cache = {}
        for item_id in df["item_id"].unique():
            item_row = item_map.get(str(item_id))
            if item_row is None:
                identity_cache[item_id] = {
                    "is_stattrak": 0, "is_souvenir": 0, "is_knife": 0,
                    "is_glove": 0, "is_sticker": 0, "is_case": 0,
                    "is_capsule": 0, "is_agent": 0, "is_music_kit": 0,
                    "is_graffiti": 0, "is_charm": 0, "is_patch": 0,
                    "quality_rank": 0,
                }
                continue

            name = item_row["name"]
            db_type = item_row["type"]
            parsed = parse_item_name(name) if name else {}

            use_type = db_type or "skin"
            identity_cache[item_id] = {
                "is_stattrak": int(parsed.get("is_stattrak", False)),
                "is_souvenir": int(parsed.get("is_souvenir", False)),
                "is_knife": int(parsed.get("is_knife", False)),
                "is_glove": int(parsed.get("is_glove", False)),
                "is_sticker": int(use_type == "sticker" or parsed.get("is_sticker", False)),
                "is_case": int(use_type == "case" or parsed.get("is_case", False)),
                "is_capsule": int(parsed.get("is_capsule", False)),
                "is_agent": int(parsed.get("is_agent", False)),
                "is_music_kit": int(use_type == "musickit" or parsed.get("is_music_kit", False)),
                "is_graffiti": int(use_type == "graffiti" or parsed.get("is_graffiti", False)),
                "is_charm": int(parsed.get("is_charm", False)),
                "is_patch": int(parsed.get("is_patch", False)),
                "quality_rank": int(parsed.get("quality_rank", 0)),
            }

        # Map identity features onto the dataframe
        if not identity_cache:
            logger.warning("  No identity features computed (empty cache)")
            identity_cols = [
                "is_stattrak", "is_souvenir", "is_knife", "is_glove",
                "is_sticker", "is_case", "is_capsule", "is_agent",
                "is_music_kit", "is_graffiti", "is_charm", "is_patch",
                "quality_rank",
            ]
            for col in identity_cols:
                df[col] = 0
            return df

        identity_df = pd.DataFrame.from_dict(identity_cache, orient="index")
        identity_df.index.name = "item_id"
        identity_df = identity_df.reset_index()

        df = df.merge(identity_df, on="item_id", how="left")

        for col in identity_cache[next(iter(identity_cache))].keys():
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0).astype(int)

        logger.info("  Added item identity features")
        return df

    def _prune_features(self, df: pd.DataFrame) -> List[str]:
        """Remove highly correlated features to reduce noise and multicollinearity.

        Identifies feature pairs with correlation > PRUNE_CORRELATION_THRESHOLD
        and drops one from each pair, keeping features with lower index (earlier
        in the list).
        """
        if len(self.feature_cols) < 2:
            return self.feature_cols

        corr = df[self.feature_cols].corr().abs()

        # Extract only the upper triangle (each unordered pair appears once)
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        high_pairs = upper.stack()
        high_pairs = high_pairs[high_pairs > self.PRUNE_CORRELATION_THRESHOLD]

        to_drop = set()
        feature_index = {name: i for i, name in enumerate(self.feature_cols)}
        for feat_a, feat_b in high_pairs.index:
            # Keep the earlier feature (lower index), drop the later one
            if feature_index[feat_a] < feature_index[feat_b]:
                to_drop.add(feat_b)
            else:
                to_drop.add(feat_a)

        pruned = [c for c in self.feature_cols if c not in to_drop]
        if pruned != self.feature_cols:
            logger.info(
                f"  Pruned {len(self.feature_cols) - len(pruned)} features "
                f"(corr>{self.PRUNE_CORRELATION_THRESHOLD}): "
                f"{len(pruned)} remaining"
            )
        return pruned

    def _validate_feature_groups(
        self, X_val: np.ndarray, y_val: np.ndarray,
        feature_names: List[str], horizon: int = 7,
        n_shuffles: int = 20, min_drop_pp: float = 0.5,
        significance_level: float = 0.05,
        offset=None,
    ) -> Dict[str, Dict]:
        """Validate feature groups via permutation importance with
        statistical significance gating.

        For each feature group, shuffles its columns on the validation set
        and measures the directional accuracy drop vs the unshuffled baseline.
        A group is kept only if it passes BOTH:

          1. **Statistical significance** — the p-value (fraction of shuffled
             trials where accuracy >= baseline accuracy) is below
             `significance_level`. This ensures the group's signal isn't
             attributable to noise.
          2. **Practical significance** — the accuracy drop exceeds
             `min_drop_pp` (default 0.5pp). This ensures the effect is large
             enough to matter for forecasting.

        Uses the p50 (median) quantile model for the given horizon.

        `offset` is N1's `-return_1d` when the model was fitted on top of it. It
        is added to both the base and the shuffled predictions, because the
        metric here is a SIGN and the sign of a residual is not the sign of a
        forecast: without it every group would be scored against a model whose
        output has the wrong zero. It is deliberately NOT permuted with the
        `price_technicals` group that owns `return_1d` — the offset is part of the
        predictor, not one of the features whose contribution is being measured.

        Returns:
            {group_name: {"drop_pp": float, "base_acc": float,
                          "shuffled_acc": float, "p_value": float,
                          "passed": bool, "feature_count": int,
                          "features": [str]}}
        """
        model_key = (horizon, 0.5)
        if model_key not in self.models:
            return {}

        model = self.models[model_key]
        if isinstance(model, list):
            model = model[0]

        groups: Dict[str, List[str]] = {}
        for i, name in enumerate(feature_names):
            g = _feature_group(name)
            groups.setdefault(g, []).append(name)

        col_to_idx = {name: i for i, name in enumerate(feature_names)}
        group_indices: Dict[str, List[int]] = {}
        for g, feats in groups.items():
            idxs = [col_to_idx[f] for f in feats if f in col_to_idx]
            if idxs:
                group_indices[g] = idxs

        _off = 0.0 if offset is None else np.asarray(offset, dtype=float)
        p50_idx = np.squeeze(model.predict(X_val)) + _off
        base_acc = np.mean((p50_idx > 0) == (y_val > 0)) * 100

        results = {}
        for group, idxs in group_indices.items():
            shuffled_accs = []
            for _ in range(n_shuffles):
                X_shuf = X_val.copy()
                for i in idxs:
                    RNG.shuffle(X_shuf[:, i])
                p50_shuf = np.squeeze(model.predict(X_shuf)) + _off
                acc = np.mean((p50_shuf > 0) == (y_val > 0)) * 100
                shuffled_accs.append(acc)

            shuffled_arr = np.array(shuffled_accs)
            mean_shuf = float(np.mean(shuffled_arr))
            drop_pp = base_acc - mean_shuf
            p_value = float(np.mean(shuffled_arr >= base_acc))
            passed = bool(p_value < significance_level and drop_pp >= min_drop_pp)

            results[group] = {
                "drop_pp": round(drop_pp, 2),
                "base_acc": round(float(base_acc), 2),
                "shuffled_acc": round(mean_shuf, 2),
                "p_value": round(p_value, 4),
                "passed": passed,
                "feature_count": len(idxs),
                "features": groups[group],
            }

            status = "PASS" if passed else "WARN"
            logger.info(
                f"  [feat group] {group}: {drop_pp:+.2f}pp when shuffled "
                f"({base_acc:.1f}% -> {mean_shuf:.1f}%) "
                f"p={p_value:.4f} [{status}]"
            )

        return results

    # Row-sampling keys this class owns. Listed so _apply_row_sampling can
    # strip every one before rewriting them — params cached in meta.json by an
    # older build carry GOSS keys for q50 and must not leak through the warm
    # retrain path.
    _ROW_SAMPLING_KEYS = ("data_sample_strategy", "top_rate", "other_rate",
                          "subsample", "bagging_fraction", "bagging_freq")

    @classmethod
    def _row_sampling_params(cls, quantile: float,
                             subsample: float = 0.8) -> Dict[str, Any]:
        """Row-sampling config for one quantile's LightGBM params.

        Every quantile uses bagging. q50 previously used GOSS, which ranks
        rows by |gradient| to decide which to keep — but the quantile
        objective emits constant ±alpha gradients, so that ranking is
        degenerate and the small-gradient rescaling injects bias instead of
        signal. Shipped symptom: 7d q50 saved 1-2 trees per ensemble member
        (best_iteration ~= 1), so the served median was effectively an
        intercept; 3d fits were biased even when they ran the full 1000
        rounds.

        A/B 2026-07-29 (`scripts/ab_test_q50_sampling.py`, 8-fold purge-gap
        CV, production params, 200 items): bagging improved q50 pinball by
        +5.21% (3d) / +1.76% (7d), MAE by 0.244 / 0.068, and directional
        accuracy by +1.13pp / +0.71pp, winning 7/8 and 5/8 paired folds —
        passing all three pre-registered gate criteria.

        `bagging_freq` MUST be >= 1: LightGBM's default is 0, which ignores
        bagging_fraction/subsample entirely and trains on every row. It is
        set for q50 only, because that is the exact configuration the A/B
        validated. q10/q90 keep their existing (freq-unset, therefore no-op)
        subsample — correcting that is a real but separate change and needs
        its own A/B before shipping; do not "tidy" it in here.
        """
        params: Dict[str, Any] = {
            "data_sample_strategy": "bagging",
            "subsample": subsample,
        }
        if quantile == 0.5:
            params["bagging_freq"] = 1
        return params

    @classmethod
    def _apply_row_sampling(cls, params: dict, quantile: float,
                            subsample: Optional[float] = None) -> dict:
        """Overwrite any row-sampling keys in `params` with the current
        strategy, mutating and returning `params`.

        Strips every key in `_ROW_SAMPLING_KEYS` first so stale GOSS settings
        from a previously-cached param dict cannot survive. Tree params are
        left untouched.
        """
        keep = subsample if subsample is not None else params.get("subsample", 0.8)
        for k in cls._ROW_SAMPLING_KEYS:
            params.pop(k, None)
        params.update(cls._row_sampling_params(quantile, keep))
        return params

    def _compute_cv_splits(self, sorted_dates, purge_days: int = 0):
        """Compute expanding-window CV fold boundaries.

        Returns list of (train_date_list, val_date_list) tuples.
        Each fold trains on an expanding window and validates on a fixed-width
        window of VALIDATION_WINDOW_DAYS at the end.

        Args:
            purge_days: Embargo gap, in calendar days, applied between the
                training window and the validation window. For an H-day
                forecast horizon this MUST be ``embargo_days(H)``: a training
                row dated T carries a target observed at T+H, so any train date
                within H days of ``val_start`` has a label that overlaps the
                validation period — classic horizon-forecasting leakage that
                inflates both accuracy and conformal calibration — and the
                resolved anchor behind that label carries 13 days further
                still. Default 0 reproduces the un-embargoed split (used only
                where the caller has no horizon, e.g. unit tests).
        """
        val_window = self.VALIDATION_WINDOW_DAYS  # 21 days
        step = self._cv_step_days()  # 150 days unless CV_STEP_DAYS overrides
        min_train = self.CV_MIN_TRAIN_DAYS

        folds = []
        for end in range(min_train, len(sorted_dates) - val_window + 1, step):
            val_d = sorted_dates[end:end + val_window]
            if len(val_d) < 7:
                continue
            val_start = val_d[0]
            # Purge any train date whose target (train_date + purge_days) would
            # land on or after the first validation date.
            if purge_days > 0:
                cutoff = val_start - timedelta(days=purge_days)
                train_d = [d for d in sorted_dates[:end] if d < cutoff]
            else:
                train_d = list(sorted_dates[:end])
            if not train_d:
                continue
            folds.append((train_d, list(val_d)))
        return folds

    @staticmethod
    def _purge_overlapping_train_rows(train_set, split_date, horizon: int):
        """Drop training rows whose forward label is drawn from the validation
        window.

        A row dated ``d`` is labelled with the price at ``d + horizon``
        (``prepare_targets`` merges on exactly that date). So every row in
        ``[split_date - horizon, split_date)`` carries a label the validation
        window already contains, and at ``horizon == VALIDATION_WINDOW_DAYS == 30``
        that is the *entire* window's worth of future prices.

        The band is ``embargo_days(horizon)`` — ``horizon + 13`` — not
        ``horizon``, because the label is a resolved anchor rather than a point
        observation and its support runs 13 days past its nominal date. See
        `embargo_days`. Widened 2026-08-08; before that this purged exactly
        ``horizon`` and left the carry inside the window.

        This is the same purge ``_compute_cv_splits(..., purge_days=...)``
        applies above, which the production split in ``_train_horizon_inline``
        was missing. It matters more than a CV fold does: that split produces the
        ``dval`` early stopping stops on, the set Optuna scores every trial
        against, and the directional classifier's stopping set — so the shipped
        tree counts, hyperparameters and stopping points were all selected
        against partly-seen labels.

        Rows inside the band are dropped, not moved: they belong to neither side.
        """
        if train_set.empty or "date" not in train_set.columns:
            return train_set
        cutoff = pd.to_datetime(split_date) - timedelta(days=embargo_days(horizon))
        return train_set[pd.to_datetime(train_set["date"]) < cutoff]

    @classmethod
    def _choose_validation_split(cls, tdf) -> Tuple[Optional[pd.Timestamp], bool]:
        """Pick the split date for the production holdout, widening if starved.

        Returns ``(split_date, floors_met)``; ``val_set`` is ``date >= split_date``.

        The default is the trailing ``VALIDATION_WINDOW_DAYS``. `prepare_targets`
        voids labels built across a fabricated archive day and the caller drops
        those rows, and the fabricated days sit inside that window — so at the
        production budget the window fell under ``MIN_VAL_ROWS`` at 3d/7d/14d and
        the split fell through to a positional 80/20 slice whose validation
        window is roughly ten months. Early stopping, the Optuna objective and
        the classifier's stopping set all read that window, so the shape of it
        is not a detail.

        Widening backwards is the graceful degradation the fallback should have
        been: validation stays a recent contiguous calendar window, just a wider
        one. ``floors_met=False`` means even ``MAX_VALIDATION_WINDOW_DAYS`` could
        not reach the floors — the frame is genuinely too small, and the caller
        still has the positional fallback for that.
        """
        if tdf is None or len(tdf) == 0 or "date" not in getattr(tdf, "columns", []):
            return None, False
        dates = pd.to_datetime(tdf["date"])
        max_date = dates.max()
        default_split = max_date - timedelta(days=cls.VALIDATION_WINDOW_DAYS)

        inside = dates >= default_split
        rows = int(inside.sum())
        n_dates = int(dates[inside].nunique())
        if rows >= cls.MIN_VAL_ROWS and n_dates >= cls.MIN_VAL_DATES:
            return default_split, True

        # Walk earlier distinct dates, widening the window one date at a time.
        counts = dates.value_counts()
        earlier = sorted((d for d in counts.index if d < default_split), reverse=True)
        floor_date = max_date - timedelta(days=cls.MAX_VALIDATION_WINDOW_DAYS)
        split = default_split
        for d in earlier:
            if d < floor_date:
                break
            rows += int(counts[d])
            n_dates += 1
            split = d
            if rows >= cls.MIN_VAL_ROWS and n_dates >= cls.MIN_VAL_DATES:
                return d, True
        return split, False

    def _build_production_split(self, tdf, horizon: int, max_rows: int,
                                per_item_row_sampling: bool = False):
        """The production train/val split: a trailing calendar window, purged.

        Returns ``(train_set, val_set)``. Three things happen here, in order:

        1. Pick the window (`_choose_validation_split`), widening it if voided
           labels have starved the default trailing window.
        2. Purge the horizon-day band of training rows whose labels come from
           inside the window (`_purge_overlapping_train_rows`).
        3. Fall back to the positional 80/20 split only when even a widened
           window cannot clear the floors — a frame that small has no recent
           window worth stopping on.

        `_train_horizon_inline` is the full Optuna + ensemble path and cannot be
        driven from a test, so the decision lives here where it can be.
        """
        dates = pd.to_datetime(tdf["date"])
        split_date, floors_met = self._choose_validation_split(tdf)

        if floors_met:
            train_set = tdf[dates < split_date]
            val_set = tdf[dates >= split_date]
            train_set = self._purge_overlapping_train_rows(
                train_set, split_date, horizon)
            span = (dates.max() - pd.to_datetime(split_date)).days
            if span > self.VALIDATION_WINDOW_DAYS:
                logger.info(
                    f"  Widened {horizon}d validation window to {span}d "
                    f"({len(val_set)} rows, {val_set['date'].nunique()} dates) — "
                    f"voided labels thinned the default "
                    f"{self.VALIDATION_WINDOW_DAYS}d window"
                )
        else:
            n_val = 0 if split_date is None else int((dates >= split_date).sum())
            logger.warning(
                f"  Validation set for {horizon}d holds only {n_val} rows even "
                f"widened to {self.MAX_VALIDATION_WINDOW_DAYS}d; using last 20% "
                f"of training data as fallback."
            )
            split_idx = int(len(tdf) * 0.8)
            train_set = tdf.iloc[:split_idx]
            val_set = tdf.iloc[split_idx:]
            # The fallback splits positionally, so it leaks the same way the
            # date split did. tdf is date-sorted, so the validation window
            # opens at val_set's first date.
            if len(val_set) and "date" in val_set.columns:
                train_set = self._purge_overlapping_train_rows(
                    train_set, pd.to_datetime(val_set["date"].min()), horizon)

        # Safety guard only: the calendar window is already bounded by the
        # stratified item subsample in build_training_data(). Sample randomly
        # (never tail()) so we don't truncate the calendar window, which would
        # silently disable expanding-window CV. Applied after the branch so the
        # fallback path is capped too — it reads from `tdf`, so it was not.
        #
        # Which draw depends on the caller. The uniform default gives an item
        # a share of the sample equal to its share of the rows; the per-item
        # quota gives every item the same depth, which is the axis the paired
        # harness measured as worth +5.72pp at 30d. See _per_item_row_sample.
        # Only train_set is thinned — thinning val would move the evaluation
        # cohort, which is the artifact that pairing exists to remove.
        if len(train_set) > max_rows:
            if per_item_row_sampling:
                train_set = self._per_item_row_sample(train_set, max_rows)
            else:
                train_set = train_set.sample(
                    n=max_rows, random_state=42).sort_values("date")

        return train_set, val_set

    def _cv_can_run(self, tdf, horizon: int) -> bool:
        """Whether this horizon has enough distinct dates for >=2 CV folds.

        `_cv_evaluate_horizon` raises when it does not, and the caller needs to
        fall back to the (weaker) holdout instead of failing the whole retrain —
        so feasibility is checked up front. Cheap: date arithmetic only, no fits.
        """
        splits = self._compute_cv_splits(
            sorted(tdf["date"].unique()), purge_days=embargo_days(horizon))
        return len(splits) >= 2

    def _optuna_search_params(self, X_train, y_train, X_val, y_val,
                               val_dates,
                               quantile: float = 0.5,
                               boosting_type: str = "gbdt",
                               n_trials: int = 15,
                               horizon: Optional[int] = None,
                               train_offset=None,
                               val_offset=None) -> Dict[str, Any]:
        """Bayesian hyperparameter search via Optuna.

        Searches over 6 key params with TPE, scored on within-date rank IC.

        The objective was `model.best_score["valid_0"]["quantile"]` under
        `lgb.early_stopping(20)` until 2026-08-09. FIXED_BOOST_ROUNDS replaced
        early stopping in training and CV on 2026-08-08 because the trailing
        validation window carries ~a dozen effective observations, but the
        Optuna objective was missed -- so hyperparameters were still selected on
        exactly the criterion the rest of the model had discarded. At 14d and
        30d the val-loss optimum is 25 rounds while rank IC peaks at 500-750.

        Args:
            val_dates: The validation split's date column. Rank IC is computed
                WITHIN date and never pooled: a pooled Spearman re-introduces
                the market factor and would select for the base-rate tracking
                the Pesaran-Timmermann test exists to reject.
            boosting_type: LightGBM's `boosting_type`. Production always passes
                `ItemForecaster.BOOSTING_TYPE` ("gbdt"); the parameter survives
                only so scripts/optuna_*_search.py can state it at the call site.
            n_trials: Number of Optuna trials. Short horizons (3d) need
                more trials due to noisy signal; 7d/14d/30d default to 15.
            horizon: Horizon in days. When provided, applies horizon-aware
                search-bounds overrides (e.g. 3d skips depth=3 and biases
                lambda_l2 away from 0, based on prior search results).
        """
        import optuna

        # Build the binned Dataset once and reuse across all trials. Only tree
        # params (num_leaves, learning_rate, ...) vary between trials; the data
        # and its binning (max_bin) are constant, so there's no need to re-bin.
        ds_params = {"max_bin": self.MAX_BIN, "feature_pre_filter": False}
        # N1's offset, when the caller passed it: the search has to score the
        # same quantity the production fit will produce, which is
        # `model.predict(X) + offset`.
        dtrain = lgb.Dataset(X_train, y_train, params=ds_params,
                             init_score=train_offset)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, params=ds_params,
                           init_score=val_offset)

        def objective(trial):
            # Horizon-aware search bounds: 3d overrides known-losing regions.
            if horizon == 3:
                _max_depth = trial.suggest_int("max_depth", 4, 8)
                _lambda_l2 = trial.suggest_float("lambda_l2", 0.5, 2.0, step=0.5)
            else:
                _max_depth = trial.suggest_int("max_depth", 3, 8)
                _lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 2.0, step=0.5)
            params = {
                "feature_pre_filter": False,
                "objective": "quantile",
                "alpha": quantile,
                "metric": "quantile",
                "boosting_type": boosting_type,
                "verbosity": -1,
                "n_jobs": -1,
                "random_state": 42,
                "max_bin": self.MAX_BIN,
                "num_leaves": trial.suggest_int("num_leaves", 15, 63, step=8),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0, step=0.5),
                "lambda_l2": _lambda_l2,
                "max_depth": _max_depth,
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 30, step=5),
                "min_gain_to_split": 0.1,
                "feature_fraction": 0.7,
            }
            # Row subsampling fraction — a distinct regularization lever from
            # feature_fraction. Searched for every quantile now that q50 uses
            # bagging too (see _row_sampling_params for the GOSS removal).
            self._apply_row_sampling(
                params, quantile,
                subsample=trial.suggest_float("subsample", 0.5, 0.9, step=0.1))
            # Fixed rounds, no early stopping, no pruner: the trailing val
            # window carries ~a dozen effective observations, so early stopping
            # trips on noise (FIXED_BOOST_ROUNDS, :551-560) and
            # LightGBMPruningCallback prunes on LightGBM's reported `quantile`
            # metric rather than on what this objective returns.
            #
            # cv=True so tuning happens at the depth CV actually reaches.
            # Tuning at production rounds and evaluating at CV rounds selects
            # params that only pay off deeper than the evaluation ever goes.
            model = lgb.train(
                params, dtrain,
                num_boost_round=self._boost_rounds(horizon, cv=True),
                callbacks=[lgb.log_evaluation(0)],
            )
            # Within-date, never pooled. A pooled Spearman re-introduces the
            # market factor and would select for the base-rate tracking the
            # Pesaran-Timmermann test exists to reject.
            trial_pred = model.predict(X_val)
            if val_offset is not None:
                trial_pred = trial_pred + val_offset
            ic = self._within_date_rank_ic(trial_pred, y_val, val_dates)
            return -(ic if ic is not None else 0.0)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # Warm-start with the known winning params (from prior 50-trial search),
        # so TPE starts near the answer instead of rediscovering it.
        if horizon == 3:
            study.enqueue_trial({
                "num_leaves": 47,
                "learning_rate": 0.01,
                "lambda_l1": 0.0,
                "lambda_l2": 1.5,
                "max_depth": 5,
                "min_data_in_leaf": 15,
            })

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_trial
        best_params = {
            "num_leaves": best.params["num_leaves"],
            "learning_rate": best.params["learning_rate"],
            "lambda_l1": best.params["lambda_l1"],
            "lambda_l2": best.params["lambda_l2"],
            "max_depth": best.params["max_depth"],
            "min_data_in_leaf": best.params["min_data_in_leaf"],
        }
        # Searched for every quantile (all use bagging).
        if "subsample" in best.params:
            best_params["subsample"] = best.params["subsample"]
        logger.info(
            f"  Optuna search ({n_trials} trials): best loss={best.value:.6f} "
            f"params={best_params}"
        )
        return best_params

    # Recovered demand/supply sidecars, joined AFTER voting so none of them
    # votes as a price. Local research dataset only; a missing file is a no-op.
    _SIDECARS = {
        "volume-panel.parquet": ["steam_volume", "steam_sale_median"],
        "bid-panel.parquet": ["buff_bid"],
        "stattrak-panel.parquet": ["st_premium"],
        "supply-history.parquet": ["buff_listing_count"],
    }

    def _attach_sidecars(self, daily: pd.DataFrame) -> pd.DataFrame:
        for fname, cols in self._SIDECARS.items():
            path = self.archive_dir / fname
            if not path.exists():
                continue
            side = pd.read_parquet(path)[["item_id", "date"] + cols]
            daily = daily.merge(side, on=["item_id", "date"], how="left")
            if "steam_volume" in cols:
                # Prefer recovered volume; keep existing where unmatched.
                daily["volume"] = daily["steam_volume"].fillna(daily["volume"])
                daily = daily.drop(columns=["steam_volume"])
        return daily

    def engineer_features(self, price_df: pd.DataFrame,
                          events_df: pd.DataFrame,
                          item_first_dates=None,
                          skip_unused_groups: bool = False) -> pd.DataFrame:
        """Engineer the full feature frame.

        ``item_first_dates`` is an optional item_id -> first-seen date mapping
        for callers that pass a truncated history. Only ``item_age_days`` reads
        it; it is the one feature here whose value depends on how far back the
        frame reaches. Omit it and the frame's own min date is used, which is
        correct for the untruncated training path.

        Every other feature here is a bounded-window computation that a
        sufficiently long tail reproduces, exactly for the row-count and
        calendar windows, and to ~1e-5 relative for the ewm()-based MACD family,
        which has no finite memory. Note this docstring covers
        ``engineer_features`` only: ``_apply_market_aggregates`` is called
        separately and contains a rolling(365) that a PREDICT_TAIL_ITEM_DAYS
        tail does not reproduce — see that constant's comment.
        """
        # Resample to one row per item per day before feature engineering.
        # Raw price_history has multiple rows per day (collection runs every 6h).
        # Without resampling, "lag_1d" is really ~6h and "mean_7d" covers ~2 days.
        # Naming only price/volume here also drops n_ask_sources if the input
        # carries it -- see the matching exclude-set entry in
        # _select_feature_cols, which no longer relies on this being the only
        # thing stopping it from leaking into feature_cols.
        if "date" in price_df.columns:
            daily = price_df.groupby(["item_id", "date"], as_index=False).agg(
                price=("price", "mean"),
                volume=("volume", "sum"),
            )
        else:
            daily = price_df
        daily = self._attach_sidecars(daily)
        # _compute_price_features is never skipped: price_technicals is the one
        # allowlisted group, and the `other` columns are computed inside it.
        skip = self._skipped_feature_groups() if skip_unused_groups else set()
        df = self._compute_price_features(daily)
        if "temporal" not in skip:
            df = self._add_temporal_features(df, item_first_dates=item_first_dates)
        if "item_identity" not in skip:
            df = self._add_item_identity_features(df)
        if "events" not in skip:
            df = self._add_event_features(df, events_df)
        if "item_metadata" not in skip:
            df = self._add_item_metadata_features(df)
        # _feature_group assigns the supply-side columns to item_identity.
        if "item_identity" not in skip:
            df = self._add_supply_side_features(df)
        df = self._add_bymykel_metadata_features(df)
        if "social" not in skip:
            df = self._add_social_features(df)
        return df

    # ------------------------------------------------------------------
    # Target preparation
    # ------------------------------------------------------------------

    # A day on which this fraction of the cross-section repeats the previous
    # day's price EXACTLY is a re-published snapshot, not a market day. Measured
    # over all 4,735 archive days, exactly two fire (2026-07-16 and 2026-07-22,
    # both at 100.00% across ~40k items) and the next-highest day sits at
    # 69.01%, so this threshold lives in a wide empty gap rather than being
    # tuned. Requires MIN_DEGENERATE_CROSS_SECTION items so a quiet day on a
    # handful of items is not mistaken for one.
    SNAPSHOT_DAY_FLAT_FRACTION = 0.99
    MIN_DEGENERATE_CROSS_SECTION = 25
    # A day on which the size of the collected item universe moves by more than
    # this is a collector cutover. Detected from the universe, never from
    # prices: a market-wide crash cannot change how many items a collector
    # returns, so this can never delete a real event. Fires 12 times in 4,735
    # days (4 in 2013 at archive startup, 1 in 2016, 7 in 2026).
    COLLECTION_SHIFT_FRACTION = 0.20

    @classmethod
    def _snapshot_dates(cls, df: pd.DataFrame) -> frozenset:
        """Dates whose prices are a copy of the previous day's.

        Unusable as a label ENDPOINT — the price is stale, so any return
        measured to or from it is fabricated. Harmless mid-window: a copied day
        shifts no level.
        """
        if df.empty or not {"item_id", "date", "price"} <= set(df.columns):
            return frozenset()
        d = df[["item_id", "date", "price"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.groupby(["item_id", "date"], as_index=False)["price"].mean()
        prev = d.copy()
        prev["date"] = prev["date"] + pd.Timedelta(days=1)
        merged = d.merge(prev, on=["item_id", "date"], suffixes=("", "_prev"))
        if merged.empty:
            return frozenset()
        merged["same"] = merged["price"] == merged["price_prev"]
        agg = merged.groupby("date")["same"].agg(["mean", "size"])
        hits = agg[(agg["mean"] >= cls.SNAPSHOT_DAY_FLAT_FRACTION)
                   & (agg["size"] >= cls.MIN_DEGENERATE_CROSS_SECTION)]
        return frozenset(ts.date() for ts in hits.index)

    @classmethod
    def _collection_shift_dates(cls, df: pd.DataFrame) -> frozenset:
        """Dates where the collected item universe changed size abruptly.

        These are source cutovers. The archive is a stitch of source regimes —
        on 2026-03-22 the mean market return reads -31.6% and on 2026-07-09/10
        +17.4% then -17.8%, against ±0.5% on a normal day — and those are basis
        changes between source sets, not price moves.

        A cutover corrupts any label whose window SPANS it, because the anchor
        is quoted on one source basis and the target on another.
        """
        if df.empty or not {"item_id", "date"} <= set(df.columns):
            return frozenset()
        d = df[["item_id", "date"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        counts = d.groupby("date")["item_id"].nunique().sort_index()
        if len(counts) < 2:
            return frozenset()
        prev = counts.shift(1)
        change = (counts - prev).abs() / prev.replace(0, np.nan)
        hits = counts.index[change > cls.COLLECTION_SHIFT_FRACTION]
        return frozenset(ts.date() for ts in hits)

    def prepare_targets(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        logger.info(f"Preparing {horizon}d targets...")
        df = df.sort_values(["item_id", "date"])

        # Date-based target lookup: find the price exactly `horizon` calendar
        # days later, rather than shifting rows. Row-based shift gives
        # incorrect horizons when item data has gaps.
        df["_date_dt"] = pd.to_datetime(df["date"])

        # Shift each row's price BACKWARD by horizon so it becomes the
        # target for the row `horizon` days earlier. E.g., the row at
        # date=Jan 8 with price=P becomes target=P for the row at date=Jan 1.
        future = df[["item_id", "_date_dt", "price"]].copy()
        future.columns = ["item_id", "date", f"target_{horizon}d"]
        future["date"] = future["date"] - pd.Timedelta(days=horizon)
        future["date"] = future["date"].dt.date

        df = df.merge(future, on=["item_id", "date"], how="left")
        df = df.drop(columns=["_date_dt"])

        # The denominator, and it is an arm. Both legs of the return use it --
        # subtracting the raw anchor from a smoothed base would leave the
        # contamination term `(S - p_raw)/S` in the label, which is the quantity
        # being removed. See label_smoothed_anchor_enabled().
        if self.label_smoothed_anchor_enabled():
            base = self._rolling_anchor_prices(df)
        else:
            base = df["price"]
        base = base.replace(0, np.nan)

        # Which rows the two bases agree on, computed BEFORE the arm above can
        # matter and independently of it. This travels with the frame so CV can
        # report a rank IC restricted to the cohort where neither basis carries
        # `p[d]/S[d]`; it is never a feature and never a label.
        df[ANCHOR_TIED_COL] = self._anchor_is_tied(df)
        df[f"target_return_{horizon}d"] = (
            (df[f"target_{horizon}d"] - base) / base * 100
        )

        # The CALIBRATION basis, always the served one, regardless of the arm
        # above. `q_hat` is fitted on residuals to this column and the training
        # label keeps its own denominator.
        #
        # Why they must differ: `predict` quotes every item from
        # `_smoothed_anchor_prices`' span-bounded median, and the backtest
        # resolves `base_price` with the same statistic (`resolve_anchors`,
        # median of the last SMOOTH_WINDOW observations), so the residual
        # production is scored on is `P[d+h]/S[d] - 1 - r_hat`. The label
        # divides by the RAW quote `p[d]`, so a residual measured against it is
        # `P[d+h]/p[d] - 1 - r_hat` -- inflated by the anchor deviation
        # `p[d]/S[d]` on every row where the two disagree, which is most of
        # them. q_hat fitted there is too wide for the basis it is served in,
        # and the band over-covers.
        #
        # This is the DENOMINATOR half of the same incoherence
        # `2026-08-11-conformal-centre-follows-serving.md` fixed for the CENTRE.
        # It is deliberately NOT the `LABEL_SMOOTHED_ANCHOR` arm: that one moves
        # the training label, which hands the model `p[d]/S[d]` as a factor it
        # can read at the anchor, and it was measured and rejected
        # (`2026-08-11-smoothed-anchor-label-measured.md`). q_hat is post-hoc --
        # it changes a band width and nothing the model learns.
        cal_base = (base if self.label_smoothed_anchor_enabled()
                    else self._rolling_anchor_prices(df).replace(0, np.nan))
        df[calibration_target_col(horizon)] = (
            (df[f"target_{horizon}d"] - cal_base) / cal_base * 100
        )
        # Winsorize extreme returns at ±500% to prevent API corruption artifacts
        # from polluting gradient estimates. The audit found 11,044 jumps >1000%,
        # 84% of which revert the next day (definitive corruption).
        winsorized = df[f"target_return_{horizon}d"].clip(-500.0, 500.0)
        n_clipped = (winsorized != df[f"target_return_{horizon}d"]).sum()
        if n_clipped:
            logger.info(
                f"  Winsorized {n_clipped} extreme targets for {horizon}d "
                f"(±500% clip)"
            )
            df[f"target_return_{horizon}d"] = winsorized
        # Same clip on the calibration column, unconditionally: a ±500% outlier
        # that survives into the conformal set moves q_hat directly, and the
        # count above is the label's, not this column's.
        df[calibration_target_col(horizon)] = (
            df[calibration_target_col(horizon)].clip(-500.0, 500.0)
        )

        # Void labels the collector fabricated. Winsorization above cannot catch
        # these: a -31.6% source-cutover return and a 0% re-published return are
        # both well inside the ±500% clip, so they survive as confident,
        # completely wrong labels. See _snapshot_dates / _collection_shift_dates.
        anchor = pd.to_datetime(df["date"])
        # pd.Timedelta(days=<int>) emits a NumPy generic-unit DeprecationWarning;
        # the explicit-unit form does not.
        h_delta = pd.to_timedelta(int(horizon), unit="D")
        bad = pd.Series(False, index=df.index)

        snapshots = self._snapshot_dates(df)
        if snapshots:
            snap = pd.to_datetime(sorted(snapshots))
            # Endpoint rule: the anchor price or the target price is stale.
            bad |= anchor.isin(snap)
            bad |= (anchor + h_delta).isin(snap)

        shifts = self._collection_shift_dates(df)
        if shifts:
            # Span rule: the basis changes somewhere inside (anchor, anchor+h],
            # so the two legs of the return are quoted on different sources.
            for s in sorted(shifts):
                s = pd.Timestamp(s)
                bad |= (anchor < s) & (s <= anchor + h_delta)

        # Frozen-price runs. The per-item analogue of the snapshot rule above:
        # _snapshot_dates voids a day on which the whole CROSS-SECTION repeats
        # yesterday, this voids a row on which THIS ITEM does. Both are the
        # Getmansky-Lo-Makarov MA(k) mechanism — a series that stopped
        # reporting, not a market that stopped moving — and both are invisible
        # to the +/-500% winsorization, because a fabricated 0% return is well
        # inside the clip.
        #
        # ENDPOINT rule, matching the snapshot branch: either leg being stale
        # ruins the return, so both are tested. Measured 2026-08-08 on the >=$1
        # cohort, the anchor leg alone catches only 65-72% of the exact-zero
        # return mass at h=3-14 and 31.2% at h=30; anchor-or-target reaches
        # 86-91% and 52.0%. The residual at h=30 is deliberate and is NOT a gap
        # in the rule — roughly half of 30d zero returns are genuine round
        # trips back to the same price, and those are labels, not artifacts.
        # Keyed on (item, day), never on day alone: staleness is a property of
        # one item's series, and a date-only set would void every item's label
        # on any day some other item happened to be frozen.
        # None disables the rule outright — the control arm in
        # scripts/ab_test_frozen_runs.py, and the only way to reproduce a
        # pre-2026-08-08 label set. Distinct from a large threshold, which
        # would still pay for the scan.
        n_stale = 0
        if (LABEL_MAX_STALE_RUN_DAYS is not None
                and not df.empty and "price" in df.columns):
            runs = stale_run_days(
                df, item_col="item_id", date_col="date", price_col="price")
            anchor_stale = (runs > LABEL_MAX_STALE_RUN_DAYS).to_numpy()

            day = anchor.dt.normalize()
            by_key = pd.Series(
                anchor_stale,
                index=pd.MultiIndex.from_arrays([df["item_id"], day]),
            )
            # max() over the key rather than a plain lookup: the frame is
            # normally voted to one row per item-day, but nothing here enforces
            # it, and a duplicated key must resolve to "stale if any copy is"
            # rather than raising on a non-unique index.
            by_key = by_key.groupby(level=[0, 1]).max()

            target_stale = by_key.reindex(
                pd.MultiIndex.from_arrays([df["item_id"], day + h_delta]),
                fill_value=False,
            ).to_numpy(dtype=bool)

            stale_leg = pd.Series(anchor_stale | target_stale, index=df.index)
            n_stale = int(
                (stale_leg & ~bad
                 & df[f"target_return_{horizon}d"].notna()).sum()
            )
            bad |= stale_leg

        pre_void_na = df[f"target_return_{horizon}d"].isna().sum()
        n_bad = int((bad & df[f"target_return_{horizon}d"].notna()).sum())
        if n_bad:
            df.loc[bad, f"target_return_{horizon}d"] = np.nan
            df.loc[bad, f"target_{horizon}d"] = np.nan
            # Voided on exactly the same rows. A void means the RETURN is
            # fabricated -- a snapshot day, a collector cutover, a frozen run --
            # and that is a property of the price series, not of which anchor
            # the denominator used. Leaving them in the conformal set would fit
            # q_hat on the artifacts the label rules exist to remove.
            df.loc[bad, calibration_target_col(horizon)] = np.nan
            frozen_note = (
                "frozen-run rule DISABLED"
                if LABEL_MAX_STALE_RUN_DAYS is None
                else (f"{n_stale} of them for a frozen price run "
                      f"(> {LABEL_MAX_STALE_RUN_DAYS}d) on either leg")
            )
            logger.info(
                f"  Voided {n_bad} {horizon}d targets spanning "
                f"{len(snapshots)} snapshot day(s) / {len(shifts)} collector "
                f"cutover(s); {frozen_note}"
            )

        voided = int(df[f"target_return_{horizon}d"].isna().sum() - pre_void_na)
        counts = self.label_voiding.get("voided_labels_by_horizon", {})
        counts[horizon] = voided
        # `voided` is the union of three rules -- snapshot days, collector
        # cutovers, and the frozen-price-run rule (`bad |= stale_leg` above)
        # -- and on the >=$1 cohort the frozen-run rule is the dominant term.
        # Publish it separately rather than folding it into `voided`, or a
        # reader attributes a large number to cutovers that cutovers did not
        # cause. Per-horizon like `voided_labels_by_horizon`, since
        # `LABEL_MAX_STALE_RUN_DAYS` and the window shape both vary by h.
        frozen_run_labels = self.label_voiding.get("frozen_run_labels", {})
        frozen_run_labels[horizon] = n_stale
        self.label_voiding = {
            "snapshot_dates": sorted(d.isoformat() for d in snapshots),
            "collection_shift_dates": sorted(d.isoformat() for d in shifts),
            "voided_labels_by_horizon": counts,
            "frozen_run_labels": frozen_run_labels,
            "frame_date_range": [
                pd.to_datetime(df["date"]).min().date().isoformat(),
                pd.to_datetime(df["date"]).max().date().isoformat(),
            ],
        }
        logger.info(
            f"  Label voiding (h={horizon}): {voided:,} labels voided "
            f"({n_stale:,} of them for a frozen price run, "
            f"{len(shifts)} collector cutovers, {len(snapshots)} snapshot days); "
            f"cutovers: {sorted(d.isoformat() for d in shifts)}"
        )
        return df

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _train_ensemble_member(params: dict, dtrain: lgb.Dataset,
                                dval: lgb.Dataset,
                                num_boost_round: int = 1000,
                                early_stopping: bool = False) -> lgb.Booster:
        """Train a single ensemble member on a pre-constructed Dataset.

        Callers MUST call `dtrain.construct()` / `dval.construct()`
        synchronously (single-threaded) before submitting this to a
        ThreadPoolExecutor — LightGBM's Dataset binning is lazy on first
        touch, and concurrent first-touch across threads is a data race.
        Forcing construction up front lets every ensemble member safely
        share one binned Dataset (read-only) instead of re-binning the
        same matrix per member.

        ``early_stopping`` defaults to False: the trailing validation window
        has an effective sample size of ~a dozen dates, so stopping on it is
        noise. See FIXED_BOOST_ROUNDS. When it is off, `dval` is not attached
        as a valid_set at all — nothing reads the per-round metric, and
        evaluating it every round is pure cost.
        """
        callbacks = [lgb.log_evaluation(0)]
        valid_sets = None
        if early_stopping and dval is not None:
            callbacks.insert(0, lgb.early_stopping(50))
            valid_sets = [dval]
        return lgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

    @staticmethod
    def _filter_dead_items(price_df: pd.DataFrame) -> pd.DataFrame:
        """Remove items that never meaningfully move (dead items at Steam floor).

        An item is 'dead' if its max price is <= $0.05 (Steam floor) AND its
        lifetime price range is < 5%. These items constitute ~41% of training
        rows but provide zero predictive signal — they dilute the model and
        waste gradient steps on constant targets.

        Also filters items where mean_price <= 0 (division-by-zero safety).
        """
        pre = len(price_df)
        price_df = price_df[price_df["price"] > 0].copy()
        if pre != len(price_df):
            logger.info(f"  Removed {pre - len(price_df)} rows with zero/negative price")

        item_stats = price_df.groupby("item_id")["price"].agg(["min", "max"])
        dead_mask = (
            (item_stats["max"] <= 0.05)
            & ((item_stats["max"] - item_stats["min"]) / item_stats["min"] < 0.05)
        )
        dead_items = set(item_stats[dead_mask].index)
        if not dead_items:
            return price_df

        filtered = price_df[~price_df["item_id"].isin(dead_items)].copy()
        removed_rows = len(price_df) - len(filtered)
        logger.info(
            f"  Filtered {len(dead_items)} dead items "
            f"({removed_rows:,} rows, {removed_rows / max(len(price_df), 1) * 100:.1f}%)"
        )
        return filtered

    @staticmethod
    def _filter_by_median_price(price_df: pd.DataFrame,
                                min_median_price: float) -> pd.DataFrame:
        """Restrict the training universe to items whose median price clears
        ``min_median_price``, measured on the voted consensus price.

        Training applies no price filter by default, so the universe is the
        pool's tier mix: 82.6% of item-days are sub-$1 while production serves
        only >= $1 (``api/serving_policy.py::MIN_SERVED_PRICE_USD``). That
        matters less as a cohort-mismatch argument — reweighting the classifier
        toward the served cohort was measured and refuted — than as a *budget*
        argument. ``_stratified_item_subsample`` spends its row budget on the
        pool, so at the 100K default only ~20 of the ~99 selected items are in
        the served cohort, and 44% of the universe is stickers and graffiti at
        a $0.03 median.

        Filtering first makes the budget buy served-cohort breadth instead:
        measured over the 1460-day window, the >= $1 cohort is 926 items /
        993,464 item-days at 1,072.9 rows/item — essentially the pool's density,
        so the floor costs no history. A budget above ~1.0M therefore covers it
        with no subsampling at all, which removes the item-draw variance the
        subsample otherwise injects (sd 1.5-3.1pp on
        ``mean_classifier_acc_ge1``).

        Median, not mean, and computed over the whole window: it is the same
        statistic the cohort was sized on, and it does not let one spike
        promote a penny item.
        """
        item_median = price_df.groupby("item_id")["price"].median()
        keep = set(item_median[item_median >= min_median_price].index)
        if len(keep) == len(item_median):
            return price_df

        out = price_df[price_df["item_id"].isin(keep)].copy()
        logger.info(
            f"  Median-price floor >= ${min_median_price:g}: "
            f"{len(keep):,}/{len(item_median):,} items, "
            f"{len(out):,}/{len(price_df):,} rows"
        )
        return out

    @staticmethod
    def _fold_median_price_items(price_df: pd.DataFrame,
                                 min_median_price: float,
                                 cutoff) -> set:
        """Items whose median price over ``date < cutoff`` clears the floor.

        The look-ahead-free counterpart to `_filter_by_median_price`, which
        takes the median over the **whole** frame and so hands every fold a
        universe nobody could name at that fold's decision time: an item that
        was a penny sticker in 2019 and $50 in 2025 enters a 2019 fold
        *because it later rose*. This selects on the pre-cutoff window only.

        ``cutoff`` must be ``val_start - embargo_days(horizon)``, never
        ``val_start``. The selection statistic is itself a function of prices,
        so computed up to the boundary it reads the embargo window — the same
        leak in a smaller form. Never pass a bare horizon; see
        `.claude/rules/labels-and-embargo.md`.

        An empty pre-cutoff window returns an empty set, not everything. A
        fold with no history to select on has no defensible universe, and
        falling back to "keep all" would silently restore the full-sample
        behaviour on exactly the early folds where the leak is largest.

        Research use only: `build_training_data` cannot call this, because a
        fold-varying universe would move the market factor and the row budget
        alongside the treatment. See
        `docs/superpowers/plans/2026-08-08-per-fold-price-filter.md`.
        """
        cutoff = pd.Timestamp(cutoff)
        past = price_df[pd.to_datetime(price_df["date"]) < cutoff]
        if past.empty:
            return set()
        item_median = past.groupby("item_id")["price"].median()
        return set(item_median[item_median >= min_median_price].index)

    def _flag_corrupt_items(self, price_df: pd.DataFrame,
                            jump_threshold: float = 500.0,
                            max_jumps: int = 10) -> set:
        """Identify items with frequent extreme price jumps (API corruption).

        Counts how many times each item's daily price jumps exceed
        ``jump_threshold`` percent. Items with more than ``max_jumps`` such
        events are flagged as corrupt and excluded from training.

        The audit found 84% of >1000% jumps revert the next day — definitive
        API corruption, not real market movement. 905 items are affected,
        151 with 10+ events.
        """
        pdf = price_df.sort_values(["item_id", "date"]).copy()
        pdf["_prev"] = pdf.groupby("item_id")["price"].shift(1)
        pdf["_pct"] = (pdf["price"] - pdf["_prev"]) / pdf["_prev"].replace(0, np.nan) * 100
        is_first = pdf.groupby("item_id").cumcount() == 0
        pdf.loc[is_first, "_pct"] = 0.0

        bad = pdf.groupby("item_id")["_pct"].apply(
            lambda s: int((s.abs() > jump_threshold).sum())
        )
        bad_items = set(bad[bad > max_jumps].index)
        if bad_items:
            logger.info(
                f"  Flagged {len(bad_items)} corrupt items "
                f"(>{max_jumps} jumps >{jump_threshold:.0f}%)"
            )
        return bad_items

    def _stratified_item_subsample(self, price_df: pd.DataFrame,
                                   max_rows: int, seed: int = 42,
                                   exclude_items: set = None) -> pd.DataFrame:
        """Subsample whole-item histories to bound the row count *before*
        feature engineering, while preserving per-item time-series continuity
        and the full calendar window.

        The old approach capped rows via ``train_set.tail(max_rows)`` *after*
        feature engineering. As the archive grew, that kept only the most
        recent ~51 calendar days (dropping 93% of voted rows) which silently
        disabled expanding-window CV and made the weekly retrain OOM because
        ``engineer_features`` still ran on all ~2.9M rows.

        This selects entire item histories (not individual rows) so lag/rolling
        features stay valid, stratified by rarity so rare items (knives, gloves)
        are retained proportionally, and keeps every calendar date intact so CV
        has enough distinct dates.
        """
        if exclude_items:
            pre = len(price_df)
            price_df = price_df[~price_df["item_id"].isin(exclude_items)].copy()
            if len(price_df) != pre:
                logger.info(f"  Excluded {pre - len(price_df)} corrupt-item rows")

        total_rows = len(price_df)
        if total_rows <= max_rows or "item_id" not in price_df.columns:
            return price_df

        rows_per_item = price_df.groupby("item_id").size()
        n_items = len(rows_per_item)
        avg_rows = total_rows / max(n_items, 1)
        target_items = max(1, int(max_rows / max(avg_rows, 1.0)))
        if target_items >= n_items:
            return price_df

        meta = self._fetch_supply_metadata()
        rarity_map = {}
        if meta is not None and not meta.empty and "rarity" in meta.columns:
            rarity_map = dict(zip(meta["item_id"], meta["rarity"].fillna("unknown")))

        items = pd.DataFrame({"item_id": rows_per_item.index})
        items["rarity"] = items["item_id"].map(rarity_map).fillna("unknown")

        rng = np.random.RandomState(seed)
        selected: List = []
        for _rarity, group in items.groupby("rarity"):
            frac = len(group) / n_items
            k = min(len(group), max(1, int(round(target_items * frac))))
            selected.extend(group["item_id"].sample(n=k, random_state=rng).tolist())

        selected_set = set(selected)
        out = price_df[price_df["item_id"].isin(selected_set)].copy()
        logger.info(
            f"  Stratified subsample: {len(selected_set):,}/{n_items:,} items, "
            f"{len(out):,}/{total_rows:,} rows (budget {max_rows:,}); "
            f"full calendar window preserved"
        )
        return out

    @staticmethod
    def _per_item_row_sample(train_set: pd.DataFrame, max_rows: int,
                             seed: int = 42) -> pd.DataFrame:
        """Spend a row budget on item breadth: an equal quota per item.

        The alternative in ``_build_production_split`` is
        ``train_set.sample(n=max_rows)``, a uniform draw in which an item's
        share of the sample is its share of the rows. Under a median-price
        floor that is the wrong axis. Measured 2026-08-07 on the paired
        harness, ``ge1_budgeted`` (71K rows/fold) beat ``ge1_full`` (728K)
        by +5.72pp against +3.50pp at 30d — ten times the rows losing at both
        horizons, which replicates the breadth harness's ``wide_unbudgeted``
        finding. What pays is item diversity per row, not row count.

        That arm could not be expressed in production because every sampler
        upstream selects *whole item histories*: a 110K budget under the floor
        buys ~93 items at full depth, never 728 items at ~98 rows each. This
        is the missing half — ``_stratified_item_subsample`` picks which items,
        this picks how deeply each is drawn.

        Rows are drawn uniformly at random *within* each item rather than from
        its tail, so the full calendar window survives; a ``tail()`` cap
        silently disabled expanding-window CV once already
        (``2026-07-16-training-window-audit.md``). Mirrors
        ``scripts/ab_test_training_breadth.py::_stratified_sample``, which is
        the implementation the +5.72pp was measured on.

        Safe to run after feature engineering only. Thinning rows before
        ``engineer_features`` would compute lags over a punctured series, and
        before ``prepare_targets`` would void the labels of any row whose
        ``date + horizon`` partner was dropped.
        """
        if len(train_set) <= max_rows or "item_id" not in train_set.columns:
            return train_set

        groups = train_set.groupby("item_id", sort=True).indices
        per_item = max(1, max_rows // len(groups))
        rng = np.random.default_rng(seed)

        keep = []
        for _item_id, idx in groups.items():
            if len(idx) <= per_item:
                keep.append(idx)
            else:
                keep.append(rng.choice(idx, size=per_item, replace=False))
        picked = np.concatenate(keep)
        picked.sort()
        out = train_set.iloc[picked]

        if len(out) > max_rows:
            # Only reachable when items outnumber the budget, where the
            # one-row-per-item floor cannot be honoured for everyone. max_rows
            # is a memory guard upstream, so it wins over the quota.
            out = out.iloc[rng.choice(len(out), size=max_rows, replace=False)]

        logger.info(
            f"  Per-item row sample: {len(groups):,} items x {per_item:,} rows "
            f"-> {len(out):,}/{len(train_set):,} rows (budget {max_rows:,})"
        )
        return out.sort_values("date")

    def build_training_data(self, days_back: int = 365,
                             backfilled_only: bool = False,
                             max_feature_rows: int = 100_000,
                             min_median_price: Optional[float] = None) -> pd.DataFrame:
        _t0 = datetime.now()
        price_df = self.fetch_price_history(days_back=days_back, backfilled_only=backfilled_only)
        logger.info(f"  fetch_price_history took {(datetime.now() - _t0).total_seconds():.0f}s")
        price_df = self._filter_dead_items(price_df)
        # Before the subsample, so the row budget is spent on the surviving
        # universe rather than on the pool. See _filter_by_median_price.
        if min_median_price:
            price_df = self._filter_by_median_price(price_df, min_median_price)
        corrupt_items = self._flag_corrupt_items(price_df)

        # NOTE: A distribution-shift guard here previously excluded ALL 2026
        # rows. It was a temporary patch for the May–June 2026 archive gap,
        # which made 2026 sparse (some single-day, ~352-item slices where the
        # 7d validation window collapsed to noise). That gap is now backfilled
        # (2026 is continuous, ~5,360 items/day, zero <50-item days), so the
        # guard was removed to let training/CV cover the current regime and
        # add walk-forward folds through 2026. See the retrain changelog for
        # the dry-run coverage evidence.

        if max_feature_rows:
            price_df = self._stratified_item_subsample(
                price_df, max_feature_rows, exclude_items=corrupt_items
            )
        else:
            price_df = price_df[~price_df["item_id"].isin(corrupt_items)].copy()

        _t1 = datetime.now()
        events_df = self.fetch_events()
        logger.info(f"  fetch_events took {(datetime.now() - _t1).total_seconds():.0f}s")

        _t2 = datetime.now()
        # Only the training path skips: the ab_test_* harnesses build their own
        # frame and need the full 123 columns to call _apply_feature_allowlist
        # on. Measured 2026-08-09: 8.5s of engineer_features' 17.5s is blocks
        # the allowlist then discards.
        skip = self._skipped_feature_groups()
        df = self.engineer_features(price_df, events_df,
                                    skip_unused_groups=True)
        logger.info(f"  engineer_features took {(datetime.now() - _t2).total_seconds():.0f}s, "
                    f"result: {len(df):,} rows, {len(df.columns)} cols")
        del price_df, events_df

        # Cross-sectional and supply-depth are applied here rather than inside
        # engineer_features, so they carry their own guards off the same
        # derived skip set.
        _t3 = datetime.now()
        if "cross_sectional" not in skip:
            df = self._add_cross_sectional_features(df)
            logger.info(f"  cross_sectional_features took "
                        f"{(datetime.now() - _t3).total_seconds():.0f}s")

        if "supply_depth" not in skip:
            df = self._add_supply_depth_features(df)

        if self.TIER_LEAD_GROUP not in skip:
            df = self._add_tier_lead_features(df)

        # Define feature columns (exclude metadata and target columns)
        self.feature_cols = self._select_feature_cols(
            df, self.HORIZONS, self._active_shelved_features())

        # Restrict to the allowlisted groups and prune correlated columns,
        # in the cheaper order.
        self._reduce_feature_cols(df)
        self._base_feature_cols = list(self.feature_cols)

        # After the allowlist and the prune, so the transform runs over the ~33
        # surviving columns rather than all 123. Two consequences worth stating:
        # the >0.95 correlation prune therefore decides on RAW values, and
        # feature_medians (computed later, at :4034) is the median of the
        # TRANSFORMED column -- which is what serving needs, since the booster is
        # fitted on ranks.
        if self.cross_sectional_rank_enabled():
            skipped: List[str] = []
            df = self._apply_cross_sectional_ranks(df, self.feature_cols,
                                                   skipped_out=skipped)
            # Recorded, and mirrored onto the artifact fields for the same
            # reason as the cohort floor: after training, this process IS the
            # artifact, and predict has to follow this run's decision rather
            # than whatever load_models() read beforehand.
            self._train_xs_rank_skipped = skipped
            self._artifact_xs_rank_skipped = skipped

        # Downcast features to float32 to halve feature matrix memory
        for col in self.feature_cols:
            if col in df.columns and df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

        logger.info(f"Feature matrix: {len(df):,} rows, {len(self.feature_cols)} features")
        logger.info(f"Features: {self.feature_cols}")

        return df

    # ------------------------------------------------------------------
    # Voted price frame cache (shared by train, predict, and backtests)
    # ------------------------------------------------------------------

    @staticmethod
    def _voted_cache_enabled() -> bool:
        return os.getenv("VOTED_CACHE", "1") != "0"

    def _archive_fingerprint(self) -> str:
        """Identify the archive by each file's row count and byte size.

        Was `name:st_size:st_mtime_ns`, which made the cache structurally
        CI-hostile: CI checks the archive out fresh every run, so every mtime
        was new and the key changed unconditionally. The miss costs ~48s a run
        (21.1s DuckDB read + 27.2s voting), paid by the daily predict path too,
        not just by the retrain. Dropping mtime is the whole fix — st_size was
        always content-derived and survives a checkout byte-identically.

        Names NO column. The archive's schema is not uniform (prices-2026-03
        and -04 carry min_price/max_price the other 19 files do not), and the
        date column is `day`, not `date` — a first attempt at this keyed on
        MAX(date) and died in the retrain with a binder error. Row count comes
        from the Parquet footer, so this stays a metadata read.

        Only prices-*.parquet feeds the voted frame — ops/ artifacts are
        rewritten by the pipeline on every run and must not invalidate it.
        """
        import duckdb
        parts = []
        con = duckdb.connect()
        try:
            for path in sorted(self.archive_dir.glob("prices-*.parquet")):
                (n,) = con.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)",
                    [str(path)]).fetchone()
                parts.append(f"{path.name}:{n}:{path.stat().st_size}")
        finally:
            con.close()
        return "|".join(parts)

    def _voted_cache_key(self, days_back: int, backfilled_only: bool,
                         backfilled_slugs: Optional[set]) -> str:
        """Everything the voted frame depends on, hashed.

        The query window is keyed by its resolved cutoff date rather than
        ``days_back`` so a day rollover invalidates the entry — the frame is
        anchored to a calendar date, not to a relative offset.
        """
        cutoff = (self._now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        slug_digest = ""
        if backfilled_slugs is not None:
            # Sorted: the slug set arrives from an unordered DB query.
            slug_digest = hashlib.sha256(
                "|".join(sorted(str(s) for s in backfilled_slugs)).encode()
            ).hexdigest()[:16]
        payload = "\n".join([
            f"v={self.VOTED_CACHE_VERSION}",
            f"cutoff={cutoff}",
            # Without this a replay reuses the live frame, which holds every
            # row after the anchor -- the leak the upper bound exists to stop.
            f"anchor={self.replay_anchor() or ''}",
            f"backfilled_only={int(backfilled_only)}",
            f"slugs={slug_digest}",
            f"archive={self._archive_fingerprint()}",
        ])
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def _voted_cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{self.VOTED_CACHE_PREFIX}{key}.parquet")

    def _load_voted_cache(self, key: str) -> Optional[pd.DataFrame]:
        if not self._voted_cache_enabled():
            return None
        path = self._voted_cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"  Voted cache at {path} unreadable ({e}) — rebuilding")
            return None
        if df.empty:
            # An empty frame here would train the next model on zero rows.
            logger.warning(f"  Voted cache at {path} is empty — rebuilding")
            return None
        logger.info(f"  Voted cache HIT ({len(df):,} rows, {df.item_id.nunique():,} "
                    f"items) — skipping DuckDB read + multi-source voting")
        return df

    def _save_voted_cache(self, key: str, df: pd.DataFrame):
        if not self._voted_cache_enabled():
            return
        path = self._voted_cache_path(key)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            df.to_parquet(path, index=False)
            logger.info(f"  Saved voted cache ({len(df):,} rows) to {path}")
            self._prune_voted_cache()
        except Exception as e:
            # A cache write failure must never fail a retrain.
            logger.warning(f"  Could not write voted cache to {path}: {e}")

    def _prune_voted_cache(self):
        """Keep only the newest few entries.

        Each is hundreds of MB on the real archive and the aggregator changes
        the fingerprint daily, so entries would otherwise accumulate without
        bound.
        """
        entries = sorted(
            Path(self.cache_dir).glob(f"{self.VOTED_CACHE_PREFIX}*.parquet"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in entries[self.VOTED_CACHE_MAX_ENTRIES:]:
            try:
                stale.unlink()
                logger.info(f"  Pruned stale voted cache {stale.name}")
            except OSError as e:
                logger.warning(f"  Could not prune {stale}: {e}")

    # ------------------------------------------------------------------
    # Feature cache for predict speed
    # ------------------------------------------------------------------

    @property
    def _engineered_cache_path(self) -> str:
        return os.path.join(self.model_dir, self.ENGINEERED_CACHE_NAME)

    def _save_engineered_cache(self, df: pd.DataFrame):
        """Save the fully-engineered feature DataFrame to Parquet cache.

        `ENGINEERED_CACHE=0` suppresses the write, for callers that cannot
        benefit from it. CI is the case: `price-forecast.yml` excludes this file
        from both the cache save and the restore (it is ~2GB against a 10GB
        per-repo Actions cache budget), so every run misses, re-engineers, and
        serializes a frame the next run will never read. Nothing reads it back
        in-process — `predict` only consults it before engineering — so
        suppressing the write costs the run nothing. Mirrors `VOTED_CACHE=0`.
        """
        path = self._engineered_cache_path
        if os.environ.get("ENGINEERED_CACHE") == "0":
            logger.info(f"  Engineered feature cache write skipped (ENGINEERED_CACHE=0)")
            return
        df.attrs["_cache_date"] = str(date.today())
        df.attrs["_cache_version"] = self.ENGINEERED_CACHE_VERSION
        logger.info(f"  Saving engineered feature cache ({len(df):,} rows) to {path}")
        df.to_parquet(path, index=False)

    def _load_engineered_cache(self) -> Optional[pd.DataFrame]:
        """Load cached engineered features. Returns None if cache is missing or stale."""
        if self.replay_anchor() is not None:
            # A live cache holds features engineered from rows after the
            # anchor. Its key fingerprints forecaster.py, which a replay does
            # not change, so it would be a perfectly valid hit carrying exactly
            # the data the upper bound exists to exclude.
            logger.info("  REPLAY_ANCHOR set — ignoring the engineered cache.")
            return None
        path = self._engineered_cache_path
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
            cache_date_str = df.attrs.get("_cache_date", "")
            if df.empty:
                logger.warning(f"  Cache at {path} is empty (0 rows) — will refresh")
                return None

            # Version before staleness: a v1 cache is full history, not a tail,
            # and is wrong regardless of how fresh it is.
            version = int(df.attrs.get("_cache_version", 1))
            if version != self.ENGINEERED_CACHE_VERSION:
                logger.info(
                    f"  Cache at {path} is v{version}, expected "
                    f"v{self.ENGINEERED_CACHE_VERSION} — will refresh"
                )
                return None
            logger.info(f"  Loaded engineered feature cache from {path} "
                        f"({len(df):,} rows, cache_date={cache_date_str})")

            # Check staleness: if cache is older than 3 days, trigger refresh
            if cache_date_str:
                try:
                    cache_date = date.fromisoformat(cache_date_str)
                    days_stale = (date.today() - cache_date).days
                    if days_stale > 3:
                        logger.info(f"  Cache is {days_stale} days stale (>3), will refresh")
                        return None
                except (ValueError, TypeError):
                    pass

            # For legacy caches without attrs (pre-cache-refactor), fall back to
            # DuckDB freshness check against the Parquet archive.
            if not cache_date_str:
                try:
                    import duckdb
                    archive_dir = self.archive_dir
                    if archive_dir.exists():
                        with duckdb.connect() as con:
                            pq_files = sorted([str(p) for p in archive_dir.glob("prices-*.parquet")])
                            if pq_files:
                                latest_archive = con.sql(
                                    "SELECT MAX(day) FROM read_parquet(?)",
                                    params=[pq_files[-1]]
                                ).fetchone()[0]
                                cache_max_date = df["date"].max() if "date" in df.columns else None
                                if cache_max_date is not None and latest_archive is not None:
                                    archive_max = pd.to_datetime(latest_archive).date()
                                    cache_max = pd.to_datetime(cache_max_date).date() if not isinstance(cache_max_date, date) else cache_max_date
                                    days_diff = (archive_max - cache_max).days
                                    if days_diff > 3:
                                        logger.info(f"  Cache max_date={cache_max} < archive max_date={archive_max} ({days_diff}d diff), refreshing")
                                        return None
                except Exception:
                    pass

            return df
        except Exception as e:
            logger.warning(f"  Failed to load feature cache: {e}")
            return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _get_feature_importance(self, model: lgb.Booster) -> pd.DataFrame:
        importance = model.feature_importance(importance_type="gain")
        feature_names = model.feature_name()
        if len(feature_names) != len(importance):
            feature_names = self.feature_cols[:len(importance)]
        fi = pd.DataFrame({"feature": feature_names, "importance": importance})
        fi = fi.sort_values("importance", ascending=False).head(20)
        return fi

    @staticmethod
    def _compute_sample_weights(tdf: pd.DataFrame, horizon: int) -> Optional[np.ndarray]:
        """Compute sample weights proportional to item price variance.

        Items that move more get higher gradient weight; flat/dead items
        get down-weighted. Uses 30-day rolling std of daily returns as the
        weight signal, clipped to [0.1, 99th percentile].

        Positive-return samples are additionally upweighted by DIRECTION_UPWEIGHT
        to counter the model's conservative bias (underpredicts "up" by ~2×).

        Weights then decay with row age at SAMPLE_WEIGHT_HALFLIFE_DAYS, so the
        current market regime dominates the gradient instead of being outvoted
        by four years of calmer history. See that constant for the measurements.

        This prevents the ~41% of historically flat items from dominating
        the loss even after dead-item filtering removes the extreme cases.
        """
        if tdf.empty or "price" not in tdf.columns:
            return None
        vol = tdf.groupby("item_id", group_keys=False)["price"].transform(
            lambda x: x.pct_change().rolling(30, min_periods=5).std()
        ).fillna(1.0).values
        vol = np.clip(vol, 0.1, np.percentile(vol, 99))

        target_col = f"target_return_{horizon}d"
        if target_col in tdf.columns and DIRECTION_UPWEIGHT != 1.0:
            target_vals = tdf[target_col].values
            direction_mult = np.where(target_vals > 0, DIRECTION_UPWEIGHT, 1.0)
            vol = vol * direction_mult

        # Recency decay. Age is measured against the newest row in THIS frame,
        # not today, so CV folds and the final fit are each internally
        # consistent (a fold ending in 2023 is not uniformly crushed).
        if SAMPLE_WEIGHT_HALFLIFE_DAYS and "date" in tdf.columns:
            dates = pd.to_datetime(tdf["date"])
            age_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
            vol = vol * np.power(0.5, age_days / float(SAMPLE_WEIGHT_HALFLIFE_DAYS))

        vol = vol / max(np.mean(vol), 1e-8)
        return vol.astype(np.float32)

    def train(self, max_rows: int = 300_000,
              max_feature_rows: int = 1_200_000,
              min_median_price: Optional[float] = 1.0,
              per_item_row_sampling: bool = False):
        logger.info("=" * 60)
        logger.info("TRAINING LIGHTGBM FORECASTER (ensemble, HP search, walk-forward)")
        logger.info("=" * 60)

        _train_start = datetime.now()
        # Two separate budgets, previously conflated into one that was then
        # discarded. max_feature_rows bounds the frame BEFORE feature
        # engineering and therefore decides how many whole item histories the
        # model learns from; max_rows caps each horizon's slice AFTER it. Only
        # the latter was ever passed, so build_training_data silently kept its
        # own 100_000 default and the caller's 700_000 did nothing.
        #
        # They are deliberately not unified, even though production now runs
        # both at 1_200_000: feeding max_rows to both would make one number
        # move coverage and the per-horizon cap at once, and the caller sets
        # them from two separate constants for that reason.
        #
        # max_feature_rows and min_median_price, by contrast, ARE one setting
        # and default together. At the $1 floor the served cohort is 926 items
        # / 993,464 item-days, so this budget covers it with no subsample —
        # which is the point, because _stratified_item_subsample's seed alone
        # moves mean_classifier_acc_ge1 by sd 1.5-3.1pp. Raising the budget
        # without the floor spends 12x the wall-clock on the pool's tier mix;
        # setting the floor without the budget leaves a smaller draw rather
        # than none. See docs/changelog/2026-08-08-training-price-floor-shipped.md
        # and docs/changelog/2026-08-04-minimal-model-results.md.
        df = self.build_training_data(days_back=1460, backfilled_only=True,
                                      max_feature_rows=max_feature_rows,
                                      min_median_price=min_median_price)

        # Recorded into the artifact because the rank transform's output is a
        # function of WHICH items are in the cross-section, and predict's frame
        # is not this one. See _reference_cohort_mask.
        self._train_min_median_price = min_median_price
        self._train_cohort_items = int(df["item_id"].nunique())
        # After training, the in-memory model IS the artifact, so the predict
        # path's cohort lookup has to see this run's floor rather than the one
        # belonging to whatever was loaded from cache beforehand. Without this,
        # train-then-predict in one process reads a stale (or absent) floor and
        # either refuses to serve or ranks against the wrong cohort.
        self._artifact_min_median_price = min_median_price
        self._artifact_cohort_items = self._train_cohort_items

        self.horizon_feature_cols = {}

        # Sigma clip bounds come from the cross-sectional distribution of the
        # TRAINING frame, then are frozen into the artifact. q_hat is
        # calibrated against clipped sigmas, so serving must clip identically.
        # Measured once here, before any horizon calibrates, so every horizon's
        # q_hat and every served row share one clip.
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_raw = (df["price_std_60d"].to_numpy(dtype=float)
                         / df["price"].to_numpy(dtype=float))
        floor, cap = conformal.sigma_bounds(sigma_raw)
        finite = sigma_raw[np.isfinite(sigma_raw) & (sigma_raw > 0)]
        self.sigma_clip = {
            "floor": floor,
            "cap": cap,
            "fallback": float(np.median(finite)),
        }
        logger.info(
            f"Sigma clip: floor={floor:.5f} cap={cap:.5f} "
            f"fallback={self.sigma_clip['fallback']:.5f}"
        )

        for hi, horizon in enumerate(self.HORIZONS, 1):
            self._train_horizon_inline(
                horizon, df, max_rows,
                per_item_row_sampling=per_item_row_sampling)

        del df

        _train_elapsed = (datetime.now() - _train_start).total_seconds()
        self.save_models()
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETE in {_train_elapsed:.0f}s ({_train_elapsed/60:.1f}min)")
        logger.info(f"{'='*60}")
        logger.info(f"  [timing] TOTAL training: {_train_elapsed:.1f}s")

    def _train_horizon_inline(self, horizon: int, df: pd.DataFrame,
                                max_rows: int = 300_000,
                                per_item_row_sampling: bool = False):
        logger.info(f"\n{'='*60}")
        logger.info(f"HORIZON {horizon}d")
        logger.info(f"{'='*60}")
        _hz_start = datetime.now()

        # Reset to the full correlation-pruned base set before each horizon.
        # Without this, 3d's permutation-importance pruning permanently
        # removes features from self.feature_cols, starving 7d/14d/30d
        # of features they might have used.
        self.feature_cols = list(self._base_feature_cols)

        tdf = self.prepare_targets(df, horizon)

        # Drop NaN targets (use percentage return as primary target)
        tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
        tdf = tdf.sort_values("date")

        # Assign regime labels for regime-switching training
        tdf["_regime"] = self._assign_regime_labels(tdf)

        for attempt in range(2):
            if attempt == 1:
                logger.info(f"  Retry {horizon}d — pruned feature groups")

            # Temporal walk-forward split: a trailing calendar window, purged of
            # the training rows it labels. See the helper.
            train_set, val_set = self._build_production_split(
                tdf, horizon, max_rows,
                per_item_row_sampling=per_item_row_sampling)

            if attempt == 0:
                logger.info(f"  {horizon}d: {len(train_set)} train, {len(val_set)} val")

            # Replace INF with NaN before imputation (division-by-zero artifacts)
            X_train_pre = train_set[self.feature_cols].replace([np.inf, -np.inf], np.nan)
            feature_medians = X_train_pre.median()
            self.feature_medians = feature_medians
            X_train = X_train_pre.fillna(feature_medians)
            y_train = train_set[f"target_return_{horizon}d"]

            X_val = val_set[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(feature_medians)
            y_val = val_set[f"target_return_{horizon}d"]

            # Sample weights: down-weight historically flat items so the
            # model focuses on items with meaningful price movement.
            train_weights = self._compute_sample_weights(train_set, horizon)
            val_weights = self._compute_sample_weights(val_set, horizon)

            # Build the binned Dataset once per horizon and reuse it across
            # all quantiles and ensemble members. X/y and binning are
            # identical for every quantile (only the objective's alpha
            # changes), so rebuilding per fit just re-bins the same matrix.
            ds_params = {"max_bin": self.MAX_BIN, "feature_pre_filter": False}
            dtrain_kw = dict(params=ds_params)
            if train_weights is not None:
                dtrain_kw["weight"] = train_weights
            dval_kw = dict(params=ds_params)
            if val_weights is not None:
                dval_kw["weight"] = val_weights
            # N1: boost from `-return_1d` when the instrument is on, so the
            # boosters fit the residual to the naive predictor. Set on `dval`
            # too, or the early-stopping/Optuna metric scores the residual
            # against the un-offset target. Every consumer of these models'
            # output has to add the offset back -- see _naive_offset_served.
            train_offset = self._naive_offset(train_set)
            val_offset = self._naive_offset(val_set)
            if train_offset is not None:
                dtrain_kw["init_score"] = train_offset
                dval_kw["init_score"] = val_offset
            dtrain = lgb.Dataset(X_train, y_train, **dtrain_kw)
            dval = lgb.Dataset(X_val, y_val, reference=dtrain, **dval_kw)
            # Construct eagerly (single-threaded) so the ensemble
            # ThreadPoolExecutor below only ever reads an already-binned
            # Dataset — construction is where the lazy-init race lives.
            dtrain.construct()
            dval.construct()

            per_quantile_params = {}

            boosting_type = self.BOOSTING_TYPE

            _hp_start = time.time()
            cached_hp = self.tuned_params.get(horizon, {})
            reuse_hp = (os.environ.get("FORCE_HP_SEARCH") != "1"
                        and all(q in cached_hp for q in self.QUANTILES))
            if reuse_hp:
                # HP reuse only. CV still runs: it is where the conformal
                # calibration records come from, and they have to be unseen.
                logger.info(f"  Reusing cached HP for {horizon}d (Optuna skipped)...")
                _warm_retrain = True
                if train_offset is not None:
                    logger.warning(
                        f"  ⚠ {horizon}d is boosting from -return_1d "
                        f"(NAIVE_INIT_SCORE=1) on hyperparameters selected "
                        f"against the UN-offset target. Read the sign of "
                        f"rank_ic_edge from this run, but confirm the size with "
                        f"FORCE_HP_SEARCH=1 before believing it.")
                for q in self.QUANTILES:
                    bp = dict(cached_hp[q])
                    bp["max_bin"] = self.MAX_BIN
                    bp["feature_pre_filter"] = False
                    bp["device"] = "cuda" if _gpu_available() else "cpu"
                    bp["boosting_type"] = boosting_type
                    # Rewrites row sampling from the current strategy and drops
                    # any GOSS keys a pre-2026-07-29 meta.json cached for q50.
                    self._apply_row_sampling(bp, q)
                    per_quantile_params[q] = bp
            else:
                _warm_retrain = False
                skip_hp = horizon in self.SKIP_HP_HORIZONS
                hz_trials = self.N_TRIALS_MAP.get(horizon, 15)
                device = "cuda" if _gpu_available() else "cpu"

                base_params_by_q = {}
                for q in self.QUANTILES:
                    base_params_by_q[q] = {
                        "device": device,
                        "feature_pre_filter": False,
                        "objective": "quantile",
                        "alpha": q,
                        "metric": "quantile",
                        "boosting_type": boosting_type,
                        "min_gain_to_split": 0.1,
                        "feature_fraction": 0.7,
                        "max_bin": self.MAX_BIN,
                        "verbosity": -1,
                        "n_jobs": -1,
                    }
                    self._apply_row_sampling(base_params_by_q[q], q)

                if skip_hp:
                    logger.info(f"  Skipping HP search for {horizon}d (SKIP_HP_HORIZONS)...")
                    best_params_by_q = {q: None for q in self.QUANTILES}
                else:
                    # NOTE: previously tried running the 3 quantiles' Optuna
                    # studies concurrently via a ThreadPoolExecutor. That
                    # deadlocked in practice — nesting a thread pool (this
                    # one) around per-trial LightGBM training, which itself
                    # runs in another nested thread pool with OpenMP
                    # (n_jobs>1), hung one horizon's worker process
                    # indefinitely (0s of CPU time over 15+ minutes) on
                    # macOS/libomp. Reverted to sequential search — still
                    # correct and safe, just not faster.
                    best_params_by_q = {}
                    for q in self.QUANTILES:
                        logger.info(f"  Searching hyperparams for {horizon}d p{int(q*100)} "
                                    f"(Optuna, {boosting_type}, {hz_trials} trials)...")
                        best_params_by_q[q] = self._optuna_search_params(
                            X_train, y_train, X_val, y_val,
                            val_dates=val_set["date"],
                            quantile=q,
                            boosting_type=boosting_type,
                            n_trials=hz_trials,
                            horizon=horizon,
                            # Or the search would select hyperparameters for a
                            # target the fit below does not have.
                            train_offset=train_offset,
                            val_offset=val_offset,
                        )

                for q in self.QUANTILES:
                    base_params = base_params_by_q[q]
                    best_params = best_params_by_q[q]

                    # Merge Optuna results into base params
                    if best_params:
                        merge_keys = ["num_leaves", "learning_rate", "lambda_l1",
                                      "lambda_l2", "max_depth", "min_data_in_leaf",
                                      "subsample"]
                        for k in merge_keys:
                            if k in best_params:
                                base_params[k] = best_params[k]
                    else:
                        base_params["num_leaves"] = 47
                        base_params["learning_rate"] = 0.01
                        base_params["lambda_l1"] = 0.0
                        base_params["lambda_l2"] = 1.5
                        base_params["max_depth"] = 5
                        base_params["min_data_in_leaf"] = 15

                    per_quantile_params[q] = dict(base_params)
                self.tuned_params[horizon] = {
                    q: dict(per_quantile_params[q]) for q in self.QUANTILES
                }

            _hp_elapsed = time.time() - _hp_start
            if reuse_hp:
                logger.info(f"  [timing] {horizon}d optuna: {_hp_elapsed:.1f}s (skipped - cached HP)")
            elif skip_hp:
                logger.info(f"  [timing] {horizon}d optuna: {_hp_elapsed:.1f}s (skipped - SKIP_HP_HORIZONS)")
            else:
                logger.info(f"  [timing] {horizon}d optuna: {_hp_elapsed:.1f}s")

            # Train ensemble members sequentially. No threading or multiprocessing
            # — LightGBM's internal OpenMP threads already utilize all cores,
            # which is exactly why this hands them ALL the cores. The `// 2`
            # this used to carry was left behind by the parallel-ensemble code
            # deleted 2026-07-21 and contradicted the line above it; on a 2-core
            # CI runner it pinned the final fits to a single thread while the
            # Optuna search and the CV folds both ran at n_jobs=-1.
            n_jobs = -1
            boost_rounds = self._boost_rounds(horizon)
            _es = self._early_stopping_enabled()
            for q in self.QUANTILES:
                pq = per_quantile_params[q]
                logger.info(f"  Training {horizon}d p{int(q*100)} ensemble ({self.N_ENSEMBLES} members)...")
                _ens_start = datetime.now()
                ensemble_models = []
                for ei in range(self.N_ENSEMBLES):
                    p = pq.copy()
                    p["random_state"] = self.ENSEMBLE_SEEDS[ei]
                    p["feature_fraction"] = self.ENSEMBLE_FEATURE_FRACTIONS[ei]
                    p["n_jobs"] = n_jobs
                    ensemble_models.append(self._train_ensemble_member(
                        p, dtrain, dval, boost_rounds, early_stopping=_es))

                self.models[(horizon, q)] = ensemble_models
                _ens_elapsed = (datetime.now() - _ens_start).total_seconds()
                fi = self._get_feature_importance(ensemble_models[0])
                logger.info(f"  Done in {_ens_elapsed:.0f}s — Top features: {fi['feature'].head(5).tolist()}")
                logger.info(f"  [timing] {horizon}d q{int(q*100)} ensemble: {_ens_elapsed:.1f}s")

            # Directional classifier: supplies the served up/flat/down call and
            # confidence (the quantile models only supply the interval).
            logger.info(f"  Training {horizon}d directional classifier (mover-weighted)...")
            # Vol-scaled labels were A/B-tested (2026-07-27) and did not beat
            # the fixed-band control; production stays fixed-band. Tooling
            # retained in scripts/ab_test_direction_labels.py.
            _dir_start = time.time()
            self.direction_models[horizon] = self._fit_direction_classifier(
                X_train, y_train, X_val, y_val, boosting_type,
                self._direction_tree_params(per_quantile_params),
                horizon=horizon,
                sigma_train=None,
                sigma_val=None,
                tier_train=(train_set["price_tier"].to_numpy()
                            if "price_tier" in train_set.columns else None),
                num_boost_round=boost_rounds,
                early_stopping=_es,
            )
            _dir_elapsed = time.time() - _dir_start
            logger.info(f"  [timing] {horizon}d direction classifier: {_dir_elapsed:.1f}s")

            # Train regime-specific models (optional: SKIP_REGIMES=1 to skip)
            #
            # `_warm_retrain` deliberately does NOT skip these, though it used to.
            # `predict` at :5584 *prefers* the regime model over the global one
            # whenever `_detect_current_regime` matches, so an artifact with no
            # regime models serves a different mid — dropping them is a change to
            # the forecast, not a cost saving. That was tolerable while warm
            # retrains only ran locally; it is not once CI restores the model
            # cache on training runs, which is what makes the warm path
            # production's steady state. Skipping them is now opt-in only.
            if os.environ.get("SKIP_REGIMES") == "1":
                # Drop any regime models this horizon carried in from load_models
                # so a skip run never re-persists stale regime artifacts.
                for key in [k for k in self.regime_models if k[1] == horizon]:
                    del self.regime_models[key]
                logger.info(f"  Regime models skipped (SKIP_REGIMES=1)")
            else:
                # Clear this horizon's regime models before refitting them. On a
                # cold run the dict is empty anyway; on a warm one it holds the
                # restored artifact's, and a regime that now falls below the
                # minimums below would otherwise keep the *previous* run's model
                # and re-persist it beside freshly trained global models.
                for key in [k for k in self.regime_models if k[1] == horizon]:
                    del self.regime_models[key]
                for regime in self.REGIMES:
                    if regime == "global":
                        continue
                    r_train = train_set[train_set["_regime"] == regime]
                    r_val = val_set[val_set["_regime"] == regime]
                    MIN_REGIME_TRAIN = 500
                    MIN_REGIME_VAL = 50
                    if len(r_train) < MIN_REGIME_TRAIN or len(r_val) < MIN_REGIME_VAL:
                        logger.info(f"  Skipping {regime} regime ({len(r_train)} train, {len(r_val)} val — "
                                    f"below minimum)")
                        continue

                    # Use global HP params (reuse Optuna results from global)
                    r_X_train = r_train[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(feature_medians)
                    r_y_train = r_train[f"target_return_{horizon}d"]
                    r_X_val = r_val[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(feature_medians)
                    r_y_val = r_val[f"target_return_{horizon}d"]

                    r_train_weights = self._compute_sample_weights(r_train, horizon)
                    r_val_weights = self._compute_sample_weights(r_val, horizon)

                    r_ds_params = {"max_bin": self.MAX_BIN, "feature_pre_filter": False}
                    r_dtrain_kw = dict(params=r_ds_params)
                    if r_train_weights is not None:
                        r_dtrain_kw["weight"] = r_train_weights
                    r_dval_kw = dict(params=r_ds_params)
                    if r_val_weights is not None:
                        r_dval_kw["weight"] = r_val_weights
                    # The same offset as the global fit. predict() prefers regime
                    # models over the global one, so a regime booster trained
                    # without it would be the thing actually served.
                    r_train_offset = self._naive_offset(r_train)
                    if r_train_offset is not None:
                        r_dtrain_kw["init_score"] = r_train_offset
                        r_dval_kw["init_score"] = self._naive_offset(r_val)
                    r_dtrain = lgb.Dataset(r_X_train, r_y_train, **r_dtrain_kw)
                    r_dval = lgb.Dataset(r_X_val, r_y_val, reference=r_dtrain, **r_dval_kw)
                    # Construct eagerly for the same reason as the global
                    # dtrain/dval above: avoid concurrent lazy-binning
                    # across the ensemble ThreadPoolExecutor.
                    r_dtrain.construct()
                    r_dval.construct()

                    logger.info(f"  Training {regime} regime models ({horizon}d, "
                                f"{len(r_train):,} train, {len(r_val):,} val)...")
                    for q in self.QUANTILES:
                        pq = per_quantile_params[q]
                        r_ensemble = []
                        for ei in range(self.N_ENSEMBLES):
                            p = pq.copy()
                            p["random_state"] = self.ENSEMBLE_SEEDS[ei]
                            p["feature_fraction"] = self.ENSEMBLE_FEATURE_FRACTIONS[ei]
                            p["n_jobs"] = n_jobs
                            r_ensemble.append(self._train_ensemble_member(
                                p, r_dtrain, r_dval, boost_rounds,
                                early_stopping=_es))
                        self.regime_models[(regime, horizon, q)] = r_ensemble

                    self.regime_feature_cols[(horizon, regime)] = list(self.feature_cols)
                    logger.info(f"  {regime} regime models for {horizon}d done "
                                f"({len(r_train)} train, {len(r_val)} val)")

            # Expanding-window CV. This is NOT optional on the production path,
            # and the reason is conformal, not metrics.
            #
            # Split conformal only guarantees coverage if the calibration
            # residuals are genuinely unseen. `X_val`/`y_val` is not: it is the
            # `dval` handed to lgb.early_stopping(50) in _train_ensemble_member
            # AND the set _optuna_search_params scores hyperparameters on. A
            # q_hat fitted there is measured on rows the model was selected
            # against, so its residuals are optimistically small, q_hat comes out
            # biased low, and the served band under-covers. Only CV's out-of-fold
            # records are unseen.
            #
            # So: SKIP_CV was removed from .github/workflows/price-forecast.yml
            # (pinned by test_ci_workflow_does_not_skip_cv), and a warm retrain
            # now reuses cached HP WITHOUT skipping CV. Those were always
            # separable concerns — HP reuse is the point of a warm retrain,
            # skipping calibration was collateral — and conflating them is what
            # put an in-sample q_hat behind the served 3d/7d band.
            #
            # SKIP_CV=1 survives as a local/dispatch speedup only. It routes to
            # the holdout leg, which warns that the band under-covers.
            _skip_cv = os.environ.get("SKIP_CV") == "1"
            _cv_feasible = self._cv_can_run(tdf, horizon)
            if _skip_cv:
                logger.info("  CV skipped (SKIP_CV=1 — local speedup; not the CI path)")
                oof_records, cv_metrics, pt_records, pt_records_clf = [], [], [], []
            elif not _cv_feasible:
                logger.info(f"  CV cannot run for {horizon}d (<2 expanding-window folds)")
                oof_records, cv_metrics, pt_records, pt_records_clf = [], [], [], []
            else:
                # Timed explicitly: post-minimal-model this is the single
                # largest phase of a warm retrain, and the only figure on
                # record for it was a REMAINDER (total minus the phases that
                # were instrumented), which lumped it with feature engineering
                # and artifact saving. A fold-count change cannot be attributed
                # against a remainder.
                _cv_t0 = time.time()
                (oof_records, cv_metrics, pt_records,
                 pt_records_clf) = self._cv_evaluate_horizon(
                    tdf, horizon, per_quantile_params,
                    per_item_row_sampling=per_item_row_sampling)
                logger.info(f"  [timing] {horizon}d conformal CV: "
                            f"{time.time() - _cv_t0:.1f}s "
                            f"({len(cv_metrics)} folds, {len(oof_records)} OOF rows)")

            # Calibrate. Order matters: q_hat sets the band width and the
            # confidence thresholds are fitted on that width, so the conformal
            # step runs FIRST and _calibrate_confidence always sees a range_pct
            # on the same scale _compute_confidence will compare against.
            if oof_records:
                records_df = pd.DataFrame(oof_records)
                calibration_source = f"CV-POOLED OOF, {len(cv_metrics)} folds"
                out_of_sample = True
            elif _skip_cv or not _cv_feasible:
                records_df = self._holdout_conformal_records(
                    horizon, X_val, y_val, val_set)
                calibration_source = "SINGLE HOLDOUT — expect UNDER-COVERAGE"
                out_of_sample = False
            else:
                raise RuntimeError(
                    f"CV ran for the {horizon}d horizon but produced no OOF "
                    f"records. This indicates a bug — _cv_evaluate_horizon "
                    f"should have raised."
                )

            # `tdf` is the frame every calibration row's `row_index` points
            # into, and the only reason it is passed: the learned scale needs
            # the features behind each residual. Ignored when the flag is off.
            q_hat = self._calibrate_conformal(horizon, records_df, tdf)
            # The WIDTH, and the exponent it was produced at. Both are here for
            # one reason: `q_hat` may not be differenced across SIGMA_EXPONENT
            # (the two arms are ~5.5x apart in units), so a paired read of that
            # flag has nothing to compare unless the width is reported. Median
            # half of `range_pct`, on exactly the rows q_hat was fitted on, which
            # is the basis the 0.87/0.86/0.84/0.77x prediction in
            # docs/changelog/2026-08-12-sigma-exponent-implemented.md was made on.
            _half_pct = 50.0 * float(np.nanmedian(
                records_df["range_pct"].to_numpy(dtype=float)))
            _cal_msg = (
                f"  Conformal calibration [{calibration_source}]: "
                f"q_hat={q_hat:.4f} (dimensionless x sigma), "
                f"beta={self.band_beta(horizon):.4f}, "
                f"median half-width={_half_pct:.2f}% of mid, "
                f"n={len(records_df)}, alpha={conformal.ALPHA}, "
                f"target coverage={conformal.NOMINAL_COVERAGE * 100:.0f}%, "
                f"basis={self.conformal_basis.get(horizon, 'unknown')}"
            )
            if out_of_sample:
                logger.info(_cal_msg)
            else:
                # WARNING, not INFO: a reader of forecast.log must be able to
                # tell that this horizon's band carries no coverage guarantee.
                logger.warning(
                    _cal_msg + " — q_hat was fitted on X_val, which is also the "
                    "early-stopping and Optuna scoring set, so those residuals "
                    "are optimistically small and this band is expected to cover "
                    "BELOW nominal. Out-of-fold CV records are the only unseen "
                    "calibration set; run without SKIP_CV=1, or give the horizon "
                    "enough distinct dates for >=2 folds."
                )
            self._calibrate_confidence(horizon=horizon, records_df=records_df)

            # The sigma axis, on the real OOF residuals. Ordered AFTER
            # calibration so it audits the same records q_hat was fitted on, and
            # it must stay report-only: see _sigma_tilt_audit.
            sigma_tilt = self._sigma_tilt_audit(horizon, records_df)

            # The expanding-window audit, on one line. `q_hat` above is fitted
            # on residuals pooled across folds whose models saw 87,224 to
            # 300,000 rows, while the shipped model trains on the full
            # TRAIN_FEATURE_ROWS budget — so if weaker fold models are what
            # makes the served band over-cover (87.2/91.8/90.6/89.0% against
            # 80%, 2026-08-12), `fold_q_hat` must FALL as `n_train` grows and
            # the pooled value must sit above the late folds'.
            #
            # ⚠️ TWO reasons this is a weak screen, and neither is fixable here.
            # `n_train` is monotone in fold index by construction, so a negative
            # rho cannot separate "more training data" from "later market
            # period" — a calmer regime late in the window produces the same
            # sign. And `CV_MAX_TRAIN_ROWS` (300,000, shipped 2026-08-09 as
            # `6b6fc81`) binds on most folds, so the x-axis is heavily tied and
            # spans at most 87K→300K against the 300K→1.2M gap that actually
            # separates a fold model from the served one.
            #
            # So a null here does NOT kill the hypothesis — it is equally
            # consistent with the cap having flattened the very axis being
            # measured. The decisive test is a `CV_MAX_TRAIN_ROWS` sweep, where
            # the fold geometry is held fixed and only the training size moves;
            # this line only says whether q_hat is sensitive to that size at all
            # over the narrow range the cap leaves.
            #
            # Reported only. `q_hat` above is what serves, unchanged.
            fold_q_hats = [(m["n_train"], m["fold_q_hat"]) for m in cv_metrics
                           if m.get("fold_q_hat") is not None]
            q_hat_trend = None
            if len(fold_q_hats) >= 3:
                _n = np.array([a for a, _ in fold_q_hats], dtype=float)
                _q = np.array([b for _, b in fold_q_hats], dtype=float)
                # Spearman: the claim is monotone decline, not a linear slope,
                # and 3-9 points cannot support a fitted slope anyway.
                _rho = float(pd.Series(_n).corr(pd.Series(_q), method="spearman"))
                q_hat_trend = {
                    "spearman_n_train_vs_q_hat": (
                        None if not np.isfinite(_rho) else round(_rho, 3)),
                    "first_fold_q_hat": round(float(_q[0]), 4),
                    "last_fold_q_hat": round(float(_q[-1]), 4),
                    # None, not inf: a fold whose residuals are all zero is a
                    # broken fold, and publishing inf under a ratio key would
                    # read as an extreme confirmation of the hypothesis.
                    "pooled_over_last_fold": (
                        None if _q[-1] <= 0 else round(float(q_hat / _q[-1]), 4)),
                    "n_folds_measured": len(fold_q_hats),
                    # How much range the screen actually had. With the cap
                    # binding, `n_train_distinct` collapses toward 1 and a rho
                    # near zero says nothing about the hypothesis -- it says the
                    # measurement had no x-axis. A reader comparing two runs
                    # needs this beside the rho, not in a separate log line.
                    "n_train_min": int(_n[0]),
                    "n_train_max": int(_n[-1]),
                    "n_train_distinct": int(len(np.unique(_n))),
                    "cv_max_train_rows": self._cv_max_train_rows(),
                }
                logger.info(
                    f"  Expanding-window audit: fold_q_hat "
                    f"{' → '.join(f'{v:.1f}' for _, v in fold_q_hats)} "
                    f"over n_train {_n[0]:,.0f}→{_n[-1]:,.0f} | "
                    f"spearman(n_train, q_hat)="
                    f"{q_hat_trend['spearman_n_train_vs_q_hat']} over "
                    f"{q_hat_trend['n_train_distinct']} distinct n_train "
                    f"(cap={q_hat_trend['cv_max_train_rows']:,}) | pooled "
                    f"{q_hat:.1f} is {q_hat_trend['pooled_over_last_fold']}x "
                    f"the last fold's. Negative rho + ratio >1 ⇒ the pooled fit "
                    f"inherits the early folds' weakness. A rho near 0 with "
                    f"n_train_distinct near 1 is NOT a null — the cap removed "
                    f"the x-axis; sweep CV_MAX_TRAIN_ROWS instead."
                )

            # Log CV fold-level metrics
            fold_accs = [m["directional_accuracy"] for m in cv_metrics]
            mean_acc = float(np.mean(fold_accs)) if fold_accs else float("nan")
            std_acc = float(np.std(fold_accs)) if len(fold_accs) > 1 else 0.0

            # Aggregate naive baselines for direct comparison. The model only
            # has a real directional edge if mean_dir_acc clears these.
            persist_accs = [m["persistence_accuracy"] for m in cv_metrics
                            if m.get("persistence_accuracy") is not None]
            mom_accs = [m["momentum_accuracy"] for m in cv_metrics
                        if m.get("momentum_accuracy") is not None]
            mean_persist = round(float(np.mean(persist_accs)), 1) if persist_accs else None
            mean_mom = round(float(np.mean(mom_accs)), 1) if mom_accs else None
            best_baseline = max([b for b in (mean_persist, mean_mom) if b is not None],
                                default=None)

            # The directional classifier is the SERVED signal, so the edge and
            # the trust warning are judged on it (not the quantile-median sign).
            clf_accs = [m["classifier_accuracy"] for m in cv_metrics
                        if m.get("classifier_accuracy") is not None]
            mean_clf = round(float(np.mean(clf_accs)), 1) if clf_accs else None
            served_acc = mean_clf if mean_clf is not None else mean_acc
            edge = round(served_acc - best_baseline, 1) if best_baseline is not None else None

            # Reported, never gated on. This is the cohort the production
            # headline scores (>=$1), so it is the only CV figure comparable to
            # it; `edge` deliberately stays on the all-tiers number above so
            # the trust warning and the confidence calibration do not move.
            # None when no fold had a >=$1 cohort, matching the per-fold rule.
            clf_ge1 = [m["classifier_accuracy_ge1"] for m in cv_metrics
                       if m.get("classifier_accuracy_ge1") is not None]
            mean_clf_ge1 = round(float(np.mean(clf_ge1)), 1) if clf_ge1 else None

            # Invariant #4. `edge` above is measured against persistence and
            # momentum, both of which the constant call beats comfortably — so
            # a positive `edge` never meant the model was useful. These are the
            # honest bars.
            def _mean_of(key, ndigits=2):
                vals = [m[key] for m in cv_metrics if m.get(key) is not None]
                return round(float(np.mean(vals)), ndigits) if vals else None

            mean_constant_call = _mean_of("constant_call_accuracy")
            mean_down_rate = _mean_of("realised_down_rate")
            rank_ic_summary = self._summarise_rank_ic(cv_metrics)
            mean_rank_ic = rank_ic_summary["mean_rank_ic"]
            mean_naive_rank_ic = rank_ic_summary["mean_naive_rank_ic"]
            mean_trees = _mean_of("n_trees", 1)
            # Deliberately measured on the quantile-median sign, NOT on
            # `served_acc`. `served_acc` falls back from the classifier to the
            # median sign when CV_DIAGNOSTIC_CLASSIFIER=0, so an edge built on
            # it would mean one thing in CI and another locally while carrying
            # the same key. This one is always the median sign, which is also
            # what `pt` below is computed from, so the two invariant-#4 numbers
            # always describe the same signal.
            #
            # The classifier is what production actually serves, so it gets the
            # same two numbers under its own keys (below) rather than displacing
            # these. Publishing both is what lets the pair be compared; making
            # one key mean either signal is what made the 2026-08-10 diagnostics
            # run report a q50 verdict under a heading that said "served".
            edge_vs_constant = (
                None if (mean_constant_call is None or not fold_accs)
                else round(mean_acc - mean_constant_call, 2))
            # Positive means the model orders items better than "bet against
            # yesterday's move". On 2026-08-08 it was negative at all four
            # horizons, which is the bar this project had never measured.
            rank_ic_edge = rank_ic_summary["rank_ic_edge_vs_naive"]

            # PT is the headline: it tests whether predictions are independent
            # of outcomes, so unlike DA it cannot be passed by a base rate. Run
            # on the pooled out-of-fold rows, clustered by forecast date exactly
            # as backtest/scoring.py does in production.
            pt = pesaran_timmermann(pt_records, MIN_FORECAST_DATES)

            # The same test on the SERVED classifier. `mean_constant_call` and
            # `mean_down_rate` are properties of the outcomes alone, so they are
            # the same bar for both signals and are not recomputed. None when
            # CV_DIAGNOSTIC_CLASSIFIER=0 -- visibly absent, never falling back to
            # the quantile sign.
            pt_clf = (pesaran_timmermann(pt_records_clf, MIN_FORECAST_DATES)
                      if pt_records_clf else None)
            edge_vs_constant_clf = (
                None if (mean_constant_call is None or mean_clf is None)
                else round(mean_clf - mean_constant_call, 2))

            if cv_metrics:
                # classifier= pools all tiers and the frame is ~83% tier-0, so
                # it reads close to the penny-item score. classifier>=$1= is
                # the one to compare against the production headline.
                logger.info(f"  CV ({len(cv_metrics)} folds): "
                            f"classifier={mean_clf}% (>=$1: {mean_clf_ge1}%) "
                            f"quantile-sign={mean_acc:.1f}% (sd={std_acc:.1f}%)")
                logger.info(f"  Baselines: persistence={mean_persist}% "
                            f"momentum={mean_mom}% → served(classifier) edge vs best={edge}pp")
                # Invariant #4: never on its own. The constant call is the bar
                # persistence and momentum were standing in for, and it is a
                # much higher one.
                logger.info(
                    f"  Invariant #4 [quantile-sign]: "
                    f"constant-call={mean_constant_call}% "
                    f"down-rate={mean_down_rate}% → edge vs constant call="
                    f"{edge_vs_constant}pp | PT excess={pt['pt_excess_pp']}pp "
                    f"t={pt['pt_t_stat']} verdict={pt['pt_verdict']}")
                # The line that describes production. Absent, not substituted,
                # when the diagnostic classifier did not run.
                if pt_clf is not None:
                    logger.info(
                        f"  Invariant #4 [SERVED classifier]: "
                        f"constant-call={mean_constant_call}% "
                        f"down-rate={mean_down_rate}% → edge vs constant call="
                        f"{edge_vs_constant_clf}pp | PT excess="
                        f"{pt_clf['pt_excess_pp']}pp t={pt_clf['pt_t_stat']} "
                        f"verdict={pt_clf['pt_verdict']}")
                else:
                    logger.info(
                        "  Invariant #4 [SERVED classifier]: not measured "
                        "(CV_DIAGNOSTIC_CLASSIFIER=0) — the line above "
                        "describes the q50 sign, NOT what production serves.")
                logger.info(
                    f"  Cross-sectional (>=$1): rank_ic={mean_rank_ic} vs "
                    f"naive(-return_1d)={mean_naive_rank_ic} → edge={rank_ic_edge} "
                    f"| mean trees/fold={mean_trees}")
                # Printed beside it, never instead of it. The line above is the
                # contaminated basis every stored A/B was ranked on; this one is
                # the cohort where `p[d]/S[d]` is 1 and the metric means what it
                # says. A reader comparing two runs should compare THIS line.
                tied_edge = rank_ic_summary["rank_ic_edge_vs_naive_tied"]
                if rank_ic_summary["mean_rank_ic_tied"] is None:
                    logger.info(
                        "  Cross-sectional (>=$1, CLEAN ANCHOR): not measured — "
                        f"{rank_ic_summary['tied_rows']:,} tied served rows over "
                        f"{rank_ic_summary['tied_dates']} usable dates. Rank an "
                        "arm on the pooled line above only if you mean to rank "
                        "it on the anchor wedge.")
                else:
                    logger.info(
                        "  Cross-sectional (>=$1, CLEAN ANCHOR): rank_ic="
                        f"{rank_ic_summary['mean_rank_ic_tied']} vs naive="
                        f"{rank_ic_summary['mean_naive_rank_ic_tied']} → "
                        f"edge={tied_edge} | "
                        f"{rank_ic_summary['tied_rows']:,} rows, "
                        f"{rank_ic_summary['tied_dates']} of "
                        f"{rank_ic_summary['rank_ic_dates']} date-folds. "
                        "← RANK ARMS ON THIS LINE.")
                # C2 lambdarank arm, read on the CLEAN ANCHOR cohort against both
                # bars: beat the naive baseline AND the q50's own ordering.
                if rank_ic_summary["mean_lr_rank_ic_tied"] is not None:
                    lr_vs_naive = rank_ic_summary["lr_rank_ic_edge_vs_naive_tied"]
                    lr_vs_q50 = rank_ic_summary["lr_rank_ic_edge_vs_q50_tied"]
                    logger.info(
                        "  Cross-sectional (>=$1, CLEAN ANCHOR) LAMBDARANK: "
                        f"rank_ic={rank_ic_summary['mean_lr_rank_ic_tied']} | "
                        f"edge vs naive={lr_vs_naive}, vs q50={lr_vs_q50} — "
                        "PASS needs BOTH > 0.")
                    if not (lr_vs_naive and lr_vs_naive > 0
                            and lr_vs_q50 and lr_vs_q50 > 0):
                        logger.warning(
                            f"  ⚠ {horizon}d LAMBDARANK does not clear both bars "
                            f"on the clean-anchor cohort (vs naive={lr_vs_naive}, "
                            f"vs q50={lr_vs_q50}) — no served step is licensed.")
                if edge_vs_constant is not None and edge_vs_constant <= 0:
                    logger.warning(
                        f"  ⚠ {horizon}d quantile sign does NOT beat the constant "
                        f"call ({edge_vs_constant}pp) — a single fixed direction "
                        f"scores {mean_constant_call}% on these folds.")
                if edge_vs_constant_clf is not None and edge_vs_constant_clf <= 0:
                    logger.warning(
                        f"  ⚠ {horizon}d SERVED classifier does NOT beat the "
                        f"constant call ({edge_vs_constant_clf}pp) — a single "
                        f"fixed direction scores {mean_constant_call}% on these "
                        f"folds. This is the signal production ships.")
                if rank_ic_edge is not None and rank_ic_edge <= 0:
                    logger.warning(
                        f"  ⚠ {horizon}d model does NOT beat ranking by "
                        f"-return_1d (rank IC {mean_rank_ic} vs "
                        f"{mean_naive_rank_ic}) — the ML stack is subtracting "
                        f"from its own best feature. Measured on the raw-anchor "
                        f"basis; read the CLEAN ANCHOR edge before acting on it.")
                if tied_edge is not None and tied_edge <= 0:
                    logger.warning(
                        f"  ⚠ {horizon}d model does NOT beat -return_1d on the "
                        f"CLEAN ANCHOR cohort either ({tied_edge}) — this is the "
                        f"basis-free read, so it is the one that counts.")
                if pt["pt_verdict"] not in ("skill",):
                    logger.warning(
                        f"  ⚠ {horizon}d Pesaran-Timmermann verdict is "
                        f"'{pt['pt_verdict']}' (t={pt['pt_t_stat']}) on the "
                        f"quantile sign — that call is not distinguishable "
                        f"from chance.")
                if pt_clf is not None and pt_clf["pt_verdict"] not in ("skill",):
                    logger.warning(
                        f"  ⚠ {horizon}d Pesaran-Timmermann verdict is "
                        f"'{pt_clf['pt_verdict']}' (t={pt_clf['pt_t_stat']}) on "
                        f"the SERVED classifier — what production ships is not "
                        f"distinguishable from chance.")
            self.cv_results[horizon] = {
                "fold_count": len(cv_metrics),
                "per_fold": cv_metrics,
                "mean_dir_acc": round(mean_acc, 1) if fold_accs else 0,
                "std_dir_acc": round(std_acc, 1) if len(fold_accs) > 1 else 0,
                "min_dir_acc": round(min(fold_accs), 1) if fold_accs else 0,
                "max_dir_acc": round(max(fold_accs), 1) if fold_accs else 0,
                "mean_classifier_acc": mean_clf,
                "mean_classifier_acc_ge1": mean_clf_ge1,
                "mean_persistence_acc": mean_persist,
                "mean_momentum_acc": mean_mom,
                "edge_vs_best_baseline": edge,
                # Invariant #4 + the cross-sectional headline.
                "mean_constant_call_acc": mean_constant_call,
                "mean_realised_down_rate": mean_down_rate,
                "edge_vs_constant_call": edge_vs_constant,
                # Which signal the two lines above describe. Always the median
                # sign, so the key does not change meaning when the diagnostic
                # classifier is skipped.
                "invariant_4_signal": "quantile_sign",
                # The same pair for the signal production actually serves.
                # None when CV_DIAGNOSTIC_CLASSIFIER=0. A consumer that wants
                # the served verdict must read these and handle the None --
                # falling back to the keys above would silently substitute the
                # q50 sign, which is the bug this pair exists to prevent.
                "edge_vs_constant_call_classifier": edge_vs_constant_clf,
                "pt_classifier": pt_clf,
                # Pooled and tied, together. The tied pair is the one a new arm
                # is ranked on; the pooled pair is kept unchanged because it is
                # the series every historical meta.json holds.
                **rank_ic_summary,
                "mean_trees_per_fold": mean_trees,
                "pt": pt,
                # The expanding-window screen. None when fewer than 3 folds
                # reported a `fold_q_hat` — visibly absent rather than a rho
                # over two points. Diagnostic; nothing builds a band from it.
                "q_hat_trend": q_hat_trend,
                # The sigma axis. None below MIN_CALIBRATION_ROWS. Read
                # `elasticity_heldout` first — the pooled legs fit and score the
                # exponent on the same rows. Diagnostic; nothing serves from it.
                "sigma_tilt": sigma_tilt,
            }

            # Validate feature groups: permutation test on the held-out set.
            # Skip entirely when the validation window is thin (MIN_VAL_ROWS /
            # MIN_VAL_DATES, the same floor the split widens to reach) —
            # permutation tests below it are pure noise and cause false-positive
            # pruning that collapses 14d/30d models to ~4 features.
            # The significance_level parameter (0.05) gates pruning further:
            # a group must pass BOTH the statistical significance test (p < α)
            # AND the practical significance test (drop_pp >= 0.5) to be kept.
            # This prevents noisy-but-spurious correlations from surviving
            # on marginal windows without fully skipping the check.
            val_dates = val_set["date"].nunique() if "date" in val_set.columns else 0
            need_retrain = False
            if _warm_retrain:
                logger.info("  Skipping feature-group validation (warm retrain)")
            elif len(val_set) < self.MIN_VAL_ROWS or val_dates < self.MIN_VAL_DATES:
                logger.info(
                    f"  Skipping feature-group validation ({len(val_set)} rows, "
                    f"{val_dates} dates — below minimum threshold)"
                )
            else:
                try:
                    X_val_np = X_val.values if hasattr(X_val, "values") else X_val
                    y_val_np = y_val.values if hasattr(y_val, "values") else y_val
                    fv = self._validate_feature_groups(
                        X_val_np, y_val_np, self.feature_cols,
                        horizon=horizon, n_shuffles=20, min_drop_pp=0.5,
                        significance_level=0.05,
                        offset=val_offset,
                    )
                    self.cv_results[horizon]["feature_validation"] = fv
                    failed = [g for g, r in fv.items() if not r["passed"]]
                    if failed:
                        if self.prune_failed_groups and attempt == 0:
                            failed_cols = set()
                            for g in failed:
                                for f in fv[g]["features"]:
                                    failed_cols.add(f)
                            pre_count = len(self.feature_cols)
                            self.feature_cols = [c for c in self.feature_cols
                                                  if c not in failed_cols]
                            # Safety net: if all features were pruned, fall
                            # back to a minimal core set so LightGBM doesn't
                            # crash with 0 columns.
                            if not self.feature_cols and pre_count > 0:
                                safe = ["price_log", "price_lag_1d", "price_lag_3d",
                                        "price_return_1d", "price_return_3d",
                                        "price_return_7d", "price_std_7d"]
                                self.feature_cols = [c for c in safe if c in tdf.columns]
                                if not self.feature_cols:
                                    self.feature_cols = tdf.select_dtypes(include=[np.number]).columns[:1].tolist()
                                logger.warning(
                                    f"  All features pruned — falling back to "
                                    f"{len(self.feature_cols)} core features as safety net"
                                )
                            price_passed = (
                                "price_technicals" in fv and fv["price_technicals"]["passed"]
                            )
                            if price_passed:
                                logger.warning(
                                    f"  Pruned {len(failed_cols)} features from groups {failed} "
                                    f"({pre_count} -> {len(self.feature_cols)}). "
                                    f"price_technicals intact — skipping retrain (no accuracy gain)."
                                )
                            else:
                                logger.warning(
                                    f"  Pruned {len(failed_cols)} features from groups {failed} "
                                    f"({pre_count} -> {len(self.feature_cols)}). Retraining."
                                )
                                need_retrain = True
                        else:
                            logger.warning(
                                f"  Feature groups with no causal signal: {failed}. "
                                f"({'Auto-prune disabled' if not self.prune_failed_groups else 'Already re-trained.'})"
                            )
                except Exception as e:
                    logger.warning(f"  Feature validation skipped: {e}")

            if not need_retrain:
                break

        base_features = list(self.feature_cols)
        excluded_groups = self.HORIZON_EXCLUDED_GROUPS.get(horizon, [])
        if excluded_groups and not _warm_retrain:
            horizon_features = [
                c for c in base_features
                if _feature_group(c) not in excluded_groups
            ]
            logger.info(
                f"  {horizon}d: excluded {len(base_features) - len(horizon_features)} features "
                f"from groups {excluded_groups} "
                f"({len(base_features)} -> {len(horizon_features)} features)"
            )
            self.horizon_feature_cols[horizon] = horizon_features
        else:
            self.horizon_feature_cols[horizon] = base_features

        _hz_elapsed = (datetime.now() - _hz_start).total_seconds()
        logger.info(f"  Horizon {horizon}d done in {_hz_elapsed:.0f}s")
        if self.cv_results.get(horizon):
            cv = self.cv_results[horizon]
            logger.info(f"  CV summary: mean={cv.get('mean_dir_acc', '?'):}% "
                        f"std={cv.get('std_dir_acc', '?'):}% "
                        f"range=[{cv.get('min_dir_acc', '?'):}%, {cv.get('max_dir_acc', '?'):}%]")
        del tdf

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _directional_accuracy(pred_returns, actual_returns) -> float:
        """Percent of rows whose predicted return direction matches the actual.

        Uses the same 3-class up/flat/down bucketing (DIRECTION_FLAT_TOLERANCE_PCT)
        as model evaluation so baseline and model numbers are directly
        comparable. Rows with NaN actuals are skipped. Returns 0.0 for an
        empty comparison.
        """
        tol = DIRECTION_FLAT_TOLERANCE_PCT
        pred = np.asarray(pred_returns, dtype=float)
        actual = np.asarray(actual_returns, dtype=float)
        hits = 0
        n = 0
        for p, a in zip(pred, actual):
            if np.isnan(a):
                continue
            a_dir = "up" if a > tol else "down" if a < -tol else "flat"
            p_dir = "up" if p > tol else "down" if p < -tol else "flat"
            hits += int(p_dir == a_dir)
            n += 1
        return round(hits / n * 100, 1) if n else 0.0

    @staticmethod
    def _select_feature_cols(df, horizons, shelved) -> List[str]:
        """The numeric columns the trainer fits on.

        Drops metadata, the per-horizon targets, and everything in *shelved*.
        Extracted from build_training_data so the shelving can be tested without
        a full training run — a feature leaving SHELVED_FEATURES and silently
        re-entering production is exactly the regression worth a test.
        """
        exclude = {"item_id", "date", "timestamp", "price", "volume",
                   "name", "release_date", DIRECTION_LABEL_VOL_COL,
                   # n_ask_sources (the composition-stability instrument's
                   # column, added to the voted frame in v5) never reaches
                   # this far in practice -- engineer_features' resample to
                   # one row per item-day names only "price" and "volume" in
                   # its groupby().agg(), which drops it -- but that is an
                   # accident of an unrelated aggregation, and
                   # _feature_group("n_ask_sources") falls through to
                   # "other", which _skipped_feature_groups() never skips.
                   # Name it here so the exclusion is a decision, not a
                   # side effect.
                   "n_ask_sources",
                   # The clean-cohort mask. `prepare_targets` adds it after
                   # this runs and the dtype filter below would drop a bool
                   # anyway, so this is belt-and-braces -- but it is a
                   # SCORING cohort, and a model that fitted on it would be
                   # reading which basis its own label is contaminated by.
                   ANCHOR_TIED_COL}
        exclude |= {f"target_{h}d" for h in horizons}
        exclude |= {f"target_return_{h}d" for h in horizons}
        # The market factor is computed from other items' FUTURE prices. It is
        # a label input, never a feature -- if it reaches feature_cols the
        # model trains on the answer.
        exclude |= {f"market_factor_{h}d" for h in horizons}
        exclude |= set(shelved)

        return [c for c in df.columns if c not in exclude
                and df[c].dtype in (np.float64, np.float32, np.int64, int, float)]

    def _skipped_feature_groups(self) -> set:
        """Groups engineer_features may skip: everything the allowlist drops.

        Derived, never a literal. Track C4 proposes re-admitting
        `cross_sectional`; a hard-coded list would then produce a frame with the
        group allowlisted and its columns absent, median-filled to zero.

        `other` is never skipped: distance_to_support / distance_to_resistance /
        high_low_range_30d group as `other` because _feature_group matches the
        prefix `support_` while the columns are named `distance_to_*`, and they
        are computed inside _compute_price_features regardless.
        """
        allowlist = set(self.FEATURE_GROUP_ALLOWLIST or [])
        if not allowlist:
            return set()
        if self.bymykel_metadata_enabled():
            allowlist.add(self.BYMYKEL_META_GROUP)
        if self.tier_lead_enabled():
            allowlist.add(self.TIER_LEAD_GROUP)
        return set(self.ALL_FEATURE_GROUPS) - allowlist - {"other"}

    def _reduce_feature_cols(self, df: pd.DataFrame) -> None:
        """Apply the allowlist and the correlation prune, in the cheaper order.

        _prune_features builds `df[self.feature_cols].corr()`, which is
        O(rows x p^2) single-threaded pandas: 25.2s over the 123 selected
        candidates against 1.65s over the 33 the allowlist keeps. Running the
        allowlist first is therefore ~23s off every retrain.

        Output-identical on the production frame, but not by construction:
        _prune_features keeps the lower-indexed member of each >0.95 pair and
        index order does not follow group, so a correlated pair straddling the
        allowlist boundary could resolve differently. ALLOWLIST_BEFORE_PRUNE
        restores the old order.
        """
        allowlist = list(self.FEATURE_GROUP_ALLOWLIST or [])
        if allowlist and self.bymykel_metadata_enabled():
            allowlist.append(self.BYMYKEL_META_GROUP)
        if allowlist and self.tier_lead_enabled():
            allowlist.append(self.TIER_LEAD_GROUP)

        def _allow():
            if not allowlist:
                return
            pre = len(self.feature_cols)
            self.feature_cols = self._apply_feature_allowlist(
                self.feature_cols, allowlist)
            logger.info(
                f"Feature allowlist {allowlist}: "
                f"{pre} -> {len(self.feature_cols)} features")

        if self.ALLOWLIST_BEFORE_PRUNE:
            _allow()
            self.feature_cols = self._prune_features(df)
        else:
            self.feature_cols = self._prune_features(df)
            _allow()

    @staticmethod
    def _apply_feature_allowlist(feature_cols, allowlist):
        """Keep only features whose _feature_group() is in ``allowlist``.

        ``allowlist`` of None/empty is a no-op (returns all features).
        """
        if not allowlist:
            return list(feature_cols)
        allow = set(allowlist)
        return [c for c in feature_cols if _feature_group(c) in allow]

    @staticmethod
    def _recenter_on_momentum(low_ret, mid_ret, high_ret, momentum_ret):
        """Recenter quantile return forecasts on the trailing (momentum) return,
        preserving each item's calibrated interval half-widths.

        Where ``momentum_ret`` is NaN (insufficient history) the model's own
        forecast is kept unchanged. Returns (low, mid, high) in return space.
        Because the inputs are already monotone (low <= mid <= high), the
        preserved non-negative offsets keep the recentred triple monotone too.
        """
        mid_ret = np.asarray(mid_ret, dtype=float)
        low_ret = np.asarray(low_ret, dtype=float)
        high_ret = np.asarray(high_ret, dtype=float)
        momentum_ret = np.asarray(momentum_ret, dtype=float)
        low_off = mid_ret - low_ret
        high_off = high_ret - mid_ret
        new_mid = np.where(np.isnan(momentum_ret), mid_ret, momentum_ret)
        return new_mid - low_off, new_mid, new_mid + high_off

    @staticmethod
    def _direction_classes(returns, threshold=DIRECTION_FLAT_TOLERANCE_PCT) -> np.ndarray:
        """Bucket % returns into 0=down, 1=flat, 2=up using a flat band of
        ``threshold`` (scalar or per-row array, percent). Scalar reproduces the
        legacy fixed-±DIRECTION_FLAT_TOLERANCE_PCT behavior."""
        r = np.asarray(returns, dtype=float)
        thr = np.asarray(threshold, dtype=float)
        return np.where(r > thr, 2, np.where(r < -thr, 0, 1)).astype(int)

    @staticmethod
    def _demean_returns(returns, factor) -> np.ndarray:
        """Subtract the market factor from % returns, giving the idiosyncratic
        residual ``e = r - m``. See ``models/market_factor.py`` for the factor.

        A missing factor demeans by zero rather than producing NaN, so the row
        keeps the raw label instead of being dropped. That matters to any paired
        arm comparison: dropping rows in one arm only changes its row counts and
        breaks the pairing.

        Not used by training or serving. It survives the removal of the
        market-relative label experiment (refuted 2026-08-06, see
        docs/changelog/2026-08-06-market-relative-labels-refuted.md) because
        scripts/ab_test_item_metadata.py calls it to re-score its arms against a
        demeaned target -- the run that amended the item-metadata conclusions in
        docs/research/accuracy-opportunities.md.
        """
        r = np.asarray(returns, dtype=float)
        if factor is None:
            return r.copy()
        m = np.asarray(factor, dtype=float)
        return r - np.nan_to_num(m, nan=0.0)


    @staticmethod
    def _has_date_coverage(forecast_dates) -> bool:
        """True when distinct non-null forecast dates reach MIN_FORECAST_DATES.

        Row count cannot substitute for this. Items sharing a forecast_date
        share one market-wide move, so a five-figure cohort on two dates is
        nearer two observations than 11,000 — see
        docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md.
        Reuses the reporting threshold so the value we fit on and the value we
        report cannot drift apart.
        """
        distinct = {d for d in forecast_dates if d is not None and not pd.isna(d)}
        return len(distinct) >= MIN_FORECAST_DATES

    @staticmethod
    def _direction_threshold(sigma, horizon: int, k: float,
                             floor: float, cap: float) -> np.ndarray:
        """Per-row flat-band threshold (percent) = clamp(k * sigma * sqrt(h),
        floor, cap). ``sigma`` is trailing daily-return std in percent; scalar
        or array. Always returns a float ndarray."""
        s = np.atleast_1d(np.asarray(sigma, dtype=float))
        raw = k * s * np.sqrt(float(horizon))
        return np.clip(raw, floor, cap)

    @staticmethod
    def _served_cohort_multiplier(base_weights, tiers, served_share: float) -> float:
        """Multiplier for served (>= $1) rows that makes them carry
        ``served_share`` of the total training weight.

        Solving ``m*W_s / (m*W_s + W_n) = share`` for m gives

            m = share * W_n / ((1 - share) * W_s)

        where W_s / W_n are the sums of *base_weights* over the served and
        non-served partitions. Taking the sums over the already-computed mover
        weights (rather than over row counts) is what makes the resulting share
        exact once the two weightings compose.

        The knob is a share and not a raw multiplier on purpose: a multiplier's
        meaning drifts with the frame's tier composition, which varies fold to
        fold, while the share is the quantity we actually mean.

        Returns 1.0 — leave the weights alone — when the target is unreachable
        or already met: no served rows (W_s == 0) and no non-served rows
        (W_n == 0) both have no multiplier that moves the share.
        """
        if not 0.0 < served_share < 1.0:
            raise ValueError(
                f"served_share must be in (0, 1), got {served_share!r}")
        w = np.asarray(base_weights, dtype=float)
        served = np.asarray(tiers) >= HEADLINE_MIN_TIER
        w_served = float(w[served].sum())
        w_other = float(w[~served].sum())
        if w_served <= 0.0 or w_other <= 0.0:
            return 1.0
        return served_share * w_other / ((1.0 - served_share) * w_served)

    @classmethod
    def _direction_sample_weights(cls, returns, threshold, mover_weight: float,
                                  tiers=None,
                                  served_share: Optional[float] = None) -> np.ndarray:
        """Up-weight clearly-moving rows (|return| > ``threshold``) by
        ``mover_weight``; flat rows keep weight 1.0. ``threshold`` scalar or
        per-row array (percent).

        When ``tiers`` and ``served_share`` are both given, rows in the served
        cohort (``price_tier >= HEADLINE_MIN_TIER``, i.e. >= $1) are then scaled
        so they carry ``served_share`` of the total weight. Production trains on
        a frame that is ~83% sub-$1 while `api/serving_policy.py` shows only
        >= $1, so untouched the classifier spends most of its capacity on rows
        no one is served. ``served_share=None`` reproduces the pre-2026-08-06
        weights exactly. See
        docs/superpowers/specs/2026-08-06-served-cohort-weighting-design.md.
        """
        r = np.asarray(returns, dtype=float)
        thr = np.asarray(threshold, dtype=float)
        w = np.ones(len(r))
        w[np.abs(r) > thr] = mover_weight
        if served_share is None or tiers is None:
            return w
        m = cls._served_cohort_multiplier(w, tiers, served_share)
        served = np.asarray(tiers) >= HEADLINE_MIN_TIER
        w = w.copy()
        w[served] *= m
        return w

    @classmethod
    def _direction_class_prior(cls, returns, threshold: float,
                               mover_weight: float, tiers=None,
                               served_share: Optional[float] = None) -> Dict[int, float]:
        """Weighted training class prior as {0: down, 1: flat, 2: up}.

        Weighted by _direction_sample_weights, because that is the
        distribution the classifier's multiclass objective actually sees —
        raw class counts would describe a model that was never trained. That
        commitment is why ``tiers``/``served_share`` are threaded through: once
        the classifier gained a served-cohort weight, a prior that ignored it
        would silently describe a different model again.
        Returns {} when not estimable, which the diagnostic
        (scripts/diagnose_direction_prior.py) reports as "prior not estimable".

        ``threshold`` must be a scalar. Production trains with
        sigma_train=None in _train_horizon_inline and _cv_evaluate_horizon,
        so the band is the fixed scalar DIRECTION_FLAT_TOLERANCE_PCT; a
        per-row band would need the same rows dropped here as in the finite
        mask below.

        ``tiers`` is filtered by the same finite mask as ``returns``, so a
        caller passing the raw column does not silently misalign the two.
        """
        r = np.asarray(returns, dtype=float)
        finite = np.isfinite(r)
        r = r[finite]
        if r.size == 0:
            return {}
        if tiers is not None:
            tiers = np.asarray(tiers)[finite]
        c = cls._direction_classes(r, float(threshold))
        w = cls._direction_sample_weights(r, float(threshold), mover_weight,
                                          tiers=tiers, served_share=served_share)
        total = float(w.sum())
        if total <= 0.0:
            return {}
        return {k: float(w[c == k].sum() / total) for k in (0, 1, 2)}

    def _warn_no_classifier(self, horizon: int, n: int, n_flat: int) -> None:
        """Announce that a horizon served directions from the dead-band fallback.

        `predict` falls back to a ±`DIRECTION_FLAT_TOLERANCE_PCT` band on
        `mid_ret` when `direction_models` holds no booster for the horizon. That
        band is not a safe default: the served mid is shrunk far harder than
        realised returns, so on 2026-07-19 it called `flat` on **64.0%** of the
        ≥$1 cross-section against a 23.1% realised flat rate — and on 2026-07-17
        99.8% of `|mid_ret|` sat inside it, which would have called almost
        everything flat. Nothing logged either fact at the time.

        Silent by design when the classifier is present, so a healthy run stays
        quiet and this line means exactly one thing.
        `docs/changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`
        """
        if n <= 0:
            return
        pct = n_flat / n * 100
        logger.warning(
            f"  h={horizon}: NO directional classifier — served the "
            f"±{DIRECTION_FLAT_TOLERANCE_PCT}% dead-band fallback for {n} items, "
            f"{n_flat} ({pct:.1f}%) called flat")

    @classmethod
    def _recenter_on_direction(cls, low_ret, mid_ret, high_ret, direction_class):
        """Recenter forecasts so the median's sign matches the classifier's call,
        preserving each item's interval half-widths.

        down(0) -> -|mid|, flat(1) -> 0, up(2) -> +|mid|. Keeping |mid| as the
        magnitude means the quantile model still sets *how much*; the classifier
        only sets *which way*. Returns (low, mid, high).
        """
        mid_ret = np.asarray(mid_ret, dtype=float)
        low_ret = np.asarray(low_ret, dtype=float)
        high_ret = np.asarray(high_ret, dtype=float)
        cls_arr = np.asarray(direction_class, dtype=int)
        low_off = mid_ret - low_ret
        high_off = high_ret - mid_ret
        mag = np.abs(mid_ret)
        new_mid = np.select(
            [cls_arr == 2, cls_arr == 0, cls_arr == 1],
            [mag, -mag, 0.0],
            default=mid_ret,
        )
        return new_mid - low_off, new_mid, new_mid + high_off

    def _fit_direction_classifier(self, X_train, y_train_ret, X_val, y_val_ret,
                                   boosting_type: str, tree_params: dict,
                                   horizon: Optional[int] = None,
                                   sigma_train=None, sigma_val=None,
                                   num_boost_round: int = 200,
                                   random_state: int = 42,
                                   tier_train=None,
                                   early_stopping: bool = False):
        """Train a 3-class (down/flat/up) LightGBM classifier on returns,
        up-weighting movers. When ``sigma_train`` is given, the flat band is
        vol-scaled per row (k_h * sigma * sqrt(h), clamped); otherwise the
        legacy fixed ±DIRECTION_FLAT_TOLERANCE_PCT band is used.

        ``early_stopping`` defaults to **False**, so this trains
        ``num_boost_round`` rounds outright and ``X_val``/``y_val_ret`` are used
        for nothing. It used to early-stop on val multi-logloss, which on the
        trailing window fitted clf_14d to 3 trees — one boosting round — while
        clf_3d and clf_7d ran to the cap. See FIXED_BOOST_ROUNDS. Callers that
        want the old behaviour pass ``early_stopping=True``.

        ``tier_train`` is the training rows' ``price_tier``. Combined with
        ``self.served_cohort_share`` it up-weights the >= $1 cohort production
        actually serves. Passed explicitly rather than read off ``X_train``
        because the >0.95 correlation prune can drop ``price_tier`` from
        ``feature_cols``, and a weight that silently switches off when a feature
        is pruned is the failure mode SHELVED_FEATURES was written to avoid.
        Validation rows are deliberately left unweighted: the early-stopping
        metric should track the whole fold, and weighting it too would move the
        stopping point along with the objective."""
        k = self.DIRECTION_VOL_MULTIPLIER_MAP.get(horizon, 1.0)
        mover_weight = self.DIRECTION_MOVER_WEIGHT_MAP.get(horizon, 3.0)
        floor, cap = DIRECTION_THRESHOLD_FLOOR_PCT, DIRECTION_THRESHOLD_CAP_PCT

        def _thr(sigma):
            if sigma is None:
                return DIRECTION_FLAT_TOLERANCE_PCT
            return self._direction_threshold(np.asarray(sigma, dtype=float),
                                              horizon, k, floor, cap)

        ds = {"max_bin": self.MAX_BIN, "feature_pre_filter": False}
        thr_train = _thr(sigma_train)
        c_train = self._direction_classes(y_train_ret, thr_train)
        w_train = self._direction_sample_weights(
            y_train_ret, thr_train, mover_weight,
            tiers=tier_train, served_share=self.served_cohort_share)
        if self.served_cohort_share is not None and tier_train is not None:
            served = np.asarray(tier_train) >= HEADLINE_MIN_TIER
            logger.info(
                f"  {horizon}d direction classifier: served-cohort weighting "
                f"to share={self.served_cohort_share:.2f} — "
                f"{int(served.sum()):,}/{len(served):,} rows are >= $1 "
                f"({served.mean() * 100:.1f}% of rows, "
                f"{w_train[served].sum() / max(w_train.sum(), 1e-12) * 100:.1f}% "
                f"of weight)"
            )
        dtrain = lgb.Dataset(X_train, c_train, params=ds, weight=w_train)
        params = dict(tree_params)
        params.update(objective="multiclass", num_class=3, metric="multi_logloss",
                      boosting_type=boosting_type, verbosity=-1, n_jobs=-1,
                      random_state=random_state)
        callbacks = [lgb.log_evaluation(0)]
        valid_sets = None
        if (early_stopping and X_val is not None and y_val_ret is not None
                and len(X_val)):
            dval = lgb.Dataset(X_val, self._direction_classes(y_val_ret, _thr(sigma_val)),
                               reference=dtrain, params=ds)
            valid_sets = [dval]
            callbacks.insert(0, lgb.early_stopping(20))
        return lgb.train(params, dtrain, num_boost_round=num_boost_round,
                         valid_sets=valid_sets, callbacks=callbacks)

    @staticmethod
    def _direction_tree_params(per_quantile_params: dict) -> dict:
        """Extract objective-agnostic tree params from the p50 quantile config
        to seed the directional classifier."""
        src = per_quantile_params.get(0.5, {}) if per_quantile_params else {}
        keys = ("num_leaves", "learning_rate", "max_depth", "min_data_in_leaf",
                "lambda_l1", "lambda_l2")
        return {k: src[k] for k in keys if k in src}

    # ------------------------------------------------------------------
    # Conformal band state
    # ------------------------------------------------------------------

    # Fallback sigma for items with no usable 60-day history, and the initial
    # clip bounds before a fit has measured the real distribution. Overwritten
    # by train() and restored by load_models().
    SIGMA_FALLBACK_DEFAULT = 0.15
    SIGMA_FLOOR_DEFAULT = 0.01
    SIGMA_CAP_DEFAULT = 2.0

    # Fewest usable rows q_hat may be fitted on. Below this the conformal
    # quantile is decided by a handful of tail draws. Falling back to a default
    # instead of raising would put an unvalidated band in front of users, which
    # is the one thing this design refuses to do.
    MIN_CALIBRATION_ROWS = 50

    def _init_conformal_state(self) -> None:
        """Sigma clip bounds and fallback, persisted with the model.

        q_hat is calibrated against CLIPPED sigmas, so serving must clip
        identically or the coverage guarantee does not transfer.
        """
        self.sigma_clip: Dict[str, float] = {
            "floor": self.SIGMA_FLOOR_DEFAULT,
            "cap": self.SIGMA_CAP_DEFAULT,
            "fallback": self.SIGMA_FALLBACK_DEFAULT,
        }
        # Which centre each horizon's q_hat was fitted around: "served" (the
        # classifier-recentred mid predict() publishes) or "q50" (a mid that is
        # only served when no classifier exists). Provenance, not a switch —
        # the served centre needs an out-of-fold direction call, which
        # CV_DIAGNOSTIC_CLASSIFIER=0 does not produce, so "q50" is a reachable
        # state and a reader of the artifact has to be able to see it.
        self.conformal_centre: Dict[int, str] = {}
        # Which DENOMINATOR each horizon's q_hat was fitted on: "served" (the
        # smoothed anchor `predict` quotes from and the backtest scores in) or
        # "raw_anchor" (the training label's own denominator, which over-covers).
        # Provenance for the same reason `conformal_centre` is: the fallback is
        # reachable whenever the conformal frame predates
        # `calibration_target_col`, and a reader of the artifact must be able to
        # tell which width they are looking at.
        self.conformal_basis: Dict[int, str] = {}
        # The EXPONENT each horizon's q_hat was fitted at, and the one `band`
        # must serve it with. NOT provenance -- this one is load-bearing
        # arithmetic. Missing means 1.0, which is every artifact written before
        # 2026-08-12 and every run with SIGMA_EXPONENT off; `sigma` is ~0.07 so
        # a q_hat applied one exponent off is wrong by roughly 5x. Written
        # together with `conformal_calibration` or not at all.
        self.conformal_beta: Dict[int, float] = {}

        # The learned band scale (LEARNED_SCALE=1), and the same matched-pair
        # rule as `conformal_beta`: a q_hat calibrated against a learned scale
        # is in that scale's units, so serving it against `sigma` gives an
        # unrelated band rather than a degraded one. All four move together or
        # none of them do, and an artifact with no scale model falls back to
        # `sigma ** beta` -- which is what every pre-2026-08-12 artifact is.
        self.scale_models: Dict[int, Any] = {}
        self.scale_norm: Dict[int, float] = {}
        self.scale_clip: Dict[int, tuple] = {}
        self.scale_features: Dict[int, List[str]] = {}

    def band_scale(self, horizon: int, rows: pd.DataFrame, sigma):
        """The learned scale for a set of rows, or None to use `sigma ** beta`.

        THE ONE ACCESSOR, for the same reason `band_beta` is one: a call site
        that reaches for `scale_models[h]` directly gets a KeyError on an
        artifact written before this existed, and one that forgets `scale_norm`
        or the clip serves a scale in different units from the one `q_hat` was
        calibrated against.

        Returns None -- not NaN, not sigma -- when this horizon has no scale
        model, so the caller passes `learned_scale=None` and `resolve_scale`
        takes the pre-existing path.
        """
        booster = self.scale_models.get(horizon)
        if booster is None:
            return None
        X = self._scale_feature_frame(rows, sigma, horizon)
        want = self.scale_features.get(horizon)
        if want is not None and list(X.columns) != list(want):
            # Refuse rather than reindex. A scale model scored against a
            # different column set returns a plausible number computed from the
            # wrong features, and the band it produces looks entirely normal.
            logger.warning(
                f"  {horizon}d learned scale: served columns do not match the "
                f"{len(want)} the model was fitted on — falling back to sigma "
                f"for this batch rather than scoring the wrong features."
            )
            return None
        s = scale_model.predict_scale(booster, X,
                                      clip=self.scale_clip.get(horizon),
                                      fallback=np.asarray(sigma, dtype=float))
        return s * float(self.scale_norm.get(horizon, 1.0))

    def band_beta(self, horizon: int) -> float:
        """The exponent to build this horizon's band with. Always a finite float.

        One accessor so that no call site can reach for `conformal_beta[h]` and
        get a KeyError on an old artifact, or a NaN into a served half-width.
        """
        b = self.conformal_beta.get(horizon, conformal.BETA_NEUTRAL)
        return (conformal.BETA_NEUTRAL if not np.isfinite(b) else float(b))

    def _check_artifact_version(self, meta: dict) -> None:
        """Fail closed on any artifact not written by this exact scheme.

        Must run before any other field is read from `meta`. An old artifact
        has no `model_artifact_version` key at all (`meta.get` returns None,
        which never equals an int and so still raises) -- absence is exactly
        as incompatible as a mismatched version, not a reason to default.
        """
        found = meta.get("model_artifact_version")
        if found == self.MODEL_ARTIFACT_VERSION:
            # Same version, but the ByMykel bundle changes the feature SET
            # without changing the meaning of any persisted field, so the
            # version alone cannot catch it. A booster trained with the nine
            # columns, loaded with the flag off, would be scored against a
            # feature matrix that no longer has them -- a train/serve mismatch
            # of exactly the kind the v3 and v4 bumps exist to prevent.
            saved = bool(meta.get("bymykel_metadata", False))
            if saved != self.bymykel_metadata_enabled():
                raise IncompatibleModelArtifact(
                    f"saved model artifact was trained with "
                    f"BYMYKEL_METADATA={'1' if saved else '0'} but this process "
                    f"has it {'on' if self.bymykel_metadata_enabled() else 'off'}. "
                    f"The feature set differs by the nine ByMykel metadata "
                    f"columns, so loading it would serve a booster against a "
                    f"feature matrix it was not trained on. Retrain, or match "
                    f"the flag."
                )
            return
        raise IncompatibleModelArtifact(
            f"saved model artifact version {found!r} != expected "
            f"{self.MODEL_ARTIFACT_VERSION}. This cache predates the minimal "
            f"model, where conformal_calibration changed from a percentage-"
            f"point addend to a dimensionless sigma multiplier and sigma_clip "
            f"was added. Loading it here would silently serve a band computed "
            f"a different way. Retrain (mode=full) rather than loading it."
        )

    def _calibration_returns(self, frame, horizon: int, fallback):
        """The served-basis realised return for *frame*, for calibration only.

        Falls back to the training label when the column is absent, which is
        what a frame built by a caller that predates `calibration_target_col`
        looks like. That is a real degradation -- q_hat goes back to being
        fitted in the raw-anchor basis and the band over-covers -- so it WARNS
        rather than substituting quietly.
        """
        col = calibration_target_col(horizon)
        if not self.conformal_served_basis_enabled():
            # The shipped default. Named rather than silent: this IS the
            # incoherence, and a reader of `conformal_basis` has to be able to
            # tell "the arm is off" from "the frame had no column".
            self.conformal_basis[horizon] = "raw_anchor"
            return np.asarray(fallback, dtype=float)
        if frame is not None and col in getattr(frame, "columns", ()):
            self.conformal_basis[horizon] = "served"
            return frame[col].to_numpy(dtype=float)
        self.conformal_basis[horizon] = "raw_anchor"
        logger.warning(
            f"  CONFORMAL_SERVED_BASIS=1 but the {horizon}d conformal set has "
            f"no {col!r}: q_hat falls back to the TRAINING label, whose "
            f"denominator is the raw anchor quote and not the smoothed anchor "
            f"predict() quotes from. The arm is NOT in effect for this horizon. "
            f"Rebuild the frame through prepare_targets()."
        )
        return np.asarray(fallback, dtype=float)

    def _conformal_records(self, mid_ret, actual_ret, sigma,
                           current_price,
                           direction_class=None,
                           residual_actual_ret=None,
                           row_index=None) -> List[Dict[str, float]]:
        """Per-row calibration records, shared by the CV and holdout paths.

        `residual_pct` / `sigma` are what q_hat is fitted on, `mid_ret` is what
        the band is rebuilt from once q_hat exists, and `change_pct` / `hit` are
        what `_calibrate_confidence` thresholds on. Both callers go through here
        so a q_hat fitted on one path is never on a different footing from the
        other.

        `direction_class` is the out-of-fold directional call, and it is what
        makes q_hat cover the band that is actually served. `predict` builds the
        band around the q50 mid and then hands it to
        `_recenter_on_direction`, which moves the centre to ±|mid| (or pins it
        to 0) while preserving both half-widths — so on every row where the
        classifier disagrees with the q50's sign, the served centre is displaced
        by up to twice |mid| and the width is unchanged. Fitting q_hat on
        residuals to the q50 mid therefore calibrates a band nobody is served:
        measured on a held-out split, 59.8% coverage against an 80% target,
        while the never-served q50-centred band covers 79.5%.

        **Only `residual_pct` moves.** `mid_ret`, `change_pct` and `hit` stay on
        the q50 mid deliberately: they feed `_calibrate_confidence`, whose
        thresholds are consumed by `_compute_confidence`, which runs *only* on
        the no-classifier fallback path — and that path serves the q50 mid, with
        no recentring. Putting them on the served centre would fit thresholds for
        a signal that path never publishes.

        `residual_actual_ret` is the realised return on the SERVED basis, and
        it is what `residual_pct` is measured from. `actual_ret` stays the
        training label and keeps feeding `hit` and `change_pct`, which are
        `_calibrate_confidence`'s inputs -- moving those too would change a
        second thing under cover of this one. When it is None the residual falls
        back to `actual_ret`, which is the pre-2026-08-12 behaviour and
        over-covers; see `calibration_target_col`.

        `row_index` is the calibration row's label in the frame it came from,
        carried so a learned scale can reach the features that produced it
        (`models/scale_model.py`). It is the row's identity and nothing else:
        no arithmetic reads it, and `_calibrate_conformal` ignores it entirely
        unless LEARNED_SCALE is on. Passing it costs one array lookup per row.

        ⚠️ It is positional. Every array reaching this function is aligned to
        the same frame -- in the CV path `fold_p50`, `actual_returns`,
        `fold_sigma` and `current_prices` are all built from `val_df` -- so
        `row_index[i]` names the row that produced record `i`. An array that is
        not on that footing would attach one item's error to another item's
        features, which would poison the scale silently and plausibly.

        Rows whose mid or current price is zero are dropped: range_pct and
        change_pct are undefined there.
        """
        mid = np.asarray(mid_ret, dtype=float)
        actual = np.asarray(actual_ret, dtype=float)
        resid_actual = (actual if residual_actual_ret is None
                        else np.asarray(residual_actual_ret, dtype=float))
        sig = np.asarray(sigma, dtype=float)
        curr = np.asarray(current_price, dtype=float)

        mid_price = curr * (1.0 + mid / 100.0)
        keep = (mid_price != 0) & (curr != 0) & np.isfinite(resid_actual)

        tol = DIRECTION_FLAT_TOLERANCE_PCT
        hit = (self._direction_classes(actual, tol)
               == self._direction_classes(mid, tol)).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            change_pct = np.abs(mid_price - curr) / curr

        # The real serving function, not a re-derivation of it: the two centres
        # cannot drift apart if only one place knows how to compute one. Low and
        # high are irrelevant here (half-widths are preserved), so the mid is
        # passed for all three and only the mid is read back.
        served_mid = None
        if direction_class is not None:
            _, served_mid, _ = self._recenter_on_direction(
                mid, mid, mid, direction_class)

        centre = mid if served_mid is None else served_mid
        idx = None if row_index is None else np.asarray(row_index)
        if idx is not None and idx.shape[0] != mid.shape[0]:
            # Loud, because the silent version attaches one item's error to
            # another item's features and every downstream number still looks
            # reasonable.
            raise ValueError(
                f"row_index has {idx.shape[0]} entries against {mid.shape[0]} "
                f"calibration rows. It is positional: a mismatched length means "
                f"the caller's arrays are not on one frame."
            )
        records = []
        for i in np.flatnonzero(keep):
            rec = {
                "mid_ret": float(mid[i]),
                "residual_pct": float(resid_actual[i] - centre[i]),
                "sigma": float(sig[i]),
                "change_pct": float(change_pct[i]),
                "hit": float(hit[i]),
            }
            if served_mid is not None:
                rec["served_mid_ret"] = float(served_mid[i])
            if idx is not None:
                rec["row_index"] = idx[i]
            records.append(rec)
        return records

    def _holdout_conformal_records(self, horizon: int, X_val, y_val,
                                   val_set) -> pd.DataFrame:
        """Calibration records from the single validation holdout.

        LAST RESORT, not a peer of the CV path. `X_val` is the early-stopping
        `dval` and the set Optuna scores HP on, so a q_hat fitted here is
        measured on rows the model was selected against: its residuals are
        optimistically small and the resulting band under-covers. The caller
        logs that at WARNING.

        Reached only when CV genuinely cannot run (fewer than 2 expanding-window
        folds) or when a local run opts in with SKIP_CV=1. It exists so a
        short-history horizon degrades loudly instead of failing the retrain.

        Raises rather than defaulting — a fabricated q_hat is worse than a
        weak one.
        """
        p50 = self._get_ensemble_prediction(horizon, 0.5, X_val.values)
        if p50 is None:
            raise RuntimeError(
                f"No median model for {horizon}d, so the conformal band cannot "
                f"be calibrated from the holdout. Refusing to serve an "
                f"uncalibrated band."
            )
        # N1's offset, or q_hat would be fitted on residuals to a mid the
        # serving path does not use. This fallback path is already the weaker of
        # the two calibrations; it must not also be on a different footing.
        holdout_offset = self._naive_offset(val_set)
        if holdout_offset is not None:
            p50 = p50 + holdout_offset
        # Same reason, for the same served mid: predict() recentres on this
        # classifier's call, so the residual has to be measured there. The call
        # is IN-SAMPLE here — the classifier was fitted on the full training set,
        # X_val included — which is the defect this whole path already carries
        # and warns about. A coherent centre on an optimistic residual beats an
        # incoherent one.
        holdout_clf = self.direction_models.get(horizon)
        holdout_cls = (None if holdout_clf is None
                       else holdout_clf.predict(X_val).argmax(axis=1))
        records = self._conformal_records(
            p50, y_val.values, self._sigma_for_rows(val_set),
            val_set["price"].values,
            direction_class=holdout_cls,
            residual_actual_ret=self._calibration_returns(
                val_set, horizon, y_val.values),
            # Carried here too. Without it a learned scale would silently have
            # no features on exactly the horizons that fell back to this path --
            # the short-history ones, which are the hardest to size a band for.
            row_index=val_set.index.to_numpy(),
        )
        if len(records) < self.MIN_CALIBRATION_ROWS:
            raise RuntimeError(
                f"Holdout for {horizon}d yielded {len(records)} usable "
                f"calibration rows (need >= {self.MIN_CALIBRATION_ROWS}). "
                f"Refusing to fabricate a conformal q_hat."
            )
        return pd.DataFrame(records)

    def _sigma_tilt_audit(self, horizon: int,
                          records_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Is the band tilted across `sigma`, and does an exponent flatten it?

        REPORTED ONLY. `q_hat` above is what serves, unchanged, and nothing in
        `predict` reads any of this.

        THE CONFIRM READ for `docs/changelog/2026-08-12-the-band-is-tilted-in-sigma.md`,
        which measured the tilt offline on a MODEL-FREE proxy (`r̂ = 0`, valid to
        first order because predicted `|return|` is median 0.95% against
        half-widths of 10-31%). That read found the elasticity at
        **0.408 / 0.401 / 0.363 / 0.327** and level-matched coverage ramping
        **62->95%** (h=3) to **58->98%** (h=30) across sigma deciles. It
        contradicts `2026-08-12-conformal-basis-follows-serving.md`, which read
        **0.798 / 0.692 / 1.034 / 1.150** off `sigma` RECONSTRUCTED as
        `half_pct / q_hat` from ~20K prod rows and called the tilt second-order.

        This runs on the real OOF residuals -- the population `q_hat` is actually
        fitted on -- so it is the tiebreak, and the exponent must not be
        implemented before it lands. The two candidate sizes differ by 3x, and
        three of the five causes already excluded in this investigation were
        refuted by a SIGN rather than a size.

        ⚠️ Read `elasticity_heldout` in preference to `elasticity`. The pooled
        legs fit and score the exponent on the same rows; the held-out leg fits it
        on every fold but the last and scores it on the last, which is the only
        one of the two that can fail. It needs `fold` on the records, so it is
        absent on the `SKIP_CV=1` holdout path -- and that path's q_hat is already
        in-sample for early stopping, so nothing there is a confirmation of
        anything.
        """
        resid = records_df["residual_pct"].to_numpy(dtype=float)
        sigma = records_df["sigma"].to_numpy(dtype=float)
        if resid.size < self.MIN_CALIBRATION_ROWS:
            return None

        beta = conformal.elasticity(resid, sigma)
        if not np.isfinite(beta):
            return None

        # The clip is the leading alternative explanation and it is cheap to
        # exclude: rows pinned at the floor or the cap carry a sigma detached
        # from the item's volatility, so if they are what produces the tilt the
        # remedy is the clip and not the exponent. Offline this moved the
        # elasticity 0.389 -> 0.395 at h=3 on 1.2% of rows.
        floor = float(self.sigma_clip.get("floor", 0.0))
        cap = float(self.sigma_clip.get("cap", 0.0))
        clipped = np.isclose(sigma, floor) | np.isclose(sigma, cap)
        beta_unclipped = conformal.elasticity(resid[~clipped], sigma[~clipped])

        prof_1, err_1, _ = conformal.coverage_by_sigma_stratum(
            resid, sigma, exponent=1.0)
        prof_b, err_b, _ = conformal.coverage_by_sigma_stratum(
            resid, sigma, exponent=beta)

        out: Dict[str, Any] = {
            "elasticity": round(beta, 4),
            "elasticity_unclipped": (None if not np.isfinite(beta_unclipped)
                                     else round(beta_unclipped, 4)),
            "pct_rows_clipped": round(100.0 * float(clipped.mean()), 3),
            "n_calibration_rows": int(resid.size),
            # Level-matched, so marginal coverage is exactly NOMINAL_COVERAGE in
            # both and only the SPREAD is comparable.
            "decile_cov_beta1": [None if not np.isfinite(v) else round(v, 4)
                                 for v in prof_1],
            "stratum_err_pp_beta1": round(err_1, 3),
            "decile_cov_beta_fit": [None if not np.isfinite(v) else round(v, 4)
                                    for v in prof_b],
            "stratum_err_pp_beta_fit": round(err_b, 3),
            "elasticity_heldout": None,
            "stratum_err_pp_beta1_heldout": None,
            "stratum_err_pp_beta_heldout": None,
            "heldout_fold": None,
        }

        if "fold" in records_df.columns:
            folds = records_df["fold"].to_numpy()
            last = folds.max()
            fit, test = (folds < last), (folds == last)
            if (fit.sum() >= self.MIN_CALIBRATION_ROWS
                    and test.sum() >= self.MIN_CALIBRATION_ROWS):
                beta_fit = conformal.elasticity(resid[fit], sigma[fit])
                if np.isfinite(beta_fit):
                    _, e1, _ = conformal.coverage_by_sigma_stratum(
                        resid[test], sigma[test], exponent=1.0)
                    _, eb, _ = conformal.coverage_by_sigma_stratum(
                        resid[test], sigma[test], exponent=beta_fit)
                    out["elasticity_heldout"] = round(beta_fit, 4)
                    out["stratum_err_pp_beta1_heldout"] = round(e1, 3)
                    out["stratum_err_pp_beta_heldout"] = round(eb, 3)
                    out["heldout_fold"] = int(last)

        def _prof(vals) -> str:
            return " ".join("--" if v is None else f"{v * 100:.0f}"
                            for v in vals)

        logger.info(
            f"  Sigma-tilt audit ({horizon}d): elasticity={beta:.3f} "
            f"(unclipped {out['elasticity_unclipped']}, "
            f"{out['pct_rows_clipped']}% of {resid.size:,} rows clipped) | "
            f"level-matched decile coverage beta=1: [{_prof(prof_1)}] "
            f"err={err_1:.2f}pp -> beta={beta:.3f}: [{_prof(prof_b)}] "
            f"err={err_b:.2f}pp | HELD-OUT fold {out['heldout_fold']}: "
            f"beta={out['elasticity_heldout']} "
            f"err {out['stratum_err_pp_beta1_heldout']}pp -> "
            f"{out['stratum_err_pp_beta_heldout']}pp. "
            f"elasticity BELOW 1 means sigma over-corrects and the LOW-sigma "
            f"deciles are the under-covered ones. Reported only; nothing is "
            f"served from it, and the held-out leg is the one that can fail."
        )
        return out

    def _fit_learned_scale(self, horizon: int, records_df: pd.DataFrame,
                           feature_frame: Optional[pd.DataFrame],
                           resid, sigma):
        """Cross-fitted scale for calibration, plus the model that will serve.

        Returns the per-row scale `q_hat` should be calibrated against, or None
        to leave the band on `sigma ** beta`.

        **Two different scales come out of this, and conflating them is the
        whole trap.** `q_hat` is calibrated against the CROSS-FITTED scale --
        every row scored by a model that never saw its fold -- because a scale
        fitted on the same residuals it normalises matches them better than it
        will match a served item's, which makes `q_hat` too small and the band
        under-cover in production while looking flawless offline. Serving then
        uses a FINAL model fitted on all folds, which is the better estimator
        but is not the one `q_hat` was measured against.

        The residual mismatch is a LEVEL, and it is normalised away: `q_hat`
        absorbs any constant factor on the scale (see `scale_model`), so
        matching the final model's median to the cross-fitted median leaves only
        the shape difference, which is the part that is genuinely better. That
        ratio is persisted as `scale_norm` and applied at serve time.
        """
        if not scale_model.enabled():
            return None
        if feature_frame is None or "row_index" not in records_df.columns:
            logger.warning(
                f"  {horizon}d LEARNED_SCALE=1 but the calibration rows carry "
                f"no feature reference — falling back to sigma. This is the "
                f"single-holdout path, which cannot cross-fit anyway."
            )
            return None
        if "fold" not in records_df.columns:
            logger.warning(
                f"  {horizon}d LEARNED_SCALE=1 but the calibration rows carry "
                f"no fold labels, so the scale cannot be cross-fitted and a "
                f"q_hat fitted against it would be optimistic. Falling back to "
                f"sigma."
            )
            return None
        if not feature_frame.index.is_unique:
            # Non-unique labels make `.loc` fan out, silently pairing residuals
            # with the wrong rows. Unique today because `prepare_targets` ends
            # in a column merge -- an incidental fact, hence the check.
            raise RuntimeError(
                f"the {horizon}d feature frame has a non-unique index, so "
                f"calibration rows cannot be matched to their features. The "
                f"learned scale would silently pair residuals with the wrong "
                f"items; refusing to fit it."
            )

        idx = records_df["row_index"].to_numpy()
        rows = feature_frame.loc[idx]
        X = self._scale_feature_frame(rows, sigma, horizon)

        t0 = time.time()
        cross, n_models = scale_model.cross_fit(
            X, resid, records_df["fold"].to_numpy(), fallback=sigma)
        if n_models == 0:
            logger.warning(
                f"  {horizon}d learned scale: no fold produced a usable model; "
                f"falling back to sigma."
            )
            return None

        final = scale_model.fit(X, resid)
        if final is None:
            return None

        raw = scale_model.predict_scale(final, X, clip=None, fallback=sigma)
        ok = np.isfinite(raw) & (raw > 0) & np.isfinite(cross) & (cross > 0)
        norm = (float(np.median(cross[ok]) / np.median(raw[ok]))
                if ok.any() else 1.0)
        served = raw * norm

        self.scale_models[horizon] = final
        self.scale_norm[horizon] = norm
        self.scale_clip[horizon] = scale_model.clip_bounds(served)
        self.scale_features[horizon] = list(X.columns)

        logger.info(
            f"  {horizon}d learned scale: {n_models} cross-fit models + 1 "
            f"serving model on {len(X):,} rows in {time.time() - t0:.1f}s, "
            f"norm={norm:.4f}, clip="
            f"[{self.scale_clip[horizon][0]:.5f}, "
            f"{self.scale_clip[horizon][1]:.5f}]. q_hat is calibrated against "
            f"the CROSS-FITTED scale and is dimensionally tied to it — never "
            f"compare it with a sigma-basis q_hat."
        )
        return cross

    def _scale_feature_frame(self, rows: pd.DataFrame, sigma,
                             horizon: int) -> pd.DataFrame:
        """The feature matrix the learned scale is fitted on AND served from.

        ONE function with two callers on purpose. A scale model trained on one
        column set and served against another is the train/serve skew this repo
        has paid for repeatedly, and the failure is quiet: LightGBM will happily
        score a frame whose columns mean something else and return a plausible
        number.

        `sigma` is included as a feature, which makes the learned scale a strict
        GENERALISATION of the current band rather than a competitor to it — the
        model can reproduce `s = sigma` if that is genuinely best, so a null
        result means "sigma was already the right variable" rather than "the
        model could not see it". That is what makes a null informative here.
        """
        cols = self.horizon_feature_cols.get(horizon, self.feature_cols)
        cols = [c for c in cols if c in rows.columns]
        X = rows[cols].copy()
        X["sigma"] = np.asarray(sigma, dtype=float)
        return X

    def _calibrate_conformal(self, horizon: int,
                             records_df: pd.DataFrame,
                             feature_frame: Optional[pd.DataFrame] = None) -> float:
        """Fit q_hat on pooled OOF records and attach the width they imply.

        Two passes are unavoidable. The nonconformity score needs only the
        residual and sigma, but `range_pct` — what `_calibrate_confidence`
        thresholds on, and what it requires to be numeric — is a function of
        q_hat, which does not exist until the first pass finishes. Deriving it
        here rather than inside CV keeps the width the confidence thresholds
        are fitted on identical to the width predict() will serve.

        `feature_frame` is the frame the calibration rows came from, needed only
        by the learned scale (LEARNED_SCALE=1) so it can reach the features
        behind each residual. When it is absent, or the flag is off, the
        denominator is `sigma ** beta` exactly as before.

        Mutates `records_df` in place by adding `range_pct`, and returns q_hat.
        """
        resid = records_df["residual_pct"].to_numpy(dtype=float)
        sigma = records_df["sigma"].to_numpy(dtype=float)

        if scale_model.enabled() and self.sigma_exponent_enabled():
            # Caught here rather than deep in `resolve_scale`, so the run dies
            # at its first calibration instead of after training four horizons.
            raise RuntimeError(
                "LEARNED_SCALE=1 and SIGMA_EXPONENT=1 are both set. They are "
                "alternative band denominators, not layers: the exponent damps "
                "sigma's over-reaction and a fitted scale has none to damp. "
                "Applying both re-tilts the band the other way — measured in "
                "docs/changelog/2026-08-12-served-sigma-profile.md. Pick one."
            )

        learned = self._fit_learned_scale(horizon, records_df, feature_frame,
                                          resid, sigma)

        # The exponent, fitted on the SAME rows q_hat is, and stored in the same
        # breath. Nothing between these two statements may raise or return, or an
        # artifact could carry one without the other -- which is worse than
        # carrying neither, because a q_hat at the wrong exponent is wrong by ~5x
        # rather than merely uncorrected.
        beta = (conformal.fit_beta(resid, sigma,
                                   min_rows=self.MIN_CALIBRATION_ROWS)
                if self.sigma_exponent_enabled() else conformal.BETA_NEUTRAL)
        q_hat = conformal.calibrate(resid, sigma, conformal.ALPHA, beta,
                                    learned_scale=learned)
        self.conformal_calibration[horizon] = q_hat
        self.conformal_beta[horizon] = beta
        if self.sigma_exponent_enabled():
            if conformal.beta_was_clamped(resid, sigma,
                                          min_rows=self.MIN_CALIBRATION_ROWS):
                # The clamp binding means the measured elasticity left
                # [0.2, 1.0], which no window of this archive has ever produced.
                # Data problem, not a tuning outcome.
                logger.warning(
                    f"  {horizon}d sigma exponent CLAMPED to {beta:.4f} — the "
                    f"raw elasticity fell outside "
                    f"[{conformal.BETA_MIN}, {conformal.BETA_MAX}], which no "
                    f"window of this archive has produced. Treat this q_hat as "
                    f"suspect and check the calibration rows."
                )
            logger.info(
                f"  {horizon}d sigma exponent beta={beta:.4f} on "
                f"{resid.size:,} calibration rows — q_hat={q_hat:.4f} is "
                f"DIMENSIONALLY TIED to it and must never be compared with a "
                f"beta=1.0 q_hat."
            )
        # Read off the records rather than off a flag: whichever centre
        # `_conformal_records` measured the residual against is the one q_hat
        # covers, and only that builder knows which it was.
        self.conformal_centre[horizon] = (
            "served"
            if ("served_mid_ret" in records_df.columns
                and records_df["served_mid_ret"].notna().all())
            else "q50"
        )
        if (self.conformal_centre[horizon] == "q50"
                and self.direction_models.get(horizon) is not None):
            # Not a fallback worth passing over in silence: predict() recentres
            # whenever this classifier exists, so the band being calibrated is
            # not the band being served, and the 80% label on it is wrong.
            logger.warning(
                f"  {horizon}d q_hat was calibrated around the q50 mid, but a "
                f"directional classifier exists for this horizon and predict() "
                f"will recentre the mid on its call — so this band is not "
                f"centred where it is served. MEASURED 2026-08-11 and the "
                f"coverage cost is nil: three arms at two anchors agree within "
                f"1.1pp, because the displacement is bounded by 2*|mid| "
                f"(predicted |return| is median 0.95%) against a half-width of "
                f"q_hat*sigma, an order of magnitude larger. So this is an "
                f"incoherence to know about, NOT a reason to pay the 932s for "
                f"CV_DIAGNOSTIC_CLASSIFIER=1, and NOT the cause of a low "
                f"IntCov. See "
                f"docs/changelog/2026-08-11-conformal-centre-follows-serving.md."
            )

        mid = records_df["mid_ret"].to_numpy(dtype=float)
        # Same `beta` AND the same learned scale as the calibrate above, or the
        # width `_calibrate_confidence` fits its thresholds on is not the width
        # predict() will serve.
        low, high = conformal.band(mid, sigma, q_hat, beta,
                                   learned_scale=learned)
        # range_pct is (high_price - low_price) / mid_price. The current price
        # is a common factor and cancels, leaving the return-space width over
        # the mid growth factor. Rows with mid_ret == -100 (a zero mid price)
        # are already dropped by _conformal_records.
        records_df["range_pct"] = (high - low) / 100.0 / (1.0 + mid / 100.0)
        return q_hat

    def _sigma_for_rows(self, rows: pd.DataFrame) -> np.ndarray:
        """Per-item sigma for a feature frame, using the persisted clip bounds.

        Reads `price_std_60d` (engineered in engineer_features) and `price`.
        Both are present on every frame that reaches training or prediction,
        so this adds no feature-engineering pass.
        """
        std = rows["price_std_60d"] if "price_std_60d" in rows.columns \
            else pd.Series(np.nan, index=rows.index)
        return conformal.sigma_from_columns(
            price_std_60d=std.to_numpy(dtype=float),
            price=rows["price"].to_numpy(dtype=float),
            floor=self.sigma_clip["floor"],
            cap=self.sigma_clip["cap"],
            fallback=self.sigma_clip["fallback"],
        )

    @staticmethod
    def _fix_quantile_crossing(low: np.ndarray, mid: np.ndarray,
                                high: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enforce low <= mid <= high via isotonic regression (PAV for 3 points).

        For each item where quantiles cross, projects [low, mid, high] onto
        the non-decreasing constraint using the Pool-Adjacent-Violators
        algorithm. This preserves item-level interval width as much as
        possible, unlike a global average-half-width imputation.

        Returns (low_fixed, high_fixed) arrays with low_fixed <= mid <= high_fixed.
        """
        v0, v1, v2 = low.copy().astype(np.float64), mid.copy().astype(np.float64), high.copy().astype(np.float64)

        # Pattern A: v0 > v1 (first two cross)
        cross_01 = v0 > v1
        if cross_01.any():
            pool_01 = (v0[cross_01] + v1[cross_01]) * 0.5
            v0[cross_01] = pool_01
            v1[cross_01] = pool_01

        # Pattern B: after fixing A, check v1 > v2 (last two cross)
        cross_12 = v1 > v2
        if cross_12.any():
            pool_12 = (v1[cross_12] + v2[cross_12]) * 0.5
            v1[cross_12] = pool_12
            v2[cross_12] = pool_12

        # Pattern C: after fixing B, check again if A was re-broken (pooled
        # v1,v2 < original v0). Only possible when all three crossed.
        recross_01 = v0 > v1
        if recross_01.any():
            pool_all = (v0[recross_01] + v1[recross_01] + v2[recross_01]) / 3.0
            v0[recross_01] = pool_all
            v1[recross_01] = pool_all
            v2[recross_01] = pool_all

        return v0, v2

    def _blend_returns_with_prior(self, low_ret_arr: np.ndarray, mid_ret_arr: np.ndarray,
                                  high_ret_arr: np.ndarray, prior: Dict[str, np.ndarray],
                                  weight: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Blend current return-space predictions toward the prior day's.

        Returns the (low, mid, high) arrays after exponentially smoothing with
        the previous forecast. Items without a prior are left untouched.
        """
        mask = prior["mask"]
        if not mask.any() or weight <= 0:
            return low_ret_arr, mid_ret_arr, high_ret_arr
        mid_ret_arr = np.where(mask, (1 - weight) * mid_ret_arr + weight * prior["mid_ret"], mid_ret_arr)
        low_ret_arr = np.where(mask, (1 - weight) * low_ret_arr + weight * prior["low_ret"], low_ret_arr)
        high_ret_arr = np.where(mask, (1 - weight) * high_ret_arr + weight * prior["high_ret"], high_ret_arr)
        return low_ret_arr, mid_ret_arr, high_ret_arr

    def _fetch_prior_forecasts(self, item_ids: np.ndarray, horizon: int) -> Dict[str, np.ndarray]:
        """Fetch the most recent prior-day forecast per item for blending.

        Returns return-space predictions (percentage return vs the prior
        ``current_price``) aligned to ``item_ids``, with NaN where no prior
        forecast exists. Used by ``predict()`` to smooth daily flip-flopping.
        """
        result = {
            "mask": np.zeros(len(item_ids), dtype=bool),
            "low_ret": np.full(len(item_ids), np.nan),
            "mid_ret": np.full(len(item_ids), np.nan),
            "high_ret": np.full(len(item_ids), np.nan),
        }
        if self.db is None:
            return result
        # _now(), not date.today(): under a replay the "prior" day is the day
        # before the ANCHOR. Reading real forecasts dated after it would blend
        # the future into a backdated prediction.
        today = self._now().date()
        try:
            rows = self.db.execute(text("""
                SELECT item_id, price_low, price_mid, price_high, current_price, forecast_date
                FROM item_forecasts
                WHERE horizon_days = :h AND forecast_date < :today
            """), {"h": horizon, "today": today}).fetchall()
        except Exception as e:
            logger.warning(f"  Prior-forecast fetch failed ({e}); skipping blend.")
            return result
        if not rows:
            return result

        # Keep the latest forecast_date seen per item.
        best: Dict[int, tuple] = {}
        for r in rows:
            iid = int(r.item_id)
            if iid not in best or r.forecast_date > best[iid][4]:
                best[iid] = (r.price_low, r.price_mid, r.price_high, r.current_price, r.forecast_date)

        # Map Parquet string slugs → integer DB IDs
        try:
            slug_rows = self.db.execute(
                text("SELECT id, item_id FROM items WHERE is_backfilled = 1")
            ).fetchall()
            slug_to_id = {r.item_id: r.id for r in slug_rows}
        except Exception:
            slug_to_id = {}

        id_to_idx = {}
        for i, slug in enumerate(item_ids):
            iid = slug_to_id.get(str(slug))
            if iid is not None:
                id_to_idx[iid] = i
        for iid, vals in best.items():
            idx = id_to_idx.get(iid)
            if idx is None:
                continue
            low, mid, high, cur, _ = vals
            if not cur or cur <= 0 or mid is None:
                continue
            result["mask"][idx] = True
            cur_f = float(cur)
            result["mid_ret"][idx] = (float(mid) / cur_f - 1.0) * 100.0
            result["low_ret"][idx] = (float(low) / cur_f - 1.0) * 100.0 if low is not None else result["mid_ret"][idx]
            result["high_ret"][idx] = (float(high) / cur_f - 1.0) * 100.0 if high is not None else result["mid_ret"][idx]
        return result

    # Prediction consumes only the last few rows per item (tail(3) for the
    # smoothed current price, last() for the feature vector), yet the whole-frame
    # path engineers all 1460 days for every item just to slice that tail off.
    # At 8,691 items / 6.1M rows that peaks well past a 16GB CI runner and gets
    # SIGKILLed — see docs and runs 30226424193 / 30666903525 / 30668690592.
    PREDICT_TAIL_ROWS = 3

    # Item-days of history per item retained for prediction. Row-based, not
    # calendar-based: features mix 180-day calendar lag joins (see LAGS in
    # _compute_price_features) with 200-row positional rollings (the
    # `for window in [100, 200]` block) over a ~48%-dense archive, so a
    # 240-*day* cutoff could yield ~115 rows and silently change every rolling
    # feature. Voting collapses to one row per item-day, so the last N item-days
    # always span >= N calendar days — one parameter satisfies both.
    # Tailing is a no-op for items holding fewer item-days than this.
    #
    # ⚠ 240 covers every feature in the **served** set only. One feature in the
    # pipeline needs more: `market_return_30d_percentile` in
    # _apply_market_aggregates uses rolling(365), so a 240-item-day tail shifts
    # it (measured 0.9068 -> 0.8381 on a synthetic series). That is currently
    # harmless because its group, `cross_sectional`, is not in
    # FEATURE_GROUP_ALLOWLIST and the column is discarded before training. It is
    # NOT harmless if the allowlist ever widens —
    # test_cross_sectional_features_are_not_served pins that dependency and will
    # fail, at which point raise this constant past 368 rather than deleting the
    # test.
    PREDICT_TAIL_ITEM_DAYS = 240

    # Calendar prefilter for the predict fetch, purely to shrink the DuckDB scan
    # and the voting pass. Measured 2026-07-31 archive, 8,691 backfilled items,
    # share of eligible items still retaining min(own_item_days, 240):
    #
    #     365d  99.9770%   1.98M rows      548d  100.0000%   2.85M rows
    #     730d 100.0000%   3.60M rows     1460d  100.0000%   6.14M rows
    #
    # 365d is the smallest window clearing a 99.9% bar, but it shortens ~2 items
    # below their entitlement — the silent feature skew this truncation design
    # exists to avoid. 730d is the conservative 100% choice: still a 1.7x scan
    # reduction, with margin if per-item density falls. Re-measure before
    # lowering it; the script is in the plan's Task 5.
    PREDICT_FETCH_DAYS = 730

    def _tail_predict_frame(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the last PREDICT_TAIL_ITEM_DAYS item-days per item.

        Selects on distinct dates rather than on row position, so a frame that
        still carries intraday duplicates cannot yield fewer calendar days than
        the window promises. engineer_features resamples to one row per item-day
        itself, so keeping every row on a retained date is safe.
        """
        before = len(price_df)
        keep = (price_df[["item_id", "date"]]
                .drop_duplicates()
                .sort_values(["item_id", "date"])
                .groupby("item_id", sort=False, group_keys=False)
                .tail(self.PREDICT_TAIL_ITEM_DAYS))
        out = price_df.merge(keep, on=["item_id", "date"], how="inner")
        logger.info(
            f"  Predict tail: {before:,} -> {len(out):,} rows "
            f"(<= {self.PREDICT_TAIL_ITEM_DAYS} item-days per item)"
        )
        return out

    @property
    def _predict_chunk_items(self) -> int:
        """Items per chunk during prediction; 0 disables chunking."""
        return int(os.getenv("PREDICT_CHUNK_ITEMS", "1000"))

    def _engineer_features_chunked(self, price_df, events_df, eligible,
                                   item_first_dates=None) -> pd.DataFrame:
        """Engineer prediction features in item chunks with bounded memory.

        Two passes are required because the cross-sectional features are per-date
        means over ALL items: pass A accumulates those means from (sum, count)
        partials, pass B applies the resulting table and keeps only each item's
        tail. Costs ~2x the feature-engineering CPU to make peak memory a
        function of chunk size rather than catalogue size. Results are identical
        to the whole-frame path — both call _apply_market_aggregates with the
        same table.
        """
        item_ids = list(eligible)
        size = self._predict_chunk_items
        chunks = [item_ids[i:i + size] for i in range(0, len(item_ids), size)]
        logger.info(
            f"  Chunked feature engineering: {len(item_ids):,} items "
            f"in {len(chunks)} chunks of <= {size:,}"
        )

        def _chunk_frame(chunk):
            return self.engineer_features(
                price_df[price_df["item_id"].isin(chunk)], events_df,
                item_first_dates=item_first_dates,
            )

        # Pass A — per-date market aggregates across every item, and the
        # per-(date, tier) index the lead-lag feature reads. Both are
        # cross-sectional, so both must be accumulated over every chunk before
        # either can be applied; folding the tier table into the existing pass
        # keeps that at two passes rather than three.
        partials = []
        tier_partials = []
        want_tier_lead = self._tier_lead_served()
        for n, chunk in enumerate(chunks, 1):
            cdf = _chunk_frame(chunk)
            partials.append(self._accumulate_market_partials(cdf))
            if want_tier_lead:
                tier_partials.append(self._accumulate_tier_partials(cdf))
            del cdf
            gc.collect()
            logger.info(f"    market pass {n}/{len(chunks)}")
        market = self._market_from_partials(partials)
        tier_lagged = (self._tier_lead_from_partials(tier_partials)
                       if want_tier_lead else pd.DataFrame())

        # Pass B — apply the market table, then discard all but each item's tail.
        tails = []
        for n, chunk in enumerate(chunks, 1):
            cdf = _chunk_frame(chunk)
            cdf = self._apply_market_aggregates(cdf, market)
            cdf = self._add_supply_depth_features(cdf)
            if want_tier_lead:
                cdf = self._apply_tier_lead(cdf, tier_lagged)
            cdf = (
                cdf.sort_values(["item_id", "date"])
                .groupby("item_id", sort=False)
                .tail(self.PREDICT_TAIL_ROWS)
            )
            tails.append(cdf)
            del cdf
            gc.collect()
            logger.info(f"    feature pass {n}/{len(chunks)}")

        df = pd.concat(tails, ignore_index=True)
        logger.info(f"  Chunked engineering complete: {len(df):,} tail rows retained")
        return df

    def predict(self, item_ids: List[int] = None) -> pd.DataFrame:
        logger.info("Generating forecasts...")

        # Try to load cached engineered features first (major speedup)
        df = self._load_engineered_cache()

        if df is not None:
            logger.info(f"  Using cached engineered features ({len(df):,} rows)")
        else:
            logger.info("  No usable cache found — running full feature engineering")
            price_df = self.fetch_price_history(
                days_back=self.PREDICT_FETCH_DAYS, backfilled_only=True)

            # Skip items without a real recent series: snapshot-tier items keep
            # only a single latest row, and a "forecast" from one data point is
            # a meaningless constant that would still be written to the DB.
            day_counts = price_df.groupby("item_id")["date"].nunique()
            eligible = day_counts[day_counts >= self.PREDICT_MIN_HISTORY_DAYS].index
            skipped = price_df["item_id"].nunique() - len(eligible)
            price_df = price_df[price_df["item_id"].isin(eligible)]
            logger.info(
                f"  {len(eligible):,} items with >= {self.PREDICT_MIN_HISTORY_DAYS} days of history "
                f"({skipped:,} skipped)"
            )

            events_df = self.fetch_events()

            # Engineer only the history the features need. Must come after the
            # eligibility filter, which counts distinct days over full history,
            # and the first-seen dates must be captured before the truncation
            # that would otherwise make every item look 239 days old.
            item_first_dates = price_df.groupby("item_id")["date"].min()
            price_df = self._tail_predict_frame(price_df)

            if self._predict_chunk_items and len(eligible) > self._predict_chunk_items:
                df = self._engineer_features_chunked(
                    price_df, events_df, eligible, item_first_dates=item_first_dates)
            else:
                df = self.engineer_features(price_df, events_df,
                                            item_first_dates=item_first_dates)

                # Add cross-sectional features (same as training)
                df = self._add_cross_sectional_features(df)

                # Add supply depth features (same as training)
                df = self._add_supply_depth_features(df)

                if self._tier_lead_served():
                    df = self._add_tier_lead_features(df)

                # Save to cache for next predict run. Only the whole-frame path
                # writes it: the chunked frame keeps PREDICT_TAIL_ROWS per item,
                # which is narrower still than the PREDICT_TAIL_ITEM_DAYS window
                # this frame carries.
                self._save_engineered_cache(df)

        # Both gated transforms run BEFORE alignment, and must: alignment adds a
        # missing column as NaN and the median fill downstream turns that into a
        # constant, so a frame that reached here without them would be served
        # silently rather than loudly.
        #
        # The engineered cache is the live way for that to happen. Its key
        # fingerprints forecaster.py's bytes, so a code change invalidates it --
        # but flipping TIER_LEAD_FEATURE is an environment change the key cannot
        # see, and a cached frame written with the flag off is otherwise a
        # perfectly valid hit. Fail instead of median-filling a feature the
        # booster was fitted on.
        tier_lead_col = self.TIER_LEAD_FEATURES[0]
        if (tier_lead_col in self.feature_cols
                and tier_lead_col not in df.columns):
            raise RuntimeError(
                f"{tier_lead_col} is in the model's feature_cols but absent from "
                f"the prediction frame. The engineered cache was almost certainly "
                f"built with TIER_LEAD_FEATURE off; delete it (or set "
                f"ENGINEERED_CACHE=0) and re-run rather than serving a "
                f"median-filled constant."
            )

        # Over the concatenated frame, so the within-date cross-section is every
        # eligible item -- not the chunk. The chunked path returns per-item tails
        # of PREDICT_TAIL_ROWS, which still holds every item at the latest date,
        # and the latest date is the only one predict reads (`groupby.last()`
        # below).
        #
        # But NOT over every eligible item as the ranking POPULATION: training
        # ranks the >= $1 cohort (916 items of 5,536 on the 2026-08-10 frame)
        # and this frame is unfiltered, so the cohort goes in as the reference
        # and the rest are placed within it. Ranking the pooled frame was the
        # defect recorded in
        # docs/changelog/2026-08-10-instrument-panel-first-read.md.
        if self._cross_sectional_rank_served():
            if self._artifact_min_median_price is None:
                raise RuntimeError(
                    "the loaded artifact enables the cross-sectional rank "
                    "transform but records no `train_min_median_price`, so the "
                    "cohort its percentiles are relative to cannot be "
                    "reconstructed. Retrain rather than serve percentiles from "
                    "a population the booster never saw."
                )
            # [] is a real answer ("nothing was skipped"); None is not.
            if self._artifact_xs_rank_skipped is None:
                raise RuntimeError(
                    "the loaded artifact enables the cross-sectional rank "
                    "transform but records no `xs_rank_skipped_cols`. The "
                    "date-constant skip would then be re-derived from the "
                    "predict frame, which ranks a different set of columns "
                    "than training did (run 31440424106: 31/32 against 32/32). "
                    "Retrain."
                )
            cohort = self._reference_cohort_mask(df)
            n_cohort = df.loc[cohort, "item_id"].nunique()
            n_all = df["item_id"].nunique()
            logger.info(
                f"  cross-sectional rank reference: {n_cohort:,}/{n_all:,} items "
                f"at the artifact's >= ${self._artifact_min_median_price:g} floor"
            )
            trained = self._artifact_cohort_items
            if trained and abs(n_cohort - trained) > 0.25 * trained:
                logger.warning(
                    f"  ⚠ the served cohort is {n_cohort:,} items against "
                    f"{trained:,} at training — a >25% move means the frames "
                    f"disagree about the universe, not that the market changed"
                )
            df = self._apply_cross_sectional_ranks(
                df, list(self.feature_cols), reference_mask=cohort,
                skip_cols=self._artifact_xs_rank_skipped)

        # Align features with training columns (add missing, drop extras)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_cols + [c for c in df.columns if c not in self.feature_cols]]

        # Use median of last 3 days for current_price to filter spike artifacts.
        # Prediction features still come from the latest row (rolling windows
        # inside the features already provide smoothing), but the base price
        # used to convert percentage returns into dollar amounts is more
        # robust when averaged over a short window.
        df = df.sort_values(["item_id", "date"])

        # Span-bounded, matching the backtest's resolver. See
        # _smoothed_anchor_prices: the observations must be near the anchor,
        # not merely the last three rows on file.
        smoothed_price = pd.Series(
            self._smoothed_anchor_prices(df, pd.to_datetime(df["date"]).max()),
            name="_smoothed_price",
        ).to_frame()
        smoothed_price.index.name = "item_id"

        # The served `current_price`, and the only site that sets it. Which of
        # the two prices lands here is arm A of
        # docs/superpowers/plans/2026-08-11-serving-anchor-freshness.md.
        latest_rows = df.groupby("item_id").last().reset_index()
        latest_rows = latest_rows.merge(smoothed_price, left_on="item_id", right_index=True, how="left")

        # BEFORE the reassignment below, and that ordering is the whole point.
        # `_serving_base_price` overwrites `price` with the served base, which
        # under the shipped arm IS `_smoothed_price` -- so a disclosure computed
        # after this line would find `p == S` for every item and publish the
        # entire catalogue as clean.
        anchor_clean, anchor_wedge_pct = self._anchor_disclosure(
            latest_rows["price"], latest_rows["_smoothed_price"])

        latest_rows["price"], outlier_mask = self._serving_base_price(
            latest_rows["price"], latest_rows["_smoothed_price"])
        n_outliers = int(outlier_mask.sum())
        if n_outliers:
            # The mask is arm-invariant, so this count sizes the deviating
            # cohort the replay's dollar gate is read on. The arm has to appear
            # beside it: it leaves no trace in `meta.json` (the artifact is the
            # same one either way), so this line is the only record of which
            # price a stored replay number was scored against.
            gated = self.outlier_gated_anchor_enabled()
            logger.warning(
                f"  {n_outliers} items have latest price >"
                f"{100 * self.ANCHOR_OUTLIER_TOLERANCE:g}% from 3d median — "
                f"smoothing {'those items only' if gated else 'every item'} "
                f"(SERVE_OUTLIER_GATED_ANCHOR={'1' if gated else '0'})")
        latest_rows = latest_rows.drop(columns=["_smoothed_price"])

        if item_ids:
            latest_rows = latest_rows[latest_rows["item_id"].isin(item_ids)]

        # Build the full feature matrix with the union of all per-horizon feature
        # sets (each horizon may have pruned different features).
        all_feature_cols = sorted(set().union(
            *[set(cols) for cols in self.horizon_feature_cols.values()]
        )) if self.horizon_feature_cols else self.feature_cols

        latest_clean = latest_rows.reindex(columns=all_feature_cols, fill_value=0).replace([np.inf, -np.inf], np.nan)
        if not self.feature_medians.empty:
            medians_aligned = self.feature_medians.reindex(all_feature_cols)
            medians_aligned = medians_aligned.where(medians_aligned.notna(), latest_clean.median())
            X_batch = latest_clean.fillna(medians_aligned)
        else:
            X_batch = latest_clean.fillna(latest_clean.median())

        item_id_arr = latest_rows["item_id"].to_numpy()
        current_price_arr = latest_rows["price"].to_numpy()
        anchor_clean_arr = anchor_clean.to_numpy()
        anchor_wedge_arr = anchor_wedge_pct.to_numpy()
        generated_at = self._now()

        # Detect current market regime for regime-aware model selection
        current_regime = self._detect_current_regime(df)
        has_regime_models = any(r == current_regime for r, _, _ in self.regime_models)
        if current_regime in self.REGIMES:
            if has_regime_models:
                logger.info(f"  Current market regime: {current_regime} (using regime models)")
            else:
                logger.info(f"  Current market regime: {current_regime} (no regime models trained, using global)")
        else:
            logger.info(f"  Current market regime: {current_regime} (using global models)")

        # Track regime vs global model usage for diagnostics
        regime_count = 0
        global_count = 0

        # One row per item, filled in horizon by horizon.
        # Per ITEM, not per horizon: the anchor is a property of the price
        # frame on the forecast date, and all four horizons are quoted from it.
        agg = {
            iid: {
                "item_id": iid,
                "current_price": float(cur),
                "anchor_clean": bool(clean),
                "anchor_wedge_pct": (float(wedge) if np.isfinite(wedge)
                                     else None),
                "forecasts": {},
                "generated_at": generated_at,
            }
            for iid, cur, clean, wedge in zip(
                item_id_arr, current_price_arr,
                anchor_clean_arr, anchor_wedge_arr)
        }

        # Per-item conformal scale. Loop-invariant: it depends only on
        # latest_rows, which does not vary by horizon (q_hat does). Clipped with
        # self.sigma_clip — the same bounds q_hat was calibrated against, or the
        # coverage guarantee does not transfer.
        sigma_arr = self._sigma_for_rows(latest_rows)

        for horizon in self.HORIZONS:
            h_features = self.horizon_feature_cols.get(horizon, self.feature_cols)
            X_horizon = X_batch[h_features]
            preds = {}
            for q in self.QUANTILES:
                all_preds = []

                # Prefer regime-specific model, fall back to global. Skip any
                # model whose feature count doesn't match the current matrix
                # (e.g. stale models left in the dir from a prior feature set)
                # so a feature-schema change can never crash prediction.
                regime_key = (current_regime, horizon, q)
                if current_regime in self.REGIMES and regime_key in self.regime_models:
                    all_preds = self._predict_ensemble_safe(
                        self.regime_models[regime_key], X_horizon)
                    if all_preds:
                        regime_count += 1
                if not all_preds and (horizon, q) in self.models:
                    all_preds = self._predict_ensemble_safe(
                        self.models[(horizon, q)], X_horizon)
                    if all_preds:
                        global_count += 1

                if all_preds:
                    preds[q] = np.mean(all_preds, axis=0)

            # Only the median is served. Gating on the presence of 0.5 rather
            # than on a quantile count keeps this correct whatever QUANTILES
            # holds — the old three-quantile count check would skip every
            # horizon once the grid collapses to [0.5].
            if 0.5 not in preds:
                continue

            # Models predict percentage returns (e.g. 5.0 means +5%); the
            # conversion to price levels happens per item further below.
            p50_ret = preds[0.5]

            # N1: a booster fitted with `init_score = -return_1d` emits the
            # RESIDUAL to that baseline, so the served level is the residual plus
            # the baseline. Read off the artifact, not the environment — a warm
            # daily run that lost the env var would otherwise publish residuals
            # as forecasts with nothing in the output to give it away.
            # `latest_rows` is X_batch's source frame, so this is positional.
            naive_offset = self._naive_offset_served(latest_rows)
            if naive_offset is not None:
                p50_ret = p50_ret + naive_offset

            # Median from the single p50 model; band from locally-weighted split
            # conformal (sigma_arr, computed once above). A band symmetric about
            # the median cannot cross, so the isotonic repair this loop used to
            # run is unnecessary. That repair is still a static method on this
            # class for walkforward's baseline arm and the evaluate/ab_test
            # scripts — do not delete it.
            #
            # The band is unbounded below: a wide enough q_hat * sigma puts the
            # low leg under -100%, i.e. a negative price. _sanitize_forecasts is
            # what guarantees the served triple stays ordered and positive;
            # low_ret is deliberately NOT floored here, because truncating one
            # side would break the symmetry the coverage guarantee rests on.
            mid_ret_arr = p50_ret
            q_hat = self.conformal_calibration.get(horizon)
            if q_hat is None:
                raise RuntimeError(
                    f"no conformal calibration for horizon {horizon}d. The band "
                    f"cannot be constructed without q_hat; refusing to serve a "
                    f"forecast with a fabricated interval."
                )
            # A q_hat from the single-holdout fallback is served too. It is
            # fitted on the early-stopping/Optuna scoring set, so it is biased
            # low and the band under-covers — train() logs that at WARNING. A
            # band that under-covers is still more useful than no band, and the
            # path is unreachable in production (a 1460-day frame yields 8-9 CV
            # folds; the fallback needs fewer than 2).
            # `band_beta` and not `conformal_beta[horizon]`: an artifact written
            # before 2026-08-12 has no exponent at all, and 1.0 is the value that
            # reproduces the band it was calibrated for.
            # `band_scale` returns None on every artifact without a learned
            # scale model, which is all of them before 2026-08-12 -- and then
            # `band` divides by `sigma ** beta` exactly as it always has.
            # Per-horizon rather than loop-invariant like `sigma_arr`: each
            # horizon's residuals have their own size, so each has its own model.
            low_ret_arr, high_ret_arr = conformal.band(
                mid_ret_arr, sigma_arr, q_hat, self.band_beta(horizon),
                learned_scale=self.band_scale(horizon, latest_rows, sigma_arr))

            # Momentum fallback for weak horizons (14d/30d): serve the trailing
            # return as the median, keeping the model's calibrated interval
            # width. Ablation showed momentum >= the ML model at these horizons.
            if horizon in self.MOMENTUM_FALLBACK_HORIZONS:
                mom_col = f"return_{horizon}d"
                if mom_col in latest_rows.columns:
                    momentum_ret = latest_rows[mom_col].to_numpy(dtype=float)
                    low_ret_arr, mid_ret_arr, high_ret_arr = self._recenter_on_momentum(
                        low_ret_arr, mid_ret_arr, high_ret_arr, momentum_ret)

            # Directional classifier: the served up/flat/down call + confidence.
            # Probabilities computed now; the median is recentered on the call
            # AFTER all return-space corrections below, so the price stays
            # coherent with the reported direction.
            dir_class_arr = None
            dir_conf_arr = None
            clf = self.direction_models.get(horizon)
            if clf is not None:
                probs = clf.predict(X_horizon)
                dir_class_arr = probs.argmax(axis=1)
                dir_conf_arr = probs.max(axis=1)

            # NOTE: the conformal widening used to be applied here, as a
            # per-horizon percentage-point addend. It is not missing — q_hat is
            # now a multiplier of the per-item sigma and is applied where the
            # band is built above. Do not re-add a widening step at this point.

            # Forecast blending / directional smoothing: blend the current
            # return-space predictions with the previous day's forecast for the
            # same item+horizon. Reduces daily direction flip-flopping. No-ops
            # when no prior forecast exists (first run / retrain).
            _disabled = self.replay_disabled()
            if _disabled:
                logger.warning(
                    f"  REPLAY_DISABLE={sorted(_disabled)}: serving transforms "
                    f"skipped. This is an attribution replay, not a forecast.")
            if "blend" not in _disabled:
                prior = self._fetch_prior_forecasts(item_id_arr, horizon)
                low_ret_arr, mid_ret_arr, high_ret_arr = self._blend_returns_with_prior(
                    low_ret_arr, mid_ret_arr, high_ret_arr, prior, self.FORECAST_BLEND_WEIGHT)

            # Per-tier bias correction: threshold-based approach (preferred).
            # Recalibrates classification boundaries to match the true outcome
            # base rate instead of shifting mid_ret (which pushes predictions
            # into the flat dead-zone). Falls back to additive correction only
            # when no threshold data exists for any tier on this horizon.
            tier_thresholds = self.bias_thresholds.get(horizon, {})
            fallback_additive = not tier_thresholds
            corrections = self.bias_corrections.get(horizon, {})
            if "bias" in _disabled:
                corrections = {}
            if corrections and fallback_additive:
                mid_ret_arr = np.array(mid_ret_arr, dtype=np.float64, copy=True)
                low_ret_arr = np.array(low_ret_arr, dtype=np.float64, copy=True)
                high_ret_arr = np.array(high_ret_arr, dtype=np.float64, copy=True)
                for i, price in enumerate(current_price_arr):
                    tier = self._get_price_tier(float(price))
                    corr = corrections.get(tier, 0.0)
                    if corr != 0.0:
                        mid_ret_arr[i] += corr
                        low_ret_arr[i] += corr
                        high_ret_arr[i] += corr

            # Recenter the median on the classifier's call so the served price
            # is coherent with the reported direction. Applied last, after all
            # return-space corrections, preserving interval half-widths.
            if dir_class_arr is not None and "recenter" not in _disabled:
                low_ret_arr, mid_ret_arr, high_ret_arr = self._recenter_on_direction(
                    low_ret_arr, mid_ret_arr, high_ret_arr, dir_class_arr)

            _dir_name = {0: "down", 1: "flat", 2: "up"}
            fallback_n = 0
            fallback_flat = 0
            for i, iid in enumerate(item_id_arr):
                low_ret, mid_ret, high_ret = (float(low_ret_arr[i]),
                                               float(mid_ret_arr[i]),
                                               float(high_ret_arr[i]))
                current_price = float(current_price_arr[i])

                # Convert return predictions to price levels
                price_low = round(current_price * (1 + low_ret / 100), 2)
                price_mid = round(current_price * (1 + mid_ret / 100), 2)
                price_high = round(current_price * (1 + high_ret / 100), 2)

                if dir_class_arr is not None:
                    # Served signal: the directional classifier.
                    direction = _dir_name[int(dir_class_arr[i])]
                    confidence = ("high" if float(dir_conf_arr[i]) >= self.DIRECTION_CONFIDENCE_HIGH
                                  else "low")
                else:
                    # Fallback (no classifier): threshold-based on mid_ret.
                    tier = self._get_price_tier(float(current_price))
                    th = tier_thresholds.get(tier, {})
                    t_down = th.get("t_down", -DIRECTION_FLAT_TOLERANCE_PCT)
                    t_up = th.get("t_up", DIRECTION_FLAT_TOLERANCE_PCT)
                    if mid_ret > t_up:
                        direction = "up"
                    elif mid_ret < t_down:
                        direction = "down"
                    else:
                        direction = "flat"
                    confidence = self._compute_confidence(
                        price_mid, price_low, price_high, current_price, horizon=horizon)
                    fallback_n += 1
                    fallback_flat += direction == "flat"

                agg[iid]["forecasts"][horizon] = {
                    "low": price_low,
                    "mid": price_mid,
                    "high": price_high,
                    "direction": direction,
                    "confidence": confidence,
                }

            self._warn_no_classifier(horizon, fallback_n, fallback_flat)

        result_df = pd.DataFrame([r for r in agg.values() if r["forecasts"]])
        if not result_df.empty:
            result_df = self._sanitize_forecasts(result_df)

        total_used = regime_count + global_count
        if total_used > 0:
            pct = regime_count / total_used * 100
            logger.info(f"  Regime model usage: {regime_count}/{total_used} "
                        f"({pct:.1f}%) regime, {global_count}/{total_used} "
                        f"({100-pct:.1f}%) global")

        logger.info(f"  Forecasts generated for {len(result_df)} items")
        return result_df

    # Floor for a `low` leg that sanitization has to replace, as a fraction of
    # the median. Anchored on `mid` rather than on `current_price` precisely
    # because `mid` can sit well below `current_price` on a strong down
    # forecast: flooring at `current_price` is what produced low > mid. A
    # fraction of `mid` satisfies both requirements — strictly positive and
    # <= mid — by construction, and is scale-free across price tiers. The value
    # is arbitrary (any positive floor is), so the WARNING below, not the
    # number, is what makes a systematically clipping sigma visible.
    SANITIZE_LOW_FLOOR_FRAC = 0.01

    def _sanitize_forecasts(self, result_df: pd.DataFrame) -> pd.DataFrame:
        for h in self.HORIZONS:
            for key in ["low", "mid", "high"]:
                vals = np.array([r["forecasts"].get(h, {}).get(key, np.nan)
                                 for r in result_df.to_dict("records")])
                mask_bad = ~np.isfinite(vals) | (vals <= 0)
                if mask_bad.any():
                    current_prices = result_df["current_price"].values
                    vals[mask_bad] = current_prices[mask_bad]
                    clamped = []
                    for i in np.where(mask_bad)[0]:
                        cf = result_df.iloc[i]["forecasts"].get(h)
                        if cf is None:
                            continue
                        cf[key] = float(vals[i])
                        clamped.append(str(result_df.iloc[i]["item_id"]))
                        if key == "mid":
                            cf["direction"] = "flat"
                            cf["confidence"] = "low"
                    if clamped:
                        logger.warning(
                            f"  {len(clamped)} {h}d forecasts had a "
                            f"non-positive or non-finite '{key}' clamped to "
                            f"current_price: {clamped[:5]}"
                            f"{' ...' if len(clamped) > 5 else ''}"
                        )

            # Ordering is enforced structurally, because the clamping above is
            # per-leg and independent: a valid `mid` alongside a non-positive
            # `low` comes back as low = current_price, which can exceed mid.
            #
            # The conformal band is unbounded below — low_ret = mid_ret -
            # q_hat * sigma, with sigma free to sit at the persisted
            # 99th-percentile cap — so this is reachable for volatile items,
            # which are exactly the ones the normalized band exists to serve
            # better. Under the old [p10, p90] base plus a percentage-point
            # widening, low_ret never got near -100 and the clamp was
            # effectively dead code.
            #
            # Enforced here rather than by flooring low_ret at construction for
            # two reasons: this is the last layer before serving, so it must
            # hold for ANY upstream input (including a band that arrives
            # inverted for some other reason); and truncating low_ret would
            # break the band's symmetry about the median, which is the property
            # conformal calibrated its coverage guarantee on.
            reordered = []
            for i in range(len(result_df)):
                cf = result_df.iloc[i]["forecasts"].get(h)
                if cf is None:
                    continue
                mid = float(cf.get("mid", np.nan))
                if not np.isfinite(mid):
                    # Nothing coherent to anchor on; the pass above already
                    # flagged it and there is no ordering to restore.
                    continue
                low = float(cf.get("low", np.nan))
                high = float(cf.get("high", np.nan))
                orig_low, orig_high = low, high

                if np.isfinite(low):
                    low = min(low, mid)
                if not np.isfinite(low) or low <= 0:
                    low = mid * self.SANITIZE_LOW_FLOOR_FRAC if mid > 0 else mid
                if not np.isfinite(high) or high < mid:
                    high = mid

                cf["low"] = float(low)
                cf["high"] = float(high)
                # NaN != NaN, so a non-finite original counts as changed.
                if low != orig_low or high != orig_high:
                    reordered.append(str(result_df.iloc[i]["item_id"]))

            if reordered:
                logger.warning(
                    f"  {len(reordered)} {h}d forecasts violated "
                    f"low <= mid <= high and were reordered around the median: "
                    f"{reordered[:5]}{' ...' if len(reordered) > 5 else ''}. "
                    f"A sigma pinned at the clip cap or an oversized q_hat is "
                    f"the usual cause."
                )

        if "volume" in result_df.columns:
            zero_vol = result_df["volume"].fillna(0) == 0
            if zero_vol.any():
                for h in self.HORIZONS:
                    for i in np.where(zero_vol.values)[0]:
                        cf = result_df.iloc[i]["forecasts"].get(h)
                        if cf and cf.get("confidence") == "high":
                            cf["confidence"] = "low"
        return result_df

    def predict_single(self, item_id: int) -> Dict[str, Any]:
        results = self.predict(item_ids=[item_id])
        if results.empty:
            return {}
        return results.iloc[0].to_dict()

    @staticmethod
    def _predict_ensemble_safe(ensemble, X) -> list:
        """Predict from each member of an ensemble, skipping any model whose
        feature count doesn't match X (guards against stale models left in the
        model dir from a different feature schema). Returns a list of arrays."""
        models = ensemble if isinstance(ensemble, list) else [ensemble]
        ncols = X.shape[1]
        out = []
        for m in models:
            try:
                if m.num_feature() != ncols:
                    continue
                out.append(m.predict(X))
            except Exception as e:
                logger.warning(f"  Skipping incompatible model in ensemble: {e}")
        return out

    def _get_ensemble_prediction(self, horizon, q, X):
        """Get averaged prediction from LGB ensemble."""
        all_preds = []
        key = (horizon, q)

        # LGB
        if key in self.models:
            ensemble = self.models[key]
            if isinstance(ensemble, list):
                all_preds.extend(m.predict(X) for m in ensemble)
            else:
                all_preds.append(ensemble.predict(X))

        if not all_preds:
            return None
        return np.mean(all_preds, axis=0)

    def _cv_evaluate_horizon(self, tdf, horizon, per_quantile_params,
                             per_item_row_sampling: bool = False):
        """Run expanding-window CV for a single horizon.

        Trains a single model (no ensemble) per fold using the best hyperparams
        found by Optuna, collects OOF predictions, and returns pooled records
        for confidence calibration plus fold-level metrics.

        Args:
            tdf: DataFrame with features + targets (from prepare_targets).
            horizon: Horizon in days.
            per_quantile_params: Dict {q: base_params} with best HP merged.

        Returns:
            (oof_records, fold_metrics) where oof_records is a list of dicts
            with mid_ret/residual_pct/sigma/change_pct/hit and fold_metrics is
            a list of per-fold accuracy dicts.

            The records carry no `range_pct`: the band width depends on a q_hat
            that does not exist until the pooled records are calibrated. The
            caller computes q_hat from residual_pct/sigma and then derives
            range_pct in a second pass, so the width the confidence thresholds
            are fitted on is the width that will actually be served.
        """
        sorted_dates = sorted(tdf["date"].unique())
        # Embargo train dates within `embargo_days(horizon)` of each validation
        # window: a train row's target is observed `horizon` days later and the
        # anchor behind it carries 13 days further, so without this gap those
        # labels overlap the validation period (leakage).
        splits = self._compute_cv_splits(
            sorted_dates, purge_days=embargo_days(horizon))
        if len(splits) < 2:
            raise RuntimeError(
                f"CV produced {len(splits)} fold{'s' if splits else 's'} "
                f"(need >=2). Cannot evaluate {horizon}d horizon — "
                f"check that training data has enough distinct dates "
                f"({len(sorted_dates)} available)."
            )

        oof_records = []
        fold_metrics = []
        pt_records = []
        pt_records_clf = []

        cv_max_rows = self._cv_max_train_rows()
        for fold_id, (train_dates, val_dates) in enumerate(splits):
            train_df = tdf[tdf["date"].isin(train_dates)]
            val_df = tdf[tdf["date"].isin(val_dates)]

            # Same cap, same draw, same reasoning as _build_production_split:
            # sample randomly (never tail()) so the calendar window survives and
            # expanding-window CV is not silently disabled. val_df is never
            # thinned -- that would move the evaluation cohort.
            if len(train_df) > cv_max_rows:
                if per_item_row_sampling:
                    train_df = self._per_item_row_sample(train_df, cv_max_rows)
                else:
                    train_df = train_df.sample(
                        n=cv_max_rows, random_state=42).sort_values("date")
            self._record_cv_fold_train_rows(len(train_df))

            if len(val_df) < 50:
                continue

            # Only the median is consumed: the band is conformal now, so a
            # fold's p10/p90 predictions (where QUANTILES still asks for them)
            # have no reader.
            fold_p50 = None

            X_train_pre = train_df[self.feature_cols].replace([np.inf, -np.inf], np.nan)
            fold_medians = X_train_pre.median()
            X_train = X_train_pre.fillna(fold_medians)
            y_train = train_df[f"target_return_{horizon}d"]

            X_val = val_df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(fold_medians)
            y_val = val_df[f"target_return_{horizon}d"]

            # Sample weights (same as main training loop)
            train_w = self._compute_sample_weights(train_df, horizon)
            val_w = self._compute_sample_weights(val_df, horizon)

            # Build the fold's binned Dataset once; reuse across all quantiles.
            ds_params = {"max_bin": self.MAX_BIN, "feature_pre_filter": False}
            dtrain_kw = dict(params=ds_params)
            if train_w is not None:
                dtrain_kw["weight"] = train_w
            dval_kw = dict(params=ds_params)
            if val_w is not None:
                dval_kw["weight"] = val_w
            # N1's offset. `fold_val_offset` is added back to every prediction
            # below, so rank_ic, the fold DA, the PT records and the conformal
            # residuals all describe `model + baseline` -- the same quantity
            # predict() serves -- rather than the residual on its own.
            fold_train_offset = self._naive_offset(train_df)
            fold_val_offset = self._naive_offset(val_df)
            if fold_train_offset is not None:
                dtrain_kw["init_score"] = fold_train_offset
                dval_kw["init_score"] = fold_val_offset
            dtrain = lgb.Dataset(X_train, y_train, **dtrain_kw)
            dval = lgb.Dataset(X_val, y_val, reference=dtrain, **dval_kw)

            for q in self.QUANTILES:
                lgb_preds = []

                # LGB
                params = per_quantile_params.get(q, {}).copy()
                params["objective"] = "quantile"
                params["alpha"] = q
                params["metric"] = "quantile"
                params["verbosity"] = -1
                params["n_jobs"] = -1
                params["random_state"] = 42

                # Fixed rounds by default, for the reason in FIXED_BOOST_ROUNDS:
                # `dval` here is one fold's 21-day window, so early-stopping on
                # it stopped 9 of 33 folds at a single tree.
                fold_callbacks = [lgb.log_evaluation(0)]
                fold_valid = None
                if self._early_stopping_enabled():
                    fold_callbacks.insert(0, lgb.early_stopping(20))
                    fold_valid = [dval]
                model = lgb.train(
                    params, dtrain,
                    num_boost_round=self._boost_rounds(horizon, cv=True),
                    valid_sets=fold_valid,
                    callbacks=fold_callbacks,
                )
                fold_pred = model.predict(X_val)
                if fold_val_offset is not None:
                    fold_pred = fold_pred + fold_val_offset
                lgb_preds.append(fold_pred)

                pred = lgb_preds[0]
                if q == 0.5:
                    fold_p50 = pred

            # The median is the only prediction this function needs. Guarding on
            # p10/p90 here would `continue` on every fold once QUANTILES == [0.5],
            # emptying oof_records and silently skipping calibration entirely.
            if fold_p50 is None:
                continue

            current_prices = val_df["price"].values
            actual_returns = y_val.values
            # The realised return on the SERVED basis, for q_hat only. Every
            # other number in this fold -- directional accuracy, rank IC, the PT
            # records -- stays on `actual_returns`, the training label, so this
            # cannot move a metric. See `calibration_target_col`.
            cal_returns = self._calibration_returns(val_df, horizon, y_val)

            # Locally-weighted split conformal: the nonconformity score is the
            # absolute median residual normalized by the item's sigma. There is
            # no p10/p90 interval to measure exceedance against any more, and no
            # crossing to repair. `val_df` is the same row order as fold_p50, so
            # sigma aligns positionally.
            fold_sigma = self._sigma_for_rows(val_df)

            # Fold-level directional accuracy
            fold_hits = 0
            for i in range(len(val_df)):
                actual_ret = float(actual_returns[i])
                mid_ret = float(fold_p50[i])
                actual_dir = "up" if actual_ret > DIRECTION_FLAT_TOLERANCE_PCT else "down" if actual_ret < -DIRECTION_FLAT_TOLERANCE_PCT else "flat"
                pred_dir = "up" if mid_ret > DIRECTION_FLAT_TOLERANCE_PCT else "down" if mid_ret < -DIRECTION_FLAT_TOLERANCE_PCT else "flat"
                if pred_dir == actual_dir:
                    fold_hits += 1

            fold_acc = round(fold_hits / len(val_df) * 100, 1)

            # Naive baselines on the same val rows, for honest comparison:
            #  - persistence: random walk in price → predict 0% return (flat).
            #    Its accuracy is just the share of genuinely flat actuals.
            #  - momentum: predict the sign of the item's trailing `horizon`-day
            #    return (a backward-looking feature, no leakage). This is the
            #    real bar a directional forecaster must clear.
            persistence_acc = self._directional_accuracy(
                np.zeros(len(actual_returns)), actual_returns)
            momentum_acc = None
            mom_col = f"return_{horizon}d"
            if mom_col in val_df.columns:
                momentum_acc = self._directional_accuracy(
                    val_df[mom_col].to_numpy(dtype=float), actual_returns)

            # Directional classifier — the actually-served signal. Trained the
            # same way as the production model (mover-weighted 3-class) so this
            # fold accuracy reflects what predict() will deliver.
            # Vol-scaled labels were A/B-tested (2026-07-27) and did not beat
            # the fixed-band control; production stays fixed-band. Tooling
            # retained in scripts/ab_test_direction_labels.py.
            #
            # It is an INPUT to q_hat since 2026-08-11, and a diagnostic
            # otherwise. `predict` recentres the served mid on this call, so its
            # out-of-fold predictions are what `_conformal_records` measures the
            # calibration residual against (F1). Two consequences: the
            # confidence thresholds still never see it — they stay on the q50
            # mid, which is what the no-classifier fallback serves — and this
            # cost flag now moves the served BAND, which is why
            # `conformal_centre` is persisted rather than inferred.
            # It is also the single most expensive thing in a
            # retrain — 3 trees per round against the median model's 1, i.e.
            # 69% of a fold's fit cost and 32-37% of the whole run (measured
            # 2026-08-08). CV_DIAGNOSTIC_CLASSIFIER=0 skips it; CI sets that,
            # local and research runs keep it.
            actual_cls = self._direction_classes(actual_returns)  # FIXED ±0.5% yardstick
            classifier_acc = None
            pred_cls = None
            if self._cv_diagnostic_classifier_enabled():
                clf = self._fit_direction_classifier(
                    X_train, y_train, X_val, y_val, self.BOOSTING_TYPE,
                    self._direction_tree_params(per_quantile_params),
                    horizon=horizon,
                    sigma_train=None,
                    sigma_val=None,
                    tier_train=(train_df["price_tier"].to_numpy()
                                if "price_tier" in train_df.columns else None),
                    num_boost_round=self._boost_rounds(horizon, cv=True),
                    early_stopping=self._early_stopping_enabled())
                pred_cls = clf.predict(X_val).argmax(axis=1)
                classifier_acc = round(float((pred_cls == actual_cls).mean()) * 100, 1)

            # The same accuracy again over the production cohort only. The
            # figure above pools every price tier and the training frame is
            # ~83% tier-0, so it is approximately the penny-item score, while
            # the production headline is >=$1 (HEADLINE_MIN_TIER). Comparing
            # the two was comparing populations, and that cohort mismatch is
            # most of the "~20pp train/serve gap" five hypotheses failed to
            # explain (docs/superpowers/specs/2026-08-05-cv-cohort-parity-design.md).
            #
            # Additive, never a replacement: mean_classifier_acc feeds the
            # edge-vs-baseline trust gate and the confidence calibration, and
            # is the series every historical meta.json holds.
            #
            # None, not 0.0, when a fold holds no >=$1 rows — an empty
            # partition has no accuracy, and a zero reads as "scored nothing
            # right". Same rule score_cohort follows for its own partitions.
            classifier_acc_ge1 = None
            if pred_cls is not None and "price_tier" in val_df.columns:
                ge1 = val_df["price_tier"].to_numpy() >= HEADLINE_MIN_TIER
                if ge1.any():
                    classifier_acc_ge1 = round(
                        float((pred_cls[ge1] == actual_cls[ge1]).mean()) * 100, 1)

            # The bar that actually matters, and the one this CV never carried.
            # `constant_call` is the best single fixed call on THIS fold, which
            # is what backend/AGENTS.md invariant #4 requires beside any DA;
            # `realised_down_rate` says whether a given DA is impressive.
            # Measured 2026-08-08: corr(fold DA, fold constant-call DA) = 0.715,
            # R^2 0.51 — half of what mean_classifier_acc moves on is the fold's
            # realised direction mix, not the model.
            fold_records = self._direction_records(
                fold_p50, actual_returns, val_df["date"])
            _, fold_constant_call = constant_call_baseline(fold_records)
            fold_down_rate = realised_down_rate(fold_records)

            # Cross-sectional signal, on the SERVED cohort. This is the metric
            # the product needs: the dashboard ranks items within a date, so
            # within-date ordering is the thing to maximise, and it is immune to
            # the base rate that dominates DA. `naive_rank_ic` is the same
            # measurement for "rank by minus yesterday's return" — the one-line
            # baseline that beat this model at all four horizons on 2026-08-08.
            served = (val_df["price_tier"].to_numpy() >= HEADLINE_MIN_TIER
                      if "price_tier" in val_df.columns
                      else np.ones(len(val_df), dtype=bool))
            naive_pred = (-val_df["return_1d"].to_numpy(dtype=float)
                          if "return_1d" in val_df.columns else None)
            fold_rank_ic, rank_ic_dates = self._within_date_rank_ic_detail(
                fold_p50, actual_returns, val_df["date"], served)
            naive_rank_ic = None
            if naive_pred is not None:
                naive_rank_ic = self._within_date_rank_ic(
                    naive_pred, actual_returns, val_df["date"], served)

            # The same two numbers on the cohort where the label's denominator
            # is not contaminated -- `p[d] == S[d]`, so neither the raw basis
            # nor the smoothed one carries the wedge. Every arm this project
            # has ranked was read on the pooled figure above, and the
            # contamination (0.03-0.24 rank IC) is larger than every effect
            # being chased, so THIS is the pair a new arm is decided on.
            #
            # None, not the pooled value, when the mask is absent: an
            # `ab_test_*` frame built before the column existed cannot tell,
            # and falling back would publish the contaminated number under the
            # clean key. Same rule `classifier_accuracy_ge1` follows.
            rank_ic_tied = naive_rank_ic_tied = None
            rank_ic_tied_dates = 0
            tied_served = np.zeros(len(val_df), dtype=bool)
            if ANCHOR_TIED_COL in val_df.columns:
                tied_served = served & val_df[ANCHOR_TIED_COL].eq(True).to_numpy()
                rank_ic_tied, rank_ic_tied_dates = \
                    self._within_date_rank_ic_detail(
                        fold_p50, actual_returns, val_df["date"], tied_served)
                if naive_pred is not None:
                    naive_rank_ic_tied = self._within_date_rank_ic(
                        naive_pred, actual_returns, val_df["date"], tied_served)

            # C2: a within-date ranker on the same folds, read against the same
            # naive baseline and (primarily) the same tied cohort. Its scores are
            # ordinal — DA/MAE are undefined — so only rank IC is stored.
            lr_rank_ic = lr_rank_ic_tied = None
            lr_rank_ic_dates = lr_rank_ic_tied_dates = 0
            if self._lambdarank_enabled():
                lr_scores = self._lambdarank_fold_scores(
                    train_df, val_df, horizon, per_quantile_params)
                lr_rank_ic, lr_rank_ic_dates = \
                    self._within_date_rank_ic_detail(
                        lr_scores, actual_returns, val_df["date"], served)
                lr_rank_ic_tied, lr_rank_ic_tied_dates = \
                    self._within_date_rank_ic_detail(
                        lr_scores, actual_returns, val_df["date"], tied_served)

            fold_metrics.append({
                "fold": fold_id + 1,
                "train_start": str(train_dates[0]),
                "train_end": str(train_dates[-1]),
                "val_start": str(val_dates[0]),
                "val_end": str(val_dates[-1]),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_trees": int(model.num_trees()),
                "directional_accuracy": fold_acc,
                "classifier_accuracy": classifier_acc,
                "classifier_accuracy_ge1": classifier_acc_ge1,
                "persistence_accuracy": persistence_acc,
                "momentum_accuracy": momentum_acc,
                "constant_call_accuracy": (
                    None if fold_constant_call is None
                    else round(fold_constant_call, 1)),
                "realised_down_rate": (
                    None if fold_down_rate is None else round(fold_down_rate, 1)),
                "rank_ic": fold_rank_ic,
                "naive_rank_ic": naive_rank_ic,
                "rank_ic_dates": rank_ic_dates,
                "rank_ic_tied": rank_ic_tied,
                "naive_rank_ic_tied": naive_rank_ic_tied,
                "rank_ic_tied_dates": rank_ic_tied_dates,
                "n_tied": int(tied_served.sum()),
                "lr_rank_ic": lr_rank_ic,
                "lr_rank_ic_dates": lr_rank_ic_dates,
                "lr_rank_ic_tied": lr_rank_ic_tied,
                "lr_rank_ic_tied_dates": lr_rank_ic_tied_dates,
            })

            # Pooled records for the Pesaran-Timmermann test. Built from the
            # quantile-median SIGN so this costs nothing even when the
            # diagnostic classifier is skipped.
            pt_records.extend(fold_records)

            # The same pooling for the SERVED classifier, when it ran. Kept in a
            # separate stream rather than replacing the one above: both verdicts
            # are then published side by side, and neither key changes which
            # signal it describes depending on CV_DIAGNOSTIC_CLASSIFIER.
            if pred_cls is not None:
                pt_records_clf.extend(self._direction_records_from_classes(
                    pred_cls, actual_cls, val_df["date"]))

            # Build per-row records for pooled calibration. Same builder the
            # single-holdout path uses, so the two calibrations are comparable.
            # `pred_cls` is None unless CV_DIAGNOSTIC_CLASSIFIER=1, and then the
            # residual is measured against the q50 mid — honest, but a mid
            # predict() does not serve. `_calibrate_conformal` warns.
            fold_conformal = self._conformal_records(
                fold_p50, actual_returns, fold_sigma, current_prices,
                direction_class=pred_cls, residual_actual_ret=cal_returns,
                # `val_df` is a boolean-mask slice of `tdf` with no
                # `reset_index`, and `tdf` carries a unique RangeIndex out of
                # `prepare_targets`' merge -- so these labels index straight back
                # into the feature frame. Every array above is built from
                # `val_df`, which is what makes them positionally aligned to it.
                row_index=val_df.index.to_numpy())
            # Which fold each calibration row came from. `_calibrate_conformal`
            # ignores it -- the pooled q_hat is unchanged -- but without it the
            # sigma-tilt audit can only fit and score its exponent on the same
            # rows, and an in-sample elasticity is not a confirmation of one.
            for _rec in fold_conformal:
                _rec["fold"] = fold_id
            oof_records.extend(fold_conformal)

            # Per-fold q_hat, reported and never served. The pooled q_hat is
            # fitted across folds whose models saw 87,224 to 300,000 rows, while
            # the shipped model trains on the full TRAIN_FEATURE_ROWS budget --
            # so the calibration residuals come from systematically WEAKER
            # models than the one production serves, and split conformal's
            # exchangeability assumption does not hold across that gap. A pooled
            # q_hat is then conservative by construction and the served band
            # over-covers uniformly, which is the shape the 2026-08-12 panel has
            # (p80 of the served nonconformity score below 1 in 19 of 20 sigma
            # strata). This is the measurement that decides it: if q_hat falls
            # as the fold's training set grows, the pooled fit is inheriting the
            # early folds' weakness.
            #
            # Reported only. Nothing reads `fold_q_hat` to build a band -- the
            # remedy, if this confirms, is a calibration-set change and it needs
            # its own decision.
            fold_q_hat = None
            fold_beta = None
            if len(fold_conformal) >= self.MIN_CALIBRATION_ROWS:
                _fr = pd.DataFrame(fold_conformal)
                try:
                    # DELIBERATELY AT beta = 1.0, EVEN UNDER SIGMA_EXPONENT.
                    # The spec listed this as a site to convert; converting it is
                    # the wrong call. This series exists to test whether q_hat
                    # falls as a fold's training set grows, which needs every
                    # fold in ONE unit -- and the pooled beta does not exist yet
                    # here (it is fitted after CV, on the pooled records), so the
                    # only available exponent is each fold's own. Using it would
                    # make consecutive fold_q_hats incomparable and silently
                    # break the published 0.94/0.91/0.92/0.84x series and the
                    # expanding-window audit built on it.
                    fold_q_hat = float(conformal.calibrate(
                        _fr["residual_pct"].values, _fr["sigma"].values,
                        conformal.ALPHA, conformal.BETA_NEUTRAL))
                    # The new information instead: beta per fold. This is the
                    # quantity the 14d/30d dispute turns on -- whether the
                    # exponent drifts between folds enough to explain a weak
                    # held-out leg -- measured on real OOF residuals rather than
                    # the model-free panel. Reported only.
                    fold_beta = float(conformal.fit_beta(
                        _fr["residual_pct"].values, _fr["sigma"].values,
                        min_rows=self.MIN_CALIBRATION_ROWS))
                except ValueError:
                    # No finite nonconformity scores in this fold. Diagnostic
                    # only, so it must never take out the CV that produces the
                    # pooled calibration.
                    fold_q_hat = None
            if fold_q_hat is not None:
                logger.info(
                    f"    [fold {fold_id + 1}] fold_q_hat={fold_q_hat:.4f} "
                    f"(beta=1.0 always, for comparability), "
                    f"fold_beta={fold_beta:.4f} on "
                    f"n_cal={len(fold_conformal):,}, n_train={len(train_df):,}, "
                    f"val {val_dates[0]}..{val_dates[-1]}"
                )
            fold_metrics[-1]["fold_q_hat"] = fold_q_hat
            fold_metrics[-1]["fold_beta"] = fold_beta

        if not fold_metrics:
            raise RuntimeError(
                f"CV evaluated zero folds for {horizon}d horizon — "
                f"every validation window had <50 rows or all quantile predictions "
                f"failed. Need at least one usable fold. Training data has "
                f"{len(sorted_dates)} distinct dates and {len(tdf)} rows."
            )

        return oof_records, fold_metrics, pt_records, pt_records_clf

    @staticmethod
    def _cv_diagnostic_classifier_enabled() -> bool:
        """Whether CV fits the per-fold directional classifier.

        Default OFF (2026-08-09). It feeds no served artifact -- only fold_p50
        reaches oof_records -- and costs 932s, 52% of a classifier-on retrain
        (872s off vs 1804s on), which put a local retrain over the project's own
        30-minute run cap. The replacement diagnostics are mean_rank_ic and the
        PT verdict, both computed from fold_p50 and unaffected.

        Set CV_DIAGNOSTIC_CLASSIFIER=1 to restore mean_classifier_acc_ge1.
        """
        return os.environ.get("CV_DIAGNOSTIC_CLASSIFIER", "0") != "0"

    LAMBDARANK_BINS = 8

    @staticmethod
    def _lambdarank_enabled() -> bool:
        """Whether CV also fits a within-date `lambdarank` ranker per fold.

        Default OFF. Pure diagnostic (C2): the ranker's scores are read through
        `_within_date_rank_ic_detail` on the SAME folds as the q50, so its rank
        IC is directly comparable, and no served artifact consumes it. The read
        that decides it is the TIED cohort, where `p[d]/S[d] == 1` — a pooled
        rank-IC win is contaminated by the anchor-deviation factor a ranker will
        happily order on, which is what made `xs_rank` a serving mirage. Set
        LAMBDARANK=1. See
        `docs/superpowers/specs/2026-08-13-lambdarank-diagnostic-design.md`.
        """
        return os.environ.get("LAMBDARANK") == "1"

    @staticmethod
    def _lambdarank_labels(dates, y, k):
        """Graded relevance + LightGBM `group` vector for a ranking Dataset.

        `dates` must arrive sorted so equal dates are contiguous (the trainer
        sorts). Relevance is the WITHIN-date rank of the forward return bucketed
        into `k` graded levels — a global bucketing would re-import the market
        factor as relevance. `group` is the per-date row count, summing to
        `len(y)`. A single-row or all-equal date has nothing to rank and
        collapses to relevance 0, contributing no pairwise signal.
        """
        from scipy.stats import rankdata
        darr = np.asarray(dates)
        yv = np.asarray(y, dtype=float)
        rel = np.zeros(len(yv), dtype=np.int32)
        group = []
        start = 0
        n_all = len(darr)
        for i in range(1, n_all + 1):
            if i == n_all or darr[i] != darr[start]:
                n = i - start
                group.append(n)
                gy = yv[start:i]
                if n > 1 and np.ptp(gy) > 0:
                    r = rankdata(gy, method="average")  # 1..n, ties averaged
                    rel[start:i] = np.clip(
                        np.floor((r - 1) / n * k), 0, k - 1).astype(np.int32)
                start = i
        return rel, group

    def _lambdarank_fold_scores(self, train_df, val_df, horizon,
                                per_quantile_params):
        """Per-fold within-date ranker scores on `val_df`, LAMBDARANK diagnostic.

        Trains one `lambdarank` booster on the fold's train rows — query group =
        the date's cross-section, relevance = within-date return bucket — and
        returns its raw scores on `val_df` in `val_df` row order, so they align
        with the same `served` / `tied_served` masks the q50 rank IC uses. The
        ranker does NOT take N1's offset: its output is an ordinal score, not a
        return, and the naive baseline it is read against is unchanged. HP mirror
        the q50's tuned tree params, held fixed across arms for comparability; a
        positive result must be re-sized with FORCE_HP_SEARCH=1.
        """
        k = self.LAMBDARANK_BINS
        tcol = f"target_return_{horizon}d"
        ts = train_df.sort_values("date", kind="stable")
        feat = ts[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        med = feat.median()
        X_tr = feat.fillna(med)
        rel, group = self._lambdarank_labels(
            ts["date"].to_numpy(), ts[tcol].to_numpy(), k)
        q50 = per_quantile_params.get(0.5, {})
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "label_gain": list(range(k)),
            # Cover the whole cross-section, not the default top-30 NDCG — rank
            # IC scores the full ordering.
            "lambdarank_truncation_level": max(group) if group else k,
            "boosting_type": self.BOOSTING_TYPE,
            "max_bin": self.MAX_BIN,
            "num_leaves": q50.get("num_leaves", 31),
            "learning_rate": q50.get("learning_rate", 0.03),
            "max_depth": q50.get("max_depth", 5),
            "min_data_in_leaf": q50.get("min_data_in_leaf", 15),
            "lambda_l1": q50.get("lambda_l1", 0.5),
            "lambda_l2": q50.get("lambda_l2", 0.5),
            "feature_fraction": q50.get("feature_fraction", 0.7),
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": 42,
        }
        ds = lgb.Dataset(
            X_tr, label=rel, group=group,
            params={"max_bin": self.MAX_BIN, "feature_pre_filter": False})
        model = lgb.train(
            params, ds,
            num_boost_round=self._boost_rounds(horizon, cv=True),
            callbacks=[lgb.log_evaluation(0)])
        X_val = val_df[self.feature_cols].replace(
            [np.inf, -np.inf], np.nan).fillna(med)
        return model.predict(X_val)

    @staticmethod
    def _direction_records(pred_returns, actual_returns, dates) -> list:
        """Rows in the shape `backtest/directional_test.py` expects.

        Built from the quantile-median sign, not the classifier, so the PT test
        and the constant-call baseline stay available when the diagnostic
        classifier is skipped. `forecast_date` is a string because
        `pesaran_timmermann` clusters on it and sorts it.
        """
        tol = DIRECTION_FLAT_TOLERANCE_PCT
        p = np.asarray(pred_returns, dtype=float)
        a = np.asarray(actual_returns, dtype=float)
        d = pd.to_datetime(pd.Series(dates).to_numpy()).strftime("%Y-%m-%d")
        pdir = np.where(p > tol, "up", np.where(p < -tol, "down", "flat"))
        adir = np.where(a > tol, "up", np.where(a < -tol, "down", "flat"))
        return [
            {"predicted_direction": str(pd_), "actual_direction": str(ad),
             "direction_correct": bool(pd_ == ad), "forecast_date": str(fd)}
            for pd_, ad, fd in zip(pdir, adir, d)
        ]

    @staticmethod
    def _direction_records_from_classes(pred_cls, actual_cls, dates) -> list:
        """The same record shape as `_direction_records`, but for the SERVED
        classifier's predictions rather than the quantile-median sign.

        `_direction_classes` buckets 0=down, 1=flat, 2=up, and it is what both
        the classifier is trained against and `actual_cls` is built from. The
        actual directions therefore agree row-for-row with `_direction_records`,
        which is what makes `constant_call_accuracy` and `realised_down_rate` --
        properties of the outcomes alone -- shared between the two signals
        rather than needing to be recomputed per signal.

        Only available when CV_DIAGNOSTIC_CLASSIFIER=1. There is no fallback on
        purpose: a PT verdict that silently described a different signal
        depending on an environment variable is the failure this exists to fix.
        """
        names = ("down", "flat", "up")
        p = np.asarray(pred_cls, dtype=int)
        a = np.asarray(actual_cls, dtype=int)
        d = pd.to_datetime(pd.Series(dates).to_numpy()).strftime("%Y-%m-%d")
        return [
            {"predicted_direction": names[int(pc)],
             "actual_direction": names[int(ac)],
             "direction_correct": bool(pc == ac),
             "forecast_date": str(fd)}
            for pc, ac, fd in zip(p, a, d)
        ]

    @staticmethod
    def _summarise_rank_ic(fold_metrics: list) -> dict:
        """The rank IC block of `cv_results`: pooled, tied, and both bars.

        Two pairs, and the second is the one an arm is decided on.
        `mean_rank_ic` / `mean_naive_rank_ic` pool the whole served
        cross-section, so both legs are measured against a label whose
        denominator is the raw anchor quote the features are also built from.
        The `_tied` pair restricts to rows where that quote equals its own local
        median, which is the only cohort where neither the raw basis nor the
        smoothed one carries `p[d]/S[d]`.

        The pooled keys are NOT redefined. They are the series every historical
        `meta.json` holds and the trust warning reads, and silently changing
        what they mean would make the trend a comparison of two quantities.

        `tied_dates` rides along because the `min_rows` bar bites harder on a
        subset: a horizon whose tied number rests on two dates is not a
        measurement, and nothing else in the payload would say so.
        """
        def _mean(key):
            vals = [m[key] for m in fold_metrics if m.get(key) is not None]
            # 4 dp: a rank IC lives in [-1, 1] and the differences that matter
            # here are third-decimal. 2 dp rounds 0.1199 to 0.12 and makes the
            # naive comparison unreadable.
            return round(float(np.mean(vals)), 4) if vals else None

        def _edge(model, naive):
            return (None if (model is None or naive is None)
                    else round(model - naive, 4))

        mean_rank_ic = _mean("rank_ic")
        mean_naive = _mean("naive_rank_ic")
        mean_tied = _mean("rank_ic_tied")
        mean_naive_tied = _mean("naive_rank_ic_tied")
        # C2 lambdarank arm (None throughout when LAMBDARANK is off). The verdict
        # pair is the tied cohort: it must beat the naive baseline AND the q50's
        # own ordering, both where `p[d]/S[d] == 1`.
        mean_lr = _mean("lr_rank_ic")
        mean_lr_tied = _mean("lr_rank_ic_tied")
        return {
            "mean_rank_ic": mean_rank_ic,
            "mean_naive_rank_ic": mean_naive,
            "rank_ic_edge_vs_naive": _edge(mean_rank_ic, mean_naive),
            "mean_rank_ic_tied": mean_tied,
            "mean_naive_rank_ic_tied": mean_naive_tied,
            "rank_ic_edge_vs_naive_tied": _edge(mean_tied, mean_naive_tied),
            "mean_lr_rank_ic": mean_lr,
            "mean_lr_rank_ic_tied": mean_lr_tied,
            "lr_rank_ic_edge_vs_naive_tied": _edge(mean_lr_tied, mean_naive_tied),
            "lr_rank_ic_edge_vs_q50_tied": _edge(mean_lr_tied, mean_tied),
            "tied_rows": sum(int(m.get("n_tied") or 0) for m in fold_metrics),
            "tied_dates": sum(int(m.get("rank_ic_tied_dates") or 0)
                              for m in fold_metrics),
            "rank_ic_dates": sum(int(m.get("rank_ic_dates") or 0)
                                 for m in fold_metrics),
        }

    @classmethod
    def _within_date_rank_ic(cls, pred, actual, dates, mask=None,
                             min_rows: int = 20) -> Optional[float]:
        """Mean within-date Spearman correlation of `pred` against `actual`.

        The cross-sectional metric: it measures whether the model orders items
        correctly on a given day, which is what the product serves, and it is
        immune to the realised direction mix that dominates DA. Dates with
        fewer than `min_rows` served rows, or with no variation in either leg,
        contribute nothing rather than a degenerate 0.
        """
        return cls._within_date_rank_ic_detail(
            pred, actual, dates, mask, min_rows)[0]

    @staticmethod
    def _within_date_rank_ic_detail(pred, actual, dates, mask=None,
                                    min_rows: int = 20
                                    ) -> "tuple[Optional[float], int]":
        """`_within_date_rank_ic`, and the number of dates it actually read.

        The count is not decoration. The same `min_rows` bar applied to a
        SUBSET of the cross-section drops dates the pooled figure keeps -- the
        tied cohort is roughly a third of the panel, and one 2026-08-11 anchor
        had 26 tied items of 669 -- so the two columns can silently describe
        different calendars while looking like a paired comparison.
        """
        p = np.asarray(pred, dtype=float)
        a = np.asarray(actual, dtype=float)
        d = pd.to_datetime(pd.Series(dates).to_numpy())
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            p, a, d = p[m], a[m], d[m]
        if len(p) < min_rows:
            return None, 0
        frame = pd.DataFrame({"d": d, "p": p, "a": a})
        ics = []
        for _, g in frame.groupby("d"):
            if len(g) < min_rows:
                continue
            if g["p"].nunique() < 2 or g["a"].nunique() < 2:
                continue
            ic = spearmanr(g["p"], g["a"]).statistic
            if np.isfinite(ic):
                ics.append(float(ic))
        if not ics:
            return None, 0
        return round(float(np.mean(ics)), 4), len(ics)

    def _calibrate_confidence(self, horizon, records_df):
        """Calibrate confidence thresholds from pre-built calibration records.

        Binary confidence (high/low only):
        - Finds a range_pct threshold where high-confidence predictions achieve
          >= target_accuracy (default 80%) while maximizing coverage.
        - A change_pct floor prevents marking near-zero-move predictions as
          "high" confidence (they're correct but uninformative).

        `records_df` must carry a numeric `range_pct`, which means
        `_calibrate_conformal` has to have run first: the width is a function of
        q_hat. There used to be a second entry point that built its own records
        from the p10/p90 spread, which put `high_range` on a different scale
        than the conformal width `_compute_confidence` compares against. Both
        callers now share one scale.
        """
        target_accuracy = 0.80
        min_coverage_pct = 0.05

        df = records_df

        # Find the widest range_pct threshold where accuracy >= target.
        # Wider threshold = more items get "high" confidence → better coverage.
        # This is the opposite of the old approach (which maximized accuracy).
        thresholds = sorted(df["range_pct"].quantile([i / 20 for i in range(1, 20)]).unique())

        best_high_threshold = 0.15
        best_high_acc = 0.0
        best_high_coverage = 0

        for t in thresholds:
            subset = df[df["range_pct"] < t]
            if len(subset) >= max(50, len(df) * min_coverage_pct):
                acc = subset["hit"].mean()
                coverage = len(subset)
                # Pick the threshold with the most coverage that meets the target
                if acc >= target_accuracy and coverage > best_high_coverage:
                    best_high_threshold = t
                    best_high_acc = acc
                    best_high_coverage = coverage

        # If no threshold meets target_accuracy, fall back to the highest accuracy
        # that still covers at least 5% of items.
        if best_high_coverage == 0:
            for t in thresholds:
                subset = df[df["range_pct"] < t]
                if len(subset) >= max(50, len(df) * min_coverage_pct):
                    acc = subset["hit"].mean()
                    coverage = len(subset)
                    if coverage > best_high_coverage or (coverage == best_high_coverage and acc > best_high_acc):
                        best_high_threshold = t
                        best_high_acc = acc
                        best_high_coverage = coverage

        # Find change_pct threshold that filters near-zero-move predictions
        # (which are trivially correct but uninformative).
        best_change_threshold = 0.0
        best_change_coverage = best_high_coverage
        if best_high_coverage > 0:
            high_set = df[df["range_pct"] < best_high_threshold]
            if len(high_set) > 0:
                change_thresholds = sorted(high_set["change_pct"].quantile(
                    [i / 10 for i in range(1, 10)]).unique())
                for ct in change_thresholds:
                    subset = high_set[high_set["change_pct"] > ct]
                    if len(subset) >= max(20, len(high_set) * 0.3):
                        acc = subset["hit"].mean()
                        coverage = len(subset)
                        # Accept the change_pct threshold if accuracy stays >= target
                        if acc >= target_accuracy and coverage >= best_high_coverage * 0.5:
                            best_change_threshold = ct
                            best_change_coverage = coverage

        self.confidence_thresholds[horizon] = {
            "high_range": best_high_threshold,
            "high_change": best_change_threshold,
            "high_accuracy": round(best_high_acc * 100, 1),
        }

        th = self.confidence_thresholds[horizon]
        logger.info(
            f"  Calibrated (binary): high_range={th['high_range']:.3f} "
            f"(acc={th['high_accuracy']:.1f}%, "
            f"coverage={best_high_coverage / len(df) * 100:.1f}%)"
        )

    def _compute_confidence(self, mid: float, low: float, high: float, current: float,
                             horizon: int = 7) -> str:
        """Binary confidence: high (tight interval, non-trivial move) or low."""
        if mid == 0 or current == 0:
            return "low"
        range_pct = (high - low) / mid
        change_pct = abs(mid - current) / current

        # Use per-horizon calibrated thresholds if available, fall back to sensible defaults
        h_thresholds = self.confidence_thresholds.get(horizon, {})
        high_range = h_thresholds.get("high_range", 0.15)
        high_change = h_thresholds.get("high_change", 0.0)

        if range_pct < high_range and change_pct > high_change:
            return "high"
        return "low"

    # ------------------------------------------------------------------
    # Concept drift monitoring
    # ------------------------------------------------------------------

    def check_concept_drift(self, horizon: int = 7, sliding_window: int = 7,
                             threshold: Optional[float] = None) -> Optional[Dict]:
        """Check if recent prediction accuracy has dropped below threshold.

        Queries the last `sliding_window` days of forecast backtest results and
        compares directional accuracy against the threshold, defaulting to
        DRIFT_DA_THRESHOLD. Logs an alert to the accuracy_alerts table if drift
        is detected.

        Rows that do not report `date_coverage_sufficient` are **ignored**, not
        averaged: a cohort spanning 1-2 forecast dates describes those dates'
        market direction rather than the model. Returns None when fewer than
        three rows survive that filter, meaning "we cannot tell".

        This is an alerting signal only. It must not gate a retrain — see
        docs/superpowers/specs/2026-08-04-remove-accidental-retrain-work-design.md
        """
        threshold = self.DRIFT_DA_THRESHOLD if threshold is None else threshold
        from database import PredictionAccuracy, AccuracyAlert
        from sqlalchemy import desc

        cutoff = (self._now() - timedelta(days=sliding_window * 2)).strftime("%Y-%m-%d")
        records = self.db.execute(text("""
            SELECT evaluation_date, metrics
            FROM prediction_accuracy
            WHERE prediction_type = 'forecast'
              AND horizon_days = :horizon
              AND evaluation_date >= :cutoff
            ORDER BY evaluation_date DESC
            LIMIT :limit
        """), {"horizon": horizon, "cutoff": cutoff, "limit": sliding_window}).fetchall()

        if not records:
            return None

        accuracies = []
        uncovered = 0
        for r in records:
            m = r.metrics if isinstance(r.metrics, dict) else json.loads(r.metrics)
            if "directional_accuracy" not in m:
                continue
            # Fail closed. Rows written before scoring.py began reporting
            # coverage carry no date attribution, and rows spanning 1-2 dates
            # describe those dates rather than the model. Either way this is
            # "we cannot tell", not "the model is fine".
            if not m.get("date_coverage_sufficient", False):
                uncovered += 1
                continue
            accuracies.append(m["directional_accuracy"])

        if uncovered:
            logger.info(
                f"  Drift check ({horizon}d): ignored {uncovered} accuracy row(s) "
                f"lacking {MIN_FORECAST_DATES}-date coverage"
            )

        if len(accuracies) < 3:
            return None

        recent_avg = sum(accuracies) / len(accuracies)
        logger.info(f"  Drift check ({horizon}d, {len(accuracies)} windows): "
                     f"avg_acc={recent_avg:.1f}%, threshold={threshold:.1f}%")

        if recent_avg >= threshold:
            # Resolve any open alert
            open_alert = self.db.query(AccuracyAlert).filter(
                AccuracyAlert.prediction_type == "forecast",
                AccuracyAlert.horizon_days == horizon,
                AccuracyAlert.resolved_at.is_(None),
            ).first()
            if open_alert:
                open_alert.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
                self.db.commit()
                logger.info(f"  Drift resolved: accuracy back to {recent_avg:.1f}%")
            return {"drifted": False, "accuracy": recent_avg, "threshold": threshold}

        # Trigger alert
        alert = AccuracyAlert(
            prediction_type="forecast",
            horizon_days=horizon,
            sliding_window_days=sliding_window,
            current_accuracy=round(recent_avg, 2),
            threshold_accuracy=threshold,
            sample_count=len(accuracies),
            triggered_at=datetime.now(timezone.utc).replace(tzinfo=None),
            details={"window_accuracies": accuracies},
        )
        self.db.add(alert)
        self.db.commit()
        logger.warning(f"  DRIFT DETECTED ({horizon}d): accuracy={recent_avg:.1f}% "
                        f"below threshold={threshold:.1f}%")
        return {"drifted": True, "accuracy": recent_avg, "threshold": threshold}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_models(self):
        os.makedirs(self.model_dir, exist_ok=True)
        for (horizon, q), ensemble in self.models.items():
            if isinstance(ensemble, list):
                for ei, model in enumerate(ensemble):
                    path = os.path.join(self.model_dir, f"lgb_{horizon}d_q{int(q*100)}_e{ei}.txt")
                    model.save_model(path)
            else:
                path = os.path.join(self.model_dir, f"lgb_{horizon}d_q{int(q*100)}.txt")
                ensemble.save_model(path)

        # Save regime-specific models
        for (regime, horizon, q), ensemble in self.regime_models.items():
            if isinstance(ensemble, list):
                for ei, model in enumerate(ensemble):
                    path = os.path.join(
                        self.model_dir,
                        f"lgb_{horizon}d_q{int(q*100)}_{regime}_e{ei}.txt"
                    )
                    model.save_model(path)
            else:
                path = os.path.join(
                    self.model_dir,
                    f"lgb_{horizon}d_q{int(q*100)}_{regime}.txt"
                )
                ensemble.save_model(path)

        # Save learned band-scale models. One per horizon, and it travels with
        # `conformal_calibration` in meta.json as a matched pair: a q_hat
        # calibrated against a learned scale is in that scale's units, so an
        # artifact carrying one without the other serves an unrelated band.
        for horizon, booster in self.scale_models.items():
            booster.save_model(
                os.path.join(self.model_dir, f"scale_{horizon}d.txt"))
        if self.scale_models:
            logger.info(f"  Saved {len(self.scale_models)} learned scale models")
        # And remove any that this run did not produce, for the same reason the
        # regime sweep below exists: a stale scale_*.txt left on disk would be
        # loaded beside a q_hat calibrated without it.
        for horizon in self.HORIZONS:
            if horizon in self.scale_models:
                continue
            stale = os.path.join(self.model_dir, f"scale_{horizon}d.txt")
            if os.path.exists(stale):
                os.remove(stale)

        # Save directional classifiers (one 3-class model per horizon)
        for horizon, clf in self.direction_models.items():
            clf.save_model(os.path.join(self.model_dir, f"clf_{horizon}d.txt"))
        if self.direction_models:
            logger.info(f"  Saved {len(self.direction_models)} directional classifiers")

        # Remove orphaned regime-model files: any lgb_*_{regime}_*.txt on disk
        # that isn't in the current self.regime_models. Without this, a
        # regime-free (SKIP_REGIMES) run would leave stale regime artifacts
        # behind, which load_models would then drag forward indefinitely.
        import glob as _glob
        expected = set()
        for (regime, horizon, q), ensemble in self.regime_models.items():
            members = ensemble if isinstance(ensemble, list) else [ensemble]
            for ei in range(len(members)):
                expected.add(f"lgb_{horizon}d_q{int(q*100)}_{regime}_e{ei}.txt")
            expected.add(f"lgb_{horizon}d_q{int(q*100)}_{regime}.txt")
        removed = 0
        for regime in self.REGIMES:
            for path in _glob.glob(os.path.join(self.model_dir, f"lgb_*_{regime}_*.txt")) \
                    + _glob.glob(os.path.join(self.model_dir, f"lgb_*_{regime}.txt")):
                if os.path.basename(path) not in expected:
                    os.remove(path)
                    removed += 1
        if removed:
            logger.info(f"  Removed {removed} orphaned regime model files")

        # Save feature columns, calibration thresholds, and imputation medians
        thresholds_serial = {}
        for horizon, th in self.confidence_thresholds.items():
            thresholds_serial[str(horizon)] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in th.items()
            }
        medians = self.feature_medians.to_dict() if not self.feature_medians.empty else {}
        medians_serial = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                          for k, v in medians.items()}
        # Compute feature importance for each horizon (average over quantiles/ensembles)
        feature_importance = {}
        for horizon in self.HORIZONS:
            all_importances = {}
            for q in self.QUANTILES:
                key = (horizon, q)
                if key not in self.models:
                    continue
                ensemble = self.models[key]
                if isinstance(ensemble, list):
                    for m in ensemble:
                        fi = self._get_feature_importance(m)
                        for _, row in fi.iterrows():
                            all_importances.setdefault(row["feature"], []).append(float(row["importance"]))
                else:
                    fi = self._get_feature_importance(ensemble)
                    for _, row in fi.iterrows():
                        all_importances.setdefault(row["feature"], []).append(float(row["importance"]))
            if all_importances:
                avg = {f: sum(v) / len(v) for f, v in all_importances.items()}
                sorted_fi = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:20]
                feature_importance[str(horizon)] = [{"feature": f, "importance": round(v, 4)} for f, v in sorted_fi]

        # Serialize CV results (convert int keys to str for JSON)
        cv_serial = {}
        for h_str, cvdata in self.cv_results.items():
            cv_serial[str(h_str)] = cvdata

        # Serialize tuned per-quantile hyperparameters so subsequent retrains
        # can reuse them and skip the Optuna search (Tier-1 speedup).
        tuned_serial = {}
        for h, qd in self.tuned_params.items():
            tuned_serial[str(h)] = {str(q): dict(params) for q, params in qd.items()}

        # Serialize regime feature cols: {"horizon_regime": [cols]}
        regime_cols_serial = {}
        for (h, regime), cols in self.regime_feature_cols.items():
            regime_cols_serial[f"{h}_{regime}"] = cols

        # Track which regimes actually have trained models
        trained_regimes = list(set(reg for (reg, h, q) in self.regime_models.keys()))

        meta = {
            "model_artifact_version": self.MODEL_ARTIFACT_VERSION,
            # Not folded into the version bump: with the flag off the feature set
            # is byte-identical to v5, so bumping would force every checkout into
            # a needless retrain for a change none of them enabled. Checked
            # separately instead — see _check_artifact_version.
            "bymykel_metadata": self.bymykel_metadata_enabled(),
            # Recorded so the predict path can follow the artifact instead of the
            # environment (see _tier_lead_served). Not part of
            # MODEL_ARTIFACT_VERSION for the same reason bymykel_metadata is not:
            # with both flags off the artifact is byte-identical to one written
            # before they existed, so bumping would force every checkout into a
            # needless retrain for a change none of them enabled.
            "tier_lead": self.tier_lead_enabled(),
            "cross_sectional_rank": self.cross_sectional_rank_enabled(),
            # The cohort the transform above was computed over. predict's frame
            # holds every backfilled item, so without these it cannot rebuild
            # the population the booster's percentiles are relative to.
            "train_min_median_price": self._train_min_median_price,
            "train_cohort_items": self._train_cohort_items,
            "xs_rank_skipped_cols": self._train_xs_rank_skipped,
            # Load-bearing for the served LEVEL, not just for the feature set: a
            # booster fitted with the offset emits a residual, so predict has to
            # know to add `-return_1d` back. See _naive_init_score_served.
            "naive_init_score": self.naive_init_score_enabled(),
            # Provenance, not a serving switch. `predict` already converts a
            # return to dollars against the smoothed anchor, so an artifact
            # trained under this flag is the COHERENT pairing and needs nothing
            # added back. What the field is for is reading a stored metric: a
            # rank IC from a run with this on is measured against a different
            # target than one with it off, and the two must never be differenced.
            "label_smoothed_anchor": self.label_smoothed_anchor_enabled(),
            "sigma_clip": dict(self.sigma_clip),
            "feature_cols": self.feature_cols,
            "horizon_feature_cols": {
                str(h): cols for h, cols in self.horizon_feature_cols.items()
            },
            "regime_feature_cols": regime_cols_serial,
            "trained_regimes": trained_regimes,
            "regime_threshold_bear": self.REGIME_RETURN_THRESHOLD_BEAR,
            "regime_threshold_bull": self.REGIME_RETURN_THRESHOLD_BULL,
            "trained_at": str(self._now()),
            "confidence_thresholds": thresholds_serial,
            "conformal_calibration": {str(h): v for h, v in self.conformal_calibration.items()},
            # The exponent each q_hat above is dimensionally tied to. Written on
            # the adjacent line, deliberately: these two are a matched pair and a
            # q_hat served at the wrong exponent is wrong by ~5x, so the writer
            # must not be able to emit one without the other. Always populated —
            # an explicit 1.0 is auditable where a missing key is ambiguous about
            # whether the run had the flag off or predates the field.
            "conformal_beta": {
                str(h): self.band_beta(h) for h in self.conformal_calibration
            },
            # The learned scale, and the same matched-pair argument one step
            # further: `conformal_beta` decides how hard to damp `sigma`, this
            # decides whether `sigma` is the variable at all. A q_hat calibrated
            # against a learned scale and served against `sigma` is not a
            # degraded band, it is an unrelated one — so the norm, the clip and
            # the column list are all written here beside the booster on disk,
            # and a horizon missing from this dict falls back to `sigma ** beta`.
            "learned_scale": {
                str(h): {
                    "norm": float(self.scale_norm.get(h, 1.0)),
                    "clip": [float(self.scale_clip[h][0]),
                             float(self.scale_clip[h][1])],
                    # The exact column order the booster was fitted on.
                    # `band_scale` refuses to score a frame that does not match
                    # it: the wrong columns give a plausible number from the
                    # wrong features and a band that looks entirely normal.
                    "features": list(self.scale_features.get(h, [])),
                }
                for h in sorted(self.scale_models)
            },
            # Which mid each q_hat covers — "served" (the classifier-recentred
            # mid predict() publishes) or "q50" (a mid served only when no
            # classifier exists). A coverage figure read without this is
            # uninterpretable, and a predict-only run has no other route to it.
            "conformal_centre": {str(h): c for h, c in self.conformal_centre.items()},
            # Which DENOMINATOR each q_hat was fitted on — "served" (the
            # smoothed anchor predict() quotes from, which is also what the
            # backtest scores in) or "raw_anchor" (the training label's, which
            # over-covers). Same argument as conformal_centre: two artifacts
            # with different values here carry bands of different widths for
            # the same model, and nothing else in the file says so.
            "conformal_basis": {str(h): b for h, b in self.conformal_basis.items()},
            "feature_medians": medians_serial,
            "n_ensembles": self.N_ENSEMBLES,
            "ensemble_seeds": self.ENSEMBLE_SEEDS,
            "ensemble_feature_fractions": self.ENSEMBLE_FEATURE_FRACTIONS,
            "training_window_days": 1460,
            "feature_importance": feature_importance,
            "cv_results": cv_serial,
            "tuned_params": tuned_serial,
            "label_voiding": self.label_voiding,
        }
        def _json_default(o):
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(os.path.join(self.model_dir, "meta.json"), "w") as f:
            json.dump(meta, f, default=_json_default)

        # Save per-tier bias corrections
        self._save_bias_corrections()

        logger.info(f"Models saved to {self.model_dir}")

    def load_models(self):
        meta_path = os.path.join(self.model_dir, "meta.json")
        if not os.path.exists(meta_path):
            logger.warning(f"No saved models found in {self.model_dir}")
            return False

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Corrupt meta.json ({e}); ignoring saved models and retraining.")
            return False

        # Must run before any other field is read: an artifact written by an
        # incompatible code version can hold fields of the same name and type
        # but a different MEANING (see IncompatibleModelArtifact), so reading
        # anything else first risks acting on it before the check fires.
        self._check_artifact_version(meta)

        self.feature_cols = meta["feature_cols"]
        self.feature_medians = pd.Series(meta.get("feature_medians", {}), dtype=np.float64)

        # Absent in artifacts written before these flags existed, which is why
        # the default is None rather than False: None means "this artifact does
        # not say", and _tier_lead_served then falls back to the environment.
        # Reading a stored False as None would be wrong in the other direction --
        # it would let a stray env var change what an existing model is served.
        self._artifact_tier_lead = meta.get("tier_lead")
        self._artifact_xs_rank = meta.get("cross_sectional_rank")
        self._artifact_min_median_price = meta.get("train_min_median_price")
        self._artifact_cohort_items = meta.get("train_cohort_items")
        self._artifact_xs_rank_skipped = meta.get("xs_rank_skipped_cols")
        self._artifact_naive_init = meta.get("naive_init_score")

        # Restore cached tuned hyperparameters (skips Optuna on retrain when present).
        self.tuned_params = {}
        raw_tp = meta.get("tuned_params", {})
        for h_str, qd in raw_tp.items():
            try:
                h = int(h_str)
            except (ValueError, TypeError):
                continue
            self.tuned_params[h] = {}
            for q_str, params in qd.items():
                try:
                    q = float(q_str)
                except (ValueError, TypeError):
                    continue
                self.tuned_params[h][q] = params

        # Load per-horizon confidence thresholds (backward compat: treat flat dict as 7d)
        raw_thresholds = meta.get("confidence_thresholds", {})
        self.confidence_thresholds = {}
        if raw_thresholds:
            first_key = next(iter(raw_thresholds))
            if isinstance(first_key, str) and first_key.lstrip("-").isdigit():
                # New nested format: {"7": {"high_range": ..., ...}, "30": {...}}
                for h_str, th in raw_thresholds.items():
                    self.confidence_thresholds[int(h_str)] = th
            else:
                # Legacy flat format: {"high_range": ..., ...} — assign to all horizons
                for h in self.HORIZONS:
                    self.confidence_thresholds[h] = dict(raw_thresholds)
        # Strict: both fields are load-bearing for the band, and
        # _check_artifact_version has already confirmed this artifact is the
        # minimal-model scheme, so a missing key here means the artifact is
        # corrupt, not that an older field name should be defaulted around.
        self.conformal_calibration = {
            int(h): float(q) for h, q in meta["conformal_calibration"].items()
        }
        self.sigma_clip = {k: float(v) for k, v in meta["sigma_clip"].items()}
        # The ONE non-provenance field that is loaded with `.get`, and the reason
        # is the opposite of laxness: every artifact written before 2026-08-12
        # lacks the key, and 1.0 is precisely the exponent those q_hats were
        # calibrated at, so defaulting reproduces their band exactly. A NaN or a
        # missing horizon resolves through `band_beta`, never into a half-width.
        self.conformal_beta = {
            int(h): float(b) for h, b in meta.get("conformal_beta", {}).items()
        }
        # The learned scale, restored as a unit. `.get` for the same reason as
        # `conformal_beta`: absent on every artifact before 2026-08-12, and
        # absence means "this q_hat was calibrated against sigma", which
        # `band_scale` reproduces by returning None. A horizon whose booster is
        # missing from disk is dropped from ALL FOUR dicts rather than kept with
        # a default, so the pair can never come apart -- an entry here without
        # its booster would serve `sigma` under a q_hat that is not in sigma's
        # units, which is the one failure this whole structure exists to stop.
        self.scale_models, self.scale_norm = {}, {}
        self.scale_clip, self.scale_features = {}, {}
        for h, cfg in meta.get("learned_scale", {}).items():
            path = os.path.join(self.model_dir, f"scale_{int(h)}d.txt")
            if not os.path.exists(path):
                logger.warning(
                    f"  meta.json claims a learned scale for {h}d but "
                    f"{os.path.basename(path)} is missing. Dropping it — the "
                    f"band for this horizon falls back to sigma, and its q_hat "
                    f"is NOT in sigma's units, so treat its width as suspect."
                )
                continue
            self.scale_models[int(h)] = lgb.Booster(model_file=path)
            self.scale_norm[int(h)] = float(cfg.get("norm", 1.0))
            clip = cfg.get("clip")
            self.scale_clip[int(h)] = (None if not clip
                                       else (float(clip[0]), float(clip[1])))
            self.scale_features[int(h)] = list(cfg.get("features", []))
        if self.scale_models:
            logger.info(
                f"  Loaded {len(self.scale_models)} learned scale models "
                f"({sorted(self.scale_models)}d) — these q_hats are in the "
                f"learned scale's units, not sigma's."
            )
        # Provenance, so `.get` and not strict: predict() reads the band from
        # conformal_calibration alone, and every artifact written before
        # 2026-08-11 lacks this key. An empty dict means "this artifact does not
        # say which mid its q_hat covers" — which is itself the honest answer.
        self.conformal_centre = {
            int(h): str(c) for h, c in meta.get("conformal_centre", {}).items()
        }
        # Same contract, same reason: absent on every artifact written before
        # 2026-08-12, and an artifact that does not say is not the same as one
        # that says "raw_anchor".
        self.conformal_basis = {
            int(h): str(b) for h, b in meta.get("conformal_basis", {}).items()
        }

        n_ensembles = meta.get("n_ensembles", 1)

        for horizon in self.HORIZONS:
            for q in self.QUANTILES:
                # Load LGB models. train() always stores a LIST, so save_models
                # writes `_e{ei}`-suffixed members even for a single member —
                # this must mirror that regardless of n_ensembles. Branching on
                # `n_ensembles > 1` here and reading an unsuffixed filename in
                # the single-member case left self.models empty once
                # N_ENSEMBLES became 1, silently serving no forecasts.
                # `range(n_ensembles)` (not a glob) is what keeps stale e1/e2
                # files from the 40-model grid out of the ensemble.
                ensemble = []
                for ei in range(max(1, n_ensembles)):
                    path = os.path.join(self.model_dir, f"lgb_{horizon}d_q{int(q*100)}_e{ei}.txt")
                    if os.path.exists(path):
                        try:
                            ensemble.append(lgb.Booster(model_file=path))
                        except (lgb.basic.LightGBMError, Exception) as e:
                            logger.warning(f"  Corrupt model {path}, skipping: {e}")
                if ensemble:
                    self.models[(horizon, q)] = ensemble
                    continue
                # Unsuffixed fallback: save_models still emits this name if a
                # bare Booster (not a list) is ever assigned to self.models.
                path = os.path.join(self.model_dir, f"lgb_{horizon}d_q{int(q*100)}.txt")
                if os.path.exists(path):
                    try:
                        self.models[(horizon, q)] = lgb.Booster(model_file=path)
                    except (lgb.basic.LightGBMError, Exception) as e:
                        logger.warning(f"  Corrupt model {path}, skipping: {e}")

        # Load regime-specific models
        trained_regimes = meta.get("trained_regimes", [])
        raw_regime_fc = meta.get("regime_feature_cols", {})
        self.regime_feature_cols = {}
        for key_str, cols in raw_regime_fc.items():
            parts = key_str.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    h = int(parts[0])
                    regime = parts[1]
                    self.regime_feature_cols[(h, regime)] = cols
                except ValueError:
                    continue

        for regime in trained_regimes:
            for horizon in self.HORIZONS:
                for q in self.QUANTILES:
                    ensemble = []
                    for ei in range(n_ensembles):
                        path = os.path.join(
                            self.model_dir,
                            f"lgb_{horizon}d_q{int(q*100)}_{regime}_e{ei}.txt"
                        )
                        if os.path.exists(path):
                            try:
                                ensemble.append(lgb.Booster(model_file=path))
                            except (lgb.basic.LightGBMError, Exception) as e:
                                logger.warning(f"  Corrupt regime model {path}, skipping: {e}")
                    if ensemble:
                        self.regime_models[(regime, horizon, q)] = ensemble

        # Load directional classifiers (one 3-class model per horizon)
        for horizon in self.HORIZONS:
            cpath = os.path.join(self.model_dir, f"clf_{horizon}d.txt")
            if os.path.exists(cpath):
                try:
                    self.direction_models[horizon] = lgb.Booster(model_file=cpath)
                except (lgb.basic.LightGBMError, Exception) as e:
                    logger.warning(f"  Corrupt classifier {cpath}, skipping: {e}")
        if self.direction_models:
            logger.info(f"  Loaded {len(self.direction_models)} directional classifiers")

        # Build per-horizon feature sets from the loaded models.
        # Each model internally stores the feature names it was trained with.
        self.horizon_feature_cols = {}
        for (horizon, q), ensemble in self.models.items():
            if horizon in self.horizon_feature_cols:
                continue
            model = ensemble[0] if isinstance(ensemble, list) else ensemble
            self.horizon_feature_cols[horizon] = model.feature_name()

        # Load per-tier bias corrections
        self._load_bias_corrections()

        total_groups = len(self.models) + len(self.regime_models)
        if self.regime_models:
            regimes_found = set(r for (r, h, q) in self.regime_models.keys())
            logger.info(f"Loaded {len(self.models)} global + {len(self.regime_models)} regime "
                        f"model groups ({regimes_found}) from {self.model_dir}")
        else:
            logger.info(f"Loaded {total_groups} model groups from {self.model_dir}")
        return total_groups > 0

    def has_models(self) -> bool:
        return len(self.models) > 0
