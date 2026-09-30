#!/usr/bin/env python3
"""
A/B test: do the shelved volume features add real directional accuracy?

Three arms on the same walk-forward folds:
  - baseline:  drop the volume columns
  - treatment: all features (including volume)
  - placebo:   the volume columns column-shuffled (capacity-inflation guard)

WHY THE DATE CUTOFF: the archive's `volume` column is 100% positive through
2026-04 and identically 0 from 2026-07 on (2026-05/06 are the archive gap).
That is why these features are shelved -- a pure train/serve gap, not a lack
of signal (docs/changelog/2026-08-06-volume-features-shelved.md). Evaluating
past the cliff would measure a column that is 0 in every late fold and answer
nothing. The frame is therefore cut at VOLUME_LIVE_THROUGH, so this A/B
answers exactly one question: **if the volume feed were repaired, would these
features earn a place in production?** It does NOT license un-shelving them
against the current dead feed.

Usage:
    python scripts/archive/ab_test_volume_features.py [--max-items 200] [--horizon 14]

Embargo (added 2026-08-08):
    This harness had **no purge gap at all** before that date: the fold split
    was `train = every date <= window_end - 1`, so every training row within
    `horizon` days of the boundary carried a label resolved from inside the
    validation window. The train side now goes through production's own
    `ItemForecaster._purge_overlapping_train_rows`, which embargoes
    `embargo_days(horizon)` = `horizon + 13` days -- the label's resolved-anchor
    support, not its nominal date. The validation window is untouched; purging
    it would empty the 21-day window at h=30.

    **Every delta this harness printed before 2026-08-08 is un-embargoed.**
    How much that inflated them is not known here. The often-quoted
    "+12.1pp unpurged -> +6.1pp purged" at h=30 is from the external review
    (docs/research/2026-08-07-cs2-forecasting-research.md), describes an
    event-calendar arm, and has never been replicated in this repo.
"""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from backtest.paired_mde import format_paired, paired_arm_contrasts, paired_metric_difference
from backtest.walkforward_records import (
    paired_records,
    without_records,
)
from database import SessionLocal
from models.forecaster import ItemForecaster, phase_collapsed_sql_filter

# The 13 shelved volume columns. The last two were never in the 47-column
# feature_cols -- the >0.95 correlation prune dropped them in favour of their
# 30d partners -- but they re-enter once the partners are dropped, so the arms
# have to name them explicitly.
NEW_PRIMITIVES = (
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
    "volume_mean_7d",
    "volume_std_60d",
)

# Last day on which the archive carries real volume. See the docstring.
VOLUME_LIVE_THROUGH = "2026-04-30"

# Minimum median price for an item to enter the universe.
#
# Without this the universe is `ORDER BY row_count DESC` = longest history,
# which selects cheap high-turnover cases and stickers: measured 15 of 200
# items >= $1, median item price $0.059, 8.22% of rows in the served cohort.
# That cohort also breaks the target -- at $0.03 one cent is a 33% move, so
# 41% of 3d forward returns are EXACTLY zero (64% for sub-$0.10 items), the
# q50 learns to emit exactly 0.0 on 39% of rows, and `sign(0)==sign(0)`
# scores those as hits: 31% of the reported DA is free. Strip the flat rows
# and the same model sits at 51.2% against a 50.1% base rate.
#
# Production reports on the >=$1 cohort, and the flat rate there is 0.15-0.53%
# depending on horizon, so this filter fixes the metric and the cohort at once.
MIN_MEDIAN_PRICE = 1.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ab_test_volume_features")

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"

# Feed-repair arm (docs/research/2026-08-17-volume-band-quality-preregistration.md).
# With IFLOW_VOLUME=1 the `volume` column is sourced from the iflow BUFF backfill's
# count_in_24 series (staging archive) instead of the native aggregator feed, which
# is identically 0 from 2026-07 on. iflow is used as the *single* volume series over
# the whole panel — never coalesced with the native feed, which would average two
# differently-scaled series inside a 30d/60d rolling window and corrupt the treatment
# arm exactly the way the cliff-straddle does. The arm structure is untouched: only
# which series populates `volume` changes; baseline/treatment/placebo still differ
# solely by whether the volume *columns* are dropped/shuffled downstream.
IFLOW_VOLUME = os.getenv("IFLOW_VOLUME") == "1"
IFLOW_VOLUME_DIR = Path(__file__).parent.parent.parent / "buff-iflow-staging" / "price-archive"
# iflow count_in_24 spans 2022-04-18 .. 2026-05-20; restrict the eval panel to
# that era so the volume features are live across every row they are scored on.
IFLOW_ERA_FROM = "2022-04-18"
if IFLOW_VOLUME:
    # iflow count_in_24 runs through 2026-05-20; the native cliff was 2026-04-30.
    VOLUME_LIVE_THROUGH = "2026-05-20"

# Live-feed repair arm (2026-09-08 accuracy note, priority 2). With
# SKINPORT_VOLUME=1 the `volume` column is sourced from the free Skinport
# completed-sale series (`volume-YYYY-MM.parquet`, source=skinport_sales,
# collected daily since 2026-08-08) instead of the dead native aggregator
# column. Same single-series rule as IFLOW_VOLUME: never coalesced with
# another definition — Skinport's trailing windows are a different quantity
# from Steam's daily counts, and averaging them inside a 30d/60d rolling
# window corrupts the treatment arm exactly the way the cliff-straddle does.
# `sales_24h` is the closest analogue of a daily sale count. Set
# SKINPORT_ERA_FROM to the first accumulated day when running.
SKINPORT_VOLUME = os.getenv("SKINPORT_VOLUME") == "1"
SKINPORT_ERA_FROM = os.getenv("SKINPORT_ERA_FROM", "2026-08-08")

# Walk-forward fold scheme, exposed as run params for the re-fold prereg
# (docs/research/2026-08-17-volume-band-quality-refold-preregistration.md).
# Defaults reproduce the original hardcoded scheme (2/3 split, 60-day step,
# 21-day window) exactly; the re-fold uses SPLIT_FRACTION=0.50 STEP_DAYS=45 to
# clear the >=14-cluster floor without overlapping validation windows (the gap
# STEP_DAYS - VAL_WINDOW_DAYS must exceed the horizon, or forward-resolved
# labels bleed across folds).
SPLIT_FRACTION = float(os.getenv("SPLIT_FRACTION", str(2 / 3)))
STEP_DAYS = int(os.getenv("STEP_DAYS", "60"))
VAL_WINDOW_DAYS = int(os.getenv("VAL_WINDOW_DAYS", "21"))

# Route-2 offline scale probe (docs above; the served band is `q_hat * scale`,
# scale defaults to sigma=price_std_60d/price). SCALE_PROBE=1 asks the one cell
# the 2026-08-12 learned-scale refutation never tested: does *volume* as a NEW
# conditioning feature in the scale flatten conditional band miscalibration
# beyond a sigma-only learned scale, and beyond a volume-shuffled placebo? It
# uses the BASELINE arm's residuals (volume is not in the mean), so it isolates
# volume-as-band-width from volume-as-mean-predictor.
SCALE_PROBE = os.getenv("SCALE_PROBE") == "1"

# Production's item universe, spelled into every archive read this harness
# makes. Before 2026-08-08 the `ab_test_*` family globbed the Parquet privately
# and saw a universe production does not train on, so an A/B advised a model it
# had not measured. The bid sources need no clause here: `aggregator_sync` is a
# single ask feed and already excludes them. See `models/item_parser.py`.
_UNIVERSE = phase_collapsed_sql_filter()


# Correlation-prune threshold. Part of the frame fingerprint: it decides the
# cached frame's column set, so changing it must invalidate the cache.
CORR_PRUNE_THRESHOLD = 0.95

# Dataset-construction params. Mirrors production (`ItemForecaster.MAX_BIN`)
# so the A/B measures the model prod actually ships, and bins 4x coarser than
# LightGBM's default 255 — cheaper to build and to train.
DS_PARAMS = {"max_bin": 63, "feature_pre_filter": False}


def _frame_fingerprint():
    """Identify the feature-producing code, so a cached frame can't outlive it.

    The frame is only as valid as `_compute_price_features`; a cache key over
    the data alone cannot see a code change. Same failure mode the voted-frame
    cache guards with VOTED_CACHE_VERSION.

    Hashes content rather than size+mtime: git checkouts rewrite mtime without
    changing bytes, and a stale-looking-but-valid cache hard-exits every
    worker. Covers this script's own frame-shaping constants too — they are
    baked into the cached column set just as much as forecaster.py is.
    """
    src = Path(__file__).parent.parent / "models" / "forecaster.py"
    h = hashlib.sha256(src.read_bytes())
    h.update(
        repr(
            (
                NEW_PRIMITIVES,
                CORR_PRUNE_THRESHOLD,
                VOLUME_LIVE_THROUGH,
                MIN_MEDIAN_PRICE,
                _UNIVERSE,
                IFLOW_VOLUME,
                IFLOW_ERA_FROM if IFLOW_VOLUME else None,
                SPLIT_FRACTION,
                STEP_DAYS,
                VAL_WINDOW_DAYS,
            )
        ).encode()
    )
    return h.hexdigest()[:16]


def build_frame(max_items=200, cache_path=None):
    """Load prices, engineer features, correlation-prune.

    Returns (df, pruned, present_new). If `cache_path` exists, reads it instead
    of rebuilding; otherwise builds and writes it.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") != _frame_fingerprint():
                raise SystemExit(
                    f"Frame cache {cache_path} was built from different "
                    f"feature code. Rebuild it with --build-cache-only."
                )
            if meta.get("max_items") != max_items:
                raise SystemExit(
                    f"Frame cache {cache_path} was built with max_items={meta.get('max_items')}, not {max_items}."
                )
            df = pd.read_parquet(cache_path)
            logger.info(f"  Loaded cached frame {cache_path} ({len(df):,} rows)")
            return df, meta["pruned"], meta["present_new"]

    df, pruned, present_new = _build_frame_uncached(max_items)

    if cache_path is not None:
        # Write-then-rename, and the parquet lands before the meta that
        # vouches for it. Without this, N workers that all miss the cache
        # interleave writes into one path and a reader can pick up a meta
        # that points at a half-written frame.
        tmp_frame = cache_path.with_suffix(f".{os.getpid()}.tmp.parquet")
        tmp_meta = cache_path.with_suffix(f".{os.getpid()}.tmp.json")
        df.to_parquet(tmp_frame, index=False)
        tmp_meta.write_text(
            json.dumps(
                {
                    "fingerprint": _frame_fingerprint(),
                    "max_items": max_items,
                    "pruned": pruned,
                    "present_new": present_new,
                    "rows": len(df),
                }
            )
        )
        os.replace(tmp_frame, cache_path)
        os.replace(tmp_meta, cache_path.with_suffix(".meta.json"))
        logger.info(f"  Wrote frame cache {cache_path} ({len(df):,} rows)")

    return df, pruned, present_new


def _load_iflow_volume(con, slugs):
    """The iflow volume series (item_id, date, iflow_volume) for `slugs`.

    The staging files carry the two eras in separate columns
    (`count_in_24_pre_restructure` / `steam_volume_post_restructure`, split at
    the 2024-02-13 upstream restructure -- see scripts/backfill_buff_iflow.py).
    The two series are incomparable and must never be merged: this loader uses
    the post-restructure series only and REFUSES a panel containing pre-era
    rows rather than silently splicing them. An experiment spanning the
    boundary must pick an era explicitly.

    Keyed on (item_slug, day); one dump/day is written by the backfill, and the
    MAX collapse is defensive against a duplicate. Universe safety is inherited
    from the caller: the left join below keeps only rows whose item_slug already
    passed `_UNIVERSE` on the price side, so phantom/phase-collapsed keys cannot
    re-enter through the volume series.
    """
    files = sorted(str(p) for p in IFLOW_VOLUME_DIR.glob("iflow-liquidity-*.parquet"))
    if not files:
        raise RuntimeError(
            f"IFLOW_VOLUME=1 but no iflow-liquidity-*.parquet in {IFLOW_VOLUME_DIR}. "
            f"Run scripts/backfill_buff_iflow.py first."
        )
    union = " UNION ALL BY NAME ".join(
        f"SELECT item_slug, day, count_in_24_pre_restructure, steam_volume_post_restructure FROM read_parquet('{f}')"
        for f in files
    )
    placeholders = ", ".join("?" for _ in slugs)
    df = con.sql(
        f"""
        SELECT item_slug AS item_id, day AS date,
               MAX(count_in_24_pre_restructure) AS vol_pre,
               MAX(steam_volume_post_restructure) AS vol_post
        FROM ({union})
        WHERE item_slug IN ({placeholders})
        GROUP BY item_slug, day
    """,
        params=slugs,
    ).df()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    n_pre = int(df["vol_pre"].notna().sum())
    if n_pre:
        raise RuntimeError(
            f"IFLOW_VOLUME=1 but {n_pre:,} joined row(s) carry the pre-2024-02-13 "
            f"`count_in_24` series, which is incomparable with the post-restructure "
            f"`steam_volume` series (backfill era split). Restrict the panel to one "
            f"era instead of merging them."
        )
    return df.rename(columns={"vol_post": "iflow_volume"})[["item_id", "date", "iflow_volume"]]


def _load_skinport_live_volume(con, slugs: list, archive_dir=None) -> pd.DataFrame:
    """The free Skinport live sale counts (item_id, date, live_volume).

    Reads ``volume-YYYY-MM.parquet`` from the archive (source=skinport_sales,
    ``sales_24h`` trailing window). Raises when empty so a stalled accumulator
    cannot present as a measured null.
    """
    base = Path(archive_dir) if archive_dir else ARCHIVE_DIR
    files = sorted(str(p) for p in base.glob("volume-[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"))
    if not files:
        raise RuntimeError(
            "SKINPORT_VOLUME=1 but no volume-*.parquet in the archive. "
            "Run scripts/run_sales_volume.py daily until the continuity "
            "gate (scripts/check_sidecar_continuity.py) reports enough dates."
        )
    union = " UNION ALL BY NAME ".join(f"SELECT item_slug, day, sales_24h FROM read_parquet('{f}')" for f in files)
    placeholders = ", ".join("?" for _ in slugs)
    df = con.sql(
        f"""
        SELECT item_slug AS item_id, day AS date,
               MAX(sales_24h) AS live_volume
        FROM ({union})
        WHERE item_slug IN ({placeholders})
        GROUP BY item_slug, day
    """,
        params=slugs,
    ).df()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _width_robustness(base_records, treat_records, *, value_key="rel_width"):
    """Gating robustness for the width contrast (refold prereg).

    Two checks the h=14 tightening must survive: (1) leave-one-fold-out -- no
    single fold may flip the CI to include 0; (2) leave-out-2025-10 -- the effect
    must not be carried by the volume-spike regime. Width is lower-better, so the
    CI excludes 0 when `ci_upper < 0`; the LOO worst case is the drop that pushes
    `ci_upper` highest (closest to 0). Returns (worst_loo, no_oct25).
    """
    folds = sorted({r["fold_id"] for r in treat_records})
    worst = None
    for f in folds:
        b = [r for r in base_records if r["fold_id"] != f]
        t = [r for r in treat_records if r["fold_id"] != f]
        d = paired_metric_difference(b, t, value_key=value_key, scale=100.0)
        if d.get("ci_upper") is None:
            continue
        if worst is None or d["ci_upper"] > worst["ci_upper"]:
            worst = {"dropped_fold": f, **d}

    def _not_oct25(r):
        ts = pd.Timestamp(r["forecast_date"])
        return not (ts.year == 2025 and ts.month == 10)

    b_no = [r for r in base_records if _not_oct25(r)]
    t_no = [r for r in treat_records if _not_oct25(r)]
    n_oct = len(treat_records) - len(t_no)
    no_oct = paired_metric_difference(b_no, t_no, value_key=value_key, scale=100.0)
    no_oct = {"n_oct25_rows_dropped": n_oct, **no_oct}
    return worst, no_oct


def _stratified_coverage(resid, denom, strat, n_strata=10, alpha=0.20):
    """Level-matched conditional coverage error across strata of `strat`.

    Mirrors `conformal.coverage_by_sigma_stratum` but DECOUPLES the stratifier
    from the denominator, so the same rows can be scored under `sigma` vs a
    learned scale while stratifying on a fixed axis. The threshold is the
    empirical 1-alpha quantile of `|resid|/denom`, so MARGINAL coverage is
    1-alpha by construction and any spread across strata is conditional
    miscalibration -- a uniformly narrower band cannot fake a win here (the
    exact trap `conformal.py:255-260` documents). Returns mean |coverage -
    (1-alpha)| in pp.
    """
    r = np.abs(np.asarray(resid, dtype=float))
    d = np.asarray(denom, dtype=float)
    st = np.asarray(strat, dtype=float)
    ok = np.isfinite(r) & np.isfinite(d) & (d > 0) & np.isfinite(st)
    r, d, st = r[ok], d[ok], st[ok]
    if r.size < n_strata:
        return float("nan")
    score = r / d
    thr = float(np.quantile(score, 1.0 - alpha))
    covered = score <= thr
    edges = np.quantile(st, np.linspace(0.0, 1.0, n_strata + 1)[1:-1])
    idx = np.searchsorted(edges, st, side="right")
    per = np.array([covered[idx == k].mean() if np.any(idx == k) else np.nan for k in range(n_strata)])
    return float(np.nanmean(np.abs(per - (1.0 - alpha)))) * 100.0


def _run_scale_probe(records, present_new):
    """Does volume in the learned scale flatten conditional coverage?

    Four denominators, all level-matched to 80% marginal by `_stratified_coverage`:
    `sigma` (production), a sigma-only learned scale (the fair "scale but no new
    info" control), a sigma+volume scale, and a sigma+shuffled-volume placebo.
    The volume scale must beat BOTH the sigma-only scale and the placebo on the
    volume-stratified error for volume to be real band information.
    """
    from models import scale_model

    rec = pd.DataFrame(records)
    resid = rec["residual_pct"].to_numpy(dtype=float)
    sigma = rec["sigma"].to_numpy(dtype=float)
    folds = rec["fold"].to_numpy()
    volcols = [c for c in present_new if c in rec.columns]
    if not volcols:
        logger.warning("  scale probe: no volume columns present; skipping")
        return None

    Xs = pd.DataFrame({"sigma": sigma})
    Xv = Xs.copy()
    for c in volcols:
        Xv[c] = rec[c].to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    Xp = Xs.copy()
    for c in volcols:
        Xp[c] = rng.permutation(rec[c].to_numpy(dtype=float))

    scale_sig, _ = scale_model.cross_fit(Xs, resid, folds, fallback=sigma)
    scale_vol, nv = scale_model.cross_fit(Xv, resid, folds, fallback=sigma)
    scale_shuf, _ = scale_model.cross_fit(Xp, resid, folds, fallback=sigma)

    # Stratify on sigma (the documented tilt axis) and on a CONTINUOUS volume
    # proxy (the new axis under test). Skip volume_missing -- a 0/1 flag has no
    # deciles and collapses the stratum error to 0. Prefer a level/z-score.
    _pref = [
        "volume_mean_30d",
        "volume_mean_60d",
        "volume_zscore_30d",
        "volume_lag_7d",
        "volume_mean_7d",
        "volume_lag_1d",
    ]
    _cont = [c for c in _pref if c in volcols] + [c for c in volcols if c != "volume_missing"]
    strat_vol_col = _cont[0] if _cont else volcols[0]
    strat_vol = rec[strat_vol_col].to_numpy(dtype=float)

    out = {}
    for name, denom in [
        ("sigma", sigma),
        ("scale_sigma_only", scale_sig),
        ("scale_volume", scale_vol),
        ("scale_placebo", scale_shuf),
    ]:
        out[name] = {
            "err_sigma_strata_pp": _stratified_coverage(resid, denom, sigma),
            "err_volume_strata_pp": _stratified_coverage(resid, denom, strat_vol),
        }
    out["_meta"] = {
        "n_rows": len(rec),
        "n_folds": int(rec["fold"].nunique()),
        "n_scale_models": nv,
        "volcols": volcols,
        "strat_vol_col": strat_vol_col,
    }
    return out


def _build_frame_uncached(max_items):
    import duckdb

    con = duckdb.connect()
    db = SessionLocal()

    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()

        # ── Load items ──────────────────────────────────────────────
        pq_files = sorted([str(p) for p in ARCHIVE_DIR.glob("prices-*.parquet")])
        pq_queries = []
        for pqf in pq_files:
            cols = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{pqf}')").fetchall()
            col_names = {r[0] for r in cols}
            if "source" in col_names:
                pq_queries.append(
                    f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}') WHERE (source IS NULL OR source = 'aggregator_sync') AND {_UNIVERSE}"
                )
            else:
                pq_queries.append(
                    f"SELECT item_slug, day, mean_price, volume FROM read_parquet('{pqf}') WHERE {_UNIVERSE}"
                )
        union_sql = " UNION ALL BY NAME ".join(pq_queries)

        rows = con.sql(f"""
            SELECT item_slug,
                   MIN(day) AS first_day,
                   MAX(day) AS last_day,
                   COUNT(*) AS row_count,
                   MEDIAN(mean_price) AS med_price
            FROM ({union_sql})
            GROUP BY item_slug
            HAVING row_count >= 90
               AND MEDIAN(mean_price) >= {MIN_MEDIAN_PRICE}
            -- item_slug breaks row_count ties. Without it DuckDB's parallel
            -- top-N picks a different item universe (and a different item
            -- order, which feeds LightGBM's row sampling) on every run —
            -- measured at ~1pp of dir-acc swing between two identical runs,
            -- the same size as the effect this A/B is trying to detect.
            ORDER BY row_count DESC, item_slug
            LIMIT {max_items}
        """).fetchall()

        items = rows
        logger.info(f"  {len(items)} items for evaluation")

        # ── Load all price data ─────────────────────────────────────
        # One scan of the archive for every item, not one scan per item.
        # Row order still matches the per-item loop this replaced: items in
        # `items` order (row_count DESC), day-ascending within each item.
        slugs = [r[0] for r in items]
        placeholders = ", ".join("?" for _ in slugs)
        all_prices = con.sql(
            f"""
            SELECT item_slug AS item_id, day AS timestamp,
                   mean_price AS price, volume
            FROM ({union_sql})
            WHERE item_slug IN ({placeholders})
        """,
            params=slugs,
        ).df()

        all_prices["timestamp"] = pd.to_datetime(all_prices["timestamp"])
        all_prices["date"] = all_prices["timestamp"].dt.date
        all_prices["item_id"] = pd.Categorical(all_prices["item_id"], categories=slugs, ordered=True)
        all_prices = all_prices.sort_values(["item_id", "timestamp"], kind="stable").reset_index(drop=True)
        all_prices["item_id"] = all_prices["item_id"].astype(str)

        # Feed repair: replace the native `volume` column with the iflow
        # post-restructure series (never coalesced -- see the IFLOW_VOLUME note,
        # and _load_iflow_volume refuses panels reaching into the pre-2024-02-13
        # `count_in_24` era rather than splicing the two).
        # Restrict the panel to the iflow era first: iflow begins 2022-04-18, so
        # pre-2022 rows can carry no volume by construction and would enter the
        # treatment arm as 68% dead-weight zeros, diluting the effect and making
        # the join-coverage gate measure the panel span, not the feed. The upper
        # bound (VOLUME_LIVE_THROUGH) is applied below. This IS the "2022-2026
        # panel" the prereg scoped.
        if IFLOW_VOLUME:
            _pre_era = len(all_prices)
            all_prices = all_prices[all_prices["timestamp"] >= pd.Timestamp(IFLOW_ERA_FROM)].reset_index(drop=True)
            logger.info(f"  iflow era >= {IFLOW_ERA_FROM}: {_pre_era:,} -> {len(all_prices):,} price rows")
            ivol = _load_iflow_volume(con, slugs)
            all_prices = all_prices.merge(ivol, on=["item_id", "date"], how="left")
            _cov = float(all_prices["iflow_volume"].notna().mean())
            logger.info(
                f"  iflow volume join: {_cov:.1%} of price rows carry post-restructure volume "
                f"({all_prices['iflow_volume'].notna().sum():,} of {len(all_prices):,})"
            )
            if _cov < 0.80:
                logger.warning(
                    f"  iflow join coverage {_cov:.1%} < 80% -- feed not repaired "
                    f"for this cohort; the prereg voids this run."
                )
            all_prices["volume"] = all_prices["iflow_volume"]
            all_prices = all_prices.drop(columns=["iflow_volume"])

        # Skinport live repair: same single-series rule, free trailing-24h sale counts.
        if SKINPORT_VOLUME:
            _pre_era = len(all_prices)
            all_prices = all_prices[all_prices["timestamp"] >= pd.Timestamp(SKINPORT_ERA_FROM)].reset_index(drop=True)
            logger.info(f"  skinport-live era >= {SKINPORT_ERA_FROM}: {_pre_era:,} -> {len(all_prices):,} price rows")
            nvol = _load_skinport_live_volume(con, slugs)
            all_prices = all_prices.merge(nvol, on=["item_id", "date"], how="left")
            _cov = float(all_prices["live_volume"].notna().mean())
            logger.info(
                f"  skinport-live volume join: {_cov:.1%} of price rows carry "
                f"sales_24h ({all_prices['live_volume'].notna().sum():,} of {len(all_prices):,})"
            )
            if _cov < 0.80:
                logger.warning(
                    f"  skinport-live join coverage {_cov:.1%} < 80% -- feed not "
                    f"repaired for this cohort; the prereg voids this run."
                )
            all_prices["volume"] = all_prices["live_volume"]
            all_prices = all_prices.drop(columns=["live_volume"])
            # The live feed is ongoing, so the archive arm's VOLUME_LIVE_THROUGH
            # (2026-04-30, the native cliff) would empty this panel. Bound it
            # instead by the last accumulated live day — anything later has no
            # volume by construction and would enter as dead weight.
            _live_through = nvol["date"].max()
            _pre_cut = len(all_prices)
            all_prices = all_prices[all_prices["date"] <= _live_through].reset_index(drop=True)
            logger.info(f"  skinport-live cutoff {_live_through}: {_pre_cut:,} -> {len(all_prices):,} price rows")

        # Cut before feature engineering, not after: a 30d/60d rolling volume
        # window that straddles the cliff would average real volume with the
        # zeros and quietly corrupt the treatment arm. Skipped for the live
        # arm, which bounds itself above (no cliff inside its era).
        _pre = len(all_prices)
        if not SKINPORT_VOLUME:
            all_prices = all_prices[all_prices["timestamp"] <= pd.Timestamp(VOLUME_LIVE_THROUGH)].reset_index(drop=True)
        logger.info(f"  Volume-live cutoff {VOLUME_LIVE_THROUGH}: {_pre:,} -> {len(all_prices):,} price rows")

        # ── Build features once ─────────────────────────────────────
        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)

        EXCLUDE = {"item_id", "date", "timestamp", "price", "volume", "name", "release_date"}
        all_feature_cols = [
            c for c in df.columns if c not in EXCLUDE and df[c].dtype in (np.float64, np.float32, np.int64, int, float)
        ]

        # Prune highly correlated
        if len(all_feature_cols) > 2:
            corr = df[all_feature_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                if col in to_drop:
                    continue
                highly_corr = upper[col][upper[col] > CORR_PRUNE_THRESHOLD].index
                to_drop.update(highly_corr)
            pruned = [c for c in all_feature_cols if c not in to_drop]
        else:
            pruned = all_feature_cols

        logger.info(f"  Full feature count: {len(all_feature_cols)} → {len(pruned)} (after corr prune)")

        present_new = [c for c in NEW_PRIMITIVES if c in pruned]
        logger.info(f"  Volume features present after prune: {present_new}")

        return df, pruned, present_new

    finally:
        con.close()


def run_evaluation(df, pruned, present_new, horizon_filter=None, arm_filter=None, n_jobs=None, q50_only=False):
    """Walk-forward evaluation over the prebuilt frame.

    Returns results[horizon][arm]. `horizon_filter` restricts to one horizon so
    the 4 horizons can run as independent processes; `n_jobs` must then be
    sized to cores/workers, or the shards oversubscribe the machine and each
    one runs slower than it would alone.
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    quantiles = [0.5] if q50_only else [0.1, 0.5, 0.9]
    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)

    try:
        # Three arms. baseline/treatment differ only by the 6 new columns.
        # placebo keeps the columns but shuffles their values (handled per-fold).
        #
        # Base is the model production serves: shelve + allowlist applied to the
        # engineered frame. The tested volume features are price_technicals but
        # SHELVED in production, so they are excluded from the base and added back
        # only in treatment/placebo — the honest "does adding these to production
        # help?" experiment. Before this the arms differed by a handful of columns
        # on a ~138-column base production does not serve. See 2026-08-13 repin.
        base_cols = [c for c in pruned if c not in ItemForecaster.SHELVED_FEATURES]
        base_cols = ItemForecaster._apply_feature_allowlist(base_cols, ItemForecaster.FEATURE_GROUP_ALLOWLIST)
        subsets = {
            "baseline": base_cols,
            "treatment": base_cols + present_new,
            "placebo": base_cols + present_new,
        }
        if arm_filter is not None:
            subsets = {k: v for k, v in subsets.items() if k == arm_filter}
        for name, cols in subsets.items():
            logger.info(f"    {name:20s}: {len(cols):>3d} features")

        # ── Horizons to evaluate ────────────────────────────────────
        horizons = ItemForecaster.HORIZONS
        if horizon_filter is not None:
            horizons = [h for h in horizons if h == horizon_filter]

        # Store results: results[horizon][config_name] = metrics dict
        results = {}

        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}")
            logger.info(f"  Evaluating {horizon}d horizon...")
            logger.info(f"  {'=' * 60}")

            tdf = forecaster.prepare_targets(df, horizon)
            tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy()
            tdf = tdf.sort_values(["item_id", "date"])

            if tdf.empty:
                logger.warning(f"    No valid targets for {horizon}d")
                continue

            dates = sorted(tdf["date"].unique())
            split_idx = int(len(dates) * SPLIT_FRACTION)

            # Fold membership by range comparison on datetime64, not
            # `date.isin(train_dates)`. train_dates is always a prefix of
            # `dates`, so the isin it replaces was hashing a list that grew to
            # thousands of entries against the whole frame, once per fold per
            # arm, to select a contiguous block.
            tdf_days = pd.to_datetime(tdf["date"]).to_numpy()
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()

            results[horizon] = {}
            _scale_probe_rows = []  # baseline-arm OOF residuals for SCALE_PROBE

            for config_name, fc in subsets.items():
                # Skip if too few features
                if len(fc) < 3:
                    logger.info(f"    [{config_name}] Skipping — only {len(fc)} features")
                    continue

                logger.info(f"\n    --- Config: {config_name} ({len(fc)} features) ---")

                # Both invariant across folds: hoisted out of the loop, and the
                # per-fold slice then copies ~len(available) columns instead of
                # every column in the engineered frame.
                available = [c for c in fc if c in tdf.columns]
                if not available:
                    logger.info(f"    [{config_name}] Skipping — no features present")
                    continue
                target_col = f"target_return_{horizon}d"
                # `item_id` rides along for the paired records' pairing key;
                # `date` for the 200K row cap's sort below. Neither is a
                # feature -- `available` is the feature list and the matrices
                # are built from it alone.
                sub = tdf[[*available, target_col, "price", "date", "item_id"]]

                directional_hits = 0
                directional_total = 0
                # Served-cohort breakout. The pooled metric is dominated by
                # sub-$1 items (82% of the training cohort) where every
                # feature -- including price momentum -- measures ~0, so a
                # pooled null cannot distinguish "no signal" from "wrong
                # cohort". Mirrors production's classifier_accuracy_ge1.
                ge1_hits = 0
                ge1_total = 0
                # Flat-actual rows are not a directional call. An exactly-zero
                # forward return makes sign(actual)=0, which the q50 matches by
                # emitting exactly 0.0 -- a free hit that inflated the pooled
                # metric by ~31pp on the old penny cohort. Score them out.
                strict_hits = 0
                strict_total = 0
                strict_ge1_hits = 0
                strict_ge1_total = 0
                mae_total = 0.0
                mae_count = 0
                interval_hits = 0
                interval_total = 0
                per_fold = []
                records = []

                step = STEP_DAYS
                for window_end in range(split_idx + 1, len(dates), step):
                    val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                    if len(val_dates) < 7:
                        continue

                    train_df = ItemForecaster._purge_overlapping_train_rows(
                        sub[tdf_days <= dates_dt[window_end - 1]], val_dates[0], horizon
                    )
                    val_df = sub[
                        (tdf_days >= dates_dt[window_end]) & (tdf_days <= dates_dt[window_end + len(val_dates) - 1])
                    ]

                    if config_name == "placebo" and present_new:
                        rng = np.random.default_rng(42)
                        train_df = train_df.copy()
                        val_df = val_df.copy()
                        for col in present_new:
                            train_df[col] = rng.permutation(train_df[col].values)
                            val_df[col] = rng.permutation(val_df[col].values)

                    if len(val_df) < 50:
                        continue

                    if len(train_df) > 200000:
                        train_df = train_df.sort_values("date").tail(200000)

                    # One median pass, used to impute both matrices (it was
                    # recomputed over the full training block for each).
                    train_median = train_df[available].median()
                    X_train = train_df[available].fillna(train_median)
                    y_train = train_df[target_col]
                    X_val = val_df[available].fillna(train_median)
                    y_val = val_df[target_col]

                    # Bin once for all quantiles: only `alpha` differs between
                    # them, and alpha plays no part in Dataset construction, so
                    # rebuilding per quantile re-binned identical data 3x.
                    Xtr_v, Xv_v = X_train.values, X_val.values
                    dtrain = lgb.Dataset(Xtr_v, y_train.values, params=DS_PARAMS, free_raw_data=False)
                    dval = lgb.Dataset(Xv_v, y_val.values, reference=dtrain, params=DS_PARAMS, free_raw_data=False)

                    models = {}
                    for q in quantiles:
                        params = {
                            "objective": "quantile",
                            "alpha": q,
                            "metric": "quantile",
                            "boosting_type": "gbdt",
                            "num_leaves": 31,
                            "max_depth": 5,
                            "min_data_in_leaf": 15,
                            "min_gain_to_split": 0.1,
                            "learning_rate": 0.03,
                            "feature_fraction": 0.7,
                            "bagging_fraction": 0.7,
                            "bagging_freq": 5,
                            "lambda_l1": 0.5,
                            "lambda_l2": 0.5,
                            "verbosity": -1,
                            "random_state": 42,
                            "n_jobs": n_jobs,
                            # LightGBM's auto row/col-wise choice picks
                            # col-wise below ~5 threads on this shape and falls
                            # off a cliff: measured 386s vs 64s for the same
                            # 7d/3-arm unit at n_jobs=3, bit-identical results.
                            # Sharding by horizon means low per-process thread
                            # counts, so this must be pinned.
                            "force_row_wise": True,
                            **DS_PARAMS,
                        }
                        # Production's trainer and round table. The old call
                        # early-stopped on `dval` and scored `Xv_v` — the same
                        # rows. `dval` is ignored unless EARLY_STOPPING=1.
                        model = ItemForecaster._train_ensemble_member(
                            params,
                            dtrain,
                            dval,
                            num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
                            early_stopping=ItemForecaster._early_stopping_enabled(),
                        )
                        models[q] = model.predict(Xv_v)

                    p50_ret = models[0.5]
                    # q50-only mode trains no interval bounds. Fall back to a
                    # zero-width band so the arithmetic below stays defined,
                    # and report coverage as unavailable rather than as a
                    # number that looks real (see `int_cov` below).
                    p10_ret = models.get(0.1, p50_ret)
                    p90_ret = models.get(0.9, p50_ret)

                    # Fix quantile crossing
                    crossing_mask = (p10_ret > p50_ret) | (p50_ret > p90_ret)
                    non_crossing = ~crossing_mask
                    low_ret = np.minimum(p10_ret, p50_ret)
                    high_ret = np.maximum(p50_ret, p90_ret)
                    if non_crossing.any():
                        avg_hw = np.mean(
                            [
                                np.mean(p50_ret[non_crossing] - p10_ret[non_crossing]),
                                np.mean(p90_ret[non_crossing] - p50_ret[non_crossing]),
                            ]
                        )
                        if avg_hw > 0:
                            low_ret[crossing_mask] = p50_ret[crossing_mask] - avg_hw
                            high_ret[crossing_mask] = p50_ret[crossing_mask] + avg_hw
                    low_ret = np.minimum(low_ret, p50_ret)
                    high_ret = np.maximum(high_ret, p50_ret)

                    cp = val_df["price"].to_numpy(dtype=float)
                    ar = y_val.to_numpy(dtype=float)

                    # Vectorized restatement of the former per-row loop. NaN
                    # returns map to sign 0, matching the old string-compare
                    # path where a NaN failed both > and < and fell to "flat".
                    fold_total = len(val_df)
                    fold_hits = int(np.count_nonzero(np.sign(np.nan_to_num(ar)) == np.sign(np.nan_to_num(p50_ret))))

                    abs_err = np.abs(cp * (1 + p50_ret / 100) - cp * (1 + ar / 100))
                    fold_mae = float(abs_err.sum())

                    actual_future = cp * (1 + ar / 100)
                    fold_int_hits = int(
                        np.count_nonzero(
                            (cp * (1 + low_ret / 100) <= actual_future) & (actual_future <= cp * (1 + high_ret / 100))
                        )
                    )
                    fold_int_total = fold_total

                    directional_hits += fold_hits
                    directional_total += fold_total

                    _match = np.sign(np.nan_to_num(ar)) == np.sign(np.nan_to_num(p50_ret))
                    _nonflat = np.asarray(ar) != 0
                    strict_hits += int(np.count_nonzero(_match & _nonflat))
                    strict_total += int(_nonflat.sum())

                    _ge1 = np.asarray(cp) >= 1.0
                    strict_ge1_hits += int(np.count_nonzero(_match & _nonflat & _ge1))
                    strict_ge1_total += int((_nonflat & _ge1).sum())

                    # Paired records for the fold-clustered interval below.
                    # The cohort is the harness's strictest one -- non-flat
                    # actuals at >=$1 -- because that is the population
                    # production serves, and it is arm-independent, so the two
                    # arms pair row for row. `window_end` rather than a running
                    # counter: a counter drifts the moment one arm skips a fold
                    # the other kept.
                    # Band-quality endpoints (the shippable axis -- invariant 4
                    # forbids DA as a claim on its own). `in_interval` is realised
                    # coverage vs the 0.80 nominal; `rel_width` is the interval's
                    # width as a fraction of price, (high-low return)/100. Both are
                    # meaningful only when quantile bounds were trained; in q50-only
                    # mode low_ret==high_ret==p50_ret so rel_width is 0 and coverage
                    # degenerate -- do not interpret the width contrast in that mode.
                    _in_interval = (
                        (cp * (1 + low_ret / 100) <= actual_future) & (actual_future <= cp * (1 + high_ret / 100))
                    ).astype(float)
                    _rel_width = (high_ret - low_ret) / 100.0
                    records.extend(
                        paired_records(
                            item_ids=val_df["item_id"].to_numpy(),
                            forecast_dates=val_df["date"].to_numpy(),
                            fold_id=window_end,
                            keep=_nonflat & _ge1,
                            direction_correct=_match,
                            in_interval=_in_interval,
                            rel_width=_rel_width,
                        )
                    )

                    # SCALE_PROBE: capture the baseline model's OOF residuals on
                    # the same >=$1 non-flat cohort. sigma and volume are joined
                    # from `df` after the loop, so nothing here depends on the
                    # arm's column set. residual_pct = actual - predicted (pct).
                    if SCALE_PROBE and config_name == "baseline":
                        _keep = _nonflat & _ge1
                        _scale_probe_rows.append(
                            pd.DataFrame(
                                {
                                    "item_id": val_df["item_id"].to_numpy()[_keep],
                                    "date": val_df["date"].to_numpy()[_keep],
                                    "residual_pct": (ar - p50_ret)[_keep],
                                    "fold": window_end,
                                }
                            )
                        )

                    if _ge1.any():
                        ge1_hits += int(
                            np.count_nonzero((np.sign(np.nan_to_num(ar)) == np.sign(np.nan_to_num(p50_ret))) & _ge1)
                        )
                        ge1_total += int(_ge1.sum())
                    mae_total += fold_mae
                    mae_count += fold_total
                    interval_hits += fold_int_hits
                    interval_total += fold_total

                    per_fold.append(
                        {
                            "fold": len(per_fold) + 1,
                            "val_start": str(val_dates[0]),
                            "val_end": str(val_dates[-1]),
                            "dir_acc": round(fold_hits / fold_total * 100, 1) if fold_total > 0 else 0,
                            "mae": round(fold_mae / fold_total, 4) if fold_total > 0 else 0,
                            "int_cov": round(fold_int_hits / fold_int_total * 100, 1) if fold_int_total > 0 else 0,
                            "n": fold_total,
                        }
                    )

                if directional_total > 0:
                    dir_acc = directional_hits / directional_total * 100
                    mae = mae_total / mae_count if mae_count > 0 else 0
                    int_cov = interval_hits / interval_total * 100 if interval_total > 0 else 0

                    fold_accs = [f["dir_acc"] for f in per_fold]
                    baseline_2class = 50.0
                    result = {
                        "directional_accuracy": round(dir_acc, 2),
                        "dir_acc": round(dir_acc, 2),
                        "dir_acc_ge1": round(ge1_hits / ge1_total * 100, 2) if ge1_total else None,
                        "n_ge1": ge1_total,
                        "dir_acc_strict": round(strict_hits / strict_total * 100, 2) if strict_total else None,
                        "n_strict": strict_total,
                        "dir_acc_strict_ge1": round(strict_ge1_hits / strict_ge1_total * 100, 2)
                        if strict_ge1_total
                        else None,
                        "n_strict_ge1": strict_ge1_total,
                        "pct_flat": round(100.0 * (1 - strict_total / directional_total), 2)
                        if directional_total
                        else None,
                        "mae": round(mae, 4),
                        # None, not a number, when there was no interval to
                        # score — a zero-width band would otherwise report a
                        # plausible-looking coverage figure that means nothing.
                        "interval_coverage": None if q50_only else round(int_cov, 2),
                        "sample_count": directional_total,
                        "effective_baseline": baseline_2class,
                        "fold_count": len(per_fold),
                        "fold_mean_dir_acc": round(np.mean(fold_accs), 1) if fold_accs else 0,
                        "fold_std_dir_acc": round(np.std(fold_accs), 1) if len(fold_accs) > 1 else 0,
                        "fold_min_dir_acc": round(min(fold_accs), 1) if fold_accs else 0,
                        "fold_max_dir_acc": round(max(fold_accs), 1) if fold_accs else 0,
                        # Retained so the merge step can compute per-fold win
                        # counts between arms, not just the pooled delta.
                        "per_fold": per_fold,
                        # The pairing input for the fold-clustered interval.
                        "records": records,
                    }
                    result["improvement_over_baseline_pp"] = round(dir_acc - baseline_2class, 1)
                    results[horizon][config_name] = result

                    logger.info(
                        f"      DirAcc={dir_acc:.1f}% ({directional_total:,} samples, "
                        f"{result['improvement_over_baseline_pp']:.1f}pp above baseline)"
                    )

            # Fold-clustered paired intervals, the harness's actual verdict.
            # Until 2026-08-08 this reported a pooled `treatment - baseline`
            # delta against a +/-0.5pp emoji threshold, which is not a test:
            # the item-level MDE here is 2.21-3.69pp, so a 0.5pp "win" is
            # inside the noise floor by a factor of five.
            arms = {
                a: r.get("records", [])
                for a, r in results[horizon].items()
                if not a.startswith("_") and r.get("records")
            }
            if "baseline" in arms and len(arms) > 1:
                # DA stays (diagnostic only, per invariant 4). The ship gate is
                # rel_width (lower better) subject to the coverage gate; both are
                # fold-clustered contrasts on the same paired records.
                contrasts = paired_arm_contrasts(arms, base="baseline")
                width = paired_arm_contrasts(arms, base="baseline", value_key="rel_width", higher_is_better=False)
                coverage = paired_arm_contrasts(arms, base="baseline", value_key="in_interval", higher_is_better=True)
                results[horizon]["_paired_vs_baseline"] = contrasts
                results[horizon]["_paired_width_vs_baseline"] = width
                results[horizon]["_paired_coverage_vs_baseline"] = coverage
                for arm in contrasts:
                    logger.info(f"      paired {arm:<10} DA:       {format_paired(contrasts[arm])}")
                    logger.info(f"      paired {arm:<10} rel_width: {format_paired(width[arm])}")
                    logger.info(f"      paired {arm:<10} coverage: {format_paired(coverage[arm])}")

                # Gating width robustness (refold prereg): treatment vs baseline
                # must survive leave-one-fold-out and leave-out-2025-10.
                if "treatment" in arms:
                    worst, no_oct = _width_robustness(arms["baseline"], arms["treatment"])
                    if worst is not None:
                        _s = "SURVIVES" if worst["ci_upper"] < 0 else "FAILS"
                        logger.info(
                            f"      robustness LOO width [{_s}]: worst drop="
                            f"fold {worst['dropped_fold']} -> "
                            f"{worst['mean_diff']:+.3f} "
                            f"[{worst['ci_lower']:+.3f}, {worst['ci_upper']:+.3f}]"
                        )
                    _s2 = "SURVIVES" if no_oct.get("ci_upper") is not None and no_oct["ci_upper"] < 0 else "FAILS"
                    logger.info(
                        f"      robustness -2025-10 width [{_s2}]: dropped "
                        f"{no_oct['n_oct25_rows_dropped']:,} rows -> "
                        f"{no_oct['mean_diff']:+.3f} "
                        f"[{no_oct['ci_lower']:+.3f}, {no_oct['ci_upper']:+.3f}]"
                    )
                    results[horizon]["_width_robustness"] = {"loo_worst": worst, "no_oct25": no_oct}
            else:
                # `--arm` shards one arm per process, so a shard has nothing to
                # contrast against. Say so: a missing verdict must not read as
                # an absent effect. `merge_price_primitives_ab.py` pairs the
                # shards at fold grain from `per_fold`.
                logger.info(
                    f"      paired: not computed — this run holds "
                    f"{sorted(arms) or 'no'} arm(s). Merge the shards with "
                    f"scripts/merge_price_primitives_ab.py for the verdict."
                )

            if SCALE_PROBE and _scale_probe_rows:
                if "price_std_60d" not in df.columns:
                    logger.warning("  scale probe: price_std_60d absent; skipping")
                else:
                    rec = pd.concat(_scale_probe_rows, ignore_index=True)
                    cols_needed = ["item_id", "date", "price", "price_std_60d"] + [
                        c for c in present_new if c in df.columns
                    ]
                    merged = rec.merge(df[cols_needed], on=["item_id", "date"], how="left")
                    with np.errstate(divide="ignore", invalid="ignore"):
                        merged["sigma"] = merged["price_std_60d"] / merged["price"]
                    probe = _run_scale_probe(merged, present_new)
                    if probe is not None:
                        results[horizon]["_scale_probe"] = probe
                        m = probe["_meta"]
                        logger.info(
                            f"      SCALE PROBE h={horizon}: {m['n_rows']:,} rows, "
                            f"{m['n_folds']} folds, {m['n_scale_models']} scale "
                            f"models, strat_vol={m['strat_vol_col']}"
                        )
                        for nm in ("sigma", "scale_sigma_only", "scale_volume", "scale_placebo"):
                            e = probe[nm]
                            logger.info(
                                f"        {nm:<17} sigma-strata "
                                f"err={e['err_sigma_strata_pp']:.2f}pp  "
                                f"volume-strata "
                                f"err={e['err_volume_strata_pp']:.2f}pp"
                            )

        logger.info("\n" + "=" * 64)
        logger.info("SHIP DECISION SUMMARY (dir-acc %, treatment vs baseline vs placebo)")
        logger.info("=" * 64)
        for h in sorted(results):
            r = results[h]
            if not all(k in r for k in ("baseline", "treatment", "placebo")):
                continue
            b = r["baseline"]["dir_acc"]
            t = r["treatment"]["dir_acc"]
            p = r["placebo"]["dir_acc"]
            logger.info(
                f"  {h:>2}d  baseline={b:5.2f}  treatment={t:5.2f}  "
                f"placebo={p:5.2f}  (t-b={t - b:+.2f}, t-p={t - p:+.2f})"
            )
            for label, key, nkey in (
                (">=$1 ", "dir_acc_ge1", "n_ge1"),
                ("STRICT", "dir_acc_strict", "n_strict"),
                ("STR>=1", "dir_acc_strict_ge1", "n_strict_ge1"),
            ):
                bg, tg, pg = (r[k].get(key) for k in ("baseline", "treatment", "placebo"))
                if None in (bg, tg, pg):
                    continue
                logger.info(
                    f"       {label} baseline={bg:5.2f}  treatment={tg:5.2f}  "
                    f"placebo={pg:5.2f}  (t-b={tg - bg:+.2f}, t-p={tg - pg:+.2f})"
                    f"  n={r['treatment'][nkey]:,}"
                )
            logger.info(f"       flat-actual rows: {r['treatment'].get('pct_flat')}%")

        return results

    finally:
        db.close()


def print_comparison(results):
    """Print a comparison table across arms (baseline/treatment/placebo) and horizons."""
    config_order = ["baseline", "treatment", "placebo"]
    config_labels = {
        "baseline": "Baseline (no primitives)",
        "treatment": "Treatment (+6 primitives)",
        "placebo": "Placebo (shuffled)",
    }

    print("\n" + "=" * 100)
    print("A/B TEST — Price Technical Primitives (volatility asymmetry + oscillator divergence)")
    print("=" * 100)

    for horizon in sorted(results.keys()):
        h_results = results[horizon]
        print(f"\n  ┌─ {horizon}d Horizon {'─' * 60}┐")

        header = f"  │ {'Arm':<26} {'DirAcc':>8} {'vs Base':>9} {'MAE':>8} {'IntCov':>8} {'Folds':>6} {'Samples':>9}"
        sep = f"  │ {'─' * 26} {'─' * 8} {'─' * 9} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 9}"
        base_dir_acc = h_results.get("baseline", {}).get("directional_accuracy", 0)

        print(header)
        print(sep)

        for cfg in config_order:
            r = h_results.get(cfg)
            if r is None:
                continue
            dir_acc = r["directional_accuracy"]
            delta = dir_acc - base_dir_acc
            delta_str = f"{delta:+.2f}pp" + (" ✅" if delta > 0.5 else " ❌" if delta < -0.5 else "  ")
            label = config_labels.get(cfg, cfg)
            cov = r.get("interval_coverage")
            cov_str = f"{cov:>6.1f}%" if cov is not None else f"{'n/a':>7}"
            print(
                f"  │ {label:<26} {dir_acc:>7.1f}% {delta_str:>9} ${r['mae']:>5.2f} "
                f"{cov_str} {r['fold_count']:>5}  {r['sample_count']:>8,}"
            )

        print(f"  └{'─' * 78}┘")

    # ── Ship decision guidance ───────────────────────────────────────
    print(f"\n  {'=' * 100}")
    print("  INTERPRETATION")
    print(f"  {'=' * 100}")
    print("")
    print("  'vs Base' compares each arm to baseline (the volume columns dropped).")
    print("  Ship the primitives only if treatment beats baseline by a meaningful,")
    print("  non-flat margin AND treatment beats placebo (rules out capacity inflation)")
    print("  AND no horizon regresses beyond the 0.5-1.5pp budget.")

    # ── Verdict: new primitives, short vs long horizons ─────────────
    short_horizons, long_horizons = [3, 7], [14, 30]
    short_deltas, long_deltas = [], []
    for h in results:
        h_res = results[h]
        base = h_res.get("baseline", {}).get("directional_accuracy")
        treat = h_res.get("treatment", {}).get("directional_accuracy")
        if base is not None and treat is not None:
            delta = treat - base
            if h in short_horizons:
                short_deltas.append(delta)
            if h in long_horizons:
                long_deltas.append(delta)

    print("\n  New Primitives (treatment vs baseline):")
    if short_deltas:
        avg_short = np.mean(short_deltas)
        print(
            f"    Short horizons (3d/7d):    avg Δ = {avg_short:+.2f}pp "
            f"{'📈 helpful' if avg_short > 0.3 else '📉 harmful' if avg_short < -0.3 else '➡️ neutral'}"
        )
    if long_deltas:
        avg_long = np.mean(long_deltas)
        print(
            f"    Long horizons (14d/30d):   avg Δ = {avg_long:+.2f}pp "
            f"{'📈 helpful' if avg_long > 0.3 else '📉 harmful' if avg_long < -0.3 else '➡️ neutral'}"
        )

    print("")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="A/B test: price technical primitives contribution per horizon")
    parser.add_argument("--max-items", type=int, default=200, help="Number of items to evaluate (default: 200)")
    parser.add_argument("--horizon", type=int, default=None, help="Only evaluate this horizon (default: all)")
    parser.add_argument(
        "--arm",
        choices=["baseline", "treatment", "placebo"],
        default=None,
        help="Only evaluate this arm (default: all three)",
    )
    parser.add_argument("--frame-cache", default=None, help="Read/write the engineered frame at this path")
    parser.add_argument(
        "--build-cache-only", action="store_true", help="Build the frame cache and exit (run once before workers)"
    )
    parser.add_argument("--out", default=None, help="Write results JSON here instead of stdout")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="LightGBM threads per process. When sharding by "
        "horizon, set this to cores/shards (default: "
        "cores/2, matching production training)",
    )
    parser.add_argument(
        "--q50-only",
        action="store_true",
        help="Train only the median quantile — 3x faster, and "
        "the ship rule only reads dir-acc. Interval "
        "coverage is not meaningful in this mode",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("A/B TEST: Shelved Volume Features (baseline/treatment/placebo)")
    logger.info("=" * 70)

    df, pruned, present_new = build_frame(max_items=args.max_items, cache_path=args.frame_cache)

    if args.build_cache_only:
        logger.info("Frame cache built; exiting before evaluation.")
        return 0

    results = run_evaluation(
        df,
        pruned,
        present_new,
        horizon_filter=args.horizon,
        arm_filter=args.arm,
        n_jobs=args.n_jobs,
        q50_only=args.q50_only,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(without_records(results), indent=2, default=str))
        logger.info(f"Wrote {args.out}")
        return 0

    print_comparison(results)
    print(f"\n  JSON: {json.dumps(without_records(results), indent=2, default=str)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
