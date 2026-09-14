#!/usr/bin/env python3
"""A/B test: `anomaly_p` as a band-WIDTH modulator on the climatology scale.

The anomaly GBM is the only head here that ever beat a featureless null
(held-out AUC +0.13-0.14 at every horizon, log loss better at 3/7/14d;
`docs/changelog/2026-09-09-anomaly-head-beats-its-null.md`). Items with high
`anomaly_p` are about to make outsized moves, so they should get wider bands.
This harness tests exactly that: multiply the served climatology scale by
`f(anomaly_p)`, a monotone transform fitted to equalize per-decile coverage.

This is a CONDITIONAL COVERAGE fix, not a sixth sigma-family denominator. The
denominators (sigma / beta / learned / exceedance / climatology) are
alternatives — `_calibrate_conformal` raises if combined — while this is a
MODIFIER of the climatology scale, in the same sense as the reactive EWMA
multiplier and the vol-rank multiplier: it has no effect unless climatology is
the served scale, and in production `q_hat` would be calibrated on the
modulated scale (matched pair). The harness scores at MATCHED 80% coverage per
arm per fold (each arm gets its own q80 on the eval rows), so no `q_hat` is
needed and a pure re-levelling cannot read as a win.

## Arms

    control:      flat K=320 climatology from production's own
                  `_build_climatology_table` on the fit split.
    anomaly_mod:  control * f(anomaly_p). The anomaly head is production's own
                  `_fit_anomaly_classifier` on the fit split; f is the
                  per-decile q80 ratio (bin q80 / global q80) on the fit split,
                  isotonised to monotone non-decreasing, normalised to mean 1.0
                  on the fit split (so `q_hat`'s level is preserved, as with
                  vol-rank), clipped to [0.25, 4.0].
    shuffled_mod: control * f_shuffled, where f_shuffled is fitted identically
                  but on PERMUTED fit anomaly_p. Same capacity (ten monotone
                  numbers), no signal. If this arm also "wins", the metric is
                  broken, not the modulator.

## Metrics (per fold, served cohort >= $1 only)

* `log_width` at matched 80% coverage — the width read. Negative paired delta
  (arm - control) means narrower at equal coverage.
* `decile_err` — mean |per-anomaly-decile coverage - 80%| in pp, evaluated at
  each arm's OWN matched q80 (level-matched, so a level shift cannot score as
  a conditional fix). Negative paired delta means more equal coverage across
  the anomaly axis. Bins are the fit decile edges applied to eval, so both
  arms are scored on identical rows in identical bins.

Fold machinery, item split, row budget, purge (h+13 embargo) and the paired
t-interval are imported wholesale from the harness family rather than
re-derived — the point is a paired read, which only holds if all arms see
identical rows.

An arm earns a production follow-up only with a negative 95% interval on
`decile_err` AND no significant widening on `log_width`. The harness wires
nothing itself.

Usage:
    python -m scripts.anomaly_band_modulator_ab --horizon 7 \\
        --frame-cache /tmp/exc_meta_frame.parquet --out /tmp/anom_mod_h7.json
    python -m scripts.anomaly_band_modulator_ab --max-folds 2  # recent folds only
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Must precede any prepare_targets call: `target_anomaly_{h}d` is only computed
# when the gate is on (forecaster.py), so an unset flag silently yields a
# harness that skips every horizon rather than one that fails.
os.environ["ANOMALY_GBM"] = "1"

import numpy as np
import pandas as pd
from api.serving_policy import MIN_SERVED_PRICE_USD
from database import SessionLocal
from models.forecaster import ItemForecaster
from scripts.ab_test_item_metadata import (
    ROW_BUDGET,
    STEP_DAYS,
    VAL_WINDOW_DAYS,
    _stratified_sample,
    assign_items,
    build_frame,
)
from scripts.exceedance_meta_ab import (
    TREE_PARAMS,
    _score,
    paired_fold_deltas,
)
from scripts.shrink_k_vol_rank_ab import matched_width

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("anomaly_band_modulator_ab")

TARGET_COVERAGE = 0.80
N_DECILES = 10
SHUFFLE_SEED = 917

#: Fewest eval rows a fold may score. The decile error is a binned read; below
#: this the bins are decided by tail draws rather than the arm.
MIN_EVAL_ROWS = 200
#: Fewest served fit rows the multiplier may be fitted on. Ten decile
#: quantiles on less than this is noise chasing.
MIN_FIT_ROWS = 500
#: Fewest rows a decile bin must hold to vote on the multiplier shape. Below
#: this the harness falls back to a flat multiplier rather than serving noise.
MIN_BIN_ROWS = 50
#: Clip bounds of the multiplier, matching the vol-rank production normalisation
#: (clip raw at 0.01, mean-1.0, clip to [0.25, 4.0]): a modifier that may only
#: widen or narrow within the range the served geometry already tolerates.
F_LO, F_HI = 0.25, 4.0


def isotonic_increasing(y, w=None):
    """Pool-adjacent-violators fit: closest non-decreasing sequence to `y`.

    Weighted least squares (weights `w`, default uniform), pure numpy so the
    harness gains no new dependency for a ten-number fit. Returns the fitted
    value per input position, in the input order.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    # Blocks carry (sum, weight, member indices); merge while the block means
    # decrease — the standard PAVA for the increasing case.
    sums = list(y * w)
    weights = list(w)
    members = [[i] for i in range(n)]
    i = 0
    while i < len(sums) - 1:
        if sums[i] / weights[i] <= sums[i + 1] / weights[i + 1]:
            i += 1
            continue
        sums[i] += sums[i + 1]
        weights[i] += weights[i + 1]
        members[i] = members[i] + members[i + 1]
        del sums[i + 1]
        del weights[i + 1]
        del members[i + 1]
        if i > 0:
            i -= 1
    out = np.empty(n)
    for s, wt, idxs in zip(sums, weights, members):
        out[idxs] = s / wt
    return out


def _flat_model():
    """The degenerate multiplier: no edges, a single value of 1.0."""
    return {"edges": np.array([]), "values": np.array([1.0])}


def fit_anomaly_multiplier(p_fit, abs_r_fit, scale_fit, n_bins=N_DECILES):
    """Monotone per-decile width multiplier from fit-split rows.

    Within each anomaly_p decile, `q80_bin` is the quantile of |r|/scale that
    reaches `TARGET_COVERAGE` there; the raw multiplier is `q80_bin /
    global_q80`, i.e. the factor that would give that decile exactly the
    global coverage. Raw ratios are isotonised to non-decreasing (high
    anomaly_p must never mean a NARROWER band — that is the mechanism), then
    normalised to mean 1.0 on the fit rows so the level `q_hat` would be
    calibrated at is preserved, then clipped to [F_LO, F_HI].

    Returns {"edges", "values"} with len(values) == len(edges) + 1. Any
    degenerate input (too few rows, no spread in p, empty bins, non-positive
    global q80) returns the flat model rather than noise — a fold that cannot
    fit a shape votes nothing, it does not vote a spurious one.
    """
    p = np.asarray(p_fit, dtype=float)
    r = np.asarray(abs_r_fit, dtype=float)
    s = np.asarray(scale_fit, dtype=float)
    ok = np.isfinite(p) & np.isfinite(r) & np.isfinite(s) & (s > 0)
    p, r, s = p[ok], r[ok], s[ok]
    if p.size < MIN_FIT_ROWS:
        return _flat_model()
    if not np.all(np.isfinite(p)) or np.min(p) == np.max(p):
        return _flat_model()
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    if edges.size != n_bins - 1 or np.any(~np.isfinite(edges)):
        return _flat_model()
    if np.any(np.diff(edges) <= 0):
        # Ties in anomaly_p collapsed a bin: dedupe and require at least
        # three distinct edges, else there is no axis to condition on.
        edges = np.unique(edges)
        if edges.size < 3:
            return _flat_model()
        n_bins = edges.size + 1
    idx = np.searchsorted(edges, p, side="right")
    scores = r / s
    q_global = float(np.quantile(scores, TARGET_COVERAGE))
    if not np.isfinite(q_global) or q_global <= 0:
        return _flat_model()
    centres, raws, counts = [], [], []
    for k in range(n_bins):
        sel = idx == k
        if int(sel.sum()) < MIN_BIN_ROWS:
            continue
        q_bin = float(np.quantile(scores[sel], TARGET_COVERAGE))
        if not np.isfinite(q_bin) or q_bin <= 0:
            continue
        centres.append(float(np.median(p[sel])))
        raws.append(q_bin / q_global)
        counts.append(int(sel.sum()))
    if len(raws) < 3:
        return _flat_model()
    order = np.argsort(centres, kind="stable")
    iso = isotonic_increasing(np.asarray(raws)[order], np.asarray(counts, dtype=float)[order])
    # Which original bin each kept position came from, in centre order.
    all_bins = np.array([k for k in range(n_bins) if int((idx == k).sum()) >= MIN_BIN_ROWS])
    kept_bins_ordered = all_bins[order]
    kept_centres = np.asarray(centres)[order]
    # Map the isotonised values onto every bin: kept bins take their own
    # fitted value, bins dropped for thinness take the nearest kept centre's,
    # so no bin is left with a hole.
    bin_to_iso = dict(zip(kept_bins_ordered.tolist(), iso.tolist()))
    full = np.empty(n_bins)
    for k in range(n_bins):
        if k in bin_to_iso:
            full[k] = bin_to_iso[k]
            continue
        sel = idx == k
        c = float(np.median(p[sel])) if sel.any() else float(np.quantile(p, (k + 0.5) / n_bins))
        full[k] = iso[int(np.argmin(np.abs(kept_centres - c)))]
    # Enforce monotonicity across the filled bins (nearest-centre mapping can
    # only break it at dropped bins) and normalise to mean 1.0 on fit.
    full = isotonic_increasing(full, np.array([(idx == k).sum() for k in range(n_bins)], dtype=float))
    mean_fit = float(np.mean(full[idx]))
    if not np.isfinite(mean_fit) or mean_fit <= 0:
        return _flat_model()
    values = np.clip(full / mean_fit, F_LO, F_HI)
    return {"edges": np.asarray(edges, dtype=float), "values": np.asarray(values, dtype=float)}


def apply_anomaly_multiplier(p, model):
    """Per-row multiplier for anomaly_p under a fitted model (flat-safe)."""
    p = np.asarray(p, dtype=float)
    edges = np.asarray(model.get("edges", []), dtype=float)
    values = np.asarray(model.get("values", [1.0]), dtype=float)
    if edges.size == 0 or values.size != edges.size + 1:
        return np.ones_like(p)
    out = values[np.searchsorted(edges, p, side="right")]
    return np.clip(out, F_LO, F_HI)


def anomaly_decile_error(abs_r, scale, p, q, edges, n_bins=N_DECILES):
    """Level-matched per-anomaly-decile coverage error, in pp.

    Coverage is evaluated at the arm's OWN matched q80 (`covered =
    |r| <= q * scale`), so marginal level differences cannot score as a
    conditional fix; rows are binned by the FIT decile edges so every arm is
    scored on identical bins. Returns NaN when nothing is scoreable, so a
    degenerate fold drops out of the paired mean instead of voting zero.
    """
    abs_r = np.asarray(abs_r, dtype=float)
    scale = np.asarray(scale, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.asarray(edges, dtype=float)
    ok = np.isfinite(abs_r) & np.isfinite(scale) & (scale > 0) & np.isfinite(p) & np.isfinite(q)
    if not ok.any() or edges.size != n_bins - 1:
        return float("nan")
    covered = abs_r[ok] <= float(q) * scale[ok]
    idx = np.searchsorted(edges, p[ok], side="right")
    per = np.array([covered[idx == k].mean() if (idx == k).any() else np.nan for k in range(n_bins)])
    if np.all(np.isnan(per)):
        return float("nan")
    return float(np.nanmean(np.abs(per - TARGET_COVERAGE))) * 100.0


def _scoring_edges(p_val, fit_edges, n_bins=N_DECILES):
    """Bin edges shared by every arm's decile error on this fold.

    The fit edges when they are usable (identical bins to the multiplier fit);
    otherwise deciles of the eval anomaly_p; NaN edges when neither exists, in
    which case the fold votes on width only.
    """
    fit_edges = np.asarray(fit_edges, dtype=float)
    if fit_edges.size == n_bins - 1 and np.all(np.isfinite(fit_edges)) and np.all(np.diff(fit_edges) > 0):
        return fit_edges
    p = np.asarray(p_val, dtype=float)
    p = p[np.isfinite(p)]
    if p.size < MIN_EVAL_ROWS or np.min(p) == np.max(p):
        return np.full(n_bins - 1, np.nan)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        return np.full(n_bins - 1, np.nan)
    return edges


def _lookup_scale(forecaster, horizon, cfg, ids, prices):
    """Eval-split climatology scale through the production lookup.

    Sets, reads, and restores the horizon entry so a mid-fold failure cannot
    leak one fold's table into the next.
    """
    prev = forecaster.climatology_scale.get(horizon)
    forecaster.climatology_scale[horizon] = dict(cfg)
    try:
        return forecaster._climatology_lookup(horizon, ids, prices)
    finally:
        if prev is None:
            forecaster.climatology_scale.pop(horizon, None)
        else:
            forecaster.climatology_scale[horizon] = prev


def run(df, pruned, horizon_filter=None, n_jobs=None, max_folds=None):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    eval_items, train_items, trained_eval = assign_items(df)

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    if not forecaster.anomaly_gbm_enabled():
        raise SystemExit("ANOMALY_GBM did not take effect — no labels to score.")
    try:
        horizons = [h for h in ItemForecaster.HORIZONS if horizon_filter is None or h == horizon_filter]
        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Anomaly modulator {horizon}d\n  {'=' * 60}")
            tdf = forecaster.prepare_targets(df, horizon)
            tcol = f"target_return_{horizon}d"
            acol = f"target_anomaly_{horizon}d"
            if tcol not in tdf.columns or acol not in tdf.columns:
                logger.warning(f"    {tcol} or {acol} absent — skipping")
                continue
            # Intersection: the climatology fit, the anomaly head, the
            # multiplier fit and the scoring must all see identical rows, or
            # the paired read across arms does not hold.
            tdf = tdf.dropna(subset=[tcol, acol]).sort_values(["item_id", "date"])
            if tdf.empty:
                logger.warning(f"    No valid rows for {horizon}d")
                continue
            base_cols = [c for c in pruned if c in tdf.columns]
            if not base_cols:
                logger.warning("    no allowlisted feature columns — skipping")
                continue
            sub = tdf[["item_id", "date", "price", tcol, acol] + base_cols].copy()

            dates = sorted(sub["date"].unique())
            split_idx = len(dates) * 2 // 3
            sub_days = pd.to_datetime(sub["date"]).to_numpy()
            dates_dt = pd.to_datetime(pd.Series(dates)).to_numpy()
            is_heldout = sub["item_id"].isin(set(eval_items)).to_numpy()
            is_trained_eval = sub["item_id"].isin(set(trained_eval)).to_numpy()
            is_train_item = sub["item_id"].isin(set(train_items)).to_numpy()

            fold_list = list(range(split_idx + 1, len(dates), STEP_DAYS))
            if max_folds is not None:
                fold_list = fold_list[-max_folds:]

            per_fold = {"control": [], "anomaly_mod": [], "shuffled_mod": []}
            for fold_idx, window_end in enumerate(fold_list):
                val_dates = dates[window_end : window_end + VAL_WINDOW_DAYS]
                if len(val_dates) < 7:
                    continue
                in_fit = sub_days <= dates_dt[window_end - 1]
                in_val = (sub_days >= dates_dt[window_end]) & (sub_days <= dates_dt[window_end + len(val_dates) - 1])
                fit_df = ItemForecaster._purge_overlapping_train_rows(
                    sub[in_fit & is_train_item], val_dates[0], horizon
                )
                val_df = sub[in_val & (is_heldout | is_trained_eval)]
                val_df = val_df[val_df["price"] >= MIN_SERVED_PRICE_USD]
                if len(val_df) < MIN_EVAL_ROWS or fit_df.empty:
                    continue
                fit_sample = _stratified_sample(fit_df, train_items, ROW_BUDGET, fold_idx)

                med = fit_sample[base_cols].median()
                X_train = fit_sample[base_cols].fillna(med)
                head = forecaster._fit_anomaly_classifier(
                    X_train,
                    fit_sample[acol].to_numpy(),
                    "gbdt",
                    dict(TREE_PARAMS, n_jobs=n_jobs),
                    horizon=horizon,
                    tier_train=None,
                    num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
                )
                if head is None:
                    continue
                p_fit = np.clip(np.asarray(head.predict(fit_df[base_cols].fillna(med)), dtype=float), 1e-6, 1.0 - 1e-6)
                p_val = np.clip(np.asarray(head.predict(val_df[base_cols].fillna(med)), dtype=float), 1e-6, 1.0 - 1e-6)

                fit_min = fit_df[["item_id", "price", tcol]]
                table, pool, g = ItemForecaster._build_climatology_table(fit_min, tcol)
                if not table:
                    continue
                cfg = {"table": table, "tier_pool": pool, "global": g}
                ctl_fit = _lookup_scale(
                    forecaster, horizon, cfg, fit_df["item_id"].to_numpy(), fit_df["price"].to_numpy(dtype=float)
                )
                ctl_val = _lookup_scale(
                    forecaster, horizon, cfg, val_df["item_id"].to_numpy(), val_df["price"].to_numpy(dtype=float)
                )

                abs_r_fit = fit_df[tcol].abs().to_numpy(dtype=float)
                served_fit = fit_df["price"].to_numpy(dtype=float) >= MIN_SERVED_PRICE_USD
                fmodel = fit_anomaly_multiplier(p_fit[served_fit], abs_r_fit[served_fit], ctl_fit[served_fit])
                rng = np.random.default_rng(SHUFFLE_SEED + fold_idx)
                fshuf = fit_anomaly_multiplier(
                    rng.permutation(p_fit[served_fit]), abs_r_fit[served_fit], ctl_fit[served_fit]
                )

                abs_r_val = val_df[tcol].abs().to_numpy(dtype=float)
                scales = {
                    "control": ctl_val,
                    "anomaly_mod": ctl_val * apply_anomaly_multiplier(p_val, fmodel),
                    "shuffled_mod": ctl_val * apply_anomaly_multiplier(p_val, fshuf),
                }
                edges = _scoring_edges(p_val, fmodel["edges"])
                auc, _, _ = _score(val_df[acol].to_numpy(dtype=float), p_val)

                row = {
                    "fold": fold_idx,
                    "val_start": str(val_dates[0]),
                    "n_fit": len(fit_df),
                    "n_eval": len(val_df),
                    "head_auc_val": auc,
                    "mean_f_val": float(np.mean(apply_anomaly_multiplier(p_val, fmodel))),
                    "f_values": [float(v) for v in np.asarray(fmodel["values"]).ravel()],
                    "f_flat": bool(np.asarray(fmodel["edges"]).size == 0),
                }
                for arm, scale in scales.items():
                    q, w = matched_width(abs_r_val, scale)
                    per_fold[arm].append(
                        dict(
                            row,
                            width=(None if not np.isfinite(w) else float(w)),
                            log_width=(None if not np.isfinite(w) or w <= 0 else float(np.log(w))),
                            decile_err=(
                                None
                                if q is None or not np.isfinite(q)
                                else float(anomaly_decile_error(abs_r_val, scale, p_val, q, edges))
                            ),
                        )
                    )

            if not [r for r in per_fold["control"] if r["width"] is not None]:
                logger.warning(f"    no usable folds at {horizon}d")
                continue
            results[horizon] = {arm: {"per_fold": rows} for arm, rows in per_fold.items()}
            results[horizon]["_paired"] = {
                f"{arm}_{metric}": paired_fold_deltas(per_fold["control"], per_fold[arm], metric)
                for arm in ("anomaly_mod", "shuffled_mod")
                for metric in ("log_width", "decile_err")
            }
            for arm in ("control", "anomaly_mod", "shuffled_mod"):
                ws = [r["width"] for r in per_fold[arm] if r["width"] is not None]
                es = [r["decile_err"] for r in per_fold[arm] if r["decile_err"] is not None]
                aucs = [r["head_auc_val"] for r in per_fold[arm] if r["head_auc_val"] is not None]
                if ws:
                    logger.info(
                        f"      {arm:12s} mean matched width={np.mean(ws):.3f}% "
                        f"mean decile_err={np.mean(es):.2f}pp "
                        f"({len(ws)} folds)"
                    )
                if aucs and arm == "control":
                    logger.info(f"      anomaly head mean val AUC={np.mean(aucs):.4f}")
        return results
    finally:
        db.close()


def print_summary(results):
    print("\n" + "=" * 78)
    print("ANOMALY_P BAND MODULATOR vs FLAT CLIMATOLOGY — paired per-fold deltas")
    print("log_width: negative = NARROWER at matched 80% (good)")
    print("decile_err: negative = more EQUAL coverage across anomaly_p (good)")
    print("=" * 78)
    for horizon, entry in sorted(results.items()):
        n_ctl = sum(1 for r in entry["control"]["per_fold"] if r["width"] is not None)
        print(f"\nh={horizon}d   control folds={n_ctl}")
        for arm in ("anomaly_mod", "shuffled_mod"):
            rows = entry[arm]["per_fold"]
            base_w = {r["fold"]: r["width"] for r in entry["control"]["per_fold"]}
            base_e = {r["fold"]: r["decile_err"] for r in entry["control"]["per_fold"]}
            w_ratios = [
                r["width"] / base_w[r["fold"]]
                for r in rows
                if r["width"] is not None and base_w.get(r["fold"]) is not None and base_w[r["fold"]] > 0
            ]
            e_ratios = [
                r["decile_err"] / base_e[r["fold"]]
                for r in rows
                if r["decile_err"] is not None and base_e.get(r["fold"]) is not None and base_e[r["fold"]] > 0
            ]
            winfo = (
                f"width ratio {np.mean(w_ratios):.4f}, narrower {sum(x < 1.0 for x in w_ratios)}/{len(w_ratios)}"
                if w_ratios
                else "no paired folds"
            )
            einfo = (
                f"decile ratio {np.mean(e_ratios):.4f}, more equal {sum(x < 1.0 for x in e_ratios)}/{len(e_ratios)}"
                if e_ratios
                else "no paired folds"
            )
            for metric, info in (("log_width", winfo), ("decile_err", einfo)):
                d = entry["_paired"][f"{arm}_{metric}"]
                if d is None:
                    print(f"  {arm:12s} {metric:10s} — too few paired folds; {info}")
                    continue
                flag = "*" if d["excludes_zero"] else " "
                better = d["n_folds"] - d["wins"]  # lower is better, both metrics
                print(
                    f"  {arm:12s} {metric:10s} d={d['mean']:+.5f} "
                    f"[{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]{flag} "
                    f"better {better}/{d['n_folds']}; {info}"
                )
    print(
        "\n* = 95% interval excludes zero. The modulator earns a production "
        "follow-up\n  only with a negative decile_err interval AND no "
        "significant widening."
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--metadata-parquet", default=None)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="score only the K most recent folds (quick read; the verdict needs the full run)",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    df, pruned, _ = build_frame(args.metadata_parquet, cache_path=args.frame_cache)
    results = run(df, pruned, horizon_filter=args.horizon, n_jobs=args.n_jobs, max_folds=args.max_folds)
    print_summary(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
