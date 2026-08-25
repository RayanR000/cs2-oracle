#!/usr/bin/env bash
#
# Local stand-in for the "Price Forecast" -> "Backtest Accuracy" chain
# (.github/workflows/price-forecast.yml, backtest-accuracy.yml). Companion to
# run_daily_local.sh, which covers the aggregator leg of the same chain.
#
# WHY THIS EXISTS. While Actions minutes are out, the forecast chain does not
# run, so item_forecasts and forecast_outcomes stop advancing — the archive
# froze on 2026-08-20 / 2026-08-17. Everything gated on panel MATURITY stalls
# with it, including the centre-vs-last-price verdict, which needs 20 forecast
# dates per horizon and has 8-11
# (docs/research/2026-08-25-gbm-retirement-decision.md). Waiting does not fix
# that; only running the chain does.
#
# WRITES PRODUCTION. Forecasts go to prod Supabase (backend/.env) and the
# refreshed archive is force-pushed to RayanR000/cs2-oracle-data, exactly as CI
# does. Everything below the env block is a transcription of the workflow, and
# the env block is the part that must not drift: a wrong flag does not fail, it
# silently serves a different model and pollutes the panel it is meant to grow.
#
# Usage:
#   ./run_forecast_local.sh                 # predict-only (the daily mode)
#   ./run_forecast_local.sh --full          # retrain + predict (the Monday mode)
#   ./run_forecast_local.sh --train-only    # retrain, no forecasts, no publish
#   ./run_forecast_local.sh --dry-run       # run against the LOCAL db, no push
#   ./run_forecast_local.sh --no-backtest   # forecasts only, skip resolution
#
# HAZARD worth knowing before running anything else against a symlinked archive:
# a predict run WRITES item_forecasts.parquet into whatever price-archive points
# at, including under ENVIRONMENT=development. This script clones fresh so that
# is contained; an ad-hoc local run pointed at a canonical checkout will dirty
# it, and only `git checkout --` puts it back.
#
set -euo pipefail

MODE="predict-only"
DRY_RUN=0
RUN_BACKTEST=1
while [ $# -gt 0 ]; do
  case "$1" in
    --predict-only) MODE="predict-only" ;;
    --train-only)   MODE="train-only" ;;
    --full)         MODE="full" ;;
    --dry-run)      DRY_RUN=1 ;;
    --serve-regime) SERVE_GLOBAL=0 ;;
    --no-backtest)  RUN_BACKTEST=0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$REPO_ROOT/backend"
PY="$BACKEND/venv/bin/python"
DATA_REPO="https://github.com/RayanR000/cs2-oracle-data.git"
ARCHIVE_LINK="$REPO_ROOT/price-archive"
MODEL_DIR="$BACKEND/models/saved_models"
REGIME_STASH="$MODEL_DIR/.regime-stashed"
STASHED=""
ARCHIVE_DIR=""
SERVE_GLOBAL=1

[ -x "$PY" ] || { echo "ERROR: venv python not found at $PY"; exit 1; }

# `db.archive.ARCHIVE_ROOT` is hardcoded to <repo>/price-archive with no env
# override, so CI points it at the fresh checkout with a symlink and so does
# this. The local price-archive/ is a real (gitignored) directory that runs
# BEHIND remote; publishing from it would force-push stale history over the
# newer side, the same trap run_daily_local.sh documents. It is moved aside for
# the run and restored on exit, so the workspace is left as it was found.
cleanup() {
  if [ -L "$ARCHIVE_LINK" ]; then rm -f "$ARCHIVE_LINK"; fi
  if [ -n "$STASHED" ] && [ -d "$STASHED" ]; then mv "$STASHED" "$ARCHIVE_LINK"; fi
  if [ -n "$ARCHIVE_DIR" ] && [ -d "$ARCHIVE_DIR" ]; then rm -rf "$ARCHIVE_DIR"; fi
  if [ -d "$REGIME_STASH" ]; then
    mv "$REGIME_STASH"/*.txt "$MODEL_DIR"/ 2>/dev/null || true
    rmdir "$REGIME_STASH" 2>/dev/null || true
  fi
  return 0
}
trap cleanup EXIT

echo "== [1/6] Clone data archive fresh =="
ARCHIVE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cs2-forecast.XXXXXX")"
git clone --depth 1 "$DATA_REPO" "$ARCHIVE_DIR"

echo "== [2/6] Point price-archive at the fresh clone =="
if [ -e "$ARCHIVE_LINK" ] && [ ! -L "$ARCHIVE_LINK" ]; then
  STASHED="$REPO_ROOT/price-archive.stashed.$$"
  mv "$ARCHIVE_LINK" "$STASHED"
  echo "stashed the stale local price-archive/ at $(basename "$STASHED")"
fi
rm -f "$ARCHIVE_LINK"
ln -s "$ARCHIVE_DIR/price-archive" "$ARCHIVE_LINK"

if [ "$MODE" = "predict-only" ]; then
  # Same unmatched-glob guard as the regime count below: an EMPTY model dir is
  # the exact case this check exists to report, so it must not die on it.
  n=$({ ls "$MODEL_DIR"/*.txt 2>/dev/null || true; } | wc -l | tr -d ' ')
  echo "Found $n booster file(s)."
  if [ "$n" -eq 0 ]; then
    echo "ERROR: no boosters in backend/models/saved_models. Re-run with --full."
    exit 1
  fi
fi

# SKIP_REGIMES only governs TRAINING. At predict time `predict()` prefers a
# regime specialist whenever `_detect_current_regime` matches, and the LOCAL
# artifact still carries lgb_*_range_* boosters from a pre-SKIP_REGIMES retrain
# — so an unguarded local run serves `model_config=regime` on 100% of the
# cohort while CI's regime-free artifact last served `global-only` (archive,
# 2026-08-20). That is a different served mid, and the deep review's item 7
# flags those specialists as plausibly DEGRADING it. `served_identity` merges
# the two labels into one cohort, so the drift would not even be visible as a
# separate panel — it would just quietly change the thing being measured.
#
# So the regime boosters are moved aside for the run and restored on exit,
# which reproduces CI's artifact rather than this laptop's. `--serve-regime`
# opts back in for a deliberate comparison.
#
# predict-only ONLY. A training mode under SKIP_REGIMES=1 trains no regime
# models and its save path already DELETES the orphaned files on disk
# (forecaster.py:10054) — stashing them there and restoring on exit would
# resurrect artifacts the retrain had just retired.
if [ "$SERVE_GLOBAL" -eq 1 ] && [ "$MODE" = "predict-only" ]; then
  # `|| true`: an unmatched glob reaches ls literally and ls exits non-zero,
  # which under `set -o pipefail` would abort the run for the ordinary case of
  # an artifact holding only SOME of the three regimes (this one has `range`).
  n_regime=$({ ls "$MODEL_DIR"/lgb_*_range_*.txt "$MODEL_DIR"/lgb_*_bull_*.txt \
                  "$MODEL_DIR"/lgb_*_bear_*.txt 2>/dev/null || true; } | wc -l | tr -d ' ')
  if [ "$n_regime" -gt 0 ]; then
    mkdir -p "$REGIME_STASH"
    mv "$MODEL_DIR"/lgb_*_range_*.txt "$MODEL_DIR"/lgb_*_bull_*.txt \
       "$MODEL_DIR"/lgb_*_bear_*.txt "$REGIME_STASH"/ 2>/dev/null || true
    echo "stashed $n_regime regime booster(s) to serve global-only, as CI does"
  fi
fi

cd "$BACKEND"

# ---------------------------------------------------------------------------
# The env block, transcribed from price-forecast.yml's "Run ML price
# forecasting" step. Read that step's comments before changing any of these;
# each one is a serving decision, not a cost knob.
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: local DB, no push. The local DB is a synthetic fixture, so"
  echo "         nothing it produces is a real forecast — this checks wiring."
  export ENVIRONMENT=development
else
  export ENVIRONMENT=production
fi
export DEBUG=false
# SKIP_CV deliberately unset: it biases q_hat low and the served band under-covers.
export CV_DIAGNOSTIC_CLASSIFIER=0
# Mirrors CI rather than the local default. The ~2GB feature cache is output-
# neutral when it hits, but a run whose whole purpose is to extend the served
# panel should not be the first place a stale cache is discovered.
export ENGINEERED_CACHE=0
export SKIP_REGIMES=1
export EXCEEDANCE_HEAD=1
export FEATURE_NATIVE_NAN=1
# Persist the serve-time quote (raw + smoothed, per item per anchor) into
# price-archive/ops/anchor_audit/, which the publish step below carries. It is
# the only record of what was actually quoted: the served current_price cannot
# be reconstructed afterwards, and that unreconstructable wedge is what drives
# published dollar coverage to 54-72% against a band calibrated to 80%.
export ANCHOR_AUDIT=1
# REQUIRED for `full`: the 14d model-age gate would otherwise read the restored
# artifact as fresh and silently degrade the retrain into a predict-only run.
if [ "$MODE" = "full" ]; then export FORCE_RETRAIN=1; else export FORCE_RETRAIN=0; fi

echo "== [3/6] Run ML price forecasting (mode: $MODE) =="
set -o pipefail
case "$MODE" in
  predict-only) "$PY" scripts/forecast_prices.py --predict-only 2>&1 | tee forecast.log ;;
  train-only)   "$PY" scripts/forecast_prices.py --train-only   2>&1 | tee forecast.log ;;
  full)         "$PY" scripts/forecast_prices.py                2>&1 | tee forecast.log ;;
esac

if [ "$RUN_BACKTEST" -eq 1 ] && [ "$MODE" != "train-only" ]; then
  echo "== [4/6] Resolve matured forecasts (backtest) =="
  "$PY" scripts/backtest_accuracy.py --type forecast 2>&1 | tee backtest.log
  # The gate ships on a different branch (docs/gbm-retirement-decision) until
  # that lands, and the first live run hit exactly that: an interpreter error
  # where the panel counts should have been. Absent is not a failure — it is
  # the expected state on this branch — so say so and carry on.
  if [ -f scripts/centre_vs_lastprice.py ]; then
    echo "-- centre maturity gate (informational; exit 1 = still immature) --"
    "$PY" scripts/centre_vs_lastprice.py --gate 2>&1 | tee centre_gate.log || true
  else
    echo "-- centre maturity gate SKIPPED: scripts/centre_vs_lastprice.py is"
    echo "   not on this branch (see docs/gbm-retirement-decision) --"
  fi
else
  echo "== [4/6] Backtest SKIPPED =="
fi

if [ "$MODE" = "train-only" ]; then
  echo "== [5/6] Publish SKIPPED (train-only writes no forecasts) =="
  echo "== [6/6] DONE =="
  exit 0
fi

echo "== [5/6] Publish archive (orphan commit, force-push main) =="
# ONE push for both legs. CI publishes twice because the forecast and backtest
# jobs are separate checkouts; here they share one, and the chain is sequential,
# so a single orphan commit carries item_forecasts AND forecast_outcomes.
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: skipping the publish. Would have published:"
  git -C "$ARCHIVE_DIR" status --porcelain | head
else
  cd "$ARCHIVE_DIR"
  git config user.name "github-actions[bot]"
  git config user.email "github-actions[bot]@users.noreply.github.com"
  git checkout --orphan flat
  git add -A
  git commit -m "archive: item_forecasts + outcomes $(date -u +%F)"
  git push --force origin HEAD:main
  cd "$BACKEND"
fi

echo "== [6/6] Verify forecasts were persisted =="
# Best-effort BY DESIGN. A manual run after UTC midnight false-fails this check
# — the forecasts still persisted and the publish above already happened. Read
# the message before believing a failure here.
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: freshness check skipped (local DB holds no served forecasts)."
elif ! "$PY" scripts/check_forecast_freshness.py; then
  echo "WARN: freshness check failed. If this run started after 00:00 UTC that"
  echo "      is the known false-fail, not a lost write — confirm against the"
  echo "      published archive before acting on it."
fi

echo "== DONE: $MODE run complete =="
