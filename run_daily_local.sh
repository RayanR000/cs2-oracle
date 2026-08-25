#!/usr/bin/env bash
#
# Local stand-in for the "Aggregator Market Update" GH Actions workflow
# (.github/workflows/aggregator-update.yml). Use when Actions minutes are
# exhausted: it runs the same daily pipeline on this machine and force-pushes
# the refreshed archive to RayanR000/cs2-oracle-data.
#
# Writes to PRODUCTION Supabase (backend/.env) and force-pushes the data repo's
# main branch (the intended once-per-day squash). Requires push access to
# RayanR000/cs2-oracle-data via your git/gh credentials.
#
# Does NOT run the forecast chain (Price Forecast -> Backtest) -- those are
# separate workflows and also need Actions minutes.
#
# --no-db: skip the DB migration + DB-coupled aggregate and produce the snapshot
#   straight from the csgotrader dumps (scripts/export_snapshot_dbfree.py). Use
#   when Supabase is unreachable. Drops the historical carry-forward and the
#   separate backfilled/OHLCV CSV (both need the items table); the raw dump is
#   the full universe, so daily coverage is unaffected.
#
# Usage:  ./run_daily_local.sh [--no-db]
#
set -euo pipefail

NO_DB=0
DATE_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --no-db) NO_DB=1 ;;
    --date) DATE_OVERRIDE="$2"; shift ;;
    --date=*) DATE_OVERRIDE="${1#*=}" ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$REPO_ROOT/backend"
PY="$BACKEND/venv/bin/python"
DATA_REPO="https://github.com/RayanR000/cs2-oracle-data.git"

export ENVIRONMENT=production
export DEBUG=false

[ -x "$PY" ] || { echo "ERROR: venv python not found at $PY"; exit 1; }
cd "$BACKEND"

echo "== [2/6] Resolve snapshot date =="
if [ -n "$DATE_OVERRIDE" ]; then
  SNAPSHOT_DATE="$DATE_OVERRIDE"
  echo "Using --date override: $SNAPSHOT_DATE (backdated; stamps today's dump under this day)"
else
  SNAPSHOT_DATE="$("$PY" -c 'from collectors.snapshot_date import resolve_snapshot_date; print(resolve_snapshot_date())')"
fi
export AGGREGATOR_SNAPSHOT_DATE="$SNAPSHOT_DATE"
echo "Snapshot day: $SNAPSHOT_DATE (runner clock: $(date -u +%FT%TZ))"

set -o pipefail
if [ "$NO_DB" -eq 1 ]; then
  echo "== [1/6] DB migration SKIPPED (--no-db) =="
  echo "== [3/6] Export snapshot from dumps (no Supabase) =="
  "$PY" scripts/export_snapshot_dbfree.py --date "$SNAPSHOT_DATE" 2>&1 | tee aggregator.log
else
  echo "== [1/6] DB migration =="
  "$PY" scripts/run_task.py migrate
  echo "== [3/6] Aggregate (writes prices to prod) =="
  "$PY" scripts/run_task.py aggregate 2>&1 | tee aggregator.log
fi

SNAP_CSV="/tmp/aggregator-snapshots-$SNAPSHOT_DATE.csv"
[ -f "$SNAP_CSV" ] || { echo "ERROR: aggregator did not produce $SNAP_CSV"; exit 1; }

echo "== [4/6] Clone data archive fresh =="
# Fresh clone every run: the local price-archive/ dir runs BEHIND remote, so
# reusing it would force-push stale history over the newer remote side.
ARCHIVE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cs2-archive-$SNAPSHOT_DATE.XXXXXX")"
git clone --depth 1 "$DATA_REPO" "$ARCHIVE_DIR"

echo "== [5/6] Append prices + supply/volume =="
"$PY" scripts/append_to_parquet.py \
  --date "$SNAPSHOT_DATE" \
  --out-dir "$ARCHIVE_DIR" \
  --snapshot-csv "$SNAP_CSV" \
  --backfilled-csv "/tmp/aggregator-backfilled-$SNAPSHOT_DATE.csv" \
  --exchange-rates-csv "/tmp/exchange-rates-$SNAPSHOT_DATE.csv"

# Best-effort, exactly like continue-on-error in CI: a single-venue outage
# (e.g. Skinport 403 from a laptop on WARP) must not abort the daily update.
echo "-- supply depth (best-effort) --"
"$PY" scripts/run_supply_depth.py --archive-dir "$ARCHIVE_DIR/price-archive" \
  || echo "WARN: supply depth failed, continuing"
echo "-- sales volume (best-effort) --"
"$PY" scripts/run_sales_volume.py --archive-dir "$ARCHIVE_DIR/price-archive" \
  || echo "WARN: sales volume failed, continuing"

echo "== [6/6] Publish archive (orphan commit, force-push main) =="
cd "$ARCHIVE_DIR"
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git checkout --orphan flat
git add -A
git commit -m "archive: prices + exchange-rates $SNAPSHOT_DATE"
git push --force origin HEAD:main

cd "$REPO_ROOT"
rm -rf "$ARCHIVE_DIR"
echo "== DONE: pushed $SNAPSHOT_DATE to cs2-oracle-data main =="
echo "NOTE: forecast chain (Price Forecast -> Backtest) still needs Actions minutes."
