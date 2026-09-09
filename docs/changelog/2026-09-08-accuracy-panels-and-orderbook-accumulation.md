# Accuracy panels + free-source accumulation (2026-09-08 note, build-out)

Implements the collection layer of `docs/research/2026-09-08-next-accuracy-indicators.md`
using FREE sources only — no paid cs2.sh feed (revised 2026-09-08: the initial
cs2.sh collectors were removed the same day).
No modelling or serving change: every new feature path is default-off and every
evaluation is gated on the required protocol (date split, H+13 embargo,
shuffled-feature placebo, served confirmation).

## What shipped (accumulation + research panels only)

- `ORDERBOOK_FEATURES=1` isolated forecaster group (`ob_*`, own allowlist
  gate like `tier_lead`/`supply_churn`, train-side A/B only, missing-as-NaN).
  Default off. These are ASK-SIDE proxies from the free supply panel's
  lis-skins ladder (slope, depth concentration, inflow, age, churn velocity,
  turnover) — the "collected but not yet evaluated" fields the doc names.
  Bid-side quantities (spread, imbalance, price impact) have no free source
  and are not represented; a null here does not test those.
- `SKINPORT_VOLUME=1` arm in `scripts/ab_test_volume_features.py`
  (single-series repair off the free `volume-YYYY-MM.parquet` Skinport
  `sales_24h`, era bound + live-day upper bound + 80% join-coverage gate,
  mirroring `IFLOW_VOLUME`). Smoke-tested 2026-09-08 on 20 items: era filter,
  100% join, frame build with 10 volume features surviving prune, and a clean
  void ("No valid targets for 7d") on the 17-day panel — the H+13 embargo
  leaves no resolvable labels yet, so the full rerun waits on accumulation.
  Live collection from this machine is 403-blocked (Cloudflare egress, known);
  CI accumulates daily.
- `scripts/build_case_panel.py` → `case-panel.parquet` (visible supply,
  date-anchored 7/30d depletion, drop-pool status; EV/ROI/openings NaN until
  feeds exist). `scripts/build_sticker_panel.py` → `sticker-panel.parquet`
  (visible supply, substitute_group, capsule status; application/craft
  velocity NaN until CSFloat queries are wired).
- `backend/collectors/reddit_events.py` + `scripts/run_reddit_events.py`:
  structured event detector (narrative tags, first-seen timestamps,
  engagement/author-quality/novelty, conventional + reversed sentiment).
  Skips loudly without Reddit creds. Generic sentiment stays off.
- `scripts/check_sidecar_continuity.py` (+ `--gate`): days/gaps/longest-run/
  per-feed-latest for the free accumulation tables (supply, volume, reddit-events).
- `ROW_COUNT_FIELDS`: `reddit_event_rows`, `case_panel_rows`, `sticker_panel_rows`.

## Measured state (2026-09-08)

- Data repo: supply 19 consecutive days (08-07→08-25, no gaps, all 5 feeds);
  Skinport sales volume 17 days with 1 gap (longest run 15). Working copy is
  behind (supply 1 day, no volume) — use the data repo / prod Postgres for figures.
- The ask-side proxy group reads the same supply panel, so its continuity gate
  is the supply table's lis-skins coverage — no new collector needed. The
  Skinport-volume rerun needs more `volume-*.parquet` days first; then
  `compute_mde.py` first (stop if MDE > ~2pp).
- Centre re-check (`centre_vs_lastprice.py --gate`, prod Postgres): panel still
  immature — 17/20/13/2 clean dates at 3/7/14/30d vs MIN 20. Signs informative
  only: GBM centre skill −0.059/−0.063/−0.078 at 3/7/14d, coverage 0.7–1.5pp
  worse than last-price at identical width. No simplification yet.

## Tests

New: `test_reddit_events`,
`test_sidecar_continuity`, `test_case_sticker_panels`, `test_orderbook_features`,
`test_skinport_ab_arm` — all pass (see verification below). Existing: supply/sales/centre (44
passed), skip/shelved groups (12 passed), churn/scale/volume-flag suites pass
except 4 `test_null_volume_features` failures that reproduce on the clean tree
(pre-existing, unrelated). One `test_forecaster.py` trend test also fails on the
clean tree.
