# Accuracy Work Closed; Supply-Depth Gate Found Permanent (2026-07-31)

## Decision

**The prediction-accuracy line of work is closed.** Not paused, not blocked on a
harness repair — finished. `docs/research/accuracy-opportunities.md` now carries a
stop banner and its "Remaining" list is empty. The two items formerly listed there
(multi-horizon joint training, ensemble expansion) are **abandoned as unmeasurable**.

## Why

Three facts that only became decisive when read together:

**1. Every measured effect is ~0pp.** Six consecutive feature groups:

| Group | Measured |
|---|---|
| Supply-side bundle (rarity + weapon_type) | +0.66pp |
| Player counts | 0pp |
| Event decay optimization | 0pp |
| Multi-source outlier voting | 0pp (train) |
| Quality spread / cross-wear | +0.16pp |
| Price technical primitives | −1.10pp |

**2. The harness floor is 1.15pp (3d) / 2.76–7.13pp (7d/14d/30d).** Established
2026-07-31, see `2026-07-31-price-primitives-decision-scale.md`.

**3. The gap does not close.** Resolving ~0.3pp effects needs a ~9x MDE reduction
= **~85x more folds**. Fold count is capped near **73** by disjoint-window
independence (`step=21`); it does not scale with `--max-items`. Unreachable.

The prior standing advice — "fix the measurement before spending compute" — was
wrong, and this entry retracts it. The measurement cannot be fixed to the precision
the remaining ideas require. A model whose last six honest experiments all returned
zero, evaluated by a design that cannot see anything smaller than 1pp, chasing
items estimated below 1pp, is at its practical ceiling given its inputs.

### A measurement idea considered and dropped

Whether paired per-fold sd shrinks with item count (which would lower MDE without
adding folds) was scoped and partly staged: the 800-item frame builds in **124.5s**
at 3.29M rows, **11.6 GB peak RSS / 22.0 GB peak footprint on a 24 GB machine** —
so the two 800-item arms cannot run concurrently. Abandoned before running the
fits: the best plausible outcome is sd halving, a **2x** MDE gain against a **9x**
requirement. Recorded so nobody re-scopes it. (The correlation prune also keeps a
slightly different feature set per item count — 137 features at 800 vs 134 at 200 —
so cross-item-count sd values would not have been strictly comparable anyway.)

## Finding: the supply-depth 30-day gate is permanent

Supply depth was dropped 2026-07-16 because `supply_change_7d` /
`supply_listings_zscore` need 30+ days of `supply_snapshots` history. The natural
question 3.5 months on was whether daily scraping had since supplied it.

**It had not — zero days accumulated in 16 days of green CI.**

| Source | Coverage |
|---|---|
| Prod Supabase `supply_snapshots` | 1 day — 2026-07-15, 35,037 rows |
| `price-archive/ops/supply_snapshots.parquet` | 1 day — 2026-07-15, 35,037 rows |

### Root cause — a green job that stores nothing

From run `30589441873` (2026-07-30):

```
Tracking 5,542 items for supply snapshots
429 at offset=0, backing off 30s
429 at offset=0, backing off 60s
429 at offset=0, backing off 120s
ERROR - Could not get total item count from Steam. Aborting.
INFO -   Stored 0 Steam snapshots
INFO - Supply scrape complete in 213.4s
```

Steam 429s the **very first** request from GitHub-hosted runner IPs. The scraper
backs off, aborts, stores 0 rows — and **exits 0**. So `supply-scraper.yml` reports
success daily, files no failure issue, and has silently stored nothing since roughly
2026-07-16. The 2026-07-15 day predates the hosted schedule. The 4m30s runtime
against the ~115 min a genuine catalog walk needs was the visible tell all along.

### Why this makes the drop permanent

The free path does not merely run slowly (the 2026-07-16 cost objection) — it
**cannot run from CI at all**, because Steam blocks datacenter IPs. Waiting
accumulates nothing.

Even a *fixed* scraper would not revive the feature: it grows history only forward,
and a feature present for the trailing 30 days is untrainable over a 1460-day window
and unmeasurable in a 26-fold walk-forward A/B, where it would be null for ~98% of
rows. **Only a paid historical backfill (CS2Cap candles `q`, $19/mo) could revive
it** — still declined. The drop is now reaffirmed on **data** grounds, which do not
expire, rather than cost grounds, which could have.

## Ops: `supply-scraper.yml` deleted

Confirmed nothing consumes supply data — no frontend reference, no API endpoint, and
**zero supply-derived columns survive into the engineered training frame** (verified
against the 800-item frame; the correlation prune drops them because one day of data
is constant). The only reader was the forecaster, which discards them.

So the workflow was **deleted**, not repaired: making a job nothing consumes fail
loudly just converts a silent no-op into a daily alert. `supply_scraper.py`, the
`supply_snapshots` table, and migration `0016` are **kept**, so a future paid-backfill
revival does not start from scratch.

A liquidity/"how many for sale" UI number, if ever wanted, needs only *current* data
— a live query at page load, not a daily archive. That hypothetical does not justify
the collection either.

## Collector audit — the silent-success shape is not unique

Audited all remaining scheduled collectors by checking **whether their output table is
actually fresh**, not whether CI is green. Two more broken jobs, one of them serious:

| Workflow | CI status | Output | Verdict |
|---|---|---|---|
| `aggregator-update` | green daily | ~358K rows/day, 41K items | ✅ healthy |
| `backtest-accuracy` | green daily | `prediction_accuracy` fresh today | ✅ healthy |
| `reddit-sentiment` | **green 3x/day** | `social_mentions` = **0 rows, ever** | 🛑 silent failure |
| `price-forecast` | **failing** | last CI success **2026-07-14** | 🛑 **34 consecutive failures** |
| `event-correlation-analysis` | green weekly | `event_correlations` stale since 2026-07-12 | ⚠️ suspect |
| `discover-new-items` | last run 2026-07-05 (failed) | no runs in 26 days | ⚠️ not firing |

### `reddit-sentiment` — same shape as supply

All three subreddits return **403 Blocked** (`old.reddit.com`, datacenter IP), and the
task reports `RESULT: {'status': 'success', 'total_mentions': 0, 'inserted': 0}`.
`social_mentions` has **never held a single row** — it doesn't even exist in the local
SQLite DB, which is why `forecaster` logs `Failed to load social mentions: no such
table` on every local build. Runs 3x/day for nothing.

### `price-forecast` — the production incident

**Last successful CI run: 2026-07-14. Every run since has failed** (34 of them), with:

```
forecast_prices - ERROR - No models available for prediction.
RESULT: {'status': 'error', 'message': 'No trained models'}
```

**Root cause:** the workflow persists trained models by committing them —
`git add backend/models/saved_models/` (`price-forecast.yml:103-109`) — but the model
files are **gitignored** (`.gitignore:68-71`, `*.txt` / `*.pkl`). So the add is a
no-op; only `meta.json` is tracked. Monday's full-retrain run predicts successfully
in-process because the boosters exist in its own workspace, then they vanish. Every
predict-only run afterwards checks out `meta.json` with no boosters and aborts. The
mechanism is certain; which commit introduced the ignore rule is less so (`345d26d`,
2026-07-20, touches it, but the last success predates that).

**Not caused by `47d3195`** ("ignore saved_models rollback snapshots", 2026-07-31) —
that is 17 days after the breakage began.

**Impact:** prod forecasts were last written **2026-07-29** (34,764 rows), and that
was a *manual local run*, not CI — as were 07-19, 07-18, 07-17. The automated forecast
pipeline has been dead for 17 days and the app has been served by hand since.

**Not fixed here.** Committing 100MB+ boosters to git is what caused the unpushable-main
incident in `47d3195`, so the fix is a real design choice — GH Actions cache, artifact
retention, or object storage — not a one-line patch.

## Files changed

- `docs/research/accuracy-opportunities.md` — stop banner; Remaining emptied and
  marked abandoned-unmeasurable; supply-depth re-check + root cause under Dropped.
- `docs/changelog/2026-07-31-accuracy-work-closed.md` — this entry.

No code changed. No model or serving behavior changed.

## Related

- `docs/changelog/2026-07-31-price-primitives-decision-scale.md` — the measurement floor
- `docs/changelog/2026-07-16-drop-supply-depth.md` — the original drop decision
- `docs/research/accuracy-opportunities.md` — Reality Check, Measurement Floor
