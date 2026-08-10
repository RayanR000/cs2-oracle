# docs/

Refreshed 2026-08-10 against the code. Where a doc and the code disagree, the code wins —
report it rather than working around it.

## Architecture (`architecture/`)

- `model.md` — the forecaster as it stands: 4 q50 LightGBM models + 4 directional
  classifiers, the split-conformal band, sequential training, age-based retrain
- `model-optimization.md` — size/speed levers, split into already-applied, still-available,
  and 🛑 do-not
- `pipeline.md` — the aggregator and the workflows chained off it; what collects and what
  no longer does
- `data.md` — Parquet archive layout, the `ops/` mirror layer, Supabase serving tables

> **No production directional-accuracy figure is currently quotable.**
> `MIN_FORECAST_DATES = 20` (`backend/backtest/scoring.py`) and live cohorts span 1–2
> distinct forecast dates, so every horizon reports NO HEADLINE. This is a calendar
> problem, not a code problem. Offline CV DA and production DA are not comparable until
> the served series accumulates ~20 dates.

## Reference (`references/`)

- `steam-api.md` — Steam Market endpoints and response formats, empirically tested.
  The rate-limit envelope applies to **residential IPs only** — hosted CI runners are
  429'd on the first request.
- `data-sources.md` — per-source status, freshness, known issues
- `data-inventory.md` — the canonical coverage audit: what is actually on disk, how much of
  the market it covers, and where the history is thin. Companion to `data-sources.md`, which
  says where the data comes from rather than what arrived
- `catalog-build.md` — Steam catalog scrape: rate-limiting strategy, gap repair
- `backfill.md` — **Dead capability.** CSMarketAPI multi-market backfill; the free-key
  quota never resets and the local DB is empty. Kept for the key-rotation and
  priority-queue design only.

## Research (`research/`)

- `2026-08-09-model-and-data-research.md` — **Start here for anything accuracy-related.** The
  current review. ⚠️ **Read its corrections banner first** — every Track A recommendation in it
  shipped within hours, three of its cost predictions failed, and its §1c gate is refuted.
- `2026-08-07-cs2-forecasting-research.md` — ⚠️ **SUBSTANTIALLY STALE, banner at the top.** Still
  the best analysis and literature review in the repo, but six of its load-bearing numbers are
  refuted or category errors — including the `~0.3pp` date-level MDE, which is a *required* effect
  size marked `[MEASURED HERE]`. 1,706-line external design review, and its C1–C5 correction block
  overturns four of its own first-pass claims. Of those, the BUFF **bid** voting into the consensus
  as an ask is **fixed** (2026-08-07) and `walkforward_backtest.py --purge` being default OFF is
  **fixed** (2026-08-08; the flag is now `--no-purge`); the synthetic 1.1607 Steam fee constant is
  **still unfixed**, tracked as `5c`. Also carries the one measured positive — expensive tiers lead
  cheap tiers by a day, z = 9.1. See `changelog/2026-08-07-cs2-forecasting-research-review.md`.
- `2026-08-09-next-steps.md` — ⭐ **the live action list.** Tracks O/G/A/C/D. **Track A is closed
  (all six cost levers shipped 2026-08-09); D1 answered — the Steam listing page works.** Read its
  status block first; it says which of the other tracks are done. **The gate is lifted** — Tracks C
  and D are unblocked. O2 is the last pre-existing item still open — G3 landed in `873148b`,
  and O4 closed 2026-08-10.
- `2026-08-10-training-cost-levers.md` — **the current cost accounting**, measured against CI
  runs `31337078991` and `31356483719`; supersedes the cost tables in
  `changelog/2026-08-09-training-cost-levers.md`. Both measured runs are over the project's
  30-minute wall-clock cap. ⚠️ **Read its own corrections banner** — three of its claims were
  overturned when levers 1 and 4 landed the same day, including that lever 1 is free of
  served effects (it is not; the regime half moves the served mid).
- `2026-08-08-model-review.md` — the model review. ⚠️ **§5's composition rows are refuted in
  place** (2026-08-09); the fall attributed to composition control was a pre-2026-vs-2026 regime
  difference caused by a NULL-unsafe comparison.
- `2026-08-09-composition-stability.md` — the corrected measurement, **complete** (`f833882`).
  **Composition control does not move the reversal** (+0.1027 stable vs +0.1023 unconditional at
  3d; equal to four decimals at 7d), so the quoting-artifact gate on accuracy work is **lifted**.
  Note what is compared: the contrast is stable-vs-unconditional, because both the
  "changed (present)" and "stable & ≥3 sources" cells are 25 dates at 3d / 19 at 7d and carry no
  number. Both are calendar waits.
- `2026-08-07-next-steps.md` — **superseded for ordering** by the 2026-08-09 doc; the descriptions
  remain valid. Steps 1–7 are DONE. Steps 8–11 are NOT STARTED **except** step 10's rank-IC half
  and step 11's reversal measurement. **Blocker 5d is refuted** — it was a false positive.
  Read step 5's "not done" list before citing any A/B result: the harnesses were repaired but
  **none has been re-run**, so every stored A/B number predates the repair.
- `lis-skins-snapshot-plan.md` — ⚠️ **built, not proposed.** Shipped 2026-08-06 as
  `collectors/supply_depth.py` and runs daily; banner records which fields were dropped.
- `accuracy-opportunities.md` — closed 2026-07-31, **reopened 2026-08-07** by the review
  above, which relocates the binding constraint from input data to measurement. The stop
  banner is intact and the tables are still a record of what was tried, not a backlog. Read
  both before proposing accuracy work.
- `2026-07-19-feature-contribution-by-horizon.md` — ⚠️ **bannered.** The ablation behind
  `HORIZON_EXCLUDED_GROUPS` (now a **no-op** — the allowlist already removes those groups at every
  horizon) and the founding `+3.5pp` behind `FEATURE_GROUP_ALLOWLIST`, which has **never been
  re-derived**. Penny cohort, un-embargoed, scored against a 50% benchmark the project rejects.
- `2026-07-21-training-time-optimization.md` — 🛑 **RETIRED.** Its phase ranking is inverted
  (Optuna 59% / CV 2%; measured is CV 50% / Optuna 4%) and levers E, G and J are refuted in code.
  Use `architecture/model-optimization.md` → "Where the time goes now".
- `volume-data.md` — ⚠️ **bannered.** The conclusion (volume adds no predictive lift) stands; the
  |r| < 0.002 reasoning at `:27`/`:142` does not, the free archive source died 2026-04-15, and
  post-2026-03-22 "volume" is a listing count.
- `competitor-analysis.md` — landscape and differentiators; four inline caveats added 2026-08-09
- `2026-07-27-direction-label-sweep-raw.txt` — raw sweep output. ⚠️ **It is a crashed run** — dies
  on an SSL timeout partway through 14d; 30d never ran and no summary line was printed. Nothing
  cites it. Every treatment arm loses to control on the three completed horizons, which is
  consistent with the vol-scaled branch being dead code (`sigma=None` on both paths).

## Design docs and plans (`superpowers/`)

`specs/` holds designs (15), `plans/` the execution checklists (13). Each shipped change is
also recorded in `changelog/`, which is the durable record. Load-bearing ones:

- `specs/2026-07-25-monthly-parquet-partitioning-design.md` — the live partitioning scheme
  in `scripts/append_to_parquet.py`
- `specs/2026-08-01-deterministic-backtest-design.md` — shared-estimator backtest
- `specs/2026-08-03-served-forecast-surface-design.md` — confidence-gate removal, $1 floor
- `specs/2026-08-04-minimal-model-design.md` — the 40→8 model collapse
- `specs/2026-08-05-cv-cohort-parity-design.md` — CV/production cohort mismatch. Its
  residual-gap table rests on 1–2 market days; read it with the NO HEADLINE caveat above.

## Changelog (`changelog/`)

Append-only dated decision records: bug fixes, features, audits, and refuted experiments.
121 entries, 2026-07-08 to 2026-08-10. Entries are never edited to match later reality —
several describe code that has since been deleted, which is the point. Per `AGENTS.md`
workflow rule 2, non-trivial decisions get a new dated note here.

## Other

- `code-review-2026-07-21.md` — **Live punch list**, findings re-verified 2026-08-05.
  Most are still open, and the security findings cluster (SQL f-strings, default secret
  key, session token in a redirect URL). Separates LIVE from DORMANT.
- `operations.md` — runbook: workflow schedules, required secrets, load-bearing steps,
  troubleshooting
- `design.md` — ⚠️ **describes the frontend deleted 2026-08-10.** Visual design system:
  OKLCH palette, typography, spacing, components. Bannered, and kept as **rebuild input
  only** — the `frontend/app/*` paths it references no longer exist
- `product.md` — ⚠️ **same: rebuild input, not a live spec.** Positioning, users, brand
  personality, design principles, written in the present tense about an interface that
  no longer ships

## Removed 2026-08-05

`historical/` (5 files) and `retrain-optimization-analysis.md` were deleted — the first
documented only resolved issues, the second optimized a 36-model quantile grid that no
longer exists. Both are recoverable from git history if needed.
