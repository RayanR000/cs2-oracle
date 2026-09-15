# Spec: work plan for the recalibration wait window (2026-09-14 → ~2026-10-12)

**Status:** PROPOSED. Four independent items; none requires a retrain or a serving change
except where stated. Written while the served-outcome `q_hat` feedback is data-blocked.

## Context

The served band over-covers. Measured 2026-09-14 on prod `forecast_outcomes`, ≥$1, the two
documented date exclusions applied, **post-2026-08-19 era only** (pooling across geometry eras
is invalid — see `served-panel-and-geometry-eras`):

| h | coverage | band width (median, % of base) | middle-80% of actual moves | ratio |
|---|---|---|---|---|
| 3 | 0.926 | 0.144 | 0.080 | 1.80× |
| 7 | 0.914 | 0.228 | 0.102 | 2.24× |
| 14 | 0.932 | 0.320 | 0.145 | 2.20× |

Target is 80%. The band is ~2.2× wider than an honest 80% interval needs to be, and the
coverage is cheap: 57%/46%/33% of items move <2% at h=3/7/14.

`models/served_recalibration.py` already computes the correcting multiplier and self-activates
at `MIN_FORECAST_DATES = 20` distinct served dates per horizon, floored at
`_geometry_floor()` = **2026-09-06** (`SHRINK_K_SERVING_START`). Current floored panel: **h=3 has
4 dates, h=7/14/30 have 0.** `served_coverage_factors()` returns `{}` — correctly dormant.

**The 20 is real, not cautious.** Bootstrapped the per-date factor over historical dates
(3,000 draws, sampling without replacement):

| days used | 5th–95th pct of recommended factor |
|---|---|
| 3 | 0.55 – 1.55 |
| 5 | 0.59 – 1.42 |
| 10 | 0.74 – 1.20 |
| 20 | 0.89 – 1.04 |

At ≤5 days the correction's **direction** is undetermined. Do not lower the gate.

⚠️ Calling `factors_from_panel` without the floor on today's panel returns
`{3: 1.44, 7: 1.23, 14: 1.30}` — *widen by 23–44%*, the exact opposite of correct, because it
averages in the under-covering pre-08-19 era. Production applies the floor and is safe. Any
change to this thread must preserve that floor.

---

## Item 1 — Resolve the outcome backlog (do first; free clock)

**Problem.** Forecasts exist whose target dates have passed but which have no
`forecast_outcomes` row. Above the 2026-09-06 geometry floor:

| h | unresolved forecast_date | rows |
|---|---|---|
| 3 | 2026-09-06, 2026-09-11 | 5,536 each |
| 7 | 2026-09-06, 2026-09-07 | 5,536 each |

(h=3 2026-09-07..09-10 are resolved; the gap is not a systematic resolver failure.)

**Why it matters.** These are dates already earned but not credited. Resolving takes h=3 from
**4 → 6** and h=7 from **0 → 2** toward the 20-date gate, and removes an incomplete-panel bias
from every headline accuracy figure.

**Change.** None to code. Run the Backtest Accuracy resolver over the backlog.

**Acceptance.**
- `served_recalibration._load_panel(db, (3,7,14,30), since=_geometry_floor())` reports
  h=3 ≥ 6 dates and h=7 ≥ 2 dates.
- No `forecast_date` below 2026-09-06 gains rows (the floor is about *input* geometry; older
  resolutions are fine for headline accuracy but must not be mistaken for gate progress).
- `scripts/check_outcome_dates.py` snapshot shows counts non-decreasing, no pruning.

**Update 2026-09-15.** Ran the resolver (Backtest Accuracy dispatch
`34984152746`, green): 09-11 (h=3) and 09-07 (h=7) resolved, but **09-06 resolves
at no horizon and never will** — its base leg `[f − staleness, f]` falls inside
the 08-28..09-05 collection outage, and the resolution gate classes those rows
unscoreable (12.6% of mature forecasts this run, "a collection gap, not a
resolver regression"). Three consecutive green runs agree, so this is structural,
not a missed run. Do not dispatch again for 09-06. Floored panel is now h=3: 5
dates (09-07..09-11), h=7: 1 (09-07). The gate clears on **new** dates only:
09-12 h=3 and 09-08 h=7 both mature 09-15 and should resolve in the next daily
runs — earliest gate ETA ~09-16, via the normal chain, no action needed.

**Hazards.** Writes to prod Postgres. The resolver is the only writer of `forecast_outcomes`;
the store is append-only (`outcomes-store-is-append-only`) so a bad run is not silently
reversible. Confirm with the user before dispatching.

**Effort.** Minutes, if the workflow run is clean.

---

## Item 2 — Restore freshness monitoring (do first; cheap insurance)

**Problem.** `Forecast Freshness Check` is `disabled_manually`. Nothing alerts on a missed
daily run. **Update 2026-09-15: stale — the workflow is `active` with green scheduled runs; no action needed.** Two clocks depend on unbroken daily continuity:

- the served recalibration gate (Item 1's 20 dates), and
- the archive-basis rank leg, which clears **2026-09-27 with a one-date margin** (13 qualifying
  + 8 pending = 21 vs `MIN_DATES = 20`). Two missed days re-VOID it.

The 2026-08-28..09-05 outage — 9 permanently lost forecast dates — is precisely this failure
mode, and it went unnoticed.

**Change.** `gh workflow enable "Forecast Freshness Check"`. **The user must run this**; the
harness classifier blocks workflow-enable from here.

**Acceptance.** Workflow shows `active` in `gh workflow list --all`; one scheduled run
completes.

**Hazards.** Known false-positive: a manual full dispatch after UTC midnight trips
`check_forecast_freshness.py` against UTC-today while the chain is healthy
(`manual-full-dispatch-freshness-false-fail`). Expect it; do not treat the first such failure
as a real outage. If the noise is unacceptable, fix the check to compare against the
*collected* anchor date rather than UTC-today — but that is a separate change, not a
prerequisite.

**Effort.** One command, plus tolerating one known false alarm.

---

## Item 3 — Reduce `q_hat` fold-composition variance (the real work)

**Problem.** The conformal width `q_hat` swings **±26% on fold composition alone** — per-fold
`q_hat` varies **1.4–1.85× at identical 300K training rows** (h=3, folds 4–9: 68.8 → 126.9),
and `spearman(n_train, q_hat)` flips sign by horizon (−0.66/+0.22/+0.76/−0.52), i.e. training
size explains approximately none of it. Root cause is split-conformal exchangeability breaking
under regime shift: a scalar `q_hat` pooled over a calibration window that mixes volatility
regimes, then served per-date.

This is **not** the same defect as the over-width in Item 1's table, and fixing one does not fix
the other. It is also a prerequisite for trusting the automatic correction: feeding a ±26%-noisy
width into a multiplier fitted on served outcomes propagates that noise into the served band.

**Explicitly out of scope — do not propose these.** Conditioning the width on anything is a
measured dead end, seven times over: sigma, sigma-exponent, learned scale, exceedance,
per-tier Mondrian conformal, bagged `q_hat`, vol-rank GBM, SHRINK_K GBM, log1p-listings,
supply-velocity. Conditioning the band **always** loses to flat pooling
(`band-width-levers-refuted`, `served-band-overcoverage-and-feedback-calib`). This item targets
the **variance of the estimator**, not the level and not per-item conditioning.

**Direction (not yet a decision).** The tractable question is whether the fold *design* — not
the fold *features* — can be made to produce a stable statistic: `CV_STEP_DAYS`
(`models/forecaster.py:809`, currently 150) controls fold count and overlap, and per-fold
`q_hat` dispersion is directly measurable. Note `CV_STEP_DAYS` and boost-round cuts were both
already refuted **as cost reductions** (`retrain-cost-profile-and-cv-folds`); that is a
different question from whether they move estimator variance, and the prior run measured cost,
not dispersion.

**Required first step.** Measure before changing anything: a harness that reports per-fold
`q_hat` dispersion per horizon under varying fold designs, on the existing archive panel. No
retrain needed for the measurement.

**Acceptance (of the measurement, not of a fix).**
- Per-fold `q_hat` dispersion reported per horizon, with a placebo arm.
- A stated, falsifiable bar for what "less noisy" means, written **before** any number is read.
- Any proposed change must show coverage is not degraded at matched width on the prod basis.

**Hazards.** Conformal CV is **57% of an 847s retrain** — iteration is slow; run from the
controller with a Bash waiter, not a subagent (`long-runs-run-from-controller`). Do not tune
against a single split; that is how the vol-rank "6–8% narrower" claim was manufactured and
later refuted. This item needs a written pre-registration before any statistic is read.

**Effort.** The bulk of the wait window.

---

## Item 4 — Fix the `count_in_24` source splice (latent, cheap)

**Problem.** `scripts/backfill_buff_iflow.py` writes a single `steam_vol` field from two
different upstreams depending on era (`scripts/backfill_buff_iflow.py:95-115`): for pre-
2024-02-13 rows it reads `count_in_24` from the BUFF "DATA" db, and for later rows
`steam_volume.volume`. Both land in one column. Any volume feature built on it silently mixes
two incomparable series at an era boundary.

**Why now.** Harmless today — the ingest is not running and nothing reads the column
(`iflow-count-in-24-splice-defect`). It becomes expensive the moment someone builds on it, and
the era boundary is exactly the kind of seam that fabricates signal.

**Change.** Emit the two series as separate, explicitly named columns; never collapse them. Do
not attempt to reconcile or rescale them into one.

**Acceptance.**
- Output carries two distinct columns with era-explicit names; neither is named `steam_vol`.
- A test asserts a pre-2024-02-13 row and a post-boundary row populate *different* columns.
- No consumer reads a merged column (currently none does — keep it that way).

**Hazards.** None to serving; the ingest is dormant. Do not run the backfill as part of this
item — expanding the served universe is a separate, budget-breaking decision
(`iflow-backfill-expands-served-universe`, `training-breadth-verdict`).

**Effort.** ~1 hour including the test.

---

## Explicitly not in this window

- **No new modelling.** The ML tracker is fully closed (#1 REFUTED, #2 SHIPPED, #3 NULL,
  #4 KILLED); it is not a backlog.
- **Nothing on the centre.** `lambda* = 0` reproduced three times on three labels/harnesses;
  the GBM centre loses to last-price at every horizon (measured again 2026-09-14: skill
  −5.6%/−6.6%/−6.9%/−12.5%, all CIs below zero).
- **Do not run the archive-basis rank leg before its gate clears.** One clean shot; re-check
  with `--gate` on ~2026-09-27, and glance at the Price Forecast run list ~09-20 to protect
  the one-date margin.
