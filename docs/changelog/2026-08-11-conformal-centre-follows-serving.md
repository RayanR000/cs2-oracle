# q_hat now covers the mid that is served, not the mid that is fitted

F1 from `docs/research/2026-08-10-next-steps.md`. Shipped 2026-08-11.

**The band was calibrated around a centre `predict` then moved.** `q_hat` is fitted on
`|actual − q50_mid| / sigma`; `predict` builds the band from it and *afterwards* hands the
triple to `_recenter_on_direction`, which sets the mid to `±|mid|` — or pins it to exactly 0
— while preserving **both half-widths**. On every row where the directional classifier
disagrees with the q50's sign, the served centre is displaced by up to twice `|mid|` and the
band is not widened to pay for it.

> ## ⚠️ MEASURED, and the coverage claim below is WRONG on production data.
> Three arms at two anchors agree within **1.1pp** in all 8 cells. The centre is an
> incoherence worth fixing and **not** a coverage defect; it is also **not** the cause of
> production's `IntCov` 34.6–61.8%. Do not pay the 932s for it. Full table and the arithmetic
> in "The three-arm read" below — read that section before acting on anything above it.

## Sized on a held-out split — in a regime that does not hold

Same generator, same q50 mids, classifier disagreeing on 40% of rows; `q_hat` fitted on one
half, coverage measured on the other **through the serving assembly** (`conformal.band`, then
`_recenter_on_direction`):

| q_hat fitted around | served-band coverage | the q50-centred band nobody is served |
|---|---|---|
| the q50 mid (before) | **59.8%** | 79.5% |
| the served centre (after) | **79.1%** | — |

Against an 80% target. The old calibration was not broken arithmetic — it was correct about a
band that is never published.

**The generator is the problem, and it is why this number did not replicate.** It draws
`mid ~ N(0, 4pp)` against residuals of `sigma*20 ≈ 3pp`, so the displacement `2*|mid|` is
*larger* than the half-width. Production is the other way round by an order of magnitude:
predicted `|return|` is median 0.95% / p90 9.01% against half-widths of ±10% to ±31%. Read this
table as "the mechanism exists and this is its shape", never as its size.

## What changed

`_conformal_records` takes an optional `direction_class` and, when it has one, measures
`residual_pct` against `_recenter_on_direction`'s own output — the real serving function, not a
re-derivation, so the two centres cannot drift apart. Both callers pass it: CV passes its
per-fold out-of-fold call, and the single-holdout fallback passes the served classifier's
(in-sample there, which is the defect that path already warns about).

**Only `residual_pct` moves.** `mid_ret`, `change_pct` and `hit` stay on the q50 mid on
purpose: they feed `_calibrate_confidence`, whose thresholds are consumed by
`_compute_confidence`, which runs **only** on the no-classifier fallback path — and that path
serves the q50 mid with no recentring. Moving them would fit thresholds for a signal that path
never publishes. This is also why F2 (the served `confidence` tag is an uncalibrated `>= 0.5`
cut) is untouched here and still open.

`meta.json` carries `conformal_centre` per horizon, `"served"` or `"q50"`. Provenance, not a
switch — a coverage figure read without it is uninterpretable, and a predict-only run has no
other route to it. Absent on every artifact written before today, which loads as `{}`
("this artifact does not say"), so no `MODEL_ARTIFACT_VERSION` bump: no existing field changed
meaning, and `predict` reads the band from `conformal_calibration` alone.

## The falsifiable read

The existing coverage test measured `q_hat` against the residuals it was fitted on, which is
`>= nominal` by construction and could not fail. `scripts/replay_serving.py` now prints a
**BAND COVERAGE** table — `low <= realised <= high` on the served cohort, the same predicate
the backtest's `in_interval` uses, with the misses split into `miss<low` / `miss>high`. Split
conformal misses symmetrically; a displaced centre misses with a sign, so the split names the
cause rather than just reporting a number. `_served_rows` carries the band alongside the mid,
which is the not-inert half: a coverage table fed a frame with no `low`/`high` is a table of
NaN that reads as "nothing to see".

## Measured in CI, run `31529688893` (all four horizons, `conformal_centre=served`)

Two non-overlapping anchors, `horizons=matrix`, every arm off. Target 80%.

| h | 2026-05-16 | miss<low / >high | 2026-07-09 | miss<low / >high |
|---|---|---|---|---|
| 3 | 85.49% | 9.7 / 4.8 | **64.44%** | 13.1 / 22.5 |
| 7 | 91.05% | 6.4 / 2.5 | **74.61%** | 15.0 / 10.4 |
| 14 | 82.00% | 14.6 / 3.4 | 81.01% | 17.9 / 1.1 |
| 30 | 88.03% | 7.3 / 4.7 | 80.52% | 17.6 / 1.8 |

n = 1061 and 1032 per cell. **Six of eight cells at or above nominal**, and the two shortfalls
are both at 07-09 on the short horizons.

**The miss split earns its place immediately.** At 07-09/h=3 the misses skew *above* the band
(22.5 vs 13.1) while at 14d and 30d the same anchor skews *below* (17.6–17.9 vs 1.1–1.8). The
sign flips with horizon on one date, which is a directional-bias signature rather than a width
one — a band that is merely too narrow misses symmetrically. And the 85.49 → 64.44 swing
between two dates at h=3 is the market-date dominance already on record
(`da-is-dominated-by-the-market-date`).

## The three-arm read — the centre does not move coverage

`CV_DIAGNOSTIC_CLASSIFIER` and `REPLAY_DISABLE` became `model-diagnostics.yml` inputs
(`cd6a93d`) precisely so the run above could be paired. Two more dispatches, same commit, same
two anchors, same matrix:

| anchor | h | served centre | q50 centre (production today) | q50, recentring off |
|---|---|---|---|---|
| 05-16 | 3 | 85.49 | 85.67 | 85.86 |
| 05-16 | 7 | 91.05 | 91.14 | 90.86 |
| 05-16 | 14 | 82.00 | 82.19 | 81.06 |
| 05-16 | 30 | 88.03 | 88.12 | 88.31 |
| 07-09 | 3 | 64.44 | 65.12 | 64.53 |
| 07-09 | 7 | 74.61 | 75.10 | 74.52 |
| 07-09 | 14 | 81.01 | 81.40 | 80.72 |
| 07-09 | 30 | 80.52 | 80.91 | 80.04 |

Runs `31529688893` / `31532517480` / `31532543867`. **Maximum spread across the three arms is
1.1pp** (14d at 05-16), most cells inside 0.5pp. The WARNING fired 4 of 4 in the q50 arm, so
the disclosure works and the arm really was the defect.

**Why it cannot matter, in one line of arithmetic.** The recentring displaces the mid by at
most `2*|mid|`. This model's predicted `|return|` is **median 0.95%, p90 9.01%** (measured on
the ops mirror for `2026-08-11-actionable-selection-is-the-base-wedge.md`), while the
half-width is `q_hat*sigma` with `q_hat` = **95.5 / 141.4 / 204.9 / 312.7** at 3/7/14/30d
against a clipped sigma of ~0.05–0.30 — call it ±10% to ±31%. The displacement is a tenth of
the half-width, so it can only flip rows sitting in a thin sliver at the band edge. This is the
same fact that collapsed the actionable cohort: **almost nothing this model predicts is large
enough to matter to a threshold**, and a band edge is a threshold.

### Three consequences

1. **The centre is not the cause of production's `IntCov` 34.6–61.8%.** The replay reads ~81%
   pooled on the same band. Two figures describing one band that far apart put the discrepancy
   in the *backtest's comparison*, not the calibration. The suspect is the basis wedge:
   `in_interval` tests an archive-resolved actual against a band built on the served
   `current_price`, and that wedge is **median 5.70% / p90 37.82%** on the ops mirror against
   half-widths of 10–31%. `2026-08-11-actionable-selection-is-the-base-wedge.md` ruled
   `in_interval` coherent because "both legs are dollars" — that ruling is about *units*, and
   it does not survive being asked about *basis*. **This is the open question F1 actually
   surfaced**, and it is worth more than anything in the option list above.
2. **Neither production change is worth making.** Paying the 932s buys ≤1.1pp; dropping the
   recentring buys ≤1.1pp and costs the price/direction sign coherence it exists for. Leave
   production as it is. The option list above is settled, not pending.
3. **What F1 delivered is coherence and disclosure**, not coverage: `q_hat` now describes the
   band that is served, `conformal_centre` says which, the WARNING says when it is the wrong
   one, and `BAND COVERAGE` makes the whole question falsifiable at a fixed anchor for the
   first time. That is worth keeping at zero cost — it is not worth 932s.

The 05-16 → 07-09 swing (85.5 → 64.4 at h=3) dwarfs every arm difference and is the market-date
dominance already on record. Two anchors is a replication, not power.

**The mid did not move**, by construction: `q_hat` sets only the half-widths, and `mid_ret`,
`change_pct` and `hit` are pinned to the q50 centre by test. Any DA or rank IC in this run is
comparable to a control on an earlier commit.

## ⚠️ Inert on the daily path, and now loud about it

`price-forecast.yml` sets `CV_DIAGNOSTIC_CLASSIFIER=0`, so production CV emits **no** OOF
direction call and the served centre cannot be computed there. Those runs keep the q50 centre
and log a WARNING naming the ~20pp of coverage it costs. `model-diagnostics.yml` already sets
`=1`, so that is where the fix is live and where the empirical number comes from.

That is a disclosed defect, not a fix. The three ways out, in cost order:

1. **Stop recentring the mid.** Serve the q50 mid and publish the classifier's call as the
   direction label only. Coverage then needs no new calibration at all, and the existing
   `q_hat` is already right. The cost is that the served price and the served direction can
   disagree in sign, which is the coherence the recentring was added for.
2. **Calibrate on the last *k* folds only**, fitting the CV classifier just for those. Cost
   scales as `k/9` of the 932s — ~310s at k=3 — against a smaller but fresher calibration set,
   floored by `MIN_CALIBRATION_ROWS`.
3. **Pay the whole 932s in production.** It is 52% of a classifier-on retrain (872s off, 1804s
   on locally) and would put a retrain **at** the project's 30-minute cap. Not obviously
   affordable.

Choosing between them wants the replay's coverage number under option 1 beside the
served-centre number, which is one diagnostics dispatch each.

**A consequence to hold onto:** `CV_DIAGNOSTIC_CLASSIFIER` now moves the served **band**, not
just a diagnostic. A cost flag that changes a served artifact is the failure mode
`.claude/rules/training-budget.md` already records for regime models; `conformal_centre` in
`meta.json` is what keeps it disclosed rather than silent. The stale comment in
`_cv_evaluate_horizon` claiming this classifier "is a DIAGNOSTIC, not an input" is corrected in
place.

## Not in this change

- **C5 (split conformal + ACI)** was always downstream of this: a new calibration *scheme*
  built on the wrong centre inherits the same defect. It is now unblocked.
- **Arm A** (`SERVE_OUTLIER_GATED_ANCHOR`) rescales every served dollar band because `q_hat` is
  fitted in return space. Its coverage under the arm is still unmeasured — the replay can now
  report it, which is the last thing that was blocking that default.

Tests: `tests/test_minimal_model_shape.py` (records, centre provenance, the WARNING, both CV
wirings, the round trip) and `tests/test_serving_replay.py` (the band rows, the coverage
predicate, the NaN cohort).
