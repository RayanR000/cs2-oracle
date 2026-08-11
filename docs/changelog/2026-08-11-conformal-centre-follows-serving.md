# q_hat now covers the mid that is served, not the mid that is fitted

F1 from `docs/research/2026-08-10-next-steps.md`. Shipped 2026-08-11.

**The band was calibrated around a centre `predict` then moved.** `q_hat` is fitted on
`|actual − q50_mid| / sigma`; `predict` builds the band from it and *afterwards* hands the
triple to `_recenter_on_direction`, which sets the mid to `±|mid|` — or pins it to exactly 0
— while preserving **both half-widths**. On every row where the directional classifier
disagrees with the q50's sign, the served centre is displaced by up to twice `|mid|` and the
band is not widened to pay for it.

## Sized on a held-out split

Same generator, same q50 mids, classifier disagreeing on 40% of rows; `q_hat` fitted on one
half, coverage measured on the other **through the serving assembly** (`conformal.band`, then
`_recenter_on_direction`):

| q_hat fitted around | served-band coverage | the q50-centred band nobody is served |
|---|---|---|
| the q50 mid (before) | **59.8%** | 79.5% |
| the served centre (after) | **79.1%** | — |

Against an 80% target. The old calibration was not broken arithmetic — it was correct about a
band that is never published. Production reports `IntCov` 34.6–61.8% on the `lgbm-v3` cohort;
this is a mechanism that produces exactly that shape of miss.

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
