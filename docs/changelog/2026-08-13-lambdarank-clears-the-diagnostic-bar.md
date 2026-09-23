# C2 lambdarank clears the diagnostic bar at 4/4 — the first arm to beat the q50's own ordering

**Date:** 2026-08-13.
**Pre-registered:** `docs/specs/2026-08-13-lambdarank-diagnostic-design.md` (spec
`d995675`, implementation `343140b`, workflow arm `+1`), on branch `lambdarank-diagnostic`.
The bar, the tied-cohort primary read, and the point prediction were fixed before the dispatch.
**Read:** run `31735646130`, `LAMBDARANK=1 cv_diagnostic_classifier=q50`, matrix over 3/7/14/30d.
One dispatch — both bars are intra-run, so no paired control is needed.
**Verdict:** PASS at all four horizons. **The pre-registered prediction (null/negative at 3/4)
was wrong.**

## Result — CLEAN ANCHOR (tied) cohort

Both bars required, both > 0: beat the naive `−return_1d` baseline AND beat the q50's own
within-date ordering, where `p[d]/S[d] == 1`.

| Horizon | lr rank IC | q50 rank IC (tied) | naive (tied) | edge vs naive | edge vs q50 | tied rows / date-folds |
|---|---|---|---|---|---|---|
| 3d  | 0.1514 | 0.0866 | 0.0181 | **+0.1333** | **+0.0648** | 45,245 / 268 of 270 |
| 7d  | 0.1309 | 0.0701 | 0.0206 | **+0.1103** | **+0.0608** | 42,160 / 240 of 240 |
| 14d | 0.1152 | 0.0739 | 0.0098 | **+0.1054** | **+0.0413** | 41,977 / 240 of 240 |
| 30d | 0.0927 | 0.0667 | 0.0170 | **+0.0757** | **+0.0260** | 41,506 / 240 of 240 |

This is the first arm in the project to clear the vs-q50 bar on the clean-anchor cohort at any
horizon, let alone all four.

## Why this is more credible than `xs_rank` was

`xs_rank` (C1) posted +0.05 CV rank IC and is closed for good, because that gain was the anchor
denominator `p/S` and never reached serving. This read is built to exclude that failure mode:

- **The tied cohort** is exactly the rows where `p/S == 1`, so the scoring carries neither the
  raw nor the smoothed anchor wedge — the cohort every prior arm should have been ranked on.
- **The vs-q50 edge controls for the training label.** Both the ranker and the q50 train on the
  same labels (whose denominator is the raw anchor), on the same folds, with the same embargo. The
  delta between them isolates the *objective* — within-date ranking vs pointwise quantile — rather
  than any shared contamination in the label. That +0.026 to +0.065 is the transferable number.

A side-finding falls out of the control line: on the tied cohort the **q50 already beats naive**
(+0.0685 / +0.0495 / +0.0641 / +0.0497), while the pooled raw-anchor line still warns it loses
(0.1813 vs 0.1943 at 3d). That reconfirms "the model loses to `−return_1d`" as a pooled-basis
artifact, not a real deficit — the same lesson `xs_rank`'s closure taught, now visible in the
control of an unrelated arm.

## What it does NOT mean — the serving transfer is unproven

**This is a CV rank-IC diagnostic. It does not reach the served product, and that is where every
prior CV winner has died.** A ranker emits an ordinal score, not a direction or a band; production
serves direction from the classifier/q50 sign and a band from the q50 + conformal. Nothing consumes
a rank score today. The spec scoped this as diagnostic-first for exactly this reason: **a pass
opens the serving question, it does not license serving.**

Two further bounds on the number:

- **Magnitude is provisional.** HP were held at the q50's tuned params (the arm is a different
  objective); the sign is what the bar tests and it is unambiguous, but the size must be confirmed
  with `FORCE_HP_SEARCH=1` before it is quoted.
- **One dispatch, no interval.** Comparability rests on the documented bit-reproducibility of the
  rank-IC diagnostic; this is not a paired harness.

## Confirm read — the edge is not a stale-HP artifact

`FORCE_HP_SEARCH=1` re-read, run `31737771209`, same arm (`lambdarank+hpsearch`). Both arms use
the re-searched q50 params, so the vs-q50 comparison stays fair; this checks the *size* against the
possibility that the cached HP (selected on a different run/label basis) flattered it.

| Horizon | vs-q50 (cached) | vs-q50 (confirm) | vs-naive (confirm) |
|---|---|---|---|
| 3d  | +0.0648 | +0.0648 | +0.1333 |
| 7d  | +0.0608 | +0.0614 | +0.1169 |
| 14d | +0.0413 | +0.0391 | +0.1018 |
| 30d | +0.0260 | +0.0297 | +0.0757 |

The vs-q50 edge survives at 4/4, moving ≤0.002 at 7/14/30d. 3d is byte-identical because
`SKIP_HP_HORIZONS = [3]` (`forecaster.py:384`) skips the Optuna phase there, so FORCE_HP_SEARCH is
a no-op at 3d — the identical numbers are expected, not a caching artifact. **The diagnostic edge
is HP-robust.** The size is now confirmed, which is what the serving design should be sized against.

## Next step

The serving-transfer pre-registration the spec deferred is now earned. It has to answer the
question that closed `xs_rank`: **how does a within-date rank score become a served direction and
band, and does the +0.026–0.065 vs-q50 edge survive the trip through `predict()`?** That is a
distinct experiment with its own bar — not a licence to ship the ranker. Before it, confirm the
magnitude with a `FORCE_HP_SEARCH=1` re-read on the tied cohort, since a serving design should be
sized against the real edge, not the HP-borrowed one.
