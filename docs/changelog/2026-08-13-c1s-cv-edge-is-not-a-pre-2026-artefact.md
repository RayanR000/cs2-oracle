# C1's CV rank-IC edge is not a pre-2026 artefact — the regime explanation is spent

**Date:** 2026-08-13
**Change:** none. Read-only decomposition of already-run C1 diagnostics artifacts. No retrain,
no dispatch.
**Bears on:** `research/2026-08-13-next-steps.md` item 1 (and, through it, items 2 and 5);
`changelog/2026-08-13-cv-folds-are-not-time-aligned-with-serving.md`;
`c1-rank-transform-refuted-on-clean-anchors`.
**Verdict:** ❌ **the pre-registered collapse condition is not met.** The C1 CV edge is present
and positive in the 2026 fold on all four horizons. The "CV measures a different (pre-2026)
regime than serving" explanation for C1's serving failure is **falsified and spent.**

## Method

The per-fold `rank_ic` is serialised in `meta.json: cv_results[h].per_fold` beside `val_start` /
`val_end`. Pulled the C1 pair from the Model Diagnostics runs — control **`31663300447`**, arm
**`31663312585`**, commit **`ee76a9c`** — aligned folds by `(val_start, val_end)`, and computed
the arm−control `rank_ic` edge per fold. Pooled edge reproduces the stored `mean_rank_ic`
difference exactly: **+0.063 / +0.090 / +0.073 / +0.042** at 3/7/14/30d. (These differ slightly
from the +0.0684 / +0.0759 / +0.0588 / +0.0408 figures quoted in the next-steps doc — a
metric/run-quote difference, immaterial to the per-fold decomposition.)

## Pre-registered bar (from next-steps item 1)

> If the pooled CV gain is carried by pre-2026 folds and collapses on the 2026-01/02 fold, the CV
> and serving legs are measuring different regimes. If the edge is flat across the fold sequence,
> this explanation is spent.

## Result — no collapse

| H | pooled edge | pre-2026 mean (n=7) | 2026 fold(s), val_end |
|---|---|---|---|
| 3  | +0.063 | +0.072 | **+0.038** (02-13), +0.028 (07-28) |
| 7  | +0.090 | +0.090 | **+0.095** (02-13) |
| 14 | +0.073 | +0.075 | **+0.060** (02-13) |
| 30 | +0.042 | +0.033 | **+0.109** (02-13) |

The single 2026 validation fold (`2026-01-15 → 2026-02-13`) is **positive on every horizon**, and
on 7d and 30d it is *larger* than the pre-2026 mean. h=3 has a second 2026 fold ending 2026-07-28
that spans the serving window, also positive (+0.028). The edge does not collapse in 2026; it does
not even weaken monotonically. The regime-artefact hypothesis is falsified.

## What this changes

The rank-IC improvement from the C1 transform is **real and regime-robust**. So C1's refutation on
serving directional accuracy (1 of 4 horizons — see `c1-rank-transform-refuted-on-clean-anchors`)
is **not** explained by a train/serve regime mismatch. The unexplained gap is narrower and more
specific: a genuine within-fold *ranking* gain does not convert to served *directional accuracy*.

- **Item 2** (anchor the CV fold grid to the frame end) will not rescue C1 by aligning regimes —
  the 2026 fold already carries the edge. Do item 2 for band/`q_hat` and PT-sample hygiene, which
  is its own justification; do not expect it to close the C1 serving gap.
- **Item 5 / C2** (`lambdarank`) is **strengthened, not weakened.** The rank-IC diagnosis that
  motivates a ranker survives: rank structure demonstrably exists and holds in the 2026 regime; it
  is the q50-level → direction serving path that fails to capture it.

## Caveat carried, not resolved

`cv-folds-are-not-time-aligned-with-serving` still holds: for 7/14/30d the last CV window ends
2026-02-13, while served arms are read on anchors `2026-04-18 … 2026-06-08`. The CV never
validates *on* the serving months. What this decomposition rules out is the stronger claim that the
edge is a *pre-2026* phenomenon — the Feb-2026 fold being positive is exactly the evidence against
it. Whether the edge would also hold on Apr–Jun 2026 folds is untested and would require item 2 to
generate those folds.
