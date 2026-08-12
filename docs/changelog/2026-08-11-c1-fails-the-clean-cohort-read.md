# C1's CV gain survives the clean cohort; its failure to reach serving survives too

**Pre-registered:** `docs/research/2026-08-11-c1-tied-cohort-preregistration.md`, committed and
pushed as `f457666` while both runs were mid-flight and no number from either was visible.
**Runs:** control `31547391395`, arm `31547400215` (`cross_sectional_rank=true`), both on
`ab08a8b`, four non-overlapping replay anchors.
**Read with:** `python -m scripts.compare_diagnostics --control 31547391395 --arm 31547400215`.

## Why this run existed

`2026-08-11-rank-transform-does-not-transfer-to-serving.md` refuted C1 on the pooled basis. The
same day put the label-denominator wedge at +0.14 rank IC — larger than the effect being judged —
so that verdict was *unreadable*, not wrong. This re-reads both layers on the tied cohort, where
neither `p[d]/S[d]` factor operates.

## The verdict: FAIL, on the decisive leg

**CV, clean-anchor cohort — passes 4 of 4.** The bar was arm edge ≥ 0 and arm > control at 3 of 4.

| h | ctl IC | arm IC | Δ | ctl edge | arm edge |
|---|---|---|---|---|---|
| 3 | +0.0764 | +0.1556 | **+0.0792** | +0.0576 | **+0.1368** |
| 7 | +0.0723 | +0.1524 | **+0.0801** | +0.0511 | **+0.1312** |
| 14 | +0.0720 | +0.1361 | **+0.0641** | +0.0612 | **+0.1253** |
| 30 | +0.0607 | +0.1116 | **+0.0509** | +0.0434 | **+0.0943** |

Note what this corrects: on the clean cohort the *control* beats `-return_1d` at all four horizons
(+0.043 to +0.061). The "the ML stack is subtracting from its own best feature" warning was itself
an artifact of the contaminated denominator.

**Serving, tied cohort — fails.** The bar was a positive arm−control delta at ≥ 3 of 4 anchors.

| h | 04-15 | 05-16 | 06-16 | 07-09 | positive |
|---|---|---|---|---|---|
| 3 | −0.0422 | +0.0062 | +0.1357 | −0.2532 | 2/4 |
| 7 | −0.1657 | +0.0088 | +0.0083 | −0.0643 | 2/4 |
| 14 | −0.2525 | −0.1990 | −0.0114 | −0.5927 | **0/4** |
| 30 | +0.0694 | +0.0181 | +0.2247 | +0.0335 | 4/4 |

Only 30d clears, and the pre-registration ruled in advance that **30d alone is not a pass** — its
clean-anchor leg is the one unconfirmed in CI and its embargo exceeds its validation window.
Dropping the 26-item 07-09 anchor does not rescue it: 2/3, 2/3, 0/3, 3/3.

**C1 stays shelved.** The 2026-08-11 refutation survives the contamination correction.

## The control reproduced a prior run to four decimals

Control tied serving IC, mean across the four anchors: **+0.1321 / +0.1562 / +0.1747 / +0.0536** —
identical at every horizon to run `31456524146` in `2026-08-11-clean-anchor-confirmed-in-ci.md`.
Different dispatch, same cached HP and folds. That bit-reproducibility is the control a paired
harness would otherwise have to supply, and it is why one dispatch per arm is readable.

It also independently re-confirms the clean-anchor signal itself: +0.13 to +0.17 at 3/7/14d, with
30d again the weak leg at +0.0536.

## What the arm does to the deviating cohort

Worth recording because it was not the question. The arm is **worse on the deviating cohort in 13
of 16 cells**, several by 0.20–0.27 (h=3 at 06-16: −0.2062; h=7 at 06-16: −0.2120). A within-date
rank transform makes the contaminated two-thirds actively worse while helping CV. That is
consistent with the transform amplifying the anchor-deviation channel the rank is computed over,
and it is a mechanism worth naming rather than a result — nothing here isolates it.

## The open question this leaves

**Why does a within-date rank transform gain +0.05 to +0.08 in CV and lose at serving?** Both
layers are now measured on the same cohort, so the contaminated-denominator explanation is spent.
The remaining candidates are the CV/serving cohort construction (folds are already cohort-filtered
and take `reference_mask=None`, so the serving branch cannot be reached from CV) and the serving
transforms inside `predict`. This is a smaller, better-posed question than the one this run opened
with.

## What did not happen

No confirm run. `force_hp_search=1` was reserved for a positive, and there is nothing to confirm.
No interval: four anchors are replications, not power, and the readable statistic remains sign
consistency.
