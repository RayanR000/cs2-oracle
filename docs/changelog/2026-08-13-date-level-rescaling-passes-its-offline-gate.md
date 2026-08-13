# The date-level rescaling passes its offline gate at 3/7/30d, and voids its own instrument at 14d

**Date:** 2026-08-13
**Pre-registered in:** `docs/research/2026-08-13-date-level-sigma-rescaling-preregistration.md`
(committed as `71ef8c1`, **before** any number below was seen). Every bar, the placebo, the
seed and the void conditions are that document's; nothing here was revised after a read.
**Instrument:** `backend/scripts/attribute_band_level.py --legs` (offline, ~17s per horizon,
writes nothing), 8 new tests in `tests/test_band_level_attribution.py` (13 total)
**Panel:** `voted_39810715096a47b1688e44ef` — 731 dates, 2024-07-09 → 2026-07-09, ≥$1 cohort
**Status:** the arm **survives the gate at h=3/7/30 and is unrefereeable at h=14**. No flag
exists yet and nothing is dispatched; this entry is the offline gate the pre-registration
requires before a runner is spent.

## Verdict against the pre-registered bars

| leg | bar | h=3 | h=7 | h=14 | h=30 |
|---|---|---|---|---|---|
| (0) fitted `b` | in [0,1], CI not spanning 0–1 | ✅ | ✅ | ✅ | ✅ |
| (V) instrument validity | MAE ≤ 3pp vs the served control | ✅ 2.33 | ✅ 0.96 | ❌ **4.53** | ✅ 0.86 |
| (P) placebo | shuffled \|Δ cov\| ≤ 1.0pp | ✅ 0.46 | ✅ 0.35 | ✅ 0.21 | ✅ 0.24 |
| (S) spread, offline preview | −30% or better | ✅ −38% | ✅ −33% | ❌ **+4%** | ~ −28% |

## (0) The level elasticity, and a pre-registered point prediction that held

`b` is the OLS slope of `log median_i |resid[i,t]|` on `log L[t]` across dates, so `b = 1`
means a date whose trailing `sigma` runs 50% high also realises 50% more forward move — the
conformal score is already level-free and production is right. The bootstrap **resamples
dates**, which is the unit of variation: the level is one observation per date, not one per row.

| h | `b` | 95% CI | `gamma = 1 − b` | dates |
|---|---:|---|---:|---:|
| 3 | 0.300 | [0.213, 0.396] | 0.700 | 724 |
| 7 | 0.315 | [0.223, 0.420] | 0.685 | 716 |
| 14 | 0.292 | [0.195, 0.392] | 0.708 | 702 |
| 30 | 0.246 | [0.163, 0.335] | 0.754 | 670 |

The pre-registration predicted **`b` ≈ 0.20–0.30, `gamma` ≈ 0.70–0.80**, derived from
`log(1.10) / log(1.50) = 0.235` on the same-items ratios. Measured: 0.246–0.315, with every CI
excluding both 0 and 1. **No horizon is void, and forward dispersion repays roughly a quarter to
a third of a trailing-volatility swing** — the other two thirds are the defect.

## (V) The panel reproduces the served control at three horizons and not at 14d

| h | 2026-04-22 | 2026-05-16 | 2026-06-16 | MAE |
|---|---|---|---|---:|
| 3 | 92.14 / 89.85 | 88.30 / 85.58 | 84.01 / 85.99 | 2.33pp |
| 7 | 93.38 / 94.04 | 91.20 / 91.52 | 87.55 / 89.45 | 0.96pp |
| 14 | 94.83 / 95.90 | **90.78 / 82.38** | 81.88 / 85.99 | **4.53pp** |
| 30 | 93.51 / 93.76 | 89.60 / 88.12 | *no rows* | 0.86pp |

(panel / served, in %.) At h=14 a single anchor is **8.40pp** off. Per the pre-registration's own
void condition — *"if it cannot reproduce the control it cannot referee the arm"* — **h=14 is
voided as a measurement here**, and its (S) failure below is therefore not evidence against the
arm either. It is not a small discrepancy in a noisy direction: the panel says 2026-05-16
over-covers at 14d and the served band says it is the *best-calibrated* anchor in the set.

⚠️ **Two of the four audited anchors never enter this leg**, so the MAE is over 3 anchors at
h=3/7/14 and **only 2 at h=30**. A two-anchor validity check is weak, and h=30's pass should be
read as "not contradicted" rather than "validated".

## The pre-registration's six-anchor set does not survive its own label rules

The prereg asserted all six anchors "resolve h=30 without crossing the 2026-03-22 consensus
break". **That is wrong for 2026-03-10**, whose h=14 target (03-24) and h=30 target (04-09) both
span it. Measured rows per anchor:

| anchor | h=3 | h=7 | h=14 | h=30 |
|---|---:|---:|---:|---:|
| 2026-02-14 | 761 | 756 | 713 | 574 |
| 2026-03-10 | 616 | 563 | **voided** | **voided** |
| 2026-04-06 | 791 | 780 | 674 | 595 |
| 2026-04-22 | 598 | 529 | 484 | 632 |
| 2026-06-16 | 563 | 458 | 480 | *past panel end* |
| 2026-07-06 | **voided** | *past panel end* | *past panel end* | *past panel end* |

So the arm is measured on **5 / 5 / 4 / 3** anchors, not six, and the four horizons' deltas are
**not like-for-like** — a longer horizon is scored on a smaller and differently-composed set.
This is the third recurrence of one defect: *anchor selection consults the collection audit and
never the label-voiding detector*, noted on 2026-08-13 for 2026-07-06 and now again for
2026-03-10. **The two checks must be run together before an anchor set is fixed.**

## (P) The placebo separates the arm from the refuted date-level class

`L[t]` shuffled across dates, 200 permutations, seed 20260813 — fixed in advance.

| h | anchors | control | arm | real Δ | shuffled mean Δ | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 5 | 87.11% | 85.04% | −2.07pp | −0.46pp | 4.5× |
| 7 | 5 | 86.84% | 84.51% | −2.33pp | −0.35pp | 6.7× |
| 14 | 4 | 86.13% | 81.41% | −4.72pp | +0.21pp | 22.7× |
| 30 | 3 | 86.45% | 77.68% | −8.77pp | +0.24pp | 36.4× |

Every horizon clears the 1.0pp bar, and the effect is 4.5–36× the shuffled mean. **The two
arguments the pre-registration offered for why this is not the refuted class therefore stand**:
`2026-08-12-the-band-is-tilted-in-sigma.md` refuted conditioning `q_hat` on a date-level *state*
scored level-matched to 80%; this rescales `sigma` itself and leaves the within-date ordering
untouched.

⚠️ **The shuffled distribution is wide** — p95 of +2.8 to +4.6pp and a max |Δ| of 8.0–13.6pp. The
bar is on the *mean*, correctly, but it follows that **a single-permutation placebo would have
been uninformative here**, and that any future read on this arm needs the full distribution.

## (S) The spread, and the overshoot risk that is now the main threat

Across-anchor coverage spread, control → arm: **25.94 → 16.03pp (−38%)** at 3d,
**29.33 → 19.76pp (−33%)** at 7d, 12.96 → 13.42pp (+4%) at 14d — the voided horizon — and
**12.85 → 9.31pp (−28%)** at 30d. Per anchor:

| anchor | h=3 | h=7 | h=14 | h=30 |
|---|---|---|---|---|
| 2026-02-14 | 86.60 → 87.25 | 85.32 → 86.38 | 83.59 → 85.13 | 80.66 → **82.75** |
| 2026-03-10 | 71.92 → 78.41 | 67.85 → 72.29 | — | — |
| 2026-04-06 | 97.85 → 94.44 | 97.18 → 92.05 | 85.61 → **74.18** | 84.54 → **73.45** |
| 2026-04-22 | 92.14 → 80.27 | 93.38 → 83.18 | 94.83 → 87.60 | 93.51 → **77.06** |
| 2026-06-16 | 84.01 → 81.17 | 87.55 → 85.15 | 81.88 → 79.79 | — |

At 3d and 7d this is what a level fix is supposed to look like: the over-covered anchors come
down, the under-covered 2026-03-10 comes **up**, and the spread contracts from both ends.

🔑 **At 30d it overshoots.** Pooled coverage lands at 77.68% and two anchors fall to 73–77%, which
is the shape that failed `SIGMA_EXPONENT` (77.80% → 69.72%). The pre-registration is explicit that
**overshoot is a failure, not a partial success**. The offline pooled number is still *closer* to
80% than the control at every horizon, so bar (A) is not breached — but 30d is one step from
breaching it, on three anchors, on a panel that cannot even validate itself there beyond two.

## The arm is servable — `L[t]` does not depend on labels

A concern the instrument raised on itself: it derives `L[t]` from the **score frame**, whose rows
survived label voiding — a set serving cannot know. Against the label-free cross-section of the
same dates (every ≥$1 row with a finite `sigma`, no target):

| h | dates | median ratio | p01 | p99 | max \|dev\| | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 724 | 0.9999 | 0.9585 | 1.0452 | 0.117 | 0.9992 |
| 7 | 716 | 0.9998 | 0.9607 | 1.0479 | 0.083 | 0.9992 |
| 14 | 702 | 0.9999 | 0.9585 | 1.0408 | 0.077 | 0.9992 |
| 30 | 670 | 0.9998 | 0.9602 | 1.0385 | 0.086 | 0.9993 |

The two are the same series to a rounding error. **`L[t]` is computable at serve time from date
`t`'s own cross-section**, so the arm as measured is the arm that could ship.

## What this licenses

**Does:** the pre-registered dispatch, at `gamma` = 0.700 / 0.685 / 0.754 for h=3/7/30, on the
served band via `model-diagnostics.yml` — two jobs of three anchors, ~25 minutes each, per the
cost note. h=14 has no readable instrument offline and would be dispatched blind.

**Does not:** license a flag default, a ship, or a fourth width scale. The three-scale refutation
(`band-over-covers-basis-unconfirmed`) stands: `sigma`, `sigma**beta` and a learned scale all
calibrate to exactly 80% and land elsewhere when served. This arm is different in kind — it moves
the *level*, not the within-date weighting — but its own level has been observed to be
period-dependent, and six anchors in one year cannot settle that. The band's **width** bar is
still not read here at all; only coverage is.

## Reading the instrument

- `fit_level_elasticity` bootstraps **dates**, `anchor_coverage` calibrates one `q_hat` per anchor
  over rows at or before `anchor − embargo_days(horizon)` (H+13, derived not re-spelled), and
  `pooled_anchor_coverage` is **row-weighted** — an equal-weight average of anchor rates would let
  the thinnest date move the bar, which is the composition trap of 2026-08-11.
- `q_hat` absorbs `L ** gamma` and is **never differenced across the flag**; only coverage is.
- The 8 new tests pin the arithmetic on constructed panels, and each was mutation-checked: making
  `gamma` a no-op, making the shuffle a no-op, fitting the elasticity on rows instead of date
  medians, and removing the bootstrap resample each fail exactly the test that claims to catch it.
- Absolute coverage here is a **stand-in** — the panel derives its own clip bounds and cohort, and
  the residual is the realised return rather than a booster's error.
