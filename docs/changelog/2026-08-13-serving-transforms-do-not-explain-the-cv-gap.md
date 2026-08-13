# The serving transforms do not explain the CV→serving gap, and they are not costing the signal either

**Date:** 2026-08-13
**Pre-registered:** `docs/research/2026-08-13-serving-transform-attribution-preregistration.md`,
committed and pushed as `320c3bd` **before** dispatch, including the point prediction that leg (2)
would fail.
**Runs:** control-OFF `31665479967`, arm-OFF `31665491037` on `320c3bd` with
`REPLAY_DISABLE=blend,bias,recenter`, against the ON pair `31663300447` / `31663312585` on
`ee76a9c`. Same six audited anchors throughout.
**Verdict:** ❌ **leg (2) FAILS** — the transforms carry part of the gap and not enough of it.
**Leg (1) is null**: turning all three off does not move the control's served rank IC.

## Leg (2) — does `predict()` explain C1's failure to transfer? No

Arm − control, tied cohort, positive-anchor counts:

| | h=3 | h=7 | h=14 | h=30 |
|---|---:|---:|---:|---:|
| transforms **ON** | 4/6 | 3/6 | 1/6 | 1/6 |
| transforms **OFF** | **5/6** | **4/6** | **3/6** | 1/6 |

**Bar: ≥4 of 6 anchors at ≥3 of 4 horizons, AND strictly more positive than ON at ≥3 of 4.** The
second condition passes (3 of 4 horizons improve); **the first does not — only 2 horizons reach
4/6.** Both were required. Leg (2) fails.

**The pre-registered prediction was right about the verdict and wrong about the mechanism.** It
argued the transforms are near rank-preserving within a date and so should show *no* effect. They
show a **consistent** one: every horizon except 30d gains a positive anchor when they come off, and
none loses one. They are simply not large enough to be the answer. On means rather than signs the
picture is muddier still — the OFF leg improves at 7d and 14d (+0.0022 → +0.0059, −0.0603 →
−0.0262) and is *worse* at 3d and 30d (+0.0587 → +0.0396, −0.0583 → −0.0799) — which is why the
pre-registered statistic was sign consistency, as this harness's own footer instructs.

🔑 **The residual is the finding.** With all three transforms removed, CV still says
**+0.0684 / +0.0759 / +0.0588 / +0.0408** at 4 of 4 while serving says **+0.040 / +0.006 / −0.026 /
−0.080** on the mean. **The gap survives the removal of everything inside `predict()` that can move
a ranking.** h=30 does not move at all: 1/6 before, 1/6 after.

## Leg (1) — are the three transforms costing the served signal? No

Control OFF − control ON, tied cohort. **These transforms had never been scored**: they run only
inside `predict()`, so no CV number has ever seen them.

| | h=3 | h=7 | h=14 | h=30 |
|---|---:|---:|---:|---:|
| mean Δ | +0.0116 | +0.0180 | +0.0071 | −0.0061 |
| positive | 5/6 | 4/6 | 4/6 | 2/6 |
| per-anchor range | 0.3425 | 0.3028 | 0.4235 | **0.0443** |

**Null, and reported as the headline anyway because the pre-registration committed to it.** The
mean effect is +0.007 to +0.018 rank IC at 3/7/14d against a per-anchor spread **20–50× larger**;
six anchors cannot resolve that. The honest statement is that **the blend, the tier bias and
`_recenter_on_direction` move the served ranking materially on individual dates — by up to ±0.23 —
with no consistent direction.** They are neither the free win nor the hidden cost.

⚠️ **h=30 is different in kind and worth noting on its own.** Its per-anchor range is **0.0443**,
an order of magnitude below the other horizons: at 30 days the three transforms are very nearly
**inert on the ranking**. That is also the horizon where C1's transfer does not improve at all when
they are removed — consistent, and it means any future 30d serving question should look somewhere
other than these three knobs.

## What is now exhausted

Three explanations have been offered for C1 gaining +0.04–0.08 rank IC in CV and not reaching
serving. All three are now measured and spent:

1. **The label denominator** — refuted 2026-08-11 by reading both layers on the tied cohort.
2. **The anchor set** — refuted 2026-08-13 on six audited anchors, where 30d actually *reversed*.
3. **The serving transforms** — refuted here; removing all three leaves the gap.

Per the pre-registration, **this opens no new question and buys no further dispatch**: the 6
attribution runs that a pass would have justified are not worth running, because there is nothing
left to attribute. What remains is the cohort geometry the two paths score over — CV folds are
cohort-filtered and take `reference_mask=None`, so the serving branch is unreachable from CV — and
that is a code-reading question, not a dispatch.

## Standing caveat

Every anchor here is a high-volatility date (1.404–2.261× the panel norm). The archive cannot
supply a clean low-volatility set without crossing a collection regime, so none of this describes a
calm market.
