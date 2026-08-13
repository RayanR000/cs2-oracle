# C1's refutation survives clean anchors — and 30d, which passed on the dirty set, reverses

**Date:** 2026-08-13
**Pre-registered:** `docs/research/2026-08-13-c1-audited-anchor-preregistration.md`, committed and
pushed as `ee76a9c` **before** either run was dispatched.
**Runs:** control `31663300447`, arm `31663312585` (`cross_sectional_rank=true`), both on `ee76a9c`,
`horizons=matrix`, six anchors each.
**Read with:** `python -m scripts.compare_diagnostics --control 31663300447 --arm 31663312585`
**Verdict:** ❌ **bar (A) FAILS at 3 of 4 horizons.** C1 stays shelved. The mis-collected anchors
were not why it failed.

## The question

C1 was shelved on a serving leg of 2/4, 2/4, 0/4, 4/4 positive anchors — and two of those four
anchors are refused by the audit that shipped the next day, carrying the two largest negatives in
the table. On the two clean anchors C1 read positive at 3 of 4 horizons instead of 1 of 4. Two
anchors are replications, not evidence, so this re-dispatched on six anchors clean on **both**
audits at all four horizons inside one collection regime.

## (A) Primary — serving, tied cohort. FAILS

Arm − control tied-cohort rank IC. Bar: positive at ≥4 of 6 anchors, at ≥3 of 4 horizons.

| anchor | h=3 | h=7 | h=14 | h=30 |
|---|---:|---:|---:|---:|
| 2026-04-18 | −0.0467 | −0.0394 | −0.0065 | −0.1163 |
| 2026-04-28 | **+0.1191** | −0.0502 | −0.0290 | −0.0999 |
| 2026-05-08 | −0.1095 | **+0.1228** | −0.1418 | **+0.0465** |
| 2026-05-19 | **+0.1365** | −0.1330 | −0.2347 | −0.1682 |
| 2026-05-29 | **+0.1461** | **+0.1091** | −0.0307 | −0.0036 |
| 2026-06-08 | **+0.1064** | **+0.0036** | **+0.0810** | −0.0085 |
| **positive** | **4/6 ✅** | 3/6 ❌ | 1/6 ❌ | 1/6 ❌ |

**One horizon of four clears. The bar needs three.**

🔑 **30d reverses, and that is the most informative cell in the read.** It was the one horizon that
passed the contaminated set at **4/4**; on audited anchors it is **1/6**, negative at five of six.
The pre-registration had already ruled that 30d alone could not carry a pass — it did not need to.
**Contamination was not biasing the read in one direction**; it manufactured a positive at 30d and
negatives at 3d/7d, and removing it moved both.

**14d fails exactly as pre-registered**, 1/6 after 0/4. That prediction is on record in `ee76a9c`.

## (B) Secondary — CV. PASSES, and that is the whole problem

| h | ctl IC | arm IC | Δ | ctl edge | arm edge | rows | dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | +0.0866 | +0.1550 | **+0.0684** | +0.0685 | +0.1369 | 45,245 | 268 |
| 7 | +0.0701 | +0.1460 | **+0.0759** | +0.0495 | +0.1254 | 42,160 | 240 |
| 14 | +0.0739 | +0.1327 | **+0.0588** | +0.0641 | +0.1229 | 41,977 | 240 |
| 30 | +0.0667 | +0.1075 | **+0.0408** | +0.0497 | +0.0905 | 41,506 | 240 |

Arm edge ≥ 0 at 4/4 and arm > control at 4/4, replicating the prior read's
+0.0792 / +0.0801 / +0.0641 / +0.0509 closely. **So the CV↔serving divergence is now confirmed on
clean anchors**: a within-date rank transform gains +0.04 to +0.08 rank IC in cross-validation and
does not reach serving. The contaminated-anchor explanation is spent, as the contaminated-denominator
one was before it.

## (C) Control reproducibility — FAILS, and the bar was mine to write badly

Prior control CV rank IC was +0.0764 / +0.0723 / +0.0720 / +0.0607; this control reads
+0.0866 / +0.0701 / +0.0739 / +0.0667, i.e. **+0.0102 / −0.0022 / +0.0019 / +0.0060** — outside the
pre-registered ≤0.002 at three of four horizons. By the letter, that voids (A).

**It does not rescue the arm, and the bar was mis-specified.** It compares across commits —
`ab08a8b` to `ee76a9c`, two days of label, universe and audit changes — while the pairing (A)
actually depends on is control-versus-arm *within this dispatch*, and that is verified directly:
both runs share the commit, the cache and the fold geometry, and all **24 anchor-horizon-cohort
cells paired with identical `n`** (`pair_replay_rows` flags any cell whose count moves; none did).
Recorded as an error in the pre-registration rather than quietly dropped: a reproducibility bar
must name the pair it is testing.

## The deviating cohort, again

The arm is worse on the deviating cohort in **19 of 24 cells**, several by 0.19–0.43 (h=14 at
05-29: −0.4273; h=7 at 05-29: −0.3461). The prior read saw 13 of 16. That is two-thirds of the
served universe getting materially worse, and it replicates.

## What this closes and what it leaves

**Closes:** the anchor-contamination explanation for C1's serving failure. Two independent
confounds have now been removed — the label denominator (2026-08-11) and the anchor set (today) —
and the arm still does not reach serving. **C1 stays shelved, now on clean evidence.**

**Leaves:** the same open question, better posed. A transform that gains +0.04–0.08 in CV and loses
at serving differs somewhere between the two paths, and both cohort-construction explanations are
now spent. What remains is inside `predict` — the prior-day blend, the tier bias,
`_recenter_on_direction`, the conformal band — which `REPLAY_DISABLE` can switch off one at a time.
That is a cheap, well-posed next instrument and it needs no retrain.

⚠️ **Every anchor here is a high-volatility date** (1.404–2.261× the panel norm). The archive cannot
supply a clean low-vol set without crossing a collection regime, so nothing in this entry describes
C1 in a calm market.
