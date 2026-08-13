# On a clean anchor set the date-level rescaling fixes the level and fails its own spread bar

**Date:** 2026-08-13
**Follows:** `2026-08-13-date-level-rescaling-passes-its-offline-gate.md`, which read the arm on
the pre-registration's six anchors. **Half that set was unreadable**, so this re-reads it on six
anchors that pass both audits inside one collection regime.
**Instruments:** `scripts/replay_serving.py` gains `cutovers_from_counts` /
`cutovers_in_outcome_window` (4 tests), and the anchor set was re-picked with them.
**Status:** ❌ **the pre-registered mechanism claim (S) FAILS.** ✅ The level effect is real, large,
and lands production's band on 80.67% / 82.12% from 86.36% / 87.76%. **Not dispatched.**

## What was wrong with the anchor set, and the check that now catches it

`audit_anchor_feed` looks ±3 days around the anchor and asks how *that day* was collected. It
cannot see a basis change 30 days out — and the replay resolves outcomes itself, so it never
consulted the label path's cutover detector either. **An anchor can therefore pass every check the
repo had and still resolve its outcome across a collector cutover**, which is a synthetic
market-wide move, not a return.

`cutovers_in_outcome_window` closes it, using `_collection_shift_dates`' own rule and threshold
(read off the class, not restated) and `prepare_targets`' own span rule, `(anchor, anchor + h]`.
`replay_serving` now drops the affected horizons and refuses the anchor when none survive.

Detected cutovers in the local archive: **2026-03-22, 04-16, 07-09, 07-10, 07-11, 07-12, 07-14,
07-15**. The pre-registered set lost 2026-03-10 at 14d/30d to the first of these.

## The archive cannot supply a regime-spanning clean set

Candidate 2026 dates that pass **both** audits at all four horizons, inside one collection regime:

| requirement | clean dates | window | `sigma` vs norm |
|---|---:|---|---|
| h=3,7 | 87 | 2026-03-24 → 07-01 | 1.079 – 2.261× |
| h=3,7,14 | 73 | 2026-03-24 → 06-24 | 1.079 – 2.261× |
| h=3,7,14,30 | 52 | 2026-04-18 → 06-08 | 1.404 – 2.261× |

🔑 **Every clean anchor is a high-`sigma` date.** The archive's item count runs ~5,200/day before
March, ~32,400 from 03-24 to 04-15, and ~25,010 after the 04-16 cutover, so a set reaching back
into the low-vol quarter also crosses two collection regimes and compares three different cohorts.
The pre-registration's premise of a regime-spanning set is **not satisfiable** on this archive:
the variation available is 2.1× and one-sided, all of it above the panel norm.

**The set used below** — all in the ~25,010-item regime, clean on both audits at h=3/7, spanning
the full available range, and including 2026-06-16 from the original audited four:

`2026-04-29` (2.255×), `2026-05-06` (2.125×), `2026-05-29` (1.619×), `2026-06-06` (1.467×),
`2026-06-16` (1.156×), `2026-06-26` (1.079×).

## The read, against the bars as written

| bar | h=3 | h=7 |
|---|---|---|
| (P) placebo, ≤1.0pp | ✅ 0.17pp (effect 32.7×) | ✅ 0.24pp (effect 23.3×) |
| (A) level, toward 80% | ✅ 86.36 → **80.67%** | ✅ 87.76 → **82.12%** |
| (S) spread, −30% | ❌ 19.46 → 16.57pp (**−15%**) | ❌ 20.89 → 28.09pp (**+34%**) |
| (D) dose-response, ρ falls | ✅ +0.714 → +0.600 | ❌ +0.486 → +0.486 |

Per anchor (control → arm, %):

| anchor | `sigma` | h=3 | h=7 |
|---|---:|---|---|
| 2026-04-29 | 2.255× | 95.79 → 88.72 | 96.71 → 91.17 |
| 2026-05-06 | 2.125× | 96.69 → 87.99 | 94.78 → 86.65 |
| 2026-05-29 | 1.619× | 81.82 → **72.15** | 75.82 → **63.08** |
| 2026-06-06 | 1.467× | 83.65 → 78.57 | 84.50 → 80.17 |
| 2026-06-16 | 1.156× | 84.01 → 81.17 | 87.55 → 85.15 |
| 2026-06-26 | 1.079× | 77.23 → 75.90 | 86.00 → 85.01 |

**The high-`sigma` anchors behave exactly as the mechanism predicts** — 04-29 and 05-06 fall 7–9pp
from 95–97%, 06-26 barely moves at 1.1×. The spread bar fails on **one anchor**: 2026-05-29 sits
mid-range in `sigma` and drops **9.7pp at 3d and 12.7pp at 7d**, from a control that was already
*under* 80% at 7d. Its `L[t]` does not predict that move, which is the one thing this arm is
supposed to be.

⚠️ **A statistic the pre-registration did not fix, reported as post-hoc.** Mean per-anchor
|coverage − 80| improves at both horizons: **7.46 → 5.21pp** at 3d and **8.95 → 7.51pp** at 7d. So
the arm does reduce per-date error on average; what it fails is max-minus-min, which one anchor
owns. **The bar as written is the one that counts** — the alternative was chosen after seeing the
numbers and cannot be substituted for it.

## Why the first read said −38% / −33% and this one says −15% / +34%

The earlier spread reduction was carried by **2026-03-10**, an under-covered anchor at 71.92% that
the arm raised to 78.41%. That anchor is one of the two the outcome-window audit now rejects at
long horizons, and it sits in a different collection regime from the rest. **A spread statistic
over six anchors is owned by its extremes, and the extremes were the contaminated dates.** Same
defect, third instance, now measured rather than argued.

## Verdict

**The arm is a level fix.** It is not the date-level mechanism it was pre-registered as: the
placebo confirms the effect is genuinely date-linked and not a constant (32.7× and 23.3× the
shuffled mean), but the date term it removes does not flatten coverage across dates, and at 7d it
sharpens the spread.

**Not dispatched.** Spending ~50 minutes of runner time to confirm a level move that the offline
instrument already reproduces to 0.96–2.33pp, while the mechanism bar fails, buys nothing the
pre-registration would accept as a result.

**What is genuinely open, and needs its own pre-registration.** Production's live defect *is* the
level — the served band covers 86–89% against 80% — and after three refuted width scales this is
the first change that lands it on **80.67% / 82.12%**. Shipping a level correction that does not
fix conditional coverage is a defensible product decision and an indefensible mechanism claim, so
it must be argued as the former. Two things such a pre-registration would have to settle:

1. **2026-05-29.** One anchor moving 10–13pp against its own `L[t]` is either a property of that
   date or a hole in the arm, and six anchors cannot tell which.
2. **The one-sided anchor pool.** Every readable anchor is a high-`sigma` date, so the arm has been
   observed only where it *lowers* coverage. Its behaviour on a low-vol date — where it must
   *raise* the band — is unmeasured, and that is the direction that produced the 30d overshoot to
   77.68% in the previous entry.
