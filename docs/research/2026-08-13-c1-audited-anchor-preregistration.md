# Pre-registration: does C1's serving failure survive an audited anchor set?

**Date:** 2026-08-13, written and committed **before** either run is dispatched.
**Arm:** `cross_sectional_rank=true` on `model-diagnostics.yml`, against a control dispatch on the
**same commit**. No code change — C1 has been built and gated since 2026-08-10.
**Re-opens:** `docs/changelog/2026-08-11-c1-fails-the-clean-cohort-read.md`.
**Bars, anchors, basis and void conditions are fixed here. Nothing below may be revised after a
number is seen.**

## Why a refuted arm is being re-read

C1 was shelved on its serving leg: a positive arm−control rank-IC delta at 2/4, 2/4, **0/4** and
4/4 anchors at 3/7/14/30d, against a bar of ≥3 of 4. Those anchors were
`2026-04-15, 05-16, 06-16, 07-09`.

**`audit_anchor_feed` shipped the following day and rejects two of them.** `2026-07-09` is the feed
substitution — the usual feed absent, a stand-in quoting the ≥$1 cohort at **1.287×** — and
`2026-04-15` is the last day of the three-source era, so every horizon resolves on the feed that
replaced it. Both are exactly the cells that sank the read:

| h | 04-15 ❌ | 05-16 ✅ | 06-16 ✅ | 07-09 ❌ | clean only |
|---|---|---|---|---|---|
| 3 | −0.0422 | +0.0062 | +0.1357 | −0.2532 | **2/2** |
| 7 | −0.1657 | +0.0088 | +0.0083 | −0.0643 | **2/2** |
| 14 | −0.2525 | −0.1990 | −0.0114 | −0.5927 | 0/2 |
| 30 | +0.0694 | +0.0181 | +0.2247 | +0.0335 | **2/2** |

The two largest negatives in the table (−0.5927, −0.2532) are the feed-substitution day. **Two
anchors are replications, not evidence** — which is why this is a re-dispatch and not a re-reading
of the same numbers. The prior verdict is not being overturned here; it is being re-measured on a
set that did not exist when it was taken.

This is the same defect that cost the band series two published signs, and it has now been found
three times (`2026-07-09`, `2026-04-15`, `2026-03-10`).

## The anchor set — fixed now, six anchors

`2026-04-18, 2026-04-28, 2026-05-08, 2026-05-19, 2026-05-29, 2026-06-08`

**Selection rule, fixed before any C1 number was re-read:** the 52 dates that pass **both**
`audit_anchor_feed` and `cutovers_in_outcome_window` at all four horizons while holding one
collection regime (23,000–27,000 items/day, the ~25,010 era after the 2026-04-16 cutover) span
`2026-04-18 → 2026-06-08`; the six are that pool's **even sextiles in TIME**. Time, not volatility
or any other covariate, precisely because it cannot correlate with the outcome being measured.

⚠️ **Neither anchor from the prior read is in this set.** `2026-05-16` is in the clean pool and was
not selected by the spacing rule; `2026-06-16` is **not** clean — its h=30 outcome crosses the
2026-07-09..07-15 cutovers. Their prior values are a **prior, not a bar**, and must not be pooled
with the new ones.

⚠️ **Every clean anchor is a high-`sigma` date** (1.404–2.261× the panel norm). The archive holds
~5,200 items/day before March and ~25,010 after 04-16, so no set that reaches a low-volatility
quarter stays inside one collection regime. C1 is therefore measured on a high-volatility stretch
of 2026 and **nothing here licenses a claim about a calm market**.

## The basis, which is not optional

Every serving bar below is on the **TIED cohort** — the item-days where the raw anchor quote and
the smoothed anchor agree, so neither leg carries the `p[d]/S[d]` factor. This is the basis the
prior read used and the reason it was readable at all: on the pooled basis the label-denominator
wedge is **+0.14 rank IC**, larger than the effect being judged. `--basis-sweep` is already
unconditional in the workflow.

## Predictions and bars, before the run

**(A) Primary — serving.** Arm − control tied-cohort rank IC must be **positive at ≥ 4 of the 6
anchors**, at **≥ 3 of the 4 horizons**. Carried unchanged from the prior pre-registration:
**30d alone is not a pass** — its clean-anchor leg is the one never confirmed in CI and its embargo
exceeds its validation window.

**(B) Secondary — CV, a replication check.** The arm's `rank_ic_edge` must stay ≥ 0 at 4 of 4, and
the arm's CV rank IC must exceed the control's at ≥ 3 of 4. The prior read measured
**+0.0792 / +0.0801 / +0.0641 / +0.0509**; a CV result materially below that says the two dispatches
are not comparable, not that C1 changed.

**(C) Control reproducibility, the leg that makes one dispatch per arm readable.** The control's
tied serving IC reproduced a prior run to four decimals (+0.1321 / +0.1562 / +0.1747 / +0.0536 on
the old anchor set). The new control cannot be compared to those — different anchors — but the two
dispatches share a commit, a cache and a fold geometry, so **the control's CV rank IC must
reproduce the prior control's to ≤ 0.002 at every horizon**. If it does not, the pair is not
paired, and (A) is void.

**Pre-registered point prediction: 14d fails.** It was negative on **both** clean anchors of the
prior read (−0.1990, −0.0114) and 0/4 overall. If 14d passes here, that is a surprise and needs its
own explanation rather than a victory lap; if 3d, 7d and 30d pass and 14d fails, bar (A) is met at
3 of 4 and **C1 is un-shelved for a confirm run, not for shipping**.

## Void conditions

- Any anchor refused at run time by `audit_anchor_feed` or `cutovers_in_outcome_window` — the CI
  archive is ahead of the local copy and its cutover set may differ.
- The two runs land on different commits, different cached hyperparameters, or different fold
  counts.
- (C) fails: the controls do not reproduce.
- Reading the **pooled** basis instead of the tied cohort, or differencing an arm against a stored
  number from any earlier run.

## Cost

Two dispatches × 4 matrix jobs (one horizon each), 6 anchor replays per job. No code change, no new
instrument. `force_hp_search` is **not** set: hyperparameters were selected against the un-offset
target and are held fixed for this read, exactly as the prior one did — a positive would need a
confirm run before its size is believed.

## What a pass would and would not mean

A pass would mean the 2026-08-11 refutation was an artifact of two mis-collected anchors, and that
a within-date rank transform reaches serving after all. It would **not** ship C1: the open
mechanism question — why the transform gains +0.05 to +0.08 in CV and behaves differently at
serving — is untouched by this read, and the arm is known to be **worse on the deviating cohort in
13 of 16 cells**, which is two-thirds of the served universe.
