# Pre-registration: do the three serving transforms explain the CV→serving gap?

**Date:** 2026-08-13, written and committed **before** dispatch.
**Instrument:** `REPLAY_DISABLE=blend,bias,recenter` on `model-diagnostics.yml`'s replay step.
No code change. **No new question is opened if it fails** — the answer is a single bit either way.
**Opened by:** `docs/changelog/2026-08-13-c1-refutation-survives-the-audited-anchors.md`.

## The one bit

C1 gains **+0.0684 / +0.0759 / +0.0588 / +0.0408** rank IC in CV and reaches serving at only
**4/6, 3/6, 1/6, 1/6** anchors. Both cohort explanations are spent — the label denominator and the
anchor set. What is left between the two measurements is `predict()`, and exactly three transforms
in it move the mid: the **prior-day blend**, the **tier bias**, and **`_recenter_on_direction`**.
`ItemForecaster.REPLAY_DISABLABLE` names them; the conformal band is deliberately excluded because
it sets `low`/`high` around the mid and cannot move a cross-sectional ranking.

**Attributing which one costs 8 dispatches.** Establishing *whether any of them does* costs 2. This
runs the cheap question first: all three off, everything else identical.

## Design

Two dispatches on one commit, `REPLAY_DISABLE=blend,bias,recenter`, `horizons=matrix`, and the
**same six anchors** as the pair that just ran — `2026-04-18, 04-28, 05-08, 05-19, 05-29, 06-08`,
clean on both audits inside one collection regime.

The transforms-ON baseline is **already measured** (control `31663300447`, arm `31663312585`,
commit `ee76a9c`), so the comparison is across four runs and the new pair supplies only the OFF leg.
Both legs use the same anchors, the same cohort split and the same commit family; the control
reproduced a prior dispatch to four decimals, which is what makes one dispatch per cell readable.

## Two legs, and the second one is not about C1

**(1) Do the serving transforms cost the served signal at all?** Control OFF vs control ON, tied
cohort. **These three transforms have never been scored** — they run only inside `predict`, so no
CV number has ever seen them. This leg is production-relevant regardless of what C1 does.
**Bar: none, it is descriptive** — but the sign and size are pre-committed as the headline whatever
they are, so a null cannot be quietly dropped.

**(2) Do they explain C1's failure to transfer?** Arm − control tied-cohort delta with the
transforms OFF, against the same delta ON (4/6, 3/6, 1/6, 1/6 positive).
**Bar: positive at ≥ 4 of 6 anchors at ≥ 3 of 4 horizons** — the same bar C1 has been held to twice
— **and** strictly more positive anchors than the ON leg at ≥ 3 of 4 horizons. Both conditions, not
either.

## Predictions, before the run

**Pre-registered point prediction: leg (2) FAILS.** The transforms are applied to control and arm
alike, and a rank transform's CV gain is a property of the fitted model, not of a monotone
post-hoc adjustment to the mid — `_recenter_on_direction` and the tier bias are close to
rank-preserving within a date, and the prior-day blend mixes in a lagged mid that would have to be
*differentially* worse for the arm to carry the whole −0.05 to −0.08. If it fails, **the
CV→serving gap is not in `predict()`**, both remaining explanations are exhausted, and the next
place to look is the cohort geometry the two paths score over.

**Leg (1) is genuinely open.** If the transforms cost the control materially, that is a finding
about production worth more than C1.

## Void conditions

- The two runs land on different commits, caches or fold counts, or any anchor is refused at run
  time by either audit.
- Any replay cell whose `n` differs between control and arm (`pair_replay_rows` flags it).
- Reading the pooled basis rather than the tied cohort.
- Comparing an OFF-leg number against a stored ON-leg number from a *different* commit — the ON leg
  named above is the only admissible baseline.

## Cost

Two dispatches × 4 matrix jobs. The last pair ran 27–34 minutes per job (≈9 min train, ≈3.5 min per
anchor-horizon replay), so this fits the same envelope: the disable flag changes what the replay
computes, not how much.

## What a pass would and would not mean

A pass would locate the CV→serving gap inside `predict()` and justify the 6 further dispatches that
attribute it to one transform. It would **not** license removing a transform: each exists for a
reason CV cannot see, and a rank-IC gain from deleting one is an argument to measure its own job,
not to delete it.
