# The 2026-03-22 consensus break is archive-wide and pre-existing, not an import artifact

**Date:** 2026-08-09
**Status:** open finding. **No decision taken and no fix prescribed.**
**Found while:** validating the seam of the staged price-history import
(`2026-08-09-price-history-import-staged.md`) — the first framing, that the import created it,
was wrong.

**On 2026-03-22 the archive's own voted consensus steps down roughly 8–10% for every item whose
coverage until then was `aggregator_sync` alone. It is one shared date across the whole
cross-section.**

## The measurement, and why it is not about the import

Two groups, both measured against the staged `tracker_steam_24h` series only as a *reference
ruler*, and one measurement that uses no staged data at all:

| Group | n | first-consensus ÷ last-staged, median |
|---|---:|---:|
| Junction items (archive coverage begins where the staged series ends) | 4,169 | **0.9466** (−5.3%) |
| **Control — unbroken archive coverage, no junction at all** | 936 | **0.8953** (−10.5%) |

Junction distribution: p10 0.8318, p25 0.8837, p75 1.0192, p90 1.1235.

**The control items step harder than the junction items.** They have no handover. And the
archive's own one-day return across the break, computed with **no staged data whatsoever** —
consensus 03-22 ÷ consensus 03-21, control items, n=501 — is a median of **0.9159**.

Meanwhile the staged series' own 03-21 → 03-22 return is exactly **1.0000** for both groups.
**The break is entirely on the archive side.**

## Mechanism: three cash venues arrive at once and outvote the single Steam feed

Before 2026-03-22 these items' consensus was `aggregator_sync` alone — Steam-basis, and the
control confirms it prices at **0.9970×** the fee-corrected Steam series. On 2026-03-22
`aggregator_buff163`, `aggregator_youpin` and `aggregator_csfloat` all begin
(`docs/references/data-inventory.md` §5) at **0.891–0.905×** and outvote the one Steam feed
**3-to-1 in the median**. For junction items the first consensus is **99.94% cash-venue, 0.06%
Steam-basis** — 4,177 of 4,184 have zero Steam-basis contributor.

Two alternatives ruled out:

- **Not a frozen-price artifact.** Only **1.4%** of control items had an identical price on
  03-21 and 03-14.
- **Not a one-day step alone.** Staged ÷ consensus for control items drifts **1.008 (03-21) →
  1.103 (03-22) → 1.171 (03-31)**, so the level shift widens over the following nine days rather
  than settling.

## Why it matters

Every label whose window spans 2026-03-22 inherits a synthetic −8 to −10% move, for every item
with cash-venue coverage, **on one shared date**. At 7/14/30-day horizons that is a large share
of labels, and because the date is common to every item it will read as a **market factor**
rather than as noise.

This repo has already established that DA is dominated by the market date and that a constant
always-down call beats the model on every stored date
(`docs/changelog/2026-08-07-pesaran-timmermann-headline.md`). A synchronised synthetic drop is
the worst possible shape against that finding.

It is also the same defect class as `aggregator_buff163_buy` voting into consensus and as the
`steam_7d/30d/90d` trailing means voting against point-in-time asks: **sources on different
bases sharing one median.** The difference is that this one is a *transition* rather than a
steady state, so a read-time source filter does not neutralise it — removing the cash venues
would remove the post-03-22 series for most items entirely.

## Measurement caveat

The consensus used throughout is a **plain per-item-day median**, not production's
outlier-voted median with the ≥3-source 2σ mask
(`models/forecaster.py::_apply_multi_source_voting`). Production's magnitude may therefore
differ. **The sign will not**, because the vote is over the same three cash sources, and on
03-22 they are three of the four members of the group.

## Not done

- **No fix.** Not a source exclusion, not a per-source basis correction, not a label void
  window. Nothing here proposes one, and the finding is recorded so that a proposal can be
  argued against evidence rather than against an impression.
- **Not re-measured under production's voting rule.** See the caveat above; that is the first
  thing to run before sizing any remedy.
- **The label-side impact is not quantified.** How many stored labels span 2026-03-22, and what
  it does to DA on the ≥$1 served tier, is unmeasured. Do not assume it; the archive's 2026
  coverage has 90 missing days inside the buff163/youpin/csfloat span, so the count is not
  simply "every item × every horizon".
- **This is independent of whether the price-history import is ever promoted.** Excluding the
  junction items would not fix it — the control group, which has no junction, steps harder.

## Related

- `docs/changelog/2026-08-09-price-history-import-staged.md` — where this was found
- `docs/references/data-inventory.md` §5 — the source table showing the three venues starting
  2026-03-22
- `docs/changelog/2026-08-07-bid-source-excluded-from-voting.md` — the same class of defect,
  steady-state
- `docs/changelog/2026-08-07-pesaran-timmermann-headline.md` — why a shared-date synthetic move
  is the worst shape
- `docs/changelog/2026-07-20-hf-dataset-merge.md` — the merge that put the three venues on
  2026-03-22
