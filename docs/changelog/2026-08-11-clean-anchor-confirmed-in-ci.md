# The clean-anchor result confirmed in CI, 2026-08-11

Run `31456524146` on `bb1a01a`: a **fresh artifact trained in CI on the canonical archive**,
replayed at four anchors chosen so the outcome windows do not overlap — 2026-04-15, 05-16, 06-16,
07-09. Independent of the local archive copy and the 2026-08-09 artifact that produced
`2026-08-11-clean-anchor-signal-replicates.md`.

## The mechanism: 16 of 16

Δ rank IC (`cv` − `served`), mean across the four anchors:

| h | tied | deviating | deviating, anchors > 0 |
|---|---|---|---|
| 3 | −0.0206 | **+0.4260** | 4/4 |
| 7 | −0.0050 | **+0.3692** | 4/4 |
| 14 | −0.0137 | **+0.3025** | 4/4 |
| 30 | +0.0208 | **+0.1945** | 4/4 |

Tied is zero to within ±0.021 at every horizon; deviating is positive in **all sixteen cells**.
The h=3 deviating figure lands at **+0.4260 against the local +0.4277** — three decimals, across
different archives, different artifacts, and non-overlapping windows. The contaminated denominator
is a property of the label, not of any one dataset.

## The signal: holds at 3/7/14, weak at 30

Served rank IC on the clean subset, mean across anchors, with the local ten-anchor figure beside
it:

| h | CI (4 anchors) | positive | local (10 anchors) |
|---|---|---|---|
| 3 | **+0.1321** | 4/4 | +0.1246 |
| 7 | **+0.1562** | 4/4 | +0.1584 |
| 14 | **+0.1747** | 4/4 | +0.1550 |
| 30 | +0.0536 | 3/4 | +0.1423 |

Three of four horizons reproduce closely. **30d does not** — it comes in at a third of the local
figure and is the only horizon with a negative anchor. 30d is also the horizon whose windows
still overlap slightly here and whose embargo exceeds its validation window
(`labels-and-embargo`), so treat +0.0536 as the honest number and the local +0.1423 as
optimistic.

## Where the earlier write-up overstated

`2026-08-11-clean-anchor-signal-replicates.md` called the deviating subset "reliably
anti-predictive" at −0.2750, 0 of 10. In CI it is **−0.2014 at 0 of 4 for h=3 — but only h=3**.
At 7/14/30 the CI means are −0.0529 / −0.1009 / −0.0563 with mixed signs, and dropping the
2026-07-09 anchor moves them to **+0.0083 / −0.0152 / +0.0379** — indistinguishable from zero.

**2026-07-09 carries a disproportionate share of the effect.** It has 26 tied items of 669 (3.9%,
against 27–56% elsewhere) and produces the four largest deviating gaps in the run (+0.757 /
+0.885 / +0.855 / +0.437). Excluding it, the mechanism still holds at every horizon — deviating
gaps of +0.3156 / +0.1974 / +0.1182 / +0.1136, all positive — but the claim that the deviating
subset is *actively wrong* survives only at h=3.

Corrected claim: **the deviating subset carries no usable signal, and at h=3 it is negative.**
That is enough to justify not serving it; it is not the "reliably backwards at every horizon"
reading.

## Status of the two conclusions

1. **The CV label's denominator is contaminated** — confirmed on canonical data, 16 cells of 16,
   reproducing the local magnitude. Ready to act on.
2. **A clean-anchor subset carries served signal near +0.13 to +0.17 at 3/7/14d** — confirmed at
   4 of 4 anchors on three horizons, unconfirmed at 30d.

Neither has a confidence interval. Four anchors and ten anchors are replications, not power: the
readable statistic is that the sign is consistent, not that the magnitude is pinned.
