# The clean-anchor signal replicates across ten anchors, 2026-08-11

> **Confirmed in CI on a fresh artifact and canonical data**, and one claim below is corrected:
> the deviating subset is negative at h=3 only, not at every horizon, and 30d's tied signal comes
> in at +0.0536 rather than +0.1423. `2026-08-11-clean-anchor-confirmed-in-ci.md`.

Ten anchors, 2026-04-15 to 2026-07-09, one artifact (local, trained 2026-08-09), same script and
flags throughout. Follows `2026-08-11-the-gap-is-the-anchor-denominator.md`, which found both
results on two anchors and asked for exactly this before either was believed.

## 1. The mechanism holds at 10 of 10

Δ rank IC (`cv` − `served`) across anchors, by subset:

| h | subset | mean | sd | t | anchors > 0 |
|---|---|---|---|---|---|
| 3 | tied | **−0.0066** | 0.0403 | −0.52 | 4/10 |
| 3 | deviating | **+0.4277** | 0.1764 | 7.67 | **10/10** |
| 7 | tied | +0.0081 | 0.0413 | 0.62 | 6/10 |
| 7 | deviating | +0.2703 | 0.2240 | 3.82 | **10/10** |
| 14 | tied | −0.0159 | 0.0560 | −0.90 | 4/10 |
| 14 | deviating | +0.2457 | 0.2306 | 3.37 | **10/10** |
| 30 | tied | +0.0015 | 0.0323 | 0.15 | 4/10 |
| 30 | deviating | +0.1304 | 0.1127 | 3.66 | **10/10** |

Where the anchor quote equals its own local median the gap is **indistinguishable from zero** at
every horizon — sign counts 4–6 of 10, which is a coin. Where it deviates the gap is positive at
**every anchor and every horizon**, 40 cells of 40. `prepare_targets`' denominator is the
contamination, and the contaminated cross-section is the two-thirds of items whose anchor quote
deviates from its local median.

## 2. The clean-anchor signal survives, smaller

Served rank IC — the full serving path, smoothed denominator, no basis contamination available:

| h | subset | mean | sd | t | anchors > 0 |
|---|---|---|---|---|---|
| 3 | tied | **+0.1246** | 0.0856 | 4.60 | 9/10 |
| 3 | deviating | **−0.2750** | 0.0942 | −9.23 | **0/10** |
| 7 | tied | **+0.1584** | 0.1152 | 4.35 | 8/10 |
| 7 | deviating | −0.0772 | 0.0961 | −2.54 | 3/10 |
| 14 | tied | **+0.1550** | 0.0992 | 4.94 | 9/10 |
| 14 | deviating | −0.0554 | 0.1522 | −1.15 | 5/10 |
| 30 | tied | **+0.1423** | 0.0928 | 4.85 | **10/10** |
| 30 | deviating | +0.0016 | 0.1334 | 0.04 | 7/10 |

It shrank from the 0.20–0.31 the first anchor pair suggested — as expected of a number found
while looking for something else — and settled at **+0.12 to +0.16 at all four horizons**,
positive at 8–10 anchors of 10. That is a real served signal on a third of the cohort, and it is
the first one this project has measured on the path that actually serves.

The deviating half at h=3 is the mirror image: **−0.2750, negative at 10 anchors of 10.** The
model is not noisy on those items, it is reliably anti-predictive. That is information about the
served price rather than about the model — the forecast is quoted against a smoothed anchor whose
spike the features read as a reversal signal that, on this basis, has already occurred.

## The overlap caveat, which bounds all of the above

**The anchors are 10 days apart and the horizons run to 30, so the outcome windows overlap.** Two
consecutive 30d anchors share two thirds of their window. The observations are therefore not
independent, the effective n is smaller than 10, and **every t above is inflated** — most at 30d,
least at 3d, where 10-day spacing exceeds the horizon and the windows are disjoint.

This is the same defect as `paired-mde-was-date-clustered`, and the same rule applies: read the
**h=3 row**, where the design is clean, and treat 7/14/30 as directionally consistent rather than
independently significant. At h=3 the tied signal is +0.1246 at 9 of 10 disjoint anchors and the
deviating signal is −0.2750 at 10 of 10. Those two rows carry the result.

## Two data notes

- **The tied share is 27–43% of the scored cross-section** at nine anchors — so the clean subset
  is a third of the served cohort, not a rump.
- **2026-07-09 is an outlier: 26 tied of 669 (3.9%).** Something about that date's collection
  makes almost every anchor quote deviate from its local median. It was left in; excluding it
  would strengthen the tied-row sign counts, which is reason enough not to.

## What follows

1. **The CV metric needs rebuilding before it ranks anything else.** Every arm in the project,
   including C1, and the `−return_1d` bar they were held to, were scored on the contaminated
   denominator. The fix is a label whose denominator is the smoothed anchor `predict` actually
   quotes. ~~which is `den_only` in the sweep, already implemented.~~ **Corrected 2026-08-11:
   `den_only` is the sweep arm that KEEPS the raw anchor** (`out_med / anchor_raw`,
   `replay_serving.py:203-207`) and it is the one that retained +0.1398 of the gap. The
   smoothed-denominator arm is `num_only`. Shipped gated as `LABEL_SMOOTHED_ANCHOR=1`:
   `2026-08-11-label-smoothed-anchor.md`.
2. **A serving policy gate on anchor cleanliness is now supported by evidence**, where before it
   would have been a guess. Serving the deviating two-thirds is publishing forecasts with rank IC
   −0.28 at h=3.
3. Neither should be built on ten local anchors and one artifact. The replay runs in CI
   (`replay_anchors`), and this wants a fresh artifact and anchors that do not overlap.
