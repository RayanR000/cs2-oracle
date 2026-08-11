# The CV→serving gap is the anchor denominator, 2026-08-11

`--basis-sweep` scores the **same served mids** against four label bases at a fixed anchor.
Artifact, anchor, items, cohort and serving transforms are all held constant, so a difference
between rows is the label definition and nothing else.

| basis | numerator | denominator |
|---|---|---|
| `served` | median over (d, d+h] | smoothed anchor (`_smoothed_anchor_prices`) |
| `cv` | raw price on exactly d+h | raw price on d — `prepare_targets`' own basis |
| `num_only` | raw price on exactly d+h | smoothed anchor |
| `den_only` | median over (d, d+h] | raw price on d |

## The result

Δ rank IC against the served basis, n = 1,028–1,033 in every cell:

| h | anchor | `cv` (both legs) | `num_only` (outcome leg) | `den_only` (anchor leg) |
|---|---|---|---|---|
| 3 | 06-01 | +0.2346 | +0.0389 | **+0.2366** |
| 7 | 06-01 | +0.1558 | +0.0374 | **+0.1337** |
| 14 | 06-01 | +0.1590 | +0.0151 | **+0.1434** |
| 30 | 06-01 | +0.0758 | −0.0054 | **+0.0830** |
| 3 | 07-05 | +0.2908 | +0.0947 | **+0.2387** |
| 7 | 07-05 | +0.1196 | −0.0126 | **+0.1633** |
| 14 | 07-05 | +0.1048 | +0.0319 | **+0.0862** |
| 30 | 07-05 | +0.0306 | +0.0007 | **+0.0331** |

Means: **`cv` +0.1464, `den_only` +0.1398, `num_only` +0.0251.**

Swapping only the denominator — the smoothed anchor for the raw one — recovers **95%** of the
entire CV↔serving gap. The outcome leg is nearly irrelevant. And on the `cv` basis the served
forecasts score +0.03 to +0.19, straddling the control's own CV figures of 0.1794 / 0.1293 /
0.1015 / 0.0907.

**So the model's signal is not lost in the serving path.** It was never measured against a
denominator that the serving path uses. `2026-08-11-serving-transforms-are-not-the-gap.md` ruled
out the arithmetic; this rules in the label.

## The mechanism, as a hypothesis

`prepare_targets` divides by the **single raw quote at d**, and the features are built from that
same quote — `return_1d` most directly, but the price level and every ratio too. A spurious high
print at d therefore does two things at once: it raises `return_1d`, and it mechanically drives
the label negative, because the label divides by it. A model that reads `return_1d` is partly
predicting the denominator's own noise, which is not a tradeable return.

Three observations consistent with it, none of them proof:

- The effect is **largest at h=3** (+0.24) and shrinks monotonically with horizon (+0.08 at 30d),
  which is the shape of a fixed-size denominator artifact diluted by growing true return variance.
- `-return_1d` is the purest expression of that channel, and it is exactly the baseline the model
  "loses to" in CV at all four horizons (`model-loses-to-minus-return-1d`). On the served basis
  the naive baseline falls from about +0.19 to +0.07.
- The smoothed denominator does not remove the channel, it attenuates it: the median at d still
  contains the quote at d, one of three.

**This is an argument, not a measurement.** The direct test is a denominator that shares no quote
with any feature — score the same mids against `raw p[d+h] / raw p[d−1]`. If the IC stays high the
mechanism is wrong and smoothing itself is the cause; if it collapses toward the served number,
the shared quote is confirmed. One flag, one run.

## What this means for the work already done

- **C1's CV gain (+0.0686 / +0.0883 / +0.0755 / +0.0483) is measured on the inflated basis.** So
  is every arm this project has ranked, and so is the `-return_1d` bar all of them were held to.
  `2026-08-11-rank-transform-does-not-transfer-to-serving.md` reads differently in this light: the
  arm did not fail to transfer so much as it was selected against a quantity the product does not
  serve.
- **The `>= $1` served panel near zero is the honest number**, subject to the caveat below.
- Nothing here says the model is worthless. It says the metric that ranked the feature work is
  contaminated, and the size of the contamination (0.03–0.24 rank IC) is larger than every effect
  the project has been chasing.

## Limits

Two anchors, one artifact, no interval. The two anchors have opposite market signs, which is why
the consistency across them is worth something, but eight cells is not a sample. Nothing here has
been re-measured in CI. And the served basis has its own wedge — `2026-08-10-serving-replay.md`
sized it at up to 0.6 rank IC at h=3 for the naive baseline — so "the served number is honest"
means "less contaminated", not "clean". The `p[d−1]` test above is what would settle which of the
two is closer to a tradeable return.
