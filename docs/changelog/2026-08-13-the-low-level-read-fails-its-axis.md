# The rescaling does not raise the band on calm dates, and the question was posed on the wrong axis

**Date:** 2026-08-13
**Pre-registered:** `docs/research/2026-08-13-low-level-anchor-preregistration.md`, committed and
pushed as `9e1e8a1` **before** any coverage was computed, including the anchor sets, the bars, the
placebo seed and the point predictions.
**Instrument:** `backend/scripts/attribute_band_level.py --low-legs` / `--select-low` (offline, ~2
min, writes nothing), 8 new tests in `tests/test_band_level_attribution.py` (25 total, 2142 in the
suite).
**Verdict:** ❌ **(L1) FAILS at 3 of 3 readable horizons** — and the leg that decides the arm's fate
is **(J)**, which passes decisively. **Nothing dispatched. No flag exists.**

## Verdict against the pre-registered bars

| leg | bar | h=3 | h=7 | h=30 | h=14 (voided) |
|---|---|---|---|---|---|
| (L1) direction | pooled coverage UP at ≥2 of 3 | ❌ −0.75 | ❌ −0.34 | ❌ −1.55 | −0.80 |
| (L2) calibration | mean \|cov−80\| falls at ≥2 of 3 | ✅ −1.24 | ✅ −1.68 | ✅ −2.58 | −1.10 |
| (L3) overshoot | no anchor thrown out of ±10pp | ✅ none | ✅ none | ✅ none | none |
| (P) placebo | shuffled \|mean Δ\| ≤ 1.0pp | ✅ +0.54 | ✅ +0.38 | ✅ −0.02 | −0.46 |
| (J) joint | mean \|cov−80\| falls at ≥2 of 3 | ✅ −2.85 | ✅ −2.84 | ✅ −2.44 | −1.38 |

## (L1) The arm lowers coverage on the calm set too — but not uniformly, and that is the finding

Control → arm, per anchor, `L[t]` as a multiple of the panel median:

| anchor | `L`/panel | h=3 | h=7 | h=30 |
|---|---:|---|---|---|
| 2024-09-21 | 0.787× | 90.56 → 87.68 | 94.52 → 92.59 | 93.61 → 87.22 |
| 2024-11-04 | 0.861× | 90.27 → 86.70 | 93.55 → 91.40 | 95.45 → 92.33 |
| 2024-12-04 | 0.859× | 71.67 → 68.13 | 87.89 → 85.61 | 86.83 → 83.29 |
| 2025-01-03 | **0.679×** | 70.14 → **72.68** | 67.32 → **70.69** | 78.11 → **79.94** |
| 2025-04-01 | 0.870× | 69.44 → 69.07 | 74.94 → 73.43 | 73.06 → 71.16 |
| 2025-07-12 | 0.822× | 76.64 → **78.99** | 71.34 → **73.23** | 71.55 → **74.03** |

The arm **raises** two anchors at all four horizons and lowers four. Pooled and row-weighted that
nets to −0.34 to −1.55pp, so the bar fails as written.

⚠️ **The failing statistic cannot resolve the effect it was asked about.** Its own pre-registered
placebo has a shuffled mean of **+0.54 / +0.38 / −0.02pp** against real effects of
**0.75 / 0.34 / 1.55pp** — at 7d the real effect is *smaller* than the shuffle's mean. **(L1) fails
by the letter and carries almost no information**, which is a defect in the bar and is recorded as
one: pooled marginal coverage over six anchors was the wrong quantity to put a primary bar on.

## The axis was wrong, and only 30d says what it was

**Post-hoc, and labelled as such.** `q_hat` is calibrated on an **expanding pool**, so the dose is
not `L[t]` against the panel median — the axis the set was selected on — but `L[t]` against the
median level of the anchor's *own* calibration window. On that axis most of the set is not calm at
all: 2024-09-21 sits at **0.921×** its own window and 2024-11-04 at **1.027×**.

Sign of the arm's move, predicted by each axis:

| axis | h=3 | h=7 | h=30 |
|---|---:|---:|---:|
| `L` vs panel median (the selected one) | 2/6 | 2/6 | 2/6 |
| `L` vs its own calibration window | 3/6 | 3/6 | **6/6** |

🔑 **At 30d the calibration-relative axis predicts every anchor's sign. At 3d and 7d it is a coin
flip**, and three anchors sitting within 1.3% of their own window's level (0.988×, 0.997×, 1.027×)
move −3.5pp anyway. So the per-date dose is the whole story at 30d and is **not** the whole story at
3d/7d, where rescaling also changes which rows dominate the calibration `p80`. **Do not carry the
30d result across.**

## (J) is the leg that decides, and it needed a placebo the pre-registration did not give it

**The pre-registered placebo was on pooled coverage; (L2) and (J) are the legs that passed, and
neither had one.** Same shuffle, same seed, on `mean |cov − 80|` — reported post-hoc:

| set | h=3 | h=7 | h=30 |
|---|---|---|---|
| (L2) low set, 6 anchors | −1.24pp, **27th pct** | −1.68pp, **17th pct** | −2.58pp, 5.5th pct |
| (J) joint, 11/11/9 anchors | −2.85pp, **0.0th pct** | −2.84pp, **0.5th pct** | −2.44pp, **0.0th pct** |

🔑 **(L2)'s pass is inside the noise and must not be quoted as a result** — on the calm set alone a
shuffled level reproduces the arm's calibration gain a quarter of the time at 3d. **(J) is outside
its null at 3 of 3**: across the full level range the arm cuts mean per-date miss by **2.4–2.9pp**
and no permutation of 200 matches it at 3d or 30d.

**So the arm's benefit is measurable only on a set that spans the volatility range, and not within
the calm end of it.** That is consistent with a level correction and it is *not* consistent with the
per-date mechanism the arm was originally pre-registered as — the third read in a row to say so.

## (L3) The overshoot the pre-registration expected did not happen

Prediction 3 was that (L3) was the leg most likely to fail: `gamma ≈ 0.70` is fitted panel-wide and
already overshot downward at high volatility (77.68% pooled at 30d). **No anchor within ±5pp of
target was thrown beyond ±10pp, at any horizon, on either leg.** Applying the arm at a 0.68–0.87×
dose is not destructive. That is the one clean positive here.

## (A) The single in-regime anchor, which carries no bar

`2026-02-19` — the **only** date of 183 in 2026 that sits in the panel's bottom quartile of `L[t]`
and passes both audits at all four horizons:

| | h=3 | h=7 | h=14 | h=30 |
|---|---|---|---|---|
| control → arm | 80.94 → 83.76 | 81.46 → 84.98 | 85.14 → 87.73 | **75.17 → 79.23** |

It rises at 4 of 4 — **toward** target at 30d and **away** from it at 3d/7d, from a control that was
already within 1.5pp. n = 1, no bar, fixed in advance; it is a replication and it is reported
because a reader will ask.

## What this closes, and what it does not

**Does not close open question 2.** The level-fix argument still has not observed the arm raising
the band on dates that are calm *relative to their own calibration window*, inside the served
collection regime — because the archive holds **one** such date. That is a property of the data, not
of the experiment: 2026's calm quarter (Q1, median `sigma` 0.0655) sits at the panel median
(0.0667), and every genuinely low-`L` date in the panel is pre-2026, where `source IS NULL` and the
collection regime differs from serving.

**Does add one thing the shipping argument can use:** (J), outside its null at 3 of 3, is the first
placebo-controlled evidence that one global `gamma` improves per-date calibration across the
volatility range rather than trading one end against the other. It is offline, on a stand-in panel,
across two collection regimes.

**What a next read would have to do differently.** Select on `L[t] / median L over the calibration
window`, not on `L[t]` against the panel — the axis the arm actually responds to, confirmed 6/6 at
30d and unconfirmed at 3d/7d. And put the primary bar on the per-date miss with its own placebo, not
on pooled marginal coverage, which six anchors cannot resolve.

## Standing caveats

- **No validity leg exists on the low set.** No served band was ever quoted on a 2024 date, so
  absolute coverage here is a stand-in and only control→arm differences are admissible. The transfer
  of the panel's 0.86–2.33pp agreement with the served control at 3/7/30d from the 2026 anchors to
  these is an **assumption**, stated in the pre-registration as one.
- **h=14 is excluded from every bar count**, per the prior read's leg (V) failing at 4.53pp MAE.
- `gamma` refit to **0.700 / 0.685 / 0.708 / 0.754**, reproducing the published values exactly, so
  no void condition fired.
- **The warm-up exclusion changed the set before any read.** The first selection pass returned
  `2024-07-09` — the panel's lowest `L[t]` by a factor of two, with **100% of its rows pinned at the
  sigma clip floor**, because `sigma` is `price_std_60d / price` and the panel's first 60 days hold
  a window that is not yet 60 days long. The rule added is the feature's own window, pinned by a
  test against the feature set.
