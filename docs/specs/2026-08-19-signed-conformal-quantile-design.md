# Design: the conformal band is two signed quantiles, not one absolute one

**Date:** 2026-08-19
**Decides:** replace the single absolute `q_hat` with a signed `(q_lo, q_hi)` pair, and remove
`_recenter_on_direction` from the serving path (the RANGE stance — chosen 2026-08-19)
**Evidence:** `research/2026-08-19-deep-model-review.md` (the width win and the biased-centre
mechanism), the `training-budget` rule's 2026-08-11 finding (the classifier recentring is
coverage-negligible), and `AGENTS.md` ("this is a RANGE forecaster, not a directional predictor")
**Status:** **implemented** 2026-08-19 (conformal.py + forecaster wiring + tests green); paired
serving replay pending

## Two corrections found during implementation

- **The `:7131` `_recenter_on_direction` call is NOT a serving path.** It lives in
  `_conformal_records` and only fires when `direction_class` is passed, i.e. under
  `CV_DIAGNOSTIC_CLASSIFIER` (off by default). Removing it would break that diagnostic, so it
  stays. The only serving removal is the one in `predict`.
- **The `recenter` REPLAY_DISABLE token was retired.** With `_recenter_on_direction` gone from
  `predict`, `REPLAY_DISABLABLE` dropped `"recenter"` (now `{"blend", "bias"}`) — a knob that
  parses but guards nothing reads as a clean attribution control and measures nothing.

## The change in one line

`conformal.calibrate` returns two signed quantiles of `res / scale` — at `α/2` and `1 − α/2` —
instead of one quantile of `|res| / scale`; `conformal.band` serves `mid + q_lo·scale` … `mid +
q_hi·scale`; and `_recenter_on_direction` is removed from `predict`, so the served sign comes from
the band itself, not a 3-class classifier.

## Why this is the free win, and what it is NOT

`calibrate` (`conformal.py:201`) folds the residual with `np.abs` before taking `q_hat`, so `band`
(`:312`) is symmetric by construction. When the residual distribution is **off-centre**, the
absolute quantile is dominated by the fatter tail and the symmetric band is inflated. Two signed
quantiles centre on the residual's own median and are **11–13% narrower** at the same coverage
(measured on prod-panel OOF records; direction robust, magnitude uncertain — the panel is
mixed-geometry post-2026-08-06 and 14 dates deep with h=30 contaminated).

**The off-centre residual is the q50's bias, not the classifier's.** `P(actual < mid)` =
0.60/0.65/0.71 because `DIRECTION_UPWEIGHT = 1.5` fits the q50 near the 58–60th percentile. The
default OOF calibration records are centred on that q50 (the classifier's `_recenter_on_direction`
only touches them under `CV_DIAGNOSTIC_CLASSIFIER`, off by default), so the 11–13% is a q50-centre
correction. This is why it does **not** contradict the 2026-08-11 rule: that finding measured the
*classifier* recentring toggle against *coverage* and found ≤1.1pp, because the serving shift is
bounded by 2·|mid| (median 0.95%) — an order of magnitude below a 10–31% half-width, and far too
small to move width by 11–13%.

So this spec is two coupled changes with different jobs:

1. **Signed quantiles** — the width win. Zero training cost (post-hoc on the existing q50's OOF
   residuals). NOT a fourth band-**width** scale: it changes the band's centre and shape, so the
   "three scales measured, width is not the lever" refutation does not apply. NOT CQR either: no
   p10/p90 boosters, so the 40–46 min OOF-conformal cost that killed CQR does not apply.
2. **Removing `_recenter_on_direction`** — coherence, not width. Coverage-negligible (≤1.1pp) on
   its own, but it **must** land with change 1: if the classifier still moves the mid *after* an
   asymmetric band is built, it drags the signed offsets with it and re-breaks the centre the band
   was just calibrated around.

**Out of scope (needs a retrain):** the durable upstream fix is `DIRECTION_UPWEIGHT → 1.0`, which
unbiases the q50 itself so the band has no bias to absorb. Signed quantiles absorb it for free
today; the retrain is a separate follow-up.

## The one invariant that makes this dangerous

**`(q_lo, q_hi)` is a matched set with `beta` and the scale, exactly as `q_hat` was** — see the
`SIGMA_EXPONENT` spec. `q_lo`/`q_hi` are dimensionless multipliers of `sigma ** beta` (or the
learned scale), so a pair served at the wrong exponent is wrong by ~5×, not partially. Consequences
the implementation must honour:

1. Every write persists `q_lo`, `q_hi`, and `beta` in one operation, or none.
2. An artifact with no `q_lo`/`q_hi` must default to the **symmetric** pair `(−q_hat, +q_hat)` —
   which reproduces today's band exactly — and must never default one while reading the other.
   Mirror the `conformal_beta` → 1.0 fallback and its WARN.
3. No stored `q_lo`/`q_hi` may be differenced across this change, across `beta`, or across
   `conformal_basis`/`conformal_centre`. Compare **coverage and WIDTH**, never a raw quantile. The
   audit line that prints `q_hat` must print the pair (and `beta`) beside it.
4. The finite-sample correction is two-sided: lower level `floor((n+1)·(α/2))/n`, upper level
   `ceil((n+1)·(1−α/2))/n`, so nominal interval coverage stays `1 − α`. Do not reuse the one-sided
   `ceil((n+1)(1−α))/n` for either leg.

## Call sites

`models/forecaster.py` unless noted. The pair has to reach every place `q_hat`/`band` does, or a
leg silently keeps the old symmetric geometry.

| # | Location | What it does | Change |
|---|---|---|---|
| 1 | `conformal.calibrate` `:179` | one abs quantile → `q_hat` | return `(q_lo, q_hi)`; keep a symmetric shim for the fallback |
| 2 | `conformal.band` `:292` | `mid ± q_hat·scale` | `mid + q_lo·scale` … `mid + q_hi·scale`; assert `q_lo ≤ q_hi` (no crossing) |
| 3 | `_calibrate_conformal` `:7555` | fits `q_hat` | fits and stores the pair; `:7611` `band` feeds `range_pct` → `_calibrate_confidence` (width only, formula unchanged) |
| 4 | `predict` `:8232` | builds the served band | serve the signed band |
| 5 | `predict` `:8303` | `_recenter_on_direction` | **remove** |
| 6 | serving path `:7131` | `_recenter_on_direction` | **remove** (the second serving call site) |
| 7 | `_calibrate_conformal` `:7581-7605` | sets `conformal_centre`, WARNs on the q50/classifier incoherence | centre is now unambiguously the calibrated q50; the WARN's condition (classifier exists) goes away with the removal |
| 8 | per-fold audit `:8935` | `calibrate(..., β=1.0)` | keep pinned at β=1.0; report the pair per fold if cheap, else keep scalar `fold_q_hat` for the published series |
| 9 | `to_meta` `:9680-9720` | persists `conformal_beta`/`centre`/`basis` | add `conformal_q_lo`/`conformal_q_hi` maps |
| 10 | `from_meta` `:9827-9880` | loads them | load the pair; absent → symmetric `(−q_hat, +q_hat)` + WARN |

Leave untouched and off: `SIGMA_EXPONENT`, `LEARNED_SCALE`, `EXCEEDANCE_SCALE`, `CONFORMAL_SERVED_BASIS`,
`LABEL_SMOOTHED_ANCHOR` — all orthogonal, all off by default. `_recenter_on_momentum` is already
unreachable (`MOMENTUM_FALLBACK_HORIZONS = []`).

## The classifier artifacts

`clf_*.txt` in `saved_models/` and `_fit_direction_classifier` become dead once the two serving
calls are gone. Do NOT delete the trainer in this change — leave it and the artifacts in place,
unwired, so the diff is "stop calling it," not "remove a subsystem." A later cleanup can drop them
once the range band is confirmed in prod. `CV_DIAGNOSTIC_CLASSIFIER` and its 932s cost are moot with
no serving consumer; note it, don't rip it out here.

## Validation

TDD on `conformal.py` first: a symmetric residual set must reproduce the old band to floating point
(guards the fallback); an off-centre set must yield `|q_lo| ≠ q_hi` and a narrower interval at the
same empirical coverage; the finite-sample levels must give exactly `1 − α` marginal coverage on the
calibration set by construction.

Then a **paired serving replay** (`scripts/replay_serving.py`): read `BAND COVERAGE` and **width**
on the arm vs control. Expect narrower width with coverage still ≥ nominal, and the removal moving
coverage ≤~1.1pp. **Confirm direction only** — the served panel is 14 dates, h=30 is contaminated
(replayed off a July frame), and geometry is mixed post-2026-08-06, so the 11–13% magnitude is not
pinnable here. Never difference a quantile across the arm; compare coverage and width.

Per repo convention: run the relevant `pytest` files, and land a dated `docs/changelog/` note.
