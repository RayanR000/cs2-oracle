# 2026-08-20 — Regime-reactive climatology band scale (built, gated off)

## Why

The climatology band scale (`CLIMATOLOGY_SCALE`, default-on since 2026-08-19) sets one
dispersion **level** per item and never moves it. The served-basis validation showed the
consequence: its width edge is **regime-dependent** — 26–47% narrower than sigma on volatile
windows, but *wider* than sigma on calm ones, because a static scale cannot react while
`price_std_60d`-derived sigma tracks recent vol. The width variable *is* the product (the
centre moves the served interval <5% of its own width), so restoring regime reactivity on top
of climatology's better level is the highest-ROI remaining scale change.

This is the review's §9 #3 ("react faster to regime shifts"), implemented as a **modifier of
the climatology scale** rather than a new denominator — so it does **not** reopen the "do not
propose a fourth band-width scale" rule, which is about *alternative* denominators.

## What

`CLIMATOLOGY_REACTIVE=1` (off by default; only `"1"` enables). When on, each row's static
climatology scale is multiplied by

    clip(sigma_row / baseline_sigma_item, CLIMATOLOGY_REACTIVE_LO, CLIMATOLOGY_REACTIVE_HI)

where `sigma` is the trailing daily-return dispersion the band already carries and
`baseline_sigma_item` is the item's **median sigma over the calibration window**. On a
normal-vol day the ratio is 1.0 and the served scale is **byte-identical to static
climatology**; it widens when current vol exceeds the item's norm and narrows when below,
clipped to [0.5, 2.0] so one wild quote can't blow the band up nor a stale-flat run collapse it.

Climatology keeps setting the cross-item **level**; the multiplier only supplies within-item
**timing**. An unseen item uses the global baseline (multiplier 1.0 at the universe-median
sigma).

## Matched pair

The multiplier is folded into the scale at **fit** time (`_fit_climatology_scale`), so `q_hat`
is calibrated against the modulated scale. Serving follows the **artifact**
(`_climatology_reactive_served` → `_artifact_climatology_reactive`), never the environment, so
the cutover is atomic at the next retrain. `meta.json` carries `climatology_reactive` and the
per-item `baseline_sigma` / `baseline_global` inside `climatology_scale_tables`. An artifact
without a baseline table makes the multiplier a silent no-op (older climatology artifacts, or
the arm off at fit time). **Never difference a `q_hat` across this flag — compare coverage and
WIDTH.**

It is a modifier: with `CLIMATOLOGY_SCALE=0` it has no effect (the climatology branch never
runs), so it does not compose with the sigma-exponent / learned-scale / exceedance denominators
and needs no mutual-exclusion guard.

## Code

`backend/models/forecaster.py`: `climatology_reactive_enabled()`,
`_climatology_reactive_served()`, `_climatology_reactive_multiplier()`, class constants
`CLIMATOLOGY_REACTIVE_LO/HI`, baseline capture in `_fit_climatology_scale`, modulation in
`band_scale`, meta save/load. Tests: `backend/tests/test_climatology_reactive.py` (11).

## Status

BUILT, gated off, byte-identical with the flag off. ❌ **REFUTED AT SERVING (2026-08-20) — do
not ship.**

### The calm-anchor A/B

Local paired replay, both artifacts trained fresh on the same archive from the same HP cache
(per-fold audit `q_hat` matched across arms, so the boosters are shared and the effect is
isolated to the calibration). Control = static climatology, arm = `CLIMATOLOGY_REACTIVE=1`.
Anchors chosen by a market realized-vol series (`backend/scripts/pick_calm_anchors.py`) and
feed-audited: CALM 2026-04-07 / 06-16 / 07-01, VOLATILE 2026-03-24 / 03-27. Read with
`backend/scripts/replay_reactive_ab.py` (BAND COVERAGE width + coverage; never difference q_hat).

Width ratio arm/ctrl and coverage Δ (target 80%):

| | h=3 | h=7 | h=14 |
|---|---|---|---|
| CALM width | ×1.157 | ×1.156 | ×1.028 |
| CALM cov Δ | +2.4pp | +3.9pp | +2.0pp |
| VOLATILE width | ×1.077 | ×1.083 | ×1.015 |
| VOLATILE cov Δ | +3.4pp | +4.5pp | +3.3pp |

The arm does the **opposite** of its design. It **widens** the band nearly everywhere (multiplier
>1 at serve) and pushes coverage **further above** the already-over-covering control — and it is
*widest on the calmest anchor* (2026-04-07 h=3 ×1.29), the exact reverse of the intent.

### Root cause (two compounding defects)

1. **The serve-time `sigma` is a lagging 60-day rolling std.** On a calm day right after a
   volatile stretch (2026-04-07 sits just after the March vol), the 60d window still contains the
   volatility, so `sigma` reads HIGH and the multiplier widens exactly when the market is calm.
   The signal points the wrong way at the moment it is supposed to help. This is the row-based /
   slow-window defect flagged in the original caveat, now shown to actively break the arm.
2. **The baseline is the item's median `sigma` over the 2023–2025 calibration folds**, so the
   ratio measures current-vs-multi-year-ago vol (a secular regime shift) rather than
   current-vs-recent-norm. 2026 runs hotter than the calibration era, so the multiplier is
   ~uniformly >1 and never isolates calm days.

Same offline-plausible / serving-negative fingerprint as the refuted band-width levers, with a
specific mechanism.

## v2 — the fix, and it WORKS (2026-08-20)

Both v1 defects are replaced by one self-normalising, **date-aware** pair of EWMA vols engineered
in `engineer_features` (`ewm_reactive_fast`/`ewm_reactive_slow`, shelved so the boosters never see
them): a short-halflife (9d) EWMA of `|return_1d|` over a long-halflife (45d) one. The multiplier
is `clip(fast / slow, 0.5, 2.0)` — 1.0 when recent vol equals the item's own recent norm, <1 on a
genuinely calm day, >1 on a spike. No lagging 60d sigma, no calibration-era baseline; the
multiplier is now **stateless** (reads two columns). `CLIMATOLOGY_REACTIVE_FAST_HL/SLOW_HL` are
env-free class constants. Tests: `tests/test_climatology_reactive.py` rewritten (12).

Same paired local A/B (fresh ctrl/arm on one archive + HP; per-fold audit q_hat matched), same
anchors, corrected parser (real `halfw%`, column 7 — the v1 table above had mis-read column 5):

| | h=3 | h=7 | h=14 |
|---|---|---|---|
| CALM width | ×0.815 | ×0.805 | ×0.860 |
| CALM cov Δ | −2.6pp | −3.2pp | −6.2pp |
| VOLATILE width | ×0.908 | ×0.936 | ×0.942 |
| VOLATILE cov Δ | +1.4pp | +1.8pp | +1.8pp |

The arm now does the right thing: it **narrows calm-day bands ~19%** (toward the 80% target, off
the over-covering 90–95%) and **barely touches volatile days** (~8% narrower, coverage neutral) —
it narrows *more* on calm than volatile, the exact reverse of v1. On the calmest anchor
(2026-04-07) it cuts h=3 width 8.28→6.52 while still covering 94%, so that narrowing is free.

⚠️ **Caveats.** Not uniformly free: where control already sat near 80% (2026-06-16) the arm dips
slightly under (h=7 77.4%, h=14 71.8% — the one over-narrowed cell; clip LO / slow halflife are the
knobs). Underpowered (3 calm / 2 volatile anchors) and on the local-fixture archive (ranking
robust, absolute widths differ from prod, per the climatology precedent). Still gated OFF and
byte-identical with the flag off.

**Status: direction-correct, positive — promote to a PROD A/B before shipping.** Next: add a
`climatology_reactive` arm to `model-diagnostics.yml` and read width-at-matched-coverage on the
served panel; if it holds, ship default-off→on like the climatology scale, with
`CLIMATOLOGY_SERVING_START`-style geometry guard for the served-coverage feedback.
