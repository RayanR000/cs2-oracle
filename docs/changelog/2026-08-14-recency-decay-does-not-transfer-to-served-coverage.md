# Recency decay is CV-positive and placebo-clean at 30d, but does NOT transfer to served coverage — do not ship

**Date:** 2026-08-14
**Item:** closes the recency-decay finish-out from
`docs/changelog/2026-08-14-recency-decay-ships-at-30d-honestly.md`, which shipped the CV read but
left three things open: a placebo, the 3/7/14d legs, and the served-coverage read. The placebo is
now run and the served read settles it.
**Verdict:** ❌ **Keep `SAMPLE_WEIGHT_HALFLIFE_DAYS = 0` (off).** The 30d gain is real in CV and
survives a permuted-recency placebo, but it is a ~0.2% band-width narrowing that moves served
coverage by ≈−0.25pp against a ~7pp over-coverage — it does not transfer. Same CV+/serving− shape
as every other band-width lever.

## What ran

1. **Local paired+placebo A/B** (`scripts/ab_test_recency_weights.py --horizon 30`) on today's
   production frame (`build_training_data(days_back=1460, backfilled_only=True,
   max_feature_rows=1_200_000, min_median_price=1.0)` → 982,173 rows / 915 ≥$1 items / 33 served
   cols — reproduces the earlier run exactly). Added a third **placebo** arm: flat's vol/direction
   weights with the decay recency-multiplier (`decay/flat` ratio) **permuted across rows per fold**
   (fixed seed), preserving decay's weight distribution while breaking its alignment to age.
2. **Paired CI served-coverage read** — `model-diagnostics.yml` dispatched twice on the same commit
   (`2e5b009`, branch `lambdarank-serving-transfer`), control `recency_halflife=0` (run
   `31839757908`) vs arm `recency_halflife=365` (run `31839759435`), `replay_anchors=2026-05-08,
   2026-05-29`, on the **durable** archive. Recency was wired in as an env-overridable arm for this
   (see "Wiring" below).

## Result — CV passes (with placebo), serving does not

**CV (local harness, 8 folds, honest trainer, fixed 750 rounds):**

```
30    flat     7.92658
30   decay     7.81878
30 placebo     7.92951
-> 30d: q50 pinball +1.44% [PASS] | paired -0.108 [-0.192, -0.026] positive [PASS] | DA +0.36pp [PASS] | folds 6/8
   PLACEBO    +0.24% paired +0.003 [-0.070, +0.106] null [null=PASS]
   VERDICT 30d: SHIP decay
```

The placebo is the new information: real decay is `-0.108 [-0.192, -0.026]` (below zero), the
permuted-recency control is `+0.003 [-0.070, +0.106]` (straddles zero). So the +1.44% is recency
**structure**, not weight variance and not a fold-count-floor artifact — which was the open
suspicion (`ab-fold-count-floor`: at ~8 folds a permuted placebo can read positive on its own; here
it does not).

**Served band coverage at 30d (durable archive, arm − control):**

| anchor | control cov% | arm cov% | Δcov | control halfw% | arm halfw% | Δhalfw |
|---|---|---|---|---|---|---|
| 2026-05-08 | 87.69 | 87.78 | **+0.09** | 36.26 | 36.34 | +0.08 |
| 2026-05-29 | 87.11 | 86.53 | **−0.58** | 29.66 | 29.30 | −0.36 |

Mean Δcov ≈ **−0.25pp**, mixed-sign across the two anchors, against a band that over-covers by
**~7pp** (87 → 80 target). The CV OOF calibration barely moved either: `q_hat` 280.07 → 277.10,
median half-width 21.94% → 21.77% of mid. The half-width narrows ~0.8% relative — real, tiny, and
nowhere near what closing the over-coverage needs.

## Why this is the expected shape, not a surprise

Recency decay is a **band/pinball** change (DA is +0.36pp, flat), and every band-width lever tried
before it — `SIGMA_EXPONENT`, `LEARNED_SCALE`, the expanding-window `q_hat`, date-conditional
`q_hat` — improved out-of-fold and **failed to move the served band**
(`band-over-covers-basis-unconfirmed`; the `.claude/rules/training-budget.md` "three scales, all
calibrate to exactly 80% on their own records, land elsewhere when served" entry). The OOF residuals
the pinball gain is measured on are a different population from the served outcomes, and a ~1% CV
pinball improvement does not carry the ~7pp of served over-coverage. Recency joins that column.

## Decision

- **`SAMPLE_WEIGHT_HALFLIFE_DAYS` stays 0 (off).** No accuracy claim survives (DA flat), and the
  band claim fails at serving.
- The finish-out is **complete**: the placebo ruled out the fold-floor artifact, and the served
  read ruled out transfer. There is no remaining unmeasured leg that would flip this — the 3/7/14d
  CV legs were dropped as not decision-relevant (the effect was always 30d-specific, and the ship
  bar is serving, not CV breadth).

## Wiring (kept, off by default)

Landed with this work and left in place as an off-by-default diagnostic arm, exactly like the other
refuted band levers:

- `forecaster.py:164` — `SAMPLE_WEIGHT_HALFLIFE_DAYS = float(os.environ.get(...) or "0.0")`.
  **Training-only**: the decay is baked into the fitted booster, so `predict`/replay read it for
  free — no `meta.json` field, no serving env to keep in sync. `ab_test_recency_weights` still
  monkeypatches the module attribute directly.
- `model-diagnostics.yml` — `recency_halflife` dispatch input + a training-step env mapping (the
  `|| '0'` fallback matters: an empty string would raise through the `float(... or "0.0")` read).
  Not set in the replay step by design.
- `ab_test_recency_weights.py` — the permuted-weight `placebo` arm; ship now requires decay to beat
  the placebo, not just flat.

`test_forecaster.py` 155 passed. CI runs `31839757908` / `31839759435` both green.
