# The served classifier is measured, and invariant #4 now describes it

**Date:** 2026-08-10
**Follows:** `docs/changelog/2026-08-10-arm64-and-scheduled-diagnostics.md`, which built
`model-diagnostics.yml` for exactly this measurement.
**Run:** `31416199251` on `main`, first execution of the workflow, all four horizons green.

## 1. The measurement

`price-forecast.yml` sets `CV_DIAGNOSTIC_CLASSIFIER=0`, so until this run no offline number
had ever described the signal production serves. `predict` takes direction from
`self.direction_models[horizon]` (`forecaster.py:5679`); every DA in every training log to
date is the q50 sign.

| h | **classifier (served)** | quantile-sign | constant-call | classifier − constant |
|---|---|---|---|---|
| 3d | **49.6%** (≥$1: 49.6) | 39.6% | 48.68% | **+0.9pp** |
| 7d | **48.8%** (≥$1: 48.6) | 41.5% | 52.88% | −4.1pp |
| 14d | **49.3%** (≥$1: 49.1) | 43.0% | 58.95% | −9.7pp |
| 30d | **52.9%** (≥$1: 52.8) | 48.3% | 68.91% | −16.0pp |

**Two findings, and the first is good news.** The served classifier beats the quantile sign
by **4.6–10.0pp at every horizon**, so every historical DA figure understated production.
But it still **loses to a constant call at three of four horizons**; 3d is the only win, at
+0.9pp. Same shape as the q50 result, now on the signal that ships.

Rank IC reproduced the 2026-08-10 retrain exactly — 0.1774 / 0.1267 / 0.1023 / 0.0967
against naive `−return_1d` 0.1933 / 0.1638 / 0.1456 / 0.1058 — which is a useful control:
identical cached HP and folds give identical numbers.

**Cost:** 2:46 / 4:17 / 6:47 / 8:53 per horizon, ~9 min wall clock in parallel, against a
60-minute timeout. The "~16 min/horizon" estimate was ~2x pessimistic.

## 2. What the run exposed

The job scored classifier *accuracy* but reported "edge vs constant call" and the PT verdict
from the **quantile sign**, under a heading that read as the served signal. Verified
arithmetically: the logged −9.1 / −11.37 / −15.9 / −20.61pp is exactly
`quantile_sign − constant_call` at all four horizons, and the PT figures
(t = 12.14 / 7.03 / 3.077 / 3.35) reproduce the q50-sign numbers digit-for-digit.

**This was deliberate, not a bug.** `forecaster.py:4435-4441` chose the quantile sign so the
key could not change meaning between CI (classifier off) and local (classifier on), and
`invariant_4_signal: "quantile_sign"` recorded that honestly. The reasoning is sound: a
metric that silently describes a different signal depending on an environment variable is
worse than one that is stably the wrong signal.

So the fix **adds** the served verdicts rather than displacing the existing ones.

## 3. The change

`_direction_records_from_classes` builds PT records from the classifier's predicted classes
(`_direction_classes`: 0=down, 1=flat, 2=up). `_cv_evaluate_horizon` accumulates them in a
second stream and now returns a **4-tuple** — `(oof_records, fold_metrics, pt_records,
pt_records_clf)`.

`_train_horizon_inline` computes `pt_classifier` and `edge_vs_constant_call_classifier`
beside the existing pair, logs `Invariant #4 [quantile-sign]` and
`Invariant #4 [SERVED classifier]` as separate lines, and warns separately when the served
signal fails either bar.

**`mean_constant_call` and `mean_down_rate` are not recomputed per signal.** They are
properties of the outcomes alone, and the two record streams are built from the same
`actual_cls` under the same ±0.5% yardstick — so both signals are scored against an
identical bar. A test pins that the two streams agree on `actual_direction` row-for-row.

**No fallback, on purpose.** `pt_classifier` is `None` when `CV_DIAGNOSTIC_CLASSIFIER=0`,
visibly absent rather than substituted, and the log says so in words. A consumer wanting the
served verdict must read the served keys and handle the `None`; falling back to the q50 keys
is the exact defect this pair exists to prevent.

## 4. What this does *not* settle

**The naive-mid question is still open.** The classifier drives direction; the q50 drives
mid and band. Scoring the classifier says nothing about which mid is better — that needs a
mid-quality metric, and neither DA nor rank IC is one.

**Band coverage is still unmeasured**, so the `CV_STEP_DAYS` fold cut stays gated. The run
logs `q_hat` against `target coverage=80%` on the pooled OOF, which is ≥80% by construction —
the same unfalsifiable check D2 hit. The `coverage=95.0%` line is the binary `high_range`
threshold, a different quantity.

## 5. An unverified cost observation

Conformal CV summed to **634s with the classifier on** (105.2 + 193.2 + 49.8 + 285.8) against
**360.2s with it off** in arm64 retrain `31407938154`. That increment — roughly 200–280s
depending on how the skipped regime models are accounted — is far below the documented
"932s, 52% of a retrain", which was measured locally on x86.

If it holds, `CV_DIAGNOSTIC_CLASSIFIER=1` on the weekly retrain lands near a 21-minute job,
inside the 30-minute cap, and the separate scheduled job becomes unnecessary.

**Treat this as a hypothesis.** It compares across configurations (these jobs ran
`SKIP_REGIMES=1`, one horizon each, own data build) against a ±25% per-phase swing — the
comparison this project's own docs warn against. One controlled run settles it.

## 6. Confirmed in CI — run `31418286692`

The corrected logging ran on `fix/served-classifier-invariant-4`, all four horizons green.
Both lines emit, and the served PT is **computed**, not a re-print of the q50's:

| h | PT t [quantile-sign] | PT t [SERVED classifier] | classifier edge vs constant |
|---|---|---|---|
| 3d | 12.1396 | **13.1775** | **+0.92pp** |
| 7d | 7.0294 | **7.9998** | −4.08pp |
| 14d | 3.0770 | **5.3976** | −9.65pp |
| 30d | 3.3506 | **4.3058** | −16.01pp |

The hand-derived edges in §1 (+0.9 / −4.1 / −9.7 / −16.0) were right to rounding. The
served-side warnings fire at 7d/14d/30d and correctly stay silent at 3d.

**The finding nobody was looking for: the served signal's PT margin is larger at every
horizon, and most at the two that were called marginal** — 14d 3.077 → 5.398, 30d
3.351 → 4.306.

This bears on `CV_STEP_DAYS 150→300`, which was rejected on one ground: "fold count is also
the sample behind PT, whose t scales with √n", with 14d marginal at t=3.077. That was the
q50 sign. On the served signal, halving folds leaves 14d near 3.82.

**Not a green light.** 30d lands near 3.05 after the √2 penalty — at the hurdle, given the
q50's 30d read `no_skill` at 2.81 — and band coverage, the other gate on that lever, is
still unmeasured. The *input* to the decision has changed; the decision has not.

**It does not transfer to boost rounds.** Those tune the q50, so the q50's PT is the correct
metric there and the standing "do not cut rounds" verdict is unaffected.

## Verification

Full backend suite green: **1773 passed**, +4 over the previous 1769. The new tests cover the
class-encoding map, that the served stream is empty when the diagnostic is off and populated
when it is on, that both streams agree on outcomes, and a source-level guard that the served
verdict has no fallback to the quantile sign. Confirmed end-to-end in CI, above.
