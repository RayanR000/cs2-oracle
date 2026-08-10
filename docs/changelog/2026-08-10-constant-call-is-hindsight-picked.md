# `constant_call_accuracy` is an oracle, and the model-vs-baseline gap is much smaller than published

**Date:** 2026-08-10
**Corrects:** the "edge vs constant call" reading in
`docs/changelog/2026-08-10-post-revote-retrain.md`,
`docs/changelog/2026-08-10-served-classifier-scored.md` §1, and the "Follows" line of
`docs/changelog/2026-08-10-tier-lead-and-rank-transform-instruments.md`.
**Does not touch:** the `−return_1d` rank-IC gap, which uses no hindsight and stands unchanged.

## 1. What the number actually computes

`backtest/directional_test.py:153`:

```python
def constant_call_baseline(records):
    counts = Counter(r["actual_direction"] for r in records)
    direction, hits = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return direction, hits / len(records) * 100
```

It counts the **outcomes**, then returns whichever fixed call would have won. `_cv_evaluate_horizon`
calls it **per fold** (`forecaster.py:6528`), so the "best single fixed call" is re-chosen with
hindsight on every fold and the per-fold results are then averaged.

**The direction genuinely flips fold to fold**, so this is not a technicality. At 30d:

| fold | `constant_call_accuracy` | `realised_down_rate` | call it implies |
|---:|---:|---:|---|
| 1 | 91.8% | 7.4% | **up** |
| 2 | 77.1% | 19.9% | **up** |
| 3 | 72.9% | 72.9% | down |
| 4 | 86.5% | 86.5% | down |
| 5 | 52.3% | 52.3% | down |
| 6 | 56.6% | 56.6% | down |
| 7 | 55.7% | 55.7% | down |
| 8 | 59.0% | 36.5% | **up** |

Four of eight 30d folds went **up** on the majority of items. The baseline books 91.8% on fold 1 by
calling up and 86.5% on fold 4 by calling down. Averaging the per-fold maxima gives
`mean_constant_call_acc = 68.99%` against a `mean_realised_down_rate` of **48.48%** — the 20.5pp
difference is entirely the hindsight re-selection. **No strategy could have produced 68.99%**; it
requires knowing each window's market direction in advance, which is the forecasting problem, not a
baseline for it.

The same pattern holds at every horizon (3d: 3 of 9 folds up; 7d: 3 of 8; 14d: 4 of 8).

## 2. The runnable baseline, and the corrected comparison

The fixed call you can actually run is "always down", and its accuracy **is** `realised_down_rate`
by construction — you are right exactly when the outcome fell. Both are computed on the same
records under the same ±0.5% flat tolerance (`_direction_records` → `_direction_classes`), so the
identity is exact, not approximate.

Served classifier (run `31418286692`) against the runnable baseline (`meta.json`
`mean_realised_down_rate`, same fold set — bit-reproducibility of that fold set was established in
`2026-08-10-served-classifier-scored.md` §1):

| h | classifier (served) | always-down (runnable) | edge | **published "edge vs constant call"** |
|---|---:|---:|---:|---:|
| 3d | 49.6% | 46.06% | **+3.5pp** | +0.92pp |
| 7d | 48.8% | 48.73% | +0.1pp | −4.08pp |
| 14d | 49.3% | 50.56% | −1.3pp | −9.65pp |
| 30d | 52.9% | 48.48% | **+4.4pp** | −16.01pp |

**The published −16.01pp at 30d is against an oracle. Against anything runnable the served
classifier is +4.4pp there, and is a wash-to-modest-positive across the four horizons.**

⚠️ **The corrected edges are not a skill claim either.** Per-fold DA spread is 4.4–16.4pp
(`std_dir_acc`), so +3.5pp and +4.4pp are well inside one fold's noise, and no paired interval has
been computed on them. The tested claim remains the PT verdict — `skill` at all four horizons,
t = 13.18 / 8.00 / 5.40 / 4.31 on the served signal. This entry removes a false negative; it does
not add a positive.

## 3. Why the repo published it anyway, and what to change

**The invariant is right; the prose built on it was not.** `.claude/rules/backtest-scoring.md`
already records that "a constant call has per-date excess identically zero, so it resolves as
`degenerate`, never as skill", and that is exactly why the published headline is PT and not DA. The
triple `(DA, constant_call_accuracy, realised_down_rate)` was mandated *so that* a reader could see
this. The failure is that several changelogs then read the first two terms as a scoreboard and
ignored the third, producing statements like "the served classifier loses to a constant call at
7d/14d/30d" — which is true of the oracle and false of anything runnable.

`constant_call_accuracy` stays. It is the right quantity for its actual purpose: an upper bound on
what a direction-free call could extract, and therefore a measure of how much of a horizon's DA is
just the base rate. **What changes is that it must never be differenced against model DA.** The
runnable comparison is against `realised_down_rate`.

Fixed here:

- `docs/architecture/model.md` — `constant_call_accuracy` is now described as hindsight-selected,
  with the runnable baseline named.
- `backend/AGENTS.md` invariant #4 and `.claude/rules/backtest-scoring.md` — state which of the
  three terms is the runnable baseline, and forbid the difference.
- Correction banners on the three changelogs listed at the top.

`edge_vs_constant_call` and `edge_vs_constant_call_classifier` remain in `meta.json`. They are not
deleted, because a stored key that vanishes is worse than one whose meaning is documented — and
their magnitude is still the diagnostic for "how much of this horizon is base rate". They are
renamed nowhere and re-read everywhere.

## 4. A second reason no accuracy figure is quotable: `model_version` fragments the panel

`docs/README.md` records the 20-date gate failing and calls it "a calendar problem, not a code
problem". **That is half the cause.** `ops/item_forecasts.parquet` holds **6 distinct forecast dates
in total**, and they are split three ways:

| `model_version` | distinct dates | span | rows |
|---|---:|---|---:|
| `lgbm-v3-regime` | 3 | 2026-07-19 → 2026-08-05 | 91,696 |
| `lgbm-v3` | 2 | 2025-12-01 → 2026-07-17 | 44,336 |
| `lgbm-v3-global-only` | 1 | 2026-07-18 | 22,168 |

`score_cohort` keys on `model_version`, so run `31409508960` scored 110,615 frozen outcomes and
reported `NO HEADLINE (insufficient_dates)` on **every** horizon — 1–5 usable dates against
`MIN_FORECAST_DATES = 20`.

**`model_version` encodes the configuration, so every config change resets the panel to zero.**
`global-only` and `regime` are not different models in the sense the gate cares about; they are the
same artifact with `SKIP_REGIMES` flipped. Waiting for the calendar does not fix this — 20 more
daily runs produce 20 dates only if nothing about the config moves for 20 days, and
`SKIP_REGIMES=1` landed today.

This is now **F3** in `docs/research/2026-08-10-next-steps.md`. It is the binding constraint on ever
publishing a live number, and it is a code problem.

## Verification

`constant_call_baseline` read at source; the per-fold table above is `meta.json`'s own
`cv_results[h].per_fold`, unmodified. The always-down ≡ down-rate identity is definitional given
both are computed from the same `_direction_records` output. Forecast-date counts are a DuckDB
`count(distinct forecast_date) group by model_version` over `ops/item_forecasts.parquet`,
cross-checked against the per-cohort date counts logged by run `31409508960` (2 / 1 / 4 at 3d,
matching).

No code changed. No model changed. No retrain is implied by this entry.
