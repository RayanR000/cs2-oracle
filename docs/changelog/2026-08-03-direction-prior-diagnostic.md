# The Training Class Prior Is Balanced — the Down-Bias Is Not the Prior (2026-08-03)

`docs/specs/2026-08-03-direction-prior-correction-design.md` proposed
correcting the direction classifier's inherited up/down class prior at serve
time, gated behind a Step 0 diagnostic able to refute the hypothesis in one run.
**The diagnostic refuted it.** The correction was never built.

## The hypothesis

`_fit_direction_classifier` in `backend/models/forecaster.py` trains with
`objective="multiclass"` and no `class_weight` or `is_unbalance`, so the
classifier inherits whatever up/down skew its training window carried and
applies it unconditionally at serve time.
`docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md` had
observed the model predicting "down" 57–87% of the time regardless of forecast
date, without explaining why. The hypothesis under test: that bias is the
inherited prior.

## The measurement

`backend/scripts/diagnose_direction_prior.py` reads the persisted engineered
frame (`engineered_data.parquet`, 6,106,622 rows, 8,691 items,
2022-07-30 .. 2026-07-25), reproduces `prepare_targets`'s date-shifted forward
return, and computes the weighted training class prior per horizon via
`ItemForecaster._direction_class_prior`:

| horizon | pi_down | pi_flat | pi_up | down/up | finite targets |
|---|---|---|---|---|---|
| 3d | 0.4108 | 0.1921 | 0.3971 | 1.035 | 98.7% |
| 7d | 0.4199 | 0.1765 | 0.4036 | 1.040 | 98.1% |
| 14d | 0.4245 | 0.1649 | 0.4106 | 1.034 | 97.0% |
| 30d | 0.4287 | 0.1524 | 0.4189 | 1.023 | 95.5% |

The gate's threshold was `SKEW_RATIO_THRESHOLD = 1.20`. All four ratios fall in
1.02–1.04. **The prior is essentially balanced, so it cannot be the source of a
57–87% down rate** — producing that would need a ratio near 2.0 or above.

## Why the refutation is trustworthy

A review verified the numbers independently rather than reading the code:

- reimplemented the prior in DuckDB (a join, not a pandas merge) and reproduced
  all four ratios to four decimal places
- replicated the bulk classifier distribution by sampling 300k rows
- measured finite-target coverage at 95.5–98.7%, ruling out a small-subset
  artifact from a broken merge
- re-measured the ratio under every plausible variant of the training row
  universe — alive-items-only, either side of the time split, 2026-only —
  obtaining 0.98–1.07 in every case

The verdict is not universe-dependent. That matters because the frame the
diagnostic reads is the **predict-path feature cache**, not the exact frame the
classifiers trained on: training additionally applies dead-item filtering,
corrupt-item flagging, stratified subsampling, a time-based split, and a target
`dropna`, none of which this script reproduces. If the ratio had moved when the
row universe changed, that gap would have been a real objection. It did not.

## What this does NOT establish

The diagnostic does **not** refute the 57–87% down-rate observation. It
measures a different quantity.

- Production predicts the **latest row per item** (~8,700 rows) after
  `reindex(...)`, `replace([inf, -inf], nan)`, and median fill.
- The diagnostic predicts **all 6.1M historical rows**, in-sample, with raw
  NaN/±inf passed straight to the booster.

Its own bulk figures (3d: 36.5% down / 33.8% flat / 29.7% up; 30d: 32.8% /
31.7% / 35.5%) are therefore **not comparable** to the production rate, and the
script names those columns `bulk_*` with a printed caveat saying so.

So: the down-bias is real and still unexplained. What this diagnostic
establishes is only that it is **not attributable to the inherited class
prior**; it did not identify what does explain it, and it does not rule
anything else in, either. The remaining candidates are all serve-time and
untested against each other: `_recenter_on_direction`, `_recenter_on_momentum`,
`_blend_returns_with_prior`, and the two-date outcome cohort itself. That is
the open question. The natural next thread is the gap between the A/B harness's
reported directional accuracy and the live serving figures — `backend/AGENTS.md`
documents `walkforward_backtest.py` as scoring a different price consensus than
production (plain mean over duplicate item-days vs production's outlier-voted
median). This changelog does not claim that gap is the cause; it is the next
thing to look at.

No A/B measurement was run and no MDE was derived, because there was no
correction to measure.

## What shipped anyway

Two guards, which stand on their own merits and were designed to ship
regardless of the gate.

### 1. A forecast-date coverage guard on the outcome fit

`update_bias_corrections_from_outcomes` fits per-tier direction thresholds by
matching the predicted up/down/flat split to the observed base rate, and was
guarded only by row counts (`n < 20`, `MIN_THRESHOLD_SAMPLE = 100`). No row
count can distinguish 11,000 independent rows from 11,000 rows on two market
days, which is what the table holds. It now reuses `MIN_FORECAST_DATES` from
`backend/backtest/scoring.py` — the same constant that gates reporting — so the
value fitted on and the value reported cannot drift apart
(`ItemForecaster._has_date_coverage`). Below coverage it logs a `WARNING`
naming the distinct date count and refuses, leaving that tier's stored
thresholds unchanged and its `ewma_state` counter untouched so a later
well-covered fit starts cold.

### 2. A provenance schema version on `bias_corrections.json`

Every threshold in that file was fitted before the guard existed, on the
two-date cohort, and some sit at the `±3.0` clamp rail (horizon 30, `$1-5`:
`t_down = -3.00`, `t_up = -2.92`). The file is **gitignored**, so a committed
edit could not fix it — a hand-edit would not reproduce on CI, on prod, or on
another machine. `BIAS_FIT_SCHEMA_VERSION = 2` now causes unversioned
thresholds and their `ewma_state` to be discarded on load, which self-heals
everywhere. A malformed `schema_version` (`null`, a string, a list, a dict) is
treated as v0 and routed down the same discard path rather than raising —
deliberately chosen over broadening the exception tuple, because that would
have routed through `_set_default_bias_corrections()` and wiped `corrections`
too.

Guard 2 matters even though those thresholds are currently unreachable:
`predict()` reads them only when no direction classifier is loaded, and all
four `clf_{3,7,14,30}d.txt` exist — but the classifier loader catches a corrupt
classifier with a `logger.warning` and falls through to exactly these
thresholds, so one bad file would silently promote a rail-clamped fit to the
served path.

As a finding in its own right, this made the **wiring gap** explicit:
`update_bias_corrections_from_outcomes` is the only feedback loop from realised
outcomes to directional bias in the codebase, and it writes to a path
production does not read while a classifier is present.

## Tests

- `backend/tests/test_bias_fit_date_guard.py` — new, 14 tests: 6 for the
  coverage guard, 4 for the schema-version discard including the save round
  trip, and 4 parametrized malformed-`schema_version` cases (`null`, a
  non-numeric string, a list, a dict)
- `backend/tests/test_direction_prior.py` — new, 8 tests for the weighted class
  prior, including the three not-estimable paths (empty input, all-non-finite,
  zero total weight)

Full suite: **417 passed**, up from a 395 baseline.

## Files changed

- `backend/models/forecaster.py` — `BIAS_FIT_SCHEMA_VERSION`, schema-versioned
  load/save of `bias_corrections.json`, `_has_date_coverage` guard in
  `update_bias_corrections_from_outcomes`, and `_direction_class_prior`
- `backend/scripts/diagnose_direction_prior.py` — new, the Step 0 diagnostic
- `backend/tests/test_bias_fit_date_guard.py` — new
- `backend/tests/test_direction_prior.py` — new

## Related

- `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md` — the
  down-bias observation this tried and failed to explain, and the source of
  `MIN_FORECAST_DATES`
- `docs/specs/2026-08-03-direction-prior-correction-design.md` —
  the design, whose Step 0 gate fired
- `docs/research/accuracy-opportunities.md` — the closed feature/architecture
  roadmap this work stayed clear of
