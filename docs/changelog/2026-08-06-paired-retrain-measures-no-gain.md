# The paired harness ran: the three 2026-08-06 model fixes measure no accuracy gain

**Date:** 2026-08-06
**Change:** no code change. This is the measurement
`2026-08-06-scale-free-features-and-fabricated-labels.md` nominated as its open
next step ("the only thing that can turn 'wrong for a documented reason' into
'measurably better'"). It ran. The answer is **no detectable gain**, and two
side effects of the change were found that the original entry does not record.

## The result

Paired `classifier_accuracy_ge1`, control -> treatment, on folds matched by
identical `(train_start, train_end, val_start, val_end)`. Percentage points;
these are CV gate numbers, not production DA.

| horizon | folds | control | treatment | paired diff | paired sd | t | MDE(80%) |
|---|---|---|---|---|---|---|---|
| 3d | 8 | 50.12 | 48.74 | **−1.39** | 4.16 | −0.94 | 4.11 |
| 7d | 8 | 48.89 | 49.26 | **+0.38** | 3.40 | +0.31 | 3.37 |
| 14d | 8 | 51.56 | 50.89 | **−0.67** | 2.95 | −0.65 | 2.92 |
| 30d | 8 | 55.02 | 53.29 | **−1.74** | 3.56 | −1.38 | 3.52 |

Pooled `classifier_accuracy` on the same folds: **−0.29 / −0.29 / +0.21 / −1.16**
(sd 1.16 / 1.37 / 1.55 / 2.25).

No horizon clears its own measurement floor on either metric. Three of four point
estimates are negative on both. By the standard this project set for the
served-cohort test — within ±1pp means the mechanism is not the binding
constraint — this is the no-effect branch, with the caveat that **30d at −1.74pp
sits outside ±1pp and on the wrong side**.

`MDE(80%)` is `2.80 * sd / sqrt(n)` (two-sided α=0.05, normal approximation). At
8 folds and this variance the harness cannot resolve anything smaller than
**~3–4pp** on the served cohort, which is well above any effect this project has
ever produced (`2026-07-31-price-primitives-decision-scale.md`).

## The mechanism fix is confirmed

Dollar-denominated share of total gain, control -> treatment:

| horizon | control | treatment |
|---|---|---|
| 3d | 84.4% | **0.0%** |
| 7d | 85.8% | **0.0%** |
| 14d | 27.0% | **0.0%** |
| 30d | 32.7% | **0.0%** |

Top features move from `price_std_30d` / `price_std_60d` / `price_std_20d` to
`trend_up_fraction_30d` and the `price_cv_*` family. The defect is gone. It
simply did not pay.

**The original entry's headline shares do not reproduce like-for-like.** It
reported 55.6 / 70.2 / 77.5 / 86.6% "rising with horizon, in step with the
served-cohort accuracy gap", read off the deployed v3 artifact. A matched control
arm — pre-change behaviour on the current tree — measures 84.4 / 85.8 / 27.0 /
32.7%, i.e. *falling* with horizon, because `trend_up_fraction_30d` takes 60%+ of
gain at 14d and 30d on its own. The monotone-with-horizon pattern is an artifact
of comparing against a differently-trained artifact, not a property of the defect.
The defect was real; the argument that it explains the horizon trend is not
supported.

## Attribution: which fix did what

Each fix alone against the same control, `classifier_accuracy_ge1`:

| fix alone | 3d | 7d | 14d | 30d | paired sd |
|---|---|---|---|---|---|
| scale-free features | −0.99 | +0.73 | −1.16 | −0.95 | 3.0–4.1 |
| purged production split | **+0.00** | **+0.00** | −0.23 | −0.05 | 0–2.0 |
| label hygiene | +0.00 | +0.89 (t=2.04) | **−1.51 (t=−2.59)** | −0.72 | 1.2–2.0 |
| all three (treatment) | −1.39 | +0.38 | −0.67 | −1.74 | 3.0–4.2 |

**This harness structurally cannot measure the purge.** `classifier_accuracy_ge1`
is computed on CV folds, and `_compute_cv_splits(..., purge_days=horizon)` has
always purged — the leak was only ever in the production holdout split. The
purge reaches CV solely through the `tuned_params` the Optuna search hands it, so
at 3d (in `SKIP_HP_HORIZONS`) and at 7d the paired diff is **exactly 0.00**, and
the 14d/30d values are HP drift, not signal. What the purge actually fixes —
unleaked early stopping, unleaked HP selection, unleaked classifier stopping
points — is upstream of every number the CV gate produces. A different instrument
is required, and the original entry naming this harness as the measurement for
all three fixes is wrong on this one.

**Label hygiene is the only fix with any nominal signal, and it contradicts
itself:** +0.89pp at 7d (t=2.04) against −1.51pp at 14d (t=−2.59). Two nominal
hits in opposite directions across four horizons is what noise looks like at this
fold count. It is recorded because the 14d sign is negative, not because it is
believed.

**The scale-free swap is what makes the whole comparison unmeasurable.** Its
paired sd is 3.0–4.1pp against 0–2.0pp for the other two fixes: it changes the
model enough to move every fold substantially, in both directions. That variance
is inherited by the combined arm and is the reason nothing here resolves.

## Two side effects the original entry does not record

### 1. The label rule destroys the production validation window at 3d/7d/14d

The 30-day production validation window is a calendar window at the end of the
archive, and the voided days (the 2026-07-16 / 07-22 re-published snapshots and
the 2026-07-09/10 cutovers) sit **inside it**. The span rule then voids a
horizon-wide band around each cutover, so the damage grows with horizon:

| horizon | control val | treatment val after voiding | outcome |
|---|---|---|---|
| 3d | 2,445 rows | 1,683 rows / 17 dates | **thin-validation fallback** |
| 7d | 2,445 rows | 1,386 rows / 14 dates | **thin-validation fallback** |
| 14d | 2,445 rows | 990 rows / 10 dates | **thin-validation fallback** |
| 30d | 2,559 rows | 3,069 rows | date split holds |

At three of four horizons the run falls through to the positional 80/20 split:
3d becomes 90,860 train / 22,788 val against control's 112,411 / 2,445. That is
not a thinner version of the same thing — the validation set stops being the
recent 30 days and becomes the last 20% of the date-sorted frame, roughly the
last ten months. Early stopping, the Optuna objective and the classifier's
stopping set are all scored on that window instead.

Reproduced in the label-hygiene-only arm at identical row counts, so it is the
voiding that causes it, not the purge. Voided target counts are 916 / 1,444 /
2,295 / 4,231 at 3/7/14/30d.

This is a live consequence of code in the working tree, not a property of the
measurement.

### 2. The change costs a CV fold at 7d and 14d

Voiding labels at the end of the archive moves `tdf`'s max date, which reschedules
the expanding-window folds at the tail. Fold counts control -> treatment:
3d 9 -> 9 (but the last fold's window **shifts**, val `2026-06-07..07-06` becomes
`2026-06-10..07-17`), 7d 9 -> **8**, 14d 9 -> **8**, 30d 8 -> 8. Fewer folds is a
strictly worse measurement floor for everything measured after this change.

## Method, and why the pairing is trustworthy

Five arms, each a full cold retrain via `FORECAST_MODEL_DIR` into a scratch
directory, `run_forecast(train_only=True)`, `TRAIN_FEATURE_ROWS` at the
production default of 100,000, run from `backend/` (so the archive and the
`is_backfilled` item list come from the same place production reads).
`models/saved_models/` was **not touched** — the deployed artifact's mtime is
unchanged and its only git diff is the pre-existing one-line working-tree change.

Arms are constructed by turning the three fixes **off** on the *current* tree,
not by checking out `HEAD`. That is what makes this cleaner than the
served-cohort arms could be: the as-of lag change (`LAG_TOLERANCE_DAYS = 3`,
`MODEL_ARTIFACT_VERSION` 4) and the served-cohort code are present and identical
in **every** arm, so neither can confound the contrast. The
`2026-08-06-served-cohort-weighting-refuted.md` provenance caveat — "whether it
was present when the two arms ran is not determinable from the diff" — does not
apply here.

The off-switches, applied to `ItemForecaster` before `scripts.forecast_prices` is
imported:

* scale-free off: `SHELVED_FEATURES = (SHELVED_FEATURES - _DOLLAR_SCALE_FEATURES)
  | {price_cv_{7,14,20,30,60}d, macd_line_rel, macd_histogram_rel}`. The new
  scale-free columns must be shelved too — they did not exist pre-change, so
  un-shelving the dollar columns alone would produce an arm that is neither
  behaviour.
* purge off: `_purge_overlapping_train_rows` -> identity.
* label hygiene off: `_snapshot_dates` and `_collection_shift_dates` -> empty set.

Arms: `control` (all off), `treatment` (all on, = the shipped tree),
`scalefree`, `purge`, `labels` (one on each).

**Pairing was verified before any metric was read.** Identical in every arm: 99
of 5,378 items, 115,763 of 5,832,742 subsampled rows, sigma clip
floor=0.00956 cap=2.60632, and the winsorization counts (962 / 1,511 / 2,434 /
4,278). On every fold matched by date bounds, `n_train` and `n_val` are identical
between arms. Folds that exist in only one arm are **excluded**, which is why
every row above reads 8 folds rather than 9.

The feature sets differ as designed: control 36 columns, treatment 33. (The
original entry says 32. The >0.95 correlation prune is data-dependent, which that
entry documents as the reason to list shelved columns exhaustively; the count is
not stable and should not be quoted as one.)

**The control arm reproduces the previous harness's control**, which validates
the setup end to end:

| horizon | this control arm | `served-cohort-weighting-refuted` control | deployed `meta.json` |
|---|---|---|---|
| 3d | 50.1 | 50.10 | 49.9 |
| 7d | 48.7 | 48.74 | 49.0 |
| 14d | 51.5 | 50.81 | 51.1 |
| 30d | 55.0 | 55.02 | 53.4 |

A cold arm at this budget costs **110–150s**, not the ~20 minutes the original
entry estimated. That number was the reason the measurement kept being deferred.

## Decision

* **The fixes stay.** They are correctness fixes with a written-down mechanism: a
  dollar-denominated feature against a percentage target is wrong whatever CV
  says, a train/val split that leaks its labels is wrong, and a label built
  across a day the collector fabricated is wrong. None of that is contingent on
  an accuracy number.
* **No accuracy claim is available, and the earlier framing must change.**
  "The causes are fixed. The improvement is not proven." becomes: *measured, and
  no improvement was detectable at a 3–4pp floor; the point estimates are
  flat-to-negative.* `docs/research/2026-08-06-model-review-plain-english.md` carries the
  old wording and the "about a 20-minute job" estimate, both now superseded.
* **The validation-window collapse in side effect 1 is a defect to fix**, not a
  measurement artifact. Voiding a label is the right call; silently demoting
  three of four horizons to a positional split whose validation window is ten
  months long is not.

## What was deliberately not done

* **The harness runner was not committed.** It is a ~50-line monkeypatch script;
  the recipe above is sufficient to rebuild it, and the reasoning that kept
  `ab_test_direction_labels.py` does not obviously extend to a script whose only
  job is to disable code that should stay enabled.
* **No production retrain, and no change to `models/saved_models/`.**
* **The purge was not re-measured with an instrument that can see it.** Doing so
  means scoring the production holdout leg itself, which is a different harness.

## Still open

* **A higher-budget arm pair is running** at `TRAIN_FEATURE_ROWS=600000` (553
  items / 604,785 rows against 99 / 115,763) to cut the per-fold variance that
  makes the read above inconclusive. **Results are not in this entry yet.** At
  100K the served-cohort slice of a fold's validation set is only a few hundred
  rows, which is the direct cause of the 3–4pp floor.
* **Fix the thin-validation fallback interaction** (side effect 1).
* **The purge's effect is unmeasured** and needs an instrument that scores the
  production holdout leg.
* **The market-date domination of DA** remains the larger unexplained term —
  30d fold 1 reads 96.3% and fold 2 reads 20.2% in *both* arms, unchanged by any
  of this (`2026-08-03-accuracy-is-clustered-by-forecast-date.md`).

## Related

* `2026-08-06-scale-free-features-and-fabricated-labels.md` — the change measured
  here, and the entry whose "Still open" this closes.
* `2026-08-06-served-cohort-weighting-refuted.md` — the harness pattern and the
  `FORECAST_MODEL_DIR` override this reuses; its control arm is the reproduction
  check above.
* `2026-07-31-price-primitives-decision-scale.md` — why a 3–4pp floor makes an
  effect of this project's typical size unresolvable.
* `2026-08-03-accuracy-is-clustered-by-forecast-date.md` — why production DA was
  not available as the gate in the first place.
