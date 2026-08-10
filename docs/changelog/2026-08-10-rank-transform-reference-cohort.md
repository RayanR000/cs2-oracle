# The rank transform ranks against the trained cohort, 2026-08-10

**Decision: sub-$1 items keep getting forecast rows.** The train/serve cohort mismatch recorded in
`2026-08-10-instrument-panel-first-read.md` is fixed by changing what the transform *ranks
against*, not by changing which items are forecast.

## The choice

Three ways to make serving match training, given training ranks 916 items and `predict` sees
5,536:

1. **Apply the $1 floor to the predict frame.** Exact, ~5 lines, and it stops writing a forecast
   row for 4,620 items.
2. **Persist a fixed per-feature quantile grid and map served values through it.** Robust to
   cohort drift, but it is *not the same transform* — a fixed grid is not a within-date rank, and
   the whole point of the instrument is that the reference distribution is re-formed each date.
3. **Rank every row against the cohort's within-date distribution.** Cohort rows get exactly the
   number training computed; the rest are placed inside that distribution.

**Taken: 3.** It is the only one that is simultaneously faithful to what the booster was fitted on
and free of a product change. 1 is faithful but pays for a modelling fix with data the product
does not have to give up; those 4,620 items are not served
(`api/serving_policy.py::MIN_SERVED_PRICE_USD = 1.0`) but they are written, scored, and cheap to
keep, and deleting a series is much harder to undo than not deleting it. 2 changes the estimand.

**What the sub-$1 rows are worth is unchanged by this.** They were out-of-cohort before and they
are out-of-cohort now — no booster in the artifact has ever seen a training row from one. This
keeps them on the same *scale* as the fitted features instead of a different one; it does not make
them trustworthy, and their forecasts should not be read as comparable to the served cohort's.

## What changed

- `_apply_cross_sectional_ranks(df, cols, reference_mask=None)`. `None` is the training path,
  unchanged and pinned by a test. With a mask, the ranks are computed on the masked rows and every
  other row is placed into that distribution by two-sided `searchsorted`, so a value tying with k
  cohort values lands mid-block — the same mid-rank convention the cohort rows get.
- Out-of-cohort values below or above the entire cohort **pin to −1 / +1**. There is no fitted
  support out there and the endpoint says so rather than pretending to interpolate.
- A date where the cohort has no observation of a characteristic yields **NaN**, not 0.0. Zero is
  the centre of this scale and would be indistinguishable from a genuine median rank.
- An empty reference cohort **raises**. Falling back to the pooled frame is the defect.
- `_reference_cohort_mask` selects by **item median over the frame**, matching
  `_filter_by_median_price`. A row-wise test would let one spike promote a penny item for a single
  date, so that date alone would rank over a population training never used.
- `save_models` records `train_min_median_price` and `train_cohort_items`; `load_models` reads
  them. `predict` **refuses to serve** a rank-transform artifact that records no floor — an
  artifact from before this change cannot say which cohort its percentiles are relative to, and
  guessing is the original bug.
- `predict` logs the served cohort against the trained count and warns past a 25% divergence. The
  defect was invisible for exactly as long as nothing counted.

## Tests

`tests/test_tier_lead_and_xs_rank.py`, 36 passed (+9). The load-bearing one is
`test_reference_rows_get_exactly_the_training_transform`: a served cohort row must receive the
number training computed for it, compared against the training path run on the cohort-only frame.
`test_out_of_cohort_rows_do_not_move_cohort_percentiles` asserts the fixture actually exercises
the defect, so it fails if someone makes the penny items harmless.

`test_predict_refuses_a_rank_artifact_that_records_no_cohort` is a **source-level** test —
`predict` needs a loaded booster and an archive. It is weak by construction and is there to fail
when the guard is deleted, which is the regression that matters.

## The refactor is training-neutral, measured

Run `31438314051` (`34091fd`, `xs_rank`, cached HP) reproduces the pre-refactor panel **exactly at
all four horizons** — rank IC 0.2489 / 0.2199 / 0.1827 / 0.1374, edge +0.0556 / +0.0561 / +0.0371
/ +0.0316, served PT excess 7.933 / 6.274 / 4.220 / 3.394pp, `32/32 features` transformed. Every
digit matches run `31430874845`.

That is the intended result and the only one this run could establish: CV folds are already
cohort-filtered, so they take the `reference_mask=None` branch and never exercise the new code.
The check is that the refactor did **not** leak into training, and it did not.

## The serving branch, exercised on real data (run `31440424106`)

`--predict-smoke` (added in the same session, see below) trains under the current flags and then
runs `predict()` writing nothing. All four horizon jobs report identically:

```
cross-sectional rank reference: 948/5,536 items at the artifact's >= $1 floor
cross-sectional rank: skipped 1 date-constant column(s) ['macd_missing']
cross-sectional rank transform applied to 31/32 features
predict-smoke: 5,536 rows, 5,536 items, 5,536 carrying a non-empty forecast dict
```

The mask resolves, all 5,536 items keep a forecast, and nothing is written. **948 against the 916
at training is expected drift, not a defect:** `predict` fetches `PREDICT_FETCH_DAYS = 730` while
training takes 1460 days, so the item medians are computed over different windows. At +3.5% it is
far inside the 25% divergence warning, which did not fire.

**The guard was also demonstrated, by accident.** The first attempt (run `31439896107`) restored a
cached artifact, skipped training on the age gate, and reached `predict` with the transform on
from the environment and no cohort in the restored meta. It refused to serve, with the intended
message. That is better evidence than the source-level test written for it — and it exposed a real
defect in the smoke mode, fixed in `5fd4892`: `predict_smoke` now forces training, because a mode
that predicts from a restored artifact is measuring the wrong model.

## Open: the skip set is re-derived at serve time, and it disagrees

**32/32 columns transformed at training, 31/32 at serving** — `macd_missing` is within-date
constant on the predict frame, so the date-constant skip fires there and not in training. The
column therefore reaches the booster **raw** at serve while it was **ranked** in fitting.

The same class of defect as the cohort mismatch, one column wide: a serving decision re-derived
from the served frame instead of following what training recorded. Today's impact is probably nil
and that is a coincidence rather than a design — training ranks an all-ties date to exactly 0.0,
and a raw all-zero flag is also 0.0, so the two agree **only while the flag's constant value is
zero**. Nothing measured which value it holds on the served frame.

The fix is the one already applied to the cohort floor: persist the training skip set in the
artifact and have serving follow it rather than re-deriving. **Not done.**

## Not done

- The skip-set inconsistency above.
- **No accuracy claim.** The smoke checks shape, not quality. Three of the four smoke jobs trained
  a single horizon beside restored boosters, and none of it says the served forecasts are good.
- `CROSS_SECTIONAL_RANK` stays off in production. What is established is that the serving path
  runs and ranks against the right population — not that shipping it improves anything served.
- The 25% divergence warning is a guess at a threshold, not a derived one.
- `train_cohort_items` counts items in the built training frame, which is post-floor and
  post-budget. On the current config the budget does not bind (916 items fit whole), so it equals
  the cohort — if the budget ever binds again, the two diverge and the warning reads low.
