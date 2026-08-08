# The published headline is a significance test, not a directional accuracy

**Date:** 2026-08-07
**Step:** 2 of `docs/research/2026-08-07-next-steps.md`. No retrain.
**Status:** landed. 1,258 backend tests green (was 1,233); frontend `tsc --noEmit` clean,
`npm run lint` 0 errors.

## What was wrong

Directional accuracy was the headline, and on this data it is not a claim about the model.

`2026-08-03-accuracy-is-clustered-by-forecast-date.md` measured an always-down constant call
scoring **29.4% on 2025-12-01 and 76.9% on 2026-07-17** at 7d, against a model that says "down"
57–87% of the time regardless of the date. A fixed DA of, say, 60% is therefore excellent on the
first date and poor on the second. Every published DA in this repo has been partly a report of
which dates its cohort happened to contain.

That observation is not a quirk to caveat around — it *is* the null hypothesis of Pesaran &
Timmermann (1992, *JBES* 10(4), 461–465). Under independence between forecast and realisation the
expected hit rate is not 50%, it is `Σ_k P(pred=k)·P(act=k)`, which is exactly what
"always-down wins when the market falls" describes. There is a published test for this and the
repo was not running it.

## What now happens

`backend/backtest/directional_test.py` (new, pure — no I/O, no clock, same contract as
`backtest/scoring.py`):

1. **Per forecast date**, excess hit rate `e_d = hit_d − P*_d`, with `P*_d` estimated from that
   date's own predicted and realised label distributions. Estimating the null *per date* is what
   removes the market effect; a pooled `P*` would still let a one-sided market period masquerade
   as skill.
2. **Over dates**, a t-statistic on the mean of `e_d` under a Newey–West (Bartlett-kernel) HAC
   long-run variance — the Blaskowitz & Herwartz (2014, *IJF* 30(1)) serial-correlation-robust
   treatment. The plain PT variance assumes independent draws and this archive's carry-forward
   prices guarantee they are not: a price repeated for six days produces six outcomes that are
   nearly one observation. Bandwidth is the published `floor(4·(T/100)^(2/9))` rule — 2 at the
   20-date floor, 3 from 61 dates.
3. **Hurdle `|t| > 3.0`** (Harvey–Liu–Zhu 2016, *RFS* 29(1)), not 1.96. This repo has run well
   over a dozen feature A/Bs against the same outcome series; 3.0 is the applicable bar.

`pt_verdict` ∈ `skill` / `no_skill` / `perverse` / `insufficient_dates` / `degenerate`, stored in
`prediction_accuracy.metrics` alongside `pt_excess_pp`, `pt_t_stat`, `pt_p_value`, `pt_nw_lag`,
`pt_n_dates`, `pt_n_dates_dropped`, `pt_hurdle_t`.

**Raw DA now ships only as a triple**, per the step's instruction: `directional_accuracy`,
`constant_call_accuracy` (the best single fixed call, with `constant_call_direction`) and
`realised_down_rate`. Note that `baseline_directional_accuracy` was never the constant-call
baseline — it is the always-*flat* call specifically. It keeps its name and definition for
continuity of the stored series, and the number DA actually has to beat is now stored beside it.

## Three properties worth stating

- **A constant call sits at excess exactly zero, on every date.** When the model says "down" every
  time, `P*_d` equals the realised down-rate and cancels the hit-rate term for term. "Always-down
  beats the model" is the null *holding*, and the verdict for that case is `degenerate` — t is
  undefined, not large. This is the single most important behaviour and it is what
  `test_a_constant_call_on_a_swinging_market_is_not_skill` pins.
- **The statistic is conservative by construction.** Both departures from textbook PT push the
  same way: the per-date null is higher than a pooled one on a trending market, and the HAC
  variance is larger than the i.i.d. one under positive autocorrelation. `hit_d` and `P*_d` also
  come from the same sample, so `e_d` carries the within-date estimation noise of both — that
  noise inflates the standard error and does not bias the mean.
- **A significantly negative statistic is reported, at warning level, as a finding.** Folding
  `perverse` into `no_skill` would discard the strongest signal the test can produce.

## Surfaces

- **Log.** The `>=$1` headline line now leads with the verdict
  (`DIRECTIONAL SKILL` / `NO DIRECTIONAL SKILL` / `PERVERSE` / `NO HEADLINE (…)`) and carries the
  triple. The `<$1` line carries its own PT so it is comparable. `_headline_line` returns
  `(level, message)`, so the perverse and no-headline cases are warnings by construction rather
  than by remembering to branch.
- **API.** New `GET /accuracy/headline` — always the `>=$1` cohort, one entry per horizon from
  that horizon's latest evaluation, with the hurdle and date floor in the payload. It exists
  because `/accuracy/latest` hands back a metrics blob a caller can pull `directional_accuracy`
  out of unaccompanied. A row written before this change carries no `pt_*` keys and is reported
  as **`untested`**, which is deliberately not the same claim as `no_skill`.
- **Frontend.** The homepage placard's "Directional — 49.6% vs baseline" cell is replaced by a
  **Directional Verdict** cell (label + t) with the hit rate demoted to a "Hit Rate vs Chance"
  cell showing the triple. Pinned to 7d, not the best-scoring horizon — picking the strongest of
  four verdicts is the multiple-testing problem the t > 3 hurdle prices in. `/accuracy` gains a
  verdict table above the error cards, and its "Directional Acc" card lost its `> 50% = good`
  colouring, which was the fallacy in miniature.

## Not done here

- **Nothing was re-scored.** These are new metric keys; existing stored rows keep their `pt_*`
  fields absent until the next `--rescore`. That run costs nothing and reads no archive.
- **No production verdict exists yet.** Every live cohort still spans 1–2 forecast dates against
  `MIN_FORECAST_DATES = 20`, so all four horizons will report `insufficient_dates`. That is the
  calendar wait the step ordering was built around, and it is now instrumented rather than
  guessed at.
- **The `pt_*` keys are not backfilled into `walkforward_backtest.py` or the `ab_test_*`
  harnesses.** Those read their own loaders; step 5 is where they get touched.

## Files

| File | Change |
|---|---|
| `backend/backtest/directional_test.py` | New. The statistic, the constant-call baseline, the down-rate. |
| `backend/backtest/scoring.py` | `score_cohort` emits the PT block and the triple. |
| `backend/scripts/backtest_accuracy.py` | `_pt_str` / `_headline_line`; the headline is the verdict. |
| `backend/api/routes/accuracy.py` | `GET /accuracy/headline`, `_headline_entry`, `untested`. |
| `backend/tests/test_directional_test.py` | New, 18 tests. |
| `backend/tests/test_accuracy_headline_route.py` | New, 7 tests. |
| `frontend/lib/api.ts` | `getAccuracyHeadline`, `AccuracyHeadline`, `HeadlineVerdict`. |
| `frontend/app/page.tsx`, `frontend/app/accuracy/page.tsx` | Verdict-first rendering. |
| `docs/architecture/model.md`, `docs/product.md` | The headline is a test. |

## References

- Pesaran, M.H. & Timmermann, A. (1992). "A Simple Nonparametric Test of Predictive Performance."
  *Journal of Business & Economic Statistics* 10(4), 461–465.
- Blaskowitz, O. & Herwartz, H. (2014). "Testing the value of directional forecasts in the
  presence of serial correlation." *International Journal of Forecasting* 30(1).
- Harvey, C.R., Liu, Y. & Zhu, H. (2016). "…and the Cross-Section of Expected Returns."
  *Review of Financial Studies* 29(1), 5–68.
- Newey, W.K. & West, K.D. (1994). "Automatic Lag Selection in Covariance Matrix Estimation."
  *Review of Economic Studies* 61(4), 631–653.
