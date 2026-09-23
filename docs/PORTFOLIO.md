# CS2 Oracle — project write-up

A production forecasting system for the Counter-Strike 2 skin market, built solo from May 2026.
It runs unattended every day, and most of the work went into the part that decides whether a
number can be trusted.

**Stack:** Python · LightGBM · split-conformal prediction · DuckDB over Parquet · PostgreSQL
(Supabase) · FastAPI · GitHub Actions · pytest (~2,600 tests)

---

## The problem

CS2 skins trade across a dozen marketplaces with no consolidated price. The same item can
quote 10–30% apart on different venues, feeds go stale or freeze for weeks, and one market
factor moves almost everything together. The obvious project is "predict whether a skin goes
up." That turned out to be the wrong project, and working out why is most of this story.

## What it does

Every night the pipeline:

1. **Collects** prices for ~5,500 items from 7 marketplaces and builds an outlier-voted
   consensus price (a >2σ rejection vote, a 3-source minimum, and per-source reliability weights).
2. **Archives** them to an append-only Parquet store kept in a separate repo that only CI
   writes to, with price history back to 2013.
3. **Forecasts** a calibrated **price range** per item at 3, 7, 14 and 30 days, plus two
   disclosed probabilities: `move_odds`, the chance the move clears round-trip trading costs,
   and `anomaly_p`, the chance of an abnormal move.
4. **Scores itself.** Every served forecast is stored, then resolved against the archive
   once its target date is covered. Outcomes are frozen, so history can't be rewritten.

The chain is Aggregator → Price Forecast → Backtest, each triggered by the previous run's
success, with freshness and schema-drift checks alongside.

## The central finding: a range forecaster, not a price predictor

Directional accuracy on this market is easy to make look good. The market trends for weeks,
so an always-down call scored **29.4% on one date and 76.9% on another** at the same horizon.
A fixed hit rate is "skill" on one date and incompetence on the next.

Once the headline became a Pesaran–Timmermann test (clustered by forecast date and quoted
only next to the runnable always-down baseline), and the market-wide factor was removed,
**there was no per-item directional signal left in the features.** The limit is not model
capacity: the 2026 serving period holds only ~14–70 *independent* market episodes. Every
cross-sectional ranking idea tried was positive in cross-validation and negative in serving.

So the product became what the data supports: a **calibrated interval**. The claim it makes
is "80% of outcomes land in this band," and the backtest checks that claim every day.

## What worked, with numbers

| Change | Result |
|---|---|
| **Climatology band scale.** A featureless per-item dispersion replaces modelled volatility as the band's width denominator. | **43–47% narrower** than the σ-scaled band at matched 80% coverage, over 285 out-of-sample test dates. Four modelled-σ denominators lost to it. |
| **Signed split conformal.** Two quantiles `(q_lo, q_hi)` replace one symmetric width. | Bands **11.5 / 11.8 / 14.0% narrower** at 3/7/14d *and* closer to nominal coverage on the production panel. Decomposition showed the real defect was a biased centre (a training upweight pushing the median to the ~60th percentile), not asymmetry. |
| **Exceedance head** (`move_odds`) | The one signal that is independent of the market factor and stable across dates. Served through an isotonic calibration map; reliability error under 1.3pp at 3d and 7d. |
| **Anomaly head** (`anomaly_p`) | The **only gradient-boosted model in the project that beat a featureless baseline**: held-out AUC **+0.13 to +0.16** over the null at every horizon, all intervals clear of zero. Served at 3/7/14d only, because the 30d head ranks well but isn't calibrated. |
| **Served-outcome feedback** | Re-runs conformal calibration on the forecasts the system actually served, gated on ≥20 resolved dates per horizon. It activated at 3d on 2026-09-21. |

## How experiments are run

Of the 63 experiments in [`experiment_log.csv`](experiment_log.csv), **33 were refuted,
3 were void, 11 were measured without a ship decision, and 16 shipped.** Refutations stay in
the log with a linked write-up, so "has this been tried?" always has an answer.

The rules that make those verdicts believable:

- **Preregistration.** Hypothesis, metric, pass bars and minimum sample size are committed
  *before* the data exists. The instrument refuses to print a statistic below its power gate.
  One result that nearly passed (ACI adaptive conformal at 3d) was recorded as a failure,
  because it failed a bar written in advance. That bar was later found to be flawed, and it was
  deliberately **not** fixed after the fact.
- **Placebos in every harness.** Three band-narrowing methods (Mondrian conformal, WACI, bagged
  `q_hat`) looked 20–26% narrower until their placebo arms matched them. All three were voided.
- **Paired, date-clustered statistics.** Forecast dates, not rows, are the unit of independence.
  Re-auditing the A/B family showed most stored verdicts were underpowered (detectable effects of
  1.15–7.13pp), so they were relabelled "unresolved" rather than "null."
- **One code path for truth.** Both legs of every realised return resolve through the same price
  resolver. Mixing two estimators once made the same cohort score 61.8% on one day and 33.7%
  on another.

Things this process killed that would otherwise have shipped: social sentiment, item metadata,
volume features, regime-specific models, market-relative labels, LambdaRank, a temporal fusion
transformer centre, Google Trends and player counts, cross-market arbitrage (no capturable
spread even at 5% fees), and every per-item or per-tier way of conditioning the band tried so
far.

## Hard engineering problems

- **The "price" was five stitched sources.** The archive's price series was assembled from
  five different estimators over time, and the switches between them created fake moves on the
  switch dates. I found them, measured which years they contaminate (2026 only), and excluded
  them from labels.
- **Frozen quotes.** In one single-source period, 18–33% of prices didn't change day to day,
  and two dates were 100% frozen. These are detected and dropped from training labels.
- **A silent three-week failure.** The served-feedback query selected a column production doesn't
  have. The exception was caught and returned "no data," so the feature did nothing for three
  weeks while looking healthy. The fix included a regression test that builds the table from the
  real ORM schema and runs the live query, so this class of bug now fails in CI.
- **A green run that wrote nothing.** A run can succeed while persisting nothing. A post-run
  check now verifies that the newest forecast date actually landed in both the database and the
  Parquet mirror. On 2026-09-20 it caught a run that re-served the previous day's forecasts from
  a stale archive.
- **Unresolvable history.** A 9-day collection outage left forecasts that can never be scored.
  The resolution gate distinguishes "not ready yet" from "can never resolve," so permanent gaps
  don't pin the error rate and fail every later run.

## Honest limits

- **The band still over-covers** (~87–92% against an 80% target). Narrowing it with a single
  pooled factor would push already under-covered volatile dates lower, so the correction goes
  through served-outcome feedback, horizon by horizon, as the data matures.
- **No directional or trading claim.** A paper-trading audit found the actionable strategy
  selects zero rows once real trading costs are applied.
- **Small independent sample.** 2026 contains few independent market episodes, so several
  follow-up tests are waiting on their preregistered date counts.
- **No frontend.** It was removed to be rebuilt; the API is the product surface.

## Where to look

- [`README.md`](../README.md): architecture and quickstart
- [`docs/changelog/`](changelog/): 200+ dated decision records with measured effects
- [`docs/research/`](research/): preregistrations and research notes
- [`experiment_log.csv`](experiment_log.csv): every shipped, refuted and void experiment
