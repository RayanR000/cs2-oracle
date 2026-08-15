# Directional accuracy: can it be improved, and do we have the data?

**Date:** 2026-08-15
**Scope:** an investigation, not a decision. Answers "how accurate is the model at direction,
can it be improved, and is the data there for the surviving ideas." No code changed.
**Related:** `docs/changelog/2026-08-15-cs2-oracle-is-a-range-forecaster.md` (the positioning this
confirms), `docs/research/2026-08-14-next-steps.md` (the ranked action list this re-reads),
`backend/AGENTS.md` invariant 4.

## Short answer

Direction is **statistically real but economically worthless, and cannot be validated at the scale
we serve at.** The fix is not a better model — it is either a different *target* or more calendar
time. Every relative/cross-sectional arm tried is CV-positive and serving-negative, and the binding
constraint is the number of independent serving episodes, which is even smaller than the
`14–70` figure quoted in `AGENTS.md`.

## Best measured DA (served classifier, run `31418286692`, 2026-08-10)

| h | DA | realised_down_rate (runnable baseline) | edge | PT t |
|---|---|---|---|---|
| 3d | 49.6% | 46.06% | +3.5pp | 13.18 |
| 7d | 48.8% | 48.73% | +0.1pp | 8.00 |
| 14d | 49.3% | 50.56% | −1.3pp | 5.40 |
| 30d | 52.9% | 48.48% | +4.4pp | 4.31 |

All four clear the `|t| > 3.0` Harvey–Liu–Zhu hurdle — so there **is** significant directional
skill. It is far too small to beat the ~15% round-trip fee. Two binding caveats:

- There is **no publishable production DA** — `MIN_FORECAST_DATES = 20` refuses a headline below 20
  distinct forecast dates, and the panel has far fewer (see below).
- This table is **not currently reproducible**: `CV_DIAGNOSTIC_CLASSIFIER=0` in CI since
  2026-08-09, so every CV-log DA since is the weaker quantile sign, which the classifier beats by
  4.6–10.0pp. Source: `docs/changelog/2026-08-10-served-classifier-scored.md`.

Correcting an earlier framing in conversation: DA is **not** ~46–50% "coin flip" — the served
classifier reads 49–53% and is significant. The problem is economics and validation power, not the
absence of a signal.

## Why direction is structurally hard (the citable part)

The decisive experiment is `docs/changelog/2026-08-06-market-relative-labels-refuted.md`: once the
market-wide factor is demeaned out, `relative_accuracy_ge1` reads **36.7/32.7/34.6/39.0** against a
majority-class baseline of 38.8/42.7/46.4/51.7 — i.e. **−2.1/−10.0/−11.8/−12.7pp**. Everything the
classifier does rides the common factor; there is no idiosyncratic per-item signal in the features.
Supporting facts with citations: effective sample size is 2 not 60,737
(`2026-08-03-accuracy-is-clustered-by-forecast-date.md`); DA excluding flat-actual rows is 51.24%
vs a 50.09% base rate; the median item sees 69 Steam sales/day (4/day above $500), so per-item daily
returns are counting noise.

⚠️ **The "14–70 independent episodes" number in `AGENTS.md` is an assertion, not a measurement** —
no changelog derives it, and its stated source (the market lead-lag read) has no standalone
changelog entry. The measured episode count is smaller (next section).

## The real binding constraint: served outcome dates (measured 2026-08-15)

Independent serving observations on the ≥$1 cohort, from `ops/forecast_outcomes`:

| h | durable | local | UNION (true bound) | ~independent windows |
|---|---|---|---|---|
| 3d | 6 | 8 | 10 | ~4 |
| 7d | 8 | 7 | 12 | ~3 |
| 14d | 4 | 4 | 6 | ~1 |
| 30d | **0** | 1 | 1 | ~0 |

The single h=30 date is **2025-12-01**, the known backdated batch — drop it and h=30 has **zero**
validated serving dates and h=14 has five. The durable `ops/forecast_outcomes` is a **~4-week
rolling window** (min forecast_date 2026-07-18, h=30 absent entirely), so under the current CI
publish h=30 can never accumulate a durable observation. The ≥$1 cohort inside the outcomes is only
~1,122 items at h=3 / 966 at h=14, ~4% of the 26,623 items priced ≥$1 on the latest day.

**This bounds any directional A/B regardless of which feature feeds it.** More features, items, or
model capacity cannot buy episodes — only 2026+ calendar time can.

## Full arm inventory

An exhaustive audit of every DA arm ever tried (182 changelogs, 28 research docs, 7 rule files) was
run. Every relative/cross-sectional arm is CV-positive and serving-negative:

- **C1 within-date rank transform** — CV +0.04–0.08 rank IC, serving 1 of 6 audited anchors,
  30d reverses on clean anchors. REFUTED, closed.
- **C2 lambdarank** — CV PASS 4/4, serving 1 of 4 (h=3 sign-flips, h=14 untrainable under the
  >10K-rows-per-query cap), net-negative after cost. REFUTED on serving.
- **Market-relative (demeaned) labels** — the decisive −2 to −13pp experiment above. REFUTED.
- **ByMykel item metadata, volume features, CSFloat basis, supply-side rarity, weapon/type
  identity, market own-momentum/mean-reversion, recency decay (band arm only)** — all NULL or
  REFUTED; several fail their shuffled placebo.
- **Non-LightGBM** — CatBoost (−18 to −20pp), Ridge (−22 to −25pp paired, actively misleading),
  residual stacking (inverted quantiles), MA-crossover (~42%). Deep sequence models
  (TFT/N-BEATS/LSTM) and per-item ARIMA/GARCH are closed on mechanism in `architecture/model.md`.

## Surviving ideas worth running (ranked)

1. **Change the target to `P(|return| > round-trip cost)`.** The economically honest target — the
   only one that makes a 3% prediction distinguishable from no prediction. Never proposed as a
   target; every arm to date optimises sign or rank. Labels already computed; this is a
   loss-function change, not a data problem. **Highest value.**
2. ~~**≥$20 liquid-tier PT read** — one query on stored `prediction_accuracy`. DA has been conditioned
   on price band, weapon, date, staleness, confidence, freshness — never on the liquid tier, where
   spreads are tightest (5% at $1000+ vs 35% sub-$1). `FLOOR_SWEEP` sentinels (`-1/-2/-3` =
   ≥$1/≥$5/≥$20) are already stored per run. No record of it being run.~~ **❌ CLOSED 2026-08-15 —
   the liquid tier does not separate (see Result below).**
3. **`tier_lead` measured as a real test** — expensive→cheap lead is the only positive
   cross-sectional structure ever found (z=9.1, Granger incremental R² 9.0%, survives market-factor
   removal), yet it shipped as +0.0078 rank IC with no CI, no folds, no placebo. Data on disk.
4. **Free PT re-read on the end-anchored fold grid** (`b1cff21`) — `invariant_4_signal` must be
   re-read, not carried. Zero marginal cost.

All four are cheap and none plausibly makes direction a shippable claim under the episode bound
above. Item 1 is the only one that changes the deliverable.

## Data availability for directional inputs

Measured against the 2026 ≥$1 cohort (durable archive read over `httpfs`):

| candidate | verdict |
|---|---|
| **StatTrak pair reconstruction** | ✅ only input clearing both bars — 13 yrs history + 214 of 216 2026 dates, 298,451 pair item-days, 2,476 pairs |
| Kaggle volume panel | ⚠️ 13 yrs deep but 9.9% of item-days, 3.2k items, ends 2026-06-15 |
| BUFF bid | ❌ 31 durable dates (2026-07-11→08-14); ~3 non-overlapping h=7 windows, 0 at h=30 |
| skinport sales/volume | ❌ 7 dates — shorter than one h=7 window |
| supply/listing counts | ❌ 29-month hole: `supply-history` ends 2024-02-16, live side is 8 durable dates |
| un-blended per-venue prices | ❌ 7 of 11 venues = 31 dates; only `aggregator_sync` (blended) spans 2026 |

Two archive facts that bind regardless of direction work:

- The in-file `volume` column **dies 2026-04-15** — zero rows with `volume > 0` on any source after
  that date. Anything trained on it is unavailable at serve time.
- `median_price` **does not exist in any archive parquet file**, contradicting the documented
  schema. `min_price`/`max_price` exist only in `prices-2026-03` and `-04`.

## ⚠️ Backup risk: four deep-history panels exist only on this laptop

`volume-panel.parquet` (97.6 MB, 10.7M rows, 13 yrs Steam volume), `stattrak-panel.parquet`
(44.9 MB), `supply-history.parquet` (18.4 MB, 15.4M rows, 2.5 yrs BUFF listing counts) and
`bid-panel.parquet` (1.7 MB) are **not in the `RayanR000/cs2-oracle-data` repo** (verified against
the complete 62-blob durable manifest) and are **not produced by CI**. They are the only copy of
13 years of volume and 2.5 years of listing counts. Losing this machine loses them.

Correcting a caveat raised in conversation: backing them up is a **one-time commit**, not a workflow
change. The publish step (`aggregator-update.yml:220`) does `git checkout` of the data repo →
append → `git add -A` → orphan commit → force-push. Because `git add -A` stages the entire
checked-out tree, any file committed to the data repo once is carried into every future orphan
commit automatically — the force-push does **not** wipe it. (A backup was started 2026-08-15 and
interrupted before pushing; not yet done.)

## Result: ≥$20 liquid-tier PT read (2026-08-15) — item 2 CLOSED

Read-only query of prod `prediction_accuracy` (Supabase pooler), latest `evaluation_date`
(2026-08-15, `lgbm-v3`), sentinels `-1` (≥$1) and `-3` (≥$20). DA quoted only beside the runnable
`realised_down_rate` baseline and the PT verdict, per invariant 4.

| h | ≥$1 DA / down-rate | ≥$20 DA / down-rate (n) | ≥$20 edge vs baseline | PT |
|---|---|---|---|---|
| 3d | 40.0 / 49.0 | 38.8 / 55.1 (n=911) | **−16.4pp** | insufficient_dates |
| 7d | 44.0 / 53.9 | 40.5 / 63.8 (n=1,120) | **−23.4pp** | insufficient_dates |
| 14d | 40.9 / 61.5 | 31.0 / 74.9 (n=529) | **−43.9pp** | insufficient_dates |
| 30d | 38.7 / 32.7 | 50.0 / 54.1 (n=122, 1 date) | −4.1pp | insufficient_dates |

Three findings:

1. **PT is `insufficient_dates` at every horizon and both tiers** — even pooled across the three
   serving-config labels (`lgbm-v3` / `-global-only` / `-regime`), distinct forecast dates run ~9
   (h=3) down to 1 (h=30), far below `MIN_FORECAST_DATES = 20`. Conditioning on ≥$20 makes the
   episode bound *worse*, not better: n collapses 9,266→911 (h=3), 990→122 (h=30).
2. **At ≥$20, DA sits below the always-down baseline at every horizon**, by 16–44pp where there is
   any sample. Tighter spreads on liquid items do not come with better direction.
3. **Measurement-basis caveat:** these are the stored *quantile-sign* DA (~40%), not the served
   classifier (~49–53%, `CV_DIAGNOSTIC_CLASSIFIER=0` in CI). The absolute level is understated, but
   the relative read — ≥$20 no better than ≥$1, both below baseline, both date-starved — holds
   within one basis. Not differenced against `constant_call` (hindsight-picked; h=30 ≥$1 shows
   64.4% const-call vs 32.7% down-rate, the trap `backtest-scoring` warns of).

The liquid tier is not a hidden pocket of directional skill. Every cohort axis DA has been
conditioned on is now exhausted, none separates.

## Next step

Item 2 is closed. **Item 1 (change the target to `P(|return| > round-trip cost)`) is the only
remaining lever** — the sole idea that changes the deliverable rather than re-slicing sign. It is a
loss-function change on labels that already exist, but by the episode bound above it cannot be
*validated* until durable ≥$1 serving dates accumulate past `MIN_FORECAST_DATES = 20` (2026+ calendar
time), which the current ~4-week durable outcomes window does not yet reach at any horizon.
