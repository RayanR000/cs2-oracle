# The serving down-skew was a market-period artifact, not a transform defect

**Date:** 2026-08-06
**Change:** no code change. The "+13.4pp serving down-skew" that survived two
prior rounds of investigation is **largely refuted** as a property of the
serving transform. Measured like-for-like it is **+1.7pp at 7d**, and negative
at 14d and 30d. The calendar-gap lag mechanism is confirmed and still live.

Measured after `7b87d84` (volume features shelved) was merged and a retrain was
run, so the model under test is the 36-column one now in production.

## What was found

The prior measurement compared the classifier's predicted class mix on **serving
rows** (the last row per item) against **interior rows** (everything else), and
read the gap as evidence that the last-row + median-fill transform biases the
model toward "down". The interior baseline spanned the full history while
serving necessarily sits on the latest day. That is a confound, and it is doing
most of the work.

Same model, same 5,542 gated items, 7d, predicted-down %:

| population | down % |
|---|---|
| interior, **full history** (the old baseline) | 37.7 |
| interior, **recent 240-day window** | 51.7 |
| raw rows on the anchor date 2026-08-04 | **60.6** |
| **production serving** (`predict()` output) | **53.4** |

Two conclusions, both pointing away from the transform:

1. **The down-rate is strongly time-varying.** By month it runs from 39.9%
   (2026-03) to 60.4% (2025-08), with 2026-08 at ~56.6%. Comparing the latest
   day against a multi-year average measures the market, not the transform.
2. **The transform *lowers* the down-rate**, 60.6% on the raw anchor rows →
   53.4% as actually served. `groupby.last()` (per-column last non-NaN), the
   smoothed 3-day anchor price and the median fill move the output *away* from
   "down".

Like-for-like inside a single window, down/flat/up:

| horizon | interior | serving | residual (down) | prior residual |
|---|---|---|---|---|
| 3d | 48.4 / 25.6 / 25.9 | 52.9 / 23.0 / 24.1 | **+4.5pp** | +12.3pp |
| 7d | 51.7 / 25.2 / 23.1 | 53.4 / 25.1 / 21.5 | **+1.7pp** | +13.4pp |
| 14d | 43.6 / 22.4 / 34.0 | 42.4 / 21.2 / 36.4 | **−1.2pp** | +16.4pp |
| 30d | 46.6 / 17.7 / 35.6 | 44.1 / 22.2 / 33.7 | **−2.5pp** | −1.7pp |

**Do not cite the +13.4pp figure.** Its serving and interior columns came from
different windows and different item universes.

## Attribution caveat

The retrain overwrote the pre-shelve boosters, so the volume shelve's own
contribution **cannot be separated** from the population correction — both
changed at once. What can be said is that the skew does not survive a
like-for-like comparison against the current model. Re-deriving the split would
require retraining the 47-column model, which is not worth the run.

## Two measurement traps, both hit

**The engineered cache silently sat 10 days stale.** `predict()` takes the
chunked path (`PREDICT_CHUNK_ITEMS`, default 1000), and by design only the
whole-frame path writes `engineered_data.parquet`. Production therefore *never*
refreshes it. The file on disk was from Jul 29, anchor **2026-07-25**, over the
pre-gate **8,691**-item universe — while a live `predict()` was serving anchor
**2026-08-04** over the gated **5,542**. The first version of this measurement
took its serving column from the fresh path and its interior column from that
stale file, which is not a comparison. Set `PREDICT_CHUNK_ITEMS=0` to force the
whole-frame path and refresh the cache before measuring anything from it.

**The chunked frame cannot supply an interior population.** `PREDICT_TAIL_ROWS`
is **3**, so the chunked path keeps three rows per item; the two non-serving
rows sit adjacent to the anchor and share its gap-fills, which understates the
contrast. The whole-frame path keeps `PREDICT_TAIL_ITEM_DAYS = 240`, giving
1,322,760 interior rows across the same 5,542 items — that is the frame the
table above uses.

## What is still live: calendar-gap lag fills

Confirmed, and unaffected by any of the above. On the 2026-08-04 anchor, **4 of
36 features are median-filled on 100% of served rows**:

| feature | interior fill % | serving fill % |
|---|---|---|
| `price_lag_1d` | 2.54 | **100.00** |
| `return_1d` | 3.60 | **100.00** |
| `log_return_1d` | 3.74 | **100.00** |
| `autocorr_1d` | 5.88 | **100.00** |

The cause is upstream of the model. `_compute_price_features` joins lags by
exact calendar date, and the archive's **entire August is 08-01 and 08-04** —
08-02 and 08-03 are missing. A missing day exactly `lag` before the anchor NaNs
every `lag`-day feature for every item simultaneously, so the affected set
rotates daily with the aggregator's day gaps. The fifth member of this set,
`volume_price_conf_1d`, is gone with the volume shelve.

This is an ingestion defect, not a model defect. See
`2026-08-06-volume-features-shelved.md` for the related finding that the archive
also stopped carrying volume entirely in 2026-05.

## Retrain results

Both runs are the 36-column model. CI trained on 101,407 feature rows against
the local run's 115,763, so small differences are subsample noise.

| horizon | ≥$1 (CI) | ≥$1 (local) | all-tiers (CI) | all-tiers (local) | all-tiers (pre-shelve) |
|---|---|---|---|---|---|
| 3d | 51.8% | 49.9% | 65.8% | 67.3% | 68.2% |
| 7d | 49.6% | 49.0% | 65.6% | 67.2% | 67.9% |
| 14d | 53.3% | 51.1% | 66.4% | 67.8% | 69.9% |
| 30d | 53.9% | 53.4% | 68.4% | 68.6% | 72.8% |

This is the **first retrain to populate `mean_classifier_acc_ge1`** — the
pre-shelve artifact carried `None`, so the ≥$1 figures predicted on 2026-08-05
are now confirmed from a real training run. They land at 49.6–53.9%, somewhat
below the 52–59% that was expected. All-tiers fell 0.7–4.4pp, which is the
expected cost of dropping features that carried real signal on pre-2026-05
training rows and are dead at serve time. No volume feature appears in any
horizon's top five; those are now `price_std_*`, `trend_up_fraction_30d` and
the MACD legs.

Production is serving this model as of run
[31074743490](https://github.com/RayanR000/cs2-oracle/actions/runs/31074743490)
(15 boosters saved to the Actions cache). None of this moves the reported
accuracy: `MIN_FORECAST_DATES = 20` still gates every headline.

## Incidental: `optuna-integration` was undeclared

The first `mode=full` dispatch (run 31074434179) failed after training but
before `save_models`, with `Could not find 'optuna-integration' for 'lightgbm'`.
`forecaster.py` imports `optuna.integration.LightGBMPruningCallback`, which
optuna 4.9 split into a separate distribution; `requirements.txt` pinned
`optuna>=3.6.0` with no upper bound, so CI drifted onto 4.9.0 without it. Latent
rather than newly broken — daily runs are predict-only and never reach the HP
search, and local venvs already had the package. Fixed in `a0d06fc`. The shim is
removed in optuna 6.0, so the import should move to `optuna_integration` before
then.
