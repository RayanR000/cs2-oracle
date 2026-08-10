# Prediction Accuracy Improvement Opportunities

Date: 2026-07-14

> ## ↩️ REOPENED 2026-08-07 — on a different constraint than the one that closed it
>
> `docs/research/2026-08-07-cs2-forecasting-research.md` relocates the binding constraint
> from **input data** to **measurement**. ~~The banner below closes this line of work partly
> because "to resolve effects of the size this project actually produces you would need an
> MDE near 0.3pp … it is not reachable". The review measures a **date-level MDE of ~0.3pp**
> against the 2.21–3.69pp item-level floor — reachable, at a different grain.~~ It also finds
> that the headline metric measures the market's base rate rather than the model (the
> Pesaran–Timmermann null), that the published backtest number is unpurged, and that three of
> the refutations underneath this banner have intervals that are wrong.
>
> ### ⚠️ The 0.3pp half of that reopening is a category error — withdrawn 2026-08-09
>
> **There is no measured date-level MDE of ~0.3pp.** The 0.3pp is this document's own **required**
> effect size — point 3 of the banner below, where it is the MDE you would *need* and is declared
> "**not reachable**". The 08-07 review carried it forward marked `[MEASURED HERE]`, which turned a
> target into a measurement.
>
> **And the changelog cited as its source retracts it.**
> `docs/changelog/2026-08-06-date-level-exogenous-ingest.md:29-38` carries a 2026-08-07 correction:
> the original "the per-item floor is the wrong denominator" framing "was wrong on its premise",
> `backtest/paired_mde.py` already resampled at a coarser grain, and its real defect was clustering
> on `forecast_date` — **not** independent, so its intervals were too *narrow*. Its conclusion:
>
> > *"The operative floor for a date-level experiment is therefore the **fold-clustered
> > 2.21–3.69pp** … **Nothing in these two tables can be resolved below roughly 2pp**."*
>
> It also names the 0.008pp seed-only placebo as the wrong instrument for the same reason: a seed
> placebo perturbs only the RNG, so it measures reseeding noise, not the floor for an arm that
> changes the training data.
>
> **What this does and does not do to the reopening.** The measurement-side reopening **survives on
> its other three legs** — the Pesaran–Timmermann null, the unpurged backtest number, and the three
> refutations with wrong intervals are all real and all independently established. What does not
> survive is *"aggregate to the date level and the floor drops 10×."* **The date-level axis is
> subject to the same ~2pp floor as everything else**, so it is testable only for large effects, and
> the "MDE near 0.3pp is not reachable" conclusion in the banner below **stands unchanged**.
>
> Anything in `2026-08-07-cs2-forecasting-research.md` justified by "the date-level MDE is an order
> of magnitude better" — §6, §9 #9, §14, §17's decomposition diagram, §23, §24 rank 15 and its final
> step 9 — inherits this and must be re-argued on effect size, not on grain.
>
> **The banner below is left intact and still governs item-level feature work.** The reopened
> directions are aggregation (a date-level factor model), cross-tier structure, and
> measurement — not new item-level features. ~~The tracked action list is
> `docs/research/2026-08-07-next-steps.md`~~ — **superseded 2026-08-09: the live list is
> `docs/research/2026-08-09-next-steps.md`.** The record is
> `docs/changelog/2026-08-07-cs2-forecasting-research-review.md`.
>
> **Six things in the tables below have moved since 2026-07-31 and would be re-proposed if you
> read them cold.** Each is annotated in place; in summary:
> **conformal prediction is shipped**; the **supply-depth DROP rationale is factually dead**
> (free bulk feeds exist and a daily collector ships — see §1, though the MDE gate still is not
> met); the **`supply-scraper.yml` ops follow-up is done**; **time-decayed loss weighting** was
> built, A/B'd and declined; **rolling retrain on degradation** was decided against (drift is
> report-only unless `ALLOW_DRIFT_RETRAIN=1`); and **ensemble expansion moved the opposite way** —
> `N_ENSEMBLES = 1` since the minimal-model rewrite.
>
> ⚠️ **Several non-item facts in this document are stale and are not annotated individually:**
> "3 ensemble seeds = 36 models" (now 1 seed, 8 models), "drift threshold 60%" (report-only),
> `max_feature_rows = 700K` (now 1.2M at a $1 floor), model version `lgbm-v2` (now `lgbm-v3`),
> Optuna "current 8 trials" (now `N_TRIALS_MAP = {3:50, 7:10, 14:15, 30:15}`, and the objective is
> within-date rank IC since `8be48c5`), early stopping (default off behind `FIXED_BOOST_ROUNDS`),
> and the Measurement Floor MDE table (superseded by fold-clustered `paired_mde`, `180c426`).

> # 🛑 THIS LINE OF WORK IS CLOSED (2026-07-31)
>
> **Do not open a new feature or model experiment against this document.** The
> accuracy roadmap is finished — not blocked, not paused. Read this box before
> acting on anything below it; the tables are kept as a record of what was tried,
> not as a backlog.
>
> **The argument, in three numbers:**
>
> 1. Every feature group ever measured here landed at **|effect| < 0.7pp** —
>    +0.66, 0, 0, 0, +0.16, −1.10 (see Reality Check). Six consecutive results
>    indistinguishable from zero. *(The **|effect| < 0.7pp for every feature group**
>    generalisation no longer holds: a static item-metadata bundle measured +0.70pp
>    at 7d and +1.92pp at 30d on held-out ≥$1 items on 2026-08-06 — a different
>    instrument, and not shipped. See the note under §1. The six numbers listed here
>    are unchanged.)*
> 2. The A/B harness's minimum detectable effect is **1.15pp at 3d and
>    2.76–7.13pp at 7d/14d/30d** (see Measurement Floor).
> 3. To resolve effects of the size this project actually produces you would need
>    an MDE near **0.3pp** — a **~9x** reduction, i.e. **~85x more folds**. Fold
>    count is capped near 73 by disjoint-window independence. **It is not
>    reachable**, and no amount of item count changes it.
>
> The gap is not a measurement bug to engineer around. It is what a model at its
> practical ceiling looks like: the remaining roadmap items are estimated at
> 1–2pp *calibrated*, which is below the floor even in the best case, and the
> calibration history says the truth is nearer 0.2pp. Further harness work is a
> way to look busy on a finished problem.
>
> **The binding constraint is input data, not model architecture and not
> measurement.** The only remaining idea with a mechanism argument for clearing
> 1pp was supply-depth change/velocity, and it requires a *paid historical
> backfill* — see the Dropped section for why waiting for free history does not
> work.
>
> **If you want to reopen this,** the bar is a data source that is genuinely new
> (not inferable from price history) *and* has multi-year history so it can be
> trained and A/B'd over the archive window. Absent that, spend the time on
> product, ops, or reliability instead.

## Current Architecture

- **Model**: LightGBM quantile regression — 4 horizons (3d, 7d, 14d, 30d) × 3 quantiles (p10, p50, p90) × 3 ensemble seeds = 36 models
- **HP Optimization**: Optuna Bayesian (8 trials per quantile — reduced from 15 for speed, ~0pp accuracy impact; see `docs/changelog/2026-07-16-training-window-fix-and-speedups.md`), expanding-window CV
- **Feature count**: ~45-120 after correlation pruning (threshold 0.95)
- **Feature categories**: Price technicals (lags, rolling stats, Bollinger, RSI, MACD, support/resistance, volume), temporal (cyclic time features), events (5 types with exponential decay), cross-sectional (market returns, regime flags)
- **Drift threshold**: 60% directional accuracy on 7-day sliding window
- **Confidence calibration**: 80% target accuracy, min 5% coverage
- **Training data**: 1460 days backfilled from Parquet archive (2013-2026). Row count is bounded by a pre-feature-engineering item-stratified subsample (`max_feature_rows=700K`) that preserves the full 1460-day calendar window; a post-split safety cap (`max_rows`, default 700K) samples randomly rather than truncating recent data. (Previously a `tail(200K)` cap silently truncated training to ~51 days — fixed 2026-07-16, see `../changelog/2026-07-16-training-window-audit.md`. Window expanded 730d→1460d 2026-07-16, see `docs/changelog/2026-07-16-quick-postprocessing-wins.md`.)
- **Retrain schedule**: Full retrain Mondays, predict other days, auto-retrain on drift

---

## 1. Feature Engineering

| Feature | Rationale | Est. Impact | Calibrated | Effort |
|---------|-----------|-------------|-------------|--------|
| Category/collection features (same weapon group, collection, case) | Items in same category move together — category returns, volatility | 2-5pp | **0pp** ✅ tested — **contested 2026-08-06**, see note below | Low |
| Steam active listing count (vs. trade volume) | 🛑 **DROPPED** (⚠️ *rationale refuted 2026-08-06 — free bulk feeds exist and a collector ships; the drop now rests on the MDE gate alone, see §1*) — supply-side, but only change/velocity variant is directionally predictive and needs 30d history/paid backfill; free source too slow. See §1 DECISION. | 3-6pp est | 0pp pursued | — |
| Item liquidity score (volume churn ratio) | Low-liquidity items have larger price impact per trade | 2-4pp | 1-2pp | Low |
| Steam player count | Core demand driver — correlates with market activity | 2-4pp | **0pp** ✅ tested | Low |
| Tournament/major timeline + results | Skins of winning teams/players spike in price | 3-8pp | 1-3pp | Medium |
| Float/wear distribution features | Different wears behave as separate markets | 1-3pp | 1-2pp | Medium |
| Price clustering / round-number resistance | Psychological price levels ($10, $50, $100) | 1-2pp | 0-1pp | Low |
| Post-spike mean reversion speed | How quickly items revert after volume spikes | 2-3pp | 0-1pp | Low |
| Listing density (spread between min ask and max bid if available) | Market depth signal | 2-4pp | 1-2pp | High |

> ⚠️ **CRITICAL DISTINCTION — "listing volume" ≠ "trade volume".**
> The supply-depth features above (active *sell_listings* count, listing density, supply-to-volume ratio) are **supply-side** signals and are the genuinely novel remaining input. They are **NOT** the same as **trade volume** (units *sold*), which was audited on 2026-07-16 and found to add **ZERO predictive lift** — every trade-volume feature correlates with forward returns at **|r| < 0.002** (statistical noise). See `docs/research/volume-data.md:25-29` and `docs/references/data-sources.md:75-83`. Trade volume's only value in this stack is confidence/liquidity weighting, never forecasting. If a future contributor reads "listing volume" and adds *sales* volume, that is the mistake to avoid — use `sell_listings` from the `supply_scraper` / `supply_snapshots` table, not traded-volume.
>
> ℹ️ **The category/collection 0pp is contested (2026-08-06).** A 7-arm A/B on an 870-item
> deep ≥$1 universe, 33 production features, evaluated on 150 **held-out** items, measured a
> static item-metadata bundle (rarity + crate id + collection id + float caps +
> StatTrak/Souvenir, from `ByMykel/CSGO-API`) at **+0.70pp at 7d [+0.02, +1.33] and +1.92pp
> at 30d [+1.29, +2.57]**, with a per-fold permuted placebo at ~0pp so it is not capacity
> inflation. Those are **held-out-item CV numbers, not production DA** and not comparable to
> the pooled figures in this table. The plausible reason for the disagreement is coverage:
> the 0pp read used `item-metadata.parquet`, which has **rarity NULL on 4,296 of its 8,691
> rows** and no crate or collection column at all, against 99.5% rarity coverage in the
> ByMykel join. Not shipped; the ingest is not built. See
> `docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md`.
>
> ℹ️ **Amended later on 2026-08-06 by a re-run with market-relative labels.** The same six
> arms, with the target demeaned by the realized market factor
> (`models/market_factor.py`, `ItemForecaster._demean_returns`), give two corrections.
> (a) **The static-metadata effect survives at about two thirds of its size** — 7d
> +0.70 → **+0.54pp [+0.04, +1.06]**, 30d +1.92 → **+1.29pp [+0.63, +2.03]**, CIs still
> excluding zero; null at 3d and 14d. Part of the raw-label gain was the market term
> arriving through crate and collection identity, most of it was not. (b) **Item age is no
> longer refuted.** The full 9-column bundle beats the 7-column static subset at all four
> horizons (+0.34 vs +0.13, +0.75 vs +0.54, +0.99 vs −0.29, +1.85 vs +1.29), so the ingest
> recommendation is the whole bundle — but age's marginal over the static subset is still
> sign-inconsistent (−0.02 / +0.27 / +0.96 / −0.73) and **no arm isolates age inside the
> bundle**, so there is no per-column attribution. Market-relative DA scores *idiosyncratic*
> direction and is not comparable to the raw-label or production numbers; the label flag
> defaults off and the same transform was refuted for production the same day
> (`docs/changelog/2026-08-06-market-relative-labels-refuted.md`).
>
> 🛑 **DECISION (2026-07-16): Supply depth is DROPPED as a prediction-accuracy improvement.** Rationale: (1) only the *change/velocity* variant (`supply_change_7d`, `supply_listings_zscore`) is mechanistically predictive of direction — the *level* feature is a liquidity signal that does not move directional accuracy (CS2Cap: "liquidity is a tradability signal, not a price forecast"); (2) the change features require 30+ days of `supply_snapshots` history or a paid historical backfill (CS2Cap candles `q`); (3) the only free source is the Steam full-catalog scrape (~115 min/day) — deemed too slow/high-effort, and no free bulk listing-count source exists; paid APIs (CSMarketCap $9.99/mo, CS2Cap $19/mo) were rejected. Expected lift was only ~+1-2pp directional. Remaining accuracy work shifts to model architecture (regime-switching, Ridge head) on existing data. The `_add_supply_depth_features` code remains but is excluded from the accuracy roadmap.

---

## 2. Model Architecture

| Approach | Expected Benefit | Calibrated | Complexity | Status |
|----------|-----------------|------------|------------|--------|
| **Linear/Ridge head ensemble** — hybrid tree + linear model | 2-5pp | 1-2pp | Low | ✅ Done |
| **Regime-switching models** — separate LGBM per bull/bear/range regime | 3-8pp | 2-4pp | Medium | ✅ Done |
| **Multi-horizon joint training** — all 4 horizons in one model | 1-3pp | 1-2pp | Medium | Pending |
| **Expand ensemble to 7-10 seeds** | 1-2pp | 1-2pp | Low | Pending |
| **N-BEATS or Temporal Fusion Transformer** — neural net ensemble | 3-8pp | 1-3pp | High | Pending |
| **Hierarchical forecast** — market → category → item | 2-4pp | 1-2pp | High | Pending |

---

## 3. Training Pipeline

| Change | Impact | Calibrated | Effort |
|--------|--------|------------|--------|
| Increase Optuna trials from 8 (current) to 50-100 | 2-5pp | 0.5-1pp | Low (compute only) — note: reduced 15→8 on 2026-07-16 for speed at ~0pp cost |
| Per-cluster models — cluster items by volatility/volume/liquidity, train specialized models | 3-8pp | 1-3pp | Medium |
| Time-decayed loss weighting (weight = α^(days_ago), α=0.99) | 2-4pp | 1-2pp | Low |
| Adversarial validation between train and serving data | Better drift detection | Low | Medium |
| Rolling retrain on any day accuracy degrades (not just triggered at 60%) | 2-3pp | 1-2pp | Low |
| Learning rate warmup + schedule decay | 1-2pp | 0-1pp | Low |
| Gradient-based feature selection (SHAP importance pruning) | Simplifies model, prevents overfit | Low | Low |

---

## 4. Post-Processing & Calibration

| Change | Impact | Calibrated | Effort |
|--------|--------|------------|--------|
| Directional smoothing — EMA on predicted direction to reduce daily flip-flopping | 1-2pp | 1-2pp | Low |
| 4-tier confidence instead of binary (high/medium/low/very-low) | Better risk stratification | Low | Low |
| Ensemble variance as confidence signal | More calibrated uncertainty | Low | Low |
| ~~Conformal prediction on p10/p90 intervals~~ ✅ **SHIPPED 2026-08-04** — `backend/models/conformal.py`; the 24 p10/p90 quantile GBMs were replaced by locally-weighted **split conformal** around a single q50. `meta.json.conformal_calibration` carries the per-horizon `q_hat`. | Better coverage guarantees | Medium | Medium |
| Forecast blending — blend current prediction with previous day's at small weight | Reduces jumpiness, 1-2pp | 1-2pp | Low |

---

## 5. Data Quality

| Change | Impact | Calibrated | Effort |
|--------|--------|------------|--------|
| Multi-source outlier voting — if 5/7 sources agree, downweight outliers | 2-4pp | 2-4pp (keep) | Low |
| Intraday high/low price range per source per day | Volatility signal, 1-3pp | 1-2pp | Medium |
| Gap-fill with interpolation instead of forward-fill | More continuous signal | Low | Low |
| Source reliability scoring — weight each source by historical accuracy | 1-3pp | 1-2pp | Medium |
| Consistent timestamps across sources (align to UTC hour) | Prevents stale-data comparisons | Low | Low |

---

## 6. External Data Sources

| Source | Signal | Difficulty |
|--------|--------|------------|
| [SteamCharts](https://steamcharts.com/) API | Player count trends | Low |
| Twitch/YouTube CS2 category metrics | Hype cycles, content trends | Medium |
| Liquipedia tournament schedule + results | Major/event anticipation & reaction | Medium |
| Reddit r/GlobalOffensive, r/csgomarketforum | Sentiment (early hype) | High |
| Steam Community Market listing count API | 🛑 **DROPPED** — supply depth not pursued (2026-07-16); paid bulk APIs (CSMarketCap $9.99, CS2Cap $19) rejected. ⚠️ **Premise refuted 2026-08-06:** free bulk listing counts exist (Skinport/Waxpeer/Bitskins, ~5s) and `collectors/supply_depth.py` collects them daily. No paid API is needed; the drop stands on power, not access. | — |

---

## Priority Order

### ✅ Completed
1. **Supply-side features** — rarity one-hot kept (+10-12pp causal). Weapon_type/player counts removed (zero causal).
2. **Event decay optimization** — **0pp**; defaults were already optimal.
3. **Auto-prune** — permutation-based feature validation prevents overfit.
4. **Multi-source outlier voting** — **0pp on training, essential for inference**.
5. **Data quality audit** — dead item filter, target winsorization, corrupt item exclusion, sample weighting, 2026 shift guard. Cumulative est. +3-8pp orthogonal gain.
6. **Regime-switching models** — separate per-regime LGBM ensembles (2026-07-18).
7. **Ridge residual stacking / DART / forecast blending** — post-processing improvements.
8. **Feature contribution by horizon** — pruned harmful cross-sectional/event features at 14d/30d (2026-07-19).

### Dropped
- 🛑 **Supply depth (`sell_listings` count)** — change/velocity variant is predictive but needs 30+ days history or paid backfill. Free source too slow. Rejected 2026-07-16.
  - **Re-checked 2026-07-31 — the 30-day gate did not expire, and cannot.** The
    obvious hope was that 3.5 months of daily scraping had since accumulated the
    history the 2026-07-16 decision lacked. It has not. Both prod Supabase and
    `price-archive/ops/supply_snapshots.parquet` hold **exactly one day**
    (2026-07-15, 35,037 items) — **zero days added in 16 days of green CI.**
  - **Root cause:** Steam returns **429 on the first request** (`offset=0`) from
    GitHub-hosted runner IPs. `supply_scraper` backs off 30s→60s→120s, logs
    `Could not get total item count from Steam. Aborting.`, stores **0 snapshots**,
    and **exits 0** — so the workflow is green daily and no failure issue is filed.
    Verified in run `30589441873` (2026-07-30). The 4m30s runtime, vs the ~115 min
    a real catalog walk takes, is the visible tell.
  - **Consequence:** the free path does not merely run slowly, it **cannot run from
    CI at all**. Waiting accumulates nothing. And even a fixed scraper only ever
    grows history *forward* — a feature present for the last 30 days is untrainable
    over a 1460-day window and unmeasurable in a 26-fold walk-forward A/B, where it
    would be null for ~98% of rows. **Only a paid historical backfill (CS2Cap
    candles `q`) could revive this**, which remains declined. Drop decision
    reaffirmed and now permanent on data grounds, not cost grounds.
  - **Ops follow-up (independent of accuracy):** ✅ **DONE 2026-07-31** (`0288568`) —
    `supply-scraper.yml` was deleted, not merely disabled. ~~Either disable
    `supply-scraper.yml` or make the abort path exit non-zero so it fails loudly.~~ Tracked in
    `docs/changelog/2026-07-31-accuracy-work-closed.md`.

> ### ⚠️ The premise of this DROP has since been refuted — 2026-08-06
>
> **"No free bulk listing-count source exists" and "cannot run from CI at all" are both false.**
> Skinport, Waxpeer and Bitskins return live per-item listing counts in bulk, free, in ~5s. A
> **daily collector shipped**: `backend/collectors/supply_depth.py` +
> `scripts/run_supply_depth.py`, wired into `aggregator-update.yml`, writing
> `price-archive/supply-YYYY-MM.parquet` across 5 feeds and covering **71.1% of the ≥$1 cohort**.
> `market.csgo.com`'s `volume` field was separately verified four ways to be a **listing count**,
> not the refuted trade volume, so it is usable as depth. Retroactive feeds also exist — Skinport
> sales history carries 90-day volume windows and lis-skins exposes 2.3M listings with
> `created_at`, which kills the "only grows forward" argument too.
>
> **What is NOT refuted, and is why this is not yet an accuracy result:** the collector is
> accumulation only, and the **MDE gate is still unsatisfied**. The item-level floor is
> 2.21–3.69pp against an expected lift of ~1–2pp. So the DROP stands *as a prediction-accuracy
> decision* — but on power, not on the data-availability grounds written above. Do not cite the
> "no free source" reasoning again.
>
> See `docs/changelog/2026-08-06-free-bulk-supply-depth-feeds-exist.md`,
> `-supply-depth-collector.md` and `-retroactive-supply-feeds.md`.

### Remaining — none. Closed 2026-07-31.

Formerly listed as remaining, now **abandoned unmeasurable**:

1. ~~**Multi-horizon joint training**~~ — all horizons in one model, 1-2pp.
2. ~~**Ensemble expansion**~~ — more seeds with column subsampling, 1-2pp.

> ⚠️ **Both are at or below the A/B harness's minimum detectable effect** (1.15pp
> at 3d, 2.76–7.13pp at 7d/14d/30d — see Measurement Floor). Even if they work,
> the current design cannot confirm it.
>
> **Resolved 2026-07-31: do not "fix the measurement first" either.** That was the
> standing advice here and it was wrong — an earlier version of this note told the
> next contributor to repair the harness before spending compute. The repair is not
> affordable: closing the gap to the ~0.3pp effects this project actually produces
> needs ~85x more folds, and disjoint-window independence caps folds near 73. Both
> items are hereby abandoned as unmeasurable rather than deferred. Shipping them on
> mechanism alone was considered and rejected — six prior groups had a mechanism
> argument too, and all six measured ~0pp.

### Tested & removed
- 🛑 **Pure-price technical primitives** (volatility asymmetry + oscillator
  divergence; 6 features) — built, A/B'd on a 40-item smoke and **shelved**, then
  re-run at decision scale (200 items) 2026-07-31. Treatment measures **negative at
  all four horizons**: 3d −0.69 / 7d −1.47 / 14d −0.88 / 30d −1.38pp, pooled
  **−1.10pp**, 43/104 fold wins, p=0.136. **Kept shelved** — the columns are still
  engineered but withheld from training via `ItemForecaster.SHELVED_FEATURES`.
  **Do not re-run at larger item counts:** fold count is set by `step=60` and the
  archive date range, not `--max-items`, so more items cannot resolve it. See
  `docs/changelog/2026-07-31-price-primitives-decision-scale.md` — and the
  measurement-floor section below, which that run produced.
- 🛑 **Market-relative (cross-sectionally demeaned) direction labels** — the
  strongest surviving idea in this document, and the one the closure memo's
  "variance reduction, not more items" escape clause pointed at. Built and
  measured 2026-08-06. It is refuted in a way that **strengthens this banner
  more than any other entry here**: training the classifier on `r − m` and
  scoring the idiosyncratic call gives `relative_accuracy_ge1` of
  **36.7 / 32.7 / 34.6 / 39.0** (3d/7d/14d/30d) against a majority-class
  baseline of **38.8 / 42.7 / 46.4 / 51.7** — *below a constant call at every
  horizon*. Once the market factor is removed, the price-technical feature set
  predicts nothing about **which item** moves which way. That is a mechanism for
  the entire null streak: these experiments were variations on a signal that is
  not present at the item level, and no feature group, reweighting or model-class
  swap addresses it. Kept in the code, defaulted off
  (`DEFAULT_MARKET_RELATIVE_LABELS = False`). Bounded by the 99-item subsample
  and the mover-weighting artifact — see
  `docs/changelog/2026-08-06-market-relative-labels-refuted.md`.
  **Corroborated on a second, independent instrument the same day.** In the
  item-metadata A/B (870-item deep ≥$1 universe, 150 held-out items, 33 features),
  a feature column holding **nothing but the calendar date ordinal** bought
  **+11.21pp [+9.30, +13.01] at 30d** under raw labels; demeaning the label by the
  realized market factor collapsed it to **+0.53pp [−0.08, +1.13]**, and to
  **+0.00pp at 3d**. The largest "feature" effect ever measured in this project was
  the market term, and subtracting it removes it. That is the mechanism above,
  measured without the production retrain path. See
  `docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md`.
- 🛑 **Served-cohort (≥$1) weighting of the directional classifier** — the one
  angle that legitimately reopened this document, since every decision recorded
  here was scored on the **pooled** metric and `classifier_accuracy_ge1` did not
  exist until 2026-08-05. Built and measured 2026-08-06 on two paired cold
  retrains (identical folds, rows and `tuned_params`; only the training weight
  vector differed). Moving the ≥$1 cohort from ~18% to **50.0%** of the
  classifier's training weight changed paired `classifier_accuracy_ge1` by
  **−0.43 / +1.18 / −0.07 / −0.97pp** (3d/7d/14d/30d) and pooled accuracy by
  **~0.1pp at every horizon** — the intended pooled-for-served trade never
  happened, because the decision function barely moved. Below the
  pre-registered +2pp bar; **kept in the code, defaulted off**
  (`DEFAULT_SERVED_COHORT_SHARE = None`). This refutes "capacity is spent on the
  wrong cohort"; it does **not** test whether *more* ≥$1 items would help. See
  `docs/changelog/2026-08-06-served-cohort-weighting-refuted.md`.
- 🛑 **Quality spread / cross-wear features** — built and A/B'd 2026-07-26
  (walk-forward, 1,500 variant-group items). **Net-flat: +0.16pp mean**
  (3d −0.78 / 7d +1.28 / 14d +0.76 / 30d −0.63pp), and **+77% feature-build
  time**. Helps 7d/14d, hurts 3d/30d. **Feature code removed entirely** the same
  day (net-flat did not justify the build cost + complexity). Design preserved in
  `docs/superpowers/specs/2026-07-25-quality-spread-features-design.md`; result in
  `docs/changelog/2026-07-25-quality-spread-experiment.md`.

---

## Notes

- **Baseline restored (2026-07-16):** the first trustworthy CV numbers are 3d 69.3% / 7d 68.0% / 14d 67.8% / 30d 67.0% directional. Earlier 60–68% figures came from a broken pipeline — training was truncated to ~51 days (fixed) AND `fetch_price_history` had mislabeled its columns so the model read source-name strings as prices (fixed). See `docs/changelog/2026-07-16-training-window-fix-and-speedups.md` Parts 1 & 3.
- **Re-baselined (2026-07-16, `lgbm-v3`):** the restored baseline used 2 expanding-window folds (all the 730d window allowed). After expanding to 1460d + 9-ensemble + 20 HP trials (`lgbm-v3`, see `docs/changelog/2026-07-16-quick-postprocessing-wins.md`), the CV now produces **6 folds** spanning 4 years. The new 6-fold numbers are **~66%** across horizons (3d 66.8%, 7d 65.7%, 14d 65.9%). The gap from the 2-fold baseline is explained by the stricter measurement — more folds test against more market regimes and produce a more pessimistic (but more honest) estimate, not a model degradation. The historical walk-forward backtest is the definitive accuracy benchmark; CV is a training-time diagnostic. **Correction (2026-07-29): the walk-forward backtest does NOT run automatically in CI.** CI runs `scripts/backtest_accuracy.py --type forecast`, which scores *stored* forecasts against matured actuals — so it lags a new model by 3–30 days and never validates one at training time. `scripts/walkforward_backtest.py` must be run manually (~60–90 min for all horizons; `--horizons 3 7` for ~30–40 min). Budget for it when shipping a model change.
- CatBoost was tested and removed (Jul 2026) — degraded accuracy by 18-20pp — do not revisit
- **Do NOT add trade/sales volume as a predictive feature** — audited 2026-07-16, |r| < 0.002 with forward returns (0pp). Supply depth (`sell_listings`) was also evaluated and **dropped (2026-07-16)** as an accuracy improvement (see §1 DECISION): only its change/velocity variant is mechanistically predictive but requires 30+ days of history or a paid backfill, which was not pursued.
- Trend analyzer was deprecated and removed (Jul 2026)
- Grid search replaced by Optuna Bayesian (Jul 2026)
- Model version is `lgbm-v2`; any architecture change should increment to `lgbm-v3`
- All changes must pass `test_forecaster.py` (28+ tests)
- Production models stored in `backend/models/saved_models/` — can serve multiple model versions simultaneously

---

## Reality Check — Calibrating Estimates

Every completed feature group was measured. The pattern is consistent:

| Feature | Estimate | Actual | Calibration Factor |
|---------|:-------:|:------:|:------------------:|
| Supply-side bundle (rarity + weapon_type) | +3-6pp | **+0.66pp** | ~15-20% of estimate |
| Player counts | +2-4pp | **0pp** (spurious +3pp A/B) | — |
| Event decay optimization | Small | **0pp** | — |
| CatBoost | not est. | **-18 to -20pp** | — |
| Multi-source outlier voting | +2-4pp | **0pp train / essential inference** | Pre-backfill estimate; 99.6% training data now single-source |
| Quality spread / cross-wear | +1-2pp | **+0.16pp mean** (net-flat; +1.28/+0.76 on 7d/14d, −0.78/−0.63 on 3d/30d) | ~8-16% of estimate; shelved for +77% build cost |
| Price technical primitives (vol asymmetry + oscillator divergence) | +1-2pp | **−1.10pp pooled** (negative at all 4 horizons; 43/104 fold wins, p=0.136) | Below the harness noise floor — see Measurement Floor |

### Root Causes

1. **Extra capacity inflation.** Adding more features gives LightGBM more leaves to split on, inflating validation accuracy by 1-4pp even when the features have zero causal signal. Player counts showed +3pp A/B → 0pp permutation. **Always pair A/B tests with permutation tests.**

2. **Existing features capture most signal.** Price technicals (lags, returns, rolling stats, Bollinger, RSI, MACD) + cross-sectional (market returns, regime) → ~55-60pp directional accuracy. Rarity adds ~+10pp causal within the model, but the marginal gain of adding it to the baseline was only ~+0.5pp because the model partially compensates. **Past ~70 features, each new group delivers 30-50% of the initial estimate.**

3. **Estimates assume independent signal. They're not independent.** When features are correlated (and most market features are), the marginal gain of any new feature shrinks as the set grows.

### Calibrated Rule

For any new feature group added to the current ~70-feature set:
- **Novel signal** (genuinely new information like source spreads): expect **30-50% of pre-estimate**, floor 1pp
- **Proxied signal** (information the model can infer from price behavior): expect **10-20% of pre-estimate**, floor 0pp
- **Data quality improvements** (outlier voting, source reliability): **not subject to diminishing returns** — improves ALL existing features. The 2026-07-17 data quality audit proved this category is the most mispriced: removing 41% dead training rows and clipping corrupt targets improves every downstream gradient step, and these gains compound with feature/model improvements.
- **Training data filtering** (dead item removal, target winsorization, corrupt item exclusion): **30-70% of pre-estimate**. Unlike feature additions, data filtering actually *removes noise* rather than adding capacity. The 41% row reduction allows the model to focus its limited leaves on signal. Initial estimates of +3-8pp are more likely to hit than feature additions because there's no "extra capacity inflation" effect.

### Measurement Floor — what the A/B harness can actually detect

> ⚠️ **The table below is superseded, and the direction of the error is the surprising part.**
> `paired_mde` clustered on `forecast_date`, which is not independent — every date in a fold's
> validation window is scored by one fitted model — so these intervals were **too narrow**, not too
> wide. Fixed 2026-08-07 (`180c426`); see `docs/changelog/2026-08-06-date-level-exogenous-ingest.md:29-38`.
> The operative floor is the **fold-clustered 2.21–3.69pp** measured on the breadth A/B at 25–26
> folds. The arithmetic below is internally correct (`MDE = 2.8·sd/√26` and the fold counts both
> check out) — it is the standard error that was wrong.
>
> **This strengthens rather than weakens the closure argument**, since a wider true floor makes the
> sub-1pp effects this project produces *less* resolvable, not more. It also means the "1.15pp at
> 3d" figure quoted in point 2 of the stop banner, in "Remaining — none", and as "the harness noise
> floor" in `docs/architecture/model.md` is **not citable** — use 2.21–3.69pp.

**Added 2026-07-31.** The calibrated rule above says what gain to *expect*. This
says what gain you can *measure*, and the two are in conflict: most ship gates in
this repo are written around **0.5–1.5pp**, which is **below the noise floor of the
harness that evaluates them**.

Measured on `ab_test_price_primitives.py` at 200 items — paired per-fold
directional-accuracy deltas between two arms on identical folds:

| Horizon | paired fold sd | min detectable effect @ 26 folds (80% power) | folds needed for 0.5pp |
|---------|:--------------:|:--------------------------------------------:|:----------------------:|
| 3d  | 2.09pp  | 1.15pp | ~137 |
| 7d  | 5.03pp  | 2.76pp | ~795 |
| 14d | 5.70pp  | 3.13pp | ~1,020 |
| 30d | 12.98pp | 7.13pp | ~5,290 |

Two structural facts make those fold counts unreachable:

1. **Fold count does not scale with `--max-items`.** It is set by the harness's
   `step` (60) and the archive date range — 26 folds at 40 items and 26 at 200.
   More items sharpen each fold's estimate; they never add a fold. *Scaling up a
   null A/B by item count does not make it conclusive.*
2. The most folds obtainable with **disjoint** 21-day validation windows is ~73
   (`step=21`), which only lowers the 7d floor to ~1.65pp. Below `step=21` the
   validation windows overlap and the folds stop being independent. Note that even
   at `step=21`, folds are only *approximately* independent at 14d/30d, where
   forward-return windows still overlap across folds.

**Consequence for reading past results.** A reported delta smaller than the
horizon's MDE is *not* evidence of "net-flat" or "no effect" — it is no evidence
either way. The quality-spread result (+0.16pp mean, ±~1.3pp per horizon) and the
price-primitives result (−1.10pp pooled) both sit inside the floor. Those changes
were correctly declined on **cost and complexity** grounds; neither was actually
*measured* to be flat, and the write-ups should not be cited as if they were.

**Before running any feature A/B here:**

1. Run two arms on **one** horizon (`--horizon 7 --arm baseline` / `--arm treatment`).
2. Compute the paired per-fold delta sd from the `per_fold` arrays in the output JSON.
3. Derive `MDE = 2.8 * sd / sqrt(n_folds)`.
4. **If the gate's target is below the MDE, do not run the full experiment** — it
   cannot return a decision. Either pursue an effect large enough to clear the
   floor, or change the design (more folds, variance reduction, a paired test at
   the row level with correlation-aware standard errors). Buying more items is not
   the fix.

This makes the shortest path to *any* further accuracy work a **measurement**
problem, not a feature problem. The remaining roadmap items below are estimated at
1–2pp calibrated — at or under the floor for 7d/14d/30d — so as things stand they
cannot be validated even if they work.

### Cumulative Ceiling

The combined improvement from completing ALL remaining work is likely **+5-8pp** (current 60-68% → 65-76%), not the +20-30pp that summing initial estimates would suggest.

**Note (2026-07-17):** The data quality fixes (dead item filter, target winsorization, corrupt item exclusion, sample weighting, 2026 shift guard) add an estimated **+3-8pp** that is orthogonal to all prior feature/model work — these gains compound on top of the ceiling. Realistic new ceiling after data quality fixes: **+8-16pp total from all completed + remaining work** (60-68% → 68-84%), though the upper end requires the remaining architecture changes (regime-switching, quality spread) to also deliver.
