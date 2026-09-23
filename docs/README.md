# docs/

Refreshed 2026-08-21 against the code. Where a doc and the code disagree, the code wins —
report it rather than working around it.

> ⭐ **Band geometry changed twice after most of the numbers below were measured.** Read every
> coverage figure in this file as *pre-cutover* unless it is dated 2026-08-19 or later:
> 1. **Signed conformal band, live 2026-08-19** — two quantiles, serving recentring dropped,
>    `DIRECTION_UPWEIGHT = 1.0` (`forecaster.py:149`). Cutover pin
>    `SIGNED_BAND_SERVING_START = "2026-08-19"` (`models/served_recalibration.py:49`).
> 2. **Climatology band scale, default ON since 2026-08-20** — a featureless per-item
>    climatology replaces the GBM `sigma` as the width variable. `CLIMATOLOGY_SCALE` defaults
>    to `"1"` (`forecaster.py::climatology_scale_enabled`; only the literal `"0"` disables it), cutover pin
>    `CLIMATOLOGY_SERVING_START = "2026-08-20"`. It is 33-46% narrower than the sigma band at
>    matched 80% coverage and better calibrated on served replay.
>    `changelog/2026-08-19-climatology-band-scale-implemented.md`,
>    `changelog/2026-08-19-climatology-band-scale-default-on.md`,
>    `research/2026-08-19-climatology-vs-gbm-band.md`.
>
> **Never difference a coverage number across either pin.** `CLIMATOLOGY_REACTIVE` (the
> regime-reactive variant) is **built and SHELVED**, default `"0"` (`forecaster.py::climatology_reactive_enabled`) —
> `changelog/2026-08-20-climatology-reactive-band-scale.md`. The exceedance probability
> `P(|move| > round-trip cost)` shipped as a **served signal** (`EXCEEDANCE_HEAD=1` on the
> nightly retrain, migration `0024_add_forecast_exceed_p`, surfaced as `move_odds` /
> `stability_label` on `/items/*` and `/items/volatility`) —
> `changelog/2026-08-20-exceedance-served-signal.md`,
> `changelog/2026-08-20-volatility-stability-tags.md`. `FEATURE_NATIVE_NAN` was added
> 2026-08-21 gated off in code (`forecaster.py::feature_native_nan_enabled`) and is **set to `1` on the nightly
> retrain** (`price-forecast.yml:228`) — `changelog/2026-08-21-feature-native-nan-built-gated-off.md`.

> ⭐ **CS2 Oracle is a RANGE (interval) forecaster, not a directional predictor (2026-08-15).**
> The deliverable per item per horizon is a calibrated price range and its center. **Directional
> accuracy is not a shippable product claim.** Direction is structurally unavailable at this data
> scale — only ~14–70 independent h-windows exist in the 2026 serving regime, and every relative /
> cross-sectional arm (C1, lambdarank, lead-lag, own-momentum) is CV-positive and serving-negative.
> The band is the part that works: **85.90 / 88.70 / 86.46%** served coverage at h=3/7/14 against an
> 80% nominal (paired `replay_serving.py`, audited anchor 2026-06-16, n ≈ 1071, ≥$1, one anchor and
> a leaky mid). **Always quote both coverage columns** — `interval_coverage` (calibrated) beside
> `interval_coverage_dollar_basis` (published); their gap is the serving-anchor wedge. Prod-confirmed
> the same day: dollar-basis **64.0 / 51.5 / 71.8%** for `lgbm-v3`, with stale-anchor rows
> (drift ≥2%) collapsing to ~19–65% while fresh rows cover ~90%+. The remaining open thread is an
> **ops/data-freshness** question, not a modelling one.
> `changelog/2026-08-15-cs2-oracle-is-a-range-forecaster.md`.

> ⚠️ **The CV metric that ranked every accuracy arm is measured against a target the
> serving path does not use (2026-08-11).** `prepare_targets` divides by the raw quote at
> the anchor; `predict` quotes against a smoothed median. Swapping only that denominator
> recovers **+0.1398 of the +0.1464** CV↔serving rank IC gap — confirmed in CI on a fresh
> artifact at four non-overlapping anchors, **16 cells of 16**. **No stored rank IC, DA or
> `−return_1d` comparison in this repo is safe to rank arms on.** The corrected label ships
> gated as `LABEL_SMOOTHED_ANCHOR=1`. ⚠️ **Measured 2026-08-11 and NOT shippable**: it swings
> pooled served rank IC to +0.17–0.31 at 4/4 anchors, but the entire gain is the anchor
> deviation `p/S` entering the label as a free factor — on the tied cohort, where that factor
> is 1, it is −0.033/−0.017/−0.020/+0.008. Both label bases are contaminated by `p/S` with
> opposite signs; read arms on the tied subset.
> `changelog/2026-08-11-smoothed-anchor-label-measured.md`.
> `changelog/2026-08-11-the-gap-is-the-anchor-denominator.md`,
> `changelog/2026-08-11-clean-anchor-confirmed-in-ci.md`,
> `changelog/2026-08-11-label-smoothed-anchor.md`.
>
> The same run found the project's **first measured served signal**: rank IC +0.13/+0.16/+0.17
> at 3/7/14d on the third of the cohort whose anchor quote equals its local median, against
> nothing usable (−0.20 at h=3) on the rest. 30d does not replicate (+0.05, 3 of 4 anchors).
>
> **Attacking `p/S` in the SERVING basis works, and it is not an accuracy lever (2026-08-11).**
> `SERVE_OUTLIER_GATED_ANCHOR=1` serves the raw quote unless it deviates >10%; dollar error on
> the deviating cohort improves at **14 of 16 cells** above a composition placebo. But the
> model's edge over simply republishing the served price moves at 10 of 16 — a coin flip — and
> on that cohort the quote **beats** the forecast in dollars at 12 of 16 control cells. Off by
> default; the blocker is the shared backtest resolver.
> `changelog/2026-08-11-serving-anchor-freshness-measured.md`.

## Architecture (`architecture/`)

- `model.md` — the forecaster as it stands: 4 q50 LightGBM models + 4 directional
  classifiers, the conformal band, sequential training, age-based retrain. ⚠️ Check it
  against the band-geometry banner above: `HORIZONS = [3, 7, 14, 30]` and
  `QUANTILES = [0.5]` still hold (`forecaster.py:347,353`), but the band is now a *signed*
  conformal interval scaled by the per-item climatology, not the symmetric split-conformal
  one on `sigma`, and there is an extra head per horizon (the exceedance classifier)
- `model-optimization.md` — size/speed levers, split into already-applied, still-available,
  and 🛑 do-not
- `pipeline.md` — the aggregator and the workflows chained off it; what collects and what
  no longer does
- `data.md` — Parquet archive layout, the `ops/` mirror layer, Supabase serving tables

> **No production directional-accuracy figure is currently quotable.**
> `MIN_FORECAST_DATES = 20` (`backend/backtest/scoring.py`) and live cohorts span 1–5
> distinct forecast dates, so every horizon reports NO HEADLINE. Offline CV DA and
> production DA are not comparable until the served series accumulates ~20 dates.
>
> ⚠️ **Corrected 2026-08-10: this is only half a calendar problem.** `item_forecasts` holds
> **6 distinct forecast dates in total**, split three ways by `model_version`
> (`lgbm-v3-regime` 3 / `lgbm-v3` 2 / `lgbm-v3-global-only` 1). `score_cohort` keys on that
> field and it encodes the *configuration*, so every config change resets the panel — twenty
> daily runs yield twenty dates only if nothing about the config moves for twenty days, and
> `SKIP_REGIMES=1` landed 2026-08-10. Tracked as **F3**.
>
> ✅ **The code half is fixed, 2026-08-11 — and the figure is still not quotable.**
> `served_identity()` keys the cohort on the artifact rather than the configuration, so the
> panel merges to **8 / 7 / 4 / 1** forecast dates at 3/7/14/30d (from a best single cohort of
> 5 / 4 / 2 / 1) and now accumulates instead of resetting. Still under 20 at every horizon:
> the remainder is genuinely the calendar, ~12 more daily runs at h=3 and 20 maturing
> 30-day-old forecasts at h=30. Quote `config_dates` beside any pooled number.
> `changelog/2026-08-11-model-version-is-not-a-config.md`.
>
> ⚠️ **And 20 dates is the wrong bar for `DA − realised_down_rate` specifically (2026-08-11).**
> That statistic's per-date sd is **14.96 / 22.40 / 21.56 / 24.39pp**, so `sd/√20` is 3.3–5.5pp
> against effects of a few pp. `MIN_FORECAST_DATES` was set to span more than one market swing —
> stated in-source as "a judgement call, not a derivation" — and it does that; it does not make
> pooled excess readable. Report the within-date term.
> `changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.

> ⚠️ **Two published metrics do not mean what they appear to (audited 2026-08-10).**
> `constant_call_accuracy` is **hindsight-selected per fold**, so `edge_vs_constant_call*` is a
> comparison to an oracle and must not be read as a defeat — the runnable baseline is
> `realised_down_rate`, against which the served classifier is +3.5 / +0.1 / −1.3 / +4.4pp at
> 3/7/14/30d. And the served band is calibrated around the q50 mid but served around a
> **recentred** mid, so its 80% coverage claim does not hold (production `IntCov` 34.6–61.8%).
> `changelog/2026-08-10-constant-call-is-hindsight-picked.md`,
> `changelog/2026-08-10-band-and-confidence-are-miscalibrated.md`.
>
> ✅ **The apparent 12–16pp production deficit is composition (2026-08-11, figures corrected
> 2026-08-12).** Decomposing per-date `excess` on the ≥$1
> `lgbm-v3*` panel (**23,073** scorable rows, **12** dates, read from prod Postgres) gives
> **52% / 82% / 103%** of the gap at
> 3/7/14d to the realised direction of the 3–8 anchor dates, against a within-date term of
> **−2.26 / +0.34 / +4.98pp** — positive at h=14, where the model beats its own call mix.
> ⚠️ **The CV-basis agreement table (+2.70 / +0.17 / +0.42 against the published +3.5 / +0.1 / −1.3)
> was NOT recomputed and is unverified**, so "the classifier half is confirmed" is not yet a result;
> what is established is that composition is the first-order term. Per-date sd of `excess` is
> 14.96–24.39pp, so **never quote `excess` on fewer than ~50 forecast dates**; report the
> within-date term, which is what the PT statistic already differences. `MIN_FORECAST_DATES = 20`
> does not rescue it. ⚠️ **The superseded figures came from the *local working copy* of
> `ops/forecast_outcomes.parquet`, which holds only 14,668 of those 23,073 rows with
> verdict-selected gaps. Do not read the local copy for panel work — and note that neither Parquet
> copy is the full panel:** the durable archive CI publishes is fresh and cell-complete (70,409 rows,
> 0.1% by the `evaluated_at > resolved_at` test) but **shallow at 10 dates**, so **the publish leg is
> not broken**. The one defect the read surfaced is
> **diagnosed**: 2026-07-19 called `flat` on 64.0%
> of items at h=3 against a **23.59%** realised flat rate (an 18.3% hit rate, not a near-certain
> miss), because `f71ffb4` shipped a global ±0.5% dead band in `predict()` 42 minutes before that
> run and no directional classifier existed yet. Still reachable as `predict()`'s no-classifier
> fallback; the guard is open. **Correction to the record: 2026-07-19 is the production daily run and
> 2026-07-18's `-global-only` rows are an ablation arm that overwrote that day's production
> forecast** — the reverse of what `da-is-dominated-by-market-date` and prose following it state.
> `changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.

> ⚠️ **The band half of that is resolved, and the recentring was not the cause (2026-08-11).**
> Served centre, q50 centre and recentring-off agree within **1.1pp** in all 8 cells. The
> 34.6–61.8% was a **basis** artifact: `in_interval` tested an archive-resolved actual against
> a band `predict` had quoted from `current_price`, two anchors that disagree on 85% of rows by
> a median 5.70% / p90 37.82% against half-widths of 10–31%. The band is now rebased before the
> predicate, and **both** figures are reported — `interval_coverage` (calibrated, the basis
> `q_hat` was fitted in and the one `replay_serving.py` reads at ~81%) beside
> `interval_coverage_dollar_basis` (published dollars, the old number). Their gap is the anchor
> wedge. **Do not difference an `interval_coverage` across 2026-08-11.**
> **Migrated in prod the same day** (25,288 rows): all ten ≥$1 cells now read **73.8–93.6%**
> against an 80% nominal, and the `$-basis` column reproduces the old 34.6–61.8% range to the
> decimal — that range was never a calibration figure. The live open question is now
> **over**-coverage: 5 of 10 cells sit at 88–94%, which is C5's problem, well posed at last.
> `changelog/2026-08-11-in-interval-basis.md`,
> `changelog/2026-08-11-conformal-centre-follows-serving.md`.

## Reference (`references/`)

- `cs2-market-domain.md` — ⭐ **the external-knowledge file: how the market itself works.** Venues
  and their fees, why "the price" is undefined without a venue/price-type/fee-side qualifier, the
  static-level vs time-varying split (the table that decides what is even forecastable), a dated
  event timeline with a **breadth** column, how money is really made, and the fingerprints
  manipulated series leave in our data. Built 2026-08-16 from four graded web passes and reconciled
  against repo measurements — where they conflict, ours wins and the conflict is stated. Two live
  conflicts worth knowing: the web claims wide spreads on expensive items and our n = 22,449 BUFF
  measurement says the exact opposite (35.5% sub-$1 → 5.2% at ≥$1k), and the Steam fee is
  **piecewise/rounding-based**, not the flat `1.1607` (that is item 5c). §9 is a do-not-encode list.
- `steam-api.md` — Steam Market endpoints and response formats, empirically tested.
  The rate-limit envelope applies to **residential IPs only** — hosted CI runners are
  429'd on the first request.
- `open-source-shortlist.md` — ranked OSS worth adding or reading (2026-09-23): conformal PID,
  scoringrules, statsforecast; licenses checked; three malware-lure repos flagged
- `data-sources.md` — per-source status, freshness, known issues
- `data-inventory.md` — the canonical coverage audit: what is actually on disk, how much of
  the market it covers, and where the history is thin. Companion to `data-sources.md`, which
  says where the data comes from rather than what arrived
- `catalog-build.md` — Steam catalog scrape: rate-limiting strategy, gap repair
- `backfill.md` — **Dead capability.** CSMarketAPI multi-market backfill; the free-key
  quota never resets and the local DB is empty. Kept for the key-rotation and
  priority-queue design only.

## Research (`research/`)

- `2026-08-16-research-docs-review.md` — ⭐ **read before trusting any older research doc.** Audit
  of all 32 docs / 9,548 lines. Two structural findings: **R13 (the cohort inversion — served
  cohort 84% sub-$1 vs a 64% ≥$1 archive) fell out of tracking on 2026-08-09 and is verified still
  unfixed** in `database.py:120`; and **both designated entry points mislead** — the 08-07 doc
  still calls C1 the strongest predictor (refuted 08-13) and its replacement rank-IC headline is
  itself invalidated, while the 08-09 "start here" doc predates the range-forecaster pivot. Also:
  the live action list's #1 item failed 08-15, `volume-data.md`'s banner inverts its own verdict
  (volume is **feed-blocked, not refuted** — +1.4–1.9pp, placebo-clean), 13 docs are unindexed,
  and every stored A/B number predates the harness repair. §7 credits 10/10 preregistrations with
  recorded outcomes; §9 lists three claims checked and refuted, so they don't get re-raised.

- `2026-08-16-listing-count-floor.md` — closes the last of the three items
  `references/cs2-market-domain.md` proposed. Thin items **are** wilder, monotonically and *within
  price tier* (P(|r₇|>5%) runs 0.68 → 0.50 across listing buckets; 797k item-days, 2,912 items, 14
  months) — but the imported ≥30 threshold lands in the flat part of that curve: it costs **69% of
  the served cohort** and buys **0.011**. The only real break is >100 listings, which costs 90%.
  ⭐ **Verdict: no floor.** It recommended `log1p(listing_count)` as a band-width conditioner —
  ⚠️ **that follow-on is now REFUTED (2026-08-18):** 0/3 horizons pass and the thin buckets it
  targets barely exist in the served ≥$1 cohort.
  `changelog/2026-08-18-listing-count-conditioner-refuted.md`. Do not re-run.
- `2026-08-16-wash-trade-screen-and-volume-spike-exceedance.md` — **two nulls, one of them
  disguised.** The wash-trade screen proposed by `references/cs2-market-domain.md` §8 is null on
  4.7M Steam item-days: volume spikes come with **larger** moves (lift **0.31×** where the
  fingerprint predicts >1). Its residual — spikes leading `P(|r|>5%)` at lift 1.81× h=3, surviving
  both the same-day and the realized-vol-decile control — dies on the episode count: **81% of all
  spikes in 13 years are 2025-10**, and excluding the crash the lift is **1.145×** on 193 spikes.
  Build neither. Its method note is the transferable part: **report distinct-months and top-month
  share before reporting a lift.**

- `2026-08-09-model-and-data-research.md` — **Start here for anything accuracy-related.** The
  current review. ⚠️ **Read its corrections banner first** — every Track A recommendation in it
  shipped within hours, three of its cost predictions failed, and its §1c gate is refuted.
- `2026-08-07-cs2-forecasting-research.md` — ⚠️ **SUBSTANTIALLY STALE, banner at the top.** Still
  the best analysis and literature review in the repo, but six of its load-bearing numbers are
  refuted or category errors — including the `~0.3pp` date-level MDE, which is a *required* effect
  size marked `[MEASURED HERE]`. 1,706-line external design review, and its C1–C5 correction block
  overturns four of its own first-pass claims. Of those, the BUFF **bid** voting into the consensus
  as an ask is **fixed** (2026-08-07) and `walkforward_backtest.py --purge` being default OFF is
  **fixed** (2026-08-08; the flag is now `--no-purge`); the synthetic 1.1607 Steam fee constant is
  **still unfixed**, tracked as `5c`. Also carries the one measured positive — expensive tiers lead
  cheap tiers by a day, z = 9.1. See `changelog/2026-08-07-cs2-forecasting-research-review.md`.
- `2026-08-15-directional-accuracy-and-data-inventory.md` — the DA question, answered: significant
  skill (served classifier 49.6/48.8/49.3/52.9%, PT `|t|`>3 at 4/4) but economically worthless vs
  the ~15% fee, and **unvalidatable** — measured ~4/3/1/0 independent serving dates at h=3/7/14/30,
  smaller than the `14–70` in `AGENTS.md`. Ranks the four surviving ideas (target → `P(|r|>cost)`
  first) and inventories data for each. Flags that four deep-history panels (volume/stattrak/supply/
  bid) exist only locally and are **not** in the data repo.
- `2026-08-16-next-steps.md` — ⭐ **the live action list.** Supersedes the 08-14 ordering, whose #1
  item failed on 08-15. Ranks seven items: **decide the R13 cohort question** (a product call —
  serve 8,691 well-calibrated-but-untradeable forecasts or 1,398 tradeable ones; recommendation is
  a two-tier label, *not* a floor), **repair the volume feed** (now the largest measured accuracy
  item open, +1.4–1.9pp), **re-run the nine repaired A/B harnesses** (none were re-run, so nine
  stored verdicts are unmeasured), ≥$1 over-coverage, cross-market basis, listing count as band
  width, and two items recovered from 08-07 that were dropped without closing. Carries an explicit
  **do-not-run** list.
- `2026-08-16-r13-cohort-inversion-measured.md` — R13 answered. The inversion is **real and worse**
  (served cohort **16.1% ≥$1**, median served price **$0.09**) and the gate never moved — but its
  premise is **refuted**: sub-$1 is the *best-calibrated tier we have* (0.832/0.822/0.824/0.882 vs
  an 80% nominal), and the over-coverage lives in the ≥$1 tiers every published metric uses. So
  raising the floor costs 84% of the catalogue and makes average calibration *worse*. A
  product-scope decision, not a data-plumbing fix.
- `2026-08-16-listing-count-floor.md` — see the entry below; ranked as item 6 in the live list.
- `2026-08-14-next-steps.md` — ⚠️ **ordering superseded** by `2026-08-16-next-steps.md`; its **item
  1 (Armory `tier × post`) FAILED** both bars on 2026-08-15. Items 2 and 3 carried forward. Re-ranks survivors-first after the
  2026-08-14 changelog run closed six of the prior list's items (harness repin, supply-rarity null,
  C1 per-fold, CV grid, recency ships at 30d, **lambdarank refuted for serving**). Only three items
  are still worth running — the Oct-22-2025 `tier × post` natural experiment (free archive read), a
  free `q_hat`/PT re-read off the next retrain, and a cheap cross-market basis diagnostic — plus a
  hygiene tier and an explicit "do not run" list. The web doc's "pivot to a ranker" thesis is closed.
- `2026-08-13-next-steps.md` — **ordering superseded** by the entry above; still the reference for
  the item-by-item detail behind the 2026-08-13 audit (the harness-family repair, the six-deep
  stacked defects, and the cost-hygiene / dead-code lists carried forward unchanged).
- `2026-08-10-next-steps.md` — **ordering superseded** by the entry above; still the reference for
  the content of Track N and Track F. Ranked by accuracy-per-minute after the
  2026-08-10 audit. Adds **Track N** (close the `−return_1d` gap: `init_score`, then a market/rank
  decomposition, then `lambdarank`) and **Track F** (three cheap fixes that gate what can be
  published). Deprioritises anything scoped as closing the constant-call gap, and further
  retrain-cost work — the warm arm64 retrain is **996.6s / 17m48s**, inside the cap, and the
  bottleneck is now experiment power.
- `2026-08-09-next-steps.md` — the previous action list; **ordering superseded**, but still the
  reference for the *content* of every O/G/A/C/D item and its cautions. **Track A is closed
  (all six cost levers shipped 2026-08-09); D1 answered — the Steam listing page works.** **The gate
  is lifted** — Tracks C and D are unblocked. ⚠️ Its "loses to a constant call" framing throughout
  is a comparison to a hindsight-selected baseline.
- `2026-08-10-training-cost-levers.md` — the cost accounting, measured against CI runs
  `31337078991` and `31356483719`; supersedes the cost tables in
  `changelog/2026-08-09-training-cost-levers.md`. ⚠️ **Read its own corrections banner** — three of
  its claims were overturned when levers 1 and 4 landed the same day, including that lever 1 is free
  of served effects (it is not; the regime half moves the served mid). ⚠️ **Also now stale on the
  headline:** both runs it budgets against (1884s, 2306s) predate the warm cache, `SKIP_REGIMES=1`
  and arm64. Run `31407938154` measured **996.6s training / 17m48s job** — inside the 30-minute cap,
  so its remaining levers are real but no longer urgent. Its §"The structural option" (serve the
  naive predictor) is superseded by **N1**, which gets the same floor via `init_score` without
  giving up the q50.
- `2026-08-08-model-review.md` — the model review. ⚠️ **§5's composition rows are refuted in
  place** (2026-08-09); the fall attributed to composition control was a pre-2026-vs-2026 regime
  difference caused by a NULL-unsafe comparison.
- `2026-08-09-composition-stability.md` — the corrected measurement, **complete** (`f833882`).
  **Composition control does not move the reversal** (+0.1027 stable vs +0.1023 unconditional at
  3d; equal to four decimals at 7d), so the quoting-artifact gate on accuracy work is **lifted**.
  Note what is compared: the contrast is stable-vs-unconditional, because both the
  "changed (present)" and "stable & ≥3 sources" cells are 25 dates at 3d / 19 at 7d and carry no
  number. Both are calendar waits.
- `2026-08-07-next-steps.md` — **superseded for ordering** by the 2026-08-09 doc; the descriptions
  remain valid. Steps 1–7 are DONE. Steps 8–11 are NOT STARTED **except** step 10's rank-IC half
  and step 11's reversal measurement. **Blocker 5d is refuted** — it was a false positive.
  Read step 5's "not done" list before citing any A/B result: the harnesses were repaired but
  **none has been re-run**, so every stored A/B number predates the repair.
- `lis-skins-snapshot-plan.md` — ⚠️ **built, not proposed.** Shipped 2026-08-06 as
  `collectors/supply_depth.py` and runs daily; banner records which fields were dropped.
- `accuracy-opportunities.md` — closed 2026-07-31, **reopened 2026-08-07** by the review
  above, which relocates the binding constraint from input data to measurement. The stop
  banner is intact and the tables are still a record of what was tried, not a backlog. Read
  both before proposing accuracy work.
- `2026-07-19-feature-contribution-by-horizon.md` — ⚠️ **bannered.** The ablation behind
  `HORIZON_EXCLUDED_GROUPS` (now a **no-op** — the allowlist already removes those groups at every
  horizon) and the founding `+3.5pp` behind `FEATURE_GROUP_ALLOWLIST`, which has **never been
  re-derived**. Penny cohort, un-embargoed, scored against a 50% benchmark the project rejects.
- `2026-07-21-training-time-optimization.md` — 🛑 **RETIRED.** Its phase ranking is inverted
  (Optuna 59% / CV 2%; measured is CV 50% / Optuna 4%) and levers E, G and J are refuted in code.
  Use `architecture/model-optimization.md` → "Where the time goes now".
- `volume-data.md` — ⚠️ **bannered.** The conclusion (volume adds no predictive lift) stands; the
  |r| < 0.002 reasoning at `:27`/`:142` does not, the free archive source died 2026-04-15, and
  post-2026-03-22 "volume" is a listing count.
- `competitor-analysis.md` — landscape and differentiators; four inline caveats added 2026-08-09
- `research/data/2026-07-27-direction-label-sweep-raw.txt` — raw sweep output. ⚠️ **It is a crashed run** — dies
  on an SSL timeout partway through 14d; 30d never ran and no summary line was printed. Nothing
  cites it. Every treatment arm loses to control on the three completed horizons, which is
  consistent with the vol-scaled branch being dead code (`sigma=None` on both paths).

### Preregistrations

Written before the run, scored after — the repo's strongest discipline artifact, and unindexed
until 2026-08-16. **All thirteen have a recorded outcome; there are no orphans.** ⚠️ Three had
their gate rewritten after the result was seen, all on 2026-08-13 — marked below.

| Preregistration | Outcome |
|---|---|
| `2026-08-11-c1-tied-cohort-preregistration.md` | serving fails — `changelog/2026-08-11-c1-fails-the-clean-cohort-read.md` |
| `2026-08-11-mean-reversion-preregistration.md` | does not replicate — `changelog/2026-08-11-mean-reversion-does-not-replicate.md` |
| `2026-08-12-conditional-qhat-preregistration.md` | **VOID on the placebo clause**, scored inline |
| `2026-08-12-marginal-coverage-attribution-preregistration.md` | half the sigma mix — `changelog/2026-08-12-marginal-over-coverage-is-half-the-sigma-mix.md` |
| `2026-08-13-c1-audited-anchor-preregistration.md` | FAILS 3/4 horizons — ⚠️ **bar rewritten post-hoc** |
| `2026-08-13-date-level-sigma-rescaling-preregistration.md` | passes 3/7/30d — ⚠️ **h=14 failed, reclassified "unrefereeable"** |
| `2026-08-13-feature-contribution-honest-trainer-preregistration.md` | Leg 1 FAILS, +3.5 does not reproduce |
| `2026-08-13-low-level-anchor-preregistration.md` | fails its axis — ⚠️ **pivoted to an unregistered h=30 axis** |
| `2026-08-13-serving-transform-attribution-preregistration.md` | transforms do not explain the CV gap |
| `2026-08-15-armory-tier-post-preregistration.md` | **FAIL** on both the primary bar and the placebo, scored inline |
| `2026-08-17-volume-band-quality-preregistration.md` | cluster-starved — `changelog/2026-08-17-volume-band-quality-cluster-starved.md` |
| `2026-08-17-volume-band-quality-refold-preregistration.md` | passes on refold — `changelog/2026-08-17-volume-band-quality-passes-on-refold.md` |
| `2026-08-18-listing-count-band-width-conditioner-preregistration.md` | REFUTED 0/3 — `changelog/2026-08-18-listing-count-conditioner-refuted.md` |

### Also in `research/`

- `2026-08-14-what-moves-skin-prices-web-reconsideration.md` — the nulls catalog reconsidered:
  which factor nulls are true (static attributes, market-wide shocks) and which were mis-tested.
  Its externals are superseded by `references/cs2-market-domain.md`; its A/B/C framing still holds.
- `2026-08-15-p-exceed-cost-target-scope.md` — scope for the `P(|return| > cost)` target, the one
  signal that is market-orthogonal and date-stable. Magnitude, not direction: use it as band width.
- `2026-08-16-refutation-power-tiers-and-iflow-backfill.md` — free ~4yr BUFF+Steam history
  (2022-04 → 2026-05) not yet ingested. Its `count_in_24` is the volume-feed repair that item 2 of
  the live list depends on, and it multiplies backtest episodes ~10×.

## Design docs and plans (`specs/`, `plans/`)

`specs/` holds designs (27), `plans/` the execution checklists (18). Each shipped change is
also recorded in `changelog/`, which is the durable record. Load-bearing ones:

- `specs/2026-08-12-sigma-exponent-design.md` — **designed, not implemented.** The band divides by
  `sigma ** beta`. Read the matched-pair invariant before touching it: `sigma` is ~0.07 so
  `sigma ** 0.4` is ~5× larger and `q_hat` absorbs that, which makes a `q_hat` applied at the wrong
  exponent wrong by ~5×, not partially fixed. Four call sites, one persisted float, `beta = 1.0` the
  no-op default for every pre-existing artifact.
- `specs/2026-07-25-monthly-parquet-partitioning-design.md` — the live partitioning scheme
  in `scripts/append_to_parquet.py`
- `specs/2026-08-01-deterministic-backtest-design.md` — shared-estimator backtest
- `specs/2026-08-03-served-forecast-surface-design.md` — confidence-gate removal, $1 floor
- `specs/2026-08-04-minimal-model-design.md` — the 40→8 model collapse
- `specs/2026-08-05-cv-cohort-parity-design.md` — CV/production cohort mismatch. Its
  residual-gap table rests on 1–2 market days; read it with the NO HEADLINE caveat above.

## Changelog (`changelog/`)

Append-only dated decision records: bug fixes, features, audits, and refuted experiments
(`ls changelog/ | wc -l` for the count). Entries are never edited to match later reality — several describe code that has since been
deleted, which is the point. Per `AGENTS.md` workflow rule 2, non-trivial decisions get a new
dated note here.

The newest:

- `2026-09-18-multi-head-champion-challenger-built.md` — ⭐ **the shadow
  forecasting backend.** GBM q50 stays production champion at every horizon;
  last-price centre and LambdaRank collect exact shadow predictions with
  frozen shared outcomes. Promotion is manual-only behind a 20-date paired
  gate (`scripts/centre_promotion_report.py`,
  `scripts/ranking_transfer_report.py`); direction is offline-only and the
  API stays neutral.

- `2026-08-21-feature-native-nan-built-gated-off.md` — LightGBM native NaN handling instead of
  the median impute, from deep-review §10.4. Built behind `FEATURE_NATIVE_NAN`
  (`forecaster.py::feature_native_nan_enabled`, off in code) and **set to `1` on the nightly retrain**
  (`price-forecast.yml:228`). ⚠️ The probe's served effect was modest and **downward**, the
  opposite of the review's bull prior; the retrain is the go/no-go.
- `2026-08-20-volatility-stability-tags.md` — the range product's first discovery surface.
  `move_odds` (straight from `exceed_p`) and `stability_label` on the item payloads, plus
  `GET /items/volatility`; `move_odds` is published only at the horizons where it is
  calibrated, and the endpoint refuses the others rather than serving an uncalibrated number.
- `2026-08-20-exceedance-served-signal.md` — Phases A-D, merged as PR #28. The exceedance
  head (`EXCEEDANCE_HEAD=1` nightly) persists `exceed_p` (migration `0024`). This is the one
  market-orthogonal, date-stable signal, and it is **magnitude, not direction**.
- `2026-08-20-climatology-reactive-band-scale.md` — ❌ **SHELVED after a prod A/B.**
  `CLIMATOLOGY_REACTIVE` narrows ~20% roughly uniformly: it wins where the band over-covers
  and *harms* recently-calm/forward-volatile dates. A fourth width lever into the same
  forward-vol wall. Default `"0"`; do not re-propose without a new mechanism.
- `2026-08-19-climatology-band-scale-default-on.md` / `2026-08-19-climatology-band-scale-implemented.md`
  — ⭐ **the featureless per-item climatology replaces the GBM `sigma` as the band-width
  variable, and it is the current default** (`CLIMATOLOGY_SCALE=1`). 33-46% narrower at
  matched 80% coverage, and better calibrated on served replay. Served-geometry guard plus
  `CLIMATOLOGY_SERVING_START = 2026-08-20`.
- `2026-08-19-signed-conformal-band.md` / `2026-08-19-direction-upweight-neutral.md` — the
  band becomes two signed quantiles and serving recentring is dropped;
  `DIRECTION_UPWEIGHT` goes to 1.0, which centres the q50. Live 2026-08-19.
- `2026-08-19-schema-drift-ci-guard.md` — `Item.is_trainable` reached the ORM with no
  migration, prod never got the column, and the nightly chain skipped silently for days. The
  `schema-drift-check` workflow now diffs migrations against `Base.metadata` on every PR.
- `2026-08-18-listing-count-conditioner-refuted.md` — ❌ closes the last standing band-width lever.
  `log1p(listing_count)` as a conditioner fails 0/3 horizons, and structurally: the thin buckets it
  targets hold <20 items each in the served ≥$1 cohort. Was item 6 of the live list; now do-not-run.
- `2026-08-18-training-breadth-is-accuracy-neutral.md` — breadth is **free at a fixed row budget**
  (narrow/mid/wide all null at 3/7/14/30d); only raw volume helps, +0.88pp at 30d only. The
  accuracy-preserving path to more served items is **Steam-consistent backfill, not the iflow merge**.
- `2026-08-17-low-fee-arb-static-retest-negative.md` — static cross-venue arb has **no capturable
  edge even at 5%**; low-fee venues are within ~2–5% of each other and a raw-feed spread scan
  returns currency/outlier garbage. Do not build a spread scanner; the temporal basis signal is a
  separate, still-open thread.
- `2026-08-15-cs2-oracle-is-a-range-forecaster.md` — ⭐ **what kind of model this is.** The
  deliverable is a calibrated range and its center; **DA is not a shippable claim.** Records the
  four things a reader needs: direction is unavailable at ~14–70 independent 2026 h-windows (not a
  tuning failure); the band over-covers at **85.90 / 88.70 / 86.46%** vs 80% on a coherent basis
  (one anchor, leaky mid); the two coverage columns must be quoted together and `in_interval` is
  **not** "broken"; and prod's honest dollar-basis **64.0 / 51.5 / 71.8%** shortfall tracks
  anchor drift, which makes the open thread an **ops/freshness** one. Also: a persistence +
  empirical-quantile band ties or beats the served band on clean cash days — **measured, not
  decided**. Carries an explicit do-not-propose list (a fourth width scale, the
  "different rows for `q_hat`" class, `SERVE_OUTLIER_GATED_ANCHOR=1` as a coverage fix at
  **+2.65 / +0.22 / +1.06pp away** from nominal, and `replay_serving.py` for the published-coverage
  question, which it structurally cannot answer). ⚠️ **Its "no fourth width scale" clause was
  overtaken:** the climatology scale (2026-08-19/20) is a fourth width lever and it *won* — it is
  now the default. `CLIMATOLOGY_REACTIVE`, the fifth, hit the wall the clause describes and is
  shelved. Read the clause as "do not re-propose a *sigma*-shaped width lever".
- `2026-08-13-the-null-verdicts-were-not-all-tested.md` — ⭐ **the "accuracy surface is nearly
  exhausted" claim is not supported.** Of ~16 accuracy verdicts, **6 survive audit, 5 were
  underpowered against their own MDE, and 5 were never measured at all** — supply depth has no
  A/B at all, and recency weights' 30d arm *passed* its gate and was declined on a mechanism
  argument. 🔑 **`source = 'STEAMCOMMUNITY'` matches zero rows in the archive**, so
  `ab_test_supply_side`, `_regime` and `_ensemble` select **zero items and cannot run today**,
  while four more silently degenerate to a NULL-only frame ending 2025-12-31. Three further
  instrument defects (six harnesses skip `_apply_feature_allowlist`; two train on the penny pool
  and filter `>= $1` only at scoring; `supply_side`/`training_breadth` have no placebo). ⚠️ Also
  clears two *suspected* defects that are not defects — the embargo delegates correctly, and the
  phantom-slug omission is inert at every harness's `MIN_ITEM_DAYS` gate. **Fix the pins before
  re-running anything.**
- `2026-08-13-cv-folds-are-not-time-aligned-with-serving.md` — ⭐ **diagnosis only, and it opens a
  fifth explanation for C1's CV→serving gap after the named four were declared spent.** It is not a
  cohort or a code-path difference — it is a **calendar** one. `_compute_cv_splits`
  (`forecaster.py::_compute_cv_splits`) strides *forward* from `CV_MIN_TRAIN_DAYS`, so the last fold lands up to
  `CV_STEP_DAYS − 1 = 149` days short of the frame end. h=3 wins that rounding by single-digit dates
  and reaches **2026-06/07**; 7/14/30d lose a full stride and end at **2026-01-11 → 2026-02-09** —
  before the 2026-03-22 consensus break and with **no overlap at all** with the
  `2026-04-18 … 2026-06-08` anchors every served arm is read on. ⚠️ **A hypothesis with a named
  instrument, not a result — it does not un-refute C1.** The instrument is free: `rank_ic` is already
  stored per fold, so the arm−control edge on runs `31663312585` / `31663300447` can be split by fold
  date with no retrain. The fix (~5 lines, anchor the grid to the frame's end) also moves `q_hat` and
  the PT sample, so neither carries across it.

- `2026-08-13-regime-training-duplicates-the-global-fit.md` — under
  `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`, `_skipped_feature_groups` never engineers
  `cross_sectional`, so `market_return_30d` is absent, so `_assign_regime_labels` returns `range` for
  **every** row and the regime booster is refitted on identical rows. **`md5`-identical to the global
  booster at 4 of 4 horizons.** CI is unaffected (`SKIP_REGIMES=1`) but every local retrain, research
  retrain and A/B harness pays **95.4s of 872s (10.9%)** for a duplicate, and none of the thirteen
  harnesses sets the flag. Corrects the workflow comment's "covers 818K-893K of ~985K rows" — it is
  **100%** — and retires the "dropping regimes changes the served mid" claim in
  `forecaster.py::_train_horizon_inline` (regime-training comment) and in `2026-08-10-training-cost-levers.md`, **for as long as the
  allowlist holds**. 🔑 **A live footgun for C4:** re-admitting `cross_sectional` silently reactivates
  an untested three-way model split inside the A/B measuring the feature change. Delete the branch,
  or gate it on the allowlist rather than an env flag.

- `2026-08-13-the-leak-is-worth-two-points-and-it-pays-the-placebo.md` — the first harness re-read
  after the trainer fix, and it **corrects the entry below**: the leak's size **does** reproduce on
  real folds. Pooled DA rises **+1.98 / +2.42 / +2.73pp** (baseline / treatment / placebo) under
  `EARLY_STOPPING=1`, matching the breadth harness's +1.5–2.7pp; the synthetic that failed to find it
  was underpowered, so **never size a fold-structured defect synthetically**. It is a **level shift**,
  so the contrast stays `null` under both trainers and **the six primitives stay shelved**. 🔑 **The
  leak pays the *shuffled* arm most** — placebo swings **+1.09pp** between trainers (−0.393 → +0.700)
  and takes the largest pooled rise, so six columns of noise buy as much as the six real ones. It
  rewards capacity, not signal, which means **an arm that measured positive under early stopping is
  the suspect case.** ⚠️ One harness, one horizon; not a reproduction of the published −1.47pp (five
  other things changed); pooled and paired disagree in sign, and the paired interval is the verdict.
  The honest trainer is **~2.4x slower** (8m22s vs 3m34s).

- `2026-08-13-ab-harnesses-follow-productions-trainer.md` — all **thirteen** training `ab_test_*`
  harnesses now use `ItemForecaster._train_ensemble_member` at
  `_boost_rounds(horizon, cv=True)`. Until now every one early-stopped on `valid_sets=[dval]` and
  then scored `X_val` — the same rows — with a trainer production abandoned on 2026-08-08, so
  **every stored verdict describes an estimator that is not served**. Three ad-hoc opt-in flags
  (`--no-early-stop` x2, `--fixed-rounds`) collapse into `EARLY_STOPPING=1`, which reproduces the
  old arm and so makes a re-read *paired*. ⚠️ **The "+1.5-2.7pp selection leak" framing is only half
  supported** — that figure is two channels at once and **neither reproduced** synthetically
  (selection −0.26pp t = −0.40, trainer +0.11pp t = +0.72), so the size is a property of the real
  archive and is left open rather than asserted. **`C7` is unblocked, not done:** nothing was
  re-run, and the defects are stacked five deep. Suite 2142 → 2194.

- `2026-08-13-cohort-geometry-is-not-c1s-gap.md` — the last named explanation for C1's CV→serving
  gap, closed by code read plus two archive counts. The asymmetry is real and located: training's
  reference cohort is the item median over **1460 days**, serving's is the same rule over the
  predict frame (**730**-day fetch, tailed to **240** observed item-days). **It measures 0.7–0.9%**
  (Jaccard 0.991–0.993), which cannot erase +0.04–0.08 rank IC. The other training-only filters are
  null. 🔑 The disagreement is **directional** — serving admits 140–173 items training excludes
  against 30–33 the other way — which is the term that grows in a rising market and which the >25%
  frame guard cannot see. **All four C1 explanations are now spent; close it.**

- `2026-08-13-the-low-level-read-fails-its-axis.md` — the date-level rescaling was only ever
  observed where it *narrows* the band, so this measured the other direction. **(L1) fails at 3
  of 3**: pooled coverage moves down on the calm set too. Two things it establishes anyway. The
  dose is `L[t]` against the anchor's **own calibration window**, not the panel median the set was
  selected on — that axis predicts the sign **6/6 at 30d** and 3/6 at 3d/7d. And **(J)**, the
  joint set spanning the whole level range, cuts mean per-date `|cov − 80|` by **2.4–2.9pp at the
  0.0th/0.5th/0.0th percentile** of its own shuffled null, while **(L2) on the calm set alone sits
  at the 27th/17th** and must not be quoted. ⚠️ **The served regime holds exactly ONE date** in the
  panel's bottom quartile of `L[t]` that passes both audits at four horizons, so the raising
  direction is not measurable in-regime at all. Nothing dispatched, no flag.

- `2026-08-12-sigma-exponent-implemented.md` — the band can divide by `sigma ** beta`, behind
  `SIGMA_EXPONENT=1`, off by default. On the real calibration path the deciles go from
  `41 61 71 78 83 87 90 94 97 99` to **flat 80 at all ten**, marginal coverage unchanged at 80.0% (the
  property that hid the tilt for months, now asserted by a test), width 0.90×. **`q_hat` moves
  833.43 → 151.18 — a 5.5× units shift** — which is why both keys are written together and a missing
  `conformal_beta` defaults to 1.0. It also found a live bug in the shipped `elasticity` diagnostic:
  `denom <= 0` does not catch a constant `sigma`, where `x - x.mean()` is 1e-16 noise, so it returned
  a plausible **0.5** that would have been persisted and served. Two documented deviations from the
  spec (the per-fold `q_hat` stays at β = 1.0 for comparability, plus a new `fold_beta`). 17 tests,
  suite 2052 → 2069. No dispatch, nothing promoted.
- `2026-08-12-the-sigma-scale-is-one-exponent-per-horizon.md` — the open 14d/30d half of the tilt
  remedy, **decided**. Walk-forward over 507–588 dates with production's 14-day refit cadence, four
  arms: shrinkage is a **no-op** (`beta`'s departure from 1 is 6–14× its standard error, so λ =
  0.996–0.999 and the shrunk arm reproduces the plain one), and flexibility **buys nothing** — a
  per-`sigma`-decile `q_hat` is worse at 3/4 and a non-parametric binned scale is better only inside
  the noise, while losing on marginal coverage at 4/4. So: **one fitted exponent per horizon**, which
  held out cuts the level-matched tilt **−89% / −94% / −84% / −73%** and narrows bands to
  **0.87 / 0.86 / 0.84 / 0.77×**. ⚠️ That **contradicts the −26% / −12% at 14d/30d on record** — 43
  independent refits here against one held-out CV fold there, on a model-free panel rather than real
  OOF residuals, so the dispatch settles it. ⚠️ **The pinned selection rule MISFIRED and picked
  production at 4/4**, because it gated on a marginal coverage that drifts 5–10pp between periods for
  every arm; reported as a misfire, with the comparison labelled post-hoc. ⚠️ The tilt and width wins
  are stable across both periods; the **marginal win is not** — on the earlier period the exponent
  arms cover 74–77%, so nothing here fixes the level. Spec:
  `specs/2026-08-12-sigma-exponent-design.md`. Nothing shipped.
- `2026-08-12-marginal-over-coverage-is-half-the-sigma-mix.md` — the seventh cause, and the first one
  that **sizes**. The served `sigma` distribution runs **1.28–1.29×** the calibration median
  (measured directly, not implied as `half_pct / q_hat`), and pushing the measured coverage-vs-`sigma`
  curve through it buys **+4.0 to +4.9pp** of the 7.2/11.8/10.6/9.0pp excess — **36–68%**, verdict
  **PARTIAL** at 4/4. Every pre-registered leg passes: validity MAE **0.89–1.43pp**, footprint
  corr **+0.50 to +0.61**, and the `β = 1` placebo at **0.0000pp**, which proves the channel is the
  tilt and nothing else. ⚠️ **Refutes the `0.93×` at 30d** — `sigma` has no horizon term, so that
  rested on production's single 30d forecast date, and the "predicts the opposite at 30d" objection
  is withdrawn (`A` −0.22 → **+0.51**). At **7d the residual excess is unreachable** from this
  channel at any market state (`k* = 2.45×` exceeds every date in two years). The remaining 32–64%
  is **named by identity**: the residual law at given `sigma`, measurable on resolved outcomes.
  Second argument for shipping `β`, now at all four horizons. Nothing shipped.
- `2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md` — ✅ the tilt below is **confirmed on real
  OOF residuals** (run `31619383780`, ~157K records): elasticity **0.429 / 0.369 / 0.350 / 0.313**,
  within **0.014–0.032** of what the model-free instrument predicted, so that instrument is now
  validated against real residuals. The **0.798 / 0.692 / 1.034 / 1.150** on record is refuted at
  4/4 and hardest at 14d/30d, which it had called fine — those are the *worst* horizons
  (deciles **57→96** and **52→97**). Not the clip. ⚠️ **The remedy's reach is narrower than the
  tilt:** held out, the conditional error falls **−76% / −62%** at 3d/7d but only **−26% / −12%** at
  14d/30d, so one global exponent is implementable at the short horizons and needs a shrunk or
  non-parametric scale at the long ones. ⚠️ **It does not explain the marginal over-coverage** —
  level-matching removes that quantity first, and after six causes 87.2/91.8/90.6/89.0% vs 80% is
  still unattributed. `q_hat` and every `fold_q_hat` are byte-identical to `31611508808`. Nothing
  shipped.
- `2026-08-12-the-band-is-tilted-in-sigma.md` — the conditional `q_hat` is designed, costed and
  measured, and **the axis was wrong**. A model-free instrument (validated at
  **1.020 / 0.958 / 0.887 / 0.785×** the shipped `q_hat`) puts 731 dates behind the question for
  seconds per candidate. Its pre-registered read is **VOID** — the shuffled-state **placebo passed
  the bar**, because `mean_d |cov[d] − 80%|` falls whenever *marginal* coverage moves toward
  target, information or not. Level-matched to 80% first, **every date-conditional scheme is worse
  than pooled** (placebos at ≤0.11pp, so there is no noise floor hiding an effect) — a
  time-varying `q_hat` is refuted, at the same wall N2 hit. The defect is the **`sigma`
  exponent**: coverage ramps **62→95%** (h=3) to **58→98%** (h=30) across `sigma` deciles,
  monotone in all ten, stratum error **8.0–9.7pp → 0.6–1.8pp** at a fitted
  **β = 0.408 / 0.401 / 0.363 / 0.327**. Not the clip (1.2% of rows; β moves 0.389→0.395
  excluding them). ⚠️ **Contradicts the 0.798 / 0.692 / 1.034 / 1.150 already on record** — that
  read implied `sigma` as `half_pct / q_hat` from ~20K rows; the magnitude is unresolved and one
  report-only dispatch settles it. Nothing shipped.
- `2026-08-12-expanding-window-refuted-for-band-width.md` — ❌ **REFUTED.** Per-fold `q_hat` is now
  reported: the pooled value sits **0.94 / 0.91 / 0.92 / 0.84×** the p80 of folds already at the
  300K cap (the hypothesis needs it *above* 1), dropping the one small-`n` fold **widens** the
  calibration to 1.06–1.13×, and at identical `n_train` the fold spread is still **1.56–2.25×**.
  That closes the whole *"calibrate on different rows"* class, "use the late folds" included
  (a trailing 2–3 fold window is worse than pooled at 1.13–1.21×). Do not read the audit rho as
  support — `CV_MAX_TRAIN_ROWS` leaves 4 distinct `n_train` values, so it is one fold's leverage.
- `2026-08-12-conformal-basis-follows-serving.md` — ❌ **diagnosed, built, and NOT confirmed.**
  The served band over-covers (**87.2 / 91.8 / 90.6 / 89.0%** against 80%) and `q_hat` is fitted
  on the raw-anchor training label while the band is served and scored on the smoothed anchor.
  The quiet-dates alternative was measured against the calibration window's own dispersion and
  **rejected** (median `rel_cal` 1.01 / 0.96 / 1.07 / 0.97). But the pre-registered check
  **failed and the paired read refuted it**: arm against control on one commit, `q_hat` moves
  *up* at 4/4 horizons where over-coverage needs it 25–39% smaller. Ships off as
  `CONFORMAL_SERVED_BASIS=1`. The wrong sign is itself the finding — it can only happen if the
  booster's own prediction carries `p[d]/S[d]`. `sigma` was then measured and is **also not the
  cause**: `p80(s)` is below 1 in **19 of 20** `sigma` strata, including the lowest quintile at
  every horizon. Leading hypothesis is now the **expanding window** — OOF residuals come from
  fold models trained on 87k–300k rows against the shipped model's 1.2M budget, so a pooled
  `q_hat` is conservative by construction. Conditional coverage (**58.2–99.2%** per
  date) is untouched and ACI cannot be validated on 1–7 forecast dates.
- `2026-08-12-served-confidence-withdrawn.md` — F2. The `confidence` tag leaves `PredictionOut`
  and `TrendAnalysisOut`. Within-date, on the 11 cells with `n_high >= 30`, the `high` cohort is
  right **29.0–45.8%** against a stated 80% target and its gap to `low` is mixed-sign — so it is
  withdrawn as uninformative and mislabelled, not as inverted. **The backtest's `conf_gap_pp` is
  pooled across dates and must not be quoted for this**; its −38.7pp at h=30 is one item. Column,
  writer and metric all kept.

Behind it, the instrument panel and the 2026-08-10 audit, both of which carry corrections that
reach back into earlier entries:

- `2026-08-10-instrument-panel-first-read.md` — ⭐ four arms on one commit. The cross-sectional
  rank transform (`C1`) is the first arm to beat `−return_1d` on rank IC, at all four horizons,
  and it lifts served PT excess 45–96%. `init_score` (`N1`) does not clear the bar and never
  touches the served classifier; `tier_lead` closes the gap nowhere. Re-ranks
  `research/2026-08-10-next-steps.md`, which had put Track N first.
- `2026-08-10-rank-transform-reference-cohort.md` — the transform is fitted on 916 items and
  `predict`'s frame holds 5,536, so serving ranked against the wrong population. Fixed by ranking
  every row against the >= $1 cohort's distribution; sub-$1 items keep their unserved forecast
  rows. `predict` now refuses an artifact that does not record its cohort.
- `2026-08-10-constant-call-is-hindsight-picked.md` — `constant_call_accuracy` is selected with
  hindsight per fold, so `edge_vs_constant_call*` compares to an oracle; the runnable baseline is
  `realised_down_rate`. Also: `model_version` fragments the scoring panel, which is why no headline
  publishes. Corrects three earlier 2026-08-10 entries in place.
- `2026-08-10-band-and-confidence-are-miscalibrated.md` — the conformal band is calibrated around
  the q50 mid then served around a recentred one; the served `confidence` label is an uncalibrated
  0.5 cut, and the thresholds that *were* fitted describe a path production does not take.
  Diagnosis only, nothing fixed.

## Other

- `research/2026-07-21-code-review.md` — **Live punch list**, line refs re-verified 2026-08-21. The
  security cluster (SQL f-strings, default secret key, session token in a redirect URL) is
  still open verbatim; findings 5, 6, 9 and 11 closed by file deletions. Separates LIVE from
  DORMANT.
- `research/2026-08-06-model-review-plain-english.md` — the non-technical companion to the 2026-08-06
  scale-free-features audit. ⚠️ **Historical, not current advice.** Its framing is directional
  accuracy, which the 2026-08-15 reclassification retired as a product claim, and its
  "Still open" list is closed.
- `operations.md` — runbook: workflow schedules, required secrets, load-bearing steps,
  troubleshooting
- `product.md` — ⚠️ **describes the frontend deleted 2026-08-10: rebuild input, not a live spec.** Positioning, users, brand
  personality, design principles, written in the present tense about an interface that
  no longer ships

## Removed 2026-08-05

`historical/` (5 files) and `retrain-optimization-analysis.md` were deleted — the first
documented only resolved issues, the second optimized a 36-model quantile grid that no
longer exists. Both are recoverable from git history if needed.
