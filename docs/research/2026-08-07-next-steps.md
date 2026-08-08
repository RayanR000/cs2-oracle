# Next steps from the 2026-08-07 research review

**Source:** `docs/research/2026-08-07-cs2-forecasting-research.md`, section
"If I were building this myself", cross-referenced against §10 "Ranked recommendations".
**Record:** `docs/changelog/2026-08-07-cs2-forecasting-research-review.md`.
**Status: steps 1, 2 and 3 are DONE (2026-08-07 —
`docs/changelog/2026-08-07-bid-source-excluded-from-voting.md`,
`docs/changelog/2026-08-07-pesaran-timmermann-headline.md` and
`docs/changelog/2026-08-07-friction-conditioned-tier-scoring.md`), and step 4 is DONE
(2026-08-08 — `docs/changelog/2026-08-08-phase-collapsed-names-dropped.md`). Steps 5 and 6
are unblocked and NOT STARTED. Steps 7–11 are NOT STARTED.**

Ordering is the review's, not a re-ranking. Numbers in the "why" column are quoted from the
review or from the changelog entry that measured them; nothing here is estimated.

---

## The two scheduling facts

**1. Steps 1–6 need no retrain and are roughly two weeks of work.** They are scoring,
filtering, harness and schema changes. They do not touch the Monday `mode=full` budget at
all, and they can run while the ≥20-forecast-date wait (`MIN_FORECAST_DATES`,
`backend/backtest/scoring.py`) elapses on its own. The review states this as the reason the
order is what it is: *"the highest-value work available is free of the Monday retrain budget
entirely"*, and *"(b) is a calendar wait, not work — which is exactly why the metric fixes
should start now, while the wait is free."*

**2. Step 1 gated everything else, and it is now done.** It was voting: every label and every
A/B from **2026-07-11 to 2026-08-07** is downstream of a displaced consensus, and needs
re-running. Steps 2–6 can now proceed — but any pre-2026-08-07 number they are compared
against carries the defect.

---

## Gated: this ran first

### 1. Is `aggregator_buff163_buy` in the consensus median? — **DONE 2026-08-07**

**It was.** `BID_SOURCES` now excludes it from `_apply_multi_source_voting` and from
`walkforward_backtest.py::_load_all_prices`; `VOTED_CACHE_VERSION` bumped 1 → 2. Measured on
the ≥$1 served cohort (1,398 items), **95.4% of item-days had a displaced consensus, median
−10.8%, and 13.6% of return pairs had their direction flipped**. Record:
`docs/changelog/2026-08-07-bid-source-excluded-from-voting.md`.

**Three claims in the original entry below were refuted by the measurement:**

1. **The level was 0.579× Steam against asks at 0.717–0.809×**, on n = 461,540 item-days
   (2026-07-11 → 2026-08-04), not 0.550× / 0.700–0.802× on 23,904 items. Same conclusion,
   re-measured denominator.
2. **The "≥3 sources" reasoning was wrong.** `vote()` was *eligible* on **99.3%** of bid
   item-days (11 sources on 55% of them) and simply failed to reject the bid **80.5%** of the
   time — an 11-source panel spanning 0.58–1.04× of Steam has a σ too wide for 2σ to catch a
   value at 0.579×. **The guard was never the fix.**
3. **Flicker was not the primary mechanism.** Day-to-day flicker in whether the bid is
   rejected is **5.6%** of return pairs; **steady-state inclusion with a drifting bid–ask
   spread is 53.5%** and carries ~4× more of the >1pp damaged mass. A third mechanism the
   review did not name — **median parity**, where one low value steps an even panel's median
   down a rung (−6.35% at n = 11) — is larger than flicker too.

- **Not done:** the bid was **not** promoted to its own column (it stays recoverable as a
  labelled archive row; the spread-feature family it would serve still has 25 days of
  history), and **the affected A/Bs were not re-run**.
- **Still open from this step:** `aggregator_steam_17mafo` (2,169,483 rows, 2026-04-16 →
  2026-07-10) is voted unfiltered and uninvestigated; if it is not an ask, the contaminated
  window predates 2026-07-11 by four months. And `ab_test_regime.py` /
  `ab_test_ensemble.py` still vote the bid through their private loaders — folded into step 5.
- §10 Tier 1 #3.

---

## No retrain, ~two weeks (steps 2–6)

### 2. Replace DA with serial-correlation-robust Pesaran–Timmermann — **DONE 2026-08-07**

Landed as specified: per-date excess hit rate, Newey–West HAC t-stat over dates, hurdle
`|t| > 3.0`, ≥$1 cohort, `n_dates ≥ 20`. `backend/backtest/directional_test.py` is the new pure
module; `pt_verdict` ∈ `skill` / `no_skill` / `perverse` / `insufficient_dates` / `degenerate` is
stored in `prediction_accuracy.metrics`, and a **significantly negative statistic is reported at
warning level as a finding** rather than folded into the null. DA now ships only as the triple
(`constant_call_accuracy`, `constant_call_direction`, `realised_down_rate`). New endpoint
`GET /accuracy/headline`; the homepage placard and `/accuracy` render the verdict, not the hit
rate. Record: `docs/changelog/2026-08-07-pesaran-timmermann-headline.md`.

**One thing the spec did not anticipate, and it is the important one:** a **constant call has
per-date excess identically zero**, because when the call never varies `P*_d` equals the realised
down-rate and cancels the hit rate term for term. So "always-down beats the model" resolves as
`degenerate` — t undefined, not t large — and the always-down straw man can never be scored as
skill. Also worth noting: `baseline_directional_accuracy` in the stored series was never the
constant-call baseline, it is the always-*flat* call; it keeps its name for continuity and the
real one sits beside it.

**Not done here:** nothing was re-scored (a `--rescore` will populate `pt_*` on existing rows, at
no archive cost), and no production verdict exists yet — every live cohort still spans 1–2
forecast dates, so all four horizons report `insufficient_dates`. The `ab_test_*` harnesses and
`walkforward_backtest.py` do not carry PT; they are step 5's problem.

- **Do:** PT as the headline, computed per forecast date with a t-stat over dates. Report raw
  DA only as a **triple** with the constant-call baseline and the realised down-rate beside
  it. Hurdle **t > 3.0** (Harvey–Liu–Zhu), ≥$1 cohort, `n_dates ≥ 20`. Report it when it
  fails — a significantly *negative* PT is a finding.
- **Why:** the realised down-rate swings hugely between forecast dates and the model's call
  does not — `2026-08-03-accuracy-is-clustered-by-forecast-date.md` measures always-down at
  **29.4% on 2025-12-01** and **76.9% on 2026-07-17** at 7d, against a model that says "down"
  57–87% of the time regardless of date. (The review quotes the swing as **32.7% → 76.9%** on
  the ≥$1 cohort; the 32.7% end is not sourced to a repo entry — see the changelog record.)
  That result *is* the PT null (Pesaran & Timmermann 1992, *JBES* 10(4), 461–465). Use the
  Blaskowitz & Herwartz (2014, *IJF* 30(1)) serial-correlation-robust variant, because
  carry-forward prices violate the plain version's assumptions.
- **Touches:** `backend/backtest/scoring.py`, `backend/api/routes/accuracy.py`, and the
  frontend surface that renders the headline.
- **Effort:** small — scoring module only, no retrain.
- **Unblocks:** every subsequent measurement. Until this lands, no accuracy number in the
  repo is interpretable. §10 Tier 1 #1.

### 3. Score by price tier, conditioned on `|predicted move| > round-trip cost` — **DONE 2026-08-07**

Landed as specified except for the two narrowings below. `price_tier` gained a cut at $1000
(six bands); `backtest/friction.py` holds the round-trip and spread constants;
`backtest/actionable.py` publishes `ActionableDA` at h ∈ {14, 30} as four numbers together
(`actionable_share_pct`, `actionable_da`, `actionable_e_net_pct`, and PT on the subset); and
`FLOOR_SWEEP = {-1: $1, -2: $5, -3: $20}` stores one `prediction_accuracy` row per floor so
"where does the headline stabilise" is auditable rather than a console line. `HEADLINE_TIER`
stays `-1` / `≥$1`, so `/accuracy/headline` and the placard are unchanged. Record:
`docs/changelog/2026-08-07-friction-conditioned-tier-scoring.md`.

**Two things the spec did not anticipate:**

1. **The spread measurement is at the wrong cuts.** The bands are `<$1`, `$1–10`, `$10–50`,
   `$50–500`, `$1000+` and `price_tier`'s cuts are 1/5/20/100/1000 — they do not align, so
   tiers 2 and 3 both borrow `$10–50`'s **17.3%** under a nearest-geometric-midpoint rule
   recorded in `SPREAD_SOURCE_BAND`. **These are not measured per tier and must not be cited
   as such.**
2. **Splitting tier 4 is a series discontinuity, not a refinement.** A stored row with
   `price_tier == 4` written before 2026-08-07 means `≥$100`; nothing migrates it, because the
   tier was all that was stored.

**Narrowed on purpose, both because the frozen-data-only scope was chosen to keep
`backtest/scoring.py` pure and `--rescore` archive-free:**

- **The staleness axis is 2 buckets, not 4.** The `actual_price == base_price` split each row
  already carries is a real staleness partition, giving a 6 × 2 grid. The quartiles need
  `stale_run_days` — **step 6's** deliverable. No quartile was invented.
- **`s_i` is the tier median, not the per-item BUFF spread.** That needs a frozen
  `buff_spread_rel`, and 25 days of bid history would leave it NULL on almost every stored
  outcome.

**Not done:** nothing was re-scored (`--rescore` populates the new keys at no archive cost, but
running it is an operational step); `MIN_SERVED_PRICE_USD` did not move, since raising the
serving floor is a decision that follows the sweep; and **no production verdict exists** — every
live cohort spans 1–2 forecast dates, so `actionable_pt_verdict` reads `insufficient_dates`
exactly as `pt_verdict` does. Also note the walkforward gate now carries `actionable_*` at
h=14/30 while still having **no purge and no embargo** (step 5), so its actionable numbers
inherit the same boundary-overlap inflation as its DA.

- **Do:** split `price_tier` above $100; report on a grid (price band × staleness quartile),
  never pooled; sub-$1 is diagnostic-only and never headline. Add the friction-conditioned
  metric `ActionableDA(v,h) = P(sign(r_act) = sign(r̂) | |r̂| > RT_v + s_i)` for
  **h ∈ {14, 30} only**, venue CSFloat (RT 2.0%), reporting `n_actionable / n_total`,
  `ActionableDA`, `E[net]` and PT on the subset together. Also test the headline at $1, $5
  and $20 floors and see where it stabilises.
- **Why:** the bid–ask spread runs **35.5% sub-$1 to 5.2% at $1000+** (n = 22,449), so a
  pooled DA number is uninterpretable and **a 3% predicted move is inside the spread for five
  of six tiers**. `price_tier` currently tops out at tier 4 = ≥$100, merging the 10.8%-spread
  and 5.2%-spread cohorts — the two most different liquidity populations in the market.
  Round trips: Steam **+16.1%**, CSFloat/DMarket **+2.0%**, Skinport **+8.7%**.
- **Expected result:** failure, on `n_actionable` first. The review says that is the
  publishable internal result and belongs at the top of any document describing what this
  product is.
- **Touches:** `backend/backtest/scoring.py`, `backend/api/serving_policy.py`, the tier
  mirror.
- **Effort:** small.
- **Unblocks:** an honest statement of what the product is. §10 Tier 1 #2.

### 4. Drop the 110 Doppler names from the item universe — **DONE 2026-08-08**

**Dropped, not split.** `PHASE_COLLAPSED_SLUG_PATTERNS` in `backend/models/item_parser.py`,
applied at `_fetch_voted_price_history` (the one read behind both `train()` and `predict()`),
both `walkforward_backtest` loaders and `api/routes/opportunities.py::_load_items`;
`VOTED_CACHE_VERSION` bumped 2 → 3. Splitting by phase needs the BUFF `doppler` sub-object
ingested as its own daily series — a collection project, not a filter.

Measured against the archive rather than the dump: the word matches **129 slugs / 47,081
rows** (0.309% of slugs, 0.227% of rows), of which **two are false positives** —
`Sticker | Doppler Poison Frog (Foil)` and its Sticker Slab twin are ordinary single assets,
hence a `sticker` exemption. In the ≥$1 served cohort the cut is **6 items of 926 (0.65%)
and 7,560 item-days of 994,432 (0.76%)**; the other ~121 real names are absent from the
cohort because they are not in the gated pool, not because of the price floor. This is a
correctness fix on a 0.65% slice, an order of magnitude under the 2.21–3.69pp MDE floor — no
A/B was run and none should be. Record:
`docs/changelog/2026-08-08-phase-collapsed-names-dropped.md`.

- **Still open from this step:** the ten-plus `ab_test_*` harnesses glob the archive
  privately and still see the names, so their universe now differs from production's —
  folded into step 5, which already owns those loaders. Stored forecasts and outcomes were
  not purged, and `price_resolution.py` was deliberately left unfiltered so pending outcomes
  stay resolvable; they drain from the scored cohort within 30 days.

<details>
<summary>The original entry</summary>

- **Do:** one filter. Drop them, or split them by phase using the `doppler` sub-object the
  BUFF dump already carries.
- **Why:** 29 base names collapse **181 distinct `paint_index` assets**. Median max/min phase
  ratio within one name is **3.25×** (p90 6.08×, max 23.5×); **87.3%** have >2× internal
  dispersion; the BUFF headline price **is** the cheapest phase **95.5%** of the time. So the
  series steps whenever the cheapest phase changes — a level shift with no asset repricing.
  Worst case `★ StatTrak™ M9 Bayonet | Doppler (Minimal Wear)` spans **$1,261 → $29,685**
  under one name. This is corrupting labels today, and it would contaminate the hedonic index
  (step 8) before it is built.
- **Touches:** the item-universe filter in the training path and the archive read.
- **Effort:** tiny.
- **Unblocks:** step 8. Review §22 D9, §25.

</details>

### 5. Purge and embargo everywhere at `H + 13` days; add `ingested_at` — NOT STARTED

- **Do:** turn `walkforward_backtest.py --purge` **on by default** and accept the
  discontinuity in the published series. Widen `_purge_overlapping_train_rows` from `H` to
  `H + 13`. Fix the ten `ab_test_*` harnesses that have neither purge nor fold clustering
  (13 exist; three were fold-threaded in `2026-08-07-training-item-universe.md`). Add an
  `ingested_at` column to `CANONICAL_PRICE_COLUMNS` while in there. **Also filter
  `BID_SOURCES` in `ab_test_regime.py` and `ab_test_ensemble.py`**, each of which carries a
  private `_load_all_prices` glob with no source filter — deferred here from step 1, and
  until it lands no A/B on those two harnesses is clean.
- **Why:** `--purge` is **default OFF**, so the *published* Backtest Accuracy number is
  unpurged. The review's measured inflation: the event-calendar arm at h=30 went **+12.1pp
  unpurged → +6.1pp purged** — half the effect was boundary overlap. *(That pair appears only
  in the review; no changelog entry carries it. Unreplicated here.)* The embargo rule:
  `H + LAG_TOLERANCE_DAYS(3) + SMOOTH_WINDOW(3) + MAX_WINDOW_SPAN_DAYS(7)` ⇒ **h=3 → 16d,
  h=7 → 20d, h=14 → 27d, h=30 → 43d**. At h=30 the embargo exceeds the validation window;
  that is the correct cost, not a bug. Separately: there is **no arrival timestamp anywhere**
  in the price archive, six backfill writers exist, and the column cannot be added backwards.
- **Touches:** `backend/scripts/walkforward_backtest.py`, `backend/scripts/ab_test_*.py`
  (ten of thirteen), `_purge_overlapping_train_rows` in `backend/models/forecaster.py`,
  `CANONICAL_PRICE_COLUMNS` in `backend/db/archive.py` (+ `backend/scripts/append_to_parquet.py`
  and `backend/scripts/normalize_price_schema.py`, which both have to agree with it).
- **Effort:** small, plus a schema migration through `aggregator-update.yml` (only CI writes
  the canonical archive).
- **Unblocks:** any A/B result being citable. §10 Tier 1 #4 and #5b; §18 L1.

### 6. Drop or downweight frozen-price runs from the label set — NOT STARTED

- **Do:** remove item-days from the **label set** at run ≥2 of bit-identical prices, or
  downweight by `1/(1+run_len)`. Compute `stale_run_days` while doing it — it is the same
  scan, it is free and 13 years deep, and it is the decisive test for step 11's staleness
  caveat.
- **Why:** bit-identical `actual_price == base_price` runs **37–42% at tier 0** and
  **0–1.8% at every tier ≥$1** (`docs/superpowers/specs/2026-08-05-cv-cohort-parity-design.md`);
  by `base_price` band it is **71.9% below $0.05** falling to **0.2% at ≥$5**
  (`2026-08-07-training-item-universe.md`). That is the Getmansky–Lo–Makarov MA(k) mechanism,
  and it is the only action on this list that **improves the measurement instrument itself** —
  it lowers the MDE for every future experiment. The current fold-clustered item-level floor
  is **2.21–3.69pp** (`2026-08-07-training-item-universe.md`) against ship gates written
  around 0.5–1.5pp.
- **Touches:** `prepare_targets` / the label path in `backend/models/forecaster.py`,
  `backend/backtest/price_resolution.py`.
- **Effort:** small.
- **Unblocks:** everything downstream, by shrinking label noise. Review §22 D2.

**After step 6 the system reports honestly. That is the point of stopping the count here.**

### Also no-retrain, from §10 Tier 1 but not in the review's own 1–11 ordering

- **5c. Fix the Steam listing-page backfill to use the real cent-ceiling schedule** —
  NOT STARTED. The 1.1607 constant is **synthetic**: measured flat at **1.1606–1.1607 across
  four orders of magnitude, IQR 0.0002** over 63,767 matched pairs, where theory must swing
  ~1.67 at $0.03 to ~1.15 at $50; **91.61% of pairs are the same numbers after dividing**, and
  the bottom three deciles read ratio **0.44** (buyer below net, impossible under any fee).
  Rows below ~$0.50 carry a real basis error, ~5% too high at $0.10–0.25. Touches
  `backend/scripts/backfill_steam_listing_history.py`. Effort: small. §10 Tier 1 #5 / C1.
- **6b. Grep for `api.dmarket.com/exchange/v1`** — NOT STARTED. Returns **410 Gone**. If it
  is present it is a dead integration; the live path is `/marketplace-api/v2/offers`. Effort:
  tiny. §10 Tier 1 #6.
- **Free bug fix, zero new data:** `distance_to_support`, `distance_to_resistance` and
  `high_low_range_30d` are computed every run and silently discarded, because `_feature_group`
  matches prefix `support_` while the columns are named `distance_to_*`. One line. Review §16.

---

## Retrain required (steps 7–11)

### 7. Ship `TRAIN_MIN_MEDIAN_PRICE` at a `TRAIN_FEATURE_ROWS ≥ 1.0M` budget — NOT STARTED

- **Do:** set the knob. **Re-derive the result first with a per-fold price filter** — the
  current version selects on a full-sample median.
- **Why:** the one surviving positive, **paired +3.50pp [+1.56, +5.98] at 30d**, null at
  3/7/14d (`2026-08-07-training-item-universe.md`). More importantly it **removes item-draw
  variance entirely** rather than shrinking it — the 99-item subsample's seed alone moves
  `acc_ge1` by sd **1.5–3.1pp**. The spread data (35.5% sub-$1) argues the honest floor is
  **above** $1, not at it.
- **Caveat that must be resolved first:** `_filter_by_median_price` runs on the **entire
  window before any split** — "items whose median 2013→2026 price is ≥$1" was not a set
  anyone could have named in 2019. The effect appears **only at h=30**, which is the
  signature a look-ahead selection produces (§18 L2).
- **Touches:** `TRAIN_MIN_MEDIAN_PRICE` and `TRAIN_FEATURE_ROWS` in
  `backend/scripts/forecast_prices.py`; `_filter_by_median_price` in
  `backend/models/forecaster.py`.
- **Effort:** small; **+7 min Monday-only**.
- **Unblocks:** a training cohort that matches the served one. §10 Tier 1 #7.

### 8. Build a hedonic market index — NOT STARTED

- **Do:** `log price ~ attributes + time dummies`, **refit per fold inside the purge
  boundary**. The time dummies are the index. Item attributes move to a separate `item_attr`
  dimension table for this consumer.
- **Why:** there is no composition-controlled market factor today. The demeaning experiment
  used a same-day median over a set that **grew 5,542 → 41,725 items mid-sample**, so a
  composition change is indistinguishable from a return. The collectibles literature settles
  the estimator (Bocart & Hafner 2018 *Econometrics* 6(3) 32; Fogarty 2011 *AEP* 50(4)
  147–156 — the hybrid index is most efficient and least volatile). **No CS2 index with a
  stated methodology has been published**, so this is novel work and a user-visible artifact.
- **Leakage risk:** high if mis-built — time dummies fit in-sample across the panel. Must be
  re-fit per fold inside the purge.
- **Depends on:** step 4 (Doppler names would contaminate it) and step 5 (the purge).
- **Effort:** medium.
- **Unblocks:** step 9. §10 Tier 2 #8.

### 9. Model the date-level market factor — NOT STARTED

- **Do:** forecast the aggregate market return (or the down-rate) for date *t+h* as its own
  model, then apply it uniformly. Keep it in a separate `market_day` table, not a `GROUP BY` —
  the market factor is currently recomputed in `_apply_market_aggregates`,
  `models/market_factor.py` and each `ab_test_*.py` independently.
- **Feed it:** the expensive→cheap lead (below), `down_rate` (per-date fraction negative),
  `breadth` (fraction above 30d MA), `dispersion` (cross-sectional sd of returns), and the
  LLM supply-shock flag from the `ISteamNews` text `ingest_steam_news.py` already caches.
- **Why:** item-level idiosyncratic signal is **measured absent** — demeaning drops DA below a
  constant call at all four horizons, 36.7/32.7/34.6/39.0 against a majority-class baseline of
  38.8/42.7/46.4/51.7 (`2026-08-06-market-relative-labels-refuted.md`). Date-level variance is
  **enormous and measured** — always-down scores 29.4% on one stored date and 76.9% on the
  other. Date-level MDE is **~0.3pp** against
  the **2.21–3.69pp** item-level floor — an order of magnitude better, and the exact figure
  `accuracy-opportunities.md` declared unreachable. **4,735 days of history.** The review
  notes this is silently 80–100% of the answer the model already gives and is **not modelled
  at all**.
- **Caveats:** an aggregate index is far more autocorrelated than item returns, so the
  purge/embargo discipline matters *more*, not less. A market-timing model right 55% of the
  time is a weaker product than a per-item pick.
- **Depends on:** steps 5 and 8.
- **Effort:** medium.
- **Unblocks:** the entry criterion for every ADVANCED technique in the review (§21: "the
  date-level target must have produced a statistically honest positive first"). §10 Tier 2 #9.

### 10. `lambdarank` within date, scored by rank-IC — NOT STARTED

- **Do:** objective swap. LightGBM already ships `lambdarank`. Metric is rank-IC / Spearman
  per date plus IC-IR; the honest bar is **IC-IR > 0.05**, not 0.5. Pair it with a within-date
  rank transform of every surviving feature (`groupby("date").rank(pct=True)`) — free, removes
  the market factor from the feature side by construction.
- **Why:** this is **not** the market-relative-label experiment already refuted. Demeaning
  changed the *label* and left a pointwise loss fighting a noisy residual; ranking changes the
  *loss*, so the model is never asked to predict a magnitude it cannot know, and the date
  effect is quotiented out structurally with no residual to swamp. Poh, Lim, Zohren & Roberts
  (arXiv:2012.07149) show LTR beats regress-then-rank on rank-IC **especially at low SNR with
  heavy-tailed noise**. The review rates it the highest-expected-value untried experiment, with
  the honest caveat that its own evidence says there may be nothing to rank.
- **Touches:** the objective in `backend/models/forecaster.py`, plus a new metric.
- **Effort:** small — objective swap.
- **Unblocks:** the "relative part" of the user question ("among ≥$1 items, this is
  top-decile"), which has never been run in a form the loss did not confound. §10 Tier 2 #10.

### 11. Only then, the cheap item-level candidates — NOT STARTED

Each is free, none needs a new source. All are §24 entries; none is expected to clear the
2.21–3.69pp item-level floor on its own.

| Candidate | Why | Note |
|---|---|---|
| **Lagged expensive-tier return → cheap-tier return** | **The only measured positive.** Lag-1 corr **+0.213, z = 9.1**, Granger incremental R² **9.0%**, stable 4 of 5 years, survives market-factor removal (0.122, R² 4.5%). Cheap→expensive is +0.043, inside noise | Belongs in step 9's **date × tier** frame, not here. Validate against staleness by dropping high-`stale_run_days` days — cheap skins have the highest zero-change rate (1.33% vs 0.16%) and the objection is *not fully closed* |
| `stale_run_days` | Highest-ranked new item-level feature; the GLM θ in discrete form | Free, 13 years deep. Already computed as a side effect of step 6 |
| `roll_spread` (Roll 1984) | The **only** liquidity variable available for the 13 pre-multi-source years and the 24,000 items with no bid feed | Free |
| `stattrak_premium_z30` | A revealed-preference weapon-usage measure, the only one available free: AK-47 **2.15×** vs P2000 **1.10×**; median ST/normal 1.43×, 14.7% below 1.0× | **z-score only, never the level** — the level is a shelved dollar-scale proxy |
| `float_range_capped` | **1,683 of 2,106 skins (79.9%)** have a truncated float range; capped finishes held through the Oct-2025 crash while easy-cap ones bled into 2026 | ByMykel `float_min/max`, free/MIT. Static |
| Cross-sectional reversal | The documented cross-sectional effect in digital collectibles is **reversal, not momentum**; the shelved price primitives were momentum-flavoured | One feature. Borri/Liu/Tsyvinski is SSRN 4052045, **working paper — do not cite as published** |
| `crate_id` as a **grouping** variable | Same-crate-different-skin residual corr **+0.084** vs **+0.003** different-crate — a **28× ratio**; Gallery 0.345, Kilowatt 0.104, legacy ≈ 0. 687 of 691 items join | **Not a column.** This is a covariance-structure fact, not a mean-prediction one — use it for hierarchical shrinkage / block-bootstrap units. Does not contradict the ByMykel refutation |

**Blocked, do not attempt yet:** the whole liquidity family that needs paired bid/ask
(`d_bid − d_ask`, `spread_resid`, `spread_change_7d`, `bid_side_depth_proxy`) has **25 days
of history**, which at `CV_STEP_DAYS = 150` produces **zero additional folds**. Case EV is
blocked the same way — **128 usable days**, because 78.7% of contained items have their first
priced day on exactly 2026-03-22. `usd_cny` has **7 days** of FX history.

---

## What I would not do at any point

Verbatim from the review's closing, plus the §10 "Do not" and §13 out-of-scope lines. These
are here so they are not re-proposed.

- **No further cosmetic item-metadata bundle.** ByMykel is refuted — the model's own
  permutation test shows the features go unused.
- **No float, paint-seed, applied-sticker or applied-charm data.** Not "expensive" —
  **structurally invisible** at the `market_hash_name` key, and each one reintroduces exactly
  the composition artifact that killed CSFloat `avg_price`. (`CS2BlueGem` is paid and useless
  at this grain regardless.)
- **No sentiment rebuild.** Refuted here, deleted, and runner IPs get 403s. The literature is
  a swamp of positive-result low-rigour single-regime studies; the closest thing to a rigorous
  negative (Bijl et al. 2016, Google search volume) finds the **opposite sign** to the
  original result.
- **No transformer, TFT, N-BEATS or LSTM.** Closed on evidence (§9) *and* mechanism (§11).
  M5: LightGBM used by all of the top 50 in both tracks. Grinsztajn et al.: trees remain SOTA
  on tabular data at ~10K-sample scale.
- **No per-item ARIMA/GARCH and no item/collection embeddings.** The first is behind the
  published work (Montero-Manso & Hyndman); the second would learn item identity against a
  percentage target — the defect that shelved 37 dollar-scale columns, in a form no property
  test catches.
- **Nothing paid.** Out of scope by constraint. Also note cs2.sh advertises Steam data back to
  2013 — the depth you would pay for is what the Steam `pricehistory` endpoint gives away.
- **No "manipulation score" feature.** It would fire on <1% of item-days against a 2.21pp MDE,
  return null, and — since trees are robust to uninformative features — return null *quietly*.
- **Do not winsorise large daily returns.** The 2025-10-22 cross-section is the only dated
  event where item attributes dominated the market factor; clipping deletes the one
  observation carrying that information. Use cross-source confirmation instead of a magnitude
  threshold.
- **No new model class, generally.** §2, §9, §11, §25 and §26 converge, and the convergence is
  the finding.

---

## Open questions this list does not answer

- The **three Appendix refutations** (CSFloat, ByMykel, training-breadth) still have
  date-clustered intervals and have **not** been re-derived. Probably still correct in sign;
  the intervals are wrong. Any citation should say so.
- The **A/B harness is not reproducible run-to-run** (`mean_diff_pp` −0.1581 → −0.0026 on
  identical commands, with `n_paired` and `n_dates` also moving). Best candidate cause is a
  changing ask-source set with no `n_ask_sources` column to detect it — the mean market return
  reads **−31.6% on 2026-03-22** and **+17.4%/−17.8% on 2026-07-09/10** against ±0.5% on a
  normal day. Hypothesis, not diagnosis. Step 1 removed one measured contributor to it — the
  bid's rejection flickered on **5.6%** of return pairs — but flicker was only a twentieth of
  that defect's effect, so this is not resolved.
- **h=30 rests on 5,461 usable rows from one backdated date.** It carries both the +3.50pp
  positive (step 7) and the largest unpurged inflation (step 5). No h=30 claim until ≥30
  forecast dates mature.
- **`|r| < 0.002`** for trade volume is quoted in nine documents and is **10–40× too small**:
  C4 measures pooled corr(vol z, fwd7) = **+0.019**, fwd30 = **+0.034**, item-fixed-effects
  identical, $1–10 tier **+0.080** at 7d, on 4.46M rows 2023–2025. The audit's *conclusion*
  survives (r² < 0.15%, economically trivial); its stated *reasoning* does not. Not corrected
  anywhere yet.
