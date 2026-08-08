# Next steps from the 2026-08-07 research review

**Source:** `docs/research/2026-08-07-cs2-forecasting-research.md`, section
"If I were building this myself", cross-referenced against §10 "Ranked recommendations".
**Record:** `docs/changelog/2026-08-07-cs2-forecasting-research-review.md`.
**Status: steps 1, 2 and 3 are DONE (2026-08-07 —
`docs/changelog/2026-08-07-bid-source-excluded-from-voting.md`,
`docs/changelog/2026-08-07-pesaran-timmermann-headline.md` and
`docs/changelog/2026-08-07-friction-conditioned-tier-scoring.md`), and steps 4 and 5 are
DONE (2026-08-08 — `docs/changelog/2026-08-08-phase-collapsed-names-dropped.md` and
`docs/changelog/2026-08-08-embargo-and-harness-hygiene.md`; step 5's outstanding "gate not
re-run" item closed the same day —
`docs/changelog/2026-08-08-embargo-discontinuity-measured.md`), and step 6 is DONE (2026-08-08 —
`docs/changelog/2026-08-08-frozen-price-runs-dropped-from-labels.md`, which also refuted the two
prevalence figures step 6 was justified on and surfaced the un-taken MA-feed voting fix).
Steps 7–11 are NOT STARTED. **6b** is DONE (verified absent, 2026-08-08) and **R11** and **R12**
are DECLINED (2026-08-08 —
`docs/changelog/2026-08-08-r11-r12-declined-and-r18-r19-recosted.md`, which also re-costed R18
and R19 and opened **5d**, a blocker on the Steam listing backfill that R11, R13 and 5c all
assumed was merely un-run).**

Ordering is the review's, not a re-ranking. Numbers in the "why" column are quoted from the
review or from the changelog entry that measured them; nothing here is estimated.

**The 1–11 ordering is not the whole of §10.** It follows the review's "If I were building this
myself", which omits six items §10 itself ranked; they are tracked in
**"From §10 Tier 2 and Tier 3, ranked by the review and never tracked (R11–R19)"** below. One of
them, **R19**, is step 7's entry criterion.

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

### 5. Purge and embargo everywhere at `H + 13` days; add `ingested_at` — **DONE 2026-08-08**

Landed as specified, in five parts. `models/forecaster.py::embargo_days(horizon)` is the
band, and the 13 is **derived at call time** from `LAG_TOLERANCE_DAYS` (3) +
`SMOOTH_WINDOW` (3) + `MAX_WINDOW_SPAN_DAYS` (7) rather than typed;
`_purge_overlapping_train_rows` and every `_compute_cv_splits` caller use it.
`walkforward_backtest.py` embargoes **by default** — `--purge` became `--no-purge`, and the
discontinuity in the stored `lgbm-v3-clustered` series was accepted rather than avoided.
`ingested_at` is in `CANONICAL_PRICE_COLUMNS` as a `TIMESTAMP`, stamped by
`append_to_parquet.py` with the run's wall clock, first-arrival-wins on a re-append. All
thirteen `ab_test_*` harnesses now read production's universe through the new
`archive_universe_sql_filter`, the six with **no embargo at all** got one, and the ten with
no significance test got a paired, fold-clustered interval. 1,466 tests pass, up from 1,258.
Record: `docs/changelog/2026-08-08-embargo-and-harness-hygiene.md`.

**Three things the spec did not anticipate:**

1. **`BID_SOURCES` was not the harnesses' main defect — the Doppler names were.** Nine of
   the twelve archive-reading harnesses already filter to a *single ask source*
   (`aggregator_sync` or `STEAMCOMMUNITY`), which excludes the bid by construction. The
   step's framing (bid filtering for `ab_test_regime.py` / `ab_test_ensemble.py`) was right
   about those two, and about `ab_test_supply_side.py`, which it did not name. What all
   twelve were missing is step 4's phase-collapsed filter.
2. **Those three harnesses were reading a raw glob, not a filtered one.**
   `read_parquet('prices-*.parquet')` narrows to the first file's schema, where `source`
   does not exist at all — so their `source = 'STEAMCOMMUNITY'` item subquery worked only
   against an already-migrated archive. They now go through `prices_relation`.
3. **The embargo width follows an environment variable.** `MAX_WINDOW_SPAN_DAYS` derives
   from `collectors.pipeline.FALLBACK_MAX_AGE_DAYS`, which is env-overridable. That is the
   deliberate single-staleness-convention coupling, but it means the fold geometry is not
   a constant.

**Not done, and each of these matters for how the result is read:**

- ~~**No harness was run.**~~ — **three of thirteen re-run 2026-08-08**, and only after fixing
  a defect the migration below introduced: the harnesses' `source` filters were not NULL-safe,
  so every migrated pre-2026 file matched nothing and the universe went to **0 items (876
  after the fix)**. It surfaced as three unrelated-looking errors — `IndexError`, a DuckDB
  parse error on `IN ()`, a stale cache fingerprint — and **not one of them said "zero rows"**
  (`e930850`). The three re-derivations all survive; see the "Open questions" entry below and
  `docs/changelog/2026-08-08-migrated-archive-emptied-eight-harnesses.md`. **The other ten
  harnesses are still un-run**, so every A/B they carry predates all five changes.
- ~~**The archive migration has not been run.**~~ — **run locally 2026-08-08**; all 21
  `price-archive/prices-*.parquet` files now carry `source` and `ingested_at`. Two things that
  does not mean. `ingested_at` is **populated on 361,525 of 21,842,207 rows (1.7%)**, only the
  ones appended since the writer changed — it is a typed NULL on everything older, which is
  the correct migration and not a fix. And the local archive is **not** the canonical one:
  only CI writes `RayanR000/cs2-oracle-data`, so whether the *published* archive is migrated
  is unverified here.
- ~~**The published gate has not been re-run under the new default**~~ — **DONE 2026-08-08**,
  and the discontinuity is measured. Record:
  `docs/changelog/2026-08-08-embargo-discontinuity-measured.md`. Two things had to be fixed
  or found first: the gate had **never persisted a row** (a session closed 30 min before the
  write, `f63ae76`), and there was consequently **no `lgbm-v3-clustered` series to step
  from**, so the contrast was run as a paired `--no-purge --skip-db` arm instead. Unpurged
  inflation on the ≥$1 headline cohort, positive = unpurged reads higher: **h=3 +0.47pp
  [−1.05, +2.06] (null, MDE 1.55)**, **h=7 +4.29pp [−0.26, +9.14] (null, MDE 4.70 — the
  point estimate is inside its own MDE, so unresolved, not clean)**, **h=14 +5.44pp
  [+1.68, +9.12] (positive)**, **h=30 +10.15pp [+5.00, +15.79] (positive)**. At h=30 the old
  split read 63.14% and the embargoed one reads 52.92% — which loses to the 52.97% always-up
  constant call. **The +12.1pp → +6.1pp pair quoted below is retired**: there is now a local
  measurement of the same quantity and the review's figure should not be cited.
- **The three raw-glob harnesses still read every ask source**, so a 2026 item-day reaches
  them ~11 times and `engineer_features` collapses the copies with a plain mean where
  production votes an outlier-rejected median. Narrowing that cohort changes what they
  measure and was left alone.

<details>
<summary>The original entry</summary>

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

</details>

### 6. Drop or downweight frozen-price runs from the label set — **DONE 2026-08-08**

**Dropped, not downweighted**, on both legs. `models/staleness.py` is the new pure module;
`LABEL_MAX_STALE_RUN_DAYS = 0` voids in `prepare_targets` inside the same `bad` mask as
`_snapshot_dates` (a gap > `MAX_WINDOW_SPAN_DAYS` breaks a run rather than continuing it);
migration `0021` freezes `forecast_outcomes.base_stale_run_days`; `score_by_staleness`
publishes the axis; `scripts/ab_test_frozen_runs.py` is the verification. 1,532 tests pass,
up from 1,466. Record:
`docs/changelog/2026-08-08-frozen-price-runs-dropped-from-labels.md`.

**Both numbers in the "why" below are wrong as applied, and that is the step's main result:**

1. **"0–1.8% at every tier ≥$1" measures resolved, 3-day-SMOOTHED anchors**, which are almost
   never bit-identical. The raw voted series the label path sees is **12–27%** at ≥$1. Both are
   real; neither sizes the other, and a citation must say which.
2. **The rate is a 2026 feed property, not a market fact** — 0.5–0.8% through 2025 against
   6–33% across 2026. So the rule voids **~30% of 2026 ≥$1 labels and ~0.6% of pre-2024 ones**:
   a 2026 filter wearing a 13-year mask. Defensible, since 2026 is what production trains and
   serves on, but not the even cleaning implied here.

**It is also not surgical:** it takes **13.7–15.9% of *non-zero* ≥$1 labels** with it, so it can
be net-harmful. The verification is therefore on paired interval **width** (the seed-only
placebo floor, once per label regime), not on accuracy.

**A new finding this step surfaced, and did NOT act on** — `aggregator_steam_7d/30d/90d` are
Steam's trailing-window **mean sale price** voting against point-in-time asks, i.e. the MA(k)
mechanism as an ingest decision, a basis error by the same argument that removed the bid. It is
**not** a substitute for the run-length rule: it clears only **2.30pp of the 20.25pp** ≥$1 stale
rate, because the smoothing is in the *fields* — `aggregator_sync` and `aggregator_steam_17mafo`
are `last_24h` falling back to those same windows, and **there is no point-in-time Steam price in
this archive**. Excluding the three windows is nearly free on coverage (670 item-days of 3.09M)
but moves the voted median on **17.13%** of 2026 ≥$1 item-days (median **−7.16%**) and flips
**5.75%** of return directions — half the bid's magnitude, same character. It needs its own step,
its own `VOTED_CACHE_VERSION` bump, and it **subsumes the `aggregator_steam_17mafo` item open
from step 1**. Dropping `aggregator_sync` as well would delete 2026-01 and 2026-02 in full —
don't.

- **Not done:** nothing re-scored or re-resolved, so `base_stale_run_days` is NULL on every
  stored outcome and `staleness_bands` reads 100% `unknown` until new outcomes mature; no
  retrain, so the rule first bites at the next Monday `mode=full`; and `stale_run_days` is
  **not** a model feature — that is step 11, and it is **leak-adjacent**, since the 2026 source
  mix changed four times and a run-length feature partly encodes the collection schedule.
- **Overlaps step 7, and not additively.** `TRAIN_MIN_MEDIAN_PRICE = 1.0` removes the sub-$1
  items outright; this rule drops ~39% of labels on the whole universe against 13.8–19.2% on the
  ≥$1 subset. Once step 7 lands, step 6's effect falls to roughly a third. Do not sum them.

<details>
<summary>The original entry</summary>

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

</details>

**After step 6 the system reports honestly. That is the point of stopping the count here.**

### Also no-retrain, from §10 Tier 1 but not in the review's own 1–11 ordering

- **6c. Stop the Steam rolling-window feeds voting in the consensus** — NOT STARTED, and it is
  **not** in the review; it was found while building step 6 (2026-08-08).
  `aggregator_steam_7d/30d/90d` are Steam's trailing-window **mean sale price** — MA(7)/MA(30)/
  MA(90) — voting on equal terms against point-in-time asks
  (`collectors/csgotrader_aggregator.py:308-312`, `collectors/pipeline.py:132-135`). Same class
  of basis error as `aggregator_buff163_buy`, which step 1 removed. **Do not scope it as a
  staleness fix**: it clears only 2.30pp of the 20.25pp ≥$1 stale rate, because the smoothing is
  in the *fields* — `aggregator_sync` and `aggregator_steam_17mafo` are `last_24h` **falling
  back** to those same windows, which fires on exactly the illiquid items, and there is **no
  point-in-time Steam price in this archive at all**. Cost: nearly free on coverage (670 lost
  item-days of 3,093,793 on the ≥$1 cohort) but it **displaces the level** — the voted median
  moves on **17.13%** of 2026 ≥$1 item-days, median **−7.16%** where it moves, and **5.75%** of
  consecutive-day return directions flip. Half the bid's magnitude, same character, so it needs
  its own step, its own changelog entry and a `VOTED_CACHE_VERSION` bump, and every A/B and label
  from 2026-03 on would sit downstream of it. **It subsumes the `aggregator_steam_17mafo` item
  still open from step 1.** Do **not** also drop `aggregator_sync`: that deletes 2026-01 and
  2026-02 in full for the ≥$1 cohort (52,048 item-days) to buy a further 1.4pp — the fix there is
  upstream, recording which field the fallback chain actually used. Effort: small, plus a
  re-vote. Record: `docs/changelog/2026-08-08-frozen-price-runs-dropped-from-labels.md`.
- **5c. Fix the Steam listing-page backfill to use the real cent-ceiling schedule** —
  NOT STARTED. The 1.1607 constant is **synthetic**: measured flat at **1.1606–1.1607 across
  four orders of magnitude, IQR 0.0002** over 63,767 matched pairs, where theory must swing
  ~1.67 at $0.03 to ~1.15 at $50; **91.61% of pairs are the same numbers after dividing**, and
  the bottom three deciles read ratio **0.44** (buyer below net, impossible under any fee).
  Rows below ~$0.50 carry a real basis error, ~5% too high at $0.10–0.25. Touches
  `backend/scripts/backfill_steam_listing_history.py`. Effort: small. §10 Tier 1 #5 / C1.
- **5d. Two things stand between the Steam listing backfill and a resumed run** — NEW
  2026-08-08, found while re-deriving R11. It **gates R11, R13 and 5c**.
  1. **`load_targets` raises on `--min-price` since the schema migration.** It filters
     `HAVING MAX(median_price)`, and the 2026-08-08 normalisation left the archive with
     `mean_price` — so the script's own documented usage line `--min-price 1.0` dies on a DuckDB
     `BinderException`. It also reaches the archive through a raw
     `SELECT * FROM read_parquet('prices-*.parquet')`, which is invariant 1 in
     `backend/AGENTS.md` and is exactly the mechanism by which a renamed column disappears
     without an error. One-line fix. Target counts on the 2026-08-07 archive day once it is
     fixed: **32,617** non-gated name-keyed items with no floor, **27,935** at ≥$1 (35,766 /
     28,594 before the mangled-key filter drops 3,149 / 659 phantom slug rows) — which
     independently corroborates R13's extrapolated ~31,590.
  2. **This IP's soft-block has not decayed in three days.** The block is not new — it is dated
     **2026-08-05** with "no decay over 5 h" (`2026-08-06-data-acquisition-ranking.md`), and the
     staged DB stops there. What is new is that it is **still in force on 2026-08-08**, so the
     decay bound is now ≥3 days rather than ≥5 h. Re-measured today: a logged-out
     `GET /market/listings/730/<name>` returns 302 → **5.0 MB** carrying 20 embedded
     `pricehistory` caches (9 variants of one base name, daily `price_median` + `purchases` back
     to **2014-02-21**) for the first ~3–4 fetches, then flips to the shell — 200, **230 KB**, no
     redirect, zero `pricehistory` — and stays flipped at every idle interval probed
     (**60 s, 180 s, 360 s, 660 s**: shell every time, byte-identical to within one byte). The
     flip is **not header-shaped**: all three rotated user-agents and the `Accept` /
     `Accept-Language` / `Accept-Encoding` variants returned the shell once flipped, while the
     same request had succeeded a minute earlier. `classify()` calls it `SOFT_BLOCK` correctly,
     so `canary_ok` aborts **before request 1** — the guard is working, and the run genuinely
     cannot proceed from here.
  - **What the staged data already proves, and it matters for R11:** 47 requests harvested
     **262 items / 528,573 daily rows back to 2013-08-15** (`runtime/steam_listing_history.db`)
     — **5.6 series per request measured**, above the docstring's ~4.5. At the docstring's ~82%
     target share that is ~4.6 targets/request, so the 32,617 targets are **~7,100 requests**,
     ~5.7 h at `REQUEST_DELAY = 2.5` **if the block allowed it**. It does not, so the delay is
     not the binding constraint and no run should be planned on that figure.
  - Effort: tiny for (1). **(2) is an egress problem, not a code problem** — the same conclusion
    `2026-08-06-data-acquisition-ranking.md` reached — so the honest options are a different
    egress or a much slower schedule, and until one is demonstrated, "the Steam listing page is
    the one live onboarding route" describes a route that has delivered 262 of 32,617 items.
- **6b. Grep for `api.dmarket.com/exchange/v1`** — **DONE 2026-08-08: verified absent.** No
  occurrence anywhere in the repo. `dmarket` appears only as fee constants in
  `backend/backtest/friction.py` and its test, so there is no dead integration to remove and
  nothing depends on the 410'd path. §10 Tier 1 #6.
- **A one-line bug, but not a free win** — corrected 2026-08-08. `distance_to_support`,
  `distance_to_resistance` and `high_low_range_30d` really are computed every run and dropped:
  `_feature_group` matches prefix `support_` while the columns are named `distance_to_*`
  (`models/forecaster.py:201` against `:1416-1420`), so they group as `other` and
  `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` filters them out — confirmed by running
  `_apply_feature_allowlist` on them. **But the 2026-07-24 ablation that set that allowlist ran
  with these three already absent**, so "fixing" the prefix does not restore something measured;
  it admits **three never-measured features** to the model. One line plus an A/B against the
  2.21–3.69pp MDE, which three price-technical columns are unlikely to clear. Review §16.

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
  **Planned 2026-08-08:** `docs/superpowers/plans/2026-08-08-per-fold-price-filter.md`. Three
  things that plan found which are not stated here. (a) `ab_test_training_breadth.py:269-276`
  carries the **same leak** — `HAVING MEDIAN(...) >= 1.0`, `COUNT(DISTINCT day) >= 180` and
  `LIMIT 870` are all full-sample — so the harness that produced the +3.50pp cannot adjudicate
  it. (b) A per-fold filter must run *after* `engineer_features`, which silently moves the
  market factor from the ≥$1 universe to the pooled one, so it is two treatments unless held
  fixed. (c) It also runs after `_stratified_item_subsample`, which **destroys the budget
  argument step 7 exists for** — hence the plan derives per-fold but would ship an *anchored*
  filter. A third `per_fold_matched` arm is required, or "the leak is gone" and "there is less
  data" are the same observation. And the whole thing must be re-derived on the post-step-6
  label set.
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

## From §10 Tier 2 and Tier 3, ranked by the review and never tracked (R11–R19)

The 1–11 ordering above is the review's "If I were building this myself", which stops at §10
#10 plus step 11's candidate table. **Six items §10 ranked are absent from it.** They are
labelled by their §10 rank (`R…`) rather than renumbered, so they cannot be read as steps.
They were dropped in transcription, not declined. Order below is §10's ascending, not a
re-ranking; two are worth knowing about before reading in order, though — **R19 is step 7's
entry criterion**, and **R13 is described by the review as a larger lever than any feature on
the list**.

**Two are now decided (2026-08-08): R11 and R12 are DECLINED**, both on coverage/fold-count
grounds and neither on licence. R18 and R19 keep their rank but their effort estimates were
wrong in opposite directions — R18 is harder than "one query", R19 is cheaper than "build the
instrument". Record: `docs/changelog/2026-08-08-r11-r12-declined-and-r18-r19-recosted.md`.

Three Tier 3 rows *are* accounted for elsewhere and are deliberately not repeated here: **#15**
(per-item Getmansky–Lo–Makarov MA coefficient) is step 6 / step 11's `stale_run_days`, "the GLM
θ in discrete form"; **#16** (cross-sectional reversal) is in step 11's table; **#17**
("volume ↑ ⇒ price ↓") is downstream of R11, and volume features are shelved
(`2026-08-06-volume-features-shelved.md`).

### R11. Recover historical volume from the kieranpoc Kaggle dump — **DECLINED 2026-08-08**

**Declined on coverage, not on licence.** The dump is frozen: `dateModified` **2024-06-15**,
data snapshot **2024-05-04**, 901,195,556 bytes, CC BY-NC-SA 4.0 (verified on the page — the
review's licence mark was unconfirmed, and this is what it says). So it supplies **nothing for
2024-06 → 2026-08**, which is the only window that matters: the `volume` column's last non-zero
day is **2026-04-15**, and it is zero on 100% of rows from 2026-05 through 2026-08 (verified,
`prices-2026-*.parquet`). Both legs of its purpose are covered elsewhere —

- **Forward** sale counts: `collectors/sales_volume.py` (Skinport, wired 2026-08-08). *Not
  independently re-verified here — the endpoint answers **403 with a 422 KB HTML challenge** to
  this machine's egress, which is the WAF behaviour that module's docstring documents for
  Cloudflare-owned egress, so it is not evidence of breakage.*
- **Historical** sale counts: the Steam listing pages already carry them, and this is
  demonstrated rather than argued. `backfill_steam_listing_history.py` parses `purchases` per day
  beside `price_median`, and its staged DB holds **262 items / 528,573 daily rows back to
  2013-08-15** from 47 requests. The dump's own source is Steam, so the set of items reachable
  only through the dump is small by construction.

**What is not established, and it is why this is a decline rather than a swap:** the listing
route's *sustainable* rate. See **5d** — this IP has been soft-blocked since 2026-08-05 with no
decay in three days, so 262 of 32,617 items are in hand and the remainder needs a different
egress or a much slower schedule. That is a reason to fix the collection route, **not** a reason
to import a dump that stops 26 months before the window in question.

<details>
<summary>The original entry</summary>

- **Do:** backfill Steam price + volume from `kieranpoc/counter-strike-market-sale-data` —
  **22,492 items, 99.3M data slices**, hourly for the trailing month and daily before it, back
  to **2013**, snapshot as of **2024-05-04**.
- **Why:** the archive's `volume` column has been **identically zero since 2026-04-16**. The
  value is *not* return prediction — that is shelved and refuted — it is the **counting-noise
  denominator** (§24 rank 16, `sale_count_24h`). §11's central finding is that
  `AK-47 | Redline (Field-Tested)`, one of the most traded skins in the game, records **96
  Steam sales in 24 hours**, while the archive's own 2025 distribution medians **69
  sales/item-day overall, 20 at $50–500, 4 at $500+**, with **60% of $500+ item-days at 1–5
  sales**. Without a sales count there is no way to weight, screen, or even report which
  item-days are statistically empty — and the thin tier *is* the served cohort.
- **Caveats:** the review's own licence mark is **unconfirmed — verify on the page**. §24 is
  explicit that the dead column **must not be reused in place**. And the `|r| < 0.002` figure
  nine docs rest on is wrong in its reasoning: C4 measures pooled **+0.019 (7d) / +0.034
  (30d)**, `$1–10` tier **+0.080** at 7d, on 4.46M rows — the *conclusion* (r² < 0.15%,
  economically trivial) survives, so this is not a route back to volume as a predictor.
- **Effort:** medium. §10 Tier 2 #11.

</details>

### R12. Backfill retroactive supply depth from `atalantus` — **DECLINED 2026-08-08**

**Declined on fold count and splicability.** The listing-count window is 2023-01-25 →
2024-01-19, **359 days**. Against `CV_STEP_DAYS = 150` and `CV_MIN_TRAIN_DAYS = 200`
(`models/forecaster.py:489-490`) that is **~2 folds** — so it does technically break the
zero-additional-folds arithmetic this entry was written to break, and that is the strongest
thing that can be said for it.

What it cannot do is join the served present. Live supply depth begins **2026-08-06**
(verified: `supply-2026-08.parquet` holds that one day, 30,330 items), leaving a **2.5-year
gap**, across a **different venue** (BUFF, CNY) and a **different quantity** than the
lis-skins / market.csgo / Waxpeer counts. The two panels cannot be concatenated into one
series, so what R12 actually buys is a **history-only, ~2-fold A/B** — which against the
2.21–3.69pp item-level MDE resolves `unresolved` by construction, not `null` and not positive.
Nothing about the licence entered this decision. One correction to the source record while
here: the raw dump is **113 MB via Git LFS**, not the 24 MB xz that `data-sources.md` carried.

<details>
<summary>The original entry</summary>

- **Do:** take **listing counts only** from `atalantus/buff-price-history-archive` — BUFF163
  min price **2021-07-26 → 2024-01-19**, with the listing count populated after **2023-01-25**.
- **Why:** supply depth is accumulation-blocked, and the block is structural rather than a
  wait: **25 days of paired history at `CV_STEP_DAYS = 150` yields zero additional folds**
  (step 11's blocked list). Roughly a year of retroactive listing counts is the only thing on
  the table that changes that arithmetic. §24 rank 25 puts `supply_change_7d` at low-medium —
  "the level is ~0pp; only velocity was ever plausible" — at h=30.
- **Not the import that was declined.** The same repo's **price** dump was declined because it
  shrinks `target_items` **521 → 426** (§10, "what not to collect"). This is a different
  column and a different use, and the two should not be conflated in either direction.
- **Caveats:** licence is **none — all rights reserved**. The 2023-01 → 2024-01 window does
  not overlap the multi-source era, so it adds folds to *history*, not observations to the
  served present. And it carries listing counts, **not bids** — there is no retroactive bid
  anywhere (§16), so it does not unblock the liquidity family.
- **Effort:** medium. §10 Tier 2 #12.

</details>

### R13. Fix the cohort inversion — NOT STARTED

- **Do:** change *which items are served*, through the `is_backfilled` gate.
- **Why:** §1's table is the review's "most consequential product fact":

  | Cohort | ≥$1 | <$1 | % ≥$1 |
  |---|---:|---:|---:|
  | All items with recent data | 26,468 | 14,955 | **63.9%** |
  | Items actually forecast | 1,423 | 7,268 | **16.4%** |

  The archive is two-thirds dollar-plus; the served cohort is **84% sub-dollar** — the tier
  carrying the **35.5% spread** and **37–42% bit-identical carry-forward** rate. The review:
  *"the gate excludes most of the items that have tradeable signal and includes mostly ones
  that don't … a larger lever than any feature, and it is a data-plumbing problem
  (`is_backfilled`), not a modelling one."*
- **Half of it is already instrumented.** Step 3's `FLOOR_SWEEP` ($1 / $5 / $20) is exactly
  the read for the cheap half — raising `MIN_SERVED_PRICE_USD` removes the bad cohort without
  adding a single item, and step 3 deliberately left that decision to follow the sweep. The
  expensive half is *adding* the ≥$1 items the gate excludes, which runs into the dead
  onboarding path; the Steam listing-page route is the only live way in (extrapolated pool
  **5,542 → ~31,590 items**, `target_items` **521 → 758**), which is also why **5c** matters
  more than its size suggests.
- **The pool figure now has a direct measurement, and the route now has a blocker (2026-08-08).**
  `load_targets` counts **32,617** non-gated, name-keyed items actively collected on 2026-08-07
  (**27,935** at ≥$1), which corroborates the review's extrapolated ~31,590 independently. But
  see **5d**: **262 of those 32,617 are in hand**, the `--min-price` path raises on a column the
  schema migration renamed, and this IP's soft-block has not decayed in three days, so the canary
  aborts the run at request 0. **R13's expensive half is blocked on egress, not on effort.**
- **Touches:** the `is_backfilled` plumbing, `backend/api/serving_policy.py`.
- **Effort:** medium. §10 Tier 2 #13.

### R14. Mechanical supply-position features — NOT STARTED

- **Do:** trade-up **fuel vs output** position, and drop-pool status. The third member of §10's
  row, `float_range_capped`, is already in step 11's table.
- **Why:** 2025-10-22 is the one dated event where item attributes dominated the market factor,
  and what dispersed the cross-section was mechanical position, not cosmetics. Valve extended
  the trade-up contract to **5 Covert → 1 knife or glove**, unannounced; knives and gloves fell
  while **Coverts rose 10–20× in extreme cases, because they became trade-up fuel**. R6 is the
  same shape and case-specific: Valve silently zeroed the rare drop pool and **discontinued
  cases went +15–57% while weapon skins were unaffected**. The map is **free and static**
  (ByMykel, MIT), and §25 is explicit that this is a *different feature class* from the refuted
  cosmetic bundle.
- **The caveat that should keep it last:** §25 rates trade-up position "real (Oct-2025)" but
  firing **only on unannounced rule changes**. A feature informative on a handful of dates in
  13 years cannot clear the **2.21–3.69pp** item-level MDE; it belongs in step 9's date-level
  frame or as a conditioning variable, not as a per-item column. The standing no-winsorising
  rule is the same finding from the other side — clipping 2025-10-22 deletes the only
  observation that carries this information.
- **Effort:** medium. §10 Tier 3 #14.

### R18. Split conformal → adaptive conformal (ACI) — NOT STARTED

- **Do:** measure band coverage per tier **and per regime window** first, then adopt Gibbs &
  Candès (2021, *NeurIPS*) Adaptive Conformal Inference — α adjusted online from realised
  coverage error — only if it fails. Xu & Xie (2021, *ICML*) EnbPI is the residual-pool-refresh
  variant.
- **Why:** the served band is **split** conformal, which assumes exchangeability, and §12's
  regime record is a list of dated exchangeability breaks. §20's gate is **|coverage error| ≤
  5pp overall and ≥60% in any single regime window**, and it says in as many words that
  failures there *are* the ACI business case. This is the only item in the entire review that
  improves **something a user actually sees**.
- **State of play:** `interval_coverage` already exists (`backend/backtest/scoring.py`), is
  stored, is served on `/accuracy`, and — since step 3 — is computed per tier and per floor.
  **What does not exist is the per-regime-window read and the gate.**
- **It is NOT "one query away" — corrected 2026-08-08, on three counts.** (1) **No regime window
  is defined anywhere in the code**: `REGIME_WINDOWS` / `regime_window` / `regime_stress` return
  zero hits across every `.py` in the repo, so §12's dated breaks exist only in prose. (2) **The
  walkforward gate cannot supply the number, by design.** Every arm gets
  `PLACEHOLDER_BAND_PCT = 10.0`, the gate never calls `models/conformal.py`, and
  `interval_coverage` is *deliberately* neither logged nor persisted there —
  `walkforward_backtest.py:195-201` and the omission comment at `:621-623` say so outright,
  because the figure would describe the placeholder rather than a model. Realised coverage can
  only come from `scripts/backtest_accuracy.py`, which scores the actually-served band.
  (3) **Those stored outcomes span 6 forecast dates**, 2025-12-01 → 2026-07-19 (verified,
  `ops/forecast_outcomes.parquet`), in two clusters — a backdated batch and one week of July.
  You cannot read coverage "per regime window" off two clusters.
- **So R18 is gated on the same calendar wait as everything else**, plus a regime-window
  definition that has to be written first. It is not a cheap decidable read, and the ordering
  claim that it is should not be repeated.
- **Effort:** small to measure *once outcomes exist*; small-to-medium to implement.
  §10 Tier 3 #18.

### R19. Deflate the accumulated A/Bs for multiplicity — NOT STARTED, and it gates step 7

- **Do:** declare the trial count (§20 puts it at **≥15 A/Bs**), apply the Deflated Sharpe /
  PBO framework (Bailey & López de Prado 2014, *JPM* 40(5)), and hold new results to
  **t > 3.0** (Harvey, Liu & Zhu 2016, *RFS* 29(1)) rather than 2.0.
- **Why:** a dozen-plus A/Bs against one panel means the single-comparison CI is the wrong
  instrument, and §3 states the consequence without hedging: the ≥$1 universe result
  (**+3.50pp at 30d**) *"is the only surviving positive … and it should still be deflated for
  multiplicity before being called established."* **Step 7 ships that result**, so this is not
  bookkeeping — it is step 7's entry criterion, and it sits alongside the look-ahead caveat
  step 7 already carries.
- **The blocker is real: there is no CPCV path in this repo** (verified — no combinatorial
  split anywhere under `backend/`). §19's design is Track A purged expanding-window
  walk-forward for shipping and calibration, Track B **CPCV (N=12 / k=2 → 66 splits, 11
  backtest paths)** for feature decisions, never swapped — *"you cannot compute PBO from a
  single walk-forward path, which is why §3's multiplicity problem currently has no
  instrument."* Building Track B is medium, not small.
- **The instrument does not have to be built — corrected 2026-08-08.** `purgedcv`
  (`github.com/eslazarev/purged-cross-validation`, **MIT**, PyPI **0.1.3** and conda-forge)
  exports `CombinatorialPurgedCV`, `PurgedGroupKFold`, `PurgedKFold`, `WalkForwardSplit`,
  `purge`, `apply_embargo`, `deflated_sharpe_ratio` and
  `probability_of_backtest_overfitting` (PBO via CSCV) — i.e. **both** Bailey/López de Prado
  instruments this entry says have no path here, plus the fold geometry. All four splitters
  satisfy the sklearn splitter protocol; dependencies are numpy / pandas / scikit-learn / scipy
  and `python >= 3.10`, so nothing new enters the image. It exists precisely because mlfinlab,
  the canonical implementation, went closed-source. **Two caveats before leaning on it:** it is
  at **0.1.3**, which is young for something a ship decision would rest on; and adopting the
  splitter still leaves this repo's own work — CPCV has to be fed the same `cluster_key` fold
  geometry `backtest/paired_mde.py` uses, or the deflation runs on a different clustering than
  the intervals it is deflating. Re-cost the entry as **small-to-medium, mostly wiring**, not
  "build Track B".
- **Cheaper partial, available now:** step 2 already adopted the t > 3.0 hurdle for PT, so
  extending it to the paired-A/B verdicts is a threshold change; and Jensen, Kelly & Pedersen
  (2023, *JF* 78(5), 2465–2518) Bayesian hierarchical shrinkage is the right tool for the
  many-small-A/Bs problem **without** CPCV (code: `github.com/bkelly-lab/ReplicationCrisis`).
- **Effort:** small for the hurdle and the trial-count declaration; medium for the instrument.
  §10 Tier 3 #19.

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

- ~~The **three Appendix refutations** (CSFloat, ByMykel, training-breadth) still have
  date-clustered intervals and have **not** been re-derived.~~ **Re-derived 2026-08-08 and all
  three survive**, under the `H + 13` embargo, the universe filter and fold-clustered
  intervals: CSFloat null at all four horizons, ByMykel `treatment_vs_placebo` null at all
  four, breadth positive at **14d only** (mid +2.33pp [+0.44, +4.74]) and now null at 30d.
  Two caveats a citation must carry: the results are **not stored in this repo** (scratchpad
  artifacts only, reproducible by a 16m28s re-run), and two sub-findings are unresolved — the
  CSFloat **14d placebo excludes zero**, and ByMykel `age_only` reads positive at 3d/14d/30d
  *against baseline*, with no placebo contrast computed. See
  `docs/changelog/2026-08-08-migrated-archive-emptied-eight-harnesses.md`.
- The **A/B harness is not reproducible run-to-run** (`mean_diff_pp` −0.1581 → −0.0026 on
  identical commands, with `n_paired` and `n_dates` also moving). Best candidate cause is a
  changing ask-source set with no `n_ask_sources` column to detect it — the mean market return
  reads **−31.6% on 2026-03-22** and **+17.4%/−17.8% on 2026-07-09/10** against ±0.5% on a
  normal day. Hypothesis, not diagnosis. Step 1 removed one measured contributor to it — the
  bid's rejection flickered on **5.6%** of return pairs — but flicker was only a twentieth of
  that defect's effect, so this is not resolved.
- **h=30 rests on 5,461 usable rows from one backdated date.** It carries both the +3.50pp
  positive (step 7) and the largest unpurged inflation (step 5, now measured on the gate at
  **+10.15pp**, 2026-08-08). No h=30 claim until ≥30 forecast dates mature.
- **`|r| < 0.002`** for trade volume is quoted in nine documents and is **10–40× too small**:
  C4 measures pooled corr(vol z, fwd7) = **+0.019**, fwd30 = **+0.034**, item-fixed-effects
  identical, $1–10 tier **+0.080** at 7d, on 4.46M rows 2023–2025. The audit's *conclusion*
  survives (r² < 0.15%, economically trivial); its stated *reasoning* does not. Not corrected
  anywhere yet.
