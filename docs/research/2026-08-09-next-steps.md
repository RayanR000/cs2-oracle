# Next steps from the 2026-08-09 research review

**Source:** `docs/research/2026-08-09-model-and-data-research.md`.
**Supersedes ordering in:** `docs/research/2026-08-07-next-steps.md` steps 8–11 and R11–R19,
which remain valid as descriptions but are re-ranked here. Items carried forward keep their old
label (`6c`, `5c`, `R13`, `R18`) so they can be traced.

Every number quoted here is measured — in the review, or in the changelog entry named beside it.

---

## The gating decision

**Accuracy work is hard-gated on label integrity.** The reversal signal falls from rank IC
**0.168 → 0.101** once source composition is held stable, and to **+0.0044 (t = 0.1)** on the
cleanest subset available — 28 dates. If the label is a quoting artifact, no feature, objective or
architecture change can pay off, because the label is not a tradeable return.

⚠️ **One finding that appeared in a draft of this roadmap is withdrawn.** The review initially
reported 2026-07-09/10/11 as a new, undocumented consensus break. It is documented, detected and
already voided — see `docs/research/2026-08-09-model-and-data-research.md` §1a. G2 below is
correspondingly much smaller than first written.

**Cost work is not gated.** Nothing in Track A touches labels, features or the served signal.

```
  Track G (gate)          G1 → G2 → G3        must return before Track C starts
  Track A (cost)          A1 … A6             independent, start now
  Track O (operational)   O1 … O3             immediate, minutes each
  Track C (accuracy)      C1 … C7             blocked on Track G
  Track D (data)          D1 … D4             blocked on Track G, except D1
```

---

## Track O — operational, do first (minutes)

### O1. Clear the Price Forecast failure — **NOT STARTED, live**

- **Do:** dispatch `price-forecast.yml` with `mode=full`.
- **Why:** three consecutive failures since 2026-08-08 07:46 UTC with `saved model artifact
  version 3 != expected 5` — the Actions booster cache predates the minimal-model rewrite and
  `load_models()` refuses it rather than serving a band computed by a different scheme.
  `item_forecasts` is frozen at forecast_date 2026-08-07 and Backtest Accuracy is `skipped`
  behind it, so **no labels are maturing** and the ≥20-forecast-date wait is not elapsing.
- **Note:** `mode=full` only trains if the artifact is ≥14 days old or `FORCE_RETRAIN=1`. Here
  the artifact fails to *load*, so the retrain path is reached regardless — but set
  `FORCE_RETRAIN=1` to be certain.
- **Cost:** one dispatch, ≈20–25 min of runner time.

### O2. Correct the stale +3.50pp citation — **NOT STARTED**

- **Do:** `docs/architecture/model.md:210` and `:533` cite **+3.50pp at 30d** for the ≥$1
  training floor as settled. It was withdrawn 2026-08-08 — the same contrast re-derives to
  **+1.642pp [−0.809, +4.505], null**, and the placebo puts the instrument's item-draw noise
  floor at ±3–4pp (`docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`).
- **Why:** it is the one live stale number in `architecture/`. `model.md` also still describes
  the 99-item / 100K-row config as production; the shipped default is
  `DEFAULT_TRAIN_MIN_MEDIAN_PRICE = 1.0` + `DEFAULT_TRAIN_FEATURE_ROWS = 1_200_000`
  (`scripts/forecast_prices.py:51,63`), i.e. 926 items with no subsample.
- **Cost:** tiny.

### O3. Publish the 17 files missing from the canonical archive — **NOT STARTED**

- **Do:** add `item-metadata.parquet`, `item-metadata-bymykel.parquet` (+ codes JSON),
  `exchange-rates-history.parquet`, `player-counts-2011…2025.parquet` (13 files) and
  `ops/{collection_runs,events,accuracy_alerts}.parquet` to whatever `aggregator-update.yml`
  publishes. Dispatch the compaction that never ran in prod — the canonical repo still holds
  `snapshots-2026-08.parquet` (11.6 MB).
- **Why:** only CI writes `RayanR000/cs2-oracle-data`, and the local copy is not canonical.
  Concrete consequence: **the ByMykel bundle cannot be enabled in CI at all**, because the file
  it reads is not published — so `BYMYKEL_METADATA=1` is untestable there regardless of the
  feature verdict. Same for `usd_cny`, which reads as "blocked on 7 days of FX" while a 13-year
  FX file sits locally.
- **Cost:** small.

---

## Track G — the gate (must return before Track C)

### G1. Power up the source-composition test — **NOT STARTED**

- **Do:** extend `docs/research/2026-08-08-model-review.md` §5 from 932 dates to the full
  archive, and from the 28-date "three agreeing sources" cell to as many dates as the multi-source
  era supports. Report rank IC of `−r_t` predicting the forward 3-day return, partitioned by
  (a) source-composition stability across t−1…t+3, (b) source count, (c) whether the window spans
  any known break date. Add `n_ask_sources` as a stored column so the partition is auditable
  rather than recomputed.
- **Why:** the effect is +0.1676 pooled, +0.1006 composition-stable, and **+0.0044 (t = 0.1)** on
  the cleanest subset. That last cell has 28 dates. Everything downstream — the objective swap,
  the rank transform, the allowlist re-derivation — is conditional on it.
- **The honest constraint:** the multi-source era is **24 days deep** (§4b of the review). Two
  windows are effectively single-feed: 2026-01/02 (`sync` alone) and 2026-04-16 → 07-10
  (`17mafo` alone). So "three agreeing sources, none changing" may not be powerable from the
  archive as it stands, and the answer may be *"unresolvable until more multi-source days
  accumulate"* — which is itself a decision-grade result.
- **Entry criteria:** none.
- **Cost:** small — archive queries only, no fits.
- **Unblocks:** all of Track C.
- **What would invalidate it:** running it across 2026-03-22 or the 2026-07-09…11 handover
  without excluding those windows. Both are already in `_collection_shift_dates`' fired set, but
  G1 reads the archive directly rather than through `prepare_targets`, so it must apply the
  exclusion itself — that is the one real dependency on G2 part 1.

### G2. Publish the detector's fired-date list, and add the 2025 regime markers — **NOT STARTED**

**Scope reduced 2026-08-09.** The original entry proposed voiding 2026-07-09/10/11 and
generalising `_collection_shift_dates` to catch basis changes. The July cutover is already
detected and already voided, so both halves are withdrawn. What survives is smaller and mostly
about legibility.

- **Do, part 1 — make the detector auditable.** `_collection_shift_dates`
  (`models/forecaster.py:2975`) fires 12 times in 4,735 days and **the list of dates it fires on
  has never been written down anywhere.** Emit it into `meta.json` and into the training log, and
  pin it in `tests/test_degenerate_label_dates.py`.
- **Why:** this review re-reported a handled defect as a new one *because* that list is not
  printed. Any future audit will make the same mistake. It is also the cheapest possible check on
  whether 2026-07-11 is caught — currently argued from a universe-size jump (27,194 → 39,366
  items, well past `COLLECTION_SHIFT_FRACTION = 0.20`) rather than measured.
- **Do, part 2 — add the two 2025 regime boundaries.** 2025-07-15 (Valve Trade Protection: market
  cap ≈ −25% in one day, third-party listings −12%, 3–4% of trades reversed) and 2025-10-23
  (trade-up extended to 5 Covert → knife/gloves: cap $609M → $337M in hours, knives −20 to −60%,
  Coverts +5–20×).
- **⚠️ These must NOT be voided.** They are real market events, not collection artifacts. The
  standing rule *"do not winsorise large daily returns"* applies with full force — 2025-10-22 is
  the only dated event in thirteen years where item attributes dominated the market factor, and
  deleting it removes the single observation that carries that information. Mark them as
  **regime boundaries**: a `regime_window` definition, which R18 established does not exist
  anywhere in the code (`REGIME_WINDOWS` / `regime_window` / `regime_stress` return zero hits).
  2025-07-15 additionally matters as a **liquidity**-regime change: any volume or listing-count
  feature crossing it is measuring two different markets.
- **The design constraint that must not be violated if part 1 ever grows into a new detector:**
  cutovers are detected from the universe size and **never from prices**, deliberately — prices
  moving cannot change how many items a collector returns, so the detector cannot mask a real
  crash (`tests/test_degenerate_label_dates.py::test_a_price_crash_is_never_flagged`). Any
  basis-change detector must key on **source-set composition** (which sources reported, per day),
  never on the magnitude of the move. That column does not exist — it is G1's `n_ask_sources`.
- **Entry criteria:** none.
- **Cost:** small. No retrain needed for part 1.
- **Plan:** `docs/superpowers/plans/2026-08-09-label-integrity.md`.

### G3. Stop the Steam MA feeds voting in the consensus — carried forward as **6c**, NOT STARTED

- **Do:** add `aggregator_steam_7d/30d/90d` to the excluded set in `_apply_multi_source_voting`,
  bump `VOTED_CACHE_VERSION`, re-vote.
- **Why:** they are Steam's trailing-window **mean sale price** — MA(7)/MA(30)/MA(90) — voting on
  equal terms against point-in-time asks. Same class of basis error as
  `aggregator_buff163_buy`, which step 1 removed on 2026-08-07. Excluding them costs almost
  nothing on coverage (**670 item-days of 3,093,793**) but moves the voted median on **17.13%** of
  2026 ≥$1 item-days, median **−7.16%**, and flips **5.75%** of consecutive-day return directions
  — half the bid's magnitude, same character.
- **Do not scope it as a staleness fix.** It clears only 2.30pp of the 20.25pp ≥$1 stale rate,
  because `aggregator_sync` and `aggregator_steam_17mafo` are `last_24h` **falling back** to those
  same windows, which fires on exactly the illiquid items. The structural finding stands: **there
  is no point-in-time Steam price in this archive at all.**
- **Do not also drop `aggregator_sync`** — that deletes 2026-01 and 2026-02 in full for the ≥$1
  cohort (52,048 item-days) to buy a further 1.4pp. The fix there is upstream: record which field
  the fallback chain actually used.
- **It subsumes** the `aggregator_steam_17mafo` open item from the 2026-08-07 step 1. That feed is
  2,161,250 rows / 27,194 items, was the **only** feed 2026-04-16 → 2026-07-10, and its raw JSON
  (`price-archive/raw/17mafo/`, 84 files, 630 MB) confirms it carries `last_24h/7d/30d/90d`.
- **Entry criteria:** G2, so the re-vote is not measured across an unvoided break.
- **Cost:** small, plus a re-vote and its own changelog entry. Every A/B and label from 2026-03
  onward sits downstream of it.

---

## Track A — cost and instrumentation (independent, start now)

Full spec: `docs/superpowers/specs/2026-08-09-training-cost-design.md`.
Full plan: `docs/superpowers/plans/2026-08-09-training-cost.md`.

### A1. Cap CV fold training rows — **NOT STARTED. The largest untaken lever.**

- **Do:** apply `max_rows` / `_per_item_row_sample` inside `_cv_evaluate_horizon` exactly as
  `_build_production_split:2756-2761` already does. Default cap **300,000**, env-overridable.
- **Why:** `max_rows` is applied only on the production split; `_cv_evaluate_horizon:5693` takes
  the whole expanding window every fold, and nine folds per horizon sum to **4.2× the entire
  frame**. A rows×rounds model predicts CV / production-q50 = 2.94× against a measured 2.77×, so
  the 439.3s phase is fully explained by this and nothing else.

  | Cap | Saving | Share of 872s |
  |---|---:|---:|
  | 600k | −56.9s | 6.5% |
  | 400k | −137.6s | 15.8% |
  | **300k** | **−194.6s** | **22.3%** |
  | 200k | −263.6s | 30.2% |
  | 100k | −345.0s | 39.6% |

- **What it does not touch:** fold count, validation rows, OOF record count, distinct forecast
  dates, rank IC, the PT statistic. Only the model each fold fits sees less data, so `q_hat`
  describes a slightly weaker model than the served one — biasing the band **wide**, which is
  over-coverage and the safe direction.
- **Strictly better than the documented lever 1** (`CV_STEP_DAYS` 150 → higher), which buys the
  same seconds by destroying folds that are also the rank-IC and PT sample.
- **This was found once and dropped**: `docs/changelog/2026-08-07-per-item-row-sampling.md` —
  *"the sampler thins the 36-second part of a 541-second run."*
- **Entry criteria:** none.
- **Verification:** empirical band coverage against `NOMINAL_COVERAGE = 0.80` before and after.
- **Cost:** small.

### A2. Re-tune Optuna against rank IC at fixed rounds — **NOT STARTED. Best accuracy-per-second.**

- **Do:** change the Optuna objective (`_optuna_search_params:2840-2853`) from
  `model.best_score["valid_0"]["quantile"]` under `lgb.early_stopping(20)` to **within-date rank
  IC at `_boost_rounds(horizon, cv=True)`**, using the `_within_date_rank_ic` helper that already
  exists at `:5958`.
- **Why:** the selector still optimises early-stopped validation pinball loss on the same thin
  trailing window the project refuted when it shipped `FIXED_BOOST_ROUNDS` on 2026-08-08. The
  `FIXED_BOOST_ROUNDS` comment at `:551-560` records the conflict: **at 14d and 30d the
  validation-loss optimum is 25 rounds while rank IC peaks at 500–750.** The two are
  anti-correlated, so the params in `meta.json` for the two noisiest horizons were chosen under a
  discarded criterion.
- **Cost:** 35s of an 872s run — Optuna is 4.0% of the retrain and the trial budget does not
  change. Requires `FORCE_HP_SEARCH=1` once to escape the cached params, and a
  `MODEL_ARTIFACT_VERSION` bump so the stale params cannot be reused.
- **Entry criteria:** none. Independent of labels — it changes *which* params are chosen, not
  what is trained on.
- **What would invalidate it:** scoring rank IC pooled rather than within-date. Pooled IC
  reintroduces the market factor and would select for exactly the base-rate tracking the PT test
  exists to reject.

### A3. Allowlist before the correlation prune — **NOT STARTED**

- **Do:** in `build_training_data` (`:3475-3489`), run `_apply_feature_allowlist` **before**
  `_prune_features`, behind a flag, logging both counts.
- **Why:** `df[self.feature_cols].corr()` (`:2402`) is O(rows × p²) single-threaded pandas.
  On 123 columns it costs **25.2s**; on the 33 allowlisted columns, **1.65s**. On the production
  frame the prune drops **zero** price_technicals features (33 in, 33 out), so the reordering is
  output-identical *here*.
- **The caveat that requires the flag:** `_prune_features` keeps the **lower-indexed** member of a
  >0.95 pair, so a price feature could in principle be dropped in favour of a non-allowlisted
  partner. Not provably identical in general. Log `pre → post` for both orderings and assert
  equality in a test on the production frame.
- **Cost:** −23.5s. Small.

### A4. Short-circuit the discarded feature blocks — **NOT STARTED**

- **Do:** add a flag to `engineer_features` that skips the eight non-`price_technicals` blocks
  when the allowlist would drop them anyway.
- **Why:** measured 8.5s of 17.5s (48%) is discarded work — temporal 0.39s, item_identity 1.69s,
  events 0.01s, item_metadata 1.15s, supply_side 0.79s, social 0.00s, cross_sectional 2.15s,
  supply_depth 2.09s.
- **⚠️ Must be a flag, never a deletion.** Seven harnesses build their own frame and then call
  `_apply_feature_allowlist` on it, so they need the full 123-column frame:
  `ab_test_training_breadth.py:341`, `ab_test_train_universe.py:348`,
  `ab_test_item_metadata.py:306`, `ab_test_csfloat_basis.py:340`,
  `ab_test_interval_sampling.py:533`, `ab_test_q50_sampling.py:412`,
  `ab_test_direction_labels.py:246`. `HORIZON_EXCLUDED_GROUPS` (`:339`) and
  `_validate_feature_groups` (`:2427`) are also written in terms of groups the allowlist removes.
- **⚠️ Also blocks C4.** If `cross_sectional` is ever re-admitted to the allowlist, this flag must
  respect the allowlist rather than a hard-coded block list.
- **Cost:** −8.5s. Small. This is the doc's lever 2, re-sized: `model-optimization.md:162` puts it
  at "~1% of the retrain" against an assumed 7s phase; the measured phase is 17.5s and the
  recoverable total across A3+A4 is ≈32s / 3.7%.

### A5. `CV_DIAGNOSTIC_CLASSIFIER` default off — **NOT STARTED**

- **Do:** flip the default at `:5934` from on to off. Keep the env var so research runs can
  re-enable it.
- **Why:** on locally, which puts a research retrain at **1804s / 30.1 min** — over the project's
  own 30-minute run cap — to populate `classifier_accuracy` / `classifier_accuracy_ge1` in
  `meta.json`, which no served artifact reads. Measured directly 2026-08-09: 872s off vs 1804s
  on, i.e. **932s / 52%**. CI already sets it off.
- **What is lost:** `mean_classifier_acc_ge1` disappears from local `meta.json`. That number has
  been the project's headline diagnostic for months, so flipping the default is a
  *reporting* change as much as a cost one — the replacement is `mean_rank_ic` and the PT verdict,
  both of which the working tree already computes from `fold_p50` and therefore survive.
- **Cost:** tiny.

### A6. Content-hash the archive fingerprint — **NOT STARTED**

- **Do:** replace `st_mtime_ns` in `_archive_fingerprint` (`:3510`) with a content-derived key —
  row count + max day per file, or the archive commit SHA.
- **Why:** the voted cache **cannot hit in CI**, for two independent reasons. `self.cache_dir` is
  `backend/data` (`:641`), gitignored, and `price-forecast.yml` caches only
  `backend/models/saved_models` (`:91-96`). And even if it survived, CI checks the archive out
  fresh every run, so **every file's mtime is new every run** and the key changes unconditionally.
  Fingerprinting on mtime is structurally CI-hostile.
- **Cost:** ~48s per run (21.1s DuckDB read + 27.2s voting) — **daily, not weekly**, since the
  predict path pays it too. Small change; requires caching `backend/data` in the workflow as well.
- **⚠️ `VOTED_CACHE_VERSION` must still be bumped by hand** whenever `_fetch_voted_price_history`
  or `_apply_multi_source_voting` changes — the key cannot see code. This is the item-universe
  rule and A6 does not change it. Currently at **4**.

---

## Track C — accuracy (blocked on Track G)

### C1. Cross-sectionally rank-transform features per date

- **Do:** `groupby("date")[col].rank(pct=True)`, then map to `2 * (pct − 0.5)` so the range is
  [−1, 1], applied to every surviving feature. This is Gu, Kelly & Xiu's footnote-29 transform.
- **Why:** the repo's features are scale-free **per item** (pinned by
  `tests/test_scale_free_features.py`), which is not the same as cross-sectionally normalised.
  Scale-free still leaves every feature loaded on the common market factor on every date — and
  "DA is dominated by the market factor" plus "demeaning by the market factor drops accuracy below
  a constant call" are both the predicted symptoms of exactly that.
- **Not the refuted experiment.** `2026-08-06-market-relative-labels-refuted.md` changed the
  **label** and left a pointwise loss fighting a noisy residual. This changes the **features** and
  leaves the label alone.
- **Side benefit:** a missing characteristic maps to the cross-sectional median by construction,
  rather than to the persisted `feature_medians` — which is the mechanism behind the
  calendar-gap lag-fill artifact.
- **Cost:** one groupby. Test on `ab_test_training_breadth`'s frame with `--fixed-rounds`.
- **Entry criteria:** G1 returns "the signal survives composition control."
- **Test it alone before C2.** It is the cheaper half of step 10 and isolates cleanly.

### C2. `lambdarank` within date, scored by rank IC — carried forward from step 10

- **Do:** objective swap. Query groups = forecast dates; rows contiguous and sorted by group.
- **Why:** quantile loss on raw % returns targets a conditional median per row; nothing in
  training optimises within-date ordering, which is what the dashboard serves. The model loses to
  ranking by `−return_1d` at all four horizons.
- **The evidence, with its caveat:** LambdaRankIC (arXiv:2605.00501, **preprint, unreviewed**,
  XGBoost not LightGBM) measures regression 0.042 → pairwise LTR **0.083** → NDCG LTR 0.086 →
  their method 0.115. The **regression → LTR jump is the larger and the replicable-looking one**,
  and it needs no custom gradient code. In the same table plain regression *underperformed OLS*,
  which is this project's experience.
- **Three setup details that will silently ruin it:**
  1. `lambdarank_truncation_level` **defaults to 30** — over a ~900-item cross-section the
     objective would ignore everything below rank 30. Raise it substantially; cost is superlinear.
  2. `label_gain` defaults to `(1<<i)-1`, exponential and capped at label 31. Bucket forward
     returns into 5–10 per-date quantiles and override with a linear gain.
  3. `lambdarank_norm` matters because the cross-section width varies by date (items enter and
     exit).
- **The structural cost nobody has priced:** a ranker emits an **uncalibrated score, not a
  return**. The conformal band and the q50 serving path both need a level, so this **adds** a
  model per horizon rather than replacing one. Budget it as +4 boosters, not 0.
- **Entry criteria:** G1, then C1 measured separately.
- **Honest bar:** ICIR > 0.05 per the review's own note, not 0.5. Qlib's published LightGBM
  benchmark on daily cross-sectional CSI300 with 158 features is **Rank IC ≈ 0.05**; this model's
  OOF rank IC is already 0.092–0.183.

### C3. Residual reversal as a feature

- **Do:** demean `return_1d` within weapon / price-tier / crate groups and add the residual as a
  feature. Keep the raw `return_1d` beside it.
- **Why:** Nagel (*RFS* 2012) shows reversal returns proxy the return to liquidity provision. Da,
  Liu & Schaumburg (*Management Science* 60(3), 2014) decompose reversal into across-industry
  momentum, within-industry expected-return variation, under-reaction to cash-flow news and a
  residual, and find **only the residual is significant** — with a strategy isolating it earning
  ~3× the standard reversal strategy's risk-adjusted return.
- **Why it is not the refuted work:** the ByMykel refutation showed *cosmetic* metadata goes
  unused; the crate finding
  (`same-crate residual corr +0.084 vs +0.003 different-crate, a 28× ratio`) says crate is a
  **covariance-structure** fact. Using it as a demeaning group is the use that fact supports,
  unlike using it as a mean-prediction column.
- **Entry criteria:** G1. Also read `_validate_feature_groups` before any paired retrain.
- **Cost:** small.

### C4. Re-derive the feature allowlist with `cross_sectional` restored

- **Do:** re-run the 2026-07-24 ablation under the `H + 13` embargo, the universe filter,
  fold-clustered intervals and `--fixed-rounds`.
- **Why:** `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` rests on `ab_test_feature_contribution`
  at **100 items, un-embargoed, no universe filter, early-stopped**, and it has never been
  re-derived. Meanwhile **the one measured positive structure in this archive** — expensive tier
  leads cheap tier, lag-1 corr **+0.213, z = 9.1**, Granger incremental R² **9.0%**, stable 4 of
  5 years, survives market-factor removal — lives in the `cross_sectional` group, which is
  engineered on every row (2.15s) and then discarded, and is *additionally* excluded outright at
  h=14 and h=30 by `HORIZON_EXCLUDED_GROUPS`.
- **Entry criteria:** G1; and A4 must respect the allowlist rather than a hard-coded block list.
- **Cost:** medium — it is a full ablation, and the MDE must be computed first.

### C5. Split conformal + ACI — carried forward as **R18**

- **Do:** replace the K-fold OOF calibration with a single split-conformal calibration slice, and
  add Gibbs & Candès (2021) Adaptive Conformal Inference for drift.
- **Why (cost):** split conformal needs **one** fit; the CV block currently pays 33. The standard
  variance objection to split conformal is calibrated to n in the hundreds, and this frame has
  ~1M item-days.
- **Why (correctness):** split conformal assumes exchangeability, and §12's regime record plus
  this review's four break dates are a list of exchangeability breaks. ACI adjusts α online from
  realised coverage and needs **no refits at all**. Zaffran et al. (*ICML* 2022, peer-reviewed)
  find ACI gives the smallest intervals at correct coverage; EnbPI attains coverage by
  over-covering there and *under*-covers in a 2026 benchmark preprint — **contradiction flagged**,
  ACI wins in both.
- **Not available:** jackknife+-after-bootstrap. It is genuinely free but requires a bagged
  ensemble of models; `bagging_fraction` resamples per *iteration*, not per model, so a single
  booster has no out-of-bag structure.
- **Ordering:** **do A1 first.** A1 is −22% for a bounded, verifiable change; C5 is −50% for a
  scheme change. R18's existing entry files it as gated on the calendar wait — that applies to the
  *coverage measurement*, not to the calibration-cost argument, which is independent.
- **Cost:** medium. Requires a per-regime-window coverage read, and **no regime window is defined
  anywhere in the code** (`REGIME_WINDOWS` / `regime_window` / `regime_stress` return zero hits).
  That definition has to be written first.

### C6. Decide the regime models and `N_ENSEMBLES`

- **Do:** either set `SKIP_REGIMES=1` in CI or justify the 95.4s / 10.9%.
- **Why:** the deployed regime set includes 1-tree and 3-tree boosters, so they are plausibly
  *harming* serving. The case for cutting rests on the degenerate boosters, not on cost. Note the
  asymmetry this creates today: CI's Monday run is always cold (`price-forecast.yml:88`), so
  regimes are always trained and served, while the documented local retrain passes
  `SKIP_REGIMES=1` — **the served model depends on where it was trained.**
- **The restore lever if accuracy needs recovering** is `N_ENSEMBLES = 2` (`:317`); the 3→1
  collapse was never measured.
- **Entry criteria:** A1, so there is budget to spend.

### C7. Deflate the accumulated A/Bs — carried forward as **R19**

- **Do:** declare the trial count, apply the t > 3.0 hurdle to paired-A/B verdicts (step 2 already
  adopted it for PT), and use `purgedcv` (MIT, PyPI 0.1.3) for `deflated_sharpe_ratio` and
  `probability_of_backtest_overfitting`.
- **Why:** a dozen-plus A/Bs against one panel means the single-comparison CI is the wrong
  instrument. The urgency is gone — the positive it was meant to deflate was withdrawn — but the
  argument stands.
- **Caveat:** `purgedcv` is at 0.1.3, young for something a ship decision rests on, and CPCV must
  be fed the same `cluster_key` fold geometry `backtest/paired_mde.py` uses or the deflation runs
  on a different clustering than the intervals it deflates.

---

## Track D — data

### D1. Re-verify the Steam listing page from this project's egress — **NOT STARTED, do early**

- **Do:** fetch `steamcommunity.com/market/listings/730/<name>` from the machine and from a CI
  runner. Confirm the dehydrated react-query cache is present and parse
  `{time, price_median, purchases}` plus `rgCompactBuyOrders` / `rgCompactSellOrders`.
- **Why:** Steam rebuilt the page as SSR/React and the old `var line1=[[...]]` array is **gone**.
  The replacement carries **daily sales counts back to 2014-02-21** and **full bid/ask depth with
  quantities**, at ~10 `market_hash_name`s per page load. In the research environment 10
  back-to-back GETs at ~1.3 req/s all returned 200 with no throttling.
- **⚠️ This contradicts blocker 5d**, which records a soft-block from this machine since
  2026-08-05 with no decay in three days. The research environment may not be representative.
  **Verify before planning any backfill on it.**
- **If it holds:** `R11`, `R13` and `5c` all unblock, at ~0.1 requests per item — a 900-item
  cohort is 100–200 fetches, not 7,100. That is the difference between a blocked project and an
  afternoon.
- **Entry criteria:** none. This is the one Track D item worth doing before the gate returns,
  because it is a five-minute check that changes the value of three other items.

### D2. Fix the csgotrader collector to the per-provider paths

- **Do:** `prices.csgotrader.app/latest/<provider>.json`. Stop expecting `steam_listing` and
  `steam_volume`.
- **Why:** `prices_v6.json` now 301s to an S3 `NoSuchKey`, and **`steam_volume.json` no longer
  exists** — which is the mechanical cause of the dead `volume` column. It died upstream, not in
  the collector. Alive: buff163, csgotrader, skinport, csgoempire, csgotm, csmoney, bitskins,
  lootfarm, swapgg, cstrade, skinwallet, csgoexo, exchange_rates. Dead: steam_listing,
  steam_volume, waxpeer, uu898. `buff163.json` still carries `starting_at` and `highest_order`.
- **Cost:** small.

### D3. Cross-check volume against the `devynpruden` Kaggle dataset

- **Do:** pull `devynpruden/cs2-skin-price-history-2013-2026` — 290 MB Parquet, **Apache 2.0**,
  updated 2026-06-16, daily median/mean/min/max **plus `volume` = units sold daily**, 2013→2026.
- **Why:** it is the dataset the kieranpoc decline was reaching for. kieranpoc was declined on
  coverage (frozen 2024-05-04, supplies nothing for 2024-06 → 2026-08); this one covers the window
  and is permissively licensed. Use it as an **independent cross-check** on D1's Steam route
  before trusting either — two sources agreeing is the standard this project has adopted for
  price basis.
- **Entry criteria:** D1, so there is something to cross-check against.
- **⚠️ Volume as a *predictor* remains refuted** (pooled corr(vol z, fwd7) = +0.019, r² < 0.15%).
  The value is a **counting-noise denominator**, not a signal.

### D4. `stattrak_premium_z30` — the cheapest unexploited item

- **Do:** compute the StatTrak / normal price ratio per matched name pair, z-scored over 30 days.
  4,686 paired names exist in the archive already.
- **Why:** a revealed-preference weapon-usage measure — AK-47 **2.15×** vs P2000 **1.10×**, median
  1.43×. **Free, 13 years deep, requires no fetch at all**, and five of seven demand drivers have
  no proxy whatsoever.
- **⚠️ z-score only, never the level** — the level is a dollar-scale proxy and belongs in
  `_DOLLAR_SCALE_FEATURES`.
- **Entry criteria:** G1. It is an item-level feature and will not clear the item-level MDE on its
  own; it belongs in a bundle or in Track C's date × tier frame.

### D5. Consolidate the experiment harnesses

- **Do:** extract, in this order — one archive loader, one fold builder that calls `embargo_days`
  internally, one fit step with **early stopping off by default**, one scorer emitting the
  invariant-4 trio plus rank IC, a power gate that refuses to run below MDE, a mandatory placebo
  arm, a committed results store, and a config surface.
- **Why:** 8,617 lines across 15 standalone scripts with essentially no shared code —
  `_build_frame_uncached` ×6, `run_evaluation` ×7, `build_frame` ×6. Every one of four documented
  archive defects had to be fixed 13–15 times independently, and three are still not fixed
  everywhere: **9 harnesses use `phase_collapsed_sql_filter()` where invariant 2 requires
  `archive_universe_sql_filter()`**, and **15 of 15 still early-stop against the window they
  score** — the defect production fixed on 2026-08-08.
- **The concrete cost of not doing it:** the three harnesses re-run on 2026-08-08 produced results
  that live **only in a session scratchpad**. The changelog numbers are reproducible only by a
  16m28s re-run, not by reading a file, so "does this reproduce?" needs a full investigation every
  time.
- **Cost:** medium-to-large. Not urgent, but it is the difference between this project scaling and
  not.

---

## Answers to the two questions that prompted this review

### "100 items vs 200 items"

**Runnable today.** `backend/scripts/ab_test_training_breadth.py`, nested arms so the item draw
differences out, MDE **0.69–2.15pp**, under 2 minutes for one horizon × two arms.
`N_NARROW`/`N_MID`/`N_WIDE` at `:105-107` are module constants — "100 vs 200" is a source edit,
there is no CLI flag. **Pass `--fixed-rounds`**; this is the only harness of 15 with that escape
hatch, and without it both arms early-stop against the window they score.

**Prior:** the 2026-08-08 re-run found breadth positive at **14d only** (+2.33pp [+0.44, +4.74]),
null at 30d, saturating by 350 items. Expect 100 → 200 to be small and 14d-only, and 350 → 700 to
be flat.

**Do not read the answer as a training-budget recommendation.** Tabular scaling laws give boosting
b ≈ 0.48, so doubling rows cuts the reducible error ~29% — and in return forecasting the reducible
component is ~0.4% R² against an irreducible floor that is essentially the whole variance. The
1.2M budget is past the knee. The *fold* budget (A1) is where the headroom is.

### "Does volume affect accuracy or speed"

**Accuracy: a clean pre-cliff window exists** — 2013-08-14 → 2025-12-31, 4,523 days, 9.42M
volume-bearing rows, 5,536 items (871 at ≥$1), zero missing calendar days. Four things would
invalidate a naive run; all four are in `docs/research/2026-08-09-model-and-data-research.md` §4f.
The short version: cut at **2025-12-31**, not at the harness's `VOLUME_LIVE_THROUGH =
"2026-04-30"`; never pool across 2026-03-22, where `ask_volume` starts masquerading as `volume`;
and know that the pre-2026 series has **literally zero zero-volume rows**, so a no-sale day is an
absent row and any feature built on it conditions on a sale having occurred.

**Speed: not an experiment.** Re-admitting 13 of ~45 columns moves the booster-fit phases, which
are 39% of an 872s retrain, and per-phase timings on this hardware swing **±25% between clean
runs at identical config**.

---

## Do not re-propose

Carried forward from `2026-08-07-next-steps.md` and extended by this review.

- **The `model-optimization.md` micro-lever table.** `max_bin` 63→31, `num_leaves` 47→31,
  `min_data_in_leaf` 15→100 and `feature_fraction` 0.7→0.4 were all measured on the production
  frame 2026-08-09 and are **dead** (26.3 / 26.2 / 27.1 / 28.9 ms per round against a 25.5
  baseline). LightGBM is memory-bandwidth bound at this shape.
- **`feature_pre_filter: True`.** `False` is deliberate at all five Dataset sites — it is what
  allows one binned Dataset to be reused across parameter sets.
- **`force_row_wise` / `force_col_wise`.** Measured inside the noise band. Pin them for log
  cleanliness if you like; not for time.
- **Reusing boosters across CV folds.** Impossible: every fold's calibration rows must be unseen
  by the model producing them.
- **Parallel horizon or ensemble training.** Deleted 2026-07-21 after OpenMP deadlocks.
- **`SKIP_CV=1` in CI.** Pinned by `test_ci_workflow_does_not_skip_cv`.
- **CatBoost, neural forecasters, sentiment, cosmetic item metadata, float/paint-seed data,
  per-item ARIMA/GARCH, item embeddings, a manipulation score, anything paid.**
- **Winsorising large daily returns.** 2025-10-22 is the only dated event where item attributes
  dominated the market factor; clipping deletes the one observation carrying that information.
- **kieranpoc (frozen 2024-05-04) and atalantus (359 usable days, ~2 folds; as a price source it
  shrinks `target_items` 521 → 426).**
