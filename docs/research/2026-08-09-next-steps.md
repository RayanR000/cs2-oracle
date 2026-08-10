# Next steps from the 2026-08-09 research review

> ⚠️ **Ordering superseded by `docs/research/2026-08-10-next-steps.md`.** This document remains the
> reference for the *content* of every `O`/`G`/`A`/`C`/`D` item and its cautions — the later one
> carries the labels forward and re-ranks them. Two things changed on 2026-08-10: the
> "loses to a constant call" framing used throughout is a comparison to a **hindsight-selected**
> baseline (`2026-08-10-constant-call-is-hindsight-picked.md`), and two serving-path calibration
> defects plus a `model_version` fragmentation blocker were found
> (`2026-08-10-band-and-confidence-are-miscalibrated.md`). The `−return_1d` rank-IC gap this
> document treats as the live bar is **unaffected and still the bar**.

**Source:** `docs/research/2026-08-09-model-and-data-research.md`.
**Supersedes ordering in:** `docs/research/2026-08-07-next-steps.md` steps 8–11 and R11–R19,
which remain valid as descriptions but are re-ranked here. Items carried forward keep their old
label (`6c`, `5c`, `R13`, `R18`) so they can be traced.

Every number quoted here is measured — in the review, or in the changelog entry named beside it.

⚠️ **Except the Track A projections, which are predictions and three of them failed.** Read
`docs/changelog/2026-08-09-training-cost-levers.md` before quoting any saving from Track A. Its
own conclusion: *"Do not quote a speedup from this work."*

---

## Status as of 2026-08-10 05:00 UTC

Every item below carries its own status line. Summary, so nothing here is re-proposed:

| Track | Done | Open |
|---|---|---|
| **O** | O1, **O4 (the post-re-vote retrain — done 2026-08-10)**, O3 (the missing files — **22**, not 17) | O2, O3's snapshot deletion |
| **G** | **G1 (instrument + result), G2 part 1, G3** | G2 part 2 |
| **A** | **A1–A6, all of them** | — |
| **C** | C2's rank-IC half | C1, C2's `lambdarank` half, C3–C7 — **all unblocked** |
| **D** | D1 (and it **answers**) | D2–D5 — **all unblocked** |

✅ **O4 is done — do not re-dispatch it.** G3 (`873148b`) changed the consensus every label is
built from, and the retrain onto it ran 2026-08-10 as run `31356483719`. The post-exclusion
baseline exists and the `voted-v6-` cache key is populated. **Read O4 under Track O below for
the result before starting Track C.** An earlier version of this block told the reader to dispatch
`mode=full` with `FORCE_RETRAIN=1`; that instruction was wrong twice over — `mode=full` hits the
14-day age gate and degrades to predict-only on a fresh artifact, and `FORCE_RETRAIN` is an
environment variable the workflow neither sets nor exposes as a dispatch input. The way to force
training is `mode=train-only` (`docs/changelog/2026-08-10-post-revote-retrain.md`).

**Track A is closed.** All six shipped 2026-08-09 on branch `training-cost`, with the measured
outcome — including three of this document's own predictions failing to reproduce — in
`docs/changelog/2026-08-09-training-cost-levers.md`. Do not re-run the cost analysis.

**D1 answered, and it unblocks work.** The Steam listing page does respond from this egress
(re-measured 2026-08-09, 0.36 requests per item). Blocker 5d of `2026-08-07-next-steps.md` was a
false positive. `R13` and `5c` are unblocked on access; neither has been built.

⚠️ **"`R11` unblocked" is shorthand and has been misread.** R11's *source* — the kieranpoc Kaggle
dump — stays **DECLINED** on coverage (frozen at a 2024-05-04 snapshot; supplies nothing for
2024-06 → 2026-08), and it is listed under "Do not re-propose" at the foot of this document. What
5d's refutation removed is the residual blocker the decline named, so R11's *objective* — a
historical sale-count panel — is live again by a **different** route: Steam listing pages plus the
`devynpruden` dump. That is **D3**, not R11.

~~**One live sequencing defect.** `VOTED_CACHE_VERSION` is **5** while `price-forecast.yml:115-116`
still keys the voted cache on `voted-v4-`.~~ **Resolved 2026-08-09 in `873148b`**, which bumped the
constant to **6** and moved the workflow key `voted-v4-` → `voted-v6-` in the same commit, leaving
no v5 key behind. It never affected a CI run: the v5 bump and the whole "Cache voted price frame"
step were unpushed at the time. Both are on `origin/main` now, and the v6 key is unpopulated, so
the next run votes cold once and hits thereafter.

---

## The gating decision — ✅ **THE GATE IS LIFTED (2026-08-09, commit `f833882`)**

> ~~**Accuracy work is hard-gated on label integrity.** The reversal signal falls from rank IC
> **0.168 → 0.101** once source composition is held stable, and to **+0.0044 (t = 0.1)** on the
> cleanest subset available — 28 dates.~~
>
> **Refuted. Those rows do not mean what they say, and the conditional's antecedent is false.**
> The partition behind them classified every item-day with a NULL `source` as "composition
> changed", and `source` is NULL for **every** archive row before 2026 (9,417,947 item-days), so
> the "stable" cell held no pre-2026 data and the 0.168 → 0.101 fall is a **2013-2025 → 2026
> regime difference**, not composition control.
>
> Re-measured with the committed instrument (`backend/scripts/measure_composition_stability.py`),
> composition defined as the **set** of source names, void dates excluded:
>
> | | n_dates | stable | unconditional |
> |---|---:|---:|---:|
> | h=3, 2026 | 181 / 185 | **+0.1027** | **+0.1023** |
> | h=7, 2026 | 167 / 172 | **+0.0842** | **+0.0842** |
>
> **Holding source composition still does not touch the signal — but read what is and is not
> compared.** The equality above is **stable vs unconditional**, not stable vs changed. The
> "composition changed (present)" cell — items whose composition genuinely differed with every
> window day observed — is **underpowered at both horizons** (25 dates at 3d, 19 at 7d), so it
> carries no number and **no directional claim about the changed population is supported.** The
> instrument also reports `Window incomplete` separately (37 dates at 3d, 166 at 7d); that is
> gap-driven reporting, not composition change, and is not part of the artifact test. Full result:
> `docs/research/2026-08-09-composition-stability.md`; record:
> `docs/changelog/2026-08-09-composition-stability-refutes-quoting-artifact.md`.
>
> **Why that still lifts the gate.** The gate's premise was that the reversal is *made of*
> composition change. If it were, removing composition change would have moved the signal — and on
> 181 and 167 dates it does not move at all. That is sufficient to reject the premise without a
> powered changed-cell contrast. What a powered changed cell would add is the *magnitude* of any
> residual composition effect, which is a refinement, not the gate.
>
> **Track C and Track D are unblocked.** Their per-item "Entry criteria: G1" lines are satisfied.
>
> **What remains open is narrower and is a calendar wait, not work.** The strictest test — the
> reversal surviving *several independent sources agreeing* — is **25 dates at 3d and 19 at 7d**,
> both under the instrument's 30-date reporting floor, so no number is quoted for either. That is
> the cell §5 leant on at 28 dates. It cannot be answered until more multi-source days accumulate;
> the multi-source era is a few dozen days deep. **It does not gate Track C** — a null there would
> not restore the composition mechanism this measurement removed.

⚠️ **One finding that appeared in a draft of this roadmap is withdrawn.** The review initially
reported 2026-07-09/10/11 as a new, undocumented consensus break. It is documented, detected and
already voided — see `docs/research/2026-08-09-model-and-data-research.md` §1a. G2 below is
correspondingly much smaller than first written.

**Cost work is not gated.** Nothing in Track A touches labels, features or the served signal.

```
  Track G (gate)          G1 ✅ → G2 (pt 1 ✅) → G3        G1 has RETURNED; C is unblocked
  Track A (cost)          A1 … A6 ✅                        closed
  Track O (operational)   O1 ✅  O2  O3 ✅                   O2 is the only one left
  Track C (accuracy)      C1 … C7                          UNBLOCKED — C1 first
  Track D (data)          D1 ✅  D2 … D5                     unblocked
```

**G3 is still worth doing before C4 and any re-vote**, not as a gate on the label's validity but
because it moves the voted median on 17.13% of 2026 ≥$1 item-days — every A/B and label from
2026-03 on sits downstream of it.

---

## Track O — operational, do first (minutes)

### O4. Retrain on the post-G3 consensus — ✅ **DONE 2026-08-10**, run `31356483719`

> **Landed.** `mode=train-only` on `a0c215e`, 39m32s, TOTAL training 2306.0s. 12 boosters cached;
> `voted-v6-` populated. PT returns **skill at all four horizons** (t = 12.14 / 7.03 / 3.08 /
> 3.35), and the model still **loses to `−return_1d`** on rank IC everywhere (−0.0159 / −0.0371 /
> −0.0433 / −0.0091) — the re-vote cleaned the label's basis without closing the baseline gap.
> Full numbers: `docs/changelog/2026-08-10-post-revote-retrain.md`.
>
> ⚠️ It got **slower**, not faster: Optuna cost **692.9s** against 63.3s in the previous CI run.
> That phase is bimodal run to run. Remaining cost levers, with the measured basis:
> `docs/research/2026-08-10-training-cost-levers.md`.
>
> ⚠️ **Served accuracy is still unmeasured.** `CV_DIAGNOSTIC_CLASSIFIER=0`, so the classifier that
> serves direction is unscored in CV and every DA figure in the log is the q50 sign. No forecasts
> were written by a `train-only` run; the first served read comes from the backtest.

- **Do:** dispatch `price-forecast.yml` with **`mode=train-only`** against `origin/main`.
  ⚠️ **Not `mode=full`.** `full` runs `forecast_prices.py` with no flag, which reaches the age
  gate at `:376-398` — and the artifact is one day old, so it would *skip* training and simply
  re-predict from the stale model. `FORCE_RETRAIN` is an environment variable that
  `price-forecast.yml` never sets and does not expose as a dispatch input, so it is unreachable
  from the Actions UI. `--train-only` sets `do_train = True` unconditionally (`:380-381`).
  Run `31337078991` retrained under `mode=full` only because the artifact failed to *load* that
  day, which is a different branch of the same gate.
- **What `train-only` costs you:** no forecasts are written on that run (the predict, publish and
  accuracy steps are all conditioned on `mode != 'train-only'`). That is fine — "Save retrained
  models" still runs, so the boosters land in the Actions cache under `forecast-models-<run_id>`,
  and the next daily `predict-only` run restores them via the `forecast-models-` prefix.
- **Why:** G3 (`873148b`) removed `aggregator_steam_7d/30d/90d` from the vote, so the consensus
  price — and therefore every label from 2026-03 onward — changed. The shipped artifact is
  `trained_at 2026-08-09 21:55:20 UTC`, before that commit, and the newest run of any workflow is
  on `895005a`, five commits behind. Production is currently serving a model fitted to the
  pre-exclusion vote while the backtest scores it against the same stale basis. Nothing in Track C
  is interpretable until a post-exclusion baseline exists.
- **Expect:** a **cold vote**. `VOTED_CACHE_VERSION` is 6 and no `voted-v6-` key has ever been
  populated, so this run pays the full ~48s vote plus the DuckDB read, on top of training. The
  last full CI retrain measured **35m24s** (`TOTAL training: 1884.2s`) — already over the
  project's 30-minute cap, and this one is cold on the cache as well.
- **After it lands:** confirm `meta.json`'s `trained_at` postdates `873148b`, then re-read
  `classifier_accuracy_ge1` and the rank IC as the **new baseline**. Every stored A/B verdict
  predates the re-vote and is not comparable to it.
- **Entry criteria:** none. **Cost:** one dispatch.

### O1. Clear the Price Forecast failure — ✅ **DONE 2026-08-09**

- **Outcome:** dispatched run `31337078991` succeeded (35m24s, 2026-08-09 21:32 UTC) and
  Backtest Accuracy `31338602398` went green behind it at 22:07 UTC. The three-failure streak is
  cleared, the artifact is `model_artifact_version: 6`, and labels are maturing again.
- **What the clearance exposed:** the artifact-version check had been failing *first* for weeks, so
  training had not actually run in CI since before the volume feed died. Behind it sat a second,
  unrelated crash — a NULL `volume` column killing feature engineering on 2026-08-08, the first
  archive day whose `volume` is mostly NULL. Cherry-picked to `main` as `6ddc268`;
  `mode=full` could not have succeeded without it. Detail in
  `docs/changelog/2026-08-09-training-cost-levers.md`.
- **Do (historical):** dispatch `price-forecast.yml` with `mode=full`.
- **Why:** three consecutive failures since 2026-08-08 07:46 UTC with `saved model artifact
  version 3 != expected 5` — the Actions booster cache predates the minimal-model rewrite and
  `load_models()` refuses it rather than serving a band computed by a different scheme.
  `item_forecasts` is frozen at forecast_date 2026-08-07 and Backtest Accuracy is `skipped`
  behind it, so **no labels are maturing** and the ≥20-forecast-date wait is not elapsing.
- **Note:** `mode=full` only trains if the artifact is ≥14 days old or `FORCE_RETRAIN=1`. Here
  the artifact fails to *load*, so the retrain path is reached regardless — but set
  `FORCE_RETRAIN=1` to be certain.
- **Cost:** one dispatch. **Measured: 35m24s** for the job, `TOTAL training: 1884.2s` (31.4 min)
  on 918/5,536 items and 987,649 rows — the first CI wall-clock for the shipped config, and
  **over the project's 30-minute cap**. The ≈20–25 min figure below was derived from the 10-core
  Mac at an assumed 1.4–1.5× thread ratio; the real ratio against the 872s local cold is **2.16×**.

### O2. Correct the stale `architecture/` claims — 🟡 **the +3.50pp half is DONE (`895005a`); the config description is not**

> ✅ **The number is fixed.** `model.md:212` and `:540` now read *"the stored +3.50pp at 30d
> re-derives to +1.642pp [−0.809, +4.505], null"*. That was the narrow item this entry was
> originally filed as, and it is also Task 7 Step 4 of
> `docs/superpowers/plans/2026-08-09-training-cost.md`.
>
> ⬜ **Still open: everything in the scope correction below** — `:189-196`, `:204-217` and
> `:529-537` still describe the 99-item / 100K-row config as production, and
> `model-optimization.md` has the same defect in five more places. Roughly 25 stale claims across
> the two files. That is a rewrite of two reference documents, not a status pass.
>
> ⚠️ **Scope correction (2026-08-09): this is not one number, it is at least eight**, so the "one
> live stale number in `architecture/`" framing below is wrong and the `model-and-data-research.md`
> §6 claim inherited the error. Also outstanding in `docs/architecture/`:
>
> - `model.md:196` and `:529` — "selects **99 of 5,377 items** — 1.8% of the pool". Shipped config
>   selects **918 of 5,536 items, 987,649 rows** (CI log, run `31337078991`).
> - `model.md:194` (`min_median_price | None`), `:209` ("It is **defaulted off**"), `:534` ("The
>   floor is defaulted off") and `model-optimization.md:125` (same) — all four false since
>   2026-08-08. `train()` now defaults `max_feature_rows=1_200_000, min_median_price=1.0`.
> - `model.md:530` — "`predict()` writes forecasts for **8,691 distinct items**". The mirror's
>   latest date has **5,536**.
> - `model.md:355-356` — "cold 250.1s / warm 176.7s", with no supersession note.
>   `model-optimization.md:28-29` at least flags the 872s figure; `model.md` does not.


- **Do:** `docs/architecture/model.md:210` and `:533` cite **+3.50pp at 30d** for the ≥$1
  training floor as settled. It was withdrawn 2026-08-08 — the same contrast re-derives to
  **+1.642pp [−0.809, +4.505], null**, and the placebo puts the instrument's item-draw noise
  floor at ±3–4pp (`docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`).
- **Why:** ~~it is the one live stale number in `architecture/`~~ — see the scope correction
  above; it is one of at least eight. Note also **which** +3.50pp this is: both `model.md:210` and
  `:533` cite the **`TRAIN_MIN_MEDIAN_PRICE`** result, the one that re-derives null. A *separate*
  `+3.5pp @30d` — the cross-sectional-removal result at
  `docs/research/2026-07-19-feature-contribution-by-horizon.md:29`, which founds
  `FEATURE_GROUP_ALLOWLIST` — has **never been re-derived** and appears nowhere in
  `docs/architecture/`. Do not treat the two as one number; only the price-floor one is settled.
  `model.md` also still describes
  the 99-item / 100K-row config as production; the shipped default is
  `DEFAULT_TRAIN_MIN_MEDIAN_PRICE = 1.0` + `DEFAULT_TRAIN_FEATURE_ROWS = 1_200_000`
  (`scripts/forecast_prices.py:51,63`), i.e. 926 items with no subsample.
- **Cost:** tiny.

### O3. Publish the 17 files missing from the canonical archive — ✅ **DONE**, one leg open

- **Outcome (verified 2026-08-09 against `RayanR000/cs2-oracle-data`):** all are published — and
  the real count was **22, not 17** (the enumeration below sums to 19, and it undercounts
  `player-counts` by 2 and omits `item-metadata-bymykel-codes.json`). Fixed by push `11e68b58` at
  20:49 UTC, which corresponds to **no workflow run** — a manual push, not a CI fix. ⚠️ **Nothing
  in `.github/workflows/` generates `item-metadata-bymykel.parquet`**; it survives only because
  `aggregator-update.yml` publishes with `git add -A` over a fresh checkout, so it will vanish the
  first time that checkout does not contain it.
  `item-metadata.parquet`, `item-metadata-bymykel.parquet` + codes JSON,
  `exchange-rates-history.parquet`, `player-counts-2011…2026` (16 files, one more than counted
  here) and `ops/` (all 7 tables: `collection_runs`, `events`, `accuracy_alerts`,
  `item_forecasts`, `forecast_outcomes`, `prediction_accuracy`, `event_impacts_denorm`) are all
  present. `supply-2026-08.parquet`, `event-calendar.parquet` and `event-news.parquet` publish too.
  **The ByMykel-in-CI blocker named below is therefore cleared** — the file it reads is now
  published, so `BYMYKEL_METADATA=1` is testable in CI (the feature verdict is unchanged: refuted).
- ⬜ **Still open — one stranded file.** The canonical repo still carries
  `snapshots-2026-08.parquet` (11,634,574 bytes, `2026-08-01`→`08-07`, 1,810,858 rows). It has
  **stopped growing**, so it is a leftover to delete rather than a live output.
  ~~the compaction was never dispatched~~ — **that framing is wrong**: compaction largely landed in
  prod. Canonical `prices-2026-08.parquet` columns are already
  `item_slug, day, source, mean_price, volume, ingested_at` (the `median_price`/`min_price`/
  `max_price` strip is done), 5 of the 6 named snapshot files are gone, and total canonical size is
  **122.6 MB** against the changelog's pre-compaction 210.7 MB. Only this one deletion is left.
- **Do (historical):** add `item-metadata.parquet`, `item-metadata-bymykel.parquet` (+ codes JSON),
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

### G1. Power up the source-composition test — ✅ **DONE 2026-08-09. It returns, and it lifts the gate.**

> **Built 2026-08-09.** `n_ask_sources` is a stored column on the voted frame (commit `48fd352`,
> `VOTED_CACHE_VERSION` 4 → 5), and `backend/scripts/measure_composition_stability.py` +
> `tests/test_composition_stability.py` are the partitioned instrument (commit `80c3e07`, 16 tests).
> It reads through `prices_relation` with the universe filter, votes through
> `_apply_multi_source_voting` rather than reimplementing it, and applies the void-date exclusion
> itself.
>
> **The measurement changed the question.** The published §5 baseline **does not reproduce**, and
> the reason is definitional, not a bug: `composition stable` was computed with a NULL source never
> equal to itself, and every pre-2026 row has `source IS NULL` (9,417,947 item-days). So the
> published "stable" cell holds **no pre-2026 data at all**, and its 775 "changed" dates are ~716
> pre-2026 dates classified as changed for want of a label. **The +0.1676 → +0.1006 "under
> composition control" is a 2013-2025 → 2026 regime difference, not composition control.**
>
> Within 2026, where composition is actually observed: **stable +0.1027 against an unconditional
> +0.1023, on 181 of 185 dates — powered, and the signal does not move.** At h=7 the two agree to
> four decimals (+0.0842 both, 167 of 172 dates). ⚠️ **The changed cell is underpowered and carries
> no number** — 25 dates at 3d, 19 at 7d, once "composition genuinely differed" is separated from
> "the window had a missing day".
>
> **The count-vs-set caveat is resolved, not outstanding.** The concern was that `n_ask_sources` is
> a count, so a source swap at constant cardinality reads stable and the instrument under-detects
> composition change. The write-up therefore made the **set** basis primary. Measured: the two bases
> disagree about **106 windows out of 1,912,978** (0.006%). Correct in principle, immaterial here.
>
> **✅ Write-up complete, all three legs committed.**
> `docs/changelog/2026-08-09-composition-stability-refutes-quoting-artifact.md`;
> `2026-08-08-model-review.md` §5 corrected in place — the four composition rows struck through and
> annotated, numbers kept rather than deleted because they reproduce exactly under the old rule; and
> `docs/research/2026-08-09-composition-stability.md` carries the full result (commit `f833882`).
> **D3 is formally answered.** Basis as agreed: primary = 2026 observable era on set-based
> composition; secondary = full archive on the count basis with its assumption stated.
>
> **The one cell that did not resolve** is "stable & ≥3 agreeing sources" — 25 dates at 3d, 19 at
> 7d, both under `MIN_DATES_TO_REPORT = 30`, so no number is quoted. Blocked on calendar time, not
> effort. Per the gating section above, this does **not** hold Track C.
>
> ⚠️ **A by-product worth carrying forward:** `_collection_shift_dates` is a function of the item
> universe handed to it. On the backfilled-only frame it fires on `2026-03-22, 07-09, 07-10, 07-11,
> 07-12`; on the full-universe frame it *also* fires on **`2026-04-16`, `2026-07-14`, `2026-07-15`**,
> and over the full archive on five 2013-2016 startup dates as well — 13 in total. "Which dates are
> void" is therefore not a property of the archive alone. Recorded, not fixed.


- **Do (historical):** extend `docs/research/2026-08-08-model-review.md` §5 from 932 dates to the
  full archive, and from the 28-date "three agreeing sources" cell to as many dates as the
  multi-source era supports. Report rank IC of `−r_t` predicting the forward 3-day return,
  partitioned by (a) source-composition stability across t−1…t+3, (b) source count, (c) whether the
  window spans any known break date. Add `n_ask_sources` as a stored column so the partition is
  auditable rather than recomputed.
- ~~**Why:** the effect is +0.1676 pooled, +0.1006 composition-stable, and **+0.0044 (t = 0.1)** on
  the cleanest subset. That last cell has 28 dates. Everything downstream — the objective swap,
  the rank transform, the allowlist re-derivation — is conditional on it.~~ **All three of those
  numbers are refuted as a composition contrast** — see the status block above and the gating
  section. Nothing downstream is conditional on them.
- **The honest constraint, and it held:** the multi-source era is **24 days deep** (§4b of the
  review). Two windows are effectively single-feed: 2026-01/02 (`sync` alone) and 2026-04-16 →
  07-10 (`17mafo` alone — with one hole, 2026-07-09 is `aggregator_sync` alone). This is exactly
  why the ≥3-source cell came back at 25 dates and could not be reported: *"unresolvable until more
  multi-source days accumulate"* is the outcome on that one cell, and it is a decision-grade result.
- **Entry criteria:** none.
- **Cost:** small — archive queries only, no fits.
- **Unblocks:** all of Track C.
- **What would invalidate it:** running it across 2026-03-22 or the 2026-07-09…11 handover
  without excluding those windows. Both are already in `_collection_shift_dates`' fired set, but
  G1 reads the archive directly rather than through `prepare_targets`, so it must apply the
  exclusion itself — that is the one real dependency on G2 part 1.

### G2. Publish the detector's fired-date list, and add the 2025 regime markers — 🟡 **PART 1 DONE, PART 2 NOT STARTED**

> **Part 1 shipped 2026-08-09** (commit `3d582ab`, + `5e7dc60`). `ItemForecaster.label_voiding`
> records `snapshot_dates`, `collection_shift_dates`, `voided_labels_by_horizon` and
> `frame_date_range` as sorted ISO strings; `save_models()` writes it into `meta.json`;
> `tests/test_label_voiding_audit.py` pins it. No detector logic changed —
> `test_a_price_crash_is_never_flagged` passes untouched.
>
> **The measurement, which was this item's whole point.** On the backfilled-only frame the
> detectors fire on: **cutovers 2026-03-22, 07-09, 07-10, 07-11, 07-12**; **snapshots 2026-07-16,
> 07-22**. So **2026-07-11 IS caught** — the open question is closed, and no special case is
> needed. 07-12 is caught too and had never been documented anywhere.
>
> ⚠️ **New finding this produced.** `_collection_shift_dates` additionally fires on **2026-04-16,
> 07-14 and 07-15** on the *full-universe* frame but not on the backfilled-only one. "Which dates
> are void" is a function of the universe handed to the detector. Previously unrecorded, and it
> means the fired-date list is not a single fact about the archive.
>
> ⬜ **Part 2 (the two 2025 regime boundaries) is not started.** `REGIME_WINDOWS` /
> `regime_window` / `regime_stress` still return zero hits, exactly as R18 recorded. Everything
> written below about them — including that they must **not** be voided — still stands.


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

### G3. Stop the Steam MA feeds voting in the consensus — carried forward as **6c**, ✅ **DONE 2026-08-09** (`873148b`)

> **Shipped.** `TRAILING_WINDOW_SOURCES` lives in `models/item_parser.py` beside `BID_SOURCES`,
> both re-exposed as `ItemForecaster` class attributes; the exclusion is applied in
> `_apply_multi_source_voting` (kept `@staticmethod`, NULL-safe via `.isin`);
> `VOTED_CACHE_VERSION` 5 → 6 with the workflow key moved `voted-v4-` → `voted-v6-` in the same
> commit. 24 targeted tests plus 280 regression tests pass. Changelog:
> `docs/changelog/2026-08-09-trailing-window-sources-excluded.md`.
>
> **The mechanism in the "Why" below is corrected there.** These feeds do not sit *below* live
> asks like the bid — measured against genuine third-party asks they sit **~32% ABOVE** them
> (median ratio 1.316, on 76.6% of item-days), the same Steam cash-out wedge every Steam-derived
> feed carries, so excluding them pulls the consensus **down**. The defect is the **time** basis,
> not the level: a trailing 90-day mean barely moves when live asks move, so it damps the
> consensus and manufactures mean-reversion — which is exactly the effect G1 was validating.
>
> **G1 was re-run against the post-exclusion vote** (`3c62a49`): every measured cell moved
> 0.0000–0.0015 rank IC, within what the ~29 exposed dates can explain, and the ruling is
> unchanged.
>
> ⚠️ **Owing: the retrain (O4).** Every label from 2026-03 onward sits downstream of this, and
> nothing has retrained on it yet. No stored A/B verdict is citable against a post-`873148b`
> number.


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

## Track A — cost and instrumentation — ✅ **CLOSED 2026-08-09. All six shipped.**

Full spec: `docs/superpowers/specs/2026-08-09-training-cost-design.md`.
Full plan: `docs/superpowers/plans/2026-08-09-training-cost.md`.
**Measured outcome: `docs/changelog/2026-08-09-training-cost-levers.md`.**

> **Read the changelog before quoting anything below.** Three of this section's own predictions did
> not reproduce, and the retrain got **slower**, not faster:
>
> - **No net speedup is established.** Cold retrain measured **1426.3s** against a predicted ≈600s.
>   Three confounds and no like-for-like control: A2 removed early stopping from the trial loop, so
>   Optuna became the single largest phase at **392.3s / 27.5%**; `FORCE_HP_SEARCH=1` forces a search
>   a normal retrain reuses from cache; and the 872s baseline was CI hardware under early stopping,
>   before `beec500`. **Do not quote a speedup from this work.**
> - **A3+A4 saved 16.5s, not the 33.7s projected** (isolated properly: same code, same frame, knobs
>   flipped — 24.5s → 8.0s, feature set identical at 33 columns both ways).
> - **The D2 coverage gate was not executable.** `meta.json` stores no coverage field, and `q_hat`
>   *is* the conformal quantile of its own calibration pool, so it cannot under-cover it. The useful
>   read was the fallback: folds 9/8/8/8, every fold capped at exactly 300,000, OOF pools six figures,
>   `q_hat` up at all four horizons — the band got **wider**, the safe direction. Verdict: proceed.
> - **The "previous artifact" baselines were unusable.** This machine's prior artifact was
>   `model_artifact_version: 3` from 2026-08-06, predating both the v5 dollar-scale migration and the
>   ≥$1 universe. Taken at face value it would have reported a false positive on A3/A4 and a false
>   "cap did nothing" on A1.
>
> Two bugs the retrain caught that a green suite did not: A6's first implementation keyed on a
> `date` column the archive does not have (it is `day`), and a NULL `volume` column crashed feature
> engineering in production — fixed as `6ddc268`, cherry-picked to `main` ahead of the branch.

### A1. Cap CV fold training rows — ✅ **DONE** (`6b6fc81`)

> `CV_MAX_TRAIN_ROWS = 300_000` at `models/forecaster.py:540`, env-overridable at `:543`. Binds at
> exactly 300,000 on all four horizons. Fold count and OOF pools unchanged; band widened.


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

### A2. Re-tune Optuna against rank IC at fixed rounds — ✅ **DONE** (`8be48c5`)

> Objective is now within-date rank IC at `_boost_rounds(horizon, cv=True)` (`forecaster.py:2947`);
> `MODEL_ARTIFACT_VERSION` 5 → 6 so the stale params cannot be reused. **The cost estimate below is
> wrong by an order of magnitude:** this was priced at 35s and measured at **392.3s**, because
> removing early stopping means every trial trains the full round budget instead of stopping near
> 25. That is a foreseeable consequence the plan did not budget for. Whether the newly selected
> params forecast *better* is unmeasured — this changed the selection criterion, not the outcome.


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

### A3. Allowlist before the correlation prune — ✅ **DONE** (`a20b5a2`)

> `ALLOWLIST_BEFORE_PRUNE = True` at `forecaster.py:360`, behind the flag as required. Feature set
> verified set-identical both ways at 33 columns, so the invariant held. Saving is **16.5s combined
> with A4**, not the 23.5s projected here.


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

### A4. Short-circuit the discarded feature blocks — ✅ **DONE** (`17306b0`)

> Default **off**; only `build_training_data` passes `skip_unused_groups=True`, so the seven
> harnesses below still get the full 123-column frame. The skip set is **derived from the
> allowlist, never a literal** — `test_skip_set_follows_the_allowlist_not_a_literal` is the guard,
> which is what keeps C4 safe. Plan discrepancy found in implementation: only 6 of the 8 blocks are
> inside `engineer_features`; `_add_cross_sectional_features` and `_add_supply_depth_features` are
> applied in `build_training_data` and carry their own guards off the same derived set.


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

### A5. `CV_DIAGNOSTIC_CLASSIFIER` default off — ✅ **DONE** (`dfafdfb`)

> `forecaster.py:6151` now reads `os.environ.get("CV_DIAGNOSTIC_CLASSIFIER", "0") != "0"`. The
> shipped artifact's `mean_classifier_acc` / `mean_classifier_acc_ge1` are `null`, as predicted —
> `mean_rank_ic` and the PT verdict are the replacement, and both survive.


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

### A6. Content-hash the archive fingerprint — ✅ **DONE** (`db8d5d5`, fixed by `1b5ee06`)

> `_archive_fingerprint` (`forecaster.py:3633`) is now row count + byte size per file, and
> `price-forecast.yml` caches `backend/data`. **It names no column at all** — the first
> implementation keyed on `MAX(date)` and died in the cold retrain with a binder error, because the
> archive's date column is `day` and the archive is not schema-uniform (`prices-2026-03` and `-04`
> carry `min_price`/`max_price` the other 19 files do not). All 1693 tests passed while it was
> broken, because both the plan's fixture and the implementer's had invented a `date` column.
>
> ~~⚠️ **The manual-bump obligation below is live and currently unmet.**~~ **Met 2026-08-09**:
> G3 (`873148b`) set `VOTED_CACHE_VERSION = 6` and the workflow key to `voted-v6-` together. The
> obligation itself stands for every future change to `_fetch_voted_price_history` or
> `_apply_multi_source_voting`.


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
  rule and A6 does not change it. ~~Currently at **4**~~ — now at **5** since `48fd352`.

---

## Track C — accuracy — ✅ **UNBLOCKED 2026-08-09** (G1 returned; see the gating decision)

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
- **Entry criteria:** ✅ **satisfied** — G1 returned "the signal survives composition control"
  (2026-08-09). This is the next thing to do.
- **Test it alone before C2.** It is the cheaper half of step 10 and isolates cleanly.

### C2. `lambdarank` within date, scored by rank IC — 🟡 **the rank-IC half is DONE; `lambdarank` is not**

> **Already shipped, so do not re-propose it:** within-date rank IC is the Optuna objective
> (`8be48c5`), and every fold now stores `rank_ic` alongside `naive_rank_ic` — the `−return_1d`
> baseline — at `forecaster.py:6088-6116`. The shipped artifact reports **`mean_rank_ic` 0.1307 vs
> naive 0.1656, edge −0.0349**, i.e. the metric is in place *and it confirms the model still loses
> to the one-line baseline*. That is the measurement C2 exists to move.
>
> ⬜ **Not started:** the objective swap itself. `lambdarank` returns zero hits in the codebase.


- **Do:** objective swap. Query groups = forecast dates; rows contiguous and sorted by group.
- **Why:** quantile loss on raw % returns targets a conditional median per row; nothing in
  training optimises within-date ordering, which is what the dashboard serves. The model loses to
  ranking by `−return_1d` at all four horizons.
- **The evidence, and it does not support the cheap version.** LambdaRankIC (arXiv:2605.00501,
  **preprint, unreviewed**, XGBoost not LightGBM, 2.7M stock-months) measures regression 0.042 →
  pairwise LTR **0.083** → NDCG LTR 0.086 → their method 0.115. Those rank-IC figures are verified
  correct against the paper. In the same table plain regression *underperformed OLS*, which is this
  project's experience.
- ⚠️ **But the paper's own portfolio results argue against the plain-LTR shortcut, and the review
  omitted them.** In its backtest columns the two stock-LightGBM-reachable arms are **worse** than
  the regression baselines: Sharpe **0.566** (pairwise) and **0.501** (NDCG) against **0.740** (OLS)
  and **0.831** (MLP), with max drawdown **81–83%** against 44–46%. Only the authors' own
  LambdaRankIC (Sharpe 0.923) dominates — and that is the arm requiring custom gradient code. The
  table also carries an **MLP regression** row at ICIR **0.806**, which beats both LTR arms and
  which the review did not report. ~~The regression → LTR jump is the larger and the
  replicable-looking one~~ — **that is the one inference this table does not license.** The rank-IC
  gain did not convert into tradeable performance.
- **What that changes:** the cheap experiment is still worth running, because rank IC is what this
  product serves and a drawdown objection does not apply to a per-item forecast surface. But
  **budget it as a metric experiment, not as an expected performance win**, and read
  `naive_rank_ic` on the same folds as the bar.
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
- **Entry criteria:** ✅ G1 satisfied; C1 must still be measured separately first.
- **Honest bar — and it is softer than it looks.** Qlib's published LightGBM benchmark on daily
  cross-sectional CSI300 is Rank IC **0.0469** (Alpha158) / **0.0499** (Alpha360), so "≈0.05" is
  right. But it is **not like-for-like**: Qlib gets that IC at Rank ICIR **0.39–0.40**, while this
  model claims Rank IC 0.092–0.183 at ICIR 0.34–1.45. A 2–4× higher IC at comparable-or-lower ICIR
  is the signature of a different sampling regime — a ~900-item daily panel with strong within-date
  co-movement versus a 300-name equity cross-section — not of a better model. That reading is
  consistent with `−return_1d` beating this model at all four horizons. **Do not use 0.05 as the
  bar; use `naive_rank_ic` on the same folds.** Currently `mean_rank_ic` 0.1307 vs naive 0.1656.
- ⚠️ **The ICIR convention quoted elsewhere in these docs ("above 0.5 good, 1.0+ excellent") is
  borrowed from the wrong statistic** — it is Grinold & Kahn's rule of thumb for the *Information
  Ratio* (annual residual return ÷ residual risk), not for ICIR (mean IC ÷ sd of IC). No standard
  ICIR convention exists. The conventions that do circulate are on IC itself: 0.02–0.07 typical for
  published academic factors, 0.10+ exceptional.

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
- **Entry criteria:** ✅ G1 satisfied. Also read `_validate_feature_groups` before any paired retrain.
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
- **Entry criteria:** ✅ G1 satisfied; A4 must respect the allowlist rather than a hard-coded block
  list — it does (`test_skip_set_follows_the_allowlist_not_a_literal`).
- **Cost:** medium — it is a full ablation, and the MDE must be computed first.

### C5. Split conformal + ACI — carried forward as **R18**

- **Do:** replace the K-fold OOF calibration with a single split-conformal calibration slice, and
  add Gibbs & Candès (2021) Adaptive Conformal Inference for drift.
- **Why (cost):** split conformal needs **one** fit; the CV block currently pays 33. The standard
  variance objection to split conformal is calibrated to n in the hundreds, and this frame has
  ~1M item-days.
- ⚠️ **Re-baseline the saving before quoting it.** The "−50%" below is CV's share of the retired
  **872s** figure (439.3s). Post-A1 the measured cold retrain is **1426.3s** with conformal CV at
  **434.8s**, so the ceiling is now **~30%**, not 50%. Note also that A1's 300k fold cap *bound on
  every fold* and the CV phase moved 439.3s → 434.8s — i.e. the −194.6s A1 projected did not
  materialise on the one adjacent measurement. Either the rows×rounds cost model in
  `2026-08-09-model-and-data-research.md` §2b is wrong, or the two runs are not comparable
  (different hardware, different round counts). **Establish a like-for-like control before pricing
  C5.**
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

- **Do:** either set `SKIP_REGIMES=1` in CI or justify the regime models on their merits.
- ⚠️ **The 95.4s / 10.9% is against the retired 872s baseline**, and regime models do not appear in
  the 1426.3s run's phase table at all, so their current cost is **unmeasured**. Do not quote a
  share until it is re-measured. The case for cutting rests on the degenerate boosters, not cost —
  which the entry already says, and which is now the *only* argument available.
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

### D1. Re-verify the Steam listing page from this project's egress — ✅ **DONE 2026-08-09. It answers.**

- **Outcome:** the page **does respond from this egress**, at **0.36 requests per item**. Blocker
  5d of `2026-08-07-next-steps.md` was a **false positive**: the 230 KB shell returned by the
  `market_hash_name` URL is normal SSR, not a soft-block. You must resolve the canonical `G<id>`
  first — the two-step `render/?…` → `G<id>` route returns the full ~5 MB page with
  `purchases` back to 2014 and full order-book depth.
- **So D1's "if it holds" branch is the live one.** `R11`, `R13` and `5c` are unblocked **on
  access**. None of them has been built, and no backfill has run.
- ⚠️ **The collector was not updated.** `backfill_steam_listing_history.py:67` still fetches the
  name URL and the block detector at `:271` still keys on the shell — so the code will keep
  reporting a block that is not there. That is the first thing any backfill needs.
- **Do (historical):** fetch `steamcommunity.com/market/listings/730/<name>` from the machine and from a CI
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
- **Entry criteria:** none. This was the one Track D item worth doing before the gate returned,
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
- **Entry criteria:** ✅ G1 satisfied. It is an item-level feature and will not clear the item-level MDE on its
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

⚠️ **Weight that citation accordingly.** arXiv:2607.21866's headline numbers check out (18 datasets,
11,536 runs, 1,648 curves, boosting median b̃ = 0.480, floors c = 0.110 RF / 0.208 boosting), but the
runs were executed by **127 graduate students under unconstrained preprocessing**, and the paper's
own third finding is that implementation variance across replications moved the fitted exponents. It
is also a **classification-error** benchmark; mapping its `c` onto a return-forecast R² is this
project's analogy, not the paper's, and the "orders of magnitude larger than for LMs" gloss is
unsourced. The conclusion is probably right; it is an argument, not a measurement.

### "Does volume affect accuracy or speed"

**Accuracy: a clean pre-cliff window exists** — 2013-08-14 → 2025-12-31, 4,523 days, 9.42M
volume-bearing rows, 5,536 items (871 at ≥$1), zero missing calendar days. Four things would
invalidate a naive run; all four are in `docs/research/2026-08-09-model-and-data-research.md` §4f.
The short version: cut at **2025-12-31**, not at the harness's `VOLUME_LIVE_THROUGH =
"2026-04-30"`; never pool across 2026-03-22, where `ask_volume` starts masquerading as `volume`;
and know that the pre-2026 series has **literally zero zero-volume rows**, so a no-sale day is an
absent row and any feature built on it conditions on a sale having occurred.

**Speed: not an experiment.** Re-admitting 13 of ~45 columns moves the booster-fit phases, and
per-phase timings on this hardware swing **±25% between clean runs at identical config**. (The
"39% of an 872s retrain" share is against the retired baseline — the measured cold retrain is now
**1426.3s**; see `docs/changelog/2026-08-09-training-cost-levers.md`.)

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
