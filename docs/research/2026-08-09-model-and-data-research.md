# Model and data research, 2026-08-09

**Method:** five parallel audits — training cost (fresh measurements on this machine against
the same 918-item / 986,065-row frame the 872s run used), data inventory (DuckDB over the local
archive + `gh api` over the canonical repo), experiment-harness capability, and two web-research
passes (forecasting literature; CS2 data sources). Every number below is either measured in this
review or cited to the changelog entry that measured it. Nothing is estimated.

**Roadmap:** `docs/research/2026-08-09-next-steps.md`.

---

> ## ⚠️ Corrections, 2026-08-09 (later the same day)
>
> **Every Track A recommendation in this review shipped within hours of it being written, and three
> of its cost predictions did not reproduce.** The review is otherwise unusually accurate — 13 of 19
> of its archive numbers reproduce to the digit on re-query, its statistical tables partition
> exactly, and no citation in it is fabricated. What follows is what changed and what was wrong.
> Numbers are kept in place rather than deleted.
>
> ### The gate in §1c is refuted — this is the important one
>
> **§Summary item 1 ("the label may be a quoting artifact… it gates every accuracy item") and §1c
> are withdrawn.** The table §1c reproduces from `2026-08-08-model-review.md` §5 is struck through
> *in that document* as refuted. The partition classified every NULL-`source` item-day as
> "composition changed", and `source` is NULL for every pre-2026 row, so the 0.1676 → 0.1006 fall is
> a **2013-2025 → 2026 regime difference, not composition control**. Re-measured on the set basis
> with void dates excluded: **stable +0.1027 vs unconditional +0.1023** at 3d (181 of 185 dates),
> and equal to four decimals at 7d (+0.0842, 167 of 172).
> `docs/research/2026-08-09-composition-stability.md`, commit `f833882`. **Track C is not gated** —
> the gate's premise was that the reversal is *made of* composition change, and removing
> composition change does not move it.
>
> Two cells did not resolve, and the write-up is explicit about both: **"composition changed
> (present)"** is 25 dates at 3d / 19 at 7d, so the comparison above is stable-vs-unconditional
> rather than stable-vs-changed and **no directional claim about the changed population is
> supported**; and **"stable & ≥3 sources"** is the same 25/19, which is the cell §1c leant on. Both
> are calendar waits, not effort.
>
> ### Three cost predictions failed
>
> | § | Predicted here | Measured (`docs/changelog/2026-08-09-training-cost-levers.md`) |
> |---|---|---|
> | §3a re-tune Optuna | "**35s** of an 872s run. Highest accuracy-per-second lever in the system" | **392.3s** — 11×, and now the single largest phase. Removing early stopping means every trial trains the full round budget instead of stopping near 25 |
> | §2b cap CV folds @ 300k | **−194.6s** (−22.3%) | conformal CV went 439.3s → **434.8s**, and the cap **bound on every fold**. The saving did not appear |
> | §2d allowlist + skip blocks | "≈**32s** recoverable" | **16.5s**, isolated properly |
>
> Cold retrain measured **1426.3s**, not the ≈600s the plan projected. That changelog's own
> conclusion: *"Do not quote a speedup from this work."* **Consequence for §3f/§2a: every share in
> this document is against the retired 872s baseline.** Conformal CV is now 30.5% of a retrain, not
> 50.4%, so the split-conformal argument's ceiling is ~30%; regime models do not appear in the new
> phase table at all, so their cost is unmeasured.
>
> ### Four other claims were settled, mostly by shipping
>
> - **§5c "uncommitted" is stale.** Rank IC and Pesaran–Timmermann are committed; `grep -c rank_ic`
>   on `HEAD` returns 23, not 0. Every element §5c enumerates exists.
> - **§1a's open question is closed.** `_collection_shift_dates` **does** fire on 2026-07-11 —
>   measured list on the backfilled frame is `2026-03-22, 07-09, 07-10, 07-11, 07-12` (commit
>   `3d582ab`). No special case is needed. A by-product: the fired set is a function of the
>   *universe handed to the detector*, so it is not a single fact about the archive.
> - **§4c's "verify from the project's own IP" is answered — it works**, at 0.36 requests per item.
>   The 230 KB shell is normal SSR; you must resolve the canonical `G<id>` first.
> - **§6's four operational defects are all resolved or misstated** — see the block below.
>
> ### Wrong on their own terms
>
> - **§2f: `_warm_retrain` suppresses four things, not three.** The fourth is
>   `HORIZON_EXCLUDED_GROUPS`, and it was already there at this document's own commit — so this was
>   never right. **It matters:** a warm retrain trains 14d/30d on the *full* feature set, so warm and
>   cold are not the same model at those horizons. The "warm ≈ 730s" figure is also inferred, not
>   measured — 872 − 35.0 − 95.4 = 741.6, and the permutation test's cost is never given. This
>   document's method claims "Nothing is estimated"; that line is an estimate.
> - **§5d: "15 of 15 still early-stop" is 14 of 15.** `ab_test_direction_labels.py` does not — it
>   calls the production estimator, whose `early_stopping` default is `False`. The 9-harness
>   wrong-universe-filter count is exact, as are all six duplication counts and the 8,617-line total.
> - **§2c: `feature_pre_filter: False` is at five Dataset sites, not four** (the regime-model site is
>   omitted).
> - **§2d's table does not sum to its own totals**: 17.33s against a stated 17.5s, and **188 columns
>   against a stated 192**. The 48%-discarded conclusion is unaffected.
> - **§2g's "elasticity 0.79–0.83" cannot be estimated from the series it cites.** Two of the three
>   points are a two-parameter fit with zero residual degrees of freedom, and the third (372 items /
>   410,780 rows) misses the fitted line by 24%. It also sources two rows to `model-optimization.md`
>   while calling them "the 2026-08-05 budget sweep". The "+62% is **entirely** the round count"
>   attribution is likewise too strong for a machine whose per-phase timings the same section says
>   swing ±25%.
>
> ### Data: what did not reproduce
>
> Re-queried 2026-08-09 through `db/archive.py::prices_relation` with `archive_universe_sql_filter()`.
> §4a's five claims, §4b's spans, §4f's cliff date, the four missing archive days and the
> NULL-source provenance all **confirmed exactly**. These did not:
>
> - **§4b silently mixes three filters and has no filter column.** Its row counts are the
>   *slug-rules-only* relation; §1a's "panel reaching 39,366 items" is the *unfiltered* csgotrader
>   count (36,099 filtered). Each is right under its own filter; none is reproducible from the table.
>   Also `aggregator_buff163_buy` is **603,243**, not "~570k", and a 13th row is missing:
>   `historical_fallback:aggregator_sync`, 12,655 rows, 2026-07-11 → 07-16.
> - **§4b/§1a: 2026-04-16 → 07-10 is single-feed on all 86 days but `17mafo` on only 85.**
>   **2026-07-09 — the date carrying the +26.46% median — is `aggregator_sync` alone, 2,373 rows.**
>   That is why n on 07-09/07-10 is exactly 2,372: the panel is the sync intersection, not the
>   27,194-item `17mafo` universe. The cutover story holds; the mechanism on 07-09 is a one-day feed
>   substitution, not the handover. (07-11's median re-measures at +14.07%, not +14.26%.)
> - **§4f's 8,622,895 zero-volume rows is not reproducible under any of eight filter combinations**
>   (8,500,521 full-filter / 9,103,764 slug-rules / 9,918,361 raw). The substance is stronger than
>   stated: 2026-04-16 → **08-07** is **100%** `volume = 0`, non-NULL. 2026-08-08 is different again
>   — 327,840 **NULLs** — which is the crash O1 uncovered.
> - **§4f's "871 at ≥$1" does not reproduce** — 898 by median price, 923 by latest, 1,050 by last
>   pre-2026 price, 3,021 ever-≥$1. The definition is load-bearing and unstated.
> - **§4f's per-source medians (117/50/335/966/1,046) are exact but are medians over `volume > 0`
>   rows only.** The literal median per row is **0** for every 2026 source — buff163 has 673,963
>   volume-bearing rows of 1,520,462. The phrasing understates how sparse 2026 volume is.
> - **§1d's `STEAM_FEE_MULTIPLIER` audit rests on a file not in the archive.**
>   `backend/runtime/steam_listing_history.db` does not exist on this machine, so the 63,767-pair
>   measurement is not reproducible. The nearest available pairing (staged `tracker_steam_24h` vs
>   `aggregator_sync`, 70,988 pairs) shows **no flat-constant signature** — median 1.163, IQR 0.088,
>   only 2.59% identical after dividing. Different pairing, so it neither confirms nor refutes, but
>   the circularity finding is currently unverifiable.
> - **§1c's 2026-03-22 sizing understates it.** −7.77% at ≥$1 (n=620), but **−18.10% pooled**
>   median / −23.51% mean. Neighbouring days are 0.00% to −3.43%.
>
> ### Citations: none fabricated, four misattributed, one misread
>
> All three 2026 arXiv IDs exist and are the papers described. **The §3c LambdaRankIC table is
> correct digit-for-digit**, as are Gu/Kelly/Xiu's 0.33–0.40%, the footnote-29 transform, Qlib's
> Rank IC ≈ 0.05, the LightGBM defaults, and every conformal-prediction guarantee. However:
>
> - **§3c's reading of LambdaRankIC is the one inference that table does not license.** In the
>   paper's own portfolio columns the two stock-LightGBM-reachable arms are **worse** than the
>   regression baselines — Sharpe **0.566** (pairwise) / **0.501** (NDCG) against **0.740** (OLS) and
>   **0.831** (MLP), max drawdown **81–83%** vs 44–46%. Only the authors' own method (0.923)
>   dominates, and that is the arm needing custom gradient code. The table also has an **MLP
>   regression** row at ICIR **0.806**, beating both LTR arms, which this review omitted. So
>   ~~"the regression → LTR jump is the larger one and the replicable-looking one"~~ is not
>   supported: the rank-IC gain did not convert into performance.
> - **§3f misattributes the cross-conformal bound.** The formula is right but is Vovk et al. **2018**
>   plus Barber et al. 2021 App. B.2.1; **Vovk (2015) explains why cross-conformal *lacks* the
>   inductive guarantee.** Also, Barber's CV+ bound is strictly ≥ 1−2α−√(2/n) — immaterial at n≈1M,
>   but the table states it without the term.
> - **§3d overstates Da/Liu/Schaumburg.** The published paper says the residual-reversal strategy is
>   **four times** the standard one, not ~3×; the 3× and the four-component decomposition both come
>   from the earlier working paper. "Only the residual is significant" holds.
> - **§3e does not note that Kelly/Malamud/Zhou is contested** (Nagel; Buncic argue the empirical
>   result is an artifact of a zero-intercept restriction plus an unconventional aggregation), and
>   "under ridge shrinkage" is imprecise — their striking case is *ridgeless*.
> - **§3f's flagged EnbPI contradiction cannot be assessed** — the "2026 benchmark preprint" is
>   uncited. The conclusion (act on ACI) does not depend on it.
> - **The Qlib bar is not like-for-like.** Qlib reaches Rank IC 0.047–0.050 at Rank ICIR 0.39–0.40;
>   this model claims 0.092–0.183 at ICIR 0.34–1.45. Higher IC at comparable-or-lower ICIR is a
>   different sampling regime, not a better model. Use `naive_rank_ic` as the bar, not 0.05.
>
> ### §6's operational defects
>
> 1. **Resolved.** Run `31337078991` succeeded 2026-08-09 21:32 UTC. But **"Backtest Accuracy is
>    `skipped` behind it, so no labels are maturing" is wrong** — Backtest has an independent cron
>    and went green on 08-08 via `schedule`, writing that day's `prediction_accuracy` row. Only the
>    `workflow_run`-chained instances skipped. The weaker claim holds: no `forecast_outcomes` were
>    evaluated on 2026-08-08. **New risk:** `MODEL_ARTIFACT_VERSION` is now **6** and is *not* on
>    `origin/main`; merging without a `mode=full` dispatch re-breaks predict-only with `5 != 6`.
> 2. **Resolved, and the count was 22, not 17** (the enumeration itself sums to 19). Fixed by a
>    **manual push**, not CI — and **no workflow generates `item-metadata-bymykel.parquet`**, so it
>    persists only via `git add -A` over a fresh checkout.
> 3. **"The one live stale number in `architecture/`" is wrong — there are at least eight**, listed
>    under O2 in the roadmap. ⚠️ And note there are **two different `+3.50pp @30d` results**:
>    `model.md:210`/`:533` both cite the `TRAIN_MIN_MEDIAN_PRICE` one (re-derived null), while §5e's
>    `ab_test_feature_contribution` `+3.5pp` — the one founding the allowlist — has **never been
>    re-derived** and is not in `architecture/` at all. §5e is right; §6.3 is about the other one.
> 4. **"Compaction was never dispatched" is wrong** — it largely landed (210.7 MB → 122.6 MB, the
>    `median_price`/`min_price`/`max_price` strip done, 5 of 6 snapshot files gone). Only
>    `snapshots-2026-08.parquet` is stranded, and it has stopped growing.
> 5. **Not claimed here but found while checking:** `ops/collection_runs.parquet` has not been
>    written by CI since **2026-07-21**. That is the aggregator's own evidence-of-collection channel.
>
> ### One thing this review's own "what it does not establish" list got right, and one it missed
>
> Four of its five open items are now closed (the Steam route, the 07-11 detector question, the CI
> wall-clock, the allowlist reorder). The CI wall-clock **now exists and the estimate was low**:
> `TOTAL training: 1884.2s` (31.4 min), job 35m24s — a **2.16×** ratio to the 872s local cold, not
> the assumed 1.4–1.5×. The conclusion (over the 30-minute cap) gets stronger.
>
> What it missed: **§4a's 38,391 and the sibling doc's 38,413 are the same filter one day apart**
> (08-07 vs 08-08), not a discrepancy — but neither doc dates its cut.

---

## Summary

The model is not underperforming for its asset class. Three things are actually wrong, and they
are not the things the roadmap has been aimed at.

1. ~~**The label may be a quoting artifact.** The composition test that decides this is
   underpowered and unresolved, and it gates every accuracy item on the roadmap.~~
   **REFUTED the same day — see the corrections banner. Composition control does not move the
   signal (+0.1027 stable vs +0.1023 unconditional at 3d), so nothing is gated on this.**
2. **The offline cost model was wrong about where the time goes.** CV folds are *uncapped*;
   nine folds per horizon sum to **4.2× the whole training frame**. That single fact explains the
   439.3s conformal-CV phase completely, and capping it is worth −22% with nothing structural lost.
   Meanwhile every micro-lever `model-optimization.md` recommends measured **dead**.
3. **The hyperparameter selector still optimises a criterion the project discarded.** Optuna
   scores trials on early-stopped validation pinball loss at 100–200 rounds. `FIXED_BOOST_ROUNDS`
   removed early stopping from training and CV on 2026-08-08 and never touched the selector.

Two further facts reframe the accuracy work:

- **A tuned LightGBM on daily cross-sectional equities gets Rank IC ≈ 0.05** (Qlib's published
  CSI300 benchmarks). This model's out-of-fold rank IC is **0.092–0.183**. Gu, Kelly & Xiu
  (*RFS* 33(5), 2020) report the best ML methods reaching monthly stock-level out-of-sample
  **R² of 0.33–0.40%**. A 50–53% DA is the normal regime, not a failure.
- **"Loses to `−return_1d`" is not an embarrassment.** Short-term reversal is a documented
  anomaly, and Nagel (*RFS* 2012) shows it is compensation for liquidity provision. Da, Liu &
  Schaumburg (*Management Science* 60(3), 2014) decompose it and find **only the residual
  component is significant**, with a strategy isolating it earning ~3× the standard one. The
  productive target is *residual* reversal, not reversal.

---

## 1. Label integrity — the gate

### 1a. The July cutover — already handled; one open question

⚠️ **A first draft of this section claimed 2026-07-09/10/11 was a new, undocumented consensus
break. That was wrong and is withdrawn.** The 2026-07-09/10 cutover is documented
(`docs/changelog/2026-08-06-validation-window-widening.md:14`), is named in
`_collection_shift_dates`' own docstring (`models/forecaster.py:2977-2981`) and in
`docs/architecture/model.md:259`, is caught by the detector — which fires 12 times in 4,735 days,
7 of them in 2026 — and the span rule already voids labels across it
(`docs/changelog/2026-08-06-paired-retrain-measures-no-gain.md:101`).

The magnitudes measured in this review differ from the published ones only in statistic:

| Date | This review: median day-over-day, 2,372 items | Published: mean market return |
|---|---:|---:|
| 2026-07-09 | +26.46% | +17.4% |
| 2026-07-10 | −20.65% | −17.8% |
| 2026-07-11 | +14.26% | not published |

Same event — the `aggregator_steam_17mafo` → 9-venue-panel handover, `17mafo`'s last day being
2026-07-10 and the panel's first 2026-07-11 — read on the median rather than the mean.

**What is genuinely open is 2026-07-11.** It is not named in any changelog and the review measured
it at +14.26%. The universe jumped from `17mafo`'s 27,194 items to a nine-venue panel reaching
39,366, which is well past `COLLECTION_SHIFT_FRACTION = 0.20`, so it is *probably* already caught
— but "probably" is not a measurement, and the detector's fired-date list has never been printed
in a doc. **Verify before building anything.** If it fires, there is nothing to do here.

### 1b. Two 2025 regime breaks the archive spans and nothing marks

Found in web research, corroborated by third-party market trackers:

| Date | Event | Measured effect |
|---|---|---|
| **2025-07-15** | Valve Trade Protection (7-day reversible trade lock, Premier S3) | market cap ≈ **−25% in one day**; third-party listings −12%; 3–4% of trades reversed |
| **2025-10-23** | Trade-up extended to 5 Covert → knife/gloves, unannounced | cap **$609M → $337M (−45%)** in hours; knives −20 to −60%; **Coverts +5–20×** |

The second matches the repo's own recorded Oct-2025 natural experiment. The first is a
**liquidity-regime change**, which matters independently: any volume or listing-count feature
crossing it is measuring two different markets.

### 1c. ~~The composition test that gates everything~~ — **REFUTED 2026-08-09 (`f833882`)**

> The table below is reproduced from `2026-08-08-model-review.md` §5, where these four rows are
> **struck through as refuted in place**. They are a 2013-2025 → 2026 regime contrast, not a
> composition contrast: the partition read a NULL `source` as never equal to itself, and every
> pre-2026 row has `source IS NULL`. Corrected measurement:
> `docs/research/2026-08-09-composition-stability.md`. **Nothing downstream is gated on this.**

`docs/research/2026-08-08-model-review.md` §5 measured the reversal signal against source
composition, voted daily series, ≥$1, 2024-01-01 onward:

| Subset | Dates | rank IC | t |
|---|---:|---:|---:|
| All rows | 932 | +0.1676 | 43.6 |
| Composition **stable** across t−1…t+3 | 188 | **+0.1006** | 14.1 |
| Composition changed | 775 | +0.1804 | 43.1 |
| Stable & **three agreeing sources** | 28 | **+0.0044** | **0.1** |

The cleanest subset available shows **no reversal at all**, on 28 dates. That is underpowered,
not refuting — but it is the measurement that decides whether any feature or objective work can
pay off, because if the label is a quoting artifact it is not a tradeable return.

### 1d. Live basis defects, sized

| Defect | Status | Size |
|---|---|---|
| MA feeds voting against asks (`aggregator_steam_7d/30d/90d`), step **6c** | measured, **unfixed** | costs 670 item-days of 3,093,793 to exclude, but moves the voted median on **17.13%** of 2026 ≥$1 item-days, median **−7.16%**, flipping **5.75%** of return directions — half the bid's magnitude, same character |
| `STEAM_FEE_MULTIPLIER = 1.1607` synthetic, step **5c** | measured, **unfixed** | flat at 1.1606–1.1607 across four orders of magnitude, IQR 0.0002, on 63,767 pairs, where the cent-ceiling schedule must swing ~1.67 at $0.03 to ~1.15 at $50; **91.61% of pairs are identical after dividing**. Rows below ~$0.50 are ~5% too high |
| `aggregator_steam_17mafo` unaudited and **dead 28 days** | **live, unflagged** | 2,161,250 rows / 27,194 items; the **only** feed 2026-04-16 → 2026-07-10; raw JSON confirms it carries `last_24h/7d/30d/90d`, so it inherits the same fallback-to-trailing-mean basis error for a three-month window |

**The structural consequence, which subsumes all three:** `aggregator_sync` and
`aggregator_steam_17mafo` are `last_24h` *falling back* to the 7d/30d/90d windows, and that
fallback fires on exactly the illiquid items. **There is no point-in-time Steam price in this
archive at all.** That is a ceiling on any return-based label, independent of features.

---

## 2. Training cost

### 2a. Where 872s goes

Published (`docs/changelog/2026-08-09-shipped-retrain-cost-measured.md`), CI-exact cold retrain,
`CV_DIAGNOSTIC_CLASSIFIER=0`, uncontended 10-core Mac:

| Phase | s | share |
|---|---:|---:|
| Conformal CV (33 folds × 4 horizons) | 439.3 | **50.4%** |
| q50 ensemble | 158.4 | 18.2% |
| Regime models | 95.4 | 10.9% |
| Direction classifier | 92.3 | 10.6% |
| Remainder | 52.0 | 6.0% |
| Optuna | 35.0 | 4.0% |

Measured in this review, the legs the "Remainder" row lumps together:

| Leg | s |
|---|---:|
| DuckDB archive read, 1460d (16.5M raw → 7.57M after the backfilled join) | 21.1 |
| `_apply_multi_source_voting` (`:1168`) → 6,080,631 voted rows | 27.2 |
| `_filter_dead_items` + `_filter_by_median_price` → 986,065 rows / 918 items | 1.0 |
| `engineer_features` (`:2891`), 9 blocks | **17.5** |
| `_prune_features` (`:2392`), 123×123 correlation | **25.2** |

**These do not reconcile with the published 52.0s remainder** — feature engineering plus the
prune alone are 42.7s, before `prepare_targets` ×4, two splits, the permutation test and
`save_models`. Treat 872s as the only solid number; the changelog itself notes per-phase timings
swing ±25% between clean runs.

### 2b. CV folds are uncapped — the largest untaken lever

`max_rows` is applied only on the production split (`_build_production_split:2756-2761`).
`_cv_evaluate_horizon:5693` does `train_df = tdf[tdf["date"].isin(train_dates)]` — the whole
expanding window, every fold. Measured fold geometry on the real frame:

| h | embargo | folds | Σ fold-train rows | ×frame | CV rounds | prod rounds |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 16 | 9 | 4,213,562 | 4.27× | 200 | 300 |
| 7 | 20 | 9 | 4,179,057 | 4.24× | 500 | 750 |
| 14 | 27 | 9 | 4,135,665 | 4.19× | 100 | 150 |
| 30 | 43 | 9 | 4,036,402 | 4.09× | 750 | 1000 |

A rows×rounds cost model predicts CV / production-q50 = **2.94×**; the measured ratio is
439.3 / 158.4 = **2.77×**. The phase is fully explained by expanding windows summing to 4.2× the
frame at 2/3 of production's rounds. Nothing else is going on.

Capping per-fold training rows the way the production split already does — ✅ **shipped as
`6b6fc81` at a 300k cap.** ⚠️ **The saving below did not materialise:** the cap bound on every
fold, and the conformal-CV phase measured 439.3s → **434.8s**. Either this cost model is wrong or
the two runs are not comparable (different hardware and round counts). Establish a like-for-like
control before reusing this table.

| Cap | Saving | Share of 872s |
|---|---:|---:|
| 600k | −56.9s | 6.5% |
| 400k | −137.6s | 15.8% |
| **300k** | **−194.6s** | **22.3%** |
| 200k | −263.6s | 30.2% |
| 100k | −345.0s | 39.6% |

**Nothing structural breaks.** Fold count, validation rows, OOF record count, distinct forecast
dates, rank IC and the PT statistic are all untouched — only the model each fold fits sees less
data. The OOF residuals then describe a slightly weaker model than the served one, so `q_hat` is
biased **wide**: over-coverage, the safe direction. This was found once before and never acted on
(`docs/changelog/2026-08-07-per-item-row-sampling.md`: *"the sampler thins the 36-second part of
a 541-second run"*).

Strictly better than widening `CV_STEP_DAYS`, the doc's lever 1, which buys the same seconds by
destroying folds that are also the rank-IC and PT sample.

### 2c. `model-optimization.md`'s lever table is refuted

Measured on the production frame, 3 reps each, ms/round against a 25.5 baseline:

| Lever | Doc claim | Measured | Verdict |
|---|---|---:|---|
| `max_bin` 63 → 31 | "~10–15%" (`:163`) | 26.3 | **dead** |
| `num_leaves` 47 → 31 | "~15–20%" (`:164`) | 26.2 | **dead** |
| `min_data_in_leaf` 15 → 100 | "~5%" (`:166`) | 27.1 | **dead** |
| `feature_fraction` 0.7 → 0.4 | — | 28.9 | **slower** |
| `force_row_wise` / `force_col_wise` | not in the doc | 38.6 vs 44.5, single-shot | inside the noise band |
| Row bagging removal | — | 14.2 vs 14.3 | regularization choice, not a cost lever |

LightGBM is memory-bandwidth bound at this frame shape — 1 thread vs all measured **1.5×**,
matching the 1.55× already recorded. `n_jobs = -1` is set everywhere (`:2821`, `:3987`, `:4739`,
`:5736`); float32 downcast is already done (`:3493-3495`) and survives the median-impute path.
There is nothing left in this family.

`feature_pre_filter: False` is **deliberate** at all five Dataset sites (`:2802`, `:3862`,
`:4723`, `:5717`) — it is what allows one binned Dataset to be reused across parameter sets.
Do not flip it.

### 2d. Feature engineering: the prune is the waste, not the engineering

Measured block by block on the production frame:

| Block | s | cols | group | survives allowlist? |
|---|---:|---:|---|---|
| `_compute_price_features` | **9.06** | 93 | price_technicals | **yes** |
| `_add_cross_sectional_features` | 2.15 | 18 | cross_sectional | no |
| `_add_supply_depth_features` | 2.09 | 5 | supply_depth | no |
| `_add_item_identity_features` | 1.69 | 13 | item_identity | no |
| `_add_item_metadata_features` | 1.15 | 7 | item_metadata | no |
| `_add_supply_side_features` | 0.79 | 15 | item_identity | no |
| `_add_temporal_features` | 0.39 | 12 | temporal | no |
| `_add_event_features` | 0.01 | 20 | events | no |
| `_add_social_features` | 0.00 | 5 | social | no |
| **total** | **17.5** | 192 | | **8.5s (48%) discarded** |

Then `_select_feature_cols` → 123 numeric candidates → `_prune_features` **25.2s** → 118 →
allowlist → **33**.

`df[self.feature_cols].corr()` (`:2402`) is O(rows × p²) single-threaded pandas. On 123 columns
it costs **25.2s**; on the 33 allowlisted columns, **1.65s**. On this frame the prune drops
**zero** price_technicals features (33 in, 33 out), so moving the allowlist ahead of the prune is
output-identical *here* — but not provably so in general, because `_prune_features` keeps the
lower-indexed member of a >0.95 pair and a price feature could be dropped in favour of a
non-allowlisted partner. Gate it and log both counts.

**Total recoverable ≈ 32s (3.7%)** — 4–5× what `model-optimization.md:162` sizes lever 2 at,
because that estimate assumed a 7s feature-engineering phase and the measurement is 17.5s.

**Guard against deletion:** seven harnesses build their own frame and call
`_apply_feature_allowlist` on it, so they need the full 123-column frame —
`ab_test_training_breadth.py:341`, `ab_test_train_universe.py:348`,
`ab_test_item_metadata.py:306`, `ab_test_csfloat_basis.py:340`,
`ab_test_interval_sampling.py:533`, `ab_test_q50_sampling.py:412`,
`ab_test_direction_labels.py:246`. `HORIZON_EXCLUDED_GROUPS` (`:339`) and
`_validate_feature_groups` (`:2427`) are also written in terms of groups the allowlist has
already removed. This must be a flag on `engineer_features`, never a deletion.

### 2e. The voted cache cannot hit in CI

Two independent reasons:

1. `self.cache_dir` is `backend/data` (`:641`), gitignored, and `price-forecast.yml` caches only
   `backend/models/saved_models` (`:91-96`, `:216-219`). The directory does not survive a job.
2. Even if it did, `_archive_fingerprint()` (`:3510`) hashes `name:st_size:st_mtime_ns` of every
   `prices-*.parquet`, and CI checks the archive out fresh every run — so **every file's mtime is
   new on every run** and the key changes unconditionally.

Cost: ~48s per run (21.1s read + 27.2s vote), every day, not just Monday. A content-derived key
(row count + max day per file, or the archive commit SHA) fixes it.

### 2f. Cold vs warm, local vs CI

`_warm_retrain` (`:3889`) suppresses exactly three things: Optuna (35.0s), regime models
(`:4031`, 95.4s), and the feature-group permutation test (`:4329`). So **cold ≈ 872s, warm ≈
730s** at the shipped config. The documented **250.1s / 176.7s** are the 100K-row / 99-item
config and are superseded.

CI's Monday run is **always cold** — the model-cache restore is `if: mode == 'predict-only'`
(`price-forecast.yml:88`). No CI wall-clock measurement of the shipped config exists anywhere in
the repo; every published number is the 10-core Mac. At 1.5× thread scaling a 2-vCPU runner is
roughly 1.4–1.5× on every fit phase, putting the real CI Monday retrain at **≈20–25 min** plus
the 48s uncached fetch — inside the 180-minute job timeout but **over the project's own
30-minute run cap**.

The local/research default has `CV_DIAGNOSTIC_CLASSIFIER` **on** (`:5934`), which is
**1804s / 30.1 min** to populate a `meta.json` field nothing reads.

### 2g. Rows ↔ items ↔ seconds

Every measured triple in the repo:

| items | rows | s | config | source |
|---:|---:|---:|---|---|
| 99 / 5,377 | 116,111 | 104.6 | 100K, warm | `model-optimization.md:111` |
| 372 | 410,780 | 232.3 | 400K | `2026-08-04-minimal-model-results.md` |
| 646 | 707,913 | 468.7 | 700K | `model-optimization.md:112` |
| 926 / 5,542 | 993,464 | 538 | $1 floor + 1.2M, early stopping | `2026-08-07-training-item-universe.md:151` |
| 926 | 984,615 | 487 | ≥$1, early stopping, CI-equiv | `2026-08-08-model-review.md:62` |
| **918** | **986,065** | **872** | ≥$1, fixed rounds, diagnostic off | `2026-08-09` changelog:28 |
| 918 | 986,065 | 1804 | same, diagnostic on | `2026-08-09` changelog:29 |

Within the one comparable series (2026-08-05 budget sweep, constant fold count) the elasticity of
seconds to rows is **0.79–0.83** — roughly linear with a ~30s floor, `t ≈ 33 + 0.62 ms × rows`.

**Two confounds that make cross-row comparisons invalid.** 538s → 872s is at *constant* rows
(993k vs 986k): the +62% is **entirely the round count**, not breadth. And the share CV carries
*falls* as the budget rises (75.8% at 100k → 61.3% at 700k), because the production q50 fit
degrades faster than linearly (3d q50 2.5s → 62.4s, 25× for 6.1× the rows).

---

## 3. Accuracy

### 3a. Optuna optimises a discarded criterion

`_optuna_search_params` (`:2840-2853`) scores every trial on
`model.best_score["valid_0"]["quantile"]` under `lgb.early_stopping(20)` at
`num_boost_round = 100 if horizon == 7 else 200`. That is early-stopped validation pinball loss
on the same thin trailing window the project refuted when it shipped `FIXED_BOOST_ROUNDS` on
2026-08-08.

The `FIXED_BOOST_ROUNDS` comment at `:551-560` records the conflict directly: at 14d and 30d the
validation-loss optimum is **25 rounds** while rank IC peaks at **500–750**. The two are
anti-correlated. So the tuned parameters in `meta.json` for the two noisiest horizons were
selected under a criterion the project has since discarded, and the fixed-rounds fix did not
touch the selector.

~~Cost to re-tune against within-date rank IC at fixed rounds: **35s of an 872s run.** Highest
accuracy-per-second lever in the system.~~ **Wrong by 11×. Shipped as `8be48c5` and measured at
392.3s** — removing early stopping means every trial trains the full round budget instead of
stopping near 25 rounds, making Optuna the largest phase of a cold retrain (27.5%). The
*diagnosis* stands and the change was made; the price was not 35s. Whether the newly selected
params forecast better is still unmeasured.


### 3b. Scale-free is not cross-sectionally normalised

Gu, Kelly & Xiu map every predictor to `2 * ((rank − min)/(max − min) − 0.5)` **within each
period**. This repo's features are scale-free *per item* (`tests/test_scale_free_features.py`
pins that property), which is a different thing: scale-free still leaves every feature loaded on
the common market factor on every date. "DA is dominated by the market factor" and "demeaning by
the market factor drops accuracy below a constant call" are both the predicted symptoms.

Cost: one `groupby("date").rank(pct=True)`. Side benefit: a missing characteristic maps to the
cross-sectional median rather than to `feature_medians`.

This is noted inside step 10 of `2026-08-07-next-steps.md` as a companion to lambdarank. It is
worth testing **on its own first** — it is the cheaper half and isolates cleanly.

### 3c. The objective does not match the product

`docs/research/2026-08-08-model-review.md` §4 measured the model losing to ranking by
`−return_1d` at all four horizons (0.183 vs 0.197 at 3d … 0.092 vs 0.102 at 30d). Quantile loss
on raw % returns targets a conditional median per row; nothing in training optimises within-date
ordering, which is what the dashboard sells.

Evidence for the fix (LambdaRankIC, arXiv:2605.00501 — **preprint, unreviewed**; 2.7M
stock-months, XGBoost not LightGBM):

| Method | Rank IC | ICIR |
|---|---:|---:|
| OLS | 0.047 | 0.666 |
| XGBoost regression | 0.042 | 0.456 |
| Pairwise LTR | **0.083** | 0.718 |
| NDCG LTR | 0.086 | 0.812 |
| LambdaRankIC (their method) | 0.115 | 1.031 |

The **regression → LTR jump is the larger one and the replicable-looking one**, and it is
available from a stock LightGBM objective with no custom gradient code. Note also that plain
regression (0.042) *underperformed OLS* (0.047) — which is this project's experience exactly.

Setup details that will bite:
- **Query groups = dates.** Rows must be contiguous and sorted by group.
- **`lambdarank_truncation_level` defaults to 30**, so over a ~900-item cross-section the
  objective ignores everything below rank 30. Must be raised substantially; cost is superlinear.
- **`label_gain` is exponential** (`(1<<i)-1`). Bucket forward returns into 5–10 per-date
  quantiles and override with a linear gain.
- A ranker emits an **uncalibrated score, not a return**. The conformal band and the q50 serving
  path both need a level, so this **adds** a model rather than removing one.

### 3d. Residual reversal, not reversal

Nagel (*RFS* 2012) shows reversal returns proxy the return to liquidity provision. Da, Liu &
Schaumburg (*Management Science* 2014) decompose reversal into across-industry momentum,
within-industry expected-return variation, under-reaction to cash-flow news, and a residual — and
find **only the residual is significant**, with a strategy isolating it earning ~3× the standard
reversal strategy's risk-adjusted return.

For CS2 that maps to demeaning `return_1d` within weapon/tier/price-bucket groups. It is a
distinct experiment from the market-relative-label one already refuted
(`2026-08-06-market-relative-labels-refuted.md`): that changed the **label** and left a pointwise
loss fighting a noisy residual; this changes a **feature**.

### 3e. More rows will not help

Tabular scaling laws (arXiv:2607.21866 — **preprint**, 18 datasets, 11,536 runs, 1,648 fitted
curves) give boosting a median exponent **b ≈ 0.48** in `error(N) = a·N^(−b) + c`, with tree
ensembles' irreducible floor **c ≈ 0.10–0.21**, orders of magnitude larger than for LMs. Doubling
rows cuts the *reducible* component ~29%; in return forecasting the reducible component is on the
order of 0.4% R² against a c that is essentially the whole variance. The 1.2M budget is well past
the knee.

**Counterweight, and it is a good one.** Kelly, Malamud & Zhou, *"The Virtue of Complexity in
Return Prediction"* (*JF* 79(1), 2024, peer-reviewed) proves and documents that simple models
**severely understate** return predictability, holding "even in extremely data-scarce
environments." It concerns *parameter* complexity under ridge shrinkage in a random-features
setting, not training rows, and is not established for GBDTs — but it is a high-quality argument
against "the model is too big" as a diagnosis. The two together say the constraint is feature
*information*, not model or data size.

### 3f. Calibration needs one fit, not 33

The CV block does double duty — diagnostic and conformal calibration set. Split conformal needs
**one** fit (Papadopoulos et al. 2002). The standard variance objection is calibrated to n in the
hundreds; this frame has ~1M item-days.

| Method | Fits | Guarantee |
|---|---:|---|
| Split conformal | **1** | ≥ 1−α |
| CV+ | K | ≥ 1−2α (Barber, Candès, Ramdas & Tibshirani, *Ann. Statist.* 49(1), 2021) |
| Cross-conformal | K | ≥ 1−2α−2(1−α)(K−1)/(n+K) (Vovk, *AMAI* 2015) |
| Jackknife+ | n | ≥ 1−2α |
| Jackknife+-after-bootstrap | **0 extra** | ≥ 1−2α (Kim, Xu & Barber, NeurIPS 2020) |

**J+aB does not apply here.** It requires a bagged ensemble of models; `bagging_fraction`
resamples per *iteration*, not per model, so a single booster has no out-of-bag structure. Using
it would mean training B boosters, defeating the purpose.

**ACI is the right drift answer.** Gibbs & Candès (2021) adjust α online from realised coverage
and need **no refits at all**. Zaffran et al. (*ICML* 2022, peer-reviewed) find ACI gives the
smallest intervals at correct coverage while EnbPI attains coverage by over-covering; a 2026
benchmark preprint reports EnbPI *under*-covering. **Contradiction flagged** — ACI wins in both,
so it is the one to act on. This matters here specifically because the archive has dated
exchangeability breaks (§1), which is exactly what split conformal assumes away.

Tracked as **R18**, currently filed as gated on the calendar wait. The calibration-*cost*
argument for it is independent of that and stronger.

Reusing boosters across folds is **impossible** — every fold's calibration rows must be unseen by
the model producing them (`:4096-4113`). Subsampling the residual set saves ~0: `q_hat` is a
quantile of ~190k OOF residuals per horizon and computing it is free. The cost is the fits.

---

## 4. Data

**The allowlist is the binding constraint, not data availability.**
`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` (`:350`) means the model consumes exactly one
stream — the voted consensus price series. Supply depth (5 feeds, 31,246 items), Skinport sale
volume (35,296 items), the event calendar (4,967 rows, 13 years), FX history (18,251 rows) and
the ByMykel bundle are all collected and all discarded before training. Adding sources changes
nothing until that changes.

### 4a. Universe and history depth

| Quantity | Value |
|---|---:|
| Distinct slugs, raw | 41,775 |
| After `archive_universe_sql_filter()` | **38,391** |
| Distinct item-days (filtered) | **13,579,293** |
| `is_backfilled` | 5,536 |
| Items forecast (latest run) | 8,691 |
| **Items trained on (≥$1)** | **926 / 993,464 item-days** |

History depth, by calendar span:

| Bucket | Items | Share |
|---|---:|---:|
| > 2 years | 3,928 | 10.2% |
| 1–2 years | 1,223 | 3.2% |
| 6–12 months | 385 | 1.0% |
| **< 6 months** | **32,855** | **85.6%** |

Distinct-date percentiles: p10 = 21, **p50 = p75 = 131**, p90 = 853. The p50=p75 plateau is one
event — 32,855 items were first observed in 2026. 3,928 + 1,223 + 385 = 5,536 is exactly the
`is_backfilled` set. **There is no middle tier**, which is why depth-only sources are declined
and breadth sources are the ones with value.

### 4b. The multi-source era is 24 days deep

| source | rows | span | days present |
|---|---:|---|---:|
| *(NULL, pre-2026 backfill)* | 9,417,947 | 2013-08-14 → 2025-12-31 | 4,523 |
| `aggregator_steam_17mafo` | 2,161,250 | 2026-04-16 → **2026-07-10** | 85 |
| `aggregator_buff163` | 1,520,462 | 2026-03-22 → 08-07 | 49 |
| `aggregator_youpin` | 1,307,213 | 2026-03-22 → 08-07 | 49 |
| `aggregator_csfloat` | 1,204,196 | 2026-03-22 → 08-07 | 49 |
| `aggregator_sync` | 1,096,317 | 2026-01-01 → 08-07 | 113 of 219 |
| `aggregator_csgotrader` | 744,394 | 2026-07-11 → 08-07 | 24 |
| `aggregator_steam_90d/30d/7d` | 713k/649k/550k | 2026-07-11 → 08-07 | 24 |
| `aggregator_skinport` | 532,173 | 2026-07-11 → 08-07 | 24 |
| `aggregator_csmoney` | 495,379 | 2026-07-11 → 08-07 | 24 |
| `aggregator_buff163_buy` (bid, excluded) | ~570k | 2026-07-11 → 08-07 | 24 |

**Two windows are effectively single-feed**: 2026-01/02 (`sync` alone) and 2026-04-16 → 07-10
(`17mafo` alone). A "consensus" over either is one quote.

### 4c. Three findings that reverse standing conclusions

**Steam rebuilt the Market listing page.** The old `var line1=[[...]]` inline array is gone.
The SSR/React page now ships a dehydrated react-query cache with `{time, price_median,
purchases}` per `market_hash_name` — `purchases` being the **daily sales count** — ranging
**2014-02-21 → 2026-08-09**, daily historically and hourly for the trailing ~30 days. One 5 MB
GET returned **10 distinct names / 36,630 history points**; 10 back-to-back GETs at ~1.3 req/s
all returned 200 with no throttling. The same page embeds
`rgCompactBuyOrders` / `rgCompactSellOrders` — **full bid and ask depth with quantities**.

If that reproduces from this project's egress it collapses blocker **5d** and unblocks **R11**,
**R13** and **5c** at ~0.1 requests per item (a 900-item cohort is 100–200 fetches, not 7,100).
⚠️ **Verify from the project's own IP before planning on it** — `2026-08-07-next-steps.md` §5d
records a soft-block since 2026-08-05 with no decay in three days, and the research environment
may not be representative. `/market/pricehistory/` JSON still returns **400 without a login
cookie**; that is unchanged.

**`prices.csgotrader.app/latest/steam_volume.json` no longer exists** — 301 → S3 `NoSuchKey`,
along with `steam_listing` and `waxpeer`. **This is the mechanical cause of the dead `volume`
column: it died upstream, not in the collector.** The dump moved to per-provider paths
(`latest/<provider>.json`, 46,041 items each; `buff163` still carries `starting_at` and
`highest_order`). Alive: buff163, csgotrader, skinport, csgoempire, csgotm, csmoney, bitskins,
lootfarm, swapgg, cstrade, skinwallet, csgoexo, exchange_rates.

**An undocumented dated archive exists** at `prices.csgotrader.app/YYYY/MM/DD/prices.json` —
advertised in the 301 error body. `2019/06/30` returns 9.9 MB / 13,017 items and carries a
`csgobackpack` block with `sold` counts for 24h/7d/30d/all_time. `prices_v6.json` under the same
path is continuous **~2020-12 → 2023-10-25**, 22,475 items, all 10 providers. Nothing dated
exists after Oct 2023, and late-2019 → late-2020 is missing. One-shot retroactive backfill, not a
live feed.

### 4d. New free sources worth knowing

| Source | What | Licence / limit |
|---|---|---|
| **`devynpruden/cs2-skin-price-history-2013-2026`** (Kaggle) | 290 MB Parquet, daily median/mean/min/max **plus `volume` = units sold daily**, 2013→2026, updated 2026-06-16 | **Apache 2.0** |
| `idomanteu/cs2-historical-item-prices-…` (Kaggle) | hourly **bid and ask** per venue + `hourly_volume` + `total_supply`, **2026-03-22 → 2026-04-15 only** | CC BY 4.0 |
| Skinport `/v1/sales/history` | 24h/7d/30d/**90d** completed-sale windows, 36,150 items | free, needs `Accept-Encoding: br` — **already shipped 2026-08-08, read by nothing** |
| Waxpeer `/v1/prices?game=csgo` | listing counts, no auth | free |
| `steamcharts.com/app/730/chart-data.json` | `[unix_ms, players]` from 2012-07-01 | free, undocumented |
| Wikimedia Pageviews API | daily since 2015 | free — **the correct Google Trends replacement**; pytrends was archived by its maintainers 2025-04-17 and Google's official Trends API is still application-gated alpha |
| Liquipedia MediaWiki API | esports calendar, JSON, no auth | free non-commercial — the working HLTV (403) replacement |

The `idomanteu` window opens **exactly on the 2026-03-22 consensus break**, which makes it a
validation instrument for that anomaly rather than training data.

**Cheapest unexploited item, requiring no fetch at all:** `stattrak_premium_z30`. 4,686 paired
names, median 1.43×, AK-47 2.15× vs P2000 1.10×, 13 years deep, computable from rows already
held. Five of seven demand drivers have no proxy at all; this is one of the two that do. Use the
**z-score only, never the level** — the level is a shelved dollar-scale proxy.

### 4e. The staged import on `feat/price-history-import`

`LukeX404/cs2-prices-tracker`, labelled `tracker_steam_24h`. Measured on disk in
`archive-staging/`:

| | |
|---|---:|
| Rows | **1,963,626** |
| Items | **5,153** |
| Range | 2025-02-17 → 2026-03-31 (403 days) |
| `new_gate_items` (would flip `is_backfilled`) | **4,196** |

It adds **zero new items** — its entire value is flipping the gate. Four open blockers stand
(`docs/changelog/2026-08-09-price-history-import-staged.md`): the **end seam is unmeasured**
(source stops 2026-03-31, consensus steps −0.78% median, 80% one-directional, on 04-01);
`report_promotion_gate`'s "consensus" is not production's; `_preserve_first_arrival` silently
NULLs `ingested_at` on non-`RangeIndex` frames; and there is no zero-row guard, so a corrupt
download imports silently. Upstream has **no LICENSE** — private training use only.

### 4f. Volume: what a clean test would need

The last day with `volume > 0` anywhere is **2026-04-15**, not 2026-05 — it dies with the switch
to `17mafo` as sole feed on 04-16. From 04-16 → 08-07, 8,622,895 rows carry `volume = 0`,
non-NULL, fabricated.

A clean pre-cliff window exists and is large: **2013-08-14 → 2025-12-31, 4,523 days, 9.42M
volume-bearing rows, 5,536 items** (871 at ≥$1), zero missing calendar days. Four things would
invalidate a naive run:

- `ab_test_volume_features.py:53` sets `VOLUME_LIVE_THROUGH = "2026-04-30"`, **15 days past the
  real cliff** — and its source filter means the volume-bearing part actually ends 2026-03-29, so
  ~32 days of fabricated zeros sit inside the evaluated window and 30d/60d rolling features
  straddle them.
- **Never pool across 2026-03-22.** `merge_hf_dataset.py:99` maps `ask_volume AS volume`, so
  post-break "volume" is a **listing count**, not a trade count. Median per row: pre-2026 = 117,
  `aggregator_sync` = 50, csfloat = 335, youpin = 966, buff163 = 1,046.
- **The pre-2026 series has literally zero zero-volume rows.** A no-sale day is an *absent row*.
  Any feature built on it silently conditions on a sale having occurred — a selection effect no
  repaired feed would reproduce.
- Nothing collects `prices.volume` today, and the repaired feed (Skinport sales, or Steam
  `purchases`) is a **different quantity** from what the pre-2026 data holds.

"Does volume affect **speed**" is not an experiment: re-admitting 13 of ~45 columns moves the
booster-fit phases, which are 39% of an 872s retrain, and per-phase timings swing ±25% between
runs.

---

## 5. Experiment capability

### 5a. The item-count question is answerable today

`backend/scripts/ab_test_training_breadth.py` is exactly that experiment and it is the only one of
15 harnesses whose treatment axis *is* item count. Nested arms `narrow=150 ⊆ mid=350 ⊆ wide=700`
from an 870-item universe (`N_UNIVERSE=870`, `MIN_ITEM_DAYS=180`, `MIN_MEDIAN_PRICE=1.0`), with
150 items held out of every arm's training set and all arms scored on **identical eval rows**
(`:539-543`), shared `ROW_BUDGET=200_000`, 25–26 folds, plus a `wide_unbudgeted` diagnostic that
separates "more items" from "more rows".

**Nesting is what makes it valid.** `assert set(arms["narrow"]) <= set(arms["mid"]) <=
set(arms["wide"])` (`:396`) makes the contrast "add items to the set you already had", so the
item draw is a common term and differences out. The applicable MDE is therefore **0.69–2.15pp**,
not the 2.21–3.69pp item-level floor (which applies to arms with *different* draws) and not the
±3–4pp `prod_pool_b` placebo floor.

Cost: the 2026-08-08 four-horizon × four-arm re-run was **5m53s**; one horizon × two arms is well
under 2 minutes. "100 vs 200" needs a source edit at `:105-107` — no CLI flag exists.

**Prior:** the 2026-08-08 re-run found breadth positive at **14d only** (+2.33pp [+0.44, +4.74])
and null at 30d, saturating by 350 items.

**Pass `--fixed-rounds`.** This harness is the only one of 15 with that escape hatch; without it
both arms early-stop against the window they score.

### 5b. No script averages over item-draw seeds

Verified across all 15 `ab_test_*.py` plus `compute_mde.py`. `compute_mde.py:52-57` and
`ab_test_frozen_runs.py:77` vary the **LightGBM** seed, not the item draw.
`ab_test_train_universe.py:129-131` runs two *disjoint* 99-item draws as a **placebo to size the
floor**, not as an average. `2026-08-07-training-item-universe.md` explicitly declined
multi-seed averaging: *"Averaging 8 draws would cost 8× the retrain to buy a number the price
floor makes unnecessary."* Nesting is the design that solves this, and it is already implemented
in one harness.

### 5c. Offline CV now computes rank IC and PT — ~~uncommitted~~ **committed**

> **Stale within hours.** `git show HEAD:backend/models/forecaster.py | grep -c rank_ic` now returns
> **23**. Every element enumerated below exists on `HEAD`; only the "uncommitted" framing was true.
> Line numbers have drifted ~+120 to +215.

~~`git show HEAD:backend/models/forecaster.py | grep -c rank_ic` → **0**.~~ In the working tree:
`pesaran_timmermann` at `:4251` stored as `cv_results["pt"]` (`:4314`); `mean_rank_ic` /
`mean_naive_rank_ic` / `rank_ic_edge_vs_naive` at `:4227-4245`; per-fold `rank_ic` /
`naive_rank_ic` at `:5875-5880` via `_within_date_rank_ic` (`:5958`); `constant_call_accuracy` /
`realised_down_rate` per fold at `:5896-5901`; three warnings when the model loses to the
constant call, to `−return_1d`, or when PT ≠ `skill` (`:4277-4292`).

`_within_date_rank_ic` is wired into **no** `ab_test_*` harness.

### 5d. Fifteen harnesses, no shared infrastructure

8,617 lines across 15 standalone scripts. `_build_frame_uncached` ×6, `run_evaluation` ×7,
`_frame_fingerprint` ×5, `build_frame` ×6, `print_summary` ×4, `_stratified_sample` ×3. The only
genuinely shared code is `backtest/paired_mde.py` and `backtest/walkforward_records.py`, both
extracted **2026-08-08** — after every stored result.

Every one of four documented archive defects had to be fixed 13–15 times independently, and three
are still not fixed everywhere: **9 harnesses use `phase_collapsed_sql_filter()` where invariant
2 requires `archive_universe_sql_filter()`**, and **15 of 15 still early-stop against the window
they score** — a defect production fixed on 2026-08-08.

### 5e. Stored A/B verdicts that are not citable

Four invalidation waves: fold clustering (2026-08-07, CIs ~5× too narrow), embargo + universe +
significance (2026-08-08 — 6 harnesses had no purge of any kind, none of 13 applied `BID_SOURCES`
or phase-collapse, 10 of 13 had no significance test at all), archive migration (8 harnesses'
universes silently went to **0 items** via a non-NULL-safe `source =` filter), and early stopping
(2026-08-09).

The two that matter most because live decisions rest on them:

| Result | Rests on it |
|---|---|
| `ab_test_price_primitives` **−1.10pp pooled** | the **1.15 / 2.76 / 3.13 / 7.13pp MDE table** in `accuracy-opportunities.md`, which `model-optimization.md` uses as the yardstick for every speed lever and `model.md:538` cites as "the harness noise floor". No ≥$1 filter, no embargo, no universe filter, early-stopped |
| `ab_test_feature_contribution` **cross-sectional +3.5pp @30d if removed** | **`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`** — the decision to discard 7 of 8 engineered groups. 100 items, un-embargoed, never re-derived |

Also: the three harnesses re-run on 2026-08-08 produced results that live **only in a session
scratchpad**; the changelog numbers are reproducible only by a 16m28s re-run, not by reading a
file.

---

## 6. Live operational defects

1. **Price Forecast has failed every run since 2026-08-08 07:46 UTC** — three consecutive
   (07:46, 08:33, 23:46); last success 2026-08-07 23:34. Cause: `saved model artifact version 3
   != expected 5` — the Actions booster cache predates the minimal model and `load_models()`
   refuses it. `item_forecasts` is frozen at forecast_date **2026-08-07**, and Backtest Accuracy
   is `skipped` behind it, so **no labels are maturing**. A `mode=full` dispatch clears it.
2. **The canonical archive is missing 17 files the local copy has** — `item-metadata.parquet`,
   `item-metadata-bymykel.parquet`, `exchange-rates-history.parquet`, 13 `player-counts-*.parquet`,
   and 3 of 7 `ops/` files. Practical consequence: **the ByMykel bundle cannot be enabled in CI
   at all**, because the file it reads is not published. The canonical repo also still holds
   `snapshots-2026-08.parquet` (11.6 MB) — compaction was never dispatched in prod.
3. **`docs/architecture/model.md:210` and `:533` still cite the retired +3.50pp @30d result.**
   It re-derives to +1.642pp [−0.809, +4.505], null
   (`2026-08-08-per-fold-price-filter-rederived.md`). This is the one live stale number in
   `architecture/`.
4. **Four archive days are still missing**: 2026-07-27, 07-30, 08-02, 08-03. Midnight-UTC cron
   drift, not collector failure. 2013-08-14 → 2026-07-26 is otherwise unbroken.

---

## What this review does not establish

- **Whether the Steam listing-page route works from this project's egress.** §4c was verified
  from the research environment only. `2026-08-07-next-steps.md` §5d records a three-day
  soft-block from this machine. Re-verify before planning.
- **Whether `_collection_shift_dates` fires on 2026-07-11.** Argued from the universe-size jump,
  not measured. The detector's fired-date list has never been printed in a doc, which is why a
  handled defect was briefly re-reported as a new one in this review.
- **Whether the 2026-03-22 break damages labels as much as its magnitude implies.** The
  day-over-day medians are measured; the label-side impact is not.
- **A CI wall-clock number for the shipped config.** The ≈20–25 min figure is the 10-core Mac
  scaled by a measured 1.5× thread ratio, not a measurement.
- **That the allowlist-before-prune reordering is output-identical in general.** It is on this
  frame; the ordering rule in `_prune_features` means it need not be.
