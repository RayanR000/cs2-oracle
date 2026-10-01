# Deep model review — ingestion, model, calibration, and whether there is an edge

> **Status (as of 2026-08-21): the §12 ranked plan is the LIVE plan, and items 1–12 have LANDED.**
> Progress, so the plan is not re-run from scratch:
>
> - **Unblock (1–3):** ✅ all done — migration `0023` for `is_trainable` (`d526ecd`), the CI
>   schema-drift guard (`3f5e811`), and the `historical_fallback:` universe filter (`0a357da`).
>   Production is no longer broken; the nightly chain is green.
> - **Free wins (4–7):** ✅ all done — sample-weight clip floor (`852cbe8`),
>   `DIRECTION_UPWEIGHT = 1.0` (`a792b3d`), and the signed two-quantile conformal band replacing
>   serving recentring (`d0dcde2`, live from `SIGNED_BAND_SERVING_START = 2026-08-19`).
> - **Measurement integrity (8–11):** the replayed 2025-12-01 cohort is dropped and h=30 is
>   bannered as unmeasured (`55ac2df`); band-geometry eras are tracked and must not be pooled.
> - **Item 12 — the climatology-vs-GBM gate:** ✅ **RUN, and the GBM LOST.**
>   `docs/research/2026-08-19-climatology-vs-gbm-band.md`. The featureless per-item climatology
>   band is 33–46% narrower at matched 80% coverage and better calibrated on served replay, and
>   it **shipped as the default on 2026-08-20**. Since this was the review's own gate on whether
>   "the rest is worth building", items 13/15 (σ modelling, re-scoring boost rounds on width)
>   should be re-scoped against the climatology baseline, not `sigma`.
> - **Item 14 — ACI at 20 served dates:** ⏳ armed but dormant; outcomes accrue ~1/day.
> - **"If a cash edge is the goal … a longer horizon":** ❌ **TESTED AND DEAD (2026-08-20).** At
>   h=90/180 the realised move clears the friction bar but selection is null (AUC 0.51–0.53 over
>   3 OOS windows, top-decile lift ≤1.0×) and the unconditional bet lost money in every window.
>   **The trade hunt is closed at every horizon; the product is range-only.**
>
> **One of the review's own recommendations has been probed and did not replicate as stated:**
> §10.4's median-impute "bull prior" was built as `FEATURE_NATIVE_NAN` (gated off, `4d3e681`);
> the served effect is modest and **downward** (mean-reversion), the opposite of the review's
> claim (`changelog/2026-08-21-feature-native-nan-built-gated-off.md`). ~~Not shipped; a retrain
> is the go/no-go.~~ **Update 2026-09-30:** shipped 2026-08-21 in #30 (`FEATURE_NATIVE_NAN: "1"` in
> `price-forecast.yml`, on for the nightly retrain; `experiment_log.csv` verdict `shipped`). The paired
> retrain showed no regression (coverage/width identical within 0.2pp / 0.06%) and the benefit is
> within noise (DA higher in 5/6 cells, centre flat).


**Date:** 2026-08-19 · **Revision 2** (supersedes revision 1 of the same day; §0b lists what changed
and why)
**Scope:** the whole product — ingestion, features, labels, training, the served band, the
evaluation layer, the research record, alternative designs, and profitability.
**Method:** eight audits (four in revision 1, four adversarial/deeper in revision 2, two of which
were tasked with *falsifying* revision 1) plus two first-hand measurement passes: one against the
**production** Postgres panel and one against the **durable Parquet archive**, both read-only.
Numbers below were computed here, not quoted from existing docs, unless attributed.

Docs-only review. No code was modified.

---

## 0. The one-paragraph answer

**The price series this model is trained on is not one quantity — it is at least five different
estimators stitched end to end, and the joins are visible as fake market-wide moves of +26.5% and
−13.7% on 2026-07-09 and 2026-07-11** (§1). That is roughly 12σ and 6σ against the 2.25% standard
deviation of the daily cross-sectional median return, it sits inside the training window, and three
of the fourteen evaluation dates are anchored on features that span it. This is a larger and better
evidenced cause of the "accuracy is clustered by forecast date" phenomenon than regime
non-exchangeability, and it was not on anyone's list.

Second: **production is broken** — an ORM column with no migration (§2). Third: the served band's
geometry **changed on 2026-08-06**, so the fourteen-date panel every calibration figure rests on
mixes two different band constructions, and only 4 dates at h=3 and 2 at h=7 use the current one
(§4). Consequently the headline over-coverage number is **not reliably measurable today**, which is
a correction to revision 1.

On profitability the answer is unchanged and it is the firmest result in the review: **no served
forecast in the ≥$1 panel has ever predicted a move that clears the round trip** (§7).

---

## 0b. Corrections to revision 1

Two subagents were tasked with falsifying revision 1. They succeeded in eight places. Recorded here
rather than silently edited, because the pattern — a mechanism confirmed by reading, a magnitude
inflated by the wrong denominator — is the same failure mode this repo's research record already
suffers from.

| # | Revision 1 said | Correct position |
|---|---|---|
| 1 | The band over-covers 5–9pp; the fix is worth 17–29% of width | Measured on the **mixed-geometry** panel. On the 4 clean h=3 dates coverage is 0.914 (scale 0.58); on the 2 clean h=7 dates it is 0.832 (scale 0.90); h=14/30 have **zero** clean dates. Direction holds, **magnitude is not measurable** (§4) |
| 2 | Signing the conformal quantile buys 11.5–14.0% width because returns are skewed | The gain is real but it is **almost entirely recentring, not skew**: recentre-only buys 11.1/11.6/13.3%, adding asymmetry buys a further **0.4/0.2/0.7pp** (§5). The mechanism was misattributed |
| 3 | `DIRECTION_UPWEIGHT = 1.5` is a sufficient cause of `P(actual<mid) = 0.60/0.65/0.71` | Not supported. `_recenter_on_direction` overwrites the median's **sign** from a 3-class classifier, so the upweight corrupts \|mid\| only. Median `r̂` of +0.0000 is 2-decimal rounding and the classifier's flat call, not a 58th-percentile quantile (§6) |
| 4 | Frozen rows voided from calibration explain a large share of over-coverage | ~**3pp** of it. Revision 1 sized it with the raw-series stale rate (12–27%) against a scored panel that is 0–1.8% stale — two different populations, which `.claude/rules/labels-and-embargo.md` explicitly warns against |
| 5 | Price-primitives MDE is 1.15/2.76/3.13/7.13pp and "placebo never run" | **Both wrong.** `accuracy-opportunities.md:428-430` retracts those figures ("not citable — use 2.21–3.69pp"), they come from a per-fold power calculation not comparable to a bootstrap half-width, and `ab_test_price_primitives.py:292` **defines a placebo arm** |
| 6 | `age_only` is positive at all four horizons | **Three.** h=7 is +0.75 [−0.07, +1.80], spanning zero. And the contrast is not free: the placebo permutes 9 columns while `age_only` adds 2, so it needs a capacity-matched arm |
| 7 | The `1.1607` Steam fee constant is unverifiable, its DB gone | **False.** `backend/runtime/steam_listing_history.db` exists (98 MB) and returns the 528,573 rows / 262 items the original cites. The circularity is real; the diagnostic is re-runnable |
| 8 | `/accuracy/outcomes/stats` reads the dangerous local parquet | **Prod-false.** `OPS_DIR` is one path that resolves to the *durable* CI-written mirror in production. The invariant-4 breach stands; the "selected on the verdict changed" hazard does not reach the deployed surface |

Also corrected: the `--sell-only` giveback is **2.6–10.6pp** on the default ≥$1 cohort (not
5.3–17.8pp); the bootstrap-overlap defect is real at **h=14** (three consecutive dates), not h=30,
which has one date and so cannot overlap with itself; the survivorship counters
(`_count_unresolvable`) **already exist** and need propagating, not building; seed-only variance
"bounds every interval" is wrong — arms are paired at fixed seed and the subsample no longer runs at
the $1 floor; and boost rounds were calibrated on **pooled**, not within-date, rank IC.

---

## 1. 🔴 The label is not one quantity — the consensus estimator changes five times

Measured directly on the durable archive (`prices-2025.parquet`, `prices-2026-0*.parquet`), source
composition by month:

| period | sources voting | what "price" means |
|---|---|---|
| ≤2025-12-31 | `NULL` only | one legacy series (Steam median-sale, per `ab-statistics`) |
| 2026-01 → 02 | `aggregator_sync` only | **one** feed, `last_24h` falling back to 7/30/90-day means |
| 2026-03 | buff163 + youpin + csfloat + sync | vote possible |
| 2026-04 | + `aggregator_steam_17mafo` | vote possible |
| **2026-05 → 06** | **`aggregator_steam_17mafo` only** | **one** backfilled feed, rescaled (see below) |
| 2026-07-11 → | 9 labels / 7 venues | outlier-voted median as documented |

So the three-or-more-source vote the README describes is **impossible for 13 of the archive's 13.5
years and again for all of May–June 2026.** For those spans "consensus" is a single quote.

**The joins are visible as market-wide moves that never happened.** Cross-sectional median daily
return, ask-side sources only:

| day | items | median daily return |
|---|---|---|
| 2026-07-08 | 26,190 | 0.00% |
| **2026-07-09** | **5,502** | **+26.47%** |
| 2026-07-10 | 26,178 | −0.00% |
| **2026-07-11** | **5,525** | **−13.73%** |
| 2026-07-12 | 8,691 | 0.00% |

The 2026 standard deviation of that statistic is **2.25%**, so these are ~12σ and ~6σ events in the
*median* item, and the item count collapses to a fifth on exactly those two days. The Feb→Mar and
Apr→May transitions are smooth (0.00%) — consistent with `merge_17mafo_gap.py` having rescaled the
backfill to match, which hides that break rather than removing it.

**Why this matters more than anything else in the review.** Every 3/7/14/30-day return spanning
2026-07-09 or 07-11 is a basis change, not a price move. Forecast dates **2026-07-17, 07-18 and
07-19** — three of the fourteen evaluation dates — carry features whose 7/14/30-day lookbacks span
it, and h=14/30 labels anchored in late June resolve across it. "Accuracy is clustered by forecast
date" has been treated as a market fact and a regime problem; a large part of it is the estimator.

**Fix.** Emit `n_ask_sources` and a source-set bitmask into the archive — the vote already computes
both (`forecaster.py:2023`) — and let loaders require constant composition. Add a composition-break
calendar and exclude spanning windows from labels the way the embargo already excludes overlapping
ones. **~1 day plus a retrain**, and until it is done no 2026 A/B is interpretable.

### 1b. The backfill that fills May–June is partly fabricated

`scripts/merge_17mafo_gap.py` wrote ~84 days × ~24k items (2,161,250 rows):
`last_24h → last_7d → last_30d` fallback written as a spot print with no flag (`:59-63`);
`volume: 0` fabricated (`:74`) — the exact zero the rest of the codebase removed
(`pipeline.py:41-50`); every price multiplied by a cross-venue basis factor interpolated from a
*different* window and different sources, with the market-wide median factor substituted where
there is no overlap (`:175-198`); `duckdb.Error` swallowed into a silent factor of 1.0 (`:162-167`);
and `append_monthly` called with no `ingested_at` (`:228`), so provably-late rows read "arrival
unknown". These rows *are* the Apr 16–Jul 10 series and cannot simply be deleted; at minimum they
need a distinguishing source label so a loader can exclude them.

### 1c. The 2σ outlier vote is near-inoperative at small n

`forecaster.py:2036-2043` computes `np.std(prices, ddof=0)` on the same points it screens —
breakdown point zero, so one outlier inflates the very threshold meant to catch it. This is the
small-sample ceiling `max|x−x̄|/σ = (n−1)/√n`, which is below 2 for n≤5; a planted 10× outlier is
missed on 55% of item-days at n=3 and 73% at n=5. And at n=2 (`:2028`) `np.median` is the **mean of
two quotes** — a different estimand whose level moves whenever either leg appears or disappears.
*Fix: MAD-based scale (`|x−med| > k·1.4826·MAD`), bump `VOTED_CACHE_VERSION`, treat as a label
change. ~2h + a paired A/B.*

### 1d. Cheaper ingestion defects worth doing first

- **`historical_fallback:` rows are re-stamped stale prices, and only production filters them.**
  `collectors/pipeline.py:236-248` writes a price up to 7 days old under **today's** `day`. Only
  `forecaster.py:1787` and `:1911` exclude the prefix — `archive_universe_sql_filter` does **not**,
  despite invariant 2 advertising it as carrying all the universe rules, and no `ab_test_*.py` or
  `walkforward_backtest.py` mentions it. 12,655 rows over exactly 6 days (2026-07-11→16), on the
  *selected* cohort that failed to match. **~30 minutes, best correctness-per-effort in the review.**
- **FX is collected and never applied.** `pipeline.py:393-404` writes exchange rates; the only
  reader in the entire backend is `scripts/backfill_buff_iflow.py:67`. CNY→USD for BUFF and youpin
  happens upstream inside CSGOTrader at an unknown rate on an unknown date, with no snapshot — a
  market-wide multiplicative factor on 2 of 7 sources with **no local detector**. *Fix: a daily
  assertion that `median(youpin/csfloat)` and `median(buff163/csfloat)` stay in band. ~2h.*
- **Item identity collapses ★ and StatTrak™.** `csgotrader_aggregator.py:109-116` strips those
  tokens when expanding match candidates and `:229-236` appends five wear suffixes taking the first
  hit — so a wear-less name binds deterministically to Factory New and a StatTrak knife can bind to
  the non-StatTrak price. Seven residual collision groups survive normalisation on 2026-07-11+.
- **The archive is not append-only.** `append_to_parquet.py:262` `drop_duplicates(keep="last")`
  overwrites a stored price for an existing `(item_slug, day, source)`, and the publish leg does
  `git checkout --orphan` + `push --force`, so no history of the previous value exists anywhere.
  `ingested_at` is populated on only **10.6%** of rows (2,394,595 of 22,499,308), all 2026-08-08
  onward — for the other 89.4% a revision is undetectable.
- **Zero-row guard holes.** `walkforward_report` returns `records_stored`, which is not in
  `ROW_COUNT_FIELDS`, so zero records exits green; a count of `None` fails the `isinstance` test and
  becomes invisible; `supply_depth` and `sales_volume` never reach the guard (`continue-on-error`);
  and `backtest_historical` passes a type the scorer rejects, so its guard fires every run.

**Checked and cleared:** "day gaps are cron drift" is *correct* — gap incidence is flat in price
tier (12.22% at <$1 → 12.98% at >$100), so there is no liquidity selection in gaps, and
`collectors/snapshot_date.py` correctly decouples the 22:00 UTC dump boundary from the 23:00 cron.
One unaccounted missing day (2026-08-02) versus the three its docstring names.

---

## 2. 🔴 Production is broken — an ORM column with no migration

`Item.is_trainable` was added at `backend/database.py:55` (commits `701a4df`, `fde8314`) and merged
to `main`. **No Alembic migration was ever written.** Prod is at `0022_add_forecast_anchor_disclosure`;
there is no `0023`, and `items` in prod has no such column. Re-verified today.

```
psycopg2.errors.UndefinedColumn: column items.is_trainable does not exist
```

The aggregator fails on its first task. `price-forecast.yml` and `backtest-accuracy.yml` are chained
off its success, so both now **skip silently**. Last green aggregator 2026-08-17 23:21 UTC; the
2026-08-18 run failed and every subsequent nightly run will fail identically until the migration
lands. Last served `forecast_date` 2026-08-17; last `evaluated_at` 2026-08-14.

This also stalls the only remedy already built: the served-outcome `q_hat` feedback self-activates
at 20 distinct served dates and the panel is at 14.

*Fix: migration `0023` (integer, default 0, backfill from `is_backfilled`), apply, re-run.
**~30 minutes.** Then a CI step that runs `alembic upgrade head` against a scratch DB and diffs
`Base.metadata` against the live schema — an ORM column with no migration passed review, tests and
merge.*

---

## 3. 🔴 Two of the fourteen evaluation dates are not what they claim

- **`2025-12-01` was created 2026-07-17** (`item_forecasts.created_at`). It is a replay, and its
  stored `current_price` matches the **2026-07-18** price frame to a median |log ratio| of **0.016**
  against **0.244** for its own nominal date. Its published dollar coverage (0.19–0.35) is an
  artifact of that; its calibrated coverage (0.89–0.91) looks fine only because the rebasing step
  divides the contamination out.
- **It is the only h=30 date in the panel.** Every h=30 figure this repo has published rests on it.

**The served quote was also stale for multi-day runs.** Asking which date's `base_price` each
forecast date's `current_price` best matches:

| forecast_date | quote actually matches |
|---|---|
| 2026-07-17 / 18 / 19 | 2026-07-18 |
| 2026-07-31 → 2026-08-07 | **2026-07-31** |
| 2026-08-09 | 2026-08-01 |
| 2026-08-11 | 2026-08-11 ✅ |

Forecasts stamped 2026-08-07 were quoted off 2026-07-31 prices. The cohort-median quote-vs-base
wedge by date runs **−16.6% to +26.1%** against band half-widths of 8–21%. `forecast_date` equals
`created_at`'s date on every historical row, i.e. the `anchor_date → today` fallback at
`scripts/forecast_prices.py:335-338` was in force. It is now instrumented (`anchor_clean` populated
from 2026-08-11) and **~34% of currently served rows still carry a dirty anchor**.

*Fix: drop the replayed cohort from every published figure and banner the docs that quote an h=30
number; make a stale anchor refuse to serve rather than disclose. ~half a day.*

---

## 4. 🟠 The band's geometry changed on 2026-08-06, so the panel mixes two products

Median `high−mid` over `mid−low` on well-formed ≥$1 rows, by forecast date:

| dates | median hi/lo | share exactly symmetric |
|---|---|---|
| 2025-12-01 → 2026-08-04 | **1.03 – 1.52** | 1–6% |
| 2026-08-05 | 1.06 | 22% |
| **2026-08-06 → 08-11** | **1.0000** | 44–50% (rest is $0.01 rounding) |

Current serving is symmetric — `conformal.band` returns `mid ± q_hat·scale` and every downstream
transform preserves both half-offsets, which an independent code read confirms. The asymmetry is a
**legacy artifact in 9 of the 14 evaluation dates**, and the two geometries are not comparable.

Restricting to clean-geometry dates changes the headline materially:

| h | cohort | dates | n | coverage | scale for 80% | P(actual<mid) |
|---|---|---|---|---|---|---|
| 3 | all minus replay | 9 | 9,319 | 0.864 | 0.749 | 0.601 |
| 3 | **clean geometry** | **4** | 3,959 | **0.914** | **0.583** | 0.576 |
| 7 | all minus replay | 11 | 11,323 | 0.893 | 0.692 | 0.646 |
| 7 | **clean geometry** | **2** | 1,909 | **0.832** | **0.895** | 0.699 |
| 14 | all minus replay | 5 | 5,121 | 0.888 | 0.763 | 0.708 |
| 14 | **clean geometry** | **0** | — | — | — | — |

At 4 and 2 dates the standard error on a coverage estimate is roughly ±20pp and ±28pp. **The band
over-covers — that direction is consistent everywhere — but its magnitude is not measurable on the
current product today.** Revision 1's "17–29% of free width" was computed on the mixed panel and
should not be quoted.

The honest consequence: **there is not yet enough clean served data to calibrate anything**, and the
accumulation is stalled by §2. Getting the pipeline back up is not merely an ops chore — it is the
precondition for every calibration decision on this list.

Two related defects, both confirmed:
- **15.3% of all served bands are degenerate (`high <= low`)**, and 22.7% have `high == mid`. This
  is entirely sub-$1: at ≥$1 it is 0.0000%. It is `round(price, 2)` collapsing a band narrower than a
  cent. Sub-$1 is 84% of the served catalogue, so the product publishes a zero-width interval for a
  large share of it — and that also explains why sub-$1 "coverage" looks near-nominal.
- **`served_recalibration.py:62-66` justifies its two-sided score by "the band is asymmetric".** It
  is not, for the population it scores; the score reduces algebraically to `|actual−mid|/half`. The
  arithmetic is fine, the justification is false, and clamped rows get an enormous `r` that biases
  the factor up against `FACTOR_MAX`.

---

## 5. 🟢 The band's centre is biased, and correcting it is the cheapest real win

Decomposing the recalibration gain on an expanding window over served dates:

| h | A: symmetric about `mid` (today) | B: recentre only, still symmetric | C: two signed quantiles |
|---|---|---|---|
| 3 | 0.894 cov / 2.044 width | 0.881 / 1.817 (**−11.1%**) | 0.866 / 1.808 (−11.5%) |
| 7 | 0.851 / 1.615 | 0.836 / 1.428 (**−11.6%**) | 0.828 / 1.423 (−11.8%) |
| 14 | 0.818 / 1.566 | 0.801 / 1.357 (**−13.3%**) | 0.806 / 1.347 (−14.0%) |

**Skew is worth 0.2–0.7pp. The centre is worth 11–13%.** Revision 1 got the number right and the
mechanism wrong. Implementing it as two signed quantiles is still correct — same code, same zero
cost, and it absorbs both — but the thing being fixed is a **biased centre**, so the durable fix is
upstream of calibration, not inside it. Caveat: measured on the mixed-geometry panel (§4); the
direction is corroborated by `P(actual<mid)` staying at 0.576/0.699 on clean dates.

---

## 6. 🟠 The served median is set by a directional classifier, after calibration

`predict` builds the conformal band at `forecaster.py:8232`, and then at `:8303` calls
`_recenter_on_direction` (`:6694-6712`), which sets

```
down(0) -> -|mid|,  flat(1) -> 0,  up(2) -> +|mid|
```

from a 3-class LightGBM classifier, preserving the half-offsets. All four `clf_*.txt` exist in
`saved_models/`, so this fires on every horizon in production.

Three consequences:

1. **The quantile model supplies only a magnitude.** The sign of every served forecast comes from a
   directional classifier — in a product whose `AGENTS.md` opens "this is a RANGE forecaster, not a
   directional predictor" and whose invariant 4 forbids quoting directional accuracy alone. That is
   a genuine contradiction between the stated stance and the implementation, and it should be
   resolved deliberately in one direction or the other.
2. **It happens after conformal calibration**, so the band is no longer centred where its coverage
   was fitted. The local `meta.json` has no `conformal_centre`/`conformal_basis` keys at all while
   all four classifiers exist — i.e. the shipped artifact is exactly the `_calibrate_conformal`
   WARNING case at `:7587-7605`. This is a better explanation of the repo's own three-for-three
   puzzle ("every scale calibrates to exactly 80% on its own records and lands somewhere else when
   served") than the width variable ever was.
3. **`DIRECTION_UPWEIGHT = 1.5` therefore corrupts |mid|, not direction.** The weighted α=0.5 pinball
   loss does minimise at the 58th–60th percentile (arithmetic confirmed independently, and the
   weight is attached on every fit path — production, CV folds, regime refit, `_fold_q50_scores`),
   but that biases the *magnitude*. Setting it to 1.0 is still worth doing and is one line; it is
   no longer the explanation for the centre bias.

### 6b. 🔴 The sample weights are degenerate — measured

`_compute_sample_weights` (`:5324-5328`) builds `vol` as the 30-day rolling std of `pct_change` — a
**fraction** — then applies `np.clip(vol, 0.1, np.percentile(vol, 99))`. Measured on the real
archive, ≥$1 cohort, n=313,753:

| statistic | value |
|---|---|
| median `vol` | **0.0398** |
| p99 `vol` | 0.387 |
| clip floor in code | **0.1** |
| **share of rows below the floor** | **89.2%** |
| NaN share (fewer than 5 obs) → `.fillna(1.0)` → clipped to p99 | 1.5% |

So **89% of training rows receive an identical weight**, the floor sits 2.5× above the median, and
the 1.5% of rows with almost no history receive ~3.9× the weight of the majority — the exact inverse
of the docstring's "flat/dead items get down-weighted". The volatility weighting production believes
it ships has effectively never been active, and `DIRECTION_UPWEIGHT` and the recency decay both
multiply this degenerate base. *Fix: one line. Highest value-per-character in the review.*

---

## 7. 🔴 There is no paper-trading edge, and accuracy is not what is blocking it

Unchanged from revision 1 and re-verified. Prod panel, ≥$1, csfloat 2% fee + tier spread
(`actionable_threshold` = `ROUND_TRIP_COST[venue] + SPREAD_BY_TIER[tier]`), long round trip,
replayed cohort dropped, date-clustered bootstrap:

| h | strategy | n | dates | gross | **net** | 95% CI |
|---|---|---|---|---|---|---|
| 3 | buy everything | 9,319 | 9 | −1.60% | **−23.41%** | [−24.76, −22.47] |
| 3 | q50 clears friction | **0** | | | | |
| 7 | buy everything | 11,323 | 11 | −2.01% | **−23.81%** | [−25.06, −22.85] |
| 7 | q50 clears friction | **0** | | | | |
| 14 | buy everything | 5,121 | 5 | −4.00% | **−25.80%** | [−27.97, −23.79] |
| 14 | q50 clears friction | **0** | | | | |

| h | median \|r̂\| | p99 \|r̂\| | friction bar | median realised \|return\| |
|---|---|---|---|---|
| 3 | 0.46% | 4.4% | 12.8–23.1% | 1.0–1.9% |
| 7 | 0.86% | 7.1% | 12.8–23.1% | 1.6–3.2% |
| 14 | 1.57% | 18.0% | 12.8–23.1% | 3.6–5.9% |

The predicted move is 15–45× smaller than the round trip, and so is the **realised** move — the
market's own volatility at these horizons does not clear the quoted spread at any tier. **A perfect
oracle would fail this test too.** Mean gross return was negative at every horizon over the served
window, so long-only loses before friction. The spread table is admitted to be measured at the wrong
cuts, but even 2% fee + a 5% spread sits above p90 `|r̂|`.

The paths that change this arithmetic are all outside the forecaster: **don't cross the spread**
(patient limit-listing converts 17–21% of cost into fill risk, and nothing here models it), **longer
horizons** (realised `|return|` grows 1.0% → 5.9% from h=3 to h=14), and the **exceedance framing**
you already identified — the band as a filter, not a trade.

---

## 8. 🟠 The paper-trading harness is untracked, unrun, and order-dependent

`backtest/papertrade.py`, `scripts/papertrade_report.py`, `tests/test_papertrade.py` and its
changelog are all `??` in `git status`; `git log --all` on them is empty; no `pt_mean_net` figure
exists anywhere. Everything the repo says about paper-trading P&L is design intent.

1. **`pt_profitable` is order-dependent.** `_cluster_mean_ci` (`papertrade.py:192`) resamples from
   `by_date.values()` while `scoring.py:291-292` deliberately sorts, *and* `_SELECT`
   (`papertrade_report.py:44-48`) has no `ORDER BY`, so Postgres row order is unpinned and successive
   runs can return different CIs on a strict `lower > 0` test. **~15 min, do this before it is ever
   run.**
2. **`min_dates` is dead** (`:215`, `:280`, `:292`), and the harness runs only at h∈{14,30} — which
   have **6 and 1** forecast dates against `MIN_FORECAST_DATES = 20`. h=30 cannot produce a CI at
   all. The report never prints the date count.
3. **Overlapping windows under-disperse the CI at h=14**, where 2026-07-17/18/19 and 07-29/07-31
   mean 6 clusters are ~3 independent blocks. `paired_mde.py` documents this exact failure.
4. **No multiple-testing account** across up to ~108 `pt_profitable` cells, while
   `directional_test.py:75` already sets `PT_T_HURDLE = 3.0` citing Harvey-Liu-Zhu.
5. **`--sell-only` is unjustified** — archive prices are ask-side (`forecaster.py:907` records that
   excluding bid sources moved the consensus −10.8% on the ≥$1 cohort), so a round trip crosses the
   full spread on the sell leg. Opt-in only; the giveback is 2.6–10.6pp on the default cohort.
6. **Survivorship is real but the counters exist.** `_count_unresolvable` (`backtest_accuracy.py:1249-1257`)
   already bins every drop; unresolvable forecasts never become `forecast_outcomes` rows, so the fix
   is to propagate a counter into the report header, not to build one.
7. **Self-contradiction:** the module docstring and the changelog both say results are reported by
   tier, yet `--by-tier` is opt-in.
8. **`GET /accuracy/outcomes/stats`** returns a bare pooled `overall_accuracy` across every horizon,
   tier, model version and date, with no PT stat and no `realised_down_rate` — an invariant-4 breach
   on a published HTTP surface. (It reads the durable mirror in prod, not the hazardous local copy.)

---

## 9. Alternative designs worth considering

The framing fact: median `|r̂|` is 0.46–1.57% against half-widths of 8–21%, so **the centre moves the
served interval by under 5% of its own width. The product is almost entirely a scale estimate.**
Judge every proposal on the scale, not the centre.

Ranked by expected value, with the trap test this repo needs (would it be CV-positive and
serving-negative?):

1. **Per-item climatological interval as the shipped default, and as the null.** Empirical quantiles
   of the item's own trailing h-day returns, centred at 0, shrunk toward the tier's pooled quantiles
   by `n_i/(n_i+k)`, conformalised with the existing machinery. No features, no boosters, no CI
   budget. **This is the null the whole programme has never had**: run it and the GBM on the same
   served dates and compare only coverage and mean width at matched coverage. If climatology sits
   inside the ±SE band and is not wider, the GBM is decorative. *Anti-trap by construction.*
2. **Partial pooling of the nuisance parameters** — σ_i, the smoothing coefficient φ_i, the freeze
   probability π_i — toward `rarity × collection × tier` groups. This is the classical answer to
   "thousands of units, few episodes each" and it appears nowhere in this repo. It directly fixes
   "`min_periods=1` gives a 14-day-old item an under-dispersed σ, so the band is narrowest where
   information is thinnest." *Structurally immune to the trap: pooling is variance reduction, not
   signal extraction, and a per-item volatility level has no date-factor confound.* Note the
   corollary — **do not pool the centre**; full pooling of the centre is `mid = 0`, which is roughly
   what the evidence supports anyway.
3. **EWMA σ (λ≈0.94) on a date-reindexed series, in place of the 60-row rolling std.** Fixes the
   row-vs-date window defect as a side effect and reacts faster to regime shifts. ~20 lines, zero
   training cost. Preferred over GARCH, which needs long clean series and will fit degenerately on
   gappy, frozen ones.
4. **`DIRECTION_UPWEIGHT = 1.0` and the sample-weight floor fix (§6b), then a z-target** `y = r/σ_i`.
   Same conditional median by construction, but splits stop being chosen by the volatile tail of the
   universe, and `q_hat` ends up fitted in the units the model trained in. **Hard ordering
   constraint: fix the σ windows first**, or the z-target explodes on exactly the noisiest rows.
5. **Pool the four horizons into one booster** with `[h, √h]` as features and target `r_h/(σ_i√(h/3))`,
   conformal still per-horizon. ~4× effective data per parameter, kills the unexplainable
   `{300, 750, 150, 1000}` boost-round spread, structurally removes the `feature_medians` bug class,
   and — because one CV pass yields OOF residuals for all four horizons — it *reduces* the phase
   that is 63% of training time.
6. **h=30: delete the booster.** 1000 rounds against a pinball optimum of 25, one resolved date, no
   validation channel. Generate the band by term-structure extrapolation `q_hat(h) = a·h^b` fitted on
   h=3/7/14 (b≈0.5 under a random walk; fitting b lets the data speak about mean reversion), centre
   at 0, and flag it `n_dates=1` in the API. Do not delete the *horizon* — long horizons are the only
   place friction could ever be cleared, and the same machinery gives h=60/90 as band-only products.
7. **Date-demeaned target** `r_it − r̄_t` with the market term pinned near 0. The bias rising
   monotonically with horizon (0.60/0.65/0.71) is the signature of a level/drift term. *Superficially
   in the refuted family, but the direction is opposite: the failed arms demeaned to **extract**
   per-item signal; this demeans to **delete** a term you are declining to forecast.* Testable with
   no retrain — it is a per-date level shift on stored OOF records.
8. **AR(1) unsmoothing for illiquid items.** Consensus over stale asks is an appraisal-smoothed
   series, so realised variance is biased **low** exactly where the band needs width. It has a free
   kill condition: measure φ̂ by liquidity bucket, and if it is not decreasing in liquidity the
   mechanism is absent. Bonus testable prediction — **`−return_1d`'s edge should be concentrated in
   illiquid items and vanish on liquid ones**; if so, the naive baseline that has beaten this model
   for a year is an artifact, not a signal. Honest caveat: unsmoothing makes bands *wider*, so ship
   it only bundled with a level correction.
9. **Hurdle / zero-inflated formulation.** Correct in principle — the outcome has an atom at zero —
   and with mass π an 80% interval only needs the continuous part to cover `(0.8−π)/(1−π)`. But
   frozen mass is 10.1/5.3/2.4% at h=3/7/14 and frozen items are *cheap*, so this is an h=3-only
   intervention worth ~1.5pp for a second model in the serving path.
10. **A supervised dispersion model** (LightGBM on `|OOF residual|`). Appealing because it subsumes
    all three refuted width variables — and **the highest trap risk on this list**: per-tier oracle
    scales differ sharply in-sample and are identical out of sample, which is the exact
    CV-positive/serving-negative fingerprint. If built at all, gate on width-at-matched-coverage
    winning on ≥8 of 10 served dates.
11. **Skip:** NGBoost and heteroskedastic Gaussian heads (parametric families on fat tails, and it
    means leaving LightGBM); log-vs-simple return as a *bias* fix (the median commutes with monotone
    transforms, so it cannot move `P(actual<mid)` — do it for tail hygiene only); Mondrian/per-tier
    conformal (measured null out of sample).

External methods that remain worth their cost once §2 and §4 are resolved: **ACI / DtACI**
([2106.00170](https://arxiv.org/pdf/2106.00170), [2208.08401](https://arxiv.org/pdf/2208.08401)) as
the upgrade to the batch served-feedback scalar, at h=3/7 only — never at h=30, where feedback lags
30 steps; **weighted / recency-decayed calibration**
([2202.13415](https://arxiv.org/abs/2202.13415)), distinct from the shelved recency verdict because
it weights *calibration scores* rather than *training rows*, and it yields a computable
departure-from-exchangeability bound — a number to replace an assertion; and the **wild cluster
bootstrap** ([MacKinnon–Nielsen–Webb 2023](https://www.sciencedirect.com/science/article/pii/S0304407622000781))
as standard discipline, since G here is 2–12 and the few-clusters regime is exactly what it is for.

---

## 10. Other confirmed modelling defects

1. **Rolling windows are row-based while lags are date-based**, and the archive is ~**48% dense** —
   the module's own `PREDICT_TAIL_ITEM_DAYS` comment says so. `price_std_60d` is therefore a
   ~120-calendar-day volatility mislabelled as 60-day, and it is the **sole input to the band's
   `sigma`**. `price_accel_7d` has the same mixed basis. *Confirmed by two independent reads; no
   `asfreq`/`resample`/date-reindex exists anywhere in the module.*
2. **Staleness contaminates features, not just labels.** `LABEL_MAX_STALE_RUN_DAYS` voids frozen
   runs from the label; nothing guards the feature path. Bit-identical consecutive-day rates on
   2026-07-11+ are 61.1% (<$1), 41.1% ($1–5), 31.6% ($5–20), 28.4% ($20–100), 28.5% (>$100), and
   median item CV by frozen share runs 0.0365 → 0.0570 → **0.0246**, so σ is *inflated* on
   partially-stale items and *deflated* on fully-stale ones. Both are wrong in the variable the band
   is built from.
3. **Boost rounds and the Optuna objective are both selected on pooled out-of-fold rank IC** — a
   metric `README.md:19` says is never safe to rank on — while the same file's table (`:836-849`)
   records that validation pinball loss is minimised at **25 rounds** at 14d and 30d and rises
   monotonically after. Production trains 150 and 1000. Both curves are read off one booster, so the
   comparison is sound. Pinball loss at α=0.5 *is* mean |residual|, the quantity `q_hat` takes the
   80th percentile of, so overshooting inflates every band at those horizons. *Note: pooled, not
   within-date — weaker than revision 1 credited, given this repo's own date-clustering finding.*
4. **Median imputation removes LightGBM's NaN handling and injects a bullish prior.** Applied on
   train, val, CV folds and predict, so no NaN ever reaches the booster. `meta.json` medians include
   `return_180d = 6.3495`, `return_120d = 4.11`, `return_90d = 2.67` — a newly-eligible item with 14
   days of history is served a coherent bull trend it never had.
5. **`self.feature_medians` is overwritten per horizon** and `predict` uses the surviving value for
   all four. Currently a **null effect** — all four horizons carry identical 33-column sets — but it
   arms the moment pruning diverges by horizon, and the fallback silently swaps a *training* median
   for a *serving cross-sectional* one under the same name.
6. **`_prune_features` is unpinned.** `feature_cols` is selected by pooled Pearson correlation at
   train time with no assertion or comparison against the previous artifact, so the served feature
   set can drift silently *between* retrains. Exposure is limited by the allowlist running first
   (33 candidates, not 123) and by per-artifact persistence, so this is drift, not train/serve skew.
7. **The distribution-free guarantee is not available** — records are pooled OOF across different
   fold models and disjoint calendar windows, effective n is the ~30 fold-dates. *The file already
   says this at `:8905-8918` and publishes the per-fold ratios; it is not a new finding, and the
   remedy was declined deliberately.*
8. **Dead weight:** `MOMENTUM_FALLBACK_HORIZONS = []` makes `_recenter_on_momentum` unreachable;
   `QUANTILES = [0.5]` and `N_ENSEMBLES = 1` leave whole loop and ensemble scaffolds; three
   mutually-exclusive off-by-default band scales are all refuted, and `EXCEEDANCE_SCALE` uses a
   one-sided upside probability to set a symmetric width, which is incoherent independent of the
   refutation. Latent trap: `distance_to_support` / `distance_to_resistance` apply
   `.replace(0, np.nan)` to the **numerator**, so "price is exactly at its 30-day low" becomes NaN
   and is median-filled — not served today, but delete or fix before anything re-admits them.

**Verified clean, do not re-audit:** the H+13 embargo is applied at both purge sites, including the
positional fallback; fold medians are train-only; `market_factor_{h}d` is excluded from features;
the CV grid is end-anchored; `q_hat` comes from pooled OOF records, never from `dval`, except a
guarded fallback that warns; per-item row capping only thins the train set; and no lookahead exists
in any of the 33 served features — every window is trailing and the label merges on an exact future
date.

---

## 11. The research record: verdicts weaker than their banners, with caveats

The item-level A/B family is largely underpowered, but revision 1 overstated the case (§0b items 5
and 6). The defensible version:

| Arm | Effect | Power | Reading |
|---|---|---|---|
| Three 2026-08-06 model fixes | −1.39 / +0.38 / −0.67 / −1.74pp | MDE **4.11 / 3.37 / 2.92 / 3.52pp**, 8 folds | Unresolved — *and the doc already says so* |
| CSFloat basis | −0.05 … −1.68 | MDE **1.19 / 1.40 / 2.69 / 3.69pp**; 4.58pp at 30d is a *different arm*; the 14d **placebo** is −0.47 [−0.93, −0.05], excluding zero | Instrument broken at 14d |
| Embargo contrast, h=7 | +4.29 [−0.26, +9.14] | MDE **4.70pp** | Point estimate inside its own MDE — `null` means unresolved, *already stated in the source* |
| Classifier HP, quality-spread, recency | "0/4", "net-flat" | **no MDE**, gates on fold-win counts | Not tests |
| Price primitives | −0.69 … −1.38pp | ⚠️ its MDE figures are **retracted by their own doc**; use 2.21–3.69pp; a placebo arm **does exist** | Weakly unresolved |
| `age_only` | +0.46 / +0.75 / +0.60 / +1.60 | 25–26 folds; positive at **three** horizons, h=7 spans zero; 14d sits in the column the same doc calls unreliable | Open, needs a **capacity-matched** placebo |

Standing caveats, with their limits stated:
- **`paired_mde` reproducibility.** The identical command moved `mean_diff_pp` −0.1581 → −0.0026,
  ~19× the reported interval, cause never identified. *But* `n_paired` and `n_dates` moved too, so
  the nondeterminism is in **frame construction**, not the estimator, and the observation is from a
  seed placebo at h=3 only.
- **Seed-only retrains move accuracy by sd 1.82/3.05/1.54/2.77pp** over 8 seeds. *But* arms are
  paired at fixed seed, and at the $1 floor the 1.2M budget covers 926 items whole, so the subsample
  never runs. This bounds cross-run absolute levels, not paired intervals, and it is already fixed.
- **Per-date sd of `excess` is 14.96/22.40/21.56/24.39pp**, and `docs/README.md:103` already states
  "never quote `excess` on fewer than ~50 forecast dates". The panel has 10/12/6/1.
- **The `1.1607` Steam fee constant's validation was circular** (91.61% of pairs are the same numbers
  after dividing) — but `backend/runtime/steam_listing_history.db` is on disk and the diagnostic is
  re-runnable, so this is a half-day of work, not a dead end.
- **Three harnesses print `(paired, dates clustered)` while passing `cluster_key="fold_id"`.** The
  code is right and the label is stale; the deferral is recorded so stored artifacts keep matching
  their source. Cosmetic, 15 minutes.

**What to do about it.** Do *not* broadly re-run the nine repaired harnesses: given §7 a 1pp DA win
is neither shippable nor profitable, and given §1 no 2026 A/B is interpretable until composition is
persisted. Instead: put the MDE beside every stored verdict and relabel underpowered nulls
UNRESOLVED; fix the frame-construction nondeterminism before quoting another paired interval; run
the one capacity-matched `age_only` placebo and close that thread; fix the three stale console
labels; and delete or re-banner `accuracy-opportunities.md`, which now carries a withdrawal of a
withdrawal and is a net negative to read.

---

## 12. Ranked plan

**Unblock — nothing else can be measured until these land:**

1. Migration `0023` for `is_trainable`; re-run the aggregator. **30 min.** (§2)
2. Schema-drift check in CI (`alembic upgrade head` on a scratch DB, diff against `Base.metadata`).
   **1h.** (§2)
3. `historical_fallback:` into `archive_universe_sql_filter`. **30 min.** (§1d)

**Free wins, ~1 day, no accuracy risk:**

4. Fix the `_compute_sample_weights` clip floor — 89% of rows currently share one weight. **1 line.**
   (§6b)
5. `DIRECTION_UPWEIGHT = 1.0`. **1 line.** (§6)
6. Pin `ORDER BY` on `papertrade_report._SELECT` and sort the clusters in `_cluster_mean_ci`, so
   `pt_profitable` stops being order-dependent before the harness is ever run. **15 min.** (§8)
7. Recentre the band (implemented as two signed conformal quantiles). **~half a day, zero training
   cost.** (§5)

**Measurement integrity — do before quoting any calibration figure again:**

8. Persist `n_ask_sources` + a source bitmask; build a composition-break calendar; exclude spanning
   windows. **~1 day + retrain.** (§1)
9. Drop the replayed 2025-12-01 cohort everywhere; banner every h=30 figure; make a stale anchor
   refuse to serve. **~half a day.** (§3)
10. Tag the pre-2026-08-06 band geometry in the panel and stop pooling across it; print `n_dates` and
    a ±SE band on every coverage figure; adopt the wild cluster bootstrap. **~half a day.** (§4)
11. Cross-venue FX ratio monitor. **2h.** (§1d)

**Then, in order:**

12. Run the **climatology-vs-GBM** width-at-matched-coverage comparison on stored OOF records and the
    served dates. No retrain, no serving-path code. **This determines whether the rest is worth
    building.** (§9.1)
13. Date-indexed rolling windows and a staleness-aware σ; then partial pooling of σ. (§10.1, §10.2, §9.2)
14. Accumulate to 20 served dates, then upgrade the feedback scalar to ACI at h=3/7 only. (§9)
15. Re-score boost rounds and the Optuna objective on width-at-matched-coverage. (§10.3)
16. Fix the remaining paper-trading defects, then **run it against prod and record the first real
    result.** (§8)

**Do not do:** Mondrian/per-tier conformal (measured null out of sample); a fourth band-width scale;
any per-item directional or cross-sectional arm; CQR with new boosters until §5 is measured;
a broad re-run of the nine repaired harnesses (§11); NGBoost or a heteroskedastic Gaussian head;
log-return as a *bias* fix.

**And accept:** improving this forecaster will not make it profitable (§7). If a cash edge is the
goal, the next experiment is a patient-limit-order execution model or a longer horizon — not a
better model.

---

## Appendix — what was measured here, and how

All read-only. Prod Postgres via `DATABASE_URL`; durable archive at
`cs2-oracle-data/price-archive` (complete to ~2026-08-14). The local `backend/cs2_market.db` is a
synthetic fixture and the local `price-archive` holds only 2026/07 — neither was used for any figure.

| § | Measurement | Source |
|---|---|---|
| 1 | Source composition by month; cross-sectional median daily return at the joins | `prices-2025.parquet`, `prices-2026-0*.parquet` via DuckDB |
| 2 | `alembic_version`, `information_schema.columns`, workflow run history | prod Postgres, `gh run list` |
| 3 | `created_at` vs `forecast_date`; quote-vs-base matching across all 14 dates | `item_forecasts`, `forecast_outcomes` |
| 4 | `high−mid` vs `mid−low` by date and model version; clean-date coverage | `forecast_outcomes` |
| 5 | Expanding-window recalibration, three arms | `forecast_outcomes` |
| 6b | 30-day rolling std of `pct_change` vs the 0.1 clip floor | `prices-2025.parquet` |
| 7 | Strategy P&L with `friction.py` constants, date-clustered bootstrap | `forecast_outcomes` |

Reproduction of stored `in_interval` from the raw columns agrees **100%**, which is what licenses
the derived coverage figures above.
