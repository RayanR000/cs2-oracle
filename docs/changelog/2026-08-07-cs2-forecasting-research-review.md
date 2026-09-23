# An external review says a bid may be voting into the labels, the headline metric measures the market, and the published backtest is unpurged

**Date:** 2026-08-07
**Review:** `docs/research/2026-08-07-cs2-forecasting-research.md` — 1,706 lines, untracked
in git at the time of writing, so it carries no commit SHA.
**Action list:** `docs/research/2026-08-07-next-steps.md` (created in the same pass).
**Bears on:** `docs/research/accuracy-opportunities.md` (its 2026-07-31 stop banner is
reopened), `2026-07-31-accuracy-work-closed.md`, `2026-08-07-paired-mde-fold-clustering.md`,
`2026-08-07-training-item-universe.md`, `2026-08-06-market-relative-labels-refuted.md`,
`2026-08-06-csfloat-basis-refuted.md`, `2026-08-06-bymykel-metadata-refuted.md`,
`2026-08-06-breadth-beats-depth-item-age-does-not.md`,
`2026-08-06-steam-listing-backfill-and-phantom-items.md`.

**No code changed in this entry.** Nothing in the review has been acted on. This record
exists so the findings are dated and tracked, and so that the numbers already published in
this repo are read with the corrections attached.

## What the review is

A commissioned from-scratch design review of the forecasting system, answering "what
features predict CS2 item prices" against three inputs: an academic literature sweep, this
repo's own measurement record (`docs/changelog/`, the memory ledger,
`docs/references/data-inventory.md`), and three web-research threads on venue
microstructure, event history and prior art. It marks every claim
**[MEASURED HERE]** / **[PUBLISHED]** / **[REPORTED]** / **[UNVERIFIED]**, and it opens with
a "Corrections to this document" block (C1–C5) in which a second pass overturns four of its
own first-pass claims. Three of those four touch live code.

It also contains original measurement taken 2026-08-07 for the review itself — §11 (venue
microstructure), §14 (cross-asset lead-lag), §25 (item attribute economics) are new numbers,
not web reading. Those are the parts that change how existing entries read.

## Five findings that change how existing numbers read

### 1. `aggregator_buff163_buy` may be voting into the consensus price as an ask (C2)

The review's first pass said "ingest BUFF's `highest_order`". Its second pass found it is
already ingested: `collectors/pipeline.py` and `collectors/csgotrader_aggregator.py` write
it as `aggregator_buff163_buy`, and `forecaster.py` feeds every non-`historical_fallback`
source into `_apply_multi_source_voting` **with no source filter**.

Measured across 23,904 items (§11), median venue price ÷ Steam price is **0.550× for the
BUFF bid** against **0.700× BUFF ask, 0.727× CSFloat, 0.802× Skinport**. A bid at 0.550 is
voting in a median whose other members sit at 0.700–0.802. `vote()` rejects >2σ outliers
only when ≥3 sources are present, so whether the bid is rejected is data-dependent and
**flickers day to day** — which manufactures returns of roughly the bid–ask wedge rather
than a constant level shift a differencing step would remove. The review dates the exposure
to **2026-07-11** on ~33k items — the same date eight of eleven live sources start, which is
why the multi-source era is 25 days deep. Every label and every A/B since then sits
downstream of this median.

**This is the review's own [UNVERIFIED] — mechanism read from code, magnitude not measured.**
It is a one-query check, and the review puts it ahead of everything else in the document for
that reason. Until the query is run, nothing here is established except that the code path
has no source filter.

### 2. Directional accuracy is measuring the market's base rate, and the null has a published test

The review reports the realised down-rate on the ≥$1 cohort as **swinging from 32.7% to
76.9% between forecast dates**, and that on every stored date a constant call matching that
date's market beat the model. *(Sourcing note: the closest repo-measured figures are in
`2026-08-03-accuracy-is-clustered-by-forecast-date.md`, whose 7d table gives an always-down
score of **29.4% on 2025-12-01** and **76.9% on 2026-07-17** — i.e. the same 76.9% top end
but 29.4%, not 32.7%, at the bottom, on a cohort the entry does not label as ≥$1. The 32.7%
end could not be sourced to a repo entry and should be treated as the review's own number
until re-measured.)* The review names the pathology: it is the null of **Pesaran & Timmermann
(1992), *JBES* 10(4), 461–465** — a nonparametric test of directional forecast value whose
null is independence between predicted and realised sign, **not 50%**. If the market is 77%
down and the model calls down 90% of the time, raw DA looks strong and the information
content is zero.

Two extensions the review says are specifically needed here: **Blaskowitz & Herwartz (2014),
*IJF* 30(1)** for the serial-correlation-robust variant (required given carry-forward
prices), and **Getmansky, Lo & Makarov (2004), *JFE* 74(3), 529–609**, whose MA(k)
illiquidity-smoothing mechanism is what the calendar-gap lag fills and frozen sub-$1 prices
are. The tier signature is already measured here: bit-identical `actual_price == base_price`
runs **37–42% at tier 0 and 0–1.8% at every tier ≥$1**
(`docs/specs/2026-08-05-cv-cohort-parity-design.md`), and by `base_price` band it
runs **71.9% below $0.05** down to **0.2% at ≥$5** (`2026-08-07-training-item-universe.md`).

This is a scoring-module change, no retrain. It reframes every accuracy number the project
has produced, including the ones in this changelog directory.

### 3. The published Backtest Accuracy number is unpurged

`walkforward_backtest.py --purge` is **default OFF**, so the published series is unpurged;
the review counts **ten further `ab_test_*` harnesses with neither purge nor fold
clustering** (§18 L1). Thirteen `ab_test_*.py` files exist and three were fold-threaded in
`2026-08-07-training-item-universe.md`, so the count of ten is consistent. Production's
`_compute_cv_splits(purge_days=horizon)` does purge — the walkforward family diverged from it.

The review's stated cost is the event-calendar arm at h=30 going **+12.1pp unpurged →
+6.1pp purged**, half the effect being boundary overlap. *(Sourcing note: this pair does not
appear in any changelog entry — `2026-08-06-date-level-exogenous-ingest.md` does not carry
it. It is the review's own measurement and is unreplicated here.)*

The review also states the embargo as a rule rather than a purge:

```
embargo_days = H + LAG_TOLERANCE_DAYS(3) + SMOOTH_WINDOW(3) + MAX_WINDOW_SPAN_DAYS(7)
             = H + 13     ⇒ h=3 → 16d, h=7 → 20d, h=14 → 27d, h=30 → 43d
```

`_purge_overlapping_train_rows` cuts at exactly `H` — correct as a purge, **short by 13 days
as an embargo**, because the as-of lag tolerance, the 3-observation scoring median and the
7-day span those observations may cover are each a real reach-back across the boundary. At
h=30 the embargo exceeds the validation window; the review calls that the correct cost, not
a bug.

### 4. The 1.1607 Steam fee constant is synthetic, and the original validation was circular (C1)

`2026-08-06-steam-listing-backfill-and-phantom-items.md` records the ratio as "a constant
1.1607 (p10 1.1565, p90 1.1656, within-item CV 0.0021)". The review ran the diagnostic its
own §11 proposed — regress the ratio on price — against
`backend/runtime/steam_listing_history.db` (528,573 rows, 262 items) joined to the archive,
**63,767 matched pairs**. The ratio is **flat at 1.1606–1.1607 across four orders of
magnitude, IQR 0.0002**, where Steam's cent-ceiling schedule must swing from ~1.67 at $0.03
to ~1.15 at $50, and sawtooth 1.150→1.168 over a single dollar. **91.61% of pairs are the
same numbers after dividing.** The two series differ by a flat multiplier baked in upstream,
so regressing one on the other recovers 1.1607 tautologically.

Consequence: any listing-page backfill row below ~$0.50 carries a real basis error (~5% too
high at $0.10–0.25), and the bottom three deciles read ratio **0.44** — buyer below net,
which no fee schedule can produce.

### 5. 110 Doppler names are not single assets, and they are corrupting labels today

The archive keys on `market_hash_name`, which does not encode Doppler phase. The BUFF dump
already fetched daily carries a `doppler` sub-object on **110 `market_hash_name`s** — 29
base names collapsing 181 distinct `paint_index` assets. Measured 2026-08-07 (§25):

| | |
|---|---:|
| Median max/min phase ratio within one name | **3.25×** |
| p90 / max | 6.08× / **23.5×** |
| Share with >2× internal dispersion | **87.3%** |
| BUFF headline price == the cheapest phase | **95.5%** |

Worst case: `★ StatTrak™ M9 Bayonet | Doppler (Minimal Wear)` spans **$1,261 → $29,685**
under one name. Because the headline tracks the cheapest phase, the series steps whenever
*which phase is cheapest* changes — a level shift with no asset repricing. Same failure that
killed CSFloat `avg_price` (float-composition noise), different attribute. The fix is a
filter, not a feature.

## The one measured positive: expensive tiers lead cheap tiers by a day

§14, **[MEASURED HERE, 2026-08-07]** — 691 items (≥$1, ≥300 days), 2021-01-02 → 2025-12-31,
1,825 dates, **969,617 daily log returns**. Stops at 2025-12-31 deliberately, because
`source` is NULL for all pre-2026 rows while the 2026 files change source composition on
three dates.

| Direction | Lag-1 corr | Verdict |
|---|---:|---|
| Cheap → expensive | +0.043 | Inside the noise band |
| **Expensive → cheap** | **+0.213 (z = 9.1)** | **Granger incremental R² 9.0%**; stable in 4 of 5 years; survives dropping all zero-change observations **and** removing the market factor (0.122, R² 4.5%) |

Three lead-lag hypotheses were tested against one archive and one window, so the review
applies the **Harvey, Liu & Zhu (2016) t > 3.0** hurdle: z = 9.1 clears it, the other two do
not and are treated as null. The folk direction (cheap leads expensive) is backwards, and
**cases do not lead skins** — the raw +0.236 at lag 1 dies on global-factor residuals
(Granger p = 0.61) and is absent in 2021–22.

Two caveats the review holds honestly. The staleness objection is **not fully closed**:
cheap skins have the highest zero-change rate (1.33% vs 0.16%) and a partially-updating
cheap index would produce this signature; the argument against is that the cheap index's own
AR(1) is only 0.125. Judged mostly real, not proven clean, with `stale_run_days` named as
the decisive test.

The candidate feature is a **date × tier** quantity, not a per-item one — which is why it
matters: it lands in the frame where the MDE is ~0.3pp rather than the 2.21–3.69pp
item-level floor.

## The strategic conclusion

Item-level idiosyncratic signal is **measured absent** — demeaning by the same-day market
factor drops directional accuracy below a constant call at every horizon
(36.7/32.7/34.6/39.0 against a majority-class baseline of 38.8/42.7/46.4/51.7,
`2026-08-06-market-relative-labels-refuted.md`). The review supplies the deepest mechanism
yet for why a dozen well-run experiments all returned null: **AK-47 | Redline (Field-Tested),
one of the most traded skins in the game, records 96 Steam sales in 24 hours**; the archive's
own distribution for the 5,542-item cohort in 2025 is a median of **69 sales/item-day
overall, 20 at $50–500, 4 at $500+**, with 60% of $500+ item-days at 1–5 sales (C5). The
thin-series problem is concentrated in the expensive tier, which is the served cohort. Per-
item daily returns are counting noise before any feature touches them.

The review's §24 ranks 32 candidate features and then tells the reader to read the shape of
the table rather than the rows: the top entries are all **date-level**, **cross-tier**, or
**structural transforms**. The liquidity family with the strongest theoretical support is
blocked on 25 days of paired bid/ask history, which at `CV_STEP_DAYS = 150` yields **zero
additional folds**. The mechanical supply-position features only fire on unannounced rule
changes. Its conclusion:

> there is no item-level feature left worth adding, and the directions that remain open are
> aggregation, cross-tier structure, and measurement.

Rank 0 — the expensive→cheap lead — is the exception that makes the point: the one measured
positive is a relationship *between tiers*, not a property of an item.

The corollary the review is equally firm about (§8): at a **+16.1% Steam round trip** (CSFloat
and DMarket +2.0%, Skinport +8.7%) against a **20.9% median bid–ask spread** ranging **35.5%
sub-$1 to 5.2% at $1000+**, and with the Steam 7-day market lock plus July-2025 trade
protection meaning **no horizon under ~8 days is executable at all**, a model at 49–53% on a
3-class problem has no economic value as a trading signal — and no realistic accuracy gain
closes that gap. This is an information product. That is a positioning statement, not a
result, but it is the frame the rest of the review is written in.

## This reopens `accuracy-opportunities.md`

That document carries a 🛑 stop banner dated 2026-07-31 whose stated reopening bar is "a data
source that is genuinely new (not inferable from price history) *and* has multi-year
history". The review does not clear that bar on its own terms — it proposes almost no new
sources — and reopens it on a different one: **the binding constraint it names is
measurement, not input data.** The banner's argument rests on an MDE of 1.15pp at 3d and
2.76–7.13pp at 7d/14d/30d from `ab_test_price_primitives.py`; the review's frame is that the
metric those MDEs were computed against is the wrong metric, that three of the refutations
underneath the banner have intervals that are wrong, and that the date-level MDE is **~0.3pp
— the exact figure the banner declares unreachable** ("you would need an MDE near 0.3pp — a
~9x reduction … it is not reachable"). It is reachable at a different grain.

A pointer has been added above the banner in that file. The banner text is left intact.

## Three published refutations whose intervals need re-deriving before citation

The review's closing Appendix restates what `2026-08-07-paired-mde-fold-clustering.md`
established — `paired_mde` clustered on `forecast_date` rather than `fold_id` until
2026-08-07, producing intervals that were too narrow — and names the three headlines
downstream of it:

- the **CSFloat basis** refutation (`2026-08-06-csfloat-basis-refuted.md`)
- the **ByMykel metadata** refutation (`2026-08-06-bymykel-metadata-refuted.md`)
- the **training-breadth** headline (`2026-08-06-breadth-beats-depth-item-age-does-not.md`)

**None has been re-derived.** The review's judgement is that they are probably still correct
in sign — they were null results, and the bug inflated significance rather than manufacturing
it — but the intervals are wrong and any future citation should say so. Note also that §11
supplies an independent *mechanism* for the CSFloat null that does not depend on the
interval at all: **CSFloat/BUFF is 0.976–1.008 across every tier**, so a CSFloat-vs-BUFF
basis feature has ~2% of range to work with.

## What was deliberately not done

- **Nothing was verified against the code or the archive.** This entry is a record of a
  document, not a measurement. C2 in particular is the review's own [UNVERIFIED], and the
  gating query has not been run. No claim here should be cited as this repo's finding until
  it is re-measured here.
- **No re-derivation of the three Appendix intervals.** The re-runs cost full sweeps and were
  already tracked as open in `2026-08-07-paired-mde-fold-clustering.md`; this entry adds a
  second source saying so, not a result.
- **The review file itself was not edited**, including the internal inconsistency noted
  below. It is a dated external artifact; corrections belong in this changelog and in the
  next-steps list, not in the source.
- **The `|r| < 0.002` figure was not corrected across the docs.** C4 measures the pooled
  within-item correlation of trailing-30 volume z-score with forward returns at **+0.019
  (7d) / +0.034 (30d)** on 4.46M rows 2023–2025, item-fixed-effects identical, and the $1–10
  tier at **+0.080** at 7d — **10–40× larger** than the `|r| < 0.002` figure that
  `docs/references/data-sources.md:108` and eight other docs rest on. C4 does **not** overturn
  the audit's conclusion (everything is still r² < 0.15%, economically trivial), only its
  stated reasoning. Correcting nine files is a separate pass and is listed as still open
  rather than done half-way here.
- **The review's Tier 2 and Tier 3 items were not re-ranked.** The next-steps list preserves
  the review's own ordering from "If I were building this myself", cross-referenced against
  §10's tiers, rather than substituting a judgement this entry has no measurement to support.
- **No spec or plan was written.** Step 1 of the next-steps list is a single query; writing a
  design document ahead of it would be work done downstream of a question that is not yet
  answered.

## Verification

None, in the sense this changelog usually means. No tests were added, no commands were run,
no production surface was checked. Every figure above is quoted from
`docs/research/2026-08-07-cs2-forecasting-research.md` and was grepped from that file rather
than restated from memory; the confidence marks (`[MEASURED HERE]`, `[UNVERIFIED]`) are
carried through from the source.

Where a figure originates in this repo's own changelog — the 2.21–3.69pp item-level MDE, the
+3.50pp [+1.56, +5.98] ≥$1 result at 30d, the 36.7/32.7/34.6/39.0 market-relative accuracies
— the citation is to the entry that measured it, not to the review.

**Two figures the review attributes to this repo's record could not be sourced to it**, and
are flagged inline above: the **32.7%** low end of the down-rate swing (the repo's nearest
table reads 29.4%), and the **+12.1pp → +6.1pp** purge effect (no changelog entry carries
it). Both are treated as the review's own, unreplicated measurements.

## One internal inconsistency in the review, reported not fixed

§1 quotes **Accuracy (≥$1, prod) 48.4 / 49.4 / 50.8 / 46.7%**. `MIN_FORECAST_DATES = 20`
(`backend/backtest/scoring.py`) and the banner in `docs/README.md` say no production DA
figure is currently quotable, because live cohorts span 1–2 distinct forecast dates and every
horizon reports NO HEADLINE. The review's own §21 restates that gate ("≥20 forecast dates
accumulated") as an MVP entry criterion, and §17 records that h=30's 5,461 usable rows all
come from a single backdated date. **Treat the §1 prod row as not quotable.** The CV row in
the same table (49.9 / 49.0 / 51.1 / 53.4% at 3/7/14/30d, ≥$1) is an offline number and is
not comparable to production DA in any case.

## Still open

- **The gating query.** Is `aggregator_buff163_buy` in the consensus median, and with what
  magnitude? Step 1 of `docs/research/2026-08-07-next-steps.md`. Everything downstream of
  2026-07-11 is conditional on the answer.
- **The three Appendix intervals**, unchanged from `2026-08-07-paired-mde-fold-clustering.md`.
- **The run-to-run divergence in the A/B harness** (`mean_diff_pp` −0.1581 → −0.0026 on
  identical commands). Still unexplained; the review's §15 offers the best candidate cause
  yet — a changing ask-source set with no `n_ask_sources` column to detect it — and notes the
  mean market return reads **−31.6% on 2026-03-22** and **+17.4%/−17.8% on 2026-07-09/10**
  against ±0.5% on a normal day. That is a hypothesis, not a diagnosis.
- **The `|r| < 0.002` correction** across nine documents.
- **`docs/README.md` and `docs/architecture/` were not otherwise reviewed against this
  document.** Only the Research index in `docs/README.md` was touched.

## Related

- `docs/research/2026-08-07-cs2-forecasting-research.md` — the review itself
- `docs/research/2026-08-07-next-steps.md` — the tracked, gated action list derived from it
- `docs/research/accuracy-opportunities.md` — reopened; stop banner intact with a pointer
- `docs/changelog/2026-08-07-paired-mde-fold-clustering.md` — the interval bug the Appendix
  turns on
- `docs/changelog/2026-08-07-training-item-universe.md` — the +3.50pp ≥$1 result the review
  calls the one surviving positive, and (§18 L2) suspects of full-sample selection
- `docs/changelog/2026-08-06-market-relative-labels-refuted.md` — the decisive experiment the
  review's one-paragraph answer rests on
