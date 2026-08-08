# A bid has been voting in the consensus price since 2026-07-11 — and the 2σ guard was eligible, ran, and kept it anyway

**Date:** 2026-08-07
**Plan:** `docs/research/2026-08-07-next-steps.md` step 1, the gating item ("do not start
anything below it first")
**Change:** `backend/models/forecaster.py` (new `BID_SOURCES`, filter in
`_apply_multi_source_voting`, `VOTED_CACHE_VERSION` 1 → 2),
`backend/scripts/walkforward_backtest.py::_load_all_prices`,
`backend/tests/test_bid_source_voting.py` (new, 9 cases).
**Suite:** 1,233 pass (`pytest tests/ -q`), 0 failures.

`aggregator_buff163_buy` is BUFF's `highest_order` — a **bid**. It has been median-voted
against asks as if it were one of them since it entered the archive on 2026-07-11. The
review recorded that as **[UNVERIFIED]**, mechanism read from code. It is verified, and the
measurement refuted the review's account of *how* the harm happens on all three points.

## What the bid is, measured

Window **2026-07-11 → 2026-08-04** (the local archive max), one row per
`(item_slug, day)`: **570,732 bid rows across 32,948 items**, first day exactly 2026-07-11.
That matches `docs/references/data-inventory.md`'s source table.

Level, as a multiple of the Steam print, n = **461,540 item-days**:

| | × Steam |
|---|---|
| `aggregator_buff163_buy` (**bid**) | **0.579** |
| the ask panel | 0.717 – 0.809 |
| bid ÷ BUFF's own ask | 0.800 |

The wedge is not constant across the cross-section: the bid/Steam ratio runs **0.500
sub-$1** to **0.713 at ≥$100**. A cheap item's bid is further below the asks it votes
against than an expensive one's, so the injected error is largest exactly where the price
level already makes a cent look like a return.

Displacement of the voted consensus, all bid-carrying item-days: the consensus **differs on
73.2%** of them, median **−8.0%** (IQR −13.0 / −2.6, p05 −20.1).

On the **≥$1 served cohort** — 1,398 items selected by `item_forecasts.current_price ≥ 1` at
`forecast_date = 2026-08-05`, the population `MIN_SERVED_PRICE_USD` actually shows:

- **95.4%** of item-days have a displaced consensus, median **−10.8%**
- **44.1%** of consecutive return pairs carry a spurious component **> 1pp**
- **13.6%** of return pairs have their **direction flipped**
- **1,395 of 1,398 items** are affected

13.6% of the served cohort's labels having the wrong sign is the number that makes this the
gating item. Direction is the product.

## Three findings that refuted the code-read hypothesis

This is the substance of the measurement. Every one of the three mechanisms the review named
turned out to be either wrong or minor, and the real channels are different.

### 1. The 2σ guard was not bypassed — it was too loose

The review's reasoning was that `vote()` (`models/forecaster.py:1048`) applies its >2σ
rejection **only at ≥3 sources**, so the bid slips through on thin panels. That is a correct
reading of the code and the wrong diagnosis of the data.

- Bid item-days with **n ≥ 3 sources: 99.3%**. With the full **11 sources: 55%**.
- The guard therefore ran on essentially every bid item-day — and **failed to reject the bid
  80.5% of the time**, rising to **83.3% at n = 11**.

The reason is arithmetic. An 11-source panel spanning **0.58 – 1.04×** of Steam has a σ wide
enough that a value at 0.579× sits inside 2σ of the median. The outlier mask cannot see the
bid because the ask dispersion is larger than the bid–ask wedge. **A tighter guard was never
the fix**, and any version of this change that tuned the threshold instead of removing the
source would have measured as an improvement while leaving four fifths of the damage in
place.

### 2. Inclusion flicker was not the main channel

The review's harm model was day-to-day flicker in *whether* the bid is rejected, which
fabricates a return of roughly the bid–ask wedge. Flicker is real. It is **5.6% of return
pairs**.

The dominant channel is **steady-state inclusion with a drifting bid–ask spread: 53.5% of
pairs**. When the bid is in the median on both days, the consensus still moves by the change
in the wedge, and the wedge moves. Weighted by damage the gap is wider still — steady-state
inclusion carries **~4× more of the >1pp damaged mass than flicker (105K vs 27K pairs)**.

Fixing only the flicker — e.g. by stabilising rejection — would have left most of the
injected variance untouched.

### 3. Median parity is a distinct mechanism, and it was unnamed

Adding one low value to an **even** ask panel does not move the median toward the bid
smoothly; it steps the median down one rung of the source ladder. Median displacement by
source count:

| bid item-days at | median displacement |
|---|---|
| n = 11 (55% of them) | **−6.35%** |
| even source counts | −1.55% |

This is a property of the estimator, not of the bid's level, and it is why the 11-source
panel — the *best*-instrumented item-days — is the worst affected. It also explains why the
first drafts of the tests passed before the fix (§ Verification).

## The change

`BID_SOURCES = frozenset({"aggregator_buff163_buy"})`, a module constant in
`models/forecaster.py`, framed as a **source-taxonomy fact rather than a quality filter**.
These rows are good data; they are not asks. Anything that wants the bid reads the labelled
source row out of the archive.

`ItemForecaster._apply_multi_source_voting` drops `BID_SOURCES` **before anything reads the
group**, so a bid counts toward neither the median nor the ≥3-source gate that enables the
outlier mask. The filter is `~df["source"].isin(BID_SOURCES)`, and the NaN behaviour is
load-bearing: `isin` is False for NaN, so the pre-2026 `source IS NULL` series — 13 years of
the archive — keeps voting. A NULL-unsafe filter here is the difference between excluding one
feed and deleting the history.

**An item-day whose only source was a bid now returns no row.** That is **2,338 item-days
across 217 items**. It is not a fallback to the bid, deliberately: a series whose basis
alternates between bid and ask fabricates the wedge as a return, which is the defect itself.

`VOTED_CACHE_VERSION` **1 → 2**, and this is mandatory rather than hygienic. The cache key
covers the `prices-*.parquet` fingerprint, the cutoff date and the backfill slug set — it
cannot see code. Without the bump a stale v1 frame would keep training the next model on the
displaced consensus with nothing in the logs to say so.

`scripts/walkforward_backtest.py::_load_all_prices` needed a separate fix. It never calls
`_apply_multi_source_voting` at all: it hands duplicate item-days to `engineer_features`,
which collapses them with a plain **mean**. A mean has no outlier rejection whatever, so the
bid entered the **published Backtest Accuracy gate number undiluted** — the one path where
the 2σ guard was genuinely absent rather than merely ineffective. It now reads through
`db/archive.py::prices_relation` (needed because `source` does not exist in the first file's
schema, which a raw `read_parquet('prices-*.parquet')` narrows to) and filters `BID_SOURCES`
in SQL.

Between them these cover all three production consumers of the consensus: training and
predict (`_fetch_voted_price_history`), the DB path, and **label resolution**
(`backtest/price_resolution.py::load_voted_prices`, which resolves both legs of every scored
outcome).

## Verification

**`backend/tests/test_bid_source_voting.py`, 9 cases, written test-first — and the first two
drafts passed before the fix existed.** They were built on a narrow three-ask toy panel,
where the asks sit close together, σ is small, and the 2σ mask catches the bid on its own:
the 19.5% case from § 1. Rebuilt on the **measured 11-source ladder** (0.579 / 0.717 / 0.724
/ 0.732 / 0.753 / 0.809 / 1.000 ×4 / 1.043), they fail correctly at **8.09 against an
ask-only 9.045 — a −10.6% displacement**, matching the −10.8% measured on the ≥$1 cohort.
Worth recording as a caution about this estimator specifically: **a toy panel is not a
substitute for the real source ladder**, because the failure depends on the ask dispersion,
not on the bid.

The file pins the refuted alternative too:
`test_bid_survives_the_outlier_mask_so_the_mask_cannot_be_the_fix` asserts on `vote()`'s own
arithmetic that the bid is *within* 2σ of the ladder median, so the claim in § 1 is checked
rather than only written down. `test_bid_does_not_count_toward_the_three_source_outlier_gate`
covers the gate channel; `test_untagged_pre_2026_rows_still_vote` covers the NaN behaviour;
two cases cover the label-resolution and published-gate loaders end to end against a
temp-dir Parquet archive.

**Full suite: 1,233 pass** (`pytest tests/ -q` from `backend/`), 0 failures.

**Real-archive spot check, 2026-07-15 → 2026-07-25** (11 days, 3,951,693 input rows /
450,690 item-days / 338,910 bid rows):

| | value |
|---|---|
| comparable item-days | 449,202 |
| consensus changed | **53.94%** |
| median displacement removed | **+8.46%** (IQR +2.67 / +14.58) |
| bid-only item-days dropped | 1,488 |
| on `price ≥ 1`: changed / median | 57.85% / +7.33% |

**These ≥$1 figures use a different denominator than the 95.4% / −10.8% above and do not
replicate them.** The measurement conditioned on bid-carrying item-days and defined the
cohort by `item_forecasts.current_price` on one forecast date; the spot check takes all
item-days in an 11-day window with `price ≥ 1`. Same direction, same rough magnitude,
different statistic. Neither number should be quoted as confirming the other.

## What was deliberately not done

- **The bid was not promoted to its own column.** Step 1 of the next-steps list asked for
  both. The bid remains fully recoverable as a labelled source row in the archive, and
  adding an unused column to the predict frame — which reaches ~2M rows and has OOM'd in CI
  before — buys nothing until spread features are actually built. Those features are blocked
  anyway: the paired bid/ask family has **25 days of history**, which at
  `CV_STEP_DAYS = 150` produces zero additional folds.
- **The 2σ threshold was not touched.** § 1 is the reason. The guard is not the mechanism and
  tuning it would have manufactured an apparent improvement.
- **No fallback for bid-only item-days.** 2,338 item-days across 217 items lose their only
  observation. A basis that alternates between bid and ask is worse than a gap, and the
  feature path already median-fills gaps.
- **The affected A/Bs were not re-run**, and the displaced series was not retro-corrected in
  the archive. The archive is source-labelled, so the correction is a read-time filter and
  needs no rewrite; the re-runs cost full sweeps.
- **`--purge` was not turned on by default** in `walkforward_backtest.py`. It is a separate
  item (step 5) and flipping it silently would break continuity of the published
  `lgbm-v3-clustered` series.
- **The three `ab_test_*` private price loaders were left unfiltered** — see below.

## Still open

- **`aggregator_steam_17mafo`: 2,169,483 rows, 2026-04-16 → 2026-07-10, also voted
  unfiltered, and not investigated.** It is 3.8× the bid's row count and it covers the A/B
  window *before* 2026-07-11, so if it is not a genuine ask the contaminated window extends
  four months further back than this entry establishes. `docs/references/data-inventory.md`
  describes `raw/17mafo/*.json` as a "Raw Steam scrape, already ingested", which argues the
  label is merely ugly rather than the data wrong — that is a docs claim, not a
  measurement, and it has not been checked against the values.
- **The `ab_test_*` harnesses still vote the bid.** `scripts/ab_test_regime.py` (item
  selection plus two price loaders) and `scripts/ab_test_ensemble.py` each carry a private
  copy of the `_load_all_prices` glob-and-mean pattern with no source filter. Deferred to
  **step 5**, which already has to touch those files for purge and fold clustering. Until
  then **no A/B run on those two harnesses is clean**, on top of their existing purge and
  clustering defects.
- **Every A/B result and every headline accuracy figure from 2026-07-11 to 2026-08-07 is
  downstream of the displaced consensus** and needs re-running. This entry records the
  invalidation; it does not correct anything.
- **Whether the fix moves production DA is unmeasured.** The spot check measures displacement
  of the price series, not accuracy. The next scored Backtest Accuracy run will be the first
  DA number computed on an ask-only consensus, and it will not be comparable to the stored
  series either — a level change in the input prices is exactly the kind of discontinuity the
  stored series has no way to attribute.

## Related

- `docs/changelog/2026-08-07-cs2-forecasting-research-review.md` — where this was raised, as
  §10 Tier 1 #3, marked [UNVERIFIED]. Three of its claims about this defect are corrected
  here and in the next-steps list.
- `docs/changelog/2026-08-07-archive-schema-and-keys.md` — `db/archive.py::prices_relation`,
  without which the gate loader could not see `source` at all.
- `docs/changelog/2026-07-11-multi-source-aggregator-and-cleanup.md` — the change that
  started writing `aggregator_buff163_buy`, and the origin date of the contamination.
- `docs/changelog/2026-08-06-volume-ab-and-harness-defects.md` and
  `2026-08-06-free-bulk-supply-depth-feeds-exist.md` — both treat the bid as a *future*
  spread feature with 21–25 days of history; neither noticed it was already voting.
- `docs/references/data-inventory.md` — the per-source row/item/span table these figures were
  read against.

## Docs touched

- `docs/research/2026-08-07-next-steps.md` — step 1 marked **DONE** with its outcome; the
  blanket "Status of everything below: NOT STARTED" header corrected; the three refuted
  claims in step 1's *Why* replaced with what was measured; steps 2–6 recorded as unblocked.
- `backend/AGENTS.md` — new gotcha for `BID_SOURCES` and the two loaders that needed
  separate fixes; the existing `VOTED_CACHE_VERSION` gotcha now names this as a second
  reason the bump is mandatory. (Edited outside `docs/`, which this pass is otherwise scoped
  to.)
- `docs/architecture/pipeline.md` — the 11-source-label table now says which label does not
  vote.
- `AGENTS.md` (root) — the full-suite count corrected **714 → 1,233**, and the documented
  command changed to `pytest tests/ -q`, because a bare `pytest -q` from `backend/` aborts on
  `scripts/test_social_signal.py` failing to collect against a missing local `thefuzz`. Both
  are pre-existing and unrelated to this change; noticed while recording the suite figure.
