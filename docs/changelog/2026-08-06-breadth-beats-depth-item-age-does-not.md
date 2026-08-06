# Breadth beats depth at a fixed row budget, and a bare date column beats both — because the date column is the market factor

**Date:** 2026-08-06
**Change:** two new harnesses, `backend/scripts/ab_test_training_breadth.py` and
`backend/scripts/ab_test_item_metadata.py`. No production code changed, no retrain, no
change to `models/saved_models/`.
**Bears on:** `docs/changelog/2026-08-06-data-acquisition-ranking.md` — this is the
measurement its Tier-1 case and its `atalantus` rejection both rested on but never had.

> **Amended 2026-08-06, later the same day.** A follow-up run of
> `ab_test_item_metadata.py --market-relative` **reverses this entry's item-age
> conclusion** and **confirms its date-column diagnosis**. The item-age refutation below
> is wrong as stated; the corrected reading, the numbers behind it and the residual
> uncertainty are in **§ Follow-up: market-relative labels absorb the date effect and the
> item-age refutation does not survive**. The filename's `item-age-does-not` slug predates
> that correction and was left in place so existing cross-references keep resolving.

Two A/B experiments, run as one investigation into what to ingest next:

1. **Breadth at a fixed row budget pays +1.0 to +1.3pp at 14d and 30d** — the acquisition
   ranking's central premise, now measured rather than argued. It **saturates by 350
   items**, which cuts the Steam listing backfill's projected payoff well below 1pp.
   Unaffected by the amendment; breadth was not re-measured under market-relative labels.
2. ~~**Item age is not worth ingesting for accuracy.**~~ **Withdrawn.** Under raw labels
   age's marginal contribution over the static metadata columns is +0.73pp at 7d and
   **−0.69pp at 30d**, which is what the refutation rested on. Under market-relative
   labels the full 9-column bundle beats the 7-column static subset at **all four**
   horizons, so the recommendation flips from "ingest the static columns, skip age" to
   **"ingest the whole bundle"**. See the follow-up section.
3. **A single column holding nothing but the calendar date buys +11.21pp at 30d** — larger
   than every feature result this project has recorded, and not a feature result at all.
   **Now attributed:** demeaning the label by the realized market factor collapses it to
   **+0.53pp [−0.08, +1.13]**. It was the market term.

## Shared method

Both harnesses reuse the corrected pattern from `ab_test_volume_features.py`
(`2026-08-06-volume-ab-and-harness-defects.md`), so the two results are directly comparable
to each other:

* **Universe:** deep ≥$1 items — median `mean_price` ≥ $1, ≥180 distinct days, first seen
  before 2026-01-01 — on the single continuous `aggregator_sync` series. 870 items,
  1,444,230 price rows.
* **Features:** production's exact set. `SHELVED_FEATURES` removed, then
  `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` applied, giving **33 columns — the same
  33 that `2026-08-06-paired-retrain-measures-no-gain.md` reports for the shipped tree**.
* **Metric:** DA(strict, ≥$1); flat-actual rows excluded, which is the fix that turned the
  volume harness's 31pp of `sign(0)==sign(0)` free hits into a real number.
* **Folds:** 2/3 split point, 21-day validation window, 60-day stride → **25–26 folds over
  523–546 dates**.
* **Significance:** `backtest/paired_mde.py::paired_da_difference`, paired on
  `(item_id, forecast_date)` with dates as the cluster unit.
* **Evaluation is on 150 held-out items in no arm's training set**, the identical rows in
  every arm. `SPLIT_SEED` and `SAMPLE_SEED` match across both scripts, so it is the same
  150 items in both experiments.

## Breadth beats depth at a fixed budget — and saturates by 350 items

The Tier-1 case for the Steam listing-page backfill is that it *lowers* rows/item
1,341 → 923 and so raises `target_items` 521 → 758; the rejection of
`atalantus/buff-price-history-archive` is the mirror image (rows/item 1,341 → 1,641,
`target_items` 521 → **426**). Both rest on "more items at a fixed budget beats more rows
per item", which had never been measured.

Nested arms — `wide ⊇ mid ⊇ narrow` — all at **200,000 training rows per fold**: `narrow`
150 items (~1,423 rows/item), `mid` 350 (~491), `wide` 700 (~264). Plus
`wide_unbudgeted`: the same 700 items with **no row cap** (~748,000 rows/fold, ~1,518
rows/item), to separate "breadth helped" from "more data helped".

Paired diff vs `narrow`, held-out items, 95% CI, percentage points:

| h | mid (350) | wide (700) | wide_unbudgeted |
|---|---|---|---|
| 3d | −0.06 [−0.58, +0.47] | +0.05 [−0.58, +0.72] | +0.27 [−0.05, +0.59] |
| 7d | −0.71 [−1.33, −0.07] | −0.16 [−0.95, +0.62] | −0.83 [−1.25, −0.47] |
| 14d | **+1.18 [+0.58, +1.83]** | **+1.28 [+0.61, +1.96]** | −0.59 [−0.96, −0.22] |
| 30d | **+0.73 [+0.16, +1.36]** | **+1.06 [+0.46, +1.70]** | −0.15 [−0.86, +0.53] |

MDE 0.32–0.78pp. `narrow` absolute DA: **54.10 / 52.73 / 60.31 / 52.89**.

Breadth pays at 14d and 30d, is null at 3d, and is inconsistent at 7d — `mid` significant
negative, `wide` null, non-monotone, so noise. The decisive column is the last one:
**`wide_unbudgeted` is worse than `wide` at 7d, 14d and 30d despite 5.8× the rows.** What
pays is item diversity per row, not row count. That is direct support for the ranking's
central claim and for rejecting depth-only sources — the `atalantus` row now has a
measurement behind it, not just an arithmetic argument about `target_items`.

**The qualifier that matters for the Steam backfill: the gain saturates.** 150 → 350 items
buys +1.18pp at 14d; 350 → 700 adds only +0.10pp. Production sits at ~521 items /
~1,343 rows/item at the 700K budget — that is `narrow`'s rows-per-item — and the backfill
would move it to ~923 rows/item, i.e. **between `narrow` and `mid`**. Interpolating the
measured arms (an interpolation, not a measurement) puts the honest projection at roughly
**+0.5 to +0.8pp at 14d/30d and nothing at 3d/7d**: above this instrument's floor, below
the 1pp bar the project has used elsewhere.

### The "just train on the excluded 2026 items" alternative cannot be run

The tempting cheaper move is to widen training onto the ~36,000 items the deep cohort
excludes. Measured on the `aggregator_sync` series, **no 2026-first-seen item has more than
22 distinct days**, because that source only widened past the deep cohort on 2026-07-09. On
the all-source series there are 15,023 shallow ≥$1 items with ≥90 days, but **11,988 sit at
exactly 128 days and 14,940 share `min_day = 2026-03-22`** — one identical window, not a
distribution, leaving ~3 disjoint folds inside 2026. That is why experiment 1 varies the
breadth axis inside the deep cohort instead of reaching for the shallow one.

## ByMykel metadata under raw labels: the static columns pay a little, item age reads inconsistent

*The conclusions in this section are the raw-label reading. The follow-up section re-runs
the same arms with the label demeaned by the realized market factor and reverses the
item-age half. The numbers here are correct as measured and were reproduced exactly by the
follow-up's control arm; it is the interpretation that changed.*

Ingested `crates.json` and `skins.json` plus **10 further dumps from the same repo**
(`stickers`, `sticker_slabs`, `graffiti`, `collectibles`, `highlights`, `music_kits`,
`patches`, `keychains`, `agents`, `collections`).

Arms, trained on 500 items (`N_TRAIN_ITEMS`) at the 200K budget, evaluated on the 150
held-out items: `baseline` (33 cols), `treatment` (+9 metadata), `age_only`
(+`item_age_days`, `rarity_meta_rank`), `static_only` (+the 7 columns with **no date
component**), `placebo` (all 9 permuted per fold in train and val), and `date_proxy`
(+ **one** column holding the raw date ordinal and nothing else).

The two control arms exist for a specific reason. **`item_age_days` is (observation date −
first sale date), which for a fixed item is the calendar date plus a constant**, and the
folds are ordered in time — so it is a date proxy as much as an item property.
`static_only` and `date_proxy` separate item metadata from that calendar term, and
`--no-early-stop` exists because early stopping scores on the very rows being measured, a
channel a date-carrying feature can exploit more than baseline can.

Held-out items, paired vs `baseline`, **no early stopping** (the honest configuration):

| arm | 7d | 30d |
|---|---|---|
| placebo | +0.08 [−0.17, +0.33] | −0.03 [−0.26, +0.18] |
| static_only (7 cols, no date) | +0.70 [+0.02, +1.33] | **+1.92 [+1.29, +2.57]** |
| age_only (age + rarity) | +1.43 [+0.94, +1.94] | +1.23 [+0.65, +1.74] |
| treatment (all 9) | +1.37 [+0.56, +2.15] | +2.68 [+2.00, +3.38] |
| **date_proxy (date ordinal only)** | +0.95 [−0.64, +2.45] | **+11.21 [+9.30, +13.01]** |

With early stopping the same contrasts read static_only +0.77 / +0.07, age_only
+1.13 / +1.77, treatment +1.74 / +4.04, date_proxy +2.15 / **+10.46** — every arm that can
use the leak gains from it, which is why the table above is the one to read. At 3d and 14d
only the four original arms ran (see provenance): treatment +0.14 [−0.38, +0.68] at 3d and
+1.27 [+0.73, +1.79] at 14d, age_only +0.07 / +0.21, placebo −0.57 / +0.04.

**Item age's marginal contribution is sign-inconsistent under these labels.**
`static_only` already contains rarity, so age's marginal contribution is `age_only` minus
`static_only`: **+0.73pp at 7d and −0.69pp at 30d**. This entry originally read that as
"item age is not worth ingesting for accuracy", and **that conclusion is withdrawn** — the
same contrast under market-relative labels is +0.27pp at 7d and −0.73pp at 30d, still
sign-inconsistent, but the full bundle containing age beats the static subset at every
horizon. See the follow-up. What remains true here is that `item_age_days` carries a
calendar term, and that the ranking's argument that item age "is not inferable from a
truncated price history" is true without by itself making it useful.

**Static item metadata is a small real effect**: +0.70pp at 7d and +1.92pp at 30d on
held-out items, CIs excluding zero, `placebo` at ~0 so it is not capacity inflation, and no
date component so it is not the calendar. This sits against the "Category/collection
features … **0pp** ✅ tested" row in `accuracy-opportunities.md` and against the
85-features-within-fold-noise result that `FEATURE_GROUP_ALLOWLIST` rests on. The plausible
difference is coverage: `item-metadata.parquet` carries **rarity NULL on 4,296 of its 8,691
rows** and has no crate or collection at all, where the ByMykel join has rarity on
**99.5%** of the training universe.

### The ingest is not the two-call one-shot the ranking describes

Coverage on the 870-item universe: rarity **99.5%**, item age **87.1%**, crate identity
**69.1%**, collection identity **81.0%**, float caps and StatTrak/Souvenir flags **73.7%**.
On the wider ≥$1 served cohort (26,428 items) item age reaches **72.5%** — but only
**45.7% from skins + crates alone**. The 10 auxiliary dumps are therefore not optional, and
the ranking's "crates.json + skins.json, ~14 MB, two calls" understates the job. Three
further costs, all measured while building the join:

* **A `market_hash_name` parser is required.** ByMykel names are base names without wear or
  StatTrak prefix; `backend/models/steam_types.py` parses the *type* string and cannot do
  this.
* **Item age is ambiguous** — more than one candidate date — for 0.9% of the training
  cohort and **11.1% of the served cohort**, entirely knives and gloves. It is derived from
  `min(crate.first_sale_date)` with `collections.json`'s `release_date` as fallback.
* **Three date formats appear in a single file**: `2013/09/20`, `2024-01-16`, and
  un-zero-padded `2014-2-19`.

## The largest number here is the calendar, not a feature

One column containing only the date ordinal buys **+11.21pp at 30d** and +0.95pp
(unresolvable) at 7d, dwarfing every feature effect this project has measured. Frame it
carefully — and the follow-up section has since identified what it is:

* It is almost certainly the same phenomenon as
  `2026-08-03-accuracy-is-clustered-by-forecast-date.md` and the "a constant always-down
  call beats the model on every stored date" finding. A date feature lets the model bet
  that the recent market-wide regime persists — a **market-timing bet, not item-level
  skill**.
* What it establishes is that **the 30d baseline leaves a large market-direction term
  unmodelled**, and it puts a measured size on that term on this universe.
* It is **not** a recommendation to add a date column. Nothing here was shipped.
* `backend/models/market_factor.py` (uncommitted, appeared in the working tree at 14:46
  today) is a principled attack on exactly this term — a forecast market index applied per
  horizon, pre-registered in `docs/superpowers/specs/2026-08-06-market-relative-labels-design.md`
  (commit `bc550e0`) with a plan at `docs/superpowers/plans/2026-08-06-market-relative-labels.md`
  (`4e6b11e`). That spec's **pre-registered** bar is > +2pp on paired
  `classifier_accuracy_ge1` at two or more horizons and not worse than −1pp at any. This
  result is independent support for that line of work, not a prediction that it clears the
  bar: demeaning a label is a different mechanism from betting on regime persistence, and
  the two numbers are on different instruments. **This last bullet is now partly settled:**
  applying that work's label transform inside this harness removes the +11.21pp almost
  entirely (§ follow-up). Separately, the market-relative *production* hypothesis was
  refuted the same day on its own pre-registered rule
  (`2026-08-06-market-relative-labels-refuted.md`) — attributing the date term and
  adopting the relabelling are different questions, and only the first went the way this
  bullet anticipated.

## Follow-up: market-relative labels absorb the date effect and the item-age refutation does not survive

Run later on 2026-08-06 with the same `ab_test_item_metadata.py`, now carrying a
`--market-relative` flag that demeans the target by the realized market factor exactly as
production's directional classifier does when `market_relative_labels` is on:
`build_market_index` over the price frame, `market_factor_for_horizon` joined per date,
then `ItemForecaster._demean_returns` on **both** train and val labels. Same universe, same
folds, same held-out 150 items, same 33-column feature set (185 engineered rather than 181
— the four `market_factor_*d` columns, which the allowlist correctly keeps out of
`feature_cols`; they are label inputs).

**Attribution is clean.** A raw-label control was re-run on the new frame and reproduced
the pre-`market_factor` numbers **exactly**: 7d baseline / treatment / date_proxy
53.24 / 54.60 / 54.18 and 30d 50.89 / 53.57 / 62.10, giving static_only +0.70 / +1.92 and
date_proxy +0.95 / +11.21 — the figures in the table above. `market_relative_labels`
defaults to `False` and the harness bypasses `build_training_data`, so the working-tree
change moved nothing on its own. Every difference below is the label transform.

Market-relative labels, held-out items, paired vs `baseline`, no early stopping, 95% CI,
percentage points:

| arm | 3d | 7d | 14d | 30d |
|---|---|---|---|---|
| baseline DA (absolute) | 57.38% | 56.89% | 58.89% | 57.80% |
| placebo | −0.21 [−0.42, −0.01] | +0.15 [−0.06, +0.36] | −0.22 [−0.44, −0.02] | −0.15 [−0.36, +0.05] |
| static_only (7 cols, no date) | +0.13 [−0.26, +0.50] | +0.54 [+0.04, +1.06] | −0.29 [−0.97, +0.31] | +1.29 [+0.63, +2.03] |
| age_only (age + rarity) | +0.11 [−0.14, +0.34] | +0.81 [+0.47, +1.15] | +0.67 [+0.35, +1.00] | +0.56 [+0.18, +0.91] |
| **treatment (all 9)** | +0.34 [−0.03, +0.72] | +0.75 [+0.27, +1.24] | +0.99 [+0.42, +1.53] | **+1.85 [+1.28, +2.47]** |
| date_proxy (date ordinal only) | +0.00 [−0.28, +0.26] | +0.60 [+0.17, +1.06] | −0.65 [−1.13, −0.23] | +0.53 [−0.08, +1.13] |

MDE 0.20–0.70pp. Market-factor coverage 97.1–98.9% of rows; market index 4,633 dates,
4,305 valid.

### The market factor absorbs the date effect almost entirely

`date_proxy` at 30d goes from **+11.21pp [+9.30, +13.01] to +0.53pp [−0.08, +1.13]** — the
CI now includes zero. At 3d it is **+0.00pp**. Only 14d moves the other way, to a
significant −0.65pp.

This is direct confirmation of the mechanism `backend/models/market_factor.py`'s own
docstring asserts — "the directional classifier is trained on `r`, whose variance is
dominated by `m` — a term the 36 price technicals carry no information about" — measured on
a separate instrument from the one that work was built for. The +11pp this entry flagged as
"not a feature result at all" **was** the market term, and subtracting it from the label
removes it. That closes the diagnosis; it says nothing about whether relabelling helps
production, which was tested separately and refuted.

### Static item metadata survives the demeaning, at about two thirds of its size

30d **+1.92 → +1.29**, 7d **+0.70 → +0.54**, both CIs still excluding zero. So part of the
raw-label static-metadata gain *was* the market term arriving through crate and collection
identity — those are coarse date proxies too, since a crate's items share a release
cohort — but the majority is not. Null at 3d and 14d in this regime (14d reads −0.29
[−0.97, +0.31], where the raw-label run had no static arm at 14d to compare against).

### The item-age refutation does not survive

The refutation rested on age's marginal contribution over `static_only` under raw labels:
+0.73pp at 7d, −0.69pp at 30d, sign-inconsistent. Under market-relative labels `age_only`
is significantly positive at three of four horizons — **+0.81 / +0.67 / +0.56** at
7d/14d/30d, CIs excluding zero — and the decisive practical point is that **`treatment`
(all 9 columns, age included) beats `static_only` at all four horizons**: +0.34 vs +0.13,
+0.75 vs +0.54, +0.99 vs −0.29, +1.85 vs +1.29. The recommendation flips from "ingest the
static columns, skip age" to **ingest the whole bundle**.

Two limits on how far that can be pushed, both of which the earlier framing got wrong in
the other direction:

* Age's **marginal over `static_only`** is still sign-inconsistent — −0.02 / +0.27 / +0.96
  / −0.73 at 3d/7d/14d/30d.
* `age_only` vs `static_only` is **not a clean marginal**: the two arms differ by six other
  columns as well as by age, and **no arm isolates age inside the full bundle**. The
  treatment-vs-static_only differences quoted above are differences of two paired diffs,
  each paired against `baseline` rather than against each other, so they carry no CI.

The honest form is therefore: **the full bundle dominates every subset tested at every
horizon, and no clean per-column attribution is available.** That is enough to decide the
ingest question — take all 9 columns — and not enough to say age is what pays.

### The placebo is not exactly zero, which sets the floor for the small arms

The per-fold permuted placebo reads −0.21 [−0.42, −0.01] at 3d and −0.22 [−0.44, −0.02] at
14d — CIs excluding zero, i.e. adding nine columns of noise costs a fifth of a point. Read
the small positives against a null band of roughly −0.22 to +0.15 rather than against 0.
That does not touch the 30d and 14d treatment effects (+1.85, +0.99), but it does mean 3d
`treatment` at +0.34 is about one placebo magnitude from nothing.

### Caveats that bound the whole follow-up

* **Absolute DA is not comparable across the flag.** Market-relative DA scores
  *idiosyncratic* direction; raw DA scores *absolute* direction. The 30d baseline rising
  from 50.89% to 57.80% is **a change of question, not a 7pp improvement**. Recovering a
  served absolute call requires rebuilding it through `forecast_market_factor`, and its
  accuracy depends on how well `m̂` forecasts `m` — not measured on this instrument.
* **These numbers do not argue for enabling the flag.** The same label transform, read
  through the live production retrain path, was killed the same day by its own
  pre-registered rule 1: `relative_accuracy_ge1` came in at 36.7 / 32.7 / 34.6 / 39.0
  against majority baselines of 38.8 / 42.7 / 46.4 / 51.7
  (`2026-08-06-market-relative-labels-refuted.md`). That run and this one use different
  metrics — 3-class agreement with a matched flat band there, DA(strict, ≥$1) with
  flat-actual rows excluded here — different cohorts (99 items on the live path vs 870
  here) and different feature counts, so they are not reconcilable arm-to-arm, and no
  attempt was made to reconcile them. `--market-relative` is used here **as an attribution
  instrument**, not as a candidate configuration.
* `market_relative_labels` is **uncommitted work in the tree and defaults to off**. These
  figures describe the regime it would create if enabled, not current production.
* Still **held-out-item CV on the 870-item deep ≥$1 universe**, still not production DA,
  still no retrain and no change to `models/saved_models/`.
* **The raw-vs-market-relative comparison exists only at 7d and 30d.** The 3d/14d
  raw-label figures with the full six-arm set were never measured — that run failed on the
  mid-session `models.market_factor` import (see provenance) and was not retried.
* The market index here reports **4,305 of 4,633 dates valid (92.9%)**, against 1,451 of
  1,455 (99.7%) on the production 1460-day window. This frame reaches back further than
  that window, so a thinner early ≥$1 cross-section is the obvious explanation, but it was
  **not verified** and the invalid dates were not enumerated.

## What was deliberately not done

* **No production retrain, no change to `models/saved_models/`, nothing written to
  `price-archive/`.** The metadata dumps and the join table live only in a scratchpad.
* **The ByMykel collector was not built.** The static-metadata effect is real but +0.70 to
  +1.92pp on a held-out CV instrument, against an ingest that needs a name parser, 12 dumps
  and a three-format date normaliser. That trade is a decision for whoever picks up Tier 2,
  not something this measurement settles.
* ~~**`item_age_days` is not recommended even though `age_only` reads +1.43pp at 7d.**~~
  **Superseded by the follow-up** — the full bundle including age dominates the static
  subset at all four horizons under market-relative labels, so the ingest recommendation is
  all 9 columns. What was not done is the arm that would isolate age *inside* the bundle
  (`treatment` minus one column); without it there is no per-column attribution, and the
  entry does not claim one.
* **The 3d/14d full-arm diagnostic was not retried after it failed.** See provenance. This
  is why the raw-vs-market-relative contrast is available only at 7d and 30d.
* **The follow-up did not re-run experiment 1 (breadth) under market-relative labels.**
  The breadth numbers therefore stay raw-label numbers, and the saturation qualifier the
  acquisition ranking now carries rests on the raw-label reading.
* **No arm was run with the market factor as a *feature*.** It is a label input built from
  other items' future prices; using it as a feature would be leakage, and
  `_select_feature_cols` excludes it for that reason.
* **The shallow-2026 breadth arm was not attempted** — the fold structure does not exist
  (§ above), so it would have produced a number with ~3 disjoint folds behind it.
* **Neither harness's default early-stopping behaviour was changed.** It is inherited from
  `ab_test_volume_features.py`. Experiment 1's arms share a feature set, so the leak is
  symmetric; experiment 2 reports both configurations because it is not.

## Provenance, and what these numbers are not

* **All successful runs in the original pass used the working tree between 14:26 and 14:45
  on 2026-08-06**, before `models/market_factor.py` and the accompanying
  `models/forecaster.py` change landed at **14:46**. The 3d/14d full-arm diagnostic failed
  with `ModuleNotFoundError: No module named 'models.market_factor'` and **was not
  retried, deliberately** — re-running would compare against a moving target. Every figure
  in the raw-label sections is against the pre-market-factor tree.
* **The concern that "the baselines will move once that work lands" did not materialise,
  and is now measured rather than feared.** The follow-up's raw-label control on the
  post-`market_factor` tree reproduced the earlier 7d and 30d numbers to the digit — 53.24
  / 54.60 / 54.18 and 50.89 / 53.57 / 62.10. `market_relative_labels` defaults to `False`
  and this harness bypasses `build_training_data`, so the flag's arrival changed nothing
  the harness reads.
* These are **held-out-item CV numbers on an 870-item deep ≥$1 universe, not production
  DA**, and not comparable to the ≥$1 tier figure the pipeline reports.
* They are also **not comparable to the volume A/B's 57–67% DA**: different universe, 33
  features rather than 143, and evaluation on items no arm trained on. The absolute level
  here (`narrow` 54.10 / 52.73 / 60.31 / 52.89) is a generalisation-to-new-items score.
* Both scripts are uncommitted at the time of writing.

## Docs updated in this pass

* `docs/changelog/2026-08-06-data-acquisition-ranking.md` — Tier-1 item 1 now carries the
  measured saturation qualifier and the interpolated projection instead of an unqualified
  breadth argument; Tier-2 item 4's item-age claim is corrected and the two-call ingest
  cost restated; the `atalantus` rejection and the "more data does not move accuracy on its
  own" section now cite the `wide_unbudgeted` arm.
* `docs/research/accuracy-opportunities.md` — the "Category/collection features … 0pp ✅
  tested" row is qualified with this result and the cohort it was measured on.

Updated again in the follow-up pass: the item-age correction and the market-factor
attribution were written back into **both** of those files.

## Still open

* ~~**Whether the static-metadata effect survives the market-factor change.**~~
  **Answered:** it does, at about two thirds of its size at 7d and 30d, null at 3d and 14d.
* **Whether item age contributes anything on its own.** The full bundle wins, no arm
  isolates age inside it, and age's marginal over `static_only` is sign-inconsistent under
  both label regimes. A leave-one-column-out arm on `treatment` would settle it.
* **Whether it survives the full production feature set.** These arms run 33 columns; the
  earlier 0pp read was taken with ~85 features in play, where dilution is a live mechanism
  (`2026-08-06-volume-ab-and-harness-defects.md` measured uninformative columns costing
  1.3–1.6pp).
* **The Steam listing backfill's egress block**, unchanged by any of this. The reason to
  finish it is now quantified at sub-1pp for 14d/30d accuracy, which weakens the accuracy
  argument and leaves the breadth-of-coverage argument standing on its own.

## Related

* `docs/changelog/2026-08-06-data-acquisition-ranking.md` — the ranking these two
  experiments test
* `docs/changelog/2026-08-06-volume-ab-and-harness-defects.md` — the harness pattern, the
  ≥$1 universe and the strict metric both of these inherit
* `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md` — the phenomenon
  `date_proxy` is a second reading of
* `docs/superpowers/specs/2026-08-06-market-relative-labels-design.md` — the pre-registered
  attack on that term
* `docs/changelog/2026-08-06-market-relative-labels-instrument.md` — `market_factor.py`,
  `_demean_returns` and the flag the follow-up borrows
* `docs/changelog/2026-08-06-market-relative-labels-refuted.md` — the same transform read
  through the production retrain path, killed on its own pre-registered rule; read it
  before treating the follow-up's demeaned baselines as an improvement
* `docs/references/data-inventory.md` — the coverage audit behind the breadth/depth framing
* `docs/changelog/2026-08-06-paired-retrain-measures-no-gain.md` — the source of the 33
  columns these arms baseline on
