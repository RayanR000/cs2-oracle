# The ByMykel metadata ingest is built, and the bundle measures better than the scratchpad build did

> **Superseded on the production path, 2026-08-06.** Every CV number below stands as
> measured, and none of it survives contact with the model: the forecaster's own
> permutation test reports that shuffling this bundle costs **−0.24 / −0.69 / −0.06 /
> +0.79pp** at 3/7/14/30d. The model does not use these columns. Read
> `2026-08-06-bymykel-metadata-refuted.md` before citing anything here.

**Date:** 2026-08-06
**Change:** new `backend/scripts/ingest_bymykel_metadata.py` and
`backend/tests/test_ingest_bymykel_metadata.py` (52 tests). Writes
`price-archive/item-metadata-bymykel.parquet` and its code book. **No production
code changed, no allowlist change, no retrain, no change to `models/saved_models/`.**
**Bears on:** `2026-08-06-data-acquisition-ranking.md` Tier-2 item 4, and
`2026-08-06-breadth-beats-depth-item-age-does-not.md`, whose metadata table was
built in a scratchpad and never committed. This is that builder, made reproducible.

The ranking put `ByMykel/CSGO-API` in Tier 2 and the follow-up measurement made it
the only acquisition candidate with a placebo-controlled positive effect. Both rest
on a metadata table that no longer existed. Rebuilding it from scratch was therefore
also a reproduction test of the numbers they report — and the reproduction is exact
where it should be.

## What was built

One script, twelve dumps, one output file keyed on `item_slug`. It writes a **new**
Parquet rather than extending `item-metadata.parquet`, which
`scripts/backfill_supply_metadata.py` owns and would clobber on its next run.

Nine columns: `item_age_first_sale_date`, `item_age_ambiguous`, `rarity_meta_rank`,
`is_meta_stattrak`, `is_meta_souvenir`, `float_meta_min`, `float_meta_max`,
`type_meta_crate_id`, `type_meta_collection_id`. 45,362 market hash names, 0.5 MB.

`item_age_days` is deliberately **not** stored. It is (observation date − first sale
date), a property of a row and not of an item; a stored "days since sale as of today"
column would freeze the calendar into a per-item constant. The consumer derives it.

`rarity_meta_rank` reuses `models/steam_types.py::RARITY_RANK` rather than inventing
a scale, because the model already carries `rarity_ordinal` on that one and a
second, differently-numbered rarity column would be silently incomparable to it.
Contraband and Default are the two ranks that map missed; Contraband is placed above
Covert, Default at 0.

### The name parser the ranking called for is not the one that was needed

`2026-08-06-breadth-beats-depth-item-age-does-not.md` records that "a
`market_hash_name` parser is required" because ByMykel names are base names without
wear or StatTrak prefix. **Ten of the twelve dumps carry `market_hash_name`
directly** and need no parsing at all. Only `skins.json` does not, and the right
direction there is *forward*: expand each skin's base name across its `wears` and its
StatTrak/Souvenir variants into market hash names.

Parsing the other way — stripping wear and StatTrak off archive names — was rejected
after measurement, not on taste. `models/item_parser.py::parse_item_name` gets knives
wrong: `★ StatTrak™ Karambit | Fade (Minimal Wear)` returns
`weapon="StatTrak™ Karambit"`, and it also reports `★ Hand Wraps` as `is_knife`
rather than `is_glove`. Forward expansion never touches that code path.

## Three date sources the scratchpad build appears to have missed

Coverage on the deep ≥$1 universe reproduced rarity, collection and float caps
immediately, but item age came in at 83.9% against the published 87.1% and crate
identity at 62.9% against 69.1%. Three gaps explain most of it:

1. **`crates.json` carries a reverse index.** `contains` and `contains_rare` name a
   crate's contents on 478 of 481 crates, and that is more complete than the forward
   `crates` field on each item. Both directions are now unioned.
2. **152 undated crates sit inside a dated collection.** Only 261 of 481 crates carry
   `first_sale_date`; a crate's contents went on sale when the crate did, so the
   collection's `release_date` is inherited one level down. The item-level fallback
   cannot reach these — it fires only when the *item* lists the collection, and items
   inside a crate frequently do not.
3. **A crate is itself a tradable item.** Its own `first_sale_date` is its own age.
   `Kilowatt Case` was coming out null.

With all three, item age reaches **90.8%**, above the published 87.1%. Crate identity
reaches 66.4% and remains the one column short of the published figure; the universe
here is 878 slugs against the published 870, which accounts for part but not all of
the gap. It is unexplained and recorded as such.

Five date formats appear in `crates.json`, not the three the earlier entry recorded:
`2024-01-16`, `2013/12/17`, `2014/5/2`, `2014/12/5`, and un-zero-padded days as well
as months. One normaliser handles both separators and both padding failures, and
returns None rather than guessing on anything else.

## The encoding defect, which is worth recording because it is silent until it is fatal

The first build encoded crate and collection identity as 31-bit blake2b hashes,
reasoning that hashing keeps codes stable when a new crate ships where an index into
a sorted id list would renumber everything after the insertion point.

**It OOM-killed the treatment arm outright** — `EXIT=137`, after LightGBM had already
said what was wrong: *"Met categorical feature which contains sparse values. Consider
renumbering to consecutive integers started from zero."* `ab_test_item_metadata.py`
declares these columns categorical (`META_CATEGORICAL`), and LightGBM allocates over
a categorical's value **range**, not its cardinality. 322 distinct crates spread over
2³¹ is not a wasteful encoding, it is an unrunnable one.

Both constraints are real and they pull against each other, so the codes are now
**dense and persisted**: `price-archive/item-metadata-bymykel-codes.json`, append-only,
existing ids keep their code forever and new ids get the next free one. Crate codes
now max at 480, collection codes at 109. The treatment arm runs in 17 s.

**That file is load-bearing.** Deleting it renumbers every crate and collection and
silently changes what an already-trained booster's splits mean. It should be deleted
only alongside a retrain.

## Results

`ab_test_item_metadata.py`, same 870-item deep ≥$1 universe, same 150 held-out items,
same 33-column baseline, `--no-early-stop`, paired on `(item_id, forecast_date)` with
dates as the cluster unit.

### The harness is verified identical, so the metadata arms are attributable

`date_proxy` holds one column — the raw date ordinal — and touches no metadata, so it
is a control on the frame rather than on the treatment. It reproduces the published
run **exactly** in both label regimes: **+0.95 / +11.21** at 7d/30d raw, and
**+0.00 / +0.60 / −0.65 / +0.53** market-relative. All eight baselines match to the
digit (raw 53.24 / 50.89; market-relative 57.38 / 56.89 / 58.89 / 57.80).

Every difference below is therefore the metadata table, not harness drift.

### Raw labels — the regime production scores

| arm | 3d | 7d | 14d | 30d |
|---|---|---|---|---|
| baseline DA (absolute) | 54.35% | 53.24% | 57.87% | 50.89% |
| placebo | −0.12 [−0.38, +0.12] | +0.05 [−0.21, +0.30] | **−0.29 [−0.50, −0.09]** | −0.09 [−0.30, +0.11] |
| static_only (7 cols) | +0.04 [−0.40, +0.52] | **+1.07 [+0.40, +1.72]** | −0.10 [−0.72, +0.54] | **+2.59 [+1.89, +3.30]** |
| age_only (age + rarity) | +0.17 [−0.19, +0.49] | **+0.97 [+0.46, +1.47]** | −0.39 [−0.82, +0.05] | **+1.41 [+0.84, +1.95]** |
| **treatment (all 9)** | +0.32 [−0.18, +0.82] | **+1.26 [+0.46, +2.02]** | −0.42 [−1.17, +0.31] | **+3.72 [+2.99, +4.47]** |
| date_proxy | −0.60 [−1.45, +0.30] | +0.95 [−0.64, +2.45] | **−7.34 [−9.08, −5.71]** | **+11.21 [+9.30, +13.01]** |

MDE 0.20–1.85pp.

**Significant at 7d and 30d, null at 3d and 14d.** Against the published raw-label
figures the better-populated table is worth more at 30d — treatment **+2.68 → +3.72**,
static_only **+1.92 → +2.59** — and is unchanged within noise at 7d (+1.37 → +1.26).

**3d and 14d are new.** The earlier entry records that run as having failed on a
mid-session `models.market_factor` import and deliberately not retried, so the
raw-label six-arm set existed only at 7d and 30d. Both new horizons are null, and 14d
is the weaker of the two: the treatment CI includes zero *and* the placebo reads
significantly negative at −0.29pp, so nine columns of noise cost about a third of a
point there. Read the 14d and 3d treatment numbers against a null band of roughly
−0.29 to +0.05, not against zero.

### Market-relative labels — attribution, not a candidate configuration

| arm | 3d | 7d | 14d | 30d |
|---|---|---|---|---|
| baseline DA (absolute) | 57.38% | 56.89% | 58.89% | 57.80% |
| placebo | −0.24 [−0.42, −0.02] | +0.00 [−0.20, +0.22] | −0.26 [−0.47, −0.06] | +0.01 [−0.17, +0.21] |
| static_only (7 cols) | +0.44 [+0.04, +0.82] | +0.65 [+0.13, +1.14] | −0.28 [−0.91, +0.28] | +1.26 [+0.55, +2.04] |
| age_only (age + rarity) | +0.26 [−0.00, +0.50] | +0.79 [+0.47, +1.13] | +0.59 [+0.26, +0.91] | +0.71 [+0.34, +1.06] |
| **treatment (all 9)** | **+0.51 [+0.12, +0.88]** | **+0.99 [+0.51, +1.48]** | **+1.44 [+0.89, +1.94]** | **+1.68 [+1.06, +2.35]** |
| date_proxy | +0.00 [−0.28, +0.26] | +0.60 [+0.17, +1.06] | −0.65 [−1.13, −0.23] | +0.53 [−0.08, +1.13] |

MDE 0.19–0.75pp. Against the published bundle (+0.34 / +0.75 / +0.99 / +1.85), this
table is better at 3d, 7d and 14d and slightly worse at 30d.

**Under market-relative labels the bundle is significantly positive at all four
horizons**, where the published run had 3d inside the placebo band.

**How much of the raw 30d gain is the market term.** Treatment goes **+3.72 → +1.68**
under demeaning, so roughly **45% survives** — a larger market component than the
published run's +2.68 → +1.85 (69%). That is the expected direction: crate and
collection identity are coarse date proxies, since a crate's items share a release
cohort, and this table assigns crate identity to more items than the old one did.
The 14d sign flip (−0.42 raw → +1.44 demeaned) is the same phenomenon read the other
way: raw 14d carries a large negative calendar term (`date_proxy` −7.34pp) that the
demeaning removes.

**These numbers do not argue for enabling `market_relative_labels`.** That transform
was killed on its own pre-registered rule the same day
(`2026-08-06-market-relative-labels-refuted.md`). It is used here as an attribution
instrument only, and absolute DA is not comparable across the flag — market-relative
DA scores *idiosyncratic* direction.

### The whole bundle still beats every subset

`treatment` beats `static_only` at all four horizons in both regimes (+0.51 vs +0.44,
+0.99 vs +0.65, +1.44 vs −0.28, +1.68 vs +1.26 demeaned). The earlier entry's
conclusion — ingest all nine columns, not the static subset — survives this rebuild.

The limits on it survive too. No arm isolates age *inside* the bundle, the
treatment-vs-static_only differences are differences of two paired diffs each paired
against `baseline` and so carry no CI, and age's marginal over `static_only` is still
sign-inconsistent. **The bundle dominates every subset tested; no per-column
attribution exists.**

## What this is not

* **Held-out-item CV on an 870-item deep ≥$1 universe, not production DA.** Not
  comparable to the ≥$1 tier figure the pipeline reports. Production trains on a
  ~99-item subsample (`training-uses-133-item-subsample`), and nothing here was
  measured through the live retrain path.
* **Nothing is wired in.** `_feature_group` routes these columns to `item_identity`,
  `item_metadata` and `temporal`, and `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`
  discards all three. **Not one of these columns reaches the production model today.**
  Shipping them means an allowlist change, a `MODEL_ARTIFACT_VERSION` bump and a
  retrain, and that decision is not settled by this measurement.
* **The output is gitignored.** `price-archive/` is a plain local directory
  (`.gitignore:65`); the durable archive is the separate `RayanR000/cs2-oracle-data`
  repo that only CI writes. Nothing written here is committed by this repo, so the
  artifact must be regenerated — `--offline` replays from `backend/runtime/bymykel/` —
  or published through the data repo before anything in CI could read it.
* **Cadence is manual by decision.** ByMykel is cross-sectional metadata that changes
  when Valve ships a case; a daily collector would add a failure surface to the
  silently-failing-collector family for data that moves monthly.

## Still open

* Whether the effect survives the **production** feature set and cohort. These arms
  run production's exact 33 columns, which removes the dilution objection that applied
  to the earlier 0pp category-features read, but not the cohort objection.
* **Crate identity coverage**, 66.4% against the published 69.1%, unexplained beyond
  the 878-vs-870 universe difference.
* A **leave-one-column-out arm** on `treatment` is still the only thing that would
  attribute the gain per column.
* Whether 14d's null is the horizon or the placebo floor — the placebo is significantly
  negative there in both regimes.

## Related

* `docs/changelog/2026-08-06-data-acquisition-ranking.md` — Tier-2 item 4
* `docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md` — the
  measurement this reproduces, and the source of every published figure quoted here
* `docs/changelog/2026-08-06-market-relative-labels-refuted.md` — read before treating
  the demeaned numbers as an improvement
* `docs/references/data-sources.md` — the ByMykel source row
