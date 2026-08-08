# Half the catalogue had no rarity for two unrelated reasons, and rank 0 meant both "base grade" and "no idea"

**Date:** 2026-08-07
**Change:** `backend/models/steam_types.py::parse_steam_type` returns `rarity_rank: None`
for an unknown rarity instead of `0`; `backend/scripts/backfill_supply_metadata.py` gains
`_normalise_key`, `build_name_lookup`, `coalesce_rarity` and `load_bymykel`;
`backend/scripts/ingest_bymykel_metadata.py` gains `rarity_token()` and a `rarity_meta`
column. `price-archive/item-metadata.parquet` and `item-metadata-bymykel.parquet`
regenerated.
**Bears on:** `2026-08-07-training-item-universe.md` (the subsample stratifies on the rarity
string this repairs), `2026-08-06-bymykel-metadata-refuted.md` (the source used for the fill,
and the reason this claims no accuracy value).

`item-metadata.parquet` carried rarity NULL on **4,296 of 8,691 items** and a `rarity_rank`
that was never NULL. Both turned out to be true, and neither was the same problem.

## Three root causes, from classifying all 4,296 NULLs

| cause | n | share |
|---|---|---|
| slug never matched the catalog's key format | 2,344 | 54.6% |
| matched fine, but `market_catalog.db` has an empty `type` | 1,926 | 44.8% |
| genuinely absent from the catalog | 26 | 0.6% |

**1. Key format — a real bug.** `market_catalog.db` is keyed by Steam display name,
`Berlin 2019 Legends (Holo/Foil)`. The price archive keys the same item
`berlin-2019-legends-holo-foil`. `build_metadata` compared the two with a bare
`if slug in name_to_meta` — no normalisation on either side, so every slug-formatted key
missed. The data was present the whole time.

The script also loads a `db_name_map` bridge from `cs2_market.db` for exactly this purpose
and **never reads it**. Measured, it would have resolved **0** of the 2,344 anyway. Left in
place rather than widening the change, but it is dead and misleading.

**2. Empty upstream data — not a bug.** These match the catalog exactly and its `type`
column is literally empty. They are real, valuable skins: `AK-47 | Asiimov (Minimal Wear)`,
`AUG | Aristocrat (Field-Tested)`. No amount of key fixing reaches them; only a second
source does.

**3. The rank-0 conflation.** `parse_steam_type` returned `rarity_rank: 0` for an unknown
rarity, and 0 is also the true rank of `base` and `highlight`. Before this change, rank 0
held **4,920 rows of which only 624 were real**. Any consumer keying on the rank rather than
the rarity string was treating half the catalogue as the lowest tier rather than as missing.
`rarity_ordinal` (`_add_supply_features`) is such a reader — currently allowlisted out of
training, so this was latent rather than live.

## The fixes

* **`parse_steam_type`** returns `rarity_rank: None` for unknown rarity, on both return
  paths. `base` and `highlight` keep their real 0.
* **`_normalise_key` / `build_name_lookup`** index every catalog row under its literal
  `display_name` and `hash_name` *and* the normalised form of both. Slugs are already in
  normal form, so this is idempotent on them and the 4,395 exact-name hits are untouched.
  Collisions resolve in favour of the entry that carries a rarity, so normalising can only
  add coverage, never replace a populated entry with an empty one.
* **`coalesce_rarity` / `load_bymykel`** fill from `item-metadata-bymykel.parquet` wherever
  Steam left the rarity empty. Steam wins wherever it has a value — the two sources agree on
  **all 4,389** overlapping items, so precedence is about keeping the column reproducible
  from the primary source, not about trusting one more. Emptiness is tested on the rarity
  string, never on a falsy rank, because rank 0 is a real value. `rarity_rank` is cast to
  pandas nullable `Int64` so the new nulls do not promote the column to float64 and hand
  `items.rarity_rank` (an Integer column) a numpy float.
* **`rarity_token()` and a `rarity_meta` column** in the ByMykel ingest. The rank alone
  cannot rebuild the string: rank 3 collapses `milspec`, `high_grade` and `distinguished`.
  Token and rank are kept in lockstep — both resolve or neither does.

## Results

Like-for-like on the same 8,691 slugs the old file held:

| | before | after |
|---|---|---|
| rarity present | 4,395 (50.57%) | **8,686 (99.94%)** |
| gained / **lost** / changed | — | 4,291 / **0** / **0** |
| rank 0 | 4,920, only 624 real | **2,502, all real** (2,494 `base` + 8 `highlight`) |
| unknown rank | 0 — hidden as rank 0 | NULL |

Full regenerated archive: **40,934/41,725 = 98.10%**. ByMykel supplied 1,933 of the 1,938
that survived the key fix (1,147 on the literal key, 786 more once normalised).

Worked examples of items that read as rank 0 before:

```
AK-47 | Asiimov (Minimal Wear)       -> covert     (6)
AK-47 | Nouveau Rouge (Minimal Wear) -> classified (5)
AUG  | Aristocrat (Field-Tested)     -> restricted (4)
```

**791 items remain unknown archive-wide, 5 within the original 8,691.** All are mangled
keys — `souvenir-nova-wurst-h-lle-field-tested` (mojibake for Würst Hülle),
`steam_sticker_|_sico_|_rio_2022`. Not pursued.

## Tests

* `backend/tests/test_supply_metadata_keys.py` — 15 cases covering normalisation in both
  directions, the slug-form lookup that used to miss, the exact-name lookup not regressing,
  collision precedence, and unknown-vs-base rank.
* `backend/tests/test_rarity_coalesce.py` — 24 cases covering the token map, token/rank
  consistency, fill-but-never-overwrite, normalised-key matching, rank 0 surviving the fill,
  row order preserved, and the `Int64` dtype guard.
* One existing schema assertion in `test_ingest_bymykel_metadata.py` updated for the new
  column. **270 pass** across all affected files.

## Files regenerated

`item-metadata-bymykel.parquet` (offline from the cached JSON dumps — no network) and
`item-metadata.parquet`. The latter went **8,691 → 41,725 rows**: `build_metadata` rebuilds
from every distinct slug in `prices-*.parquet` and the archive has grown since the old file
was written on 2026-07-15. The new rows include the known phantom slug-keyed duplicates
(`slug-keyed-items-are-duplicates`), and their lower coverage is what pulls the full-archive
figure to 98.10% against the subset's 99.94%. A backup of the pre-change file was taken to
the session scratchpad.

## The caveat that limits all of it

**This buys no measured accuracy.** The ByMykel metadata features were refuted by the
model's own permutation test, and item type and specific weapon are dead axes
(`2026-08-07-training-item-universe.md`). What the repair actually buys is two things:

* a correct stratification key for `_stratified_item_subsample`, which buckets on the rarity
  **string** — so the previous state put 49% of items into a single `"unknown"` bucket and
  called it stratified;
* the removal of the rank-0 conflation, which was a live misreading waiting for the first
  consumer to key on the rank.

## Related

* `2026-08-06-bymykel-metadata-ingest.md` / `-refuted.md` — the fill source, and why better
  coverage of it is not an accuracy claim.
* `2026-08-07-training-item-universe.md` — the entry that surfaced the `"unknown"` bucket
  while auditing what the subsample stratifies on.
