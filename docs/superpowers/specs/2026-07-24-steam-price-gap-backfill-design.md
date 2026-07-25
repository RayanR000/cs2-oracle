# Steam Price-History Gap Backfill (Apr 16 – Jul 8, 2026)

**Date:** 2026-07-24
**Status:** Approved design — ready for implementation plan
**Author:** brainstorming session

## Problem

`price-archive/prices-2026.parquet` — the price history the forecaster reads
(`forecaster.py:480–515`, `read_parquet('price-archive/prices-*.parquet')`) —
has a hard gap. Data runs continuously to **2026-04-15** (where the previous
`merge_hf_dataset.py` backfill left off), then nothing until **2026-07-09**,
when live collection resumed. The missing window is **2026-04-16 → 2026-07-08**
(84 days), across *all* sources.

This gap is the root cause behind the stale-model / interval-coverage symptoms:
a clean retrain cannot produce correct intervals while ~3 months of the serving
window is empty (see memory `q90-goss-interval-bug-is-stale`). Filling it is a
prerequisite to the retrain + `backtest_accuracy` confirmation.

### Constraints

- **Must be free.** No paid API (cs2.sh Scale $200/mo, Pricempire ~$240/mo,
  CSGOSKINS €279/mo were all researched and rejected on cost).
- Fill must land in `price-archive/prices-2026.parquet` (+ `snapshots-2026.parquet`),
  the files the forecaster actually reads — not the dead-end `runtime/ssr_history.db`.

## Research summary (what was ruled out)

- **Userless Steam scrape** — dead (memory `steam-userless-pricehistory-dead`).
- **Free static datasets** (Kaggle leawind/kieranpoc, HF, GitHub archives) — all
  stale (≤2024/mid-2025), none reach the window.
- **HF `idomanteu` continuation** (source of the *previous* gap-fill) — does not
  exist; it was a one-off promotional dump of paid cs2.sh data.
- **Paid aggregators** (cs2.sh, Pricempire, CSGOSKINS, CSMarketAPI) — all cover
  the window but gate history behind $200–280/mo. Rejected: must be free.
- **Wayback / csgotrader `prices_v6.json`** — archiving stopped 2023; dead.

## Chosen source: `17mafo/cs-price-tracker` (GitHub, free, public)

Repo: <https://github.com/17mafo/cs-price-tracker>
Raw path: `https://raw.githubusercontent.com/17mafo/cs-price-tracker/main/static/prices/YYYY-MM-DD.json`

**Verified independently (2026-07-24):**

- All 84 daily files `2026-04-16.json … 2026-07-08.json` exist (HTTP 200), zero
  missing days.
- Each file (~7 MB) is a JSON object keyed by Steam `market_hash_name`; value is
  `{"steam": {"last_24h", "last_7d", "last_30d", "last_90d", "last_ever"}}`.
  `last_24h` is that calendar day's Steam price.
- ~27,261 items on Apr 16 → ~28,337 on Jul 8 (grows over the window).
- **Keys are already `item_slug`** — the JSON key `"AK-47 | Redline (Field-Tested)"`
  is byte-identical to the archive's `item_slug`, StatTrak™ prefix included. No
  name→slug mapping needed.
- **Currency/scale aligns with our own Steam capture.** 17mafo Jul-08 `last_24h`
  = 43.63 vs. archive `aggregator_steam_7d` Jul-11 = 43.76 (0.3% match). No
  conversion required; the "EUR" label on the source is immaterial at this scale.
- High-value items above Steam's ceiling (e.g. Dragon Lore FN) carry `null`
  recent values — skipped (illiquid, negligible for the model).
- The repo's per-item `static/pricehistory/{sha1}.json` files are stale/unreliable
  and MUST NOT be used. Only the dated `static/prices/` files are authoritative.

## Design

A single new script, `backend/scripts/merge_17mafo_gap.py`, modeled directly on
the trusted `merge_hf_dataset.py` precedent. Pure network-pull + local transform;
**no Steam scraping, no session cookies, no ban risk.**

### Data flow

```
17mafo raw JSON (84 files)  →  merge_17mafo_gap.py  →  price-archive/prices-2026.parquet
                                       │                price-archive/snapshots-2026.parquet
                                       └──────────────→  price-archive/raw/17mafo/*.json  (local copy)
```

### Steps

1. **Fetch + cache.** Download `2026-04-16.json … 2026-07-08.json` to
   `price-archive/raw/17mafo/` (so we never depend on the repo staying up, and
   re-runs are offline). Fetch is idempotent: skip a date whose file already
   exists locally. Fail loudly if any of the 84 dates is missing or non-200.

2. **Transform.** For each file (day = filename date), for each item:
   - price = `steam.last_24h`; skip if `null`/missing.
   - Emit one row: `item_slug=<key>`, `day=<file date, TIMESTAMP_NS>`,
     `mean_price = median_price = min_price = max_price = last_24h`
     (single price/day — no intra-day spread available),
     `volume = 0` (not provided; consistent with existing `aggregator_steam_*`
     rows, which are also `volume=0`),
     `source = 'aggregator_steam_17mafo'`.
   - Ingest **all** items present (~27K), not just the 4.8K modeled subset —
     broader coverage at zero extra cost.

3. **Append (idempotent).** Reuse `merge_hf_dataset.py`'s `_append_parquet`
   contract exactly:
   - `prices-2026.parquet` ← price rows, columns
     `[item_slug, day, mean_price, median_price, volume, min_price, max_price, source]`,
     dedup keys `["item_slug", "day", "source"]`, `keep="last"`.
   - `snapshots-2026.parquet` ← `[item_slug, day, source, price, volume]` where
     `price = last_24h`, dedup keys `["item_slug", "day", "source"]`.
   Because the dedup key includes our unique `source` label, re-runs are safe and
   nothing else in the archive is touched.

4. **CLI.** `--start-date` / `--end-date` (default the gap bounds),
   `--out-dir` (default `../price-archive`), `--dry-run` (fetch + report counts,
   no write), `--refresh` (re-download cached files).

### Source label

`aggregator_steam_17mafo` — provenance-explicit, distinct from the live
`aggregator_steam_7d/30d/90d` windows and from `aggregator_sync`. Makes the
backfilled rows trivially filterable/removable later.

## Validation gate (must pass before retrain)

The script prints and asserts:

1. **Day coverage** — all 84 days Apr 16 – Jul 8 present under
   `source='aggregator_steam_17mafo'` in `prices-2026.parquet`.
2. **Item coverage** — per-day distinct item count ≈ 27K (within a sane band;
   warn if any day drops materially).
3. **Seam continuity** — for a spot-check set (≥10 liquid items incl.
   AK-47 | Redline (FT)), the Apr-15→Apr-16 and Jul-08→Jul-09 price steps are
   within a plausible band (no discontinuity vs. adjacent real data). The
   AK-Redline seam check already passed by hand (43.63 vs 43.76).

Then, out of scope for this script but the reason it exists:
re-run `backtest_accuracy` and confirm ~80% interval coverage (per memory
`q90-goss-interval-bug-is-stale`).

## Fallback (only if needed)

If validation shows specific **modeled** items missing from 17mafo, patch just
those with the existing `backfill_ssr_history.py --modeled-only` (a small
targeted Steam scrape of the missing set, then a `ssr_history.db → parquet`
bridge for that subset) — not a 4.8K-item run. Expected to be unnecessary given
27K-item coverage.

## Out of scope

- Retrain / model artifacts (separate task, gated on this).
- Backfilling sources other than Steam for the window (the previous gap used
  BUFF/CSFloat/YouPin from HF; not required here — the model reads a blended set
  and Steam is the anchor).
- Any change to the live collection pipeline.

## Risks

| Risk | Mitigation |
|---|---|
| Trusting a third party's capture | Steam-sourced; seam-verified against our own data within 0.3%; local raw archive kept |
| Repo disappears mid-fill | Files cached to `price-archive/raw/17mafo/` on first fetch; re-runs offline |
| No volume data | `volume=0`, consistent with existing `aggregator_steam_*` rows |
| Some modeled items absent | Targeted `backfill_ssr_history.py --modeled-only` fallback |
| Currency drift over window | Seam alignment confirms scale parity; single-scale, no per-day FX applied |
