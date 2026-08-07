# Next step: start accumulating lis-skins snapshots

**Date:** 2026-08-06
**Status:** proposed, not built
**Context:** `docs/changelog/2026-08-06-retroactive-supply-feeds.md`

## Why this and nothing else

The 08-06 analysis found the three depth feeds are one feature, not three
(Spearman 0.65–0.82 between them), and that every measured correlation is
**trailing** — the archive ends 08-04, the feeds are live-only snapshots, so no
forward test is possible today. Nothing about supply depth can be settled by
pulling harder on the current data.

Two questions block the whole line of work, and one action answers both:

1. **Is `created_at` listing age, or last-reprice time?** 99.3% of lis-skins
   inventory reads as created in the last ~37 days. Either the book turns over
   monthly or the field resets on relist. These give opposite meanings to every
   age feature, and a single snapshot cannot distinguish them.
2. **Does any of it lead price?** Needs ≥2 snapshots separated in time.

Consecutive daily snapshots answer (1) directly — track individual listing `id`s
across days and see whether `created_at` moves for a listing that persists — and
accumulate the series needed for (2).

## What to build

A collector that pulls `https://lis-skins.com/market_export_json/api_csgo_full.json`
(no auth, ~173 MB, ~44 s) once daily and writes **aggregates only**, not the 2.3M
raw listings. Per `(item_slug, day)`:

- `listing_count`
- `min_ask`, `p25_ask`, `median_ask`
- `depth_5pct`, `depth_10pct` — listings within 5% / 10% of `min_ask`
- `age_median_days`, `age_p90_days` from `created_at`
- `inflow_24h`, `inflow_7d` — listings created in that window
- `n_listings_with_float`, and the listing-`id` set hash needed for the
  `created_at` diagnostic in (1)

`item_slug` == `market_hash_name`, so it joins to the archive directly.

## Constraints carried over

- **Do not run it from `backend/`** — `.env` there points at production Supabase.
- **Untested from a GitHub runner.** lis-skins is not a Steam host, so the 429
  that killed `supply-scraper.yml` should not apply — verify this before wiring
  anything into the daily chain, because a silently-empty collector is the
  failure mode this repo keeps hitting.
- **The task must return row counts.** `scripts/run_task.py`'s zero-row guard is
  the only thing standing between a dead collector and a green badge.
- Anchor min_ask against the archive price: 358 items currently show
  `min_ask > 10x` the archive price (worst case 1,154x), so the ladder anchor
  needs an outlier guard or depth/tightness are contaminated.

## What this does not claim

No lift. Prior evidence stands unchanged: trade volume at |r| < 0.002 over 4.47M
rows, 85 non-price features within fold noise, hence
`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`. Listing age and ladder shape
are a *different quantity* from the refuted volume level — untested, not
promising.

## Revisit

~6–8 weeks of snapshots (early October 2026). At that point run
`backend/scripts/compute_mde.py` **first**; if the MDE is above ~2pp, stop — the
harness cannot resolve the effect being looked for and the accumulated data buys
nothing measurable. Only if it clears does a **permutation** A/B on the
change/velocity features follow.
