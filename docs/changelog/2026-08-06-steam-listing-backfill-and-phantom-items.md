# Steam listing-page backfill, and the 3,149 phantom items

**2026-08-06.** Research into free bulk history sources produced one viable new
source and one significant data-quality finding. Neither is merged; the backfill
is blocked on IP reputation and the phantom-item fix needs prod access.

## 1. Steam listing pages serve full history with no cookie

`GET steamcommunity.com/market/listings/730/<market_hash_name>` (follow the 302)
embeds the complete price series in its dehydrated React-Query cache:
`{time, price_median, purchases}`, daily, back to 2013. The cookie-gated
`/market/pricehistory/` endpoint returns **HTTP 400** without a session; this
route does not. Every community thread says logged-out history is impossible —
it is undocumented, not impossible.

Pages carry multiple wear/StatTrak/Souvenir variants, all harvested per request:
**4.52 series/request, of which 3.69 are themselves targets**.

New script: `backend/scripts/backfill_steam_listing_history.py`. Staging SQLite
only (`runtime/steam_listing_history.db`); never touches prod or the archive.

**Collected so far: 262 items, 528,573 rows, back to 2013-08-15.**

### Two traps hit and fixed

**Price basis.** The page quotes the **buyer** price; the archive stores **net**.
Ratio is a constant 1.1607 (p10 1.1565, p90 1.1656, within-item CV 0.0021 over
30,875 rows). Uncorrected → 16.08% median error; corrected → **0.038%, corr
1.000000**. Volume needs no correction (77.8% exact match raw).

**Silent soft-block.** Steam does not send 429. It serves HTTP 200 with a
stripped ~250 KB shell — no 302, zero `pricehistory`, no `Retry-After`. A
300-request run logged *0 failures, 0 429s, "80% EMPTY"* while throttled
throughout. Now detected in `classify()`, with a **canary** item checked before
the run and every 50 requests that aborts rather than recording false negatives.

### Efficiency bug found on review

The loop re-requested items already harvested as page variants — only the
*resume* path deduped, not the in-run path. Left in, a full run would have made
~27,741 requests instead of ~7,520 (**3.7x**), which invalidated the original
20-hour estimate. Fixed with an in-run skip seeded from the staging DB. Target
ordering turns out to be irrelevant once the skip exists.

### Current targeting

Three conditions — actively collected, not gated, real name:

1. present on the latest archive day (the daily aggregator still writes it)
2. outside the `is_backfilled` gate
3. a real `market_hash_name`, excluding **both** mangled key formats

→ **27,945 items**, ~7,520 requests, ~19.4 h at `--delay 8`.

Condition 3 matters more than it looks: the slug regex `^[a-z0-9-]+$` does *not*
match the `steam_…|…` form, so excluding slugs alone would have sent 4 phantom
keys as real targets and inflated the EMPTY rate.

### Blocked on IP class

Three egresses tested. Residential worked then earned a >5 h penalty; two
commercial VPNs (DataPacket US, NForce NL) were stripped on their **first**
request. Steam gates this on hosting-ASN class, so **no VPN will work** and the
route can never run in CI. See `docs/references/data-sources.md` for the table,
plus the currency hazard on non-US exits.

## 2. 3,149 archive items are phantom duplicates

While reconstructing `market_hash_name` for the slug-keyed items — to add them to
the backfill — it turned out they need no backfill at all.

The archive's `item_slug` **is** `items.item_id`, copied verbatim
(`append_to_parquet.py:33`, `export_daily_snapshot.py:24`). Two writers put
malformed ids in that table:

| Format | Count | Origin |
|---|---|---|
| `sealed-graffiti-popdog-battle-green` | 3,145 | `migrate_historical_data.py:345` — `"item_id": slugify(mn)`, while every other inserter uses `item_id = hash_name`. Committed `464c6fc` 2026-07-07; phantom rows start 2026-07-09. |
| `steam_sticker_\|_sico_\|_rio_2022` | 4 | `real_data_collector.py:340` — `f"steam_{hash_name.lower().replace(' ', '_')}"`. Added `02e4141` 2026-05-22, file deleted `92f5aa9` 2026-05-27 — a 5-day window, hence only 4. |

`slugify` is invertible against the archive's own name list: **3,139/3,145
resolve unambiguously, 0 unmatched**. All resolve to items **already inside the
gate** with far deeper history, and the series are identical — 62,780 overlapping
(item,day) cells at **0.0000% median difference, correlation 1.000000, 100.0%
equal**.

`init_local_db.py::populate_items` then mirrors archive keys back with
`name = item_id`, which is why the DB shows `item_id == name == 'sticker-sherry'`.
That identity is the mirror step, not the origin.

**This is the 3,149 cohort** from the serving down-bias work. Proved against
`saved_models/engineered_data.parquet` (the 07-29 cache): it holds exactly 8,691
items = 5,542 + 3,149, and the slug set's row counts are **median 15, min 14,
max 15**, matching that cohort's description exactly. Not one of the 3,149 was a
real item.

### The leak is ongoing

Both writers are dormant (one deleted, one unwired), but the bad ids remain in
**prod's items table**, so the daily aggregator re-emits them every run:
**3,149 phantom items, ~32,700 rows/day — 228,570 rows since 2026-07-25, 9.03%
of recent archive writes.** Fixing the scripts changes nothing.

**Remediation (not done, needs prod access):** repoint or delete those 3,149
`item_id`s in prod, then purge the phantom rows from the archive (686,035 rows,
3.31% of 20.7M). Until then, treat any statistic computed over recent archive
days as ~9% contaminated.

## Other sources evaluated

See the table in `docs/references/data-sources.md`. Headlines: the BUFF163 git
archive is real and large (21,954 items, 15.4M rows, one 24 MB download) but
depth-only, so it *shrinks* `target_items` 521 → 426; CSFloat's
`/api/v1/history/<name>/graph` is free, unauthenticated, unblocked, and daily
back to 2020-04 with sale counts — the best unblocked fallback if the Steam route
stays gated.
