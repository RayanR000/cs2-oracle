# Refutation power tiers, and the iflow BUFF backfill

> **Status (as of 2026-08-21): the power-tier framing was VINDICATED and sharpened; the iflow
> payoff was mostly NEGATIVE.** Item by item:
>
> - **"Not every refutation was tested on rich enough data"** — confirmed and generalised.
>   `docs/research/2026-08-19-deep-model-review.md` §11 finds the A/B family was **never
>   powered** (MDEs 1.15–7.13pp; some arms have no MDE at all), so most stored "null" verdicts
>   should read **UNRESOLVED**. ⚠️ But the prescribed remedy inverted: a **broad re-run of the
>   nine repaired harnesses is now explicitly on the do-not-do list** (§12) — put the MDE beside
>   each verdict and relabel instead. No 2026 A/B is interpretable at all until the
>   consensus-estimator composition breaks (§1) are controlled.
> - **iflow `count_in_24` as the volume-feed repair** — built, tested, **closed net-negative**
>   (`changelog/2026-08-17-volume-in-scale-is-net-negative.md`). ⚠️ The backfill script also
>   **splices BUFF `count_in_24` and Steam volume into one mislabeled column** — emit them
>   separately on any future depth ingest.
> - **iflow as a training-breadth expansion** — measured **accuracy-neutral** at a fixed row
>   budget (`changelog/2026-08-18-training-breadth-is-accuracy-neutral.md`). The breadth path is
>   Steam-consistent backfill, not iflow.
> - **iflow as a Buff price source for the cross-venue basis** — the basis feature was
>   **shelved 2026-08-19**; do not build the ingest
>   (`docs/research/2026-08-16-cross-venue-basis-steam-buff.md`).


*2026-08-16*

Two linked findings: (1) the project's "null/refuted" feature verdicts are **not all
equally trustworthy** — they split into well-powered row-level tests and
underpowered/never-actually-tested directional ones; and (2) a free ~4-year BUFF+Steam
history source (`api.iflow.work`) is exactly the rich data needed to re-run the
underpowered tests, plus a **live trade-volume series** that could settle the project's
one open positive lead.

## 1. Not every refutation was tested on rich enough data

The refutations sort into two tiers.

### Tier A — well-powered, trust these
Tested at the **row level** on millions of observations; independent of the scarce
forecast-date / episode count.

- **Item metadata (ByMykel)** — within-run permutation shuffled the whole group at
  700K rows / 646 items and cost ~0 DA vs `price_technicals` at p=0.0000
  (`docs/changelog/2026-08-06-bymykel-metadata-refuted.md`). The model genuinely does
  not use those columns. More data will not change this.
- **Trade volume as a *linear* correlation** — |r| < 0.002 across 4.47M rows
  (`docs/changelog/2026-08-08-volume-null-not-zero-and-skinport-sales.md`). A
  well-powered *linear* null.

### Tier B — underpowered, or never actually tested
- **Order-book depth / bid-ask / supply & listing counts** — *not refuted on merit;
  failed on missing data*. Bid-ask had ~31 dates; supply-history has a 29-month hole
  (ends 2024-02-16); live depth is unwired/untested
  (`docs/research/2026-08-15-directional-accuracy-and-data-inventory.md:104-105`;
  `docs/changelog/2026-08-14-supply-side-rarity-is-null-and-fails-the-placebo.md:462-465`).
  These are **untested for lack of data**, not disproven.
- **Trade volume as a *directional* signal — genuinely unresolved.** The +1–2pp,
  placebo-clean-at-4/4-horizons read was on a *static, dead Kaggle feed* and
  "does NOT transfer to production until the A/B is re-run on a live series"
  (`docs/changelog/2026-08-15-volume-features-remeasured.md`). The linear null and the
  directional positive can coexist (a nonlinear/conditional effect a linear test misses).
- **Events / game updates / player counts** — structurally underpowered: events are
  rare, MAU had little variation, and every *directional* A/B is capped by the same
  ~14–70 non-overlapping 2026 windows. Soft nulls.

### The limit of re-testing
Richer data fixes **power, not a true zero**. The structural finding —
"once the market-wide factor is removed, there is no idiosyncratic per-item signal"
(`docs/changelog/2026-08-15-cs2-oracle-is-a-range-forecaster.md`) — is something more
data *tests* but may just confirm. A re-test can still return null; the gain is a
*trustworthy* null instead of an underpowered one.

## 2. The iflow BUFF backfill (candidate source, not yet ingested)

`EricZhu-42/SteamTradingSiteTracker-Data` / `api.iflow.work`.

- **Access (free, no auth):** `GET https://api.iflow.work/export/list?dir_name=priority_archive`,
  then `.../export/download?dir_name=priority_archive&file_name=YYYY-MM-DD-HH-MM.zip`
  (allow 301). ~2,990 dumps, **2022-04-18 → 2026-05-20**, 12h cadence, ~7.8 MB zip /
  ~33 MB JSONL each, ~16,600 records/dump (CS + Dota; filter `game=="csgo"` / `appid==730`).
  **Stale since 2026-05-20 → historical backfill only**, not a live feed.
- **Per-record fields:** `hash_name` (= Steam `market_hash_name` = archive `item_slug`
  verbatim, no name-matching needed), `buff_reference_price` (**raw CNY**),
  `count_in_24` (**real 24h trade count** — a live volume series), `buy_order_list` /
  `sell_order_list` (Steam order-book depth), `buff_buy_num` / `buff_sell_num` (supply
  counts), plus IGXE/C5/UUYP prices.

### Why it matters, headline first
1. **`count_in_24` is the live trade-volume series the open volume lead needs** — the
   one experiment that could resolve Tier B's volume question. (Hedge: volume is
   linearly refuted at |r| < 0.002; the positive may still be CV-positive/serving-negative.)
2. **4 years of price history for the liquid cohort** — feeds the one proven axis
   (price volatility → exceedance) and multiplies backtest episodes ~10× (2026-only →
   2022–2026), which is what every underpowered directional test was missing.
3. **Depth / supply counts become testable for the first time** — iflow fills the data
   holes that made those Tier-B verdicts untrustworthy.

### Integration notes
- Slug maps for free. **CNY→USD is net-new**: existing BUFF arrives pre-converted
  upstream; convert via `exchange-rates-history.parquet` (cols `day, currency, rate`;
  `rate` = CNY per USD, so `usd = cny / rate`), forward-filled, `MAX_FILL_DAYS=7`.
- Format (zip/JSONL/12h) does not fit the `price_history_sources` adapter → use a
  **standalone script** modeled on `backend/scripts/merge_17mafo_gap.py`, source label
  `buff_iflow`, one dump/day nearest a fixed UTC time to match the daily archive.
- Validate via `import_price_history_source.py --report` seam check → `compute_mde.py`
  → paired walkforward A/B; ship to durable only on `verdict == "positive"` clearing the
  ~2–4pp item-level MDE. Every stored A/B verdict predates the 2026-08-08 stats fix — re-run.
- **Two first-class fields: price + `count_in_24`.** Keep summarized order-book/counts
  (top-of-book + depth, not raw ladders) as inputs to the Tier-B re-tests — not as
  shippable features.

### Caveats
- Cohort is **dynamically selected for liquidity** → survivorship-flavored panel
  (history only for items liquid at the time).
- The genuinely-untested *fundamental* mover (per external evidence: supply creation) is
  **case unboxing volume**, which iflow does NOT carry — that is CS2 Case Tracker
  (`csgocasetracker.com`), a separate future source.
