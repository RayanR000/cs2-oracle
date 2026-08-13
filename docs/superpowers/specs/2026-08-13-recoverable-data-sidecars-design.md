# Recovering volume, supply-depth, bid, and StatTrak signals — sidecar design

**Date:** 2026-08-13.
**Goal:** bring the demand/supply signals the model currently lacks into a **local research
dataset**, integrated with the existing archive, and **measure** whether each helps before any of
it reaches production. The premise: the archive is rich in price (ask) data and poor in the
demand/supply observables that distinguish one item from another — the same gap the null streak
keeps hitting.
**Scope decisions (fixed with the user):**
- Full scope: volume + BUFF bid + StatTrak premium + retroactive supply-depth.
- **Local research dataset first**, MDE-gated. Nothing lands in the durable CI archive
  (`cs2-oracle-data`, CI-only force-push) until it clears a paired-A/B bar. That promotion is a
  separate, later decision, explicitly **out of scope here**.
- Volume source is the 2026 `devynpruden` Kaggle parquet; the 2024-ending `kieranpoc` dump is
  **dropped** (superseded — reaches 2026, single parquet, carries volume + parsed metadata).
- Integration is via **sidecar parquets** joined at feature-engineering time, never as new rows in
  the price series.

## The four feeds — coverage decides serve-vs-train

A feed that dies before 2026 can train a feature but cannot serve it, and creates a train/serve
availability gap. Coverage is therefore the organizing axis:

| Feed | Source | Coverage (verified 2026-08-13) | Serveable? |
|---|---|---|---|
| **Volume** | `devynpruden/cs2-skin-price-history-2013-2026` (Kaggle parquet) | 2013-08-14 → **2026-06-15**, `volume>0` 100% every year, 4,342 items | ✅ covers the 2026-04→06 serving anchors |
| **BUFF bid** | `aggregator_buff163_buy`, already in the archive | 2026+ daily (~185k rows/mo) | ✅ |
| **StatTrak premium** | computed from paired ST / non-ST archive prices | full history | ✅ |
| **Supply-depth history** | `atalantus/buff-price-history-archive` (GitHub, xz-JSON) | 2023-01-25 → **2024-01-19** only | ⚠️ **training-only** — see caveat |

Verified: all 4,342 `devynpruden` `market_hash_name` keys match the archive's `item_slug`
**exactly** (`item_slug ≡ market_hash_name`), so every join below is a plain `(item_slug, day)`
equijoin with no fuzzy matching. Kaggle CLI + credentials are configured locally.

## Architecture — four sidecars, one uniform join

All recovered signals land as **sidecar parquets** under the local `price-archive/`, each keyed
`(item_slug, day)`, each built by its own script, each **left-joined onto the voted price frame
after voting** and before `engineer_features`. NULL where a sidecar has no row. This keeps every
recovered quantity **out of the price-voting consensus** — volume, bid, and supply must never vote
as a price, which the item-universe invariant already enforces for the bid and the trailing-window
means. The uniform pattern also means one cache-fingerprint rule covers all four.

| Sidecar | Built by | Input | Columns (beyond the key) | External? |
|---|---|---|---|---|
| `volume-panel.parquet` | `scripts/ingest_volume_panel.py` (new) | devynpruden parquet | `steam_volume INT`, `steam_sale_median DOUBLE` | download |
| `supply-history.parquet` | `scripts/ingest_supply_history.py` (new) | atalantus xz-JSON | `buff_listing_count INT` | download |
| `bid-panel.parquet` | `scripts/build_bid_panel.py` (new) | archive's own `aggregator_buff163_buy` rows | `buff_bid DOUBLE` | internal extract |
| `stattrak-panel.parquet` | `scripts/build_stattrak_panel.py` (new) | archive voted prices, ST↔non-ST pairing | `st_premium DOUBLE` | internal compute |

Why the bid and StatTrak are *also* sidecars despite being internal: it makes the feature-join
uniform (one code path, one cache rule) and keeps the bid extraction from perturbing
`_fetch_voted_price_history`, which the item-universe rule guards. The bid panel reads the raw
pre-voting rows for the one excluded source; the StatTrak panel pairs each `StatTrak™ …` slug with
its base slug on the same day and stores the price ratio.

### Component boundaries

- **Ingest/build scripts** are pure `input → sidecar parquet`. No feature knowledge, no model
  dependency, independently runnable and testable. Each writes an idempotent parquet and logs row
  counts and cohort coverage (how many of the 926-item ≥$1 cohort it covers).
- **The join** is one new helper (e.g. `_attach_sidecars(price_df)`), called once after voting.
  Left joins, dtype-checked, NULL-filling documented per column. It is the only seam feature
  engineering sees.
- **The features** read the joined columns behind flags (below). They never read a sidecar path
  directly — they read the merged frame, so a missing sidecar degrades to NULL features, not an
  error.

## Feature wiring — all behind flags, off by default

Matching how every arm in this repo ships (gated, read-only until measured):

- **Volume** (`VOLUME_FEATURES=1`): un-shelve the 11 existing volume features
  (`forecaster.py:2114-2193`, currently in `SHELVED_FEATURES` because the column was identically 0).
  They now read `steam_volume`. No new feature code — the shelf comes off.
- **BUFF bid** (`BID_FEATURES=1`): new features — bid level, bid/ask spread, bid-relative position
  — from `buff_bid` against the served ask. The market's only direct demand-side observable.
- **StatTrak premium** (`STATTRAK_FEATURE=1`): `st_premium` as a per-item usage/demand proxy.
- **Supply-depth history**: extends the existing live supply feed; read through whatever flag the
  supply-depth collector already uses. Diagnostic-only (see caveat).

## Measurement plan — the gate

Each feed gets a **paired A/B on the tied ≥$1 cohort** (`p/S ≡ 1`, the only basis where rank IC
means what it says), with its **MDE computed before the run** (the noise-floor / fold-count
discipline: the harness cannot resolve sub-1pp effects, and a placebo must be read before the
treatment). Priority order, by how measurable and serveable each is:

1. **Volume** — first. The only recovered feed that reaches the serving anchors, so the only one
   whose serving transfer is measurable at all. It also has the strongest prior (Oct-2025 crash was
   priced by supply position; volume is the closest observable).
2. **BUFF bid** — serveable daily; direct demand signal.
3. **StatTrak premium** — serveable, full history.
4. **Supply-depth history** — training-only; a CV-rank-IC diagnostic at best.

No feed is proposed for the durable archive until it clears its bar. A null is a valid, cheap
outcome that retires the feed — the same closure the feature audit bought elsewhere.

## Caveats written into the design

- **Supply-depth history is training-only.** The atalantus panel ends 2024-01-19; the live
  supply-depth collector started ~2026-07. The 2024–2026 hole means a supply-depth feature built
  from it cannot be served — it is a diagnostic, not a serving path. Do not un-shelve it as a served
  feature on the strength of a CV read.
- **Volume snapshot ends 2026-06-15; today is 2026-08-13.** The serving anchors (2026-04→06) are
  fully covered, so the *research* read is valid. But production serving beyond 2026-06 would need a
  live volume feed (Steam `market/pricehistory`, no login — `HilliamT/scm-price-history` /
  `somespecialone/steam-market-ids` are the free routes). That live feed is **out of scope**; it is
  only needed if volume clears its gate and is promoted.
- **Licenses.** devynpruden and atalantus carry no clear redistribution license. **Private/local
  training only** — never commit their raw rows to the durable archive; a promotion would ship
  *derived* features, not the source data.
- **Availability leakage.** A feature that is dense in training and NULL at serving teaches the
  model to lean on something it cannot see. The join NULL-fills, and the flag defaults off, but the
  A/B must confirm the feature is non-NULL on the tied serving cohort before any serving claim —
  not just on the training folds.

## Out of scope (named so they are not silently assumed)

- Any write to the durable `cs2-oracle-data` archive or `aggregator-update.yml`.
- A live volume/supply collector for dates past each snapshot.
- The `kieranpoc` 22k-CSV dump (dropped).
- Promotion of any feed to production — a separate decision after the gate.
