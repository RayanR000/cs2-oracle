# Drop Supply Depth as Prediction-Accuracy Improvement (2026-07-16)

## Decision

Supply depth (active `sell_listings` count) is **dropped** as a prediction-accuracy
improvement for the CS2 market forecaster. It was previously the top remaining item
in the accuracy backlog (`docs/research/accuracy-opportunities.md` #6).

## Rationale

- **Only the change/velocity variant is directionally predictive.** Mechanistic
  analysis of the CS2 economy (CS2REF market-economics, 2026) shows short-run price
  moves are driven by the *rate at which holders list*, not the listing level. The
  level feature (`supply_listings_log`, `supply_to_volume_ratio`) is a
  liquidity/tradability signal — CS2Cap explicitly states listing count is "not a
  price forecast." The level therefore adds ~0pp directional accuracy, analogous to
  the already-rejected trade volume (audited |r| < 0.002, 0pp).
- **Predictive variant needs history we don't have.** `supply_change_7d` and
  `supply_listings_zscore` (30d rolling) require ≥30 days of `supply_snapshots`, or a
  paid historical backfill (CS2Cap candles `q`). Without it, only the level feature is
  computable → no directional lift.
- **Free source is too slow; no free bulk source exists.** The only free path is the
  Steam full-catalog burst scrape (~115 min/day to walk 34K items for 5.5K tracked).
  Steam `priceoverview` is per-item (~8h for 5.5K) and does not return listing count.
  No free bulk listing-count API exists.
- **Paid APIs rejected.** CSMarketCap ($9.99/mo, 1 call = all items) and CS2Cap
  ($19/mo, batch 100/req) provide bulk listing counts, and CS2Cap candles may allow
  historical backfill — but paying was declined.
- **Expected lift was modest.** Calibrated ~+1-2pp directional even in the best case;
  not worth the cost/hassle given the 30-day gate.

## Impact

- `docs/research/accuracy-opportunities.md`: supply depth moved from top remaining (#6)
  to **Dropped**; regime-switching models is now the top remaining item. §1 carries a
  DECISION note and the trade-volume/supply-depth distinction warning.
- `docs/research/feature-engineering.md`: Phase 4 (supply/liquidity) marked Dropped.
- `docs/2026-07-14-remaining-accuracy-improvements.md`: recommendation #6 marked Dropped.
- `docs/changelog/2026-07-16-multi-source-outlier-voting.md`: top-ROI note updated.
- `docs/references/data-sources.md`: supply scraper marked "output unused".
- Realistic cumulative ceiling for remaining work is **unchanged: +5-8pp**
  (60-68% → 65-76% directional).

## Open actions (not yet done — out of scope for this doc update)

- `_add_supply_depth_features` in `backend/models/forecaster.py` (called at
  `forecaster.py:1148` and `forecaster.py:1387`) remains but is excluded from the
  accuracy roadmap. Consider removing the call so zeroed supply features stop being
  added (avoids wasted columns / capacity).
- The `supply-scraper.yml` workflow still runs ~115 min/day for output that is no
  longer used for prediction. Consider disabling it to save compute, unless the
  `supply_snapshots` data is wanted for non-prediction uses (e.g. liquidity display).
- If supply depth is ever revisited: use a paid bulk API + historical backfill and
  test **only** the change/velocity features via a permutation A/B (never a plain A/B,
  per the capacity-inflation lesson from player counts).

## Related

- `docs/research/accuracy-opportunities.md` — §1 DECISION, Priority Order
- `docs/research/volume-data.md` — trade-volume audit (0pp)
- `docs/references/data-sources.md` — supply scraper status
