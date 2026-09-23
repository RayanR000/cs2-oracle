# Competitor Analysis

Date: 2026-07-15

---

## Market Context

The CS2 skin market is valued at approximately $4.5-7B (2026). Multiple platforms aggregate prices and provide trading tools, but none offer ML-driven price forecasting. This document maps the competitive landscape and identifies features worth building that align with the project's prediction-first strategy.

---

## Competitive Landscape

### CSMarketCap (csmarketcap.com)

**What they do:** Price comparison across 20+ marketplaces, market cap tracker, trade-up calculator, inventory checker, sales volume stats.

**Data access:** Paid GraphQL API (Standard/Pro/Business tiers). Endpoints for price recommendations, market analytics, min/max prices, live prices via WebSocket.

**Your differentiator vs them:** No ML forecasts, no technical indicators, no opportunity detection.

---

### SteamAnalyst (steamanalyst.com)

**What they do:** Price comparison across 15+ marketplaces, case simulator, trade-up calculator, inventory tracker, sticker calculator, 3D viewer, float/pattern checker, loadout builder, top 500 inventories.

**Data access:** Free Hobby API tier (100 req/day, non-commercial only). Paid Business/Enterprise tiers. Terms explicitly prohibit scraping for commercial purposes.

**Your differentiator vs them:** No ML forecasts. Their tools are engagement-focused (simulators, viewers), not predictive.

---

### TradeUp Academy (tradeupacademy.org)

**What they do:** Trade-up calculators, investment ROI tools, case ROI calculator, price tables, market cap index, educational articles/guides.

**Data access:** No API. Educational-only site. Prices are estimates.

**Your differentiator vs them:** Their "investment guides" are manually written. Yours can be ML-generated.

---

### CS2Insight (cs2insight.com)

**What they do:** Lightweight price tracking and trade-up tools.

**Data access:** No API.

**Your differentiator vs them:** Minimal feature set, no advanced analytics.

---

## Feature Recommendations (Prediction-Aligned)

### Strong fit — build these

| Priority | Feature | Rationale |
|----------|---------|-----------|
| P0 | **Trending / price movers page with ML overlays** | Natural front door to forecasts. Show actual price movement + "ML predicts +X% in 7d" alongside. SteamAnalyst and CSMarketCap both have this but without predictions. |
| P0 | **ML-generated investment picks** | Replace TradeUp Academy's manually written guides. "Top 10 undervalued items this week" sourced from your model output. Directly showcases your core value. |
| P1 | **Price recommendation signals** | CSMarketCap has a dedicated API endpoint for this. You can surface ML-driven buy/hold/sell signals per item using your forecast confidence intervals. |
| P1 | **Prediction-powered price alerts** | No competitor offers this. "Notify me when ML predicts item X will drop >10% in 7 days." Could use webhook, email, or in-app notification. Strong differentiator. |
| P2 | **Portfolio tracker with projected value** | SteamAnalyst and TradeUp Academy have basic portfolio tracking. Add "projected portfolio value in 30 days" using your forecasts. |
| P2 | **Market cap estimate + forecast direction** | Vanity metric that drives engagement. CSMarketCap and TradeUp Academy both show it. You could additionally forecast whether the overall market trends up/down. |

### Medium fit — feed into models or enhance prediction UX

| Feature | How it fits |
|---------|-------------|
| **Sales volume data** | ~~Feature input for ML models — volume predicts volatility and price momentum~~ — **refuted.** All 13 volume features were shelved 2026-08-06; see `docs/research/volume-data.md` and `docs/changelog/2026-08-06-volume-features-shelved.md` |
| **Cross-market price spread** | ~~Arbitrage patterns feed into forecasts~~ — the `cross_sectional` group is discarded by `FEATURE_GROUP_ALLOWLIST` before training (`backend/models/forecaster.py:353`) |
| **Most liquid skins** | ~~Context for prediction reliability (low liquidity = wider confidence intervals)~~ — the confidence gate was removed; the served band is split-conformal and not liquidity-conditioned (`docs/specs/2026-08-03-served-forecast-surface-design.md`) |

### Skip — zero prediction angle

Trade-up calculator, case simulator, 3D viewer, float checker, pattern checker, loadout builder, inventory leaderboard, skin inspector.

---

## Data Access Summary

| Site | API | Free Tier | Scraping Policy |
|------|-----|-----------|-----------------|
| SteamAnalyst | REST | 100 req/day (hobby, non-commercial) | Explicitly prohibited for commercial use without license |
| CSMarketCap | GraphQL + REST | None (paid only) | No explicit clause, but they sell API access |
| TradeUp Academy | None | N/A | Standard "no automated access" assumed |
| CS2Insight | None | N/A | Standard "no automated access" assumed |

**Recommendation:** Do not scrape competitors. Your data pipeline (multi-source aggregation via CSGOTrader API) already provides fresher, more granular data direct from the source marketplaces. Competitor data would be a stale derivative of the same raw sources.

---

## Data Sources (Your Pipeline vs Competitors)

| Source | You | SteamAnalyst | CSMarketCap |
|--------|-----|--------------|-------------|
| Steam Community Market | via CSFloat/CSGOTrader | Yes | Yes |
| Skinport | via CSGOTrader | Yes | Yes |
| Buff163 | via CSGOTrader | Yes | No |
| CSFloat | via CSGOTrader | No | No |
| CSMoney | via CSGOTrader | Yes | No |
| Youpin | via CSGOTrader | No | No |
| Waxpeer | — | Yes | Yes |
| DMarket | — | Yes | Yes |
| MarketCSGO | — | Yes | Yes |
| Lis-Skins | — | Yes | Yes |
| WhiteMarket | — | No | Yes |

**Stale as of 2026-08-09:** the pipeline now carries **9 ask feeds plus 1 excluded bid
feed** (`docs/research/2026-08-09-model-and-data-research.md` §4b). Waxpeer, Lis-Skins and
MarketCSGO are marked "—" above but are all collected daily today — as listing depth, not
price (`backend/collectors/supply_depth.py:479-486`).

**Observation:** You cover 6 major sources. Competitors cover more total marketplaces but at shallower depth. Your advantage is the historical Parquet archive (2013+), ML pipeline, and direct source access.
