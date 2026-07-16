# Feature Engineering Deep Dive — Prediction Accuracy Improvements

Date: 2026-07-14
Extends: `docs/research/accuracy-opportunities.md`

---

## Table of Contents

1. [Current Feature Landscape](#1-current-feature-landscape)
2. [Category & Item Identity Features](#2-category--item-identity-features)
3. [Supply & Liquidity Features](#3-supply--liquidity-features)
4. [Steam Ecosystem Features](#4-steam-ecosystem-features)
5. [Market Microstructure Features](#5-market-microstructure-features)
6. [Cross-Sectional Enhancement](#6-cross-sectional-enhancement)
7. [Event Feature Overhaul](#7-event-feature-overhaul)
8. [Seasonal & Calendar Features](#8-seasonal--calendar-features)
9. [Interaction Features](#9-interaction-features)
10. [Social Sentiment Analysis (Reddit/Twitter)](#10-social-sentiment-analysis-reddittwitter)
11. [Implementation Plan](#11-implementation-plan)

---

## 1. Current Feature Landscape

### Currently engineered in `_compute_price_features()` (lines 139-292)

| Category | Features | Count |
|----------|----------|-------|
| Lag prices | 1d, 3d, 7d, 14d, 30d, 60d | 6 |
| Returns | 1d, 3d, 7d, 14d, 30d, 60d (winsorized ±500%) | 6 |
| Rolling mean | 7d, 14d, 20d, 30d, 60d | 5 |
| Rolling std | 7d, 14d, 20d, 30d, 60d | 5 |
| Rolling min/max | 7d, 14d, 20d, 30d, 60d | 10 |
| Z-score | price_zscore_30d | 1 |
| Vol regime | vol_regime_60_30 | 1 |
| Trend divergence | trend_divergence_30_60 | 1 |
| Price acceleration | price_accel_7d | 1 |
| Log returns | log_return_1d, log_return_7d | 2 |
| Autocorr proxy | autocorr_1d, autocorr_7d | 2 |
| Bollinger Bands | bb_upper, bb_lower, bb_pct_b, bb_width | 4 |
| RSI | rsi_14 | 1 |
| MACD | macd_line, macd_signal, macd_histogram | 3 |
| Support/Resistance | distance_to_support, distance_to_resistance, high_low_range_30d | 3 |
| Volume | lags, means, stds, zscore, log change, price confirmation | 12 |
| Missing flags | volume_missing, rsi_missing, macd_missing | 3 |

### Currently in `_add_cross_sectional_features()` (lines 381-421)

| Feature | Description |
|---------|-------------|
| market_return_{1/7/14/30}d | Mean return across all items per date |
| item_return_vs_market_{lag}d | Item return minus market return |
| market_volatility_30d | Mean of item-level 30d price std |
| market_volume_mean_30d | Rolling 30d mean of daily market volume |
| item_volume_vs_market_30d | Item volume / market volume |
| market_regime_{bull,bear,range} | Market state based on ±5% threshold |

**Key gap**: There is no concept of **category/group/collection** anywhere. Every item is treated independently with only a global market signal. This is the single biggest missing signal.

---

## 2. Category & Item Identity Features

### 2.1 Item Metadata Extraction

Item names follow a parseable convention. Examples:

```
AK-47 | Redline (Field-Tested)
StatTrak™ M4A4 | Desolate (Factory New)
★ Bayonet | Doppler (Factory New)
Sticker | s1mple (Holo) | Paris 2024
Glove | Driver Gloves | Imperial Plaid (Field-Tested)
Agent | Special Agent Ava | FBI
```

Parse into structured fields:

```python
weapon: str        # "AK-47", "M4A4", "Bayonet", "Driver Gloves"
skin_name: str     # "Redline", "Desolate", "Doppler", "Imperial Plaid"
quality: str       # "Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"
is_stattrak: bool  # True if "StatTrak" prefix
is_souvenir: bool  # True if "Souvenir" prefix
is_knife: bool     # True if "★" prefix (knife/glove)
is_sticker: bool   # True if "Sticker |" prefix
is_agent: bool     # True if "Agent |" prefix
is_glove: bool     # True if "Glove |" or "★ Glove" (but ★ covers it)
is_case: bool      # True if ends with "Case" or "Capsule"
```

Already partially implemented in `_parse_item_name()` in `api/routes/market.py:59`. Needs to be extracted into a shared utility and enhanced.

**Proposed features from item identity:**

| Feature | Type | Description |
|---------|------|-------------|
| `is_stattrak` | binary | StatTrak items trade at premium |
| `is_souvenir` | binary | Souvenir items have different supply dynamics |
| `is_knife` | binary | Knives are high-value, low-volume |
| `is_sticker` | binary | Stickers have different pricing (tournament-driven) |
| `is_glove` | binary | Gloves are rare, high-value |
| `is_case` | binary | Cases have deterministic supply (drop rotation) |
| `is_agent` | binary | Agents are utility items |
| `quality_rank` | int | 1-5 ordinal: FN=5, MW=4, FT=3, WW=2, BS=1 |
| `has_stattrak_float` | binary | StatTrak FN vs non-StatTrak FN have different premia |

### 2.2 Category-Level Aggregates (Highest ROI)

Group items by `weapon` (all AK-47 skins, all AWP skins) and compute category-level signals parallel to the market-level ones.

```python
cat_groups = df.groupby(["date", "weapon"])

features:
    weapon_return_{1/7/14/30}d       — mean return of all items with this weapon
    item_return_vs_weapon_{lag}d     — item return minus weapon return
    weapon_volatility_30d            — std of weapon-group prices
    weapon_volume_mean_30d           — mean volume across weapon group
    item_volume_vs_weapon_30d        — item volume / weapon-group mean
```

Same pattern for `collection`, `case_origin`, and `skin_name`:

| Group-by | Rationale |
|----------|-----------|
| `weapon` | All AK-47 skins move together due to gameplay popularity |
| `collection` | Items from same case/collection are released together |
| `skin_name` | Same skin across wears (FN vs FT spread predicts direction) |
| `quality` | All FN items share a premium/discount factor |

**Category return correlation estimation:**
Before implementing, run a diagnostic: for each weapon category, compute the mean pairwise return correlation across its items. If >0.3, the category signal is worth adding. This confirms the hypothesis.

### 2.3 Cross-Quality Spread Features

For items with multiple wear levels of the same skin:

```python
# For each (weapon, skin_name) group, compute quality spreads
quality_spread_fn_ft = price_fn / price_ft - 1  # FN premium over FT
quality_spread_fn_mw = price_fn / price_mw - 1
quality_spread_mw_ft = price_mw / price_ft - 1
quality_spread_ft_ww = price_ft / price_ww - 1

# Direction of quality spread (widening/narrowing)
quality_spread_change_7d = quality_spread_fn_ft - quality_spread_fn_ft_7d_ago

# If FN premium is widening, FN prices are rising faster → bullish for FN, bearish for FT
```

Why this works: In bull markets, higher-wear items appreciate first (cheaper entry), then FN catches up. In bear markets, FN holds value better. The spread is mean-reverting and predicts direction.

### 2.4 Skin Popularity Score

Compute a trailing 90-day volume rank for each skin (not item — aggregate across wears):

```python
skin_total_volume_90d = sum(volume across all wears of this skin)
skin_volume_percentile = percentile_rank(skin_total_volume_90d among all skins)
```

Popular skins have different volatility profiles than niche skins. Combined with `is_knife`, `is_stattrak` etc., this helps the model learn different regimes for different item tiers.

---

## 3. Supply & Liquidity Features

### 3.1 Sell Listings (Active Supply)

The CSMarketAPI backfill captures `sell_listings` per item (`csmarketapi_backfill.py:109`). This is the number of active sell orders on Steam — a direct supply measure. Currently **not used** in feature engineering.

```python
# From the items table / backfill metadata
supply_listings          — raw count of active listings
supply_listings_log      — log(1 + listings) for normalization
supply_listings_zscore   — z-score vs item's own history
supply_change_1d/7d      — % change in listings
supply_to_volume_ratio   — listings / volume (overhang ratio)
```

**Why it's powerful**: A rising `supply_to_volume_ratio` means supply is growing faster than demand → bearish. A falling ratio means demand is absorbing supply → bullish.

### 3.2 Multi-Source Supply Signals

The CSGOTrader aggregator fetches from 7 sources. Some sources provide buy/sell depth:

- **Buff163**: provides `highest_order` (buy order price, i.e., bid) — see `csgotrader_aggregator.py:297-301`
- **Steam**: provides `last_24h`, `last_7d`, `last_30d`, `last_90d` average prices

Proposed source-level features:

```python
# Spread: Buff163 sell price vs Buff163 buy price (bid-ask spread)
buff_spread = (buff163_price - buff163_buy_price) / buff163_price

# Source divergence: std of prices across all sources
source_price_divergence = std(steam_price, skinport_price, buff163_price, csfloat_price, ...)

# Source count: how many sources have data for this item
source_count

# Source reliability: for each item, track which sources consistently match
source_agreement_7d = fraction of sources within 5% of median price
```

Wider spreads → lower liquidity → higher volatility risk. Source divergence may indicate stale data or market fragmentation.

### 3.3 Volatility Skew

```python
# Semi-deviation (downside vs upside volatility)
downside_returns = returns[returns < 0]
upside_returns = returns[returns > 0]
downside_sigma_30d = std(downside_returns)  # if enough observations
upside_sigma_30d = std(upside_returns)
vol_skew_30d = upside_sigma_30d / downside_sigma_30d

# If vol_skew > 1, upside moves are more volatile (possible bubble)
# If vol_skew < 1, downside moves are sharper (possible panic)
```

### 3.4 Price Level Features

```python
# Log price (scale-invariant)
price_log = log(price)

# Price tier
price_tier = 0  # < $1
price_tier = 1  # $1-$5
price_tier = 2  # $5-$20
price_tier = 3  # $20-$100
price_tier = 4  # $100+

# Minimum tick size impact
# Cheap items move in larger % increments due to Steam's price granularity
tick_size_impact = 0.01 / price  # 1 cent as fraction of price
```

---

## 4. Steam Ecosystem Features

### 4.1 Player Count (SteamCharts)

CS2 player count is a leading indicator for market activity. More players → more games → more skin demand → more cases opened → more supply.

```python
# Fetch from Steam API or SteamCharts
player_count                    — current concurrent players
player_count_change_7d          — % change in players over 7 days
player_count_zscore_30d         — deviation from recent norm
player_count_ma_30d             — smoothed player count

# Detrended: player count relative to 90-day average
player_count_detrended = player_count / player_count_ma_90d
```

Implementation: Add a scheduled task (weekly) that fetches player count from the Steam API and stores it in the events table or a new `steam_metrics` table. The forecaster then joins this data by date.

**Update Jul 16:** Permutation test showed **zero causal impact** for player count features (real accuracy matched shuffled accuracy to within 0.03pp across all horizons). The features were removed from the forecaster. The collector pipeline is preserved for monitoring/dashboard use but does not feed the model.

### 4.2 Tournament Calendar Features

Currently, events are in the DB but only used as exponential decay of "has an event happened." Made more powerful:

```python
# For each item that is a sticker, collector item, or case:
tournament_active          — binary: is a major/event happening right now
tournament_phase           — "qualifier", "group_stage", "playoff", "final", "off"
days_until_next_major      — integer (hype builds before majors)
days_since_last_major      — integer (post-major dip)
major_prize_pool           — larger prize pools create more hype
major_region               — "europe", "americas", "asia"

# Team/player relevance
# For player stickers: does the team/player have a match today?
team_match_today           — binary
team_tournament_performance — wins/losses in current tournament
team_eliminated            — binary (team eliminated → sticker prices crash)
```

This is especially powerful for stickers, which can 10x during majors and crash after.

### 4.3 Steam Sale & Seasonal Calendar

```python
# Steam seasonal sales affect overall spending
is_steam_summer_sale       — binary (late June)
is_steam_winter_sale       — binary (late December)
is_steam_spring_sale       — binary
is_steam_autumn_sale       — binary
days_to_next_steam_sale    — integer
days_since_last_steam_sale — integer

# CS2 operation timing
is_operation_active        — binary
days_since_operation_start — integer
operation_week             — operation week number (1-16)
```

Steam sales create a massive dip in CS2 skin prices (players sell skins to fund game purchases) followed by a recovery. This is a highly predictable pattern.

---

## 5. Market Microstructure Features

### 5.1 Price Level Dynamics

```python
# Round-number support/resistance
dist_to_round_number = abs(price - round(price)) / price
# Items near round numbers ($10, $25, $50, $100) have more resistance

# Historical resistance levels from past 90 days
# Density of prices near current level
price_density_30d = number of days price was within 2% of current level
```

### 5.2 Momentum Decay Features

```python
# How quickly momentum decays (mean reversion speed)
return_decay_7_14 = return_7d - return_14d
# If this is large positive, the item had a spike that's now fading

# RSI divergence
rsi_divergence_7d = rsi_14 - rsi_14.shift(7)
# RSI falling while price rising → bearish divergence

# MACD crossover
macd_crossover = (macd_line > macd_signal).astype(int)
macd_crossover_strength = macd_histogram / abs(macd_signal).replace(0, np.nan)
```

### 5.3 Volume Price Divergence

Currently there's `volume_price_conf_1d` and `_7d` which multiply return by a binary volume flag. More nuanced:

```python
# On-Balance Volume (OBV) — cumulative volume * direction
price_direction = np.sign(price_change)
obv = (price_direction * df["volume"]).groupby(df["item_id"]).cumsum()
obv_zscore_30d = zscore(obv, 30)

# OBV divergence: price going up but OBV going down → weak rally
obv_divergence_7d = obv_change_7d - return_7d

# Volume-weighted average price (VWAP) deviation
vwap_7d = (price * volume).rolling(7).sum() / volume.rolling(7).sum()
vwap_deviation = (price - vwap_7d) / vwap_7d
```

### 5.4 Velocity & Churn

```python
# Volume-to-listings ratio (churn)
# How many trades happen relative to active listings
churn_ratio = volume_30d / max(sell_listings, 1)
churn_ratio_change_7d = churn_ratio - churn_ratio_7d_ago

# Price velocity (returns per unit volume)
price_velocity_7d = abs(return_7d) / (volume_mean_7d + 1)
# High velocity = large price moves with little volume (manipulation risk)
```

---

## 6. Cross-Sectional Enhancement

### 6.1 Beyond Global Market — Category Cross-Section

Currently only global market features exist. Add category-level cross-section:

```python
# Weapon category features
for lag in [1, 7, 14, 30]:
    df[f"weapon_return_{lag}d"] = weapon_group[ret_col].transform("mean")
    df[f"item_return_vs_weapon_{lag}d"] = df[ret_col] - df[f"weapon_return_{lag}d"]

# Collection features (same logic)
# Quality tier features (same logic)

# Cross-sectional rank features
item_return_rank_7d = weapon_group[ret_col].rank(pct=True)
item_volume_rank_30d = weapon_group["volume"].rank(pct=True)

# Replace global rank signals with category-relative signals:
# "this AWP skin is in the top 10% of all AWP skins by volume"
```

### 6.2 Market State Refinement

Current regime detection uses a hard ±5% threshold on median market 30d return. Replace with:

```python
# Market momentum state (more granular)
market_return_30d              — current value
market_return_30d_percentile  — percentile vs rolling 365-day history
market_volatility_percentile   — percentile vs rolling 365-day vol

# Regime: 5 states instead of 3
market_regime = "crash"       # return < -10%
market_regime = "bear"        # return < -3%
market_regime = "range"       # return between -3% and +3%
market_regime = "bull"        # return > +3%
market_regime = "mania"       # return > +10%

# Regime duration (how long have we been in this regime?)
market_regime_duration_days   — count of consecutive days in same regime
```

### 6.3 Leader-Follower Dynamics

Items within the same category often have leader-follower relationships. The most liquid/high-volume item leads price moves.

```python
# For each weapon category, find the highest-volume item as "leader"
# Compute leader return vs item return
weapon_leader_item = weapon group item with max trailing 30d volume
weapon_leader_return_1d       — return of the leader
weapon_leader_return_7d
item_return_lag_vs_leader_1d  — item return minus leader return (1d lag)
item_return_lead_vs_leader_1d  — item return minus leader return (next day)

# "Catch-up" signal: if leader moved up but item didn't → item likely to rise
catch_up_score = (weapon_leader_return_3d - item_return_3d) if leader_rose else 0
```

---

## 7. Event Feature Overhaul

### 7.1 Current State

Events are used with fixed decay constants and simple density counts (lines 318-379). There are 5 types: `major`, `operation`, `case_drop`, `update`, `game_update`.

Problems:
- Decay constants are hardcoded (not learned)
- All items get the same event signal regardless of relevance
- No event magnitude/importance signal
- No pre-event anticipation signal

### 7.2 Item-Specific Event Relevance

```python
# For each item, does this event type matter?
event_relevance_major = 1 if is_sticker else 0.5 if is_case else 0
event_relevance_operation = 1 if is_case else 0.5 if is_skin else 0
event_relevance_case_drop = 1 if is_case else 0.5 if is_skin else 0
event_relevance_update = 0.5 if is_skin else 0.2  # gameplay balance changes

# Modified event decay with relevance weighting
event_decay_major_weighted = event_decay_major * event_relevance_major
```

### 7.3 Event Magnitude

```python
# Tournament prize pool (larger → more hype)
major_prize_pool_millions

# Case age (newer cases have more volatile prices)
case_age_days = today - case_release_date

# Operation case inclusion
is_in_active_drop_pool — binary: is this case still dropping?
```

### 7.4 Pre-Event Anticipation

```python
# Days until next major/operation
days_to_next_major
days_to_next_operation

# Anticipation effect: stickers start rising ~14 days before a major
anticipation_window = max(0, 14 - days_to_next_major) / 14
```

### 7.5 Learnable Decay Constants

Instead of hardcoded τ values, make decay constants learnable per event type by treating them as a hyperparameter. Approach:

1. For each event type, try τ values [7, 14, 21, 30, 60, 90] as separate features
2. Train and let feature importance reveal the best τ
3. Collapse to the top-2 most important τ values per type

This can be run as an offline experiment without deploying all variants.

---

## 8. Seasonal & Calendar Features

### 8.1 Current State

Simple time features: day_of_week, month, quarter, day_of_year, cyclic encodings, is_weekend, item_age_days.

### 8.2 Academic Calendar Effects

```python
# School holidays → more play time → more skin trading
is_school_holiday       — binary (summer, winter break, spring break)
is_north_america_holiday — Christmas, Thanksgiving, Black Friday
is_europe_holiday
is_china_holiday         — Chinese New Year (huge impact on CS2 market)
```

China is a massive CS2 market. Chinese New Year causes significant market disruption as players liquidate or take breaks.

### 8.3 Payday Cycles

```python
# End of month / beginning of month = more disposable income
days_until_end_of_month
days_since_start_of_month
day_of_month
is_first_week_of_month
is_last_week_of_month
```

### 8.4 Weekly Seasonality (Enhanced)

Current `is_weekend` is binary. More granular:

```python
# Trading volume and price patterns vary by day of week
day_of_week_onehot — 7 binary columns (or use existing sin/cos)

# Weekend ramp-up: Friday-Monday
is_pre_weekend     — Thursday/Friday (people get ready to play)
is_weekend         — Saturday/Sunday (peak play time)
is_post_weekend    — Monday/Tuesday (post-weekend dip)

# Time of month for case openings
# More cases are opened on weekends → more supply on the following Tuesday
supply_shock_day = (day_of_week == 1).astype(int)  # Tuesday
```

---

## 9. Interaction Features

### 9.1 Price x Volume Regime

```python
# Four regimes based on price and volume direction
price_up_volume_up     = (return_1d > 0) & (volume_log_change_1d > 0)     # strong bull
price_up_volume_down   = (return_1d > 0) & (volume_log_change_1d < 0)     # weak bull
price_down_volume_up   = (return_1d < 0) & (volume_log_change_1d > 0)     # strong bear
price_down_volume_down = (return_1d < 0) & (volume_log_change_1d < 0)     # weak bear
```

### 9.2 Event x Item Type

```python
# Major event x sticker item interaction
major_x_sticker = event_decay_major * is_sticker

# Case drop x case item
case_drop_x_case = event_decay_case_drop * is_case

# Operation x skin item
operation_x_skin = event_decay_operation * (1 - is_sticker - is_case)
```

### 9.3 Volatility x Volume

```python
# Low-volume = high-volatility interaction
low_volume_high_vol = (volume_percentile < 0.2) * (price_std_30d / median_std)

# Market vol x item beta
market_beta_30d = covariance(item_return, market_return, 30) / variance(market_return, 30)
# If beta > 1, item amplifies market moves
market_stress_beta = market_beta_30d * market_volatility_30d
```

### 9.4 Quality Spread x Market Regime

```python
# Quality spread narrowing/widening relative to market regime
# In bull markets, lower-quality items outperform (quality spread narrows)
# In bear markets, higher-quality items hold value (quality spread widens)
quality_spread_market_interaction = quality_spread_fn_ft * (
    1 if market_regime_bull else -1 if market_regime_bear else 0
)
```

---

## 10. Social Sentiment Analysis (Reddit/Twitter)

### 10.1 Why Sentiment Matters for CS2 Skin Prices

The CS2 skin market is heavily driven by hype cycles. Many price movements are preceded by social media activity:

- **Reddit post on r/GlobalOffensive or r/csgomarketforum** mentioning a skin → price spike within hours
- **Pro player using a skin in a major tournament** → tweet/clip goes viral → demand surge
- **YouTuber/streamer showcase** → direct price impact (documented 5-30% bumps)
- **Twitter breaking news** (operation teaser, case reveal, trade-up contract discovery)
- **Whale manipulation signals** — coordinated buy/sell walls discussed in Discord servers often leak to Reddit

Sentiment acts as a **leading indicator** — social media moves minutes to hours before market prices update on the 6-hour collection cycle.

### 10.2 Architecture

```
┌───────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Reddit API      │     │  Twitter API     │     │  Discord /       │
│   (Pushshift +    │     │  (X API v2)      │     │  Telegram        │
│    PRAW)          │     │                  │     │  (scrape)        │
└────────┬──────────┘     └───────┬──────────┘     └───────┬──────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Collection Layer (run every 30-60 min)           │
│                                                                      │
│  • Fetch posts from subreddits (GlobalOffensive, csgomarketforum,    │
│    csgotrade, skins, csgobetting)                                    │
│  • Search keywords: skin names, weapon names, "case", "operation",   │
│    major team names, "market", "crash", "pump"                       │
│  • Filter: posts with specific item mentions (regex match item names)│
│  • Fetch tweets matching similar keywords (via API search)           │
│  • Store raw posts in `social_mentions` table                        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     NLP / Sentiment Scoring Layer                    │
│                                                                      │
│  • Classify polarity: positive (bullish), negative (bearish),        │
│    neutral                                                           │
│  • Score each mention: -1 (strong bear) to +1 (strong bull)         │
│  • Extract entities: item name, weapon, tournament, player           │
│  • Compute:                                                          │
│    - Mention velocity (posts per hour per item)                      │
│    - Sentiment moving average (3h, 12h, 24h)                        │
│    - Sentiment divergence from price trend                           │
│    - Author credibility score (karma, age, follow count)            │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Feature Store (daily snapshot for forecaster)    │
│                                                                      │
│  • Aggregate to daily per-item features:                            │
│    - sentiment_mean_{3/7/14/30}d                                     │
│    - mention_count_{lag}d                                            │
│    - sentiment_std_{lag}d (disagreement = uncertainty)              │
│    - sentiment_change_1d (momentum of sentiment)                     │
│    - is_social_trending (binary: above 95th percentile mentions)    │
│    - top_source_weight (Reddit / Twitter / Discord split)           │
└──────────────────────────────────────────────────────────────────────┘
```

### 10.3 Database Schema

Add to `backend/database.py`:

```python
class SocialMention(Base):
    __tablename__ = "social_mentions"

    id = Column(Integer, primary_key=True)
    item_id = Column(String(255), ForeignKey("items.item_id"), nullable=False)
    source = Column(String(50), nullable=False)        # "reddit", "twitter", "discord"
    post_id = Column(String(255), nullable=False)      # unique post identifier
    author = Column(String(255))
    title = Column(String(1000))
    body = Column(Text, nullable=True)
    url = Column(String(1000))
    posted_at = Column(DateTime, nullable=False)
    collected_at = Column(DateTime, default=utcnow_naive)

    # NLP outputs
    sentiment_score = Column(Float, nullable=True)     # -1 to +1
    sentiment_label = Column(String(20), nullable=True) # "bullish", "bearish", "neutral"
    entity_type = Column(String(50), nullable=True)     # "item", "weapon", "tournament", "player"

    # Engagement signals
    score = Column(Integer, default=0)                 # Reddit upvotes / Twitter likes
    num_comments = Column(Integer, default=0)

    __table_args__ = (
        Index('idx_social_item_id', 'item_id'),
        Index('idx_social_posted_at', 'posted_at'),
        Index('idx_social_source', 'source'),
        UniqueConstraint('source', 'post_id', name='uq_source_post'),
    )


class SocialAggregate(Base):
    """Daily aggregate of social sentiment per item — consumed by forecaster."""
    __tablename__ = "social_aggregates"

    item_id = Column(String(255), ForeignKey("items.item_id"), primary_key=True)
    date = Column(Date, primary_key=True)
    source = Column(String(50), primary_key=True)

    mention_count = Column(Integer, default=0)
    sentiment_mean = Column(Float)
    sentiment_std = Column(Float)
    mean_engagement = Column(Float)                    # avg upvotes/likes
    positive_ratio = Column(Float)                     # fraction of bullish mentions
    negative_ratio = Column(Float)                     # fraction of bearish mentions
```

### 10.4 NLP Approaches (Ordered by Complexity)

#### Option A: Keyword Polarity Heuristic (Simplest, ~100 lines)

```python
BULLISH_KEYWORDS = {"invest", "buy", "hold", "moon", "underrated", "sleeping",
                    "pump", "rising", "demand", "rare", "low supply", "ez profit",
                    "going up", "rally", "breakout", "bulish", "to the moon"}

BEARISH_KEYWORDS = {"sell", "crash", "dumping", "overrated", "overpriced",
                    "scam", "dead", "drop", "falling", "panic", "manipulation",
                    "bubble", "exit", "rug", "pump and dump", "bearish", "trap"}

def score_text(text):
    text = text.lower()
    bull_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
    bear_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
    total = bull_count + bear_count
    if total == 0:
        return 0.0
    return (bull_count - bear_count) / total
```

**Accuracy**: ~60-65%. Fast, no dependencies. Good enough for directional signal.

#### Option B: Fine-Tuned DistilBERT (Recommended, ~300 lines)

Use a lightweight transformer fine-tuned on financial/crypto sentiment data (social media domain adapts well to CS2 market):

```python
from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment",
    max_length=512,
    truncation=True
)

def score_post(title, body):
    text = f"{title}. {body[:500]}" if body else title
    result = sentiment_pipeline(text)[0]
    label = result["label"]    # "positive", "negative", "neutral"
    score = result["score"]
    if label == "positive":
        return score       # 0 to 1
    elif label == "negative":
        return -score      # -1 to 0
    return 0.0
```

**Accuracy**: ~75-80%. Runs modestly on CPU (no GPU needed for batch processing 500 posts/hour). DistilRoBERTa is 82MB — feasible for a serverless function or background task.

#### Option C: Fine-Tune on CS2 Market Data (Best, ~500 lines)

Collect 2,000 historical Reddit posts + corresponding price movements. Manually label as bullish/bearish/neutral. Fine-tune a small model. This captures CS2-specific slang ("case god", "blue gem", "purple pattern", "craft", "trade up", "stattrak premium").

**Accuracy**: ~85-90% on CS2-specific text.

### 10.5 Feature Generation for Forecaster

The forecaster reads from `social_aggregates` and generates features at the same daily frequency as price data:

```python
def _add_social_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add social sentiment features to the feature matrix."""

    # Load daily aggregates per item
    social = self._load_social_aggregates()  # item_id, date, sentiment_mean, mention_count, etc.

    # Merge onto main dataframe
    df = df.merge(social, on=["item_id", "date"], how="left")

    # Fill missing with 0 (no mentions is itself informative)
    for col in ["sentiment_mean", "mention_count"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)

    # Rolling features
    for window in [3, 7, 14, 30]:
        grouped = df.groupby("item_id")["sentiment_mean"]
        df[f"sentiment_ma_{window}d"] = grouped.rolling(window, min_periods=1).mean().values
        df[f"sentiment_std_{window}d"] = grouped.rolling(window, min_periods=1).std().values
        df[f"mention_count_{window}d"] = (
            df.groupby("item_id")["mention_count"]
            .rolling(window, min_periods=1).sum().values
        )

    # Sentiment change (momentum)
    df["sentiment_change_1d"] = (
        df["sentiment_mean"] - df.groupby("item_id")["sentiment_mean"].shift(1)
    )

    # Sentiment divergence: price rising but sentiment falling = warning
    df["sentiment_price_divergence_7d"] = (
        df["return_7d"] - df["sentiment_ma_7d"] * 5  # scale sentiment to % return scale
    )

    # Social trending flag: top 5% of items by mention volume
    df["is_social_trending"] = (
        df.groupby("date")["mention_count_7d"]
        .transform(lambda x: x > x.quantile(0.95))
    ).fillna(0).astype(int)

    return df
```

### 10.6 Leading Indicator Validation

Sentiment data is unique because it's **not aligned to daily close** like price data. A Reddit post at 2pm can affect the same day's price. To properly capture the leading nature:

```python
# Option: Use yesterday's sentiment to predict today's return
# This guarantees no lookahead bias and tests the leading hypothesis
df["sentiment_mean_lag_1d"] = df.groupby("item_id")["sentiment_mean"].shift(1)

# Better: Use morning sentiment to predict daily close
# Partition day: collect all mentions before 12:00 UTC, predict close at 23:00 UTC
```

**Expected impact**: +2-5pp directional accuracy for items with high social volume (stickers during majors, newly released cases, popular skins). Zero impact for long-tail items with no social mentions.

### 10.7 Integration into Pipeline

```
┌────────────────────────────────────────────────────────────┐
│ Social Collection Container (new, runs every 30 min)       │
│                                                            │
│  backend/collectors/social_collector.py                    │
│  • RedditCollector — PRAW + Pushshift for historical       │
│  • TwitterCollector — X API v2 (free tier: 1500 posts/mo) │
│  • DiscordCollector — optional webhook scrape              │
│  • SocialLogger — writes to social_mentions table          │
│                                                            │
│  Dependencies to add to requirements.txt:                  │
│    praw>=7.7.0                                             │
│    transformers>=4.30.0                                    │
│    torch>=2.0.0 (cpu-only)                                 │
│    tweepy>=4.14.0                                          │
│    scikit-learn>=1.3.0 (already present)                    │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│ NLP Pipeline (runs after each collection batch)           │
│                                                            │
│  backend/collectors/social_nlp.py                          │
│  • Load unprocessed mentions                               │
│  • Score sentiment (DistilBERT or keyword heuristic)       │
│  • Extract item entities (fuzzy-match against item names)  │
│  • Write scores back to social_mentions table              │
│  • Upsert daily aggregates to social_aggregates            │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│ Forecaster Integration (daily forecast run)                │
│                                                            │
│  • forecaster.py reads social_aggregates by date           │
│  • _add_social_features() merges onto feature matrix       │
│  • Features enter the same train/predict pipeline           │
└────────────────────────────────────────────────────────────┘
```

### 10.8 Social Sentiment Dashboard

The frontend could surface social signals alongside predictions (new API endpoint):

```json
GET /items/{item_id}/social
{
    "item_id": "AK-47 | Redline (Field-Tested)",
    "last_24h": {
        "mention_count": 47,
        "sentiment_mean": 0.32,
        "top_source": "reddit",
        "top_posts": [
            {
                "title": "Redline is so undervalued right now",
                "score": 342,
                "sentiment": "bullish",
                "url": "...",
                "posted_at": "2026-07-14T14:30:00Z"
            }
        ]
    },
    "trending_since": "2026-07-12",
    "social_volume_percentile": 0.92
}
```

### 10.9 Practical Considerations

| Concern | Solution |
|---------|----------|
| **Rate limits** | Twitter: 1500 posts/month free; Reddit: 60 req/min via PRAW with cache; stagger collection across 30-min intervals |
| **Storage** | ~500 posts/day = ~50MB/year for mentions table; aggregates are tiny |
| **Latency** | NLP runs async after collection; features are ready by next forecast run (23:00 UTC) |
| **Noise** | Spam, bots, reposts — filter by author karma (>100), post age (>1 min), remove duplicates |
| **Item matching** | Fuzzy match post text against item names using same `_normalize_name` logic from aggregator |
| **Data freshness** | Social data is useless if stale → 30-min collection; 1-hour max staleness |
| **Twitter API cost** | Use free tier initially; if valuable, upgrade to Basic ($100/mo for 10K posts/mo) |
| **False signals** | Coordinated manipulation ("pump and dump" groups) — track author-level patterns |

---

## 11. Implementation Plan

### Phase 1 — Quick Wins (1-2 days)

| Feature | Location | Lines to change |
|---------|----------|-----------------|
| Extract item metadata (is_stattrak, is_knife, is_sticker, etc.) | New utility + `_add_cross_sectional_features()` | New ~40 lines |
| Add item identity binary features | `_add_cross_sectional_features()` | ~20 lines |
| Add quality tier ordering | `_add_cross_sectional_features()` | ~15 lines |
| Add price_log, price_tier | `_compute_price_features()` | ~5 lines |
| Add event_relevance weighting | `_add_event_features()` | ~15 lines |
| Refine market regime from 3-state to 5-state | `_add_cross_sectional_features()` | ~10 lines |

### Phase 2 — Weapon Category Features ✅ (Completed 2026-07-15)

| Feature | Status | Notes |
|---------|--------|-------|
| Parse weapon from item name | ✅ Done | Via `models/steam_types.py` — extracts weapon_type from Steam `type` field |
| weapon_return_{1/7/14/30}d | ✅ Done → **Removed** | Permutation test showed zero causal signal (shuffling changed accuracy ≤0.05pp). Removed Jul 16. |
| item_return_vs_weapon_{lag}d | ✅ Done → **Removed** | Same — removed with the rest of weapon-type cross-sectional |
| weapon_volatility_30d | ✅ Done → **Removed** | Same |
| weapon_volume_mean_30d | ✅ Done → **Removed** | Same |
| item_volume_vs_weapon_30d | ✅ Done → **Removed** | Same |
| Rarity one-hot (11 dummies) | ✅ Done → **Kept** | Strong causal signal verified by permutation test (+10-12pp). Rarity alone carries the signal. |
| item_return_rank_7d (category percentile) | ❌ Not yet | Would require rank transform |
| Leader-follower features | ❌ Not yet | Higher effort |
| Parse skin_name from item name | ❌ Not yet | Shared utility not extracted |
| **A/B test result** | **+0.66pp avg** | 3d: +1.92pp, 14d: +0.79pp. Later permutation test showed signal was entirely from rarity; weapon-type features added only noise. |

### Phase 3 — Quality Spread & Cross-Wear Features (2-3 days)

| Feature | Effort |
|---------|--------|
| Parse (weapon, skin_name, quality) triples | Extend Phase 1 utility |
| Compute cross-quality spreads for each skin group | ~50 lines |
| Quality spread change over time | ~20 lines |
| Skin popularity score (90d volume rank) | ~15 lines |
| Quality × market regime interaction | ~10 lines |

### Phase 4 — Supply & Liquidity (2-3 days)

| Feature | Effort |
|---------|--------|
| Incorporate `sell_listings` from Item metadata | ~15 lines |
| Supply_to_volume ratio | ~10 lines |
| Supply change features | ~10 lines |
| Multi-source spread (Buff163 bid-ask) | ~20 lines |
| Source divergence (std across sources) | ~15 lines |
| Churn ratio | ~10 lines |

### Phase 5 — Social Sentiment (4-6 days)

| Component | Effort |
|-----------|--------|
| Database schema (social_mentions, social_aggregates) | ~40 lines |
| Reddit collector (PRAW + item name matching) | ~150 lines |
| Twitter collector (tweepy + rate limit handling) | ~120 lines |
| NLP pipeline (DistilBERT or keyword heuristic) | ~100 lines |
| Daily aggregation + feature generation | ~80 lines |
| Forecaster integration (_add_social_features) | ~60 lines |
| Frontend API endpoint for social data | ~50 lines |
| Total | ~600 lines |

### Phase 6 — Advanced Features (ongoing)

| Feature | Effort |
|---------|--------|
| OBV and volume-price divergence | ~25 lines |
| Volatility skew (semi-deviation) | ~15 lines |
| Learnable event decay constants | Experiment design |
| Interaction feature combinations | ~30 lines |
| VWAP deviation | ~15 lines |

### Feature Count Summary

| Phase | New Features | Est. Impact | Calibrated Est. | Status |
|-------|-------------|:-----------:|:----------------:|--------|
| Current | ~70 | baseline | baseline | ✅ |
| Phase 1 (identity) | ~15 | +1-3pp | +0-1pp | ✅ |
| Phase 2 (category) | ~35 | +2-5pp | **+0.66pp** actual (rarity only; weapon-type removed) | ✅ |
| Phase 3 (cross-wear) | ~10 | +2-4pp | **1-2pp** | Pending |
| Phase 4 (supply) | ~15 | +3-6pp | **1-3pp** | 🛑 **Dropped (2026-07-16)** |
| Phase 5 (sentiment) | ~20 | +2-5pp | **1-3pp** | Pending |
| Phase 6 (advanced) | ~20 | +1-3pp | **0-2pp** | Pending |
| **Total** | **~115 new** | **~5-10pp cumulative** | **~3-7pp** | 2 of 6 phases done |

**Note:** Phase 2 delivered +0.66pp — ~20% of the 2-5pp estimate. Existing features (cross-sectional, price technicals, ~55 features) already captured most signal. The marginal gain of Phase 2 was entirely from rarity dummies (12 features); weapon_type (22 features) and cross-sectional (6 features) were pure noise and removed. Later phases should be calibrated to **30-50% of pre-estimate** for novel signal, and must be validated via A/B + permutation test before full integration.

### Validation Approach

Each phase should be validated independently:

1. **Walk-forward evaluation**: Run `evaluate_forecaster.py` before and after each phase
2. **Feature importance**: Track which new features enter the top-10 by importance
3. **Ablation**: Remove one feature group at a time to measure marginal contribution
4. **Correlation check**: Ensure new features pass the existing 0.95 correlation pruning
5. **Directional accuracy lift**: Target +2pp minimum per phase to justify complexity

### Implementation Order Recommendation

```
Completed: Phase 1 → Phase 2
Remaining: Phase 4 → Phase 3 → Phase 5 → Phase 6
           (supply)  (quality) (sentiment) (advanced)
```

Phases 1 and 2 complete. ~~Phase 4 (supply/liquidity — listing counts, churn ratio, source spreads) was the highest remaining ROI opportunity~~ 🛑 **Dropped (2026-07-16)** — see `docs/research/accuracy-opportunities.md` §1 DECISION and `docs/changelog/2026-07-16-drop-supply-depth.md`. Top remaining work is now model architecture (regime-switching, Ridge head). Phase 5 (sentiment) remains lower priority.
