# Next accuracy and market-indicator opportunities

**Date:** 2026-09-08  
**Scope:** Research note only. No modelling or serving change is authorized by this document.

## Bottom line

Further improvement is more likely to come from better market-flow data and category-specific
targets than from a larger LightGBM model or additional price technicals.

The project's existing evidence says that price-only directional prediction is near its useful
limit. The GBM centre was 6–11% worse than a last-price/random-walk centre at 3d, 7d, and 14d on
the available served replay, while the featureless per-item climatology produced narrower,
better-calibrated ranges than the modelled volatility scale. The latest volatility-ranking GBM is
better aligned with the product: its local A/B reported 8.1%, 6.1%, and 3.6% narrower bands at
matched 80% coverage for 3d, 7d, and 14d, and a neutral result at 30d. It remains gated pending
future served validation.

The practical implication is:

1. Optimize range quality and move/exceedance probabilities rather than raw direction.
2. Add information that is not already encoded in the price series.
3. Model cases, stickers, and ordinary skins separately where their economic mechanisms differ.
4. Require every apparent improvement to survive a date-clustered, leakage-safe, placebo-controlled
   test and a later served-data confirmation.

## Ranked opportunities

### 1. Steam order-book dynamics

This is the highest-value genuinely new dataset.

Candidate features:

- Bid/ask spread and spread change.
- Bid-depth versus ask-depth imbalance at 1%, 5%, and 10% from the book.
- Book slope and concentration near the best price.
- Estimated price impact for fixed $25, $100, and $500 orders.
- Listing additions, removals, and replacement/churn rates.
- Median listing age, 24-hour listing inflow, and stale-listing share.
- Cancel-to-sale or removal-to-sale ratios when transaction matching permits them.

The existing supply collector already records ask ladder quantiles, depth within 5% and 10%,
listing age, 24-hour inflow, and an order-independent listing-ID digest. The current forecaster
largely reduces the supply panel to listing count, its rolling z-score/change, and a
supply-to-volume ratio. The richer fields are therefore collected but not yet evaluated as model
inputs.

The local `supply-2026-08.parquet` contains only one snapshot day (2026-08-06), so it cannot
support a temporal test yet. As of June 2026, cs2.sh documents full-depth Steam bid/ask order-book
history at hourly and daily resolution from 2026-06-09 onward. Start accumulating it now, but do
not train until enough independent forecast dates exist.

Primary targets should be forward absolute return, interval miss probability, and band width at
fixed coverage. Direction can remain a secondary diagnostic.

Sources:

- <https://cs2.sh/docs/changelog/dedicated-steam-endpoints>
- `backend/collectors/supply_depth.py`
- `backend/models/forecaster.py::_add_supply_depth_features`

### 2. Repair native Steam transaction volume

This is the strongest previously measured candidate for a directional improvement. The corrected
historical A/B found approximately +1.50pp, +1.88pp, +1.44pp, and +0.96pp directional accuracy at
3d/7d/14d/30d respectively, with a null shuffled-feature placebo at every horizon. It was not
shipped because the historical Steam-volume sidecar did not have a production-equivalent live
feed.

Recommended features from native Steam sales:

- Within-item log-volume surprise versus a trailing median.
- Turnover: sales divided by visible listings or near-touch depth.
- Volume acceleration over 1d/7d/30d windows.
- An explicit no-sale/missing-print state rather than silently converting absence to zero.
- Price movement per unit volume.
- Volume surprise interacted with spread expansion and order-book imbalance.

cs2.sh documents native Steam median sale price and volume back to 2013 at daily resolution and
hourly data from May 2026. That source is much closer to the quantity used in the positive
historical A/B than Skinport's different volume definition.

Before enabling anything, rerun the original baseline/treatment/shuffled-placebo harness on the
new feed and then confirm it on future served dates. A historical-source win is not sufficient.

Sources:

- `docs/changelog/2026-08-15-volume-features-remeasured.md`
- <https://cs2.sh/docs/changelog/dedicated-steam-endpoints>

### 3. Case-specific supply-consumption model

Community case analysis consistently focuses on supply consumption rather than technical
indicators. Candidate variables include:

- Estimated openings divided by visible circulating supply.
- Net visible supply depletion over 7d and 30d.
- Active, rare, or discontinued drop-pool status.
- Estimated new weekly drops from player activity and drop-pool share.
- Case expected value and ROI relative to the key/opening cost.
- Investor accumulation: falling visible supply without a corresponding opening increase.
- Substitution and cannibalization following new cases, terminals, rental changes, or Armory
  releases.

These features should feed a case-only model or head. They are undefined for ordinary skins and
would be sparse/noisy in the global model.

Reddit provides useful hypotheses here, including analyses that track case openings, Steam visible
supply, and their ratio. These figures are estimates and visible listings are not total supply,
so they must be treated as noisy covariates rather than ground truth.

Sources:

- <https://www.reddit.com/r/csgomarketforum/comments/1ftx3gl/discussion_september_case_observations/>
- <https://www.reddit.com/r/csgomarketforum/comments/1f6hqdc/discussion_september_case_observations/>
- <https://www.reddit.com/r/csgomarketforum/comments/136uge5/discussion_trying_to_estimate_the_supply_of_cases/>

### 4. Sticker application and craft velocity

For stickers, actual consumption is more informative than generic social sentiment.

Candidate features:

- Daily/weekly change in total known applications.
- Application velocity divided by visible market supply.
- 4x/5x craft creation velocity.
- Market sales relative to new applications.
- Price and availability of visually similar substitutes.
- Capsule sale status and time since removal from the in-game store.
- Team/player tournament performance or roster events, interacted with the affected sticker.

CSFloat's database/listing metadata can support applied-sticker and craft queries, although database
coverage is observational rather than a complete census. Substitute products must be modelled:
the September 2026 Cologne discussion shows that omitting cheaper ranked versions can invert the
interpretation of circular-sticker application demand.

This should also be a sticker-only head, with rankings or relative value scored within the relevant
collection rather than against all CS2 items.

Sources:

- <https://www.reddit.com/r/csgomarketforum/comments/1w80av2/d_cologne_2026_application_rate_update/>
- <https://www.reddit.com/r/csgomarketforum/comments/1n98pwb/q_how_do_you_get_info_about_application_rate_and/>
- <https://docs.csfloat.com/>

### 5. Reddit and social data as an event detector

Do not restore the old generic positive/negative sentiment features. Production has no social
rows, and the previous mention-count and sentiment features ranked outside the top 20 at every
horizon. Reddit is more useful for discovering structured events and hypotheses than as a direct
sentiment score.

A better authenticated text pipeline would detect:

- Item/category mention-velocity shocks.
- Supply-disappearance, whale-buying, manipulation, and pump/exhaustion narratives.
- Newly discovered trade-up paths.
- Pro-player, streamer, or tournament exposure.
- Valve-update and drop-pool speculation.
- Disagreement: rising attention paired with falling price or weakening order-book depth.

Each event must retain its first-seen timestamp, source, engagement velocity, entity mapping,
author/history quality, and novelty relative to prior posts. Test conventional and reversed
sentiment separately; never let posts published after the forecast cutoff enter a feature.

The June 2026 CSTrader preprint reports useful ablations for liquidity and reversed sentiment, but
it evaluates a single highly volatile 2025 period and its public configuration models a 2% fee.
It is evidence that the hypothesis is testable, not evidence that it will transfer to this system's
universe, fee assumptions, horizons, or served regime.

Sources:

- `backend/AGENTS.md` social-feature warning
- `docs/changelog/2026-07-22-social-feature-audit.md`
- <https://arxiv.org/abs/2606.31461>
- <https://github.com/IatomicreactorI/CSGOTrading>

## Model and product recommendations

### Keep the range-first objective

The strongest product is still a calibrated range forecaster. Prefer these metrics:

- Interval coverage beside median and p90 width.
- Conditional coverage by price tier, volatility, liquidity, and item family.
- Weighted interval score or CRPS where practical.
- Calibration and Brier score for `exceed_p`/`anomaly_p`.
- Recall and precision for unusually large moves.
- Fee-clearing lift for any claim presented as tradeable.

Raw directional accuracy is secondary and must always be shown beside the realised-down-rate
baseline and the Pesaran–Timmermann result, per the backend invariant.

### Prefer specialized heads over one universal feature matrix

Use shared price-history features but separate heads or models for:

- Ordinary skins: order-book/transaction flow and range quality.
- Cases: openings, drop supply, EV, and inventory depletion.
- Stickers/capsules: application consumption, capsule supply, substitutes, and tournament events.
- Rare instance-sensitive skins: only after reliable float/pattern-level observations exist.

This prevents category-specific drivers from being diluted across thousands of irrelevant rows.

### Re-evaluate the centre simplification

The August served replay found that a last-price centre beat the GBM centre at 3d/7d/14d and that
the 30d result had only two clean dates. Re-run `centre_vs_lastprice.py --gate` once every horizon
has enough fresh dates. If the result holds, simplifying the centre to last price may improve point
accuracy while removing substantial training and artifact complexity.

## Required experimental protocol

For every candidate:

1. Define the target and decision rule before running the experiment.
2. Preserve the production universe and the `horizon + 13` embargo.
3. Split by forecast date; never treat thousands of correlated items on one day as independent.
4. Report distinct dates/months and the fraction of observations from the largest month.
5. Include a capacity-matched shuffled-feature placebo.
6. Test both pooled performance and the stratum where the mechanism should operate.
7. Require no material regression at other horizons or on core coverage guardrails.
8. Confirm any offline win on future served dates before enabling it.
9. Include transaction costs, spread, depth, and executable price whenever a trade claim is made.
10. Treat missing source data as missing, not as a numeric zero.

## Priority order

1. Begin or verify continuous Steam order-book collection.
2. Restore production-equivalent native Steam sales volume and rerun its existing A/B.
3. Accumulate enough order-book dates to test turnover, imbalance, depth, age, and churn.
4. Build a small case-only research panel for openings, visible supply, and drop-pool state.
5. Build a sticker-only application-velocity panel if CSFloat coverage and rate limits permit it.
6. Add structured, authenticated Reddit/event collection only after the market-flow panels are
   reliable.

Low priority: more price technicals, model capacity, generic sentiment, Google Trends, or player
counts. These either repeat already weak information or have already shown poor out-of-sample
transfer.
