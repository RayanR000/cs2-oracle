# Product

<!-- impeccable:product-schema 1 -->

## Register

product

## Platform

web

## Users

Traders and collectors navigating the CS2 skin market. Traders seek short-to-medium-term price signals, volatility windows, and liquidity data to execute informed flips. Collectors track long-term value trends, wear spreads, and portfolio composition. Both share the need for clarity — turning raw market noise into actionable conviction without drowning in data.

The operator is currently the only real user. The product is built to be opened to CS2 traders later, and it doubles as a portfolio piece that demonstrates the engineering and analytical depth behind it. Design work must therefore hold up for a first-time visitor who has never seen the pipeline, not only for the person who built it.

## Product Purpose

A precise analytical dashboard for CS2 market intelligence. Users search items, inspect price history through high-quality charts, and surface trends that inform buying, selling, or holding decisions. Success looks like a user closing a session with confident conviction — whether to trade, collect, or wait.

## Positioning

**Probabilistic forecasts with published accuracy.** Neighboring CS2 price sites report what a skin costs now and what it cost before. CS2 Oracle issues quantile forecasts — q10 / q50 / q90 at 3, 7, 14, and 30-day horizons — and then publishes how well those forecasts did, from an automated daily backtest that scores MAE, MAPE, and directional accuracy against resolved outcomes.

The accountability is the position, not the prediction. A competitor can copy a forecast number; it cannot copy a public error record it has not been keeping. This obliges the product to show forecast uncertainty and measured accuracy honestly, including when the numbers are unflattering.

## Operating Context

- **Daily pipeline, not a live ticker.** Prices are daily closes assembled by a scheduled aggregator; forecasts and backtests run downstream of it on GitHub Actions. The interface reflects a market as of a date, and should never imply real-time tick data.
- **Sessions are analytical, not transactional.** Users compare, chart, and decide inside CS2 Oracle, then execute the trade somewhere else. The product never handles a transaction.
- **Extended sittings.** Comparing wear tiers and horizons across many items is long-dwell work, which is why visual comfort is a functional requirement here rather than a theme preference.
- **Operator-run infrastructure.** One person runs the collection, model training, and backtest schedule. Anything the interface promises has to survive an unattended pipeline.

## Capabilities and Constraints

**Confirmed capabilities**

- Item catalog with search, trending, and per-item detail; price history across sources with wear-tier selection.
- ML price forecasts: LightGBM quantile ensemble producing q10/q50/q90 at horizons `[3, 7, 14, 30]` days (`backend/models/forecaster.py`).
- Automated accuracy backtesting written to `prediction_accuracy` and exposed at `/accuracy/*`.
- Market signals: undervalued, overheated, and momentum. Market-event timeline with correlation scoring.
- Steam OpenID sign-in and a read-only Steam inventory snapshot for the portfolio view.
- Multi-source daily collection across 7 markets, archived to Parquet and queried via DuckDB.

**Constraints future work must respect**

- **The catalog is effectively closed.** There is no working path to add new items to the backfilled cohort — Steam discovery is disabled and broken, and the CSMarketAPI free-tier quota is exhausted without resetting. Do not design flows that assume the item universe grows, and do not promise coverage of an arbitrary skin.
- **Accuracy is reported per price tier.** `backtest/scoring.py` tiers items at $1 / $5 / $20 / $100; production accuracy reporting uses the ≥$1 cohort. Any accuracy figure shown to a user must state the cohort it describes, because the all-tiers number and the ≥$1 number differ materially.
- **Not every item has a forecast.** Coverage is bounded by archive history, and items can be excluded from scoring. The interface needs a real "no forecast for this item" state, not a hidden or zero-filled one.
- **Green CI is not evidence of collection.** Collectors have stored zero rows while their workflows reported success. Freshness claims in the UI must come from the data, never from a workflow badge.
- **Daily granularity is the floor.** No intraday prices exist. Do not design ranges or controls finer than one day.
- **Auth is Steam-only**, and the portfolio view depends on a publicly visible Steam inventory.

**Undecided**

- Whether the product is ever opened to real external users, and on what terms. Recorded as open; do not assume a pricing, account, or subscription model exists.

## Brand Commitments

- **Name:** CS2 Oracle.
- **Personality:** analytical, precise, calm. The authority of a professional terminal without the coldness of one. Precise enough for serious analysis, calm enough for extended sessions. Communicates trust through restraint — every element earns its place.
- **Binding anti-references:**
  - Compact and overwhelming "data-vomit" dashboards that bury signals in density.
  - Generic AI-generated website patterns (flat, uninspired layouts, identical card grids, side-stripe borders).
  - Neon/hyper-enthusiastic "gaming" aesthetics that prioritize flash over clarity.
  - Trading platform cliches (Robinhood green/red exuberance, TradingView overload, "stonks" culture).
  - SaaS-cookie-cutter patterns (hero-metric templates, numbered section markers, identical feature cards).

## Evidence on Hand

**Real, and available to show**

- `price-archive/` — multi-year daily price archive in Parquet, spanning roughly 13 years of history, queryable via DuckDB.
- Measured forecast accuracy from the automated backtest (`prediction_accuracy`, surfaced at `/accuracy/*`): MAE, MAPE, and directional accuracy by horizon and price tier.
- Live multi-source pricing across 7 markets, plus market-event correlation output and supply-depth data.
- Real Steam item imagery and real Steam inventory data for signed-in users.

**Absent — never fabricate**

- No users, testimonials, case studies, press, logos, or adoption numbers exist.
- No pricing, plan, subscription, or revenue exists.
- No uptime, SLA, or "trusted by" claim is supportable.
- **No illustrative accuracy figures.** Every accuracy, MAE, or hit-rate number shown anywhere must be read from the backtest, with its cohort and horizon named. A plausible-looking placeholder here is the single most damaging thing this product could ship, because published accuracy is its entire position.

## Product Principles

- **Clarity over density.** Whitespace and hierarchy guide the eye. The interface never competes with the data — it reveals it.
- **Honest uncertainty.** A forecast is shown as a range with a measured track record, never as a confident single number. Where the model is weak, the interface says so.
- **Asset-grounded data.** High-quality skin imagery anchors every analysis. The tool stays connected to the CS2 ecosystem and never abstracts into pure numbers.
- **Precision tools.** Every control behaves the same way everywhere — predictably, consistently, without surprise. The tool disappears into the task.
- **Purposeful information architecture.** Every route earns its existence. Published accuracy is a product surface because trust is the product; internal model instrumentation (`/ab-test/*`) is a diagnostic and stays out of the primary path.

Visual principles — typography, palette, motion, component rules — live in `docs/design.md`, not here.

## Accessibility & Inclusion

- WCAG AA (Standard) compliance as a baseline.
- High contrast for body text on dark backgrounds.
- Respect for reduced motion — all animations degrade gracefully.
- Comfortable extended reading: the soft dark surface is a functional accessibility requirement for long analytical sessions, not a style preference.
