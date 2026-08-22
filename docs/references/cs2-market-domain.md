# CS2 market domain reference

**What this is.** The durable external knowledge about how the CS2 skin market works — venues,
fees, what moves prices, how people make money, and how prices get faked. Built 2026-08-16 from
four independent web-research passes, each claim cross-checked and graded, then reconciled against
what this repo has *measured*. **Repo-side claims and citations re-verified 2026-08-21**; the
external magnitudes are as-of 2026-08-16 and were not re-researched.

**What this is not.** A findings doc. Our own results live in `research/`; this file is the
outside world. Where the two disagree, the disagreement is stated and **our measurement wins** —
every external magnitude here is a hypothesis until we reproduce it.

**Evidence grades.** `[V]` verified, 2+ independent sources. `[S]` single source. `[A]` asserted —
trade blog, forum, or marketing; direction may be real, the number is not evidence. `[MEASURED]`
measured in this repo, with the citation.

> ⚠️ **The one thing to internalise:** almost everything the market talks about is a *price level*
> attribute (rarity, float, pattern, StatTrak, craft). Those are cross-sectional and static — they
> cannot be forecast because they do not change. The forecastable surface is small: supply-pool
> state changes, Valve rule changes, and liquidity/regime shifts. §3 is the table that matters.

---

## 1. There is no such thing as "the price"

A quoted CS2 price is meaningless without three qualifiers. Mixing them across sources is the
single most common way to manufacture a fake signal.

| Qualifier | Options | Why it bites |
|---|---|---|
| **Venue** | Steam, Buff163, YouPin, CSFloat, Skinport, DMarket, market.csgo, Waxpeer, Bitskins | Steam payout is **Steam-Wallet-locked and non-withdrawable** `[V]`. Every other venue cashes out. Steam therefore trades at a structural *premium* — the gap is a wallet-lock discount, **not** an arbitrage. |
| **Price type** | last sale / lowest ask / highest bid / median of listings | Steam's graph is **last sale**; most aggregators publish **lowest ask** and say so (csgoskins.gg) `[V]`. A "spread" computed across two sources that mean different things is noise. |
| **Fee side** | gross (buyer pays) / net (seller receives) | Steam listing pages serve the **buyer** price; our archive stores **net** (`collectors/price_history_sources/cs2_prices_tracker.py:23`, `scripts/backfill_steam_listing_history.py:124`). |

Our own archive already carries this hazard twice over: `aggregator_sync` is a **blended**
construct across venues, and post-2026-03-22 "volume" from the HF import is a **listing count**,
not a trade count (`research/volume-data.md`). Never pool across 2026-03-22.

### Venue fees

| Venue | Seller fee | Payout | Notes |
|---|---|---|---|
| Steam | **15% = 5% Steam + 10% publisher** `[V]` | Wallet only, non-withdrawable `[V]` | Round trip **16.1%** as we model it (`backtest/friction.py`) |
| Buff163 | 2.5% seller + 2.5% buyer `[V]` | CNY, China-domestic | The de-facto global reference price `[A]` |
| YouPin898 | ~1% `[S]` | CNY, China-domestic | One source puts its **volume above Buff163** `[S]` — if true, trader shorthand ("the Buff price") points at the wrong venue |
| CSFloat | 2% `[V]` | Cash | Round trip **2.0%** (`friction.py`); sources conflict on whether a buyer fee exists |
| Skinport | 6–12%, tiered `[V-ish, rate disputed]` | Cash | Round trip **8.7%** (`friction.py`) |
| DMarket / Waxpeer / market.csgo / Bitskins | ~5% | Cash | Bitskins reportedly took withdrawals offline mid-2026 `[S]` — verify before using it as a cash-out reference |

**The cash-out wedge.** Getting *cash* out of a Steam sale costs the 15% fee **plus** a commonly
cited 20–30% discount to convert wallet balance to money `[A]`. Any strategy priced against Steam
quotes but settled in cash faces ~35–45% of round-trip drag, not 15%.

## 2. The Steam fee is piecewise, not a constant

Steam computes the 5% and 10% components with independent floor-rounding, so the **effective rate
rises sharply as price falls**: ~16% at $0.50, up to ~66% at $0.03 `[V, consistent across 3 fee
calculators]`. Our `1.1607` multiplier is a flat approximation of that stack.

This is the outstanding **item 5** of §10 in `research/2026-08-07-cs2-forecasting-research.md`
("Fix the listing-page backfill to use Steam's real cent-ceiling schedule" — there is no item
"5c"; the sub-item that exists is 5b, `ingested_at`). The external evidence says the fix is not
a better constant — it is a **rounding-aware piecewise function**, and the error is concentrated
exactly where our cohort floor already cuts (<$1). Low priority for the ≥$1 served cohort;
disqualifying for any sub-$1 work.

## 3. Static level vs time-varying — the only split that matters

| Attribute | Effect | Forecastable? |
|---|---|---|
| Rarity tier, weapon identity | Order-of-magnitude level jumps `[A]` | **No** — static, and absorbed by the market factor |
| Exterior / float bucket | Discrete jump at bucket edges; 20–60% within-FN spread `[S]` | **No** — static per instance |
| StatTrak | +20–60%, occasionally 1.5–3× `[V]`; ~10% of unboxes `[V]` | **No** — static flag |
| Souvenir | Not a fixed premium; near-parity to 5–20×, driven by tournament/player/sticker `[V]` | **No** — but the *variance* is a warning: a souvenir "price" per item name is a mixture |
| Pattern index (blue gem, Doppler phase, Fade %, Fire & Ice) | 8–15× for gem phases `[V]`; tier-1 Case Hardened seeds $50k–$1M+ `[V]` | **No** — and see the hazard below |
| Sticker craft premium | Dominates the very top of the market — Kato 2014 holos $50k–80k *per sticker*, a 6× iBUYPOWER craft ~$400k `[V]` | **No** — per-instance |
| **Supply pool state** (case active vs retired, collection vaulting, capsule print ended) | Retired = fixed and shrinking supply → structural drift `[V]` | **Yes** |
| **Valve rule changes** (trade-up mechanics, drop-pool edits, trade holds) | Largest observed repricings, and they hit *categories* in opposite directions | **Yes** — as regime shifts, not daily dummies |
| **Lifecycle stage** | Release spike → decay → floor `[V]` (no source quantifies the curve) | **Yes**, weakly |

> 🔑 **The finite-supply rule (the screener's scarcity axis; replaces the earlier age proxy).**
> As of 2026 the **active weekly drop pool is only 5 containers** (~20% each): Kilowatt, Revolution,
> Dreams & Nightmares, Sealed Dead Hand Terminal, Sealed Genesis Terminal. The **rare-drop pool was
> removed ~2026-01-08/09**, so every other/legacy case now gets **zero** new supply. So: a
> case-supplied item is **finite** unless its case is one of those 5 `[V, community trackers Mar
> 2026; no Valve statement — re-verify, the active set rotates]`. To map an item to its case in
> this repo: parquet `type_meta_crate_id` is a **dense project code**, not the raw id — invert
> `price-archive/item-metadata-bymykel-codes.json["crate"]` (rawid→dense), then look up the name in
> `backend/runtime/bymykel/crates.json`. ⚠️ Metadata gives only the *earliest* crate, so an item
> also present in an active case can be mislabelled finite.

> 🔑 **The pattern hazard.** For Case Hardened, Fade, Doppler, Marble Fade and knife patterns, a
> single price series *per `market_hash_name`* is structurally wrong — the true distribution is
> multimodal on hidden pattern-index/float metadata that no aggregate feed exposes `[V]`. These
> item names carry irreducible quote noise. They are candidates for exclusion or for a wider band,
> not for better features.

## 4. Liquidity and spread — where the market's folklore is wrong

Trade sources uniformly claim wide spreads on rare/expensive items and tight spreads on liquid
mid-tier skins `[V, directionally consistent — and no source gives a number]`.

**Our measurement says the opposite, monotonically:**

| Tier | <$1 | $1–5 | $5–20 | $20–100 | $100–1k | ≥$1k |
|---|---|---|---|---|---|---|
| Median relative spread | **35.5%** | 21.1% | 17.3% | 17.3% | 10.8% | **5.2%** |

`[MEASURED]` — BUFF163 paired `starting_at`/`highest_order`, n = 22,449 items (`backtest/friction.py`).

Expensive items are the *liquid* ones; sub-$1 quotes are midpoints of a book nobody could transact
in. Not one web source found this, and it inverts retail intuition — but it is the reason a DA
pooled across tiers is uninterpretable, and it is measured. **Trust ours.** (Caveat, from the
module's own docstring: the bands don't align with `scoring.price_tier`'s cuts and are assigned by
a nearest-band rule.)

**Cost hurdle for any claimed edge:** best-venue round trip (2.0% CSFloat) + the tier spread. At
$1–5 that is ~23%; on Steam it is ~37%. A 3-day forecast edge of a few percent is not tradeable.

## 5. Event timeline

Only events with an observed market reaction. **Breadth is the column that matters** — market-wide
shocks are absorbed by our date effect by construction and are true nulls for feature work; a
category rotation is not.

| Date | Event | Reaction | Breadth |
|---|---|---|---|
| 2023-09-27 | CS2 replaces CS:GO | −15–30% over ~90 days; CS:GO-exclusive items up `[V]` | Market-wide + rotation |
| Late 2024 → 2025-05 | Bull run | +15% by Jan 2025 `[V]`; +17.4% trailing 90d in May `[S]` | Market-wide |
| 2025-07 | "Trade Protected Items" — 7-day lock on trade-received items, reversible `[V]` | Friction/liquidity, no price-level shock | Market-wide |
| 2025-10 | Market cap crosses $6B `[V]` | Pre-crash peak | — |
| **2025-10-22/23** | **Trade-up contracts accept 5× Covert → knife/glove** `[V]` | Cap **$6B → $3.5B in ~30h** `[V]`; knives/gloves **−70–80%**; low-tier Coverts **up sharply** (one MP7 $8.77 → $104) `[V, direction; magnitudes vary $2B–$2.5B across outlets]` | **Category rotation inside a market-wide drop** |
| ~2025-12-17 `[date disputed]` | Rare/legacy cases removed from the weekly drop pool `[V]` | Legacy case prices spiked | Category (discontinued cases) |
| Late 2025 | Valve bans gambling/case-site branding at licensed events `[S]` | Demand-channel restriction | Diffuse |
| H1 2026 | Trade cooldown moves from a synchronised 9am CET daily unlock to a **per-item 168h unlock at the trade hour** `[V, month disputed]` | Kills the daily supply-dump spike | Market-wide, microstructure |
| 2026-02-25 | **NY AG sues Valve** over loot boxes; injunction, restitution, treble damages `[V]` | No measured price effect yet | Regulatory overhang |
| 2026-03-09 | Federal consumer class action on loot boxes `[S]` | — | Regulatory overhang |
| Mid-2026 | Cap variously reported $6.9B–$8B `[S per figure]` | Recovery | — |

> ⚠️ **Two drop-pool dates in this file do not reconcile, and neither matches our own research
> doc.** §3 dates the rare-drop-pool removal ~2026-01-08/09; the row above says ~2025-12-17; and
> `research/2026-08-07-cs2-forecasting-research.md` R6 dates Valve's silent zeroing of the rare
> drop pool **2024-12-17/18**. Unresolved — do not build a regime dummy on any of the three until
> one is pinned to an archive-observable break.

> **Date reconciliation for the `tier × post` experiment.** Press dates the trade-up crash
> **Oct 23 2025**; our `ISteamNews` capture is **2025-10-22T23:00:20Z**
> (`research/2026-08-07-cs2-forecasting-research.md` C3). Same event. It landed **~20 seconds after
> the 23:00 UTC aggregator snapshot boundary**, so the first archive row that can contain it is
> 10-23. Assign `post` from 10-23, not 10-22. (That experiment is separately data-blocked — see
> `research/2026-08-15-armory-tier-post-preregistration.md`.)

## 6. Macro drivers — mostly the market factor

- **Chinese demand leads.** Repeatedly observed: gloves +25–50%, Doppler knives +13–31% in a week
  on Chinese buying `[V, direction]`. But **no measured lag exists in any source** — every
  "Buff leads Steam by X" claim is folklore `[A]`. Do not encode a lag we have not measured.
- **Crypto / risk appetite.** Skins co-move with risk assets in drawdowns — they are *not* a hedge
  `[V]`. A 6-year backtest has skins +54.5% vs BTC +990% and S&P +154% `[S]`, which contradicts the
  widely relayed "skins beat crypto" claim `[S, low provenance]`. Either way: a common factor.
- **Seasonality** (Steam sales, CNY, summer softness, autumn case bump) — every claim found was
  narrative-grade with no magnitude or statistics `[A]`. Weak priors at best.
- **Total market size** is not a knowable number: trackers disagree $3–4B vs $8B+ in mid-2026 with
  undisclosed methodology `[V that the disagreement exists]`. Never cite a market cap as a fact.

All of the above are **market-wide** and therefore already absorbed by our date effect. This is
bucket B in `research/2026-08-14-what-moves-skin-prices-web-reconsideration.md` — true nulls.

## 7. How money is actually made

| Strategy | Verdict | Data fingerprint |
|---|---|---|
| Holding retired cases/capsules on fixed supply | **Real structural edge** (scarcity), horizon in years. ROI claims of 5–20%/yr are blog-grade `[A]` | Slow low-vol drift, decaying volume, step changes at retirement dates |
| Pattern/float sniping | **Real** — information asymmetry on metadata the feeds don't carry `[V]` | Extreme within-name dispersion; the multimodality of §3 |
| Large-scale cross-venue arb / market making | **Real, but only for operators** who net in-kind and avoid repeated cash-out `[V]` | Buff/Steam ratio ~0.75–0.90, mean-reverting |
| Buying liquidity-driven crashes | **Real when the shock is friction, not supply** `[V]` | Breadth: whole market moves at once, on a patch date |
| Trade-locked item discounts | Edge shrank materially after the 2025 lock update `[V]` | The lock discount *is* a direct read on time-value of liquidity — widening = stress |
| Manual Buff→Steam arbitrage | **Treadmill** once the wallet-lock discount is netted `[V]` | — |
| Trade-up contracts | **Not structurally +EV** post-Oct-2025; a variance product. All "X% profitable" claims are marketing `[A]` | — |
| Sticker crafting | Real but path-dependent; craft value tracks team results, and a bad craft is irreversible | Premiums spike/decay with esports results |

**The through-line:** what survives is (1) fixed-supply scarcity, (2) information edges on hidden
metadata, (3) scale operators who avoid the fee stack. Everything retail-facing is a treadmill
after frictions — which is consistent with our own finding that directional skill exists and is
economically worthless against the fee.

## 8. Manipulation, and what it looks like in our data

This is the actionable half. We ingest public quote series; some of them are fabricated.

| Activity | Fingerprint in a price series |
|---|---|
| **Pump and dump** on thin-float items | Near-vertical rise on *low or falling* volume → volume spike concurrent with the crash (distribution) → settles above pre-pump baseline, far below peak. Documented 2025 cases: an EG sticker $few → $200+ → $10 in a week `[V, direction]`; M4A4 Buzz Kill $26 → $1,459 `[S, manipulation unconfirmed]` |
| **Wash trading** | **Volume jumps with no price discovery** — the transferable heuristic from NFT/crypto wash-trade literature is >500% day-over-day volume with <5% price change `[V concept, arXiv 2312.16603 / 2102.07001; the exact threshold is illustrative]` |
| **Money laundering** | Closed-loop repeated round-trips among a small account set at off-market prices, never resold outward. **Graph structure, not price** — invisible to us; we have no trade-graph data. A 2020 CS:GO study found one item at 14.1% of all trades in a 5-day window `[S, methodology unverified]` |
| **Gambling-site inventory promos** | Demand spikes correlated with promotional calendars rather than patches or esports events |
| **Stolen-inventory dumping** | High-tier item listed below market and sold near-instantly |

> **The wash-trade screen above was run 2026-08-16 and is NULL** `[MEASURED]`. On 4.7M Steam
> item-days, volume spikes come with **larger** moves, not flat ones — P(|Δp|<5%) is 0.243 on
> spike days vs 0.780 otherwise, a lift of **0.31×** where the fingerprint predicts >1. Max
> per-item flag rate is 0.8%. Caveat that matters: Steam is the venue where washing is most
> expensive (15% + 7-day hold), and the baseline filter removes exactly the thin items where the
> risk is alleged — so this is a null on the cohort we can see, not on the market. The laundering
> fingerprint is *graph structure*, which we have no data for at all.
> `research/2026-08-16-wash-trade-screen-and-volume-spike-exceedance.md`.

**Filters serious analysts use** (Skin Trackers' published index methodology `[V]`): price floor
≥$5, **listing count ≥30–50**, ≥12–24 months of history, 7-day rolling averages, and trimming the
top/bottom 1% of trades.

Against ours: our ≥$1 floor is looser than their ≥$5, and we apply **no listing-count filter**.
⚠️ **Both gaps were measured 2026-08-16 and neither should be closed by copying the methodology.**
Their ≥30–50 is a **Steam** count on an index of a few hundred items; ours is a max over four
non-Steam venues with structurally smaller books (cohort median **18** listings), so the same
number cuts **69% of the served cohort** and buys 0.011 of exceedance rate. The underlying effect
is real and monotone within price tier, but the follow-on this doc used to recommend — put
`log1p(listing_count)` in the **band width** instead of in a filter — was pre-registered and
**measured dead 2026-08-18, 0 of 3 horizons**: the thin buckets that carry the whole gradient
(1–5, 6–15 listings) hold <20 items in the served ≥$1 cohort and drop out, and the per-item band
scale — since 2026-08-19 a per-item **climatology** (`CLIMATOLOGY_SCALE`, default on), not the GBM
sigma — already absorbs listing information for the population actually served.
`research/2026-08-16-listing-count-floor.md`,
`changelog/2026-08-18-listing-count-conditioner-refuted.md`.

Most-manipulable tiers, in order: new/tournament stickers, freshly retired collections, thin-float
pattern items, anything sub-$5. These deserve wider bands, not point forecasts — which our product
already is.

## 9. Do not encode these

Repeated everywhere, measured nowhere. Every one is `[A]`.

- "Buff leads Steam by N hours/days" — no measurement exists in any public source.
- "StatTrak appreciates 15–25% faster", "cases return X%/yr", "pattern premiums are 10–100×" —
  price *levels* stated as *drift rates*.
- Any total market cap figure.
- Seasonality magnitudes (summer −5–10%, CNY bumps).
- Trade-up EV claims.
- The Oct-2025 crash's exact dollar magnitude ($2B / $2.5B / $1.7B / $615M all appear in print).

## 10. What would actually change our model

Ranked by leverage, and each is a data question, not a feature question:

1. ~~**Listing-count floor + volume/price co-movement screen**~~ — ⚠️ **all three parts measured
   2026-08-16; none survives as proposed.** The wash screen is null (spikes come with *larger*
   moves, lift 0.31×), the volume-spike exceedance feature it turned up is one episode (81% of 13
   years of spikes are Oct 2025), and the floor's ≥30 threshold is imported from a **Steam**
   index and does not transfer to our four-venue max — it costs 69% of the cohort to buy 0.011.
   The last remnant — `log1p(listing_count)` as a band-**width** conditioner — was then
   **measured dead 2026-08-18, 0/3 horizons**: the thin buckets that carry the gradient hold <20
   items in the served ≥$1 cohort. **Nothing in this item survives.** Reopen only if the iflow
   backfill materially grows the thin-listing ≥$1 cohort.
   `research/2026-08-16-wash-trade-screen-and-volume-spike-exceedance.md`,
   `research/2026-08-16-listing-count-floor.md`,
   `changelog/2026-08-18-listing-count-conditioner-refuted.md`.
2. ~~**Un-blended per-venue closes.**~~ — ⚠️ **tested and SHELVED 2026-08-19.** The wallet-lock
   wedge (§1) does make the Steam−Buff spread a *real, persistent, mean-reverting* quantity, and
   "Buff leads, Steam follows" stands as *description*. The **predictive** claim does not: four
   skeptical controls each moved it the wrong way, and breaking the shared quote collapses the
   cross-sectional IC ~70% to ~−0.06 (h ≤ 14), ~0 by h=30. Do not build the ingest.
   `research/2026-08-16-cross-venue-basis-steam-buff.md`.
3. **A rounding-aware Steam fee function** (item 5) — only matters if we ever go below $1.
4. **Pattern-sensitive item names flagged and widened**, not modelled.
5. A per-instance float/pattern feed and a pro-usage/attention feed remain the two genuinely
   missing datasets. Neither exists for free.

---

**Method note.** Four parallel web passes (venue microstructure; item attributes; event timeline;
strategies and manipulation), each instructed to grade sources and report contradictions rather
than resolve them. Contradictions that survived are stated above as contradictions. Where an
external claim met a repo measurement, the measurement is quoted with its file. Trade-blog
magnitudes are hypotheses; this document never promotes one to evidence.
