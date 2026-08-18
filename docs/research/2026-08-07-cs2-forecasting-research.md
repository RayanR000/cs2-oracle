# CS2 Item Price Prediction — Research Review

> 🔴 **BANNER ADDED 2026-08-16. Two things this doc presents as its strongest results are dead;
> its own existing banner stops at 2026-08-09 and does not cover either.**
>
> **1. C1 / cross-sectional reversal is refuted.** `:652` and `:668` (R16) still call it "the
> project's strongest measured predictor," which "beats the model at all four horizons" and
> "survives composition control." It does not:
> `changelog/2026-08-13-c1-refutation-survives-the-audited-anchors.md:9` — **"FAILS at 3 of 4
> horizons. C1 stays shelved."** The CV-positive/serving-negative gap is structural, not a cohort
> or anchor artifact (`…-cohort-geometry-is-not-c1s-gap.md`, `…-c1s-cv-edge-is-not-a-pre-2026-artefact.md`).
> This is the doc's most-cited positive result and every cross-reference to it inherits the
> refutation.
>
> **2. The replacement headline in the 2026-08-09 banner is itself invalid.** That banner
> withdraws the accuracy figure and substitutes `mean_rank_ic 0.1307 vs naive −return_1d 0.1656`
> as the trustworthy number. The 2026-08-11 label-denominator finding (`docs/README.md:19`)
> concludes **"No stored rank IC, DA or −return_1d comparison in this repo is safe to rank arms
> on"** — which includes that substitute.
>
> **3. Every `ab_test_*`-sourced figure here predates a broken harness family.**
> `changelog/2026-08-13-harness-family-repinned-off-steamcommunity.md`: three harnesses matched 0
> rows on `source='STEAMCOMMUNITY'`, six more never applied the production feature allowlist. They
> were repaired and **none re-run**. Treat the feature-contribution, supply, regime and ensemble
> deltas below as unmeasured.
>
> Still the best literature review in the repo — read it for the reasoning and the mechanisms,
> not for the rankings. `research/2026-08-16-research-docs-review.md` §3.

**2026-08-07.** Commissioned as a from-scratch design review of a CS2 item price
forecasting system. It is not written from scratch: this repo has already run most of the
standard playbook and killed it, so the review is organised around *what the existing
evidence permits*, not around a generic feature checklist.

Sources: a completed academic literature sweep; the repo's own measurement record
(`docs/changelog/`, memory ledger); `docs/references/data-inventory.md`. Three web-research
threads (venue microstructure, event history, prior art) are appended in §11–§13.

Confidence is marked throughout: **[MEASURED HERE]** = this repo measured it,
**[PUBLISHED]** = peer-reviewed or established preprint, **[REPORTED]** = single source,
**[UNVERIFIED]** = plausible, not checked.

---

> ## ⚠️ Corrections — 2026-08-09. Substantially stale verdict, not a retired document.
>
> **The analysis and the literature sweep are still the best in this repo** — §9's citations, §12's
> event history, §13's prior-art survey and §26's demand-proxy audit are unaltered. What has gone
> stale is the *verdict*: six load-bearing numbers are withdrawn or misattributed, the premise of
> the composition argument is refuted, and twelve items written as open have shipped or reversed.
> Original numbers are kept in place below and struck where short.
>
> **`docs/research/2026-08-09-next-steps.md` is the live action list.** The 2026-08-07
> second-pass corrections (**C1–C5**) immediately below are unchanged; where a point here is
> already settled there it says so rather than repeating it.
>
> ### The six load-bearing numbers
>
> **1. `~0.3pp` date-level MDE is not a measured MDE — it is a *required* effect size, and
> its own source calls it unreachable.** Appears at `:450-459`, `:640`, `:1133`, `:1245`,
> `:1553`, `:1857`, marked `[MEASURED HERE]`. The figure comes from
> `docs/research/accuracy-opportunities.md:57-60`, where 0.3pp is the MDE you would *need* —
> "**It is not reachable**, and no amount of item count changes it." And the changelog cited
> as its source retracts it: `docs/changelog/2026-08-06-date-level-exogenous-ingest.md:31-38`
> carries a 2026-08-07 correction putting the operative date-level floor at the fold-clustered
> **2.21–3.69pp** — "**Nothing in these two tables can be resolved below roughly 2pp**."
> Everything routed to the date-level frame *because the MDE is 10× better* — §6, §9 #9,
> §14's "what to do with it", §17's decomposition diagram, §23's closing instruction, §24
> rank 15, and final step 9 — rests on a number that does not exist. The date-level case
> survives on **variance**; it does not survive on measurability, and it faces the same ~2pp
> floor as everything else.
>
> **2. The `+3.50pp [+1.56, +5.98] at 30d` ≥$1 positive is withdrawn.** Cited at `:316`,
> `:364-366`, `:633`, `:1236`, `:1791` and `:1849-1851` as "the one positive" / "the only
> surviving positive". Re-derived on the current instrument as **+1.642pp [−0.809, +4.505],
> null**, with the instrument's own item-draw floor at **±3–4pp**
> (`docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`: "the stored +3.50pp is an
> un-purged number… not recoverable in isolation and should not be cited again"). Its stated
> *explanation* also fails: `:1287` L2 reads the h=30-only effect as "the look-ahead
> signature". The look-ahead is real — 876 items were used where 319 were knowable — and it
> is worth **−0.004pp [−0.664, +0.851]**. The floor shipped anyway, on determinism
> (`docs/changelog/2026-08-08-training-price-floor-shipped.md`).
>
> **3. Blocker 5d — the Steam soft-block — was a FALSE POSITIVE.** ~~`:695-702`: Steam
> "soft-blocks with HTTP 200 and a stripped shell — there is no 429", the block is IP-scoped
> and multi-hour, keep delay ≥6 s, stop on the first stripped page.~~ The page **answers from
> this egress at 0.36 requests per item**; the 230 KB shell is normal SSR. You must resolve
> the canonical `G<id>` first — the `market_hash_name` URL returns an empty shell that looks
> exactly like a block. See `docs/research/2026-08-09-next-steps.md` **D1**. This unblocks the
> doc's own "highest-value free source" and R13's expensive half. ⚠️ No code changed:
> `backfill_steam_listing_history.py:67` still fetches the name URL and the detector at `:271`
> still keys on the shell.
>
> **4. The `99 items / 116,111 rows` training config at `:261-265` is not production.**
> Subsampling stopped 2026-08-08: `backend/scripts/forecast_prices.py:51,63` set
> `DEFAULT_TRAIN_FEATURE_ROWS = 1_200_000` and `DEFAULT_TRAIN_MIN_MEDIAN_PRICE = 1.0`,
> measured on the shipped frame as **918 items / 986,065 rows**
> (`docs/changelog/2026-08-09-shipped-retrain-cost-measured.md:21`). The
> `Accuracy (≥$1, CV) 49.9 / 49.0 / 51.1 / 53.4` row is no longer producible either:
> `CV_DIAGNOSTIC_CLASSIFIER` defaults off (`dfafdfb`) and the shipped `meta.json` carries
> `mean_classifier_acc*: null`. The replacement headline is **`mean_rank_ic` 0.1307 vs naive
> `-return_1d` 0.1656, edge −0.0349** at 7d — i.e. the model loses to a one-line baseline at
> all four horizons. Minor arithmetic: 116,111 / 5.8M is **2.0%**, not 1.8%.
>
> **5. The `0–1.8% at every tier ≥$1` staleness split measures the wrong series.** `:396-400`,
> `:1203`, `:1579`, marked `[MEASURED HERE]`, used as "the cleanest split between real items
> and artifacts" and as the whole mechanism for `stale_run_days` (§24 rank 4, "highest-ranked
> new item-level feature").
> `docs/changelog/2026-08-08-frozen-price-runs-dropped-from-labels.md:18-21`: 0–1.8% is
> **resolved, 3-day-smoothed backtest anchors**. On the raw voted series the label path
> actually sees, ≥$1 staleness is **12–27%** (20.25pp overall at ≥$1). **The 37–42% vs 0–1.8%
> contrast does not exist on the series being modelled**, and neither number may be used to
> size the other. The feature shipped on the label side regardless — see the table below.
>
> **6. The `1.1607` Steam fee constant is synthetic — and §8 still presents the circular
> measurement as evidence.** `:512-518` marks `[MEASURED HERE]` "your own archive establishes
> the Steam fee empirically… a constant **1.1607** (p10 1.1565, p90 1.1656, within-item CV
> 0.0021)" — which is exactly what **C1** below retracts ("91.61% of pairs are the same
> numbers after dividing"). §8 carries no pointer to C1, and ~~`:751` "**Do this:** regress
> the ratio on price"~~ — C1 says done. The two passes also report incompatible *n* for the
> same measurement: **30,875** rows (§8, §11) vs **63,767** matched pairs (C1). *Consequence:*
> the **+16.1%** Steam round-trip breakeven at `:726` / `:742` is derived from the synthetic
> constant and is repeated ~8 times, including the closing framing at `:1874`. The published
> fee rule (buyer/net = 1.15 exactly) gives **+15.0%**. The qualitative argument — no
> realistic accuracy gain makes this tradeable — is unaffected.
>
> ### The composition / quoting-artifact premise is refuted
>
> `:487-489` ("your set moved from 5,542 to 41,725 items during 2026" — the premise for §7's
> hedonic index) and `:1161-1165` are the source-composition argument. It has now been measured
> and it does not bite: composition-stable rank IC is **+0.1027 on 181 dates** against an
> unconditional **+0.1023** on 185 at 3d, and the two agree to four decimal places at 7d
> (**+0.0842 vs +0.0842**) — `docs/research/2026-08-09-composition-stability.md`, commit
> `f833882`. §7's *product* case (a composition-controlled, publishable index nobody else
> has) survives untouched; its stated *empirical* motivation does not.
>
> ### Shipped since — items this doc writes as open
>
> | Where | Item | Settled by |
> |---|---|---|
> | `:628` T1 #3 | Stop the BUFF bid voting into consensus | DONE 2026-08-07 — `forecaster.py:1242` drops `BID_SOURCES` before the vote |
> | `:629`, `:1286`, `:1315-1316`, `:1789` | `--purge` default OFF; purge cuts at exactly `H`, "short by 13 days" | DONE 2026-08-08 — `embargo_days(h) = h + 13` (`forecaster.py:72`) feeds `_purge_overlapping_train_rows`; `walkforward_backtest.py:710` is `set_defaults(purge=True)` with `--no-purge` as the legacy escape |
> | `:631`, `:1276-1282`, `:1794`, `:1838` | "**No arrival timestamp exists anywhere**"; the 5-tuple `CANONICAL_PRICE_COLUMNS` | `backend/db/archive.py:40` now has **six** columns including `ingested_at` (2026-08-08; NULL for every earlier row, by construction) |
> | `:632` T1 #6 | Grep for `api.dmarket.com/exchange/v1` | **No-op.** No such reference exists; only a fee constant at `backend/backtest/friction.py:29` |
> | `:1509`, `:1638-1654`, `:1793`, `:1834` | Drop the 110 Doppler names | DONE 2026-08-08 — `phase_collapsed_sql_filter` in `backend/models/item_parser.py:138` |
> | `:1504`, `:1842` D2 | Drop / downweight frozen-price runs | DONE 2026-08-08 — `backend/models/staleness.py:54::stale_run_days` + label voiding. (As a *feature* it is still open; its evidence is item 5 above and does not hold) |
> | `:402-404`, `:1373-1383`, `:1449`, `:1824` | PT and rank IC as things to build | Both committed — `_within_date_rank_ic`, per-fold `rank_ic` beside `naive_rank_ic`, and Optuna's objective **is** within-date rank IC (`8be48c5`) |
> | `:1161-1165` | `n_ask_sources` as a column to add | Shipped (`48fd352`, `VOTED_CACHE_VERSION` 4 → 5). And 2026-03-22 / 07-09 / 07-10 **cannot** be "the best available guess at the cause of §3 defect 3": `_collection_shift_dates` (`forecaster.py:3081`) already fires on 03-22, 07-09, 07-10, 07-11 and 07-12 and voids the spanning labels |
> | `:1594` §24 #19 | `usd_cny` "Blocked: 7 days of FX history" | `price-archive/exchange-rates-history.parquet` is **13 years** deep and published |
> | `:1288` L3 | "early stopping stops late" as a leakage consequence | **Direction reversed.** Early stopping on the thin trailing window *destroyed* 23–88% of rank IC and produced 1-tree boosters; replaced by `FIXED_BOOST_ROUNDS` (`forecaster.py:618`) |
> | `:1188-1192` §16 | "**117** numeric columns survive shelving" | Now **123** (`docs/research/2026-08-09-model-and-data-research.md:222`). The `distance_to_*` vs `support_` prefix bug reported beside it is **still live and correctly reported** — `forecaster.py:213` |
> | `:1286`, `:1789` | "**ten** further `ab_test_*` harnesses" | **15** exist. The live residuals differ from the ones named: **9 of 15** call `phase_collapsed_sql_filter()` where invariant 2 requires `archive_universe_sql_filter()`, and **14 of 15** still early-stop against the window they score — only `ab_test_direction_labels.py` does not, because it calls the production estimator (`_fit_direction_classifier`, `early_stopping` default `False`). `ab_test_frozen_runs.py` has no direct reference but inherits one through `walkforward_backtest.py:490` |
>
> ### Smaller factual corrections
>
> - `:238-242`, `:835-843`, `:1803` — "AK-47 | Redline (FT)… 96 Steam sales in 24 hours" as "the
>   deepest explanation for your null streak". This doc's own **C5** measures the archive
>   median at **69 sales/item-day**, so the flagship is 1.4× the median, not an outlier. The
>   ceiling argument is really about the **$50–500 (20/day)** and **$500+ (4/day)** tiers —
>   which is where the product lives, so the conclusion is *sharper*, but the headline number
>   is doing no work.
> - `:1729` — drop odds "each adjacent tier exactly 5:1". The last ratio is **2.46:1**
>   (Covert 0.64% / Rare Special 0.26%), from this table's own numbers.
> - **Three enumerations read as complete and are not.** `:337-352` §3 says "Four defects…
>   three of them inflating significance" then lists four of which only **#1 and #2** inflate
>   (#3 is irreproducibility, #4 is item-draw noise). `:1284-1298` §18 numbers L1–L10, L12,
>   L13 — **L11 is missing**. `:1373-1383` §20 numbers 1–8 then 10 — **9 is missing**.
> - `:761-768`, `:827-830` §11 — the premium/spread tables omit the **$500–1000** tier and
>   sum to **23,586** against a stated ALL of **23,904**; and the liquidity paragraph's
>   "10,152 (41%) are sub-$1" does not reconcile with its own stated universe of 46,631
>   (= **21.8%**). `:1358` repeats "41% of the archive is sub-$1" against §1's **36.1%**.
> - `:1321-1339` — the regime stress suite (S1–S6) and §20 metric 4 (`:1378`, "per regime
>   window") are graded against PASS criteria that require a `regime_window` definition.
>   `REGIME_WINDOWS` / `regime_window` / `regime_stress` return **zero hits** across every
>   `.py` in the repo. The suite is prose, not a gate — and `:1378`'s coverage criterion
>   cannot be supplied by the walkforward gate at all, which gives every arm
>   `PLACEHOLDER_BAND_PCT = 10.0`
>   (`docs/changelog/2026-08-08-r11-r12-declined-and-r18-r19-recosted.md:51-62`).
>
> ### And the standard this document is held to
>
> **This doc does not hold the "nothing here is estimated" standard its successor claims for
> it.** It marks `[MEASURED HERE]` on a *required* MDE (0.3pp), on a tautology its own
> corrections section retracts (1.1607), and on a different quantity than the one being
> reasoned about (the 0–1.8% staleness split). Read the confidence tags as claims to check,
> not as provenance.
>
> **R11–R19 status:** see the block under §10's Tier 3 table.

---

## Corrections to this document (2026-08-07, second pass)

Follow-up work overturned four claims made earlier in this file. They are corrected in
place below, and listed here because three of them touch live code.

**C1. The 1.1607 Steam fee constant is synthetic, and the original "measurement" was
circular.** §11 proposed regressing the ratio on price as a diagnostic. It was run against
`backend/runtime/steam_listing_history.db` (528,573 rows, 262 items) joined to the archive —
63,767 matched pairs. The measured ratio is **flat at 1.1606–1.1607 across four orders of
magnitude**, IQR 0.0002, where Steam's real cent-ceiling schedule must swing from ~1.67 at
$0.03 to ~1.15 at $50. At single-cent granularity ($1.00→$1.12) theory sawtooths 1.150→1.168
and the measurement does not move in the fourth decimal. **91.61% of pairs are the same
numbers after dividing.** The two series differ by a flat multiplier baked in upstream, so
regressing one on the other recovers 1.1607 tautologically. *Consequence:* any listing-page
backfill row below ~$0.50 carries a real basis error (~5% too high at $0.10–0.25), and the
bottom three deciles show ratio 0.44 — buyer below net, which no fee can produce.

**C2. BUFF's `highest_order` is already ingested — and it is voting into the consensus
price as if it were an ask.** §10 Tier 1 #3 said "ingest it." Wrong: `collectors/pipeline.py`
and `collectors/csgotrader_aggregator.py` already read it and write it as
`aggregator_buff163_buy`, and `forecaster.py` feeds every non-`historical_fallback` source
into `_apply_multi_source_voting` **with no source filter**. The bid sits at 0.550× Steam
while asks sit at 0.700–0.802×. Worse, `vote()` rejects >2σ outliers only when ≥3 sources
are present, so **whether the bid is rejected is data-dependent and flickers day to day** —
fabricating returns of roughly the bid–ask wedge. The corrected action is *exclude it from
voting, then promote it to its own column.* **[UNVERIFIED — mechanism read from code,
magnitude not measured. This is a one-query check and it should precede everything else in
this document.]**

**C3. The 2025-10-22 trade-up was in Valve's machine-readable feed the same day.** §12 filed
it under "unannounced / zero warning." Correct about *advance* warning, wrong about
observability: `ISteamNews` for appid 730 carries it at **2025-10-22 23:00:20 UTC** with the
mechanic verbatim ("*Extended functionality of the 'Trade Up Contract' to allow exchanging 5
items of Covert quality…*"). Same for Trade Protection (2025-07-16 00:00:58 UTC) and the
souvenir trade-up (2026-05-22 00:40:34 UTC). Recall on the dated supply-shock record is
**4 of 6, and 3 of the 4 largest** — the two misses were enforcement/config actions, not code
changes. A *contemporaneous* supply-shock detector is buildable; `ingest_steam_news.py`
already pages to 2012 and already caches the full announcement text, unused.

**C4. "Higher volume associates with lower prices" is not supported — the within-item sign is
positive.** Tested on 4.46M rows, 2023–2025, trailing-30 volume z-score, strictly lagged.
Pooled corr(vol z, fwd7) = **+0.019**, fwd30 = **+0.034**; item-fixed-effects version
identical (+0.019/+0.034), so not a composition artifact. The $1–10 tier reads **+0.080** at
7d. Only the log *level* at 30d and the cross-sectional level relation are negative
(−0.015, −0.038). Everything is economically trivial (r² < 0.15%). *Separately:* these are
**10–40× larger** than the `|r| < 0.002` figure `docs/references/data-sources.md` rests on.
The audit's conclusion (volume is not usable) survives; its stated reasoning does not, and
that number should not be quoted again.

**C5. Sharpened, not overturned — the counting-noise argument is strongest exactly where the
product lives.** §0 cites one live probe (AK-47 Redline FT, 96 Steam sales/24h). The archive
gives the distribution for the 5,542-item cohort in 2025: median **69** sales/item-day
overall, but **20** at $50–500 and **4** at $500+, with 60% of $500+ item-days at 1–5 sales.
So the thin-series problem is *not* uniform — it is concentrated in the expensive tier, which
is the served cohort. Also important: **`volume = 0` never appears** (min is 1). A day with
no sale produces an **absent row**, not a zero, so the archive cannot distinguish "no trades"
from "no observation." Row presence is the answerable version: ~89% of the cohort prints on
a given day.

---

## 0. The one-paragraph answer

The question "what features predict CS2 item prices" has already been answered here, and
the answer is *none of the item-level ones*. Demeaning returns by the same-day market
factor drops directional accuracy **below a constant call at every horizon** (36.7/32.7/
34.6/39.0 vs a majority-class baseline of 38.8/42.7/46.4/51.7) **[MEASURED HERE,
`2026-08-06-market-relative-labels-refuted.md`]**. Everything the model was doing was
riding a common factor. This is not a defect and not unusual: Gu, Kelly & Xiu (2020)
report a best out-of-sample monthly R² of **0.4%** on 30,000 US equities with 60 years of
history **[PUBLISHED]**. With ~5,500 items and one usable year of multi-source data, the
item-level cross-section is not where the signal is.

Fresh measurement taken for this review supplies the deepest explanation yet: **AK-47 |
Redline (Field-Tested), one of the most traded skins in the game, records 96 Steam sales in
24 hours** (§11). Per-item daily price changes here are built on tens of transactions. That
is counting noise, and no feature set recovers signal from a series that thin — which is
why a dozen well-run experiments all returned null.

The three moves that remain open are (a) fix the metric, because the current one measures
the market's base rate rather than the model, (b) move the target from item-direction to
within-date *ranking*, which cancels the market factor by construction instead of by
subtraction, and (c) model the **date-level market factor itself** — the only place
variance has been demonstrated to live, and the only way to aggregate past the
counting-noise floor. The project has never attempted (c).

---

## 1. What this system already is

From `docs/references/data-inventory.md` and the memory ledger, all **[MEASURED HERE]**:

| | |
|---|---|
| Archive | 20,756,038 rows, 41,725 items, 2013-08-14 → 2026-08-04, 4,735 of 4,739 days present |
| Model | LightGBM, one q50 regressor per horizon (3/7/14/30d) + a 3-class direction classifier; band by split conformal |
| Trained on | ~~**99 items / 116,111 rows** — 1.8% of the 5.8M-row pool, a rarity-stratified draw~~ **Stale.** Subsampling stopped 2026-08-08; production is **918 items / 986,065 rows** (floor $1, 1.2M budget). And 116,111 / 5.8M is 2.0%. Banner item 4 |
| Served | 8,691 items, filtered to ≥$1 at serve time |
| Accuracy (≥$1, CV) | ~~49.9 / 49.0 / 51.1 / 53.4% at 3/7/14/30d~~ **No longer producible** — the CV diagnostic classifier is off by default (`dfafdfb`) and `meta.json` carries `mean_classifier_acc*: null`. Replacement headline: `mean_rank_ic` **0.1307 vs naive 0.1656, edge −0.0349** at 7d |
| Accuracy (≥$1, prod) | 48.4 / 49.4 / 50.8 / 46.7% |
| Accuracy (all tiers) | 67.3 / 67.2 / 67.8 / 68.6% — **this is the penny-item score, not the product's** |

### The three structural facts that constrain everything

**1. The depth cliff.** The archive is two disjoint populations: 5,542 items with ~12
years (median 1,507 days) and ~36,000 items first seen in 2026 and capped at ~136 days.
There is no middle tier. Median across all items is 128 days.

**2. The multi-source era is 25 days deep, not 4.5 months.** Eight of eleven live sources
start 2026-07-11. Cross-venue features (spreads, lead-lag, arbitrage gaps) have almost no
history to train on. Bid/ask exists only as `aggregator_buff163_buy` vs
`aggregator_buff163` — 30,377 paired items, **25 days**, median ask/bid 1.283.

**3. Volume is dead.** `volume` is non-zero for 100% of rows 2013–2025 and **identically
zero for every row since 2026-04-16**. The column is still written and still read by
`backtest/price_resolution.py:249`. Any volume feature is training on a field that is
present in history and absent at serve time — a silent train/serve mismatch.

### The cohort inversion — the most consequential product fact

| Cohort | ≥$1 | <$1 | % ≥$1 |
|---|---:|---:|---:|
| All items with recent data | 26,468 | 14,955 | **63.9%** |
| Items actually forecast | 1,423 | 7,268 | **16.4%** |

The archive is two-thirds dollar-plus. The served cohort is 84% sub-dollar. **The gate
excludes most of the items that have tradeable signal and includes mostly ones that
don't.** This is a larger lever than any feature, and it is a data-plumbing problem
(`is_backfilled`), not a modelling one.

---

## 2. The refutation ledger — do not re-propose these

Every row **[MEASURED HERE]**, with the caveat in §3 about interval width.

| Hypothesis | Result |
|---|---|
| Market-relative (cross-sectionally demeaned) labels | **Refuted.** Below a constant call at all four horizons. The decisive experiment. |
| CSFloat price-history basis feature | Refuted — null and harmful at 30d. Also: budget is 500 req/**day**, and `avg_price` is float-composition noise |
| ByMykel item metadata bundle | Refuted — the model's own permutation test shows the features go unused |
| Volume features | Shelved; and the underlying column is dead since 2026-04-16 |
| Six "price primitives" | Refuted at all four horizons at 200 items |
| Reddit sentiment | Refuted, then deleted. Runner IPs get 403s anyway |
| Item age | Null; and it carries a calendar term that the unpurged harness inflated |
| Weapon identity / per-weapon effects | Dead axis — the apparent per-weapon effect *is* the market factor |
| Served-cohort weighting | Refuted |
| Serving-transform down-skew | Largely refuted — the "+13.4pp skew" was a market-period artifact |
| Rarity metadata repair | Cannot move accuracy — perturbs the item draw by exactly the seed-noise amount |
| Event calendar (item-level) | Functions as a **clock**; null once the trend is stripped |
| Breadth vs depth of training set | Breadth wins; depth-only sources actively shrink the trainable universe |
| ≥$1 training universe | ~~**The one positive.** Paired **+3.50pp [+1.56, +5.98] at 30d**, null at 3/7/14d~~ **WITHDRAWN 2026-08-08.** Re-derived as **+1.642pp [−0.809, +4.505], null**, against an item-draw floor of ±3–4pp; the look-ahead it was attributed to is worth −0.004pp. The floor shipped on determinism, not accuracy. `2026-08-08-per-fold-price-filter-rederived.md` |

The pattern is unmistakable: **a dozen experiments were variations on a signal that is not
present.** The literature says the same thing prospectively — Grinsztajn et al. (2022)
**[PUBLISHED]** show trees are robust to uninformative features, which is precisely why
adding refuted feature groups produced null rather than negative results and therefore
never generated a clear stop signal.

---

## 3. The measurement problem — read this before running any experiment

This is the part most projects never reach, and it is where the real risk sits.

**The harness cannot resolve the effects the ship gates ask for.** Paired per-fold
directional-accuracy sd, clustered correctly on `fold_id`, gives an operative MDE of
**2.21–3.69pp** **[MEASURED HERE]**. Ship gates in this repo are written around 0.5–1.5pp.
Runs at that target return coin-flip fold-win counts and get misread as "unproven" when
they were never resolvable. Fold count is set by `step` and the archive date range, **not**
by `--max-items` — raising items sharpens each fold but adds none.

**Four defects have been found in the A/B harnesses, three of them inflating significance:**

1. **`paired_mde` clustered on `forecast_date`, not `fold_id`, until 2026-08-07.** Dates
   inside one fold share a fitted model, so intervals were too **narrow**. Caught by a
   seed-only placebo returning CI [−0.3099, −0.0140] — excluding zero on pure noise.
   *Every "CI excludes zero" this module produced before 2026-08-07 is suspect, including
   the CSFloat, ByMykel and training-breadth headlines. None have been re-derived.*
2. **No purge/embargo at the train boundary** in `ab_test_csfloat_basis.py`,
   `ab_test_item_metadata.py`, `ab_test_training_breadth.py`, `walkforward_backtest.py`.
   Production's `_compute_cv_splits(purge_days=horizon)` does purge; the walkforward family
   diverged. Measured cost: the event-calendar arm at h=30 went **+12.1pp unpurged →
   +6.1pp purged**. Half the effect was overlap.
3. **The harness is not reproducible run-to-run** and nobody knows why. Identical commands
   moved `mean_diff_pp` from −0.1581 to −0.0026, with `n_paired` and `n_dates` differing.
4. **The item draw alone moves `acc_ge1` by sd 1.5–3.1pp.** A single retrain is not a
   measurement.

**The literature names all of this.** López de Prado's purging and embargo **[PUBLISHED,
*Advances in Financial ML* ch. 7]** is defect 2 exactly — the embargo must be ≥ the label
horizon, so ≥30 days at h=30. Bailey & López de Prado's Deflated Sharpe and PBO
**[PUBLISHED, *JPM* 40(5), 2014]** address the multiplicity you have now accumulated: a
dozen A/Bs against the same panel means the single-comparison CI is the wrong instrument.
Harvey, Liu & Zhu (2016) **[PUBLISHED, *RFS* 29(1)]** put the hurdle for a new factor at
**t > 3.0**, not 2.0. Nikolopoulos (2026, arXiv:2604.15531) formalises the placebo practice
you arrived at independently: run the *whole workflow* against synthetic
zero-predictability data, not just permuted labels.

**Practical consequence:** the ≥$1 training universe result (+3.50pp at 30d) is the only
surviving positive, it was derived post-fix with fold clustering, and it should still be
deflated for multiplicity before being called established.

---

## 4. Fix the metric first — this is the highest-value single change

Your headline metric is directional accuracy. The realised down-rate on the ≥$1 cohort
**swings from 32.7% to 76.9% between forecast dates**, and on every stored date a constant
predictor matching that date's market beat the model **[MEASURED HERE]**.

This is not a curiosity. It is a known, named pathology with a published test.

**Pesaran & Timmermann (1992), *JBES* 10(4), 461–465** **[PUBLISHED]** — a nonparametric
test of directional forecast value whose **null is independence between predicted and
realised sign, not 50%**. The benchmark is the hit rate implied by the marginal base rates.
If the market is 77% down and you call down 90% of the time, raw DA looks strong and the
information content is zero. PT nets this out.

Your `da-is-dominated-by-market-date` finding *is* the PT null. Reporting PT would have
surfaced it years earlier and would have prevented a dozen experiments run against a metric
that was measuring the market, not the model.

**Two extensions you specifically need:**
- **Blaskowitz & Herwartz (2014), *IJF* 30(1)** — the serial-correlation-robust variant.
  Given stale carry-forward prices, this is the one to use.
- **Getmansky, Lo & Makarov (2004), *JFE* 74(3), 529–609** **[PUBLISHED]** — illiquidity
  makes reported returns *smoother* than true returns via an MA(k) mechanism. Your
  calendar-gap lag fills and frozen sub-$1 prices are literally this. The per-item
  smoothing coefficient θ is (a) a diagnostic for which items' DA is an artifact and (b) a
  candidate feature. ~~It also explains the tier-0 vs ≥$1 split mechanically: bit-identical
  `actual_price == base_price` runs 37–42% at tier 0 and 0–1.8% at every tier ≥$1
  **[MEASURED HERE]**.~~ **Corrected 2026-08-08:** 0–1.8% is measured on resolved backtest
  anchors, which are 3-day smoothed medians. On the raw voted series the label path sees,
  ≥$1 staleness is **12–27%** (20.25pp overall), so the split does not exist on the series
  being modelled — `2026-08-08-frozen-price-runs-dropped-from-labels.md:18-21`, banner item 5.

**Do this:** replace raw DA with the serial-correlation-robust PT statistic as the headline,
and report it *per forecast date* with a t-stat over dates. Cost: a scoring-module change,
no retrain. It reframes every number the project has produced.

---

## 5. Change the target — ranking, not direction

You tested market-relative labels by **subtracting** the market factor and found nothing
left. There is a second way to remove the same factor that behaves very differently under
noise: **rank items within each date.**

**Poh, Lim, Zohren & Roberts, "Building Cross-Sectional Systematic Strategies by Learning
to Rank" (arXiv:2012.07149, *Journal of Financial Data Science*)** **[PUBLISHED]** —
regress-then-rank is sub-optimal because a pointwise loss ignores the ordering you actually
act on. Learning-to-rank objectives beat regression on rank-IC **especially at low
signal-to-noise and with heavy-tailed noise**, which is your regime exactly.

Why this is not the same experiment as the one you already refuted:
- Demeaning changes the *label* and leaves a pointwise squared/log loss fighting a noisy
  residual. Ranking changes the *loss*, so the model is never asked to predict a magnitude
  it cannot know.
- The date effect is quotiented out structurally. There is no residual to be swamped.
- The metric becomes rank-IC / Spearman per date, which has a clean null (zero) and is not
  vulnerable to the base-rate pathology in §4.

LightGBM already ships `lambdarank`. This is an objective swap plus a metric, not new data.
**I rate this the highest-expected-value untried experiment**, with the honest caveat that
your own evidence says there may be nothing to rank — but ranking is the version of that
test that is not confounded by the loss function.

**Second untried cross-sectional effect: reversal.** Borri, Liu & Tsyvinski, "The Economics
of NFTs" (SSRN 4052045, working paper — **not** peer-reviewed, do not cite as published)
find **size and return-reversal** effects in the near-universe of NFT transactions. Reversal,
not momentum, is the documented cross-sectional effect in digital collectibles. Your shelved
"price primitives" were momentum-flavoured. Rank on trailing k-day return and predict the
*opposite* — one feature, cheap to test.

---

## 6. Model the market factor itself — the genuinely unexplored direction

Everything above is still item-level. But your own evidence says the variance is at the
**date** level, and you have never modelled it.

The logic is straightforward:
- Item-level idiosyncratic signal: **measured, absent** **[MEASURED HERE]**.
- Date-level variance: **measured, enormous** — the market down-rate swings 32.7% → 76.9%.
- ~~Date-level MDE: **~0.3pp, i.e. measurable** **[MEASURED HERE,
  `date-level-exogenous-ingest-built`]** — an order of magnitude better than the 2.21–3.69pp
  item-level floor, because a date-level series has no item-draw variance and no
  cross-sectional noise.~~ **Wrong, and it is the load-bearing error in this section.** 0.3pp
  is a *required* effect size that `accuracy-opportunities.md:57-60` declares "**not
  reachable**", and the changelog cited here retracts it: the operative date-level floor is
  the same fold-clustered **2.21–3.69pp**, and "nothing in these two tables can be resolved
  below roughly 2pp" (`2026-08-06-date-level-exogenous-ingest.md:31-38`). See banner item 1.
  **The argument for §6 is the variance, not the MDE** — the date leg faces the same floor as
  everything else, so it needs a large effect, not a sensitive instrument.

So the target becomes: **forecast the aggregate market return** (or the down-rate) for date
*t+h*, then apply it uniformly. This is a single time series with **4,735 days of history**
— a vastly more favourable data shape than 99 items × 33 technicals over one year.

This also finally makes the exogenous variables meaningful. Reichenbach (2025), *Finance
Research Letters* 83, 107670 **[PUBLISHED]** — >3,000 Counter-Strike items, 2015–2025 —
finds diversified skin portfolios returned up to 66.9% p.a. and attributes the drivers to
**player-base growth and supply contraction**. Both are date-level, not item-level. Your
event calendar was null at item level because it *is* a clock there; against a date-level
target it is measuring the thing it was built to measure.

Hogan-Hennessy, Xenopoulos & Silva (arXiv:2210.07970, 2022) **[PUBLISHED preprint]** give
the causal template from Old School RuneScape: a developer-side **item sink inflated luxury
prices without reducing volume**, while a transaction tax did not change volume. Developer
supply/sink shocks are the dominant exogenous driver in a game economy — the analogue of a
Valve drop-pool or trade-lock change.

**Honest caveats.** (a) An aggregate index is far more autocorrelated than item returns, so
the purge/embargo discipline matters *more*, not less. (b) A market-timing model that is
right 55% of the time is a different, and much weaker, product than a per-item pick. (c) You
need a proper index first — see §7.

---

## 7. Build a real market index (hedonic), because you don't have one

You have no principled market factor. The demeaning experiment used a same-day median,
which is a composition-weighted statistic that moves when the *set* of observed items moves
— and your set moved from 5,542 to 41,725 items during 2026.

The collectibles literature solved this decades ago:
- **Bocart & Hafner et al. (2018), *Econometrics* 6(3), 32** — combining hedonic and
  repeat-sales information for fine art. **[PUBLISHED]**
- **Fogarty (2011), *Australian Economic Papers* 50(4), 147–156** — hedonic vs repeat-sales
  vs hybrid for wine. **[PUBLISHED]**

Both conclude the **hybrid index is most efficient and least volatile**, and that pure
repeat-sales gives significantly *higher* return estimates than hedonic because of selection
toward traded items. Skins are a textbook hedonic panel: wear, float, StatTrak, souvenir,
rarity, collection are exactly the attribute set the method assumes. Regress log price on
attributes plus **time dummies**; the time dummies *are* the index.

This gives you three things at once: a composition-controlled market factor for §6, a
correct demeaning basis if you ever revisit §5, and a defensible user-facing "CS2 market
index" — which is a product feature, not just an internal one. **No such index has been
published for CS2** (see §9), so this is genuinely novel work.

---

## 8. Transaction costs — why 51% accuracy is not a trading system

~~**[MEASURED HERE]** Your own archive establishes the Steam fee empirically: across 30,875
overlapping rows the Steam listing-page buyer price divides by a constant **1.1607** to
reach the net price (p10 1.1565, p90 1.1656, within-item CV 0.0021).~~ **See C1: this is
circular — 91.61% of the pairs are the same numbers after dividing, and 1.1607 is a baked-in
upstream multiplier, not a fee.** (C1 also reports a different *n* for the same measurement:
63,767 matched pairs, not 30,875.) The published rule is buyer/net = **1.15** exactly. That is
a ~13.0% haircut on the buyer price on the sell side alone.

Round-trip on Steam requires **+16.1%** to break even before spread — **read +15.0%, from the
published rule; the +16.1% here and everywhere below is derived from the synthetic constant** —
CSFloat and DMarket
are **+2.0%**, Skinport **+8.7%** (§11). Add the bid–ask spread, freshly measured across
22,449 items: **20.9% median**, ranging **35.5% sub-$1 to 5.2% at $1000+**. A Steam round
trip on a mid-tier item therefore needs something like a 25–40% move.

Against that, a model at **49–53% directional accuracy on a 3-class problem** has no
economic value as a trading signal at any horizon you currently forecast. The gap is not
close, and no realistic accuracy improvement closes it.

**Two further constraints make this stricter than it first looks.** The Steam 7-day market
lock plus the July-2025 trade protection mean **no horizon under ~8 days is executable at
all** — h=3 and h=7 describe moves nobody could trade. And a 3% predicted move is *inside
the spread* for five of the six price tiers, so most of your predictions are not
economically distinguishable from no prediction.

**This is not a reason to stop.** It is a reason to be precise about what the product is:
an **information/analytics dashboard**, where being modestly better than naive at
describing where the market is going has genuine user value, and where calibration and
honest uncertainty bands matter more than hit rate. That is what this repo has actually
built. The failure mode to avoid is drifting into implying tradeability.

Third-party venue fees are materially lower than Steam's and are the only place a
friction-aware backtest could be non-absurd — see §11 for the verified fee table.

---

## 9. Where the literature is thin — and what that means

From the completed literature sweep, these gaps are real findings:

1. **No published work does cross-sectional ML on skins.** Every CS2/CS:GO ML paper fits
   one model per item on 28–12,000 independent series. Your global-model design is *ahead*
   of the published work — and Montero-Manso & Hyndman (2021), *IJF* 37(4), 1632–1653
   **[PUBLISHED]** is the theoretical justification: a single global model matches or beats
   per-series local models with no similarity assumption.
2. **No skin paper uses purged/embargoed CV or any multiple-testing correction.** Zero.
3. **No skin paper reports DA against a constant-sign benchmark.** Your
   `da-is-dominated-by-market-date` result appears genuinely novel for this asset class.
4. **No microstructure work on the Steam market at all** — no bid-ask estimates, no
   staleness analysis, no volume-vs-listing-count disambiguation. Your archive is uniquely
   positioned to fill this.
5. **No hedonic or repeat-sales price index for skins**, despite float/wear/pattern being a
   perfect hedonic attribute set. See §7.
6. **No skin study handles survivorship or item-onboarding bias.**

### The two published CS2 ML papers are both cautionary

- **Guede-Fernández et al. (2025), *Frontiers in AI* 8, 1702924** **[PUBLISHED, open
  access]** — 12,000 skins, per-skin LSTM and NHITS, Backtrader, 10% transaction cost.
  Reports **Sharpe 10–12**. A Sharpe of 10–12 on a daily strategy is a near-certain leakage
  or selection artifact: single year, one venue, no purging, threshold selection over 12,000
  candidates with no multiplicity correction. **Read it as a worked failure**, and cite it
  whenever someone brings you a high-return skin-ML result.
- **Pettersson (2025), master's thesis, Doria 10024/193781** — 640,145 daily observations,
  28 skins, 2013–2025. RF/XGB reach **out-of-sample R² ≈ 0.50**; LSTM much weaker. An R² of
  0.50 on daily *returns* would be extraordinary; on daily *levels*, or with MA features on
  a near-random-walk, it is trivial. Useful for exactly two things: the tree-beats-LSTM
  result independently replicates your LightGBM choice, and it is the concrete warning that
  **R² on a price level is not evidence**.

### Model choice is settled — stop revisiting it

- **M5 competition (Makridakis et al., *IJF* 38(4), 2022)** **[PUBLISHED]**: first
  M-competition swept by pure ML; **LightGBM used by all of the top 50** in both the
  accuracy and uncertainty tracks. Caveat the authors themselves make: at the *individual
  product* level, trivial benchmarks stay competitive — your situation.
- **Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022, arXiv:2207.08815)** **[PUBLISHED]**:
  trees remain SOTA on tabular data at ~10K-sample scale; NNs are notably sensitive to
  uninformative features.

Combined with your data scale, this closes the "should we try a transformer / TFT / N-BEATS"
question. **No.** The answer is not a better model class; there is no signal for a better
model class to find.

### One counterweight worth a single experiment

**Kelly, Malamud & Zhou (2024), *Journal of Finance* 79(1), 459–503** **[PUBLISHED]** — "The
Virtue of Complexity": ridgeless heavily-parameterised models beat simple ones out-of-sample,
and simple models systematically *understate* predictability. Contested, and the ridge
penalty is doing the work — but it directly contradicts your model-shrinking decisions and is
worth one A/B given you already have the budget knobs.

**Also worth adopting:** Jensen, Kelly & Pedersen (2023), *JF* 78(5), 2465–2518
**[PUBLISHED]**, whose Bayesian hierarchical shrinkage estimator is the right statistical
tool for your many-small-A/Bs problem. Code: `github.com/bkelly-lab/ReplicationCrisis`.

### One improvement to a user-visible output

You calibrate the served band with **split** conformal, which assumes exchangeability. Under
regime shifts that assumption breaks and coverage drifts. **Gibbs & Candès (2021, NeurIPS),
Adaptive Conformal Inference** adjusts α online from realised coverage errors; **Xu & Xie
(2021, ICML), EnbPI** refreshes the residual pool. ACI is a small change and it improves
something a user actually sees, unlike everything else here. **[PUBLISHED]**

---

## 10. Ranked recommendations

Ordered by expected value per unit of effort, given what is already known.

**Tier 1 — do these first. All cheap, and each one changes how existing numbers read.**

| # | Action | Why | Effort |
|---|---|---|---|
| 1 | **Replace DA with serial-correlation-robust Pesaran–Timmermann**, per date with a t-stat over dates | Current headline measures the market's base rate, not the model | Small — scoring only |
| 2 | **Score by price tier, and condition on `\|predicted move\| > round-trip cost`** | Spread runs **35% sub-$1 to 5% at $1000+**. A pooled DA number is uninterpretable; a 3% call is inside the spread for 5 of 6 tiers | Small |
| 3 | **STOP `aggregator_buff163_buy` voting into the consensus price**, then promote it to its own column *(revised — see C2; it is already ingested)* | The BUFF **bid** is being voted as if it were an ask, and outlier rejection makes its inclusion flicker day to day — fabricating returns of roughly the bid-ask wedge on ~33k items since 2026-07-11. **Every label and every A/B since then sits downstream of this median** | Tiny to check, small to fix |
| 4 | **Add purge/embargo ≥ horizon + 13 days** to `walkforward_backtest.py` (`--purge` is **default OFF**, so the *published* number is unpurged) **and the ten `ab_test_*` harnesses that have neither purge nor fold clustering** | Published fix for a measured +6pp inflation at h=30. Larger blast radius than previously stated — see §18 L1 and §19 | Small |
| 5 | ~~Regress the 1.1607 ratio on price~~ **DONE — it is synthetic (C1). Fix the listing-page backfill to use Steam's real cent-ceiling schedule** | The measured ratio is flat to 4dp where theory swings 1.150→1.168; the two series are the same numbers. Rows below ~$0.50 carry a real basis error | Small |
| 5b | **Add `ingested_at` to the price archive schema** | There is **no arrival timestamp anywhere**, so every revision/backfill leakage vector is unauditable retroactively. One column, cheap now, impossible to add backwards | Tiny |
| 6 | **Grep for `api.dmarket.com/exchange/v1`** | Returns **410 Gone**. Dead integration if present | Tiny |
| 7 | **Ship `TRAIN_MIN_MEDIAN_PRICE` with `TRAIN_FEATURE_ROWS ≥ 1.0M`** — and consider a floor **above** $1 | The one surviving positive (+3.50pp at 30d); removes item-draw variance entirely. The spread data argues the honest floor is higher than $1 | Small; +7 min Monday-only |

**Tier 2 — the substantive research moves.**

| # | Action | Why | Effort |
|---|---|---|---|
| 8 | **Build a hedonic market index** (log price ~ attributes + time dummies) | You have no composition-controlled market factor; your demeaning used a median over a set that grew 5,542 → 41,725 items mid-sample. No published CS2 index exists — this is novel | Medium |
| 9 | **Forecast the date-level market factor** as its own model | The only place variance is demonstrated to live. 4,735 days of history. ⚠️ ~~date-level MDE **~0.3pp** vs 2.21–3.69pp item-level~~ — **withdrawn, see banner item 1; the date-level floor is the same 2.21–3.69pp** | Medium |
| 10 | **Try `lambdarank` within-date**, scored by rank-IC | Cancels the market factor structurally rather than by subtraction; LTR is specifically better at low SNR | Small — objective swap |
| 11 | **Recover historical volume** from the kieranpoc Kaggle dump | Your `volume` column has been zero since 2026-04-16. This is Steam price+volume back to 2013 for 22,492 items | Medium |
| 12 | **Backfill retroactive supply depth** from `atalantus/buff-price-history-archive` | BUFF listing counts 2023-01-25 → 2024-01-19. Kills the ~30-day accumulation wait. Different use from the price-source evaluation that declined it | Medium |
| 13 | **Fix the cohort inversion** | 84% of what you serve is sub-dollar; 64% of the archive is dollar-plus — and the sub-dollar tier has a 35% spread | Medium — `is_backfilled` plumbing |

**Tier 3 — speculative but cheap.**

| # | Action | Why |
|---|---|---|
| 14 | **Mechanical supply-position features**: trade-up fuel vs output, drop-pool status, float-range cap | The Oct-2025 cross-section dispersed on *these*, not on cosmetic metadata. ByMykel was refuted; this is a different feature class |
| 15 | **Per-item MA smoothing coefficient** (Getmansky–Lo–Makarov) | Diagnostic for which items' DA is a staleness artifact; candidate feature |
| 16 | **Cross-sectional reversal** — ~~Tier 3, speculative but cheap~~ **MEASURED 2026-08-09 and this ranking was wrong: it is the project's strongest measured predictor** | The documented effect in digital collectibles; you tested momentum, not reversal. **Rank IC +0.1023 at 3d (+0.1941 full archive), survives composition control, beats the model at all four horizons** |
| 17 | ~~**Test "volume ↑ ⇒ price ↓"**~~ **DONE and REFUTED by this document's own C4** — pooled +0.019 / +0.034, item-FE identical, r² < 0.15%, and the sign is *positive*. This row was never updated | ~~The panel study's counterintuitive finding, testable on your intact pre-2026 volume series~~ |
| 18 | **Split conformal → adaptive conformal (ACI)** | Fixes band coverage under regime shift; improves a user-visible output |
| 19 | Deflate past A/Bs for multiplicity (DSR/PBO); adopt a t>3 hurdle | A dozen A/Bs against one panel means single-comparison CIs are the wrong instrument |

> **Status of R11–R19 as of 2026-08-09.** Tracked in `docs/research/2026-08-07-next-steps.md`
> and re-ranked in `docs/research/2026-08-09-next-steps.md`. Rows 11–19 above are the
> unchanged proposals; this is what happened to them.
>
> | # | Status | Correction to the row above |
> |---|---|---|
> | **R11** kieranpoc volume | **REFUTED as written** | The dump is frozen at snapshot **2024-05-04** (`dateModified` 2024-06-15), so it supplies nothing for 2024-06 → 2026-08, the only window that matters. Licence at `:1014` is not "unconfirmed": it is **CC BY-NC-SA 4.0**, non-commercial. The *premise* is also superseded — no aggregator feed ever carried a volume field; `steam_volume.json` ceased to exist, 2026-04-16 is the day after the last one-shot backfill ran out, and the column is now NULL rather than 0. `docs/changelog/2026-08-08-r11-r12-declined-and-r18-r19-recosted.md` |
> | **R12** atalantus listing counts | **REFUTED** | 2023-01-25 → 2024-01-19 is **359 usable days** ⇒ **~2 folds** at `CV_STEP_DAYS = 150` / `CV_MIN_TRAIN_DAYS = 200`; a 2.5-year gap to live depth; different venue and different quantity. Source-record fix: the raw dump is **113 MB via Git LFS**, not 24 MB |
> | **R13** cohort inversion | **OPEN, now unblocked** | Blocker 5d was a false positive (banner item 3). Nothing built |
> | **R14** supply-position features | **OPEN** | Correctly ranked last |
> | **R15** `theta_ma1` / staleness | **PARTLY DONE** | The label-side use shipped 2026-08-08 (`staleness.py::stale_run_days`). As a *feature* it is still open — but its stated evidence is the 0–1.8% staleness split, which does not hold (banner item 5) |
> | **R16** cross-sectional reversal | **DONE — and it inverted the finding** | Filed here as Tier 3 "speculative but cheap". It is now the project's **strongest measured predictor**: rank IC **+0.1023 at 3d** on 2026, **+0.1941 over the full archive**, it survives composition control, and it **beats the model at all four horizons**. `docs/research/2026-08-09-composition-stability.md`, `f833882` |
> | **R17** "volume ↑ ⇒ price ↓" | **REFUTED by this document's own C4** | Tier 3 #17 proposed it as untried while `:204-212` reports it done — pooled **+0.019 / +0.034**, item-FE identical, r² < 0.15%, and the *sign is positive*. The row went un-updated for two days; struck in place at `:653` |
> | **R18** ACI | **OPEN, mis-costed here** | `:614` calls ACI "a small change". Corrected 2026-08-08 on three counts: no `regime_window` exists in code; the walkforward gate cannot supply coverage **by design** (`PLACEHOLDER_BAND_PCT = 10.0`); and stored outcomes span **6 forecast dates in two clusters**, which cannot be partitioned into regime windows. Carried as **C5** of `2026-08-09-next-steps.md` |
> | **R19** deflate for multiplicity | **OPEN, urgency gone** | It was framed as the gate on the +3.50pp positive. That positive was withdrawn, so the *ordering* claim is void; the argument stands on its own. Carried as **C7** of `2026-08-09-next-steps.md` |

**Do not:** propose another item-level cosmetic feature group, another model class, or a
deep-learning architecture. §2, §9 and §11 all converge on the same conclusion, and §11
supplies the deepest version of it — a flagship item trades **96 times a day**, so per-item
daily returns are counting noise before any feature touches them.

**A scope note on horizons.** The Steam 7-day market lock plus the July-2025 7-day trade
protection mean **no horizon under ~8 days is executable**. Your h=3 and h=7 forecasts
describe moves nobody could have traded. That is fine for an information product — but h=14
and h=30 are the only horizons where an accuracy claim could ever carry an economic
interpretation, and they should be the ones you optimise.

### What to collect (all free)

Already verified working, no auth **[MEASURED HERE]**: Skinport `/v1/items` (needs
`Accept-Encoding: br`), Waxpeer `/v1/prices`, Bitskins `/market/insell/730`, CSFloat
`/api/v1/history/<name>/graph` (daily completed-sale count + avg price back to 2020-04,
500 req/**day**), Steam `priceoverview`, Steam `itemordershistogram` (full bid/ask book,
needs `item_nameid`), Steam market **listing pages** (full daily price + real `purchases`
volume back to 2013, sessionless).

The Steam listing-page route is the highest-value free source you have, for one reason:
it is the only one that **grows the trainable universe without shrinking it**. Extrapolated,
pool 5,542 → ~31,590 items with `target_items` 521 → 758. Two hard operational constraints:
~~Steam **soft-blocks with HTTP 200 and a stripped shell — there is no 429**, the block is
IP-scoped and multi-hour, so keep delay ≥6 s and stop on the first stripped page~~ **— FALSE
POSITIVE, refuted 2026-08-09. The page answers from this egress at 0.36 requests per item; the
stripped 230 KB shell is normal SSR, returned because the `market_hash_name` URL is not the
real page. Resolve the canonical `G<id>` first** (`2026-08-09-next-steps.md` D1); and the
page serves the **buyer** price, so divide by ~~1.1607~~ **1.15 — the 1.1607 is synthetic
(C1)**.

### What not to collect

Paid sources are out of scope by constraint. Beyond that: BUFF's historical dump (shrinks
`target_items` 521 → 426), `cs2-prices-tracker` (adds +1.6% rows to the gated set and
*shrinks* item count 141 → 139; also unlicensed), CSMarketAPI (quota permanently burned),
and any sentiment source (refuted, and the literature on sentiment-leads-price is a swamp of
positive-result low-rigour single-regime studies — the closest thing to a rigorous negative,
Bijl et al. 2016 on Google search volume, finds the *opposite sign* to the original result).

---

## 11. Venue microstructure and transaction costs

The strongest material in this whole review. Most of it is **original measurement taken
2026-08-07** against free bulk feeds, not web reading.

### Fees and round-trip cost

`RT` = minimum gross % move to break even buying then selling on the same venue.

| Venue | Seller fee | Proceeds | **RT breakeven** | Confidence |
|---|---|---|---:|---|
| **Steam Community Market** | 15% (5% Steam + 10% CS2) | **Steam Wallet — walled** | **+16.1%** | [VERIFIED on split] |
| **YouPin898** | 1% | cashable CNY | +1.0% | [REPORTED] |
| **CSFloat** | 2% | cashable | +2.0% | [REPORTED, sources conflict] |
| **DMarket** | 2% | cashable | +2.0% | [REPORTED — fee page 403'd; was 5% historically] |
| **Waxpeer** | 2% | cashable | +2.0% | [REPORTED, single source] |
| **BUFF163** | 2.5% | cashable CNY, Chinese ID | +2.6% | [REPORTED] |
| **Skinport** | **8%** (cut from 12%), 6% high-tier | cashable EUR/USD, KYC | +8.7% | [VERIFIED, skinport.com/faq/sales-fee] |
| **Bitskins** | 5% | **withdrawals reportedly offline June 2026** | +5.3% | [REPORTED, unverified] |
| **market.csgo.com** | 5% | cashable | +5.3% | [REPORTED] |

Steam Wallet is walled — this is primary-sourced from the Subscriber Agreement: *"Steam
Wallet funds… have no value outside Steam… have no cash value and are not exchangeable for
cash."*

### A falsifiable check on your own 1.1607 constant

The published rule gives buyer/net = `1.15` **exactly**. Your archive measured **1.1607**,
1.07pp higher. Direction and magnitude corroborate the 5%+10% rule, but the discrepancy is
diagnostic:

Steam ceilings each fee component to the cent with a $0.01 minimum, which makes the true
ratio **vary with price** (on a $0.03 sale it is 3.0, not 1.15). **A genuinely constant
1.1607 across 30,875 rows cannot come from rounding — it would have to be a hardcoded
multiplier in the source feed.**

~~**Do this:** regress the ratio on price.~~ **DONE — see C1. The slope is ≈ 0 and it is
synthetic**, so the Steam listing-page normalisation in `backfill_steam_listing_history.py`
is applying a constant where a price-dependent correction is needed, and rows below ~$0.50
carry a real basis error (~5% too high at $0.10–0.25). Still unfixed; the live-ingestion-path
part of this paragraph stands.

### The Steam premium is ~43%, not ~20% — measured across 23,904 items

Median venue price ÷ Steam price:

| Tier | n | BUFF163 ask | BUFF163 **bid** | CSFloat | Skinport |
|---|---:|---:|---:|---:|---:|
| <$1 | 9.4k | 0.658 | 0.448 | 0.750 | 0.812 |
| $1–10 | 7.8k | 0.713 | 0.554 | 0.726 | 0.813 |
| $10–50 | 3.0k | 0.729 | 0.582 | 0.719 | 0.797 |
| $50–500 | 3.3k | 0.716 | 0.634 | 0.704 | 0.778 |
| $1000+ | 86 | 0.699 | 0.651 | 0.688 | 0.771 |
| **ALL** | **23,904** | **0.700** | **0.550** | **0.727** | **0.802** |

Three consequences:

1. **Steam/BUFF ≈ 1.43, stable across tiers.** The commonly repeated "20–30% Buff discount"
   understates it by half.
2. **The wedge is structurally unarbitrageable.** Buying on BUFF at 0.70 and selling on
   Steam nets `1.00/1.1607 = 0.862` — *in Steam Wallet funds, which cannot be withdrawn*.
   The reverse leg is impossible (you cannot fund BUFF from Steam Wallet). This is why a 43%
   gap persists indefinitely and why it is a level shift, not a signal.
3. **This mechanistically explains your CSFloat basis refutation.** CSFloat/BUFF is
   **0.976–1.008 across every tier** — the cash venues agree with each other to within 2%.
   A CSFloat-vs-BUFF basis feature has ~2% of range to work with. It measured null because
   there is nothing there to measure. Consider that refutation closed with a mechanism, not
   just a p-value.

### Bid–ask spread — the one two-sided book, and it is already in a feed you poll

BUFF163 publishes **both** a lowest ask (`starting_at`) and a **highest buy order**
(`highest_order`), and both are in the free `prices.csgotrader.app/latest/buff163.json` dump
— **46,041 items**. Measured spread (n=22,449):

| Tier | median spread | p25 | p75 |
|---|---:|---:|---:|
| <$1 | **35.5%** | 9.1% | 100.0% |
| $1–10 | 21.1% | 9.8% | 64.6% |
| $10–50 | 17.3% | 8.3% | 59.7% |
| $50–500 | 10.8% | 6.2% | 24.8% |
| $1000+ | **5.2%** | 3.6% | 7.9% |

**Spread tightens monotonically with price — 35% sub-$1 down to 5% at $1000+.** This
inverts retail intuition: **expensive items are the liquid ones.** Sub-$1 items are
midpoints of a book nobody could transact in.

This is the strongest independent justification for `TRAIN_MIN_MEDIAN_PRICE` yet found —
and it suggests the honest floor is **above** $1, not at it.

### Free observability — live-probed 2026-08-07

| Endpoint | Auth | Status | Gives |
|---|---|---|---|
| `csgotrader.app/latest/buff163.json` | none | 200, 46,041 | **ask AND bid** + Doppler phases |
| `csgotrader.app/latest/csfloat.json` | none | 200, 46,041 | price + phases |
| `csgotrader.app/latest/steam.json` | none | 200, 34,413 | 24h/7d/30d/90d avg |
| `api.skinport.com/v1/items` | none | 200, 25,163 | min/max/mean/median + `quantity` (needs `Accept-Encoding: br`) |
| `api.skinport.com/v1/sales/history` | none | **429** | the only free universe-wide **completed-sale volume**; rate-limits hard, no `Retry-After` |
| `market.csgo.com/api/v2/prices/USD.json` | none | 200, 27,503 | price + listing count |
| `api.waxpeer.com/v1/prices` | none | 200, 22,119 | `count`, `min`, `steam_price` |
| `api.bitskins.com/market/insell/730` | none | 200, 10,920 | price + `quantity` |
| `steamcommunity.com/market/priceoverview/` | none | 200 | `volume` = 24h **sale count** |
| `steamcommunity.com/market/itemordershistogram` | none | **429 from this IP** | order-book depth; `item_nameid` map free at `somespecialone/steam-market-ids` (26,935 entries, WTFPL, verified 200) |
| `csfloat.com/api/v1/listings` | **key** | 403 | out of scope — 500 req/day |
| `api.dmarket.com/exchange/v1/market/items` | none | **410 Gone** | **retired — check the repo for dead references**; use `/marketplace-api/v2/offers` |

**Bid/ask exists on exactly two venues:** BUFF163 (free bulk, already polled) and Steam
(per-item, IP-fragile). Every other free feed gives **supply only**.

### Liquidity stratification

Universe: **46,631 distinct `market_hash_name`** across all free feeds. Only **395 items
(0.9%)** trade above $500; **10,152 (41%)** are sub-$1.

Median live listing count is roughly **flat across tiers (~8–26)** while spread collapses
from 35% to 5%. That combination means **listing count is not a liquidity proxy — it is
inventory**, and inventory ≈ trade rate × dwell time. A high count on a cheap item means
*slow*, not liquid. Any supply-depth feature is unsigned without a volume denominator.

**The number that reframes the whole project:** AK-47 | Redline (Field-Tested), one of the
most traded skins in the game, showed **`volume: 96` over 24 hours** on Steam and **10 sales
in 24h** on Skinport [VERIFIED, live calls]. Double-digit daily sale counts on a *flagship*
item is the ceiling on what any per-item daily model can resolve. Most of the universe is
single-digit or zero.

**This is probably the deepest explanation for your null streak.** It is not that the
features are wrong. It is that a per-item daily price change built on ~10–100 transactions
is counting noise, and no feature set recovers signal from a series that thin.

### Implications, condensed

| Fact | Effect on the pipeline |
|---|---|
| Steam price is a walled currency at ~1.43x cash | Your primary label is a *different asset*, not a noisy read of the same one |
| Cash venues agree to within 2% | Cross-venue basis features have no range — CSFloat refutation explained |
| Spread 5%→35% by tier | **A DA number pooled across tiers is uninterpretable.** A 3% predicted move is inside the spread for 5 of 6 tiers |
| 7-day market lock + 7-day trade protection | **No horizon under ~8 days is executable.** Your h=3 and h=7 forecasts describe moves nobody could trade |
| RT is +16.1% Steam / +2.0% CSFloat | Report accuracy conditioned on `\|predicted move\| > RT`, or the metric flatters a losing strategy |
| Only BUFF and Steam publish a bid | **BUFF `highest_order` is the one unexploited two-sided series, and it is in a dump you already fetch daily** |
| Flagship item = 96 sales/day | Per-item daily series are counting-noise dominated |

### Not verified — do not encode

Steam Support pages (7-day lock, 15-day authenticator hold) are a JS shell to automated
fetch; every such claim here is secondary. The Valve post announcing Trade Protection was
never seen at source. **The "8-day cooldown" has no source found at all — do not encode it.**
Regional restrictions and the Perfect World China client were not researched. Lead-lag
between venues cannot be measured from a snapshot — but you archive the csgotrader dumps
daily, so it is an internal analysis, not a web question.

## 12. Event history and market regimes

Grading below is the researcher's: **[DOCUMENTED]** = real numbers with identifiable
method, **[WIDELY REPORTED]** = many sites, all quoting Steam Market, no method shown,
**[FOLKLORE]** = universally asserted, no data found.

**A warning that applies to every number here:** market-cap trackers disagree by ~2x on the
same event (Pricempire $5.90B→$4.20B vs Esports News UK $6.06B→$3.08B vs CSMarketCap
$609M→$337M). They index different universes. Use direction and rough magnitude only.

### The finding that matters most: 2025-10-22/23

Valve extended the trade-up contract to **5 Covert → 1 knife or glove**, unannounced,
instantly marketable. This destroyed the case-only monopoly on knife/glove supply overnight.
**[DOCUMENTED — multiple independent indices plus Forbes]**

- EsportFire300: ~$50,000 → **$28,062.67, −45% in under 48h**
- **Knives and gloves −20% to −70%** (entry knives roughly halved: Safari Mesh $100–130 →
  $40–60)
- **Coverts UP 10–20x in extreme cases** — they became trade-up fuel
- **Cases barely participated**
- Recovery: ~+50% off the lows within 24h, 77% of pre-crash cap by 10-30
- Persistence: aggregate recovered; **standard knives/gloves are permanently repriced
  lower**. Float-capped finishes (Crimson Web, Case Hardened, Blue Steel, Damascus) held;
  easy-float-cap knives kept bleeding into 2026

**Why this is the most valuable single fact in this document.** It is the one date where the
cross-section dispersed enormously *and sign-flipped* for an identifiable, mechanical
reason. Knives fell and Coverts rose because of their **position in the supply mechanics** —
whether an item is trade-up *fuel* or trade-up *output* — not because of anything cosmetic.

That is a different feature class from anything this project has tested. The ByMykel bundle
was refuted, but ByMykel supplies *cosmetic* metadata (rarity, collection, weapon type).
What priced the October cross-section was **mechanical position**: is this item trade-up
input or output? is its case still in the active drop pool? is its float range capped? Those
are constructible from public data and they are not in the current feature set.

The honest counter is that this is one date, and a feature that only fires on unannounced
supply-rule changes cannot be predicted forward. But it is a real, dated, well-evidenced
cross-sectional dispersion event, and it is the only evidence found anywhere that item
attributes ever dominate the market factor.

### Regime series — for stress-testing and regime-break dating

| Regime | Dates | Driver | Class behaviour | Grade |
|---|---|---|---|---|
| R2 melt-up | 2023-03 → 09 | CS2 announced; speculation | **Cases led +50–75%**, skins lagged. Record 50.3M cases opened Apr 2023 — the rally created its own supply | Prices [WIDELY REPORTED], unboxing [DOCUMENTED] |
| R3 correction | 2023-09-27 → mid-Oct | CS2 launch disappointment | **Inverted: expensive skins fell hardest, cases barely moved.** EsportFire300 −10%+. Bottom ~2 weeks | [DOCUMENTED] |
| R6 case regime | 2024-12-17/18 → 2025 | Valve **silently** zeroes the rare drop pool — no announcement, found by data-miners | Cleanest supply shock in the record and **case-specific**: discontinued cases +15–57%, weapon skins unaffected | [WIDELY REPORTED] |
| R7 liquidity shock | 2025-07-15 | Trade Protection — all trades reversible 7 days | −25% in **one day**; hit liquid flip-traded items hardest, illiquid collectibles least; mostly reverted in ~4 days | [WIDELY REPORTED, single source] |
| R9 **the crash** | 2025-10-22/23 | Knife/glove trade-up | See above | [DOCUMENTED] |
| R10 rotation | 2025-11 → 2026-08 | Bot purge (2026-03-26, 960k accounts); souvenir trade-up (2026-05-22) | Capital concentrates into low-supply collectibles | [WIDELY REPORTED] |

**Your archive spans 2013-08-14 → 2026-08-04, so it contains every one of these.** They are
dated, labelled regime breaks sitting in data you already have — which makes them directly
usable as stress-test windows (§15) and as change-point anchors for a date-level factor
model (§6).

### Schedulability — the leakage boundary

**Known in advance (usable as forward-looking features):** Major tournament dates (months of
lead); sticker capsules land ~2 weeks *before* the Major, champion stickers ~2 weeks
*after*; souvenir supply windows (deterministic — souvenirs mint only during live Major
matches); operation start/end dates; the post-Major sticker sale start (but **not** its end).

**Surprises (unusable):** **every single one of the largest moves.** The Oct 2025 trade-up,
the Dec 2024 rare-pool removal (silent), the Jul 2025 Trade Protection, the May 2026
souvenir trade-up, the Mar 2026 bot purge, the 2016 C&D — all zero warning. Drop-pool
rotations, drop-rate changes, trade-rule changes, ban waves: all surprises.

**This vindicates your event-calendar refutation.** The researcher found **no dated,
quantified price series showing a Major or an operation moving the aggregate market.** The
calendar-known events don't have a documented aggregate effect; the events that do aren't
calendar-known. Your finding that the calendar feature behaves as a clock is exactly what
this evidence predicts.

What remains buildable without leakage: a **contemporaneous regime-break indicator** (a
market-factor level shift dated to a known update — backward-looking, not forward), and the
**souvenir/sticker supply windows**, which are genuinely deterministic but confined to that
sub-market.

### Two testable claims that cut against trader intuition

1. **Higher trading volume associates with *lower* prices** — from the fixed-effects panel
   study (≈10 years daily, 13 skins). Directly testable in your archive on the pre-2026
   volume series, which is intact. **[DOCUMENTED]**
2. **The 2016 gambling crackdown did NOT move prices.** PCGamesN and PC Gamer sampled Steam
   Market items; the ~10% dip on 2016-07-10 was the Cologne crate flood, and it fully
   recovered by 07-15. A *negative* result better evidenced than most positive ones.
   **[DOCUMENTED]**
3. **Player count does not lead prices.** MAU plateaued through 2025 while top-end prices
   rose; CS2 ATH was 1,862,531 concurrent on 2025-04-12. The null is better supported than
   the correlation. Consistent with your dead player-count collector being no real loss.

### Graded as folklore — do not build features on these

- **"Discontinued cases double in 2–3 years then grind up forever."** The single
  most-repeated and least-evidenced claim in the space. No index, no cohort study, no
  survivorship control.
- **"Major stickers return 50–150% in 3–6 months, 500–2000% over 5 years."** Cherry-picked
  from Katowice 2014 survivors. The 2026 Major-shop change may have killed the mechanism
  outright — Stage-1 team sticker revenue was ~$60k at Cologne 2026 vs ~$600k at Budapest
  **[DOCUMENTED, HLTV]**.
- **The 2020–21 COVID bull run.** Universally asserted, zero index evidence found.
- **"Duping/exploits move supply."** Investigated and found nothing supply-relevant; the
  real incidents were theft or gameplay bugs.

### Unresolved

Exact date of the rare-pool removal (2024-12-17 vs 12-18) — unannounced, no primary source.
No price data at all for Shattered Web / Broken Fang / Riptide despite verified dates. The
SteamAnalyst "2026 report" is internally inconsistent (a $2.4M "total market value") and
should not be cited.

## 13. Prior art and free data sources

### Prior art: there is nothing to catch up to

The entire GitHub population in this space is **0–17 stars**. One peer-reviewed paper, and
it does not survive scrutiny (§9). The recurring weak approaches, ranked by frequency:

1. **Cross-sectional price-*level* regression from static attributes.** Scores a flattering
   R² because price spans five orders of magnitude, and says nothing about returns. By far
   the most common failure.
2. Per-item LSTM/GAN on a few hundred daily points, no pooled model, no trivial benchmark.
3. Exogenous kitchen sink (Google Trends, Reddit, player counts) with no ablation.
4. No Steam fee, no trade lock, no costs. Only the Frontiers paper handles this.
5. No walk-forward, no purge/embargo.
6. RMSE/MAE in dollars on price level — dominated by expensive items.
7. Survivorship filtering to items with a complete record over the window.

**Your project is already past every one of these.**

Two worth knowing about specifically. `Sapphirine/202412-22-CSGO-Skin-Price-Predictor`
(13★, Columbia course project) is the one that tried the same exogenous stack — Google
Trends + Reddit sentiment + player counts — with four models, no baseline, no metrics.
Negative confirmation for your Reddit deletion. `morganaxmu/steamcommunity-market-price-
forecaster` (9★, MIT) is the most statistically literate small repo found — `auto.arima` +
GARCH with ADF and McLeod-Li diagnostics — and it **reports no accuracy metric at all**.

The arbitrage cluster (`kalekdev/CSGO-Trader`, ~15 zero-star Telegram scanners) shares one
sin: computing spread from **listing prices on both legs**, ignoring that the buy side is
one lowest ask with one unit of depth.

### Free data — the four finds that matter

| Asset | What | License |
|---|---|---|
| **`kieranpoc/counter-strike-market-sale-data`** (Kaggle) | 22,492 items, 99.3M data slices, Steam median price **+ volume**, hourly trailing month / daily before, back to 2013. Snapshot as of **2024-05-04** | **unconfirmed — verify on page** |
| **`atalantus/buff-price-history-archive`** (GitHub, 13★) | BUFF163 min price 2021-07-26 → 2024-01-19, **plus listing count populated after 2023-01-25** | **none — all rights reserved** |
| **`EricZhu-42/SteamTradingSiteTracker-Data`** (99★) | BUFF + Steam for ~3,000 deliberately liquid items, 2022-04-18 → 2024-02-14 (~10GB) | **MIT** — the only permissive archive found |
| **`somespecialone/steam-market-ids`** (90★) | 26,935 `item_nameid` entries — required for `itemordershistogram` | **WTFPL** |

**Two of these solve open problems in this repo.**

**Your `volume` column has been identically zero since 2026-04-16.** The kieranpoc Kaggle
dump is Steam median price *with volume* back to 2013 for 22,492 items. That is a real
historical trade-volume panel covering the entire period before your multi-source era —
i.e. exactly the field you lost, over the window where you have depth.

**Your supply-depth work is waiting on ~30 days of accumulation.** The atalantus archive
carries **BUFF listing counts for a full year, 2023-01-25 → 2024-01-19**, retroactively.
Note this is a *different use* of that repo from the one already evaluated here: as a
**price** source it was declined because it shrinks `target_items` 521 → 426. As a **listing-
count panel** it is the only free retroactive supply-depth series found anywhere, and the
`target_items` objection does not apply.

Also useful: `ModestSerhat/cs2-marketplace-ids` (248★) for BUFF/YouPin name normalisation,
`somespecialone/cs2-items-schema` (64★, MIT) as a ByMykel alternative,
`HilliamT/scm-price-history` (9★, MIT) for **login-free** Steam history fetching, and
`redlfox/awesome-cs2-trading` (105★, CC0) as an index of the space.

### Dead or out of scope

`csgobackpack.net/api/` now returns a Cloudflare 403 — unusable for automation.
`prices.csgotrader.app/latest/prices_v6.json` 301s to the website; the per-market files you
already use are the surviving path. `LukeX404/cs2-prices-tracker`'s `pricehistory/` leg is
**capped at 500 points and frozen at 2025-05-26** — a one-off snapshot, not maintained.

Paid and therefore out of scope: Pricempire ($119.90–239.90/mo, history Enterprise-only and
capped at 180 days), CSGOSKINS.GG (€179–279), cs2.sh (€70–185/mo), SteamWebAPI (€25–120/mo),
SteamApis, CSMarketCap ($9.99/mo). One telling detail: cs2.sh advertises Steam data back to
2013 — **the depth you would be paying for is exactly what the Steam `pricehistory`
endpoint gives away**, and what the kieranpoc dataset already scraped.

No commercial index (Pricempire, Skinflow, CSGOStocks, SteamDT) publishes a constituent
list, weighting scheme, or rebalancing rule. They are marketing charts. **This is why §7 —
building a real hedonic index — is genuinely novel rather than duplicative.**

---

## 14. Cross-asset structure — the one positive result

**[MEASURED HERE, 2026-08-07]** — not web research. 691 items (≥$1, ≥300 days),
2021-01-02 → 2025-12-31, 1,825 dates, **969,617 daily log returns**. Deliberately stops at
2025-12-31 because `source` is NULL for all pre-2026 rows while the 2026 files change source
composition on three dates — splicing those injects composition breaks straight into returns.

*(The float/pattern/sticker/trade-up-EV economics portion of this thread was not delivered;
§14 covers the cross-asset half only. Note the caveat in §15 that the archive keys on
`market_hash_name` and collapses float and pattern entirely, so most of that material would
have been unusable at this grain regardless.)*

### Is the hierarchy real or imposed? Partly real, and asymmetric

| Level | Verdict |
|---|---|
| **Global** | Real but weak — median item R² **11.5%**, β≈1. **Knives/gloves are not in it** (R² 1.1%) |
| **Class** | Real for **CASE only** (leave-one-out R² **13%**; within-class residual corr +0.18 vs −0.01 across). SKIN ≈ 0 — but partly by construction, since skins are 469 of 691 items and therefore *are* the market factor |
| **Crate** | **Real, and not a wear artifact.** Same-crate-different-skin residual corr **+0.084** vs **+0.003** different-crate — a **28× ratio** |

The crate row is the actionable one: **crate membership is a genuine risk factor for recent
releases and has decayed to nothing for old ones.** Pooled incremental R² is ~1%, which hides
enormous dispersion — Gallery **0.345**, Kilowatt **0.104**, legacy crates zero. It also
confirms the metadata join is live: `item-metadata-bymykel.parquet` carries
`type_meta_crate_id`/`type_meta_collection_id` and **687 of 691 items join**.

Note the tension with §2's ByMykel refutation, and that it is not a contradiction: the
permutation test showed the model does not *use* ByMykel's cosmetic columns as features.
This is a different claim — that `crate_id` defines a **correlated residual group**. That is a
covariance-structure fact, not a mean-prediction fact, and it argues for crate as a *grouping
variable* (hierarchical shrinkage, block bootstrap unit) rather than as a column.

### Lead-lag: both folk hypotheses refuted, one real effect survives

**Cases do NOT lead skins.** Raw +0.236 at lag 1 looks convincing and dies on global-factor
residuals (**Granger p = 0.61**), and is absent in 2021–22. Cases lead *the market* by a day
because they are **22× the median daily volume** of any other class — there is no
case-specific information about skins.

**Cheap does NOT lead expensive — the sign is reversed.**

| Direction | Lag-1 corr | Verdict |
|---|---|---|
| Cheap → expensive | +0.043 | Inside the noise band |
| **Expensive → cheap** | **+0.213 (z = 9.1)** | **Granger incremental R² 9.0%**; stable in 4 of 5 years; survives dropping all zero-change observations **and** removing the market factor (0.122, R² 4.5%) |

Same direction as SKIN → KNIFE_GLOVE. The STICKER → SKIN lead (pooled z=3.5) **flips sign
across years** — an averaging artifact.

### The organising principle is liquidity ordering, not price or asset class

**The most liquid corner prices information first; everything else catches up a day later.**
This is consistent with the counting-noise mechanism in §11 — thin series cannot incorporate
information promptly — and it is the **first positive structure measured in this archive**.

It also explains a result that previously had no mechanism: §11 found spread tightens
monotonically with price (35.5% sub-$1 → 5.2% at $1000+), i.e. **expensive items are the
liquid ones**. Expensive→cheap lead is exactly what that predicts.

### Two caveats, held honestly

**The staleness objection is not fully closed.** Cheap skins have the highest zero-change rate
(1.33% vs 0.16%), and a partially-updating cheap index would produce this signature. The
argument against is that the cheap index's own AR(1) is only 0.125 — too small a footprint for
a partial-adjustment model generating a 0.21 lag-1 cross-correlation. **Judged mostly real,
not proven clean.** The decisive test is §16's `stale_run_days`: drop high-stale-run item-days
and re-measure.

**Multiplicity applies.** One archive, one window, and **three lead-lag hypotheses were
tested**, so §3's t > 3.0 hurdle binds. z = 9.1 clears it comfortably; the other two do not
and should be treated as null.

### What to do with it

The candidate feature is **lagged expensive-tier return as a predictor of cheap-tier return**
— a *date × tier* quantity, not a per-item one. That places it in the §6 date-level frame
where the MDE is ~0.3pp rather than the item-level frame where it is 2.21–3.69pp. It is the
best-evidenced new feature in this document and it is cheap: two tier indices and a lag.

## 15. Dataset design

### Row grain

| Grain | Does the data exist? |
|---|---|
| **item × day** | **Yes — the only grain that exists.** `engineer_features` already resamples to it on line 1 |
| item × hour | **No.** Collectors run ~every 6h ⇒ 4 obs/day max, and those are *quote snapshots, not trades*. 20 of 24 hourly rows would be forward-filled. Pre-2026 history is one row per item-day |
| item × 5 min | **No, and not obtainable.** 99.7% interpolation of nothing |
| per transaction | **No universe-wide free source.** CSFloat graph is *daily* (500 req/**day** ⇒ ~92 years to sweep 46k items); Steam listing pages are *daily*; Skinport `sales/history` 429s. Theoretically correct, unobtainable |
| event-based | **Data exists, sample does not** — ~6 real regime breaks in 13 years. Stress-test windows, never a modelling grain |

**Recommendation: item × day, plus a second, much smaller `market_day` table.** The argument
is stronger than "daily is fine": **item×day is the finest grain at which a row is an
observation rather than an interpolation.** Going finer adds forward-fill, and forward-fill
on a thin series is the Getmansky–Lo–Makarov MA(k) mechanism — it *manufactures*
autocorrelation the model will read as momentum.

The `market_day` table is the new work. §6 needs a 4,735-row series with a
composition-controlled index; that is a different key, not a `GROUP BY`. Keeping it separate
is what stops the market factor being recomputed inconsistently in five places — it currently
is (`_apply_market_aggregates`, `market_factor.py`, and each `ab_test_*.py`).

### Two integrity columns that are not features

- **`n_ask_sources`** — the voting median's basis *changes with the source set*, and the
  source set moved 5,542 → 41,725 items mid-sample. Without this, a basis change is
  indistinguishable from a return. The mean market return reads **−31.6% on 2026-03-22** and
  **+17.4%/−17.8% on 2026-07-09/10** against ±0.5% on a normal day. This is the best
  available guess at the cause of §3 defect 3 (harness not reproducible).
- **`label_void_reason`** — `_snapshot_dates` / `_collection_shift_dates` already run inside
  `prepare_targets` and are recomputed independently by every harness. Materialising the
  result is what makes the harnesses agree with production.

### Cut, with reasons

`volume` as currently written (identically 0 since 2026-04-16, stored as 0 not NULL, which
defeats every `notna()` guard — replace under a *different name* so a stale artifact cannot
silently read the dead column) · `min_price`/`max_price` (100% NULL outside a 39-day window)
· `player_count` (collector dead; §12 finds the null better supported) · all `social_*` (five
columns computed as constant 0.0) · Google Trends / Twitch (§13's recurring failure mode) ·
per-item forward event calendar · Doppler phase splits (multiply the key ~5× for a few
hundred items with no label coverage) · intraday timestamps (averaged away in
`engineer_features`' first statement).

**Item attributes move to a separate `item_attr` dimension table** rather than being cut.
They are useless for *returns* (ByMykel permutation test) and **mandatory for a *level*
index** (§7's hedonic regressors). Same fields, opposite verdicts, different consumers — not
a contradiction.

## 16. Feature engineering

**The current production set is 33 columns, all `price_technicals`** — verified by running
`_select_feature_cols` → `_apply_feature_allowlist`. ~~117~~ **123** numeric columns survive
shelving (re-counted 2026-08-09; the prefix bug below is still live and still correct);
the allowlist discards eight whole groups (`item_identity` 26, `other` 17, `events` 15,
`cross_sectional` 13, `social` 5, `supply_depth` 5, `temporal` 3) *after paying full compute*.

**A free bug fix found in the read:** `distance_to_support`, `distance_to_resistance` and
`high_low_range_30d` are computed every run and silently discarded, because `_feature_group`
matches prefix `support_` while the columns are named `distance_to_*`. Three scale-free
features, zero new data, one-line fix.

### The new/untested candidates that matter

| Feature | Why | Blocked on |
|---|---|---|
| **`stale_run_days`** | Consecutive bit-identical prices. Mechanically the GLM θ: a stale series under-reports, so the next change is a catch-up with predictable sign continuation. ~~Also the cleanest split between real items and artifacts (37–42% at tier 0 vs 0–1.8% at ≥$1)~~ — **the split does not exist on the voted series (12–27% at ≥$1); banner item 5** | **Nothing — free, 13 years deep.** ~~Highest-ranked new item-level feature~~ **Shipped 2026-08-08 as a label-side filter (`staleness.py::stale_run_days`); still open as a feature, but without the evidence quoted here** |
| **`roll_spread`** | Roll (1984) recovers an *effective spread from price alone* — the only route to a liquidity variable for the 13 pre-multi-source years and the 24,000 items with no bid feed | Nothing |
| **`spread_resid`** | `spread_rel` minus its tier-conditional median. Raw spread is **monotone in price** (35.5%→5.2%), so unresidualised it is a price-tier proxy — the exact defect that shelved the dollar-scale features | 25 days of bid history |
| **`d_bid − d_ask`** (order-flow imbalance) | Decomposes a mid move into demand-side vs supply-side. A mid that rose because the *bid* rose is a different event. Canonical short-horizon predictor in equity microstructure | 25 days of bid history |
| **`steam_buff_ratio_z30`** | The *level* of the Steam wedge is unarbitrageable and therefore dead; the *deviation from its own 30d mean* is not. The only survivor of the cross-market family | Nothing |
| **`is_tradeup_fuel` / `is_tradeup_output` / `drop_pool_active` / `float_range_capped`** | The Oct-2025 cross-section sorted on exactly these. Float-capped finishes *held* while easy-cap knives bled into 2026 — a documented mechanical split | Public collection data |
| **Within-date rank transform of every surviving feature** | `groupby("date").rank(pct=True)`. Removes the market factor from the **feature side** by construction, pairs with `lambdarank`, makes everything scale-free without a shelving list, kills the heavy tails on `autocorr_*` | **Nothing — cheapest structural change available** |

**Hard constraint on the whole liquidity family:** 25 days of paired bid/ask history. At
`CV_STEP_DAYS = 150` that produces **zero additional folds**. Not testable on the current
archive, and the atalantus backfill carries listing counts, not bids — no retroactive bid
source was found anywhere.

## 17. Target variable

| Target | Verdict |
|---|---|
| Future raw price | **Never.** §13's #1 recurring failure. Pettersson's R²≈0.50 is this |
| % return | Current primary. Asymmetric; on a penny item one cent is 20% |
| Direction, 3-class | Current. The ±0.5% flat band is **inside the bid-ask spread for five of six tiers**, so "flat" is partly a quoting artifact |
| **P(exceed X%)** with X = round-trip cost | **The economically honest target.** The only one that makes a 3% prediction distinguishable from no prediction |
| **Within-date rank** | Cancels the market factor structurally. `lambdarank` ships in LightGBM |
| **Volatility** | **Better supported than direction, and half-shipped already** — `price_cv_60d` *is* the conformal σ, pinned by test. Nobody proposed it as an explicit target |
| Risk-adjusted return | **Not recommended** — σ is the only estimable term, so r/σ collapses to 1/σ, which is a liquidity sort in disguise |

### Horizons

| h | Label coverage | Executable? | Verdict |
|---|---|---|---|
| <24h | none | No | Impossible — 4 quote snapshots/day |
| 3d | 22,093 usable | **No** (inside the lock) | **Deprecate.** Best-covered horizon, describes untradeable moves, and it is what makes the pooled DA look healthy |
| 7d | 22,143 usable | No (at the boundary) | Demote to informational |
| **14d** | 11,040 usable | **Yes** | **PRIMARY** — shortest executable horizon with real coverage |
| **30d** | **5,461, all from one backdated date** | Yes | **SECONDARY, blocked on measurement.** Carries both the +3.50pp positive and the +6pp unpurged inflation, on 5,461 rows from one date. No h=30 claim until ≥30 forecast dates mature |
| 90d | zero | Yes | **Do not build** — spans a regime break more often than not |

**Recommended target — decompose rather than replace:**

```
P(up_{i,t+h}) = calibrate( base_rate_{t+h}  +  β · rank_{i,t}(score) )
                            ↑                        ↑
                  date-level model (§6)      lambdarank within date (§5)
                  4,735 days, MDE ~0.3pp     cancels the market factor
```

This is the only decomposition consistent with the measurement record. Asking one model to
supply both halves is asking the item-level half to carry variance it demonstrably does not
have — which is what "everything the model was doing was riding a common factor" means. Both
halves get metrics with clean nulls (PT on the date leg, rank-IC on the item leg). Cost: an
objective swap, one small date-level model, and isotonic calibration on existing OOF folds.
**No new data source is on the critical path.**

**Secondary: the return distribution.** Best-evidenced target in the document and nobody
proposed it. Three caveats to encode: centre it on the date-level base rate, not the q50
regressor (which loses to a constant call); **widen** bands for high-`stale_run_days` items
because GLM smoothing biases realised vol *down*; and report net of tier-specific friction,
or the interval describes a move the user cannot capture.

### The two user questions, answered straight

**"Is this item likely to increase over the next X days?"** — Three parts, two answerable.
(1) *The market part*: "CS2 is more likely than usual to be up over 14 days" is real,
measurable, backed by 4,735 days, and **currently not modelled at all** while being silently
80–100% of the answer. (2) *The relative part*: "among ≥$1 items, this is top-decile" is what
`lambdarank` would establish or refute — never run in a form the loss did not confound.
(3) *The item-specific-direction part*: **measured absent.** The product should say (1) and
(2) and stop.

**"What is the expected return distribution?"** — This one the data genuinely supports, and
it is the better product.

## 18. Data leakage — ranked for this repo

**The first fact is schema-level:** ~~`CANONICAL_PRICE_COLUMNS = ("item_slug","day","source",
"mean_price","volume")`. **There is no ingestion or arrival timestamp anywhere in the price
archive**~~ — **fixed 2026-08-08: `backend/db/archive.py:40` now carries six columns including
`ingested_at`.** The retroactive half of the problem is permanent — the column is NULL for
every row written before that date and cannot be reconstructed backwards — so every vector
below that depends on *when a row landed* remains **unauditable for the historical panel**,
and auditable from 2026-08-08 forward.

| # | Vector | Live? | Contaminates |
|---|---|---|---|
| **L1** | **Date proxies × unpurged boundary** | ~~**Yes.** `walkforward_backtest.py --purge` is **default OFF**~~ — **fixed 2026-08-08: `set_defaults(purge=True)`, embargo `h+13`, `--no-purge` as the legacy escape.** ~~and **ten further `ab_test_*` harnesses**~~ — **15 exist**, and the live residuals are different: 9 of 15 call `phase_collapsed_sql_filter()` where invariant 2 requires `archive_universe_sql_filter()`, and 14 of 15 still early-stop against the window they score (`ab_test_frozen_runs.py` inherits one via `walkforward_backtest.py:490`; only `ab_test_direction_labels.py` is exempt) | The published series; ≥10 ledger entries |
| **L2** | **Full-sample item selection.** `_filter_dead_items`, `_filter_by_median_price`, `_flag_corrupt_items`, `_stratified_item_subsample` all run on the **entire window before any split**. "Items whose median 2013→2026 price is ≥$1" is not a set anyone could have named in 2019 | **Yes** — measured: 876 items used where 319 were knowable | ~~**The +3.50pp ≥$1 result.** The effect appears *only* at h=30 — the signature a look-ahead selection produces~~ **Both halves refuted 2026-08-08.** The leak is real but worth **−0.004pp [−0.664, +0.851]**, and the +3.50pp it was offered to explain re-derives as **+1.642pp, null**. Banner item 2 |
| **L3** | **Overlapping labels.** Daily rows with a 30d target ⇒ each price appears in 30 labels. CIs ~√30 too narrow; ~~early stopping stops late~~ **direction reversed — early stopping on the thin trailing window stopped *at round 1*, destroying 23–88% of rank IC; replaced by `FIXED_BOOST_ROUNDS` (`forecaster.py:618`)** | **Yes** | Every interval outside `paired_mde` |
| **L4** | **Revision/backfill with no arrival time.** Six backfill writers. The Steam listing page returns Steam's *2026* rendering of 2015, then divides by a constant now known to be synthetic (C1) | **Yes, unauditable** | Reproducibility — this is the best candidate for §3 defect 3 |
| **L5** | **Survivorship via `is_backfilled`** (derived from `source IS NULL`). Every pre-2026 training row is conditioned on the item still being collected in 2026. "Long history" and "survived" are the same variable | **Yes** | The whole pre-2026 panel |
| **L6** | **`events_next_30d_*` is an explicit forward count** over `(date, date+30d]`. If any §12 surprise is in the `events` table typed `update`, the model knew a −45% shock was coming | **[NEEDS AUDIT]** — one query settles it | Event-calendar results |
| L7 | Metadata as-of-today (rarity is the *stratification key*, so it perturbs which items are drawn) | Yes, small | Item draw, not features |
| L8 | Cross-source derived duplicates — voting **rewards agreement**, so a venue republishing another's price manufactures artificial precision | **[NEEDS AUDIT]** | Voting; any lead-lag work |
| L9 | `volume == 0` is a perfect indicator for "after 2026-04-16"; `_apply_market_aggregates` gates on a whole-frame flag | Yes | Any arm trained across the cutover |
| L10 | Carry-forward fills | Yes — **but backward-only and verified clean.** `merge_asof(direction="backward")`, labels use an *exact* merge. The damage is GLM smoothing, a metric problem, not leakage |
| **L12** | **Random splitting** | **No — verified clean.** `sample()` operates only on `train_set` after the date split; `val_set` is never thinned. Recorded so nobody spends a day on it |
| L13 | Conformal | No — `q_hat` fits on CV-pooled OOF records; `SKIP_CV` removal is pinned by test. Best-guarded part of the pipeline |

## 19. Validation strategy

**Dual track, one rule: ship and calibrate on purged expanding-window walk-forward; decide
features on CPCV; never swap them.** Track A must stay causal because a conformal band
calibrated on a non-causal split has no coverage meaning. Track B (CPCV, N=12/k=2 → 66
splits, **11 backtest paths**) gives a *distribution*, which is what PBO and the Deflated
Sharpe need as input — you cannot compute PBO from a single walk-forward path, which is why
§3's multiplicity problem currently has no instrument.

### The embargo rule, stated as a rule

```
embargo_days = H + LAG_TOLERANCE_DAYS(3) + SMOOTH_WINDOW(3) + MAX_WINDOW_SPAN_DAYS(7)
             = H + 13
```

Concretely: **h=3 → 16 days; h=7 → 20; h=14 → 27; h=30 → 43.** `_purge_overlapping_train_rows`
currently cuts at exactly `H` — correct as a purge, **short by 13 days as an embargo**. Each
term is a real reach-back: the as-of lag tolerance, the 3-observation scoring median, and the
7-day span those observations may cover. At h=30 the embargo (43d) exceeds the validation
window (30d) — that is the correct cost, not a bug.

### Regime stress-test suite

Protocol: train ends at `break_date − embargo_days`; test opens at `break_date`. **No PASS is
an accuracy threshold** — §8 settles that this is an information product, so what must
survive a break is calibration, not hit rate.

| # | Break | Discriminating question | PASS |
|---|---|---|---|
| S1 | 2023-09-27 CS2 launch | Cross-section **inverts** (expensive hit hardest, cases flat) | Coverage ≥70% on tiers 3–4; model does **not** call the same direction for cases and coverts — a single-factor model will, and that failure is the finding |
| S2 | 2024-12-17/18 rare-pool removal | Purest test: unannounced, class-specific, no calendar feature can help | Rank-IC on the case sub-universe not significantly negative. The one break where a supply-position feature could show signal — measure both arms |
| S3 | 2025-07-15 Trade Protection | One-day −25%, mostly reverted in 4 days | **No direction flip between day 1 and day 5** — the model must not chase the shock into the reversion |
| **S4** | **2025-10-22/23 crash** | **The discriminating test of the whole project.** Does the model rank Coverts above knives on 10-23? | A market-factor-only model **cannot**, and that null is the correct, publishable result. Do not set an accuracy PASS on 10-23 — no leak-free model predicts an unannounced rule change |
| S5 | 2026-03-26 bot purge | Does `_collection_shift_dates` fire? | **It must**, if the universe moved. If not, the label-voiding rule has a blind spot |
| S6 | 2026-05-22 souvenir trade-up | Does anything learned at S4 **transfer**? | Souvenirs no worse than a matched control. A significant negative means the model memorised S4's sign |

**Aggregate:** the model may fail to *predict* all six — expected, not a defect. It fails the
suite if coverage drops below 60%, the direction call becomes degenerate (>90% one class), or
PT goes significantly **negative** (anti-informative). And `n = 6` supports no significance
claim; the suite catches catastrophes and forces each cross-section to be looked at.

### Liquidity stratification

`price_tier` tops out at tier 4 = ≥$100, **merging the 10.8%-spread and 5.2%-spread cohorts
— the two most different liquidity populations in the market.** Split it. Report on a 2-D
grid (6 price bands × 4 staleness quartiles), never pooled. Sub-$1 is diagnostic-only and
never headline. Test the headline at $1, $5 and $20 floors and see where it stabilises.

## 20. Evaluation metrics

### Why low RMSE means nothing here — three independent reasons

**Scale.** Prices span $0.03 → $10,000+. One $10,000 knife mispriced 5% contributes 250,000
to the squared sum; a $0.50 sticker mispriced **100%** contributes 0.25. **It takes a million
perfectly-wrong stickers to equal one mildly-wrong knife.** RMSE on levels measures how
little the model moves the top 0.9% of items, and the optimal RMSE strategy is to predict
last price — the naive benchmark, scoring beautifully with zero information.

**The inverse failure.** 41% of the archive is sub-$1, where the $0.01 tick is a 33% quantum.
MAPE/SMAPE there measure price quantisation. So **dollar metrics are dominated by 1% of items
and percent metrics by 41%, and neither population is the product.** No weighting fixes this;
only stratification does.

**Zero error and zero utility are compatible.** RMSE of 0.5% at h=30 on a $50 item is an
excellent forecast and worthless: round trip is +16.1% Steam / +2.0% CSFloat and the tier
spread is 10.8%. A 3% predicted move is inside the spread for five of six tiers. **No error
metric can see this, because friction is not in the loss.**

### The reported suite, in priority order

Every metric per horizon × per tier, **fold-clustered, with `n_dates` and `n_clusters`
printed**. A metric quoted without its cluster count should be treated as unquoted.

| # | Metric | Threshold |
|---|---|---|
| 1 | **Pesaran–Timmermann** (Blaskowitz–Herwartz robust variant), per date, t-stat over dates | **t > 3.0** (Harvey–Liu–Zhu), ≥$1, `n_dates ≥ 20`. Report even when it fails — a *negative* PT is a finding |
| 2 | **Within-date rank-IC** + IC-IR | Mean IC > 0, fold-clustered CI excluding zero. **IC-IR > 0.05** is the realistic bar for this noise level, not 0.5 |
| 3 | **Brier skill score vs the per-date realised base rate** | BSS > 0 |
| 4 | **Conformal coverage vs nominal 80%**, per tier and per regime window | \|error\| ≤ 5pp overall; ≥60% in any single regime window. Failures here are the ACI business case |
| 5 | **Friction-conditioned actionable accuracy** (below) | see below |
| 6 | Per-class precision/recall with base rate printed beside | No class >80% of calls |
| 7 | R²_OOS **on returns** vs the per-date market mean | >0. Expect ≈0. **Never on levels** |
| 8 | Raw DA **+ constant-call baseline + realised down-rate**, always as a triple | DA is never quoted alone |
| 10 | Deflated Sharpe + PBO from CPCV's 11 paths | Trial count (≥15 A/Bs) declared |

**Dropped:** RMSE/MAE on levels, MAPE, R² on levels, raw Sharpe, max drawdown, pooled ROC-AUC.

### The friction-aware metric

`ActionableDA(v,h) = P( sign(r_act) = sign(r̂) | |r̂| > RT_v + s_i )`, with `s_i` the item's
BUFF relative spread (free, already fetched) falling back to the tier median. Report **four
numbers together**: `n_actionable / n_total`, `ActionableDA`, `E[net]`, and PT on the subset.

Scope: **h ∈ {14, 30} only** (an actionable metric at h=3 is a category error), venue =
CSFloat (RT 2.0%), cohort ≥$1 and also at a $20 floor.

**Expected result: failure, on `n_actionable` first.** That is a publishable internal result,
and it converts §8's qualitative argument into a number the dashboard can carry.

## 21. Architecture and maturity tiers

Read as a change list, not a build list. Everything below exists; the column that matters is
the guard, because this project's losses have all come from stages that were built correctly
and then silently stopped being true.

### The monitoring rule

The single most repeated failure here is **green CI that is not collecting** — three
scheduled jobs reported success while storing zero rows; the supply scraper stored 0 rows for
a month behind a green badge; event correlation has written 0 rows since 2026-07-19 because
it queries a table that is empty by design. In every case the workflow did what it was told
and the badge was accurate. **Badge colour measures the runner, not the data.**

Monitor the **output table's freshness and row count, plus the input table's freshness for
any derived job.** Nothing else is evidence.

| Signal | Alert | Caught by history |
|---|---|---|
| `archive_max_day()` **at origin** | not advanced in 24h | Mirror never pushed; 4 missing days across 40/45 green runs |
| Rows/day **per source label** | any label <50% of trailing median | A venue retiring behind a 200 is invisible in the total |
| **Per-column degeneracy** — fraction equal to column mode | >0.99 | **`volume` dead as zeros defeats every `notna()` guard.** This guard catches the next dead column with no code change |
| Distinct items per source | ±20% | The collector-cutover signature — universe size, never prices |
| Serve-time median-fill rate per feature | any booster-reaching feature >20% | 100%-filled lag features on 2026-08-04 |
| Realised vs **requested** training universe in `meta.json` | mismatch | The 1.8% subsample no-op |
| Band coverage vs 80% | drift >5pp over 30d | p10/p90 delivered 39–48% against 80% for months |
| Chained run `headSha` | skipped ≥2 days | 17-day outage; fixes twice sat unpushed |

**Mechanism (free):** one `pipeline-health.yml` on its own cron, **deliberately not chained
off the daily chain — a chain that stops running cannot alert on itself.** Files a labelled
GitHub issue via `gh`, the mechanism `backtest-accuracy.yml` already proved works.

**Two structural guards:** a task returning no count fields defeats the zero-row guard, so
that should be a test over the task registry, not a convention. And **assert the artifact at
its destination** — push-then-verify, always that order, against origin.

### Where this system actually is

| Layer | Tier |
|---|---|
| Data & pipeline engineering | **STRONG, approaching ADVANCED** — past every failure in §13's prior-art list |
| **Measurement** | **BELOW MVP** |
| Modelling | MVP-adequate but exhausted |

**A system that honestly reports "no edge" is above MVP. This one reports 67–68% (a
penny-item score) and three refutations whose intervals are wrong.** That is the below-MVP
condition, and it is fixable in days **without a single retrain** — which is the key
scheduling fact: the highest-value work available is free of the Monday retrain budget
entirely.

**MVP entry → STRONG:** headline is PT-based and tier-conditioned with `n_dates`; ≥20 forecast
dates accumulated; two identical harness invocations return identical `mean_diff_pp`; a
seed-only placebo returns a CI containing zero. Note (b) is a calendar wait, not work — which
is exactly why the metric fixes should start now, while the wait is free.

**STRONG entry → ADVANCED:** a composition-controlled index exists and is stable when the
observed item set changes; the date-level model has been scored against persistence with a
t-stat over dates; train and serve cohorts are the same population.

### Advanced techniques — what is actually appropriate

| Technique | Verdict |
|---|---|
| **Hierarchical / partial pooling** | **Appropriate — across *experiments*, not items.** Jensen–Kelly–Pedersen shrinkage for a dozen A/Bs against one panel. Item-level hierarchy is not: that layer is measured empty |
| **Regime detection** | **Appropriate on the index only**, and **contemporaneous only** — the forward version is unbuildable |
| **Adaptive conformal** | **Appropriate and user-visible.** Exchangeability is exactly what six regime breaks violate |
| **Event detection** | **Appropriate in a narrow form** — mechanical supply-position features, plus deterministic souvenir/sticker windows |
| Item/collection embeddings | **Inappropriate.** Would learn item identity against a percentage target — the defect that shelved 37 dollar-scale columns, in a form no property test catches |
| Deep sequence models (TFT/N-BEATS/LSTM) | **Inappropriate and closed** on evidence (§9) *and* mechanism (§11) |
| Per-item ARIMA/GARCH | **Inappropriate** — Montero-Manso & Hyndman; your design is already ahead of the published work |
| "Virtue of complexity" ridgeless | **Worth exactly one A/B**, after MVP. It contradicts every model-shrinking decision made here and the knobs already exist |

**Entry criterion into any ADVANCED item: the date-level target must have produced a
statistically honest positive first.** Absent that, advanced machinery is applied to a series
with no demonstrated predictability — precisely the failure mode §9 identifies in both
published CS2 papers.

## 22. Manipulation and data quality

**The disposition that matters:** given a 2.21–3.69pp item-level MDE, **no manipulation-
detection *feature* will clear the gate.** But removing bad quotes from the **label** shrinks
noise variance, which raises resolvable effect size for everything else. That is the only
route by which this work pays.

### What is documented vs. asserted

Valve's 2019-10-28 key-trade restriction is primary-sourced and unambiguous: *"nearly all key
purchases that end up being traded or sold on the marketplace are believed to be
fraud-sourced."* **[DOCUMENTED]** Beyond that, **no study, dataset, or documented case of
wash trading on Steam Market or any skin venue exists** **[DOCUMENTED negative — OpenAlex
sweep returns only gambling-harm and loot-box literature]**. The adjacent quantification:
>70% of volume on unregulated crypto exchanges is wash trades, but NFT wash trading is only
**2.04% of sale transactions** — and the NFT number is the better prior, because wash trading
is cheap where fees are near zero and expensive where they are 2–16%. **A 15% Steam
round-trip is a strong structural deterrent.** Low-fee venues (2%) are where it would be
economic.

Cornering, pump groups and bait listings are described consistently across sources but every
number is unsourced trade-blog content. **Directionally real, quantitatively worthless.**

### Detection rules, with dispositions

| # | Anomaly | Informative or noise? | **Disposition** |
|---|---|---|---|
| D1 | Cross-source disagreement (`\|z\|>4` on MAD, ≥3 consecutive days) | Noise if one source deviates; **informative if the median itself jumps** | Remove the source-day from the vote. Production rejects >2σ; `walkforward_backtest.py::_load_all_prices` uses a plain **mean** and does not |
| **D2** | **Frozen-price runs** (≥3d bit-identical) | **Noise — the MA(k) mechanism, 37–42% of tier 0** | **Remove from the label set at run ≥2, or downweight by 1/(1+run_len). The single highest-value cleaning action — it improves the measurement instrument itself** |
| D3 | Price-change outlier | **Both** — that is the rule's job | **Never blanket-remove.** Confirmed on ≥3 venues ⇒ **keep** (it is a supply shock). Unconfirmed ⇒ remove the quote, not the day |
| D4 | Benford / round-number clustering | Source-quality signal | **One-off diagnostic across all 11 sources**, not a feature. The only published validated test for fabricated venue data |
| D6 | **Buy-order spoofing** | Noise | **Gate `highest_order` on implied spread inside its tier's [p10,p90] before using it.** A naked bid feed is the most spoofable series in the stack |
| D7 | Cornering (listing count −50% while price rises) | **Informative — a real supply shock** | **Model separately.** Tag and exclude from training, report separately. Not a "cornering score" feature |
| **D9** | **Doppler/phase name collision** — one `market_hash_name` covers Ruby / Sapphire / Black Pearl / P1–P4, with measured within-name dispersion of thousands of dollars | **Noise — a composition artifact.** The price moves when the *phase mix on offer* changes, with no change in any asset's value | **Remove from the item universe entirely. These names are not single assets.** Same failure that killed CSFloat `avg_price` (float-composition noise), different attribute — and it would contaminate the §7 hedonic index before it is built |
| D11 | Trade-bot / operator-set prices | Noise | **Source-level allowlist: ingest only executable orders or completed sales.** At least one competing aggregator ingests CS.MONEY, which is reported at a 33% average discount. This repo does not — keep it that way |

### Two things not to do

**Do not winsorise large daily returns.** The 2025-10-22 cross-section is the only dated
event where item attributes dominated the market factor. Clipping outliers deletes the one
observation carrying information about mechanical supply position. Use D3's cross-source
confirmation gate instead of a magnitude threshold.

**Do not add a "manipulation score" feature.** It would fire on <1% of item-days against a
2.21pp MDE, return null, and — since trees are robust to uninformative features — return null
*quietly*, becoming another entry in the refutation ledger.

## 23. Sentiment and event detection

**The sentiment premise is unsupported rather than refuted, because there is no literature.**
OpenAlex on Counter-Strike/skins/virtual-item markets returns gambling-harm research
exclusively; arXiv `all:"CS:GO" AND all:"skin"` returns **0 results**. Combined with this
repo's own measurement and the counting-noise argument, **there is no case for rebuilding
sentiment.**

**The strong case is event detection from official Valve sources, and it is better than
expected** (see C3). `api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=730` is live,
free, and returns **1,752 items back to 2012-03-16** with full body text when `maxlength` is
omitted. `feed_type == 1` isolates Valve's own announcements. The legacy
`blog.counter-strike.net` RSS is **dead — newest item 2023-04-25**; do not use it.

**Regex cannot do this and it has been measured here: 33% recall (14/42)**, misfiring on both
drop-pool *removals* ("Removed Gallery Case") and incidental prose ("Fixed a case where…").
Both failure modes are exactly what an LLM fixes — lexical collision and sign-blindness.

Practical shape: **1,752 items over 14 years is a one-off backfill**, not a daily cost (~1–3
posts/week incremental). The task is **structured extraction, not sentiment** — output
`{supply_affecting, direction, mechanism, scope, confidence}`. **Validation is available and
cheap**: 6 dated ground-truth events plus 42 ByMykel crate dates. Score recall/precision
before trusting anything.

**Two operational traps.** Valve posts land **21:00–01:00 UTC**, straddling the day boundary
and the 23:00 UTC aggregator cron — date alignment must be handled explicitly or events land
on the wrong day roughly half the time. And `news_events` dedupes on `(day, title)` while
"Counter-Strike 2 Update" repeats, so two updates in one UTC day currently collapse to one.

**Feed the output as a date-level regime indicator into the §6 market-factor model** — where
the MDE is ~0.3pp and the effect is measurable — **not** as an item-level feature, where the
2.21–3.69pp floor guarantees a null.

HLTV's schedule is genuinely forward-looking but TLS-fingerprint blocked, and §12 already
found no dated series showing a Major moving the aggregate market — fetch fragility plus a
documented null means don't. SteamDB's update feed leads by minutes-to-hours, which is worth
nothing on a daily panel.

---

## 24. Ranked candidate features

Ranked by expected value = (plausibility that it carries information) × (whether it is
measurable against a 2.21–3.69pp item-level floor) ÷ (effort). **Anything already refuted is
excluded** — see §2 for that ledger; re-proposing them is the failure mode this table exists
to prevent.

Horizon column uses the §17 finding that only h=14/30 are executable.

| # | Feature | Family | Expected importance | Horizon | Data difficulty | Leakage risk |
|---:|---|---|---|---|---|---|
| **0** | **Lagged expensive-tier return → cheap-tier return** | cross-asset | **Highest — the only measured positive.** z=9.1, Granger incremental R² 9.0%, survives market-factor removal (R² 4.5%), stable 4/5 years | date × tier | **Trivial — two tier indices and a lag** | None. But **staleness not fully excluded** — validate by dropping high-`stale_run_days` days (§14) |
| 0b | **`crate_id` as a grouping/shrinkage variable** (not a column) | cross-asset | High for **recent** crates only — same-crate residual corr +0.084 vs +0.003, a 28× ratio; Gallery 0.345, Kilowatt 0.104, legacy ≈ 0 | 14, 30 | Trivial — 687/691 items already join | None |
| 1 | **`index_return_{k}d`** (hedonic, time dummies) | market-wide | **High** — the only place variance is demonstrated | date-level | Medium — must be built | **High if mis-built**: time dummies fit in-sample across the panel; must be re-fit per fold inside the purge |
| 2 | **`down_rate`** (per-date fraction negative) | market-wide | **High** — swings 32.7→76.9%; also the PT base rate | date-level | Trivial | None |
| 3 | **Within-date rank transform of all features** | all | **High (structural)** — removes the market factor from the feature side | 14, 30 | Trivial — one `groupby.rank` | None (same-date only) |
| 4 | **`stale_run_days`** | price | **High** — GLM θ in discrete form; ~~37–42% vs 0–1.8% tier split~~ **that split is on smoothed anchors, not the modelled series (12–27% at ≥$1); banner item 5** | 14, 30 | **Trivial, 13 years deep** — label-side use shipped 2026-08-08 | None |
| 5 | **`dispersion`** (cross-sectional sd of returns) | market-wide | High — the Oct-2025 signature; separates "market down" from "market re-sorting" | date-level | Trivial | None |
| 6 | **`d_bid − d_ask`** (order-flow imbalance) | liquidity | **High if measurable** — canonical short-horizon predictor | 14 | **Blocked: 25 days ⇒ zero extra folds** | None |
| 7 | **`spread_resid`** (spread minus tier median) | liquidity | High | 14, 30 | Blocked: 25 days | None |
| 8 | **`theta_ma1`** (Getmansky–Lo–Makarov) | liquidity | Medium-high — continuous form of #4 | 14, 30 | Low | None (trailing) |
| 9 | **`roll_spread`** (effective spread from price alone) | liquidity | Medium-high — **the only liquidity variable available for the 13 pre-multi-source years** | 14, 30 | Low | None |
| 10 | **`breadth`** (fraction above 30d MA) | market-wide | Medium-high | date-level | Trivial | None |
| 11 | **Cross-sectional reversal rank** | price | Medium — the documented digital-collectibles effect; momentum was tested, reversal was not | 14, 30 | Trivial | None |
| 12 | **`drop_pool_active`** | supply/metadata | Medium — cleanest supply shock in the record (+15–57% on discontinued cases) | 30 | Low — public | **Metadata-as-of-today**: status changed silently in Dec 2024, needs `valid_from` |
| 13 | **`is_tradeup_fuel` / `is_tradeup_output`** | supply/metadata | Medium — the Oct-2025 cross-section sorted on this | 30 | Low — public | Static; low |
| 14 | **`float_range_capped`** | metadata | Medium — capped finishes held through the crash; easy-cap bled | 30 | Low (ByMykel `float_min/max`) | Static |
| 15 | **LLM supply-shock flag from ISteamNews** | event | Medium at **date level**, null at item level | date-level | Medium — one-off over 1,752 docs | **UTC-boundary trap**: posts land 21:00–01:00 |
| 16 | **`sale_count_24h`** | volume | Medium — the counting-noise denominator | 14, 30 | Medium — kieranpoc backfill | Dead column must not be reused in place |
| 17 | **`turnover` = sales / listings** | liquidity | Medium — the *signed* version of supply depth | 14, 30 | Blocked on #16 | None |
| 18 | **`dwell_time` = listings / sales** | liquidity | Medium | 30 | Blocked on #16 | None |
| 19 | **`usd_cny` return** | market-wide | Medium — the two largest-coverage sources are CNY-denominated | date-level | ~~**Blocked: 7 days of FX history**~~ **Unblocked** — `price-archive/exchange-rates-history.parquet` is 13 years deep and published | None |
| 20 | **`n_ask_sources`** | integrity | Medium (as a *guard*, not a feature) | — | Trivial | Prevents a basis change reading as a return |
| 10b | **`stattrak_premium_z30`** (z-scored ST/normal ratio) | metadata | Medium-high — **a revealed-preference weapon-usage measure**, the only such signal available free. AK 2.15× vs P2000 1.10×, 14.7% below 1.0× | 14, 30 | **Trivial — names already in the archive, 13 years deep** | None. Use the z-score, never the level (level = shelved dollar proxy) |
| 10c | **`case_ev_ratio`** (contained-item EV ÷ case price) | cross-asset | Medium — disperses **0.72×–4.06×**; genuinely unmeasured, no free publisher | date × case | Low — ByMykel map is free/MIT | **128 usable days ⇒ zero extra folds.** Use the fixed-composition sub-EV, or reframe as a §14 lead-lag question |
| 14b | **`souvenir_premium_z30`** | metadata | Low-medium — 2.39× median but bimodal, 30.4% below normal | 30 | Trivial | None |
| 21 | **`distance_to_support` / `_resistance` / `high_low_range_30d`** | price | Low-medium | 14 | **Zero — already computed, silently dropped by a prefix mismatch** | None |
| 22 | **`steam_buff_ratio_z30`** | cross-market | Low-medium — the one survivor of this family; the *level* is dead, the *deviation* is not | 14 | Low | None |
| 23 | **`regime_id` / `days_since_regime_break`** | event | Low-medium | 14, 30 | Low | **Contemporaneous only** — forward version is unbuildable |
| 24 | **`bid_side_depth_proxy`** (bid / consensus ask) | liquidity | Low-medium | 30 | Blocked: 25 days | None |
| 25 | **`supply_change_7d`** on a repaired feed | supply | Low-medium — level is ~0pp; only velocity was ever plausible | 30 | Medium — atalantus backfill unblocks 2023–24 | None |
| 26 | **`spread_change_7d`** | liquidity | Low-medium | 14 | Blocked: 25 days | None |
| 27 | **Souvenir / sticker mint windows** | event | Low — real but confined to that sub-market | 30 | Low | **Genuinely deterministic** — the one leak-free forward event feature |
| 28 | **`tick_cluster_30d`** (round-cent fraction) | liquidity | Low, speculative — thin-book tell | 30 | Trivial | None |
| 29 | **`n_venues_quoting`** | cross-market | Low, speculative — delist/relist is a real supply event | 30 | Trivial | None |
| 30 | **`index_drawdown`** | market-wide | Low | date-level | Trivial | Contemporaneous only |
| 31 | **`log_sale_count` z-score** | volume | Low — measured +0.019/+0.034 (real, r²<0.15%) | 14, 30 | Blocked on #16 | None |
| 32 | **Per-item `item_first_seen` from a release table** | temporal | Low — fixes `item_age_days` being a mix of true and observation age | — | Low | Currently a date proxy; §18 L1 |

**Read the shape of this table, not just the rows.** The top entries are all either
**date-level**, **cross-tier**, or **structural transforms** — not new item-level columns.
Ranks 6–9, the liquidity family with the strongest theoretical support, are **blocked on 25
days of bid history that produces zero additional folds**. Ranks 12–14, the mechanical
supply-position features, only fire on unannounced rule changes. That distribution restates
the whole document: **there is no item-level feature left worth adding, and the directions
that remain open are aggregation, cross-tier structure, and measurement.**

Rank 0 is the exception that proves it — the one measured positive is a **relationship
between tiers**, not a property of an item.

## 25. Item attribute economics — what survives the name key

**The governing constraint:** the archive keys on `market_hash_name`, which encodes only
weapon + skin + wear bucket + StatTrak/Souvenir. Float, paint seed, applied stickers and
applied charms are **per-asset attributes that do not change the name**. Measurements below
are original, taken 2026-08-07 against the free BUFF dump (46,041 items / 34,326 priced
names) and ByMykel's static bundle.

| Attribute | Observable at name grain? |
|---|---|
| Wear bucket, StatTrak, Souvenir | **Yes — encoded in the name** |
| Float-range cap, crate/collection, trade-up position, case EV | **Yes — free static map (ByMykel, MIT)** |
| Float value within bucket, paint seed / pattern, applied stickers, applied charms | **No — per-listing, structurally invisible** |
| Doppler phase | **No on Steam — but the BUFF dump disaggregates it** |

### The finding that needs acting on: Doppler phases are corrupting labels today

**[DOCUMENTED, measured here.]** 29 base names collapse 181 distinct `paint_index` assets.
The BUFF dump you already fetch daily carries a `doppler` sub-object on **110
market_hash_names**, with ask *and* bid per phase:

| | |
|---|---:|
| Median max/min phase ratio **within one name** | **3.25×** |
| p90 / max | 6.08× / **23.5×** |
| Share with >2× internal dispersion | **87.3%** |
| **BUFF headline price == the cheapest phase** | **95.5%** |

Worst case: `★ StatTrak™ M9 Bayonet | Doppler (Minimal Wear)` spans **$1,261 → $29,685**
under one name. Because the headline tracks the cheapest phase, **the series steps whenever
which phase is cheapest changes — a level shift with no asset repricing.** This is §22's D9
quantified, and the free feed already carries the fix: drop the 110 names, or split them.

### StatTrak and Souvenir — and a free weapon-popularity proxy

**[DOCUMENTED, measured here; 4,686 paired names.]** Median ST/normal = **1.43×**, but
**14.7% trade below normal**. The driver is *weapon usage*, not rarity — a StatTrak counter
only accrues on a gun you actually shoot:

| Weapon | ST/normal | % below 1.0× |
|---|---:|---:|
| AK-47 | **2.15×** | 12% |
| USP-S | 2.06× | 10% |
| AWP | 1.90× | 11% |
| Butterfly / M9 / Karambit | **1.06–1.07×** | ~11% |
| **P2000** | 1.10× | **38%** |
| **CZ75-Auto** | 1.17× | **37%** |

Discounts concentrate in **Factory New (24% below)**, where the clean look competes with the
counter.

**This solves a problem §26 otherwise has no answer for.** No free machine-readable
weapon-usage source exists — Valve publishes nothing, and Leetify/csstats/scope all sit
behind Cloudflare. But the ST premium *is* a revealed-preference usage ranking, priced,
derivable from names already in the archive, 13 years deep, free. Test the **z-scored
deviation**, never the level (the level is a shelved dollar-scale proxy).

Souvenir: median **2.39×**, bimodal, **30.4% below normal** (the $1–10 band reads 44.2%
below). Two dated 2026-05-22 supply mechanics, both primary-sourced from `ISteamNews`: the
"Souvenir-O-Matic" (souvenir supply is no longer mint-window-constrained) and souvenirs
becoming trade-up inputs (a sink). Both are date-level shocks, name-grain observable.

### Float-range cap — free, static, and mechanically implicated

**[DOCUMENTED, computed from ByMykel `skins.json`.]** Of 2,106 skins with a float range,
**1,683 (79.9%) are truncated**; 51.5% cannot reach Battle-Scarred, 20.8% cannot reach
Well-Worn, 2.9% cannot reach Factory New. This is the mechanical variable behind §12's
observation that float-capped knife finishes held through the Oct-2025 crash while easy-cap
ones bled into 2026.

### Charms are a separate asset class, definitively

Release notes 2024-10-02: *"Use a Charm Detachment to detach the Charm from a weapon. The
Charm will return to your inventory and can be reused."* Unlike stickers, charms **survive
removal**, so they are tradeable in their own right — **78 `Charm | …` names** are already
priced in the BUFF dump and are already ordinary rows in your universe. Applied to a weapon
they become invisible; standalone they need no special handling.

### Level vs returns — the same shape as the refutation ledger

| Attribute | Price level | **Future returns** |
|---|---|---|
| Float within bucket / paint seed | Large / very large | **Not testable at name grain** |
| Doppler phase | 3.25× median internal | **Negative — it corrupts the label. Fix, don't feature** |
| StatTrak flag | 1.43× | No evidence. Only the *ratio deviation* is testable |
| Souvenir flag | 2.39×, 30% <1× | No evidence |
| Applied stickers / charms | Large / small | **Not observable** |
| Trade-up position | Real (Oct-2025) | **Only on unannounced rule changes** |
| Float-range cap | Small | Plausible, untested, conditional on a shock |
| Crate/collection | Small | **Real — as a grouping variable (§14), not a column** |
| Case EV | Definitional | **Testable — below** |

**Every row has a large level effect and an empty or conditional returns column.** Same shape
as §2, same reason: these are *level* attributes against a *percentage* target. §15's
disposition — item attributes into an `item_attr` dimension for the hedonic index, not the
return model — is correct and nothing here changes it.

**Do not pursue float, pattern, sticker or applied-charm data.** Not "expensive" —
structurally invisible at your key, and each one reintroduces exactly the composition
artifact that killed CSFloat `avg_price`. (`CS2BlueGem` does publish a keyed blue-gem pricing
API, but it is paid and therefore out of scope, and useless at this grain regardless.)

### Case EV — computable, and computed

**[DOCUMENTED, measured here.]** ByMykel `crates.json` gives **42 weapon cases, all 42 with
`contains` and `contains_rare`**, free and MIT. Drop odds (Mil-Spec 79.92%, Restricted
15.98%, Classified 3.20%, Covert 0.64%, Rare Special 0.26% — ~~each adjacent tier exactly
5:1~~ **the last ratio is 2.46:1, from these same numbers**)
are **[WIDELY REPORTED]**; I could not fetch a Valve-hosted page stating them, so treat them
as a fixed vector to verify in-client once, not as a sourced number.

Wear probabilities from `min_float`/`max_float` under a uniform draw, ST at 10%, skins tier
only:

| Case | Ask | Skin-only EV | **EV / price** |
|---|---:|---:|---:|
| Kilowatt | $0.15 | $0.609 | **4.06×** |
| Revolution | $0.24 | $0.557 | 2.32× |
| Recoil | $0.31 | $0.623 | 2.01× |
| Fracture | $0.51 | $0.609 | 1.19× |
| Dreams & Nightmares | $1.20 | $0.862 | **0.72×** |

Zero price lookups failed. **The ratio disperses 0.72×–4.06×** — a real cross-sectional
quantity, fully name-grain observable, recomputable daily.

**No free publisher of case EV exists, and no study measures whether case price tracks or
leads its own EV.** This is genuinely unmeasured.

**The constraint that decides it:** all 42 cases match the archive (median 3,583 days) and
all 6,138 contained names match — but **78.7% of contained items have their first priced day
on exactly 2026-03-22**. Usable depth is **128 days**, which at `CV_STEP_DAYS = 150` yields
**zero additional folds** — the same wall blocking the liquidity family (§16). Two partial
escapes: a **fixed-composition sub-EV** over the 1,057 contained names with >1,000 days
(biased by non-random omission, but the bias is a level term that differences out when taken
as a ratio to its own trailing mean); and reframing to **"does a case lead its own
contents?"** — a lead-lag question belonging in the §14 date-grain frame where z=9.1 effects
were resolvable, not the 2.21pp item frame. §14 found cases lead *the market* on volume alone
with no case-specific information; whether a case leads *its own contents* is the sharper
untested version, and it runs on 128 days.

## 26. Demand proxies — five of seven do not exist

| Driver | Free machine-readable proxy? |
|---|---|
| **Pro player skin usage** | **No.** No site tracks pro loadouts as data. HLTV publishes match stats, not inventories, and is Cloudflare/TLS-blocked. Only route is OCR over VODs. Not buildable |
| **Esports viewership** | **Partial and already refuted.** Dates are calendar-known and measured null; viewership numbers are paywalled estimates |
| **Streamer exposure** | **Category level only.** Twitch Helix is free with OAuth but returns viewers for "Counter-Strike 2" as a category. **Skin-level attribution is impossible** without frame-level CV. Also §13's canonical prior-art failure mode |
| **Weapon popularity** | **No published source** — but see §25: **the StatTrak premium is a revealed-preference usage measure already latent in your price data** |
| **New-player inflow** | **Weak, already refuted** — §12's null |
| **Collector vs speculator demand** | **Not separable.** Needs per-account holding periods. Closest observable is dwell time, blocked on the dead `volume` column |
| **Sticker crafting demand** | **No** — requires per-listing sticker data |

Against a 2.21–3.69pp item-level MDE, none of these would resolve even if they existed.

---

## 27. Final recommendations

### F. The ten biggest risks that could make this model misleading

Ordered by how much of the published record each one contaminates.

| # | Risk | Why it is live |
|---|---|---|
| **1** | **The headline metric measures the market, not the model** | DA's realised base rate swings 32.7%→76.9% between dates, and a constant call beat the model on **every stored date**. Until PT replaces it, every accuracy number is uninterpretable — and this is the risk that silently generated the other nine |
| **2** | **A bid is being voted into the consensus price** | `aggregator_buff163_buy` at 0.550× Steam votes in a median with asks at 0.700–0.802×, and outlier rejection makes inclusion flicker day to day. **Every label and every A/B since 2026-07-11 sits downstream** |
| **3** | **The published backtest number is unpurged** | `walkforward_backtest.py --purge` is default OFF, and ten more harnesses have neither purge nor fold clustering. Measured inflation: +12.1pp → +6.1pp at h=30 |
| **4** | **Intervals were too narrow until 2026-08-07** | `paired_mde` clustered on `forecast_date`, not `fold_id`. A seed-only placebo returned CI [−0.3099, −0.0140] on pure noise. Three published refutations have never been re-derived |
| **5** | **The one surviving positive is selected on full-sample data** | `_filter_by_median_price` uses a 2013→2026 median. "Items ≥$1 over the whole window" was unnameable in 2019 — and the +3.50pp appears **only at h=30**, the look-ahead signature |
| **6** | **h=30 rests on one backdated date** | 5,461 usable rows, all from 2025-12-01. It carries both the surviving positive and the largest unpurged inflation |
| **7** | **Doppler names are not single assets** | 110 names, median 3.25× internal dispersion, headline tracks the cheapest phase 95.5% of the time. The series steps on composition, not price |
| **8** | **No arrival timestamp exists anywhere in the archive** | Six backfill writers, revisions, and a synthetic fee constant — and none of it is auditable retroactively. Point-in-time reconstruction is impossible |
| **9** | **Green CI is not evidence of collection** | Three jobs reported success storing zero rows; `volume` died as *zeros* and defeated every `notna()` guard for a year |
| **10** | **Implying tradeability** | 49–53% on a 3-class problem against a +16.1% Steam round trip, with a 3% call inside the spread for five of six tiers, and nothing under 8 days executable at all |

### I. The ten most important research findings

| # | Finding |
|---|---|
| **1** | **Item-level idiosyncratic signal is measured absent** — demeaning drops accuracy below a constant call at all four horizons. Not a defect: Gu/Kelly/Xiu's best is 0.4% monthly R² on 30k equities with 60 years |
| **2** | **A flagship item trades 96×/day; the $500+ tier medians 4/day.** Per-item daily returns are counting noise before any feature touches them — and the problem is *worst in the served cohort* |
| **3** | **Expensive tiers lead cheap tiers by one day** — z=9.1, Granger R² 9.0%, survives market-factor removal. The first positive structure in the archive, and the folk direction is backwards |
| **4** | **"Always-down beats the model" is the Pesaran–Timmermann null**, a named pathology with a published test and a serial-correlation-robust variant |
| **5** | **Spread runs 35.5% sub-$1 to 5.2% at $1000+** — expensive items are the liquid ones, inverting retail intuition and making any pooled metric uninterpretable |
| **6** | **Cash venues agree to within 2%** (CSFloat/BUFF 0.976–1.008), which mechanistically explains the CSFloat basis null. Steam is the outlier at ~1.43×, and structurally unarbitrageable |
| **7** | **Every large market move was an unannounced Valve supply-rule change** — but 4 of 6 were in the free machine-readable news feed *the same day*. Forward event features are unbuildable; contemporaneous detection is not |
| **8** | **Nobody has published a CS2 market index with a stated methodology**, and the collectibles literature (hedonic + repeat-sales) supplies a ready-made estimator. Genuinely novel work |
| **9** | **No published skin paper uses purged CV, a multiple-testing correction, or a constant-sign benchmark.** The one peer-reviewed CS2 ML paper reports Sharpe 10–12. Your design is ahead of the literature |
| **10** | **Cases do not lead skins, volume does not predict lower prices, player count does not lead price, and the 2016 gambling crackdown did not move prices** — four widely-repeated beliefs, all refuted with data |

---

## If I were building this myself

In this order. Weeks 1–2 need no retrain at all, which is the point — the highest-value work
is free of the Monday retrain budget.

**1. Run one query: is `aggregator_buff163_buy` in the consensus median?** Everything below is
worthless if the labels are fabricated. Half a day. If it is, exclude it from voting and
re-run the affected A/Bs before anything else.

**2. Fix the metric.** Serial-correlation-robust Pesaran–Timmermann as the headline, per date
with a t-stat over dates. Report DA only as a triple with the constant-call baseline and the
realised down-rate. This reframes every number the project has produced and costs one scoring
module.

**3. Score by tier, and condition on `|predicted move| > round-trip cost`.** Split `price_tier`
above $100 — it currently merges the 10.8% and 5.2% spread cohorts. Expect `n_actionable` to
be near zero at short horizons; that is the result, and it belongs at the top of any document
describing what this product is.

**4. Drop the 110 Doppler names.** One filter. They are not single assets and they are moving
your labels on composition.

**5. Purge and embargo everywhere — `H + 13` days.** Turn `--purge` on by default, accept the
discontinuity in the published series, and fix the ten unpurged harnesses. Add `ingested_at`
to the archive schema while you are in there; it is one column and it cannot be added
backwards.

**6. Drop or downweight frozen-price runs from the label set.** 37–42% of tier-0 outcomes are
bit-identical carry-forwards. This is the only action that *improves the measurement
instrument itself* — it lowers the MDE for every future experiment.

Now you have a system that reports honestly. Everything above is roughly two weeks and no
retrain.

**7. Ship `TRAIN_MIN_MEDIAN_PRICE` at a ≥1.0M budget.** The one surviving positive, and more
importantly it removes item-draw variance entirely rather than shrinking it. Re-derive it
first with a per-fold price filter — the current version selects on a full-sample median.

**8. Build the hedonic index.** Log price ~ attributes + time dummies, refit per fold inside
the purge boundary. This is the composition-controlled market factor you have never had, it
is a user-visible product artifact, and no one has published one for CS2.

**9. Model the date-level factor.** 4,735 days, MDE ~0.3pp, and it is silently 80–100% of the
answer your model already gives. Feed it the expensive→cheap lead (§14), `down_rate`,
`breadth`, `dispersion`, and the LLM supply-shock flag from the news text you already cache.

**10. Try `lambdarank` within-date, scored by rank-IC.** An objective swap. It is the version
of the market-demeaning test that is not confounded by the loss function, and IC-IR > 0.05 is
the honest bar.

**11. Then, and only then, the cheap item-level candidates:** `stale_run_days`, `roll_spread`,
the z-scored StatTrak premium, `float_range_capped`, cross-sectional reversal. Each is free
and none needs a new source.

**What I would not do at any point:** add another cosmetic metadata bundle, chase float or
pattern data, rebuild sentiment, try a transformer, or pay for anything. §2, §9, §11, §25 and
§26 all converge, and the convergence is the finding.

**And the framing I would hold onto throughout:** this is an information product, not a
trading system. At a +16.1% Steam round trip against a model at 49–53%, no realistic accuracy
gain makes it tradeable — but a well-calibrated 55% probability with an honest band, an
index nobody else publishes, and a system that says "no call" when it should, is a real
product. The failure mode to avoid is drifting into implying otherwise.

---

## Appendix: claims that need re-deriving before being cited

Because `paired_mde` clustered on `forecast_date` rather than `fold_id` until 2026-08-07,
producing intervals that were too narrow:

- The CSFloat basis refutation
- The ByMykel metadata refutation
- The training-breadth headline

These are probably still correct in sign — they were null results, and the bug inflated
significance rather than manufacturing it — but the *intervals* are wrong and none have been
re-derived. Any future citation should say so.
