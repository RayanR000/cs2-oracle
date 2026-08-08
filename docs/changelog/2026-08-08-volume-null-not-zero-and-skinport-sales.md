# The `volume` column was never collected, and the absence was stored as `0`

**Date:** 2026-08-08
**Context:** `docs/references/data-sources.md`, `docs/changelog/2026-08-06-volume-features-shelved.md`
**Related:** `docs/changelog/2026-08-06-supply-depth-collector.md` — the stock to this flow

## Read this first: no lift is claimed

Trade volume as a *predictor* is refuted at **|r| < 0.002 across 4.47M rows** and nothing
here reopens it. The C4 correction in the 2026-08-07 review re-measured the pooled
within-item sign as **+0.019 (7d) / +0.034 (30d)** — 10–40× the quoted figure, and still
r² < 0.15%, i.e. economically trivial. The conclusion survives; only its stated reasoning
was wrong.

What this change buys is a **counting-noise denominator**: knowing which item-days are
statistically empty. That matters because the thin tier *is* the served cohort — the
archive's 2025 distribution medians **69 sales/item-day overall, 20 at $50–500 and 4 at
$500+**, with 60% of $500+ item-days at 1–5 sales. No model was retrained, no accuracy
number moved, and `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` is unchanged.

## The finding

**No aggregator feed carries a volume field at all.** Probed live 2026-08-08:

| Feed | Fields returned |
|---|---|
| `prices.csgotrader.app/latest/steam.json` | `last_24h`, `last_7d`, `last_30d`, `last_90d` — **prices** |
| `.../buff163.json` | `starting_at`, `highest_order` (+ `doppler`) |
| `.../csgotrader.json` | `price` (+ `doppler`) |

`collectors/csgotrader_aggregator.py` contains no reference to `volume` because there has
never been anything to parse. **This is an absence, not a regression** — the daily
collector has never collected volume, and the "identically zero since 2026-04-16" figure
is just the day after the last one-shot backfill ran out.

Every non-zero volume in the archive came from a backfill, all three expired:

| Source | Volume span | Rows |
|---|---|---:|
| pre-2026 (`source IS NULL`) | → 2025-12-31 | 9,429,275 |
| `aggregator_sync` | 2026-01-01 → **2026-03-29** | 404,563 |
| buff163 / youpin / csfloat | 2026-03-22 → **2026-04-15** | 1,967,619 |

**The last window is not the same quantity as the first.** `merge_hf_dataset.py:99` maps
`ask_volume AS volume` and line 122 sums it — that is a **listing count**, not a trade
count. So the column already holds two different measurements under one name, before the
zeros. Anything modelling across 2026-03-22 should treat it as a break, not a series.

## Why the `0` mattered

`pipeline.py` wrote `volume if volume is not None else 0`, and four raw-source writers
hardcoded `0`. A real zero **never occurs** — a day with no sale produces an *absent row*,
and the observed minimum is 1 — so every zero in the live era was fabricated.

It defeated the guards built to catch exactly this. From `models/forecaster.py`:
`has_volume` tests `notna()` so it stayed True, and `volume_missing` reported 0, meaning
"present". That is the documented reason the eleven volume features are shelved. The
column did not merely lack data; it asserted data it did not have.

## What landed

| Path | Change |
|---|---|
| `backend/collectors/pipeline.py` | `VOLUME_NOT_OBSERVED = None`; the `PriceHistory` path and all five CSV writers emit an empty field |
| `backend/scripts/append_to_parquet.py` | `_sum_observed` (`min_count=1`) replaces `("volume", "sum")`; `Int64` cast |
| `backend/collectors/sales_volume.py` | **new** — Skinport `/v1/sales/history` → `volume-YYYY-MM.parquet` |
| `backend/scripts/run_sales_volume.py` | **new** — entry point (`--dry-run`, `--date`, `--archive-dir`) |
| `backend/tests/test_sales_volume.py` | **new** — 12 tests |
| `backend/tests/test_append_to_parquet.py` | 3 tests for the NULL path |
| `backend/scripts/run_task.py` | `volume_rows` registered in `ROW_COUNT_FIELDS` |
| `.github/workflows/aggregator-update.yml` | `Collect sales volume`, between supply depth and the publish |

**The `min_count=1` half is not incidental.** A bare `.sum()` returns `0` for an all-NaN
group, so fixing the collector alone would have converted every NULL straight back to a
zero at aggregation. The test that caught it (`test_missing_volume_lands_as_null`) failed
with `got 0.0` against a collector that was already emitting `None`.

## Decisions taken, and their alternatives

**Fix forward only.** Stored rows keep their fabricated zeros. Repairing them means
rewriting ~7M rows across the 2026 files while not touching the three genuinely-populated
series, and the zeros age out of any training window on their own. Declined on risk, not
on principle.

**A separate `volume-YYYY-MM.parquet`, not `prices.volume`.** The dead column already
carries two meanings (Steam sale count, HF listing count) plus the zeros; a third would
make it unmodellable. §24 of the research review is explicit that the dead column must not
be reused in place. The cost is that nothing reading `prices.volume` picks this up
automatically — including `backtest/price_resolution.py:249`, which still reads it and
should either be pointed at the new file or stop reading it.

## Schema

```
volume-YYYY-MM.parquet
  item_slug VARCHAR | day DATE | source VARCHAR ('skinport_sales')
  sales_24h/7d/30d/90d BIGINT | median_30d DOUBLE | collected_at TIMESTAMPTZ
```

Dedup on `(item_slug, day, source)`, `keep="last"`, so re-running a day is idempotent.

**The windows are trailing and cumulative.** `sales_7d` is not `7 × sales_24h`, and
differencing across days does not give a daily sale count. `day` labels the observation —
the endpoint has no as-of parameter, exactly as with supply depth. And this is **one cash
venue**, not the market.

**A null window means zero here, unlike in `prices.volume`.** Skinport reports a null
window for an item it has never sold; that is a real "no sales", not a missing
observation. The two cases are deliberately encoded differently and that distinction is
the point of the whole change.

## Not verified

**The response shape was not observed.** This machine egresses through `104.28.167.5`, a
Cloudflare IP, and Skinport's WAF 403s AS13335 on *every* endpoint — so the parser is
written against the documented contract, not against bytes. CI runners are unaffected
(`/v1/items` returned 24,913 items from a runner the same day).

The parser therefore **fails the run** rather than returning short: a non-list payload,
zero parsed rows, or a parse rate under 50% all raise, and a renamed window key yields
NULL rather than `0`. Four of the twelve tests exist only for this. **On the first CI run,
check `raw_payload_entries` against `distinct_items` — both should read ~36,000.**

## Open

- `backtest/price_resolution.py:249` still reads the dead `prices.volume`.
- Supply depth has **one snapshot day** (2026-08-07); its CI step first ran 2026-08-08, so
  day-over-day accumulation is still unproven for both collectors.
- The MDE gate is untouched and unsatisfied. Run `scripts/compute_mde.py` before any A/B
  that uses these counts, and use a permutation design.
