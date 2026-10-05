# FX ratio monitor on the Aggregator

**Date:** 2026-10-05
**Script:** `backend/scripts/check_fx_ratios.py` (new), step "FX ratio monitor" in
`aggregator-update.yml`
**Status:** shipped, warn-only. No modelling or serving change.

Deep review §1d (`research/2026-08-19-deep-model-review.md`): BUFF163 and youpin price in CNY,
and CSGOTrader converts them to USD upstream at an unknown rate on an unknown date. A wrong or
stale rate is a market-wide multiplicative error on two voted sources, and nothing local would
see it. The collected exchange rates are never applied, so they can't detect it either. The
review's fix was "a daily assertion that `median(youpin/csfloat)` and `median(buff163/csfloat)`
stay in band". This is that check (§12 item 11).

## What it does

After the day's append, it reads the trailing three weeks through `prices_relation` and
`archive_universe_sql_filter`. For each venue it takes the median over items ≥ $1 of
`venue / csfloat` on the same day, and compares the newest day with the median of the prior 14
days (each day needs ≥ 200 paired items).

- `cny_breach`: buff163 or youpin moved more than 3%.
- `fx_signature`: both CNY venues moved past 3% in the same direction while skinport and
  csmoney (USD) stayed within it. A csfloat move shifts every ratio, controls included, so it
  reads as a breach, not a signature.

A breach prints a `::warning::` annotation plus a JSON summary line. The exit code stays 0, and
the step is `continue-on-error`: Price Forecast chains off the Aggregator's success, and a real
market move must not stop the day's forecasts.

## Where the threshold comes from

Local archive, 2026-07-11 → 09-08 (49 days):

| ratio to csfloat | mean | std | median daily move | largest daily move |
|---|---|---|---|---|
| buff163 | 1.017 | 0.007 | 0.15% | 1.3% |
| youpin | 1.027 | 0.007 | 0.24% | 1.3% |
| skinport | 1.126 | 0.018 | 0.16% | 3.0% |
| csmoney | 1.042 | 0.012 | 0.16% | 3.3% |

A replay over every archived day from 2026-03-22 raises **zero** alarms. The largest CNY move
against the 14-day baseline is +2.9% (buff163) / +2.7% (youpin), on 2026-07-11, the first day of
the `aggregator_*` sources. That is a real composition break (it is on the break calendar), so a
3% line sitting just above it is about right. A wrong CNY rate of a few percent would clear it.

The USD venues are controls only and are never alarmed on: skinport alone moved −4.1% on
09-08 with no CNY move.

## Reproduce

From `backend/`:

```
venv/bin/python -m scripts.check_fx_ratios --archive-dir ../price-archive
```

The replay calls `daily_ratios(Path("../price-archive"), 400)` and evaluates each day
cumulatively with `evaluate(..., 14, 0.03)`.
