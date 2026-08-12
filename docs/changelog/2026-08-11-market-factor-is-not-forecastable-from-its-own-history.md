# N2 first read: the market factor is not forecastable from its own history

**Read, not a build.** `models/market_factor.py` already carried everything this needed —
`build_market_index`, the `market_factor_for_horizon` label, and the pre-registered
`forecast_market_factor` estimator with two diagnostics explicitly excluded from the decision.
All of it shipped for the 2026-08-06 market-relative-labels experiment, whose pre-registered
kill switch fired at stage 1, so **stage 2 was never interpreted**. This interprets it.

Archive only, no DB session, nothing written. Index built over the ≥ $1 cohort from 2024-01-01:
**947 days, 943 valid, median 860 items/day**, daily log-return sd 0.01731.

## The result: null, and on the wrong side rather than merely noisy

Scored on non-overlapping windows, with the first 180 valid days dropped as estimator warm-up.

| h | n (non-ovl) | corr | MAE drift | MAE flat-zero | DA% | realised down-rate |
|---|---|---|---|---|---|---|
| 3 | 250 | −0.120 | 1.472 | **1.415** | 40.8 | 40.8% |
| 7 | 108 | −0.152 | 2.641 | **2.495** | 38.0 | 40.7% |
| 14 | 54 | −0.162 | 3.897 | **3.626** | 48.1 | 48.1% |
| 30 | 25 | −0.093 | 5.037 | **4.575** | 40.0 | 48.0% |

The estimator loses to a **flat-zero forecast** on MAE at all four horizons, correlates
**negatively** with the outcome at all four, and its directional accuracy is at or below the
realised down-rate at all four.

**The mechanism, not just the number.** The trailing-180d drift is negative on **75–78%** of
dates, while the realised factor is **up-majority** (median move +0.168 / +0.347 / +0.243 /
+0.618%). The estimator is systematically on the wrong side of a mild upward drift, and its
magnitudes — median −0.108 to −0.873% — are an order of magnitude smaller than the realised
sd of 2.776 / 4.079 / 5.329 / 6.177%. Trailing drift is an anti-signal here, not a weak one.

## What this closes, and what it does not

**Closes:** the "forecast the market factor from its own history" leg of N2. The next-steps doc
anticipated exactly this outcome and said it was worth knowing — *"it would mean the constant
call's apparent strength is unreachable in principle, which closes the question rather than
leaving it open."* That is what happened.

**Does not close:** forecasting the factor from **date-level exogenous** data — FX, the event
calendar, player counts. That ingest exists (`2026-08-06-date-level-exogenous-ingest.md`) and
none of it was used here. This read refutes one estimator family, not the layer. Note the prior
warning attached to that ingest: the event calendar is a clock and goes null once stripped.

**Does not refute the decomposition itself.** Splitting the served return into a market term
and a relative term is still coherent; this says the market term has no history-only forecast
worth serving, so the decomposition currently buys nothing over a zero market term.

## The one thing worth a follow-up, and why it is not a result

The correlation is negative at **4 of 4** horizons (−0.093 to −0.162), which is a mean-reversion
signature: index drift up predicts forward down, and vice versa. That is a **post-hoc
observation on a read designed to test the opposite sign**, so it is recorded and not acted on.
Pursuing it means a fresh pre-registration with its own bar — the same discipline that made
today's C1 read decidable.

Two caveats bound it regardless: the non-overlapping sample is 250 / 108 / 54 / 25 windows, so
30d cannot resolve much; and this window contains both the Oct 2025 crash and the 2026-03-22
consensus break, either of which can manufacture an index-wide move that no estimator should
be credited or blamed for.

## Reproduce

The read script is `n2_market_factor_read.py`, kept out of the repo deliberately — it is a
one-shot analysis over library code that is already tested (`tests/test_market_factor.py`),
not a harness anything should depend on. Everything it calls is production code.
