# Paper-trading P&L harness

Added `backtest/papertrade.py` and `scripts/papertrade_report.py`: an offline
execution harness that replays every served forecast as a long round trip and
reports what it would have earned **net of real venue friction**. It is an
evaluation instrument, not a product surface — no new tables, no route, writes
nothing.

## Why

Every error metric this repo reports is blind to friction: a forecast can be
accurate and imply no trade. `backtest/actionable.py` conditions directional
accuracy on `|r̂| > round trip + spread`; this harness takes the next step and
prices the trade as P&L, so the question becomes "would this have made cashable
money," which is the only question a would-be trader has.

Consistent with the range-forecaster stance (`AGENTS.md`): the harness is built
to be *able* to conclude "no cash edge," not to manufacture one.

## What it encodes (all from research §11)

- **P&L on the resolver basis.** Buy `base_price`, sell `actual_price` — both
  frozen legs from one estimator. The trade *decision* uses the served basis
  (`current_price` → `base_price` fallback) and the served band; the two-bases
  split is the 2026-08-11 invariant.
- **Friction charged every trade:** venue fee + the tier bid-ask spread
  (`friction.py`), crossed once. CSFloat +2%; spread 35.5% sub-$1 → 5.2% at
  $1000+.
- **Steam Wallet is walled.** Steam is in `WALLED_VENUES`; every payload carries
  `pt_cashable`, and the report prints Steam on a separate WALLET-ONLY line. A
  Steam return is never a cash P&L.
- **Executable horizons only** — `{14, 30}` (7-day market lock + trade
  protection). The unsourced "8-day cooldown" is deliberately not encoded.
- **No cross-venue arbitrage.** The ~1.43× Steam premium is a structurally
  unarbitrageable level shift; single-venue round trips only.
- **Frozen anchors bucketed apart.** `base_stale_run_days` 0 / >0 / NULL →
  fresh / stale / unknown; a frozen anchor's return is the MA artifact, and NULL
  (pre-2026-08-08) is never read as 0.

## Strategies compared

Against a buy-and-hold-everything baseline: `exceedance` (served quote below the
band floor) and `q50_clears` (predicted median move clears friction, the
actionable bar). Profitability is a **date-clustered** bootstrap CI on the mean
net return (returns share a market factor within a date, so an iid row bootstrap
under-disperses — same reasoning as `paired_mde`); `pt_profitable` is
`ci_lower > 0`.

The cross-sectional long-short read-out stays in `backtest/longshort.py` and is
given no P&L here: the short leg is not executable on this market.

## Read-outs

`venv/bin/python -m scripts.papertrade_report [--min-price 1.0] [--by-tier]`.
Reads whatever `DATABASE_URL` resolves to (read-only) — the local DB is a
synthetic fixture, so real figures need the prod panel. Results by tier, not
pooled, since spread and liquidity both improve with price.

Tests: `tests/test_papertrade.py` (13). Expected production result: no cash
strategy clears friction.
