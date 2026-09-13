# Supply-depth velocity is killed at the venue-transfer gate — do not run the GBM

**Date:** 2026-09-13

Closes ML tracker #4, the only item on that list neither refuted nor shipped. It was
blocked on "consecutive daily snapshots after the billing pause". The chain (Aggregator +
Forecast) is green Sep 08→12, with supply landing Sep 06→11 (6-day run, 5 velocity
pairs/item). Backtest is red on an unrelated 20%-unresolved resolution gate. **No September
wait was needed:** the August panel already holds 19 consecutive days (08-07→08-25), which
is the transfer test, and it fails with the wrong sign.

## Panel health (origin/main, read 2026-09-12)

- Supply Aug: 22 days, longest run 19 (08-07→08-25, no gaps). Supply Sep: 7 days
  (09-02 isolated, then 09-06→09-11 unbroken).
- Trailing 6-day run: 154k max-collapsed pairs over 31,354 items, 48% nonzero churn.
- Venues: market_csgo / skinport / lis_skins / waxpeer. **bitskins dead since 08-24
  (0 rows) — and frozen before that: 0.0% nonzero churn 08-07→08-23 at a flat
  10,920 rows/day, so it contributes no velocity signal on any day.** Skinport intermittent (0 rows 08-23, 08-27, 09-02; 22% nonzero vs
  60–67% on the other three — stale, not dead).

## Transfer spike — live max-churn vs forward |return|, August panel

Max-collapsed `Δlog(1+listings)` joined to median-across-feeds price, 571k item-days /
31.5k items. BUFF paper (`research/2026-08-17-supply-churn-volatility-signal.md`) had
`|Δlistings|` corr **+0.11** with forward |7d return|, ramping 9.6%→14.3% across churn
octiles. Live:

| h | n | \|churn\| corr | signed | level | trail vol | nz-share | nz corr | R² trail → +churn |
|---|---|---|---|---|---|---|---|---|
| 3 | 420,723 | **−0.016** | +0.005 | −0.009 | +0.137 | 49% | −0.041 | 0.01871 → 0.01884 (+0.0001) |
| 7 | 300,520 | **−0.023** | +0.003 | +0.003 | +0.161 | 51% | −0.067 | 0.02581 → 0.02613 (+0.0003) |

Churn octiles run **backwards** at both horizons (h=3 forward 5.4%→3.0%,
h=7 9.5%→5.1% smallest→largest churn) while trailing vol is flat-to-falling —
the mirror image of the BUFF ramp. Level is null, as before.

## Caveats (stated so this stays killed honestly)

Spike-grade, not the paired harness: median-price proxy (no voted consensus, no
universe filter, no H+13 embargo), 14d trailing-vol proxy vs the paper's 30d,
August window only. None of these explain a sign inversion over 300–420k rows with
8/8 octiles monotone-wrong-way at both horizons — and trailing vol replicates
(+0.14/+0.16), so the test has power.

## Verdict

**KILLED at the transfer gate. Do not run the band-width GBM** (no
`SUPPLY_CHURN_FEATURES` widening, no `ob_churn` promotion, no
`shrink_k_vol_rank_ab`-style paired replay vs the flat-K=320 climatology null).
The BUFF venue's churn signal does not survive the move to the max-collapsed
free-venue series — max-collapse attenuates moves by construction, and one of the
four surviving feeds reads stale. This is the 6th band-width death (5 GBMs +
log1p-listings); the width variable is not the lever.

Reopen only if: (a) a per-venue (not max-collapsed) churn replicates +0.1 on the
same August panel first, with the stale-feed handling stated up front — independently
reproduced 2026-09-13 on the full 22-day August panel, this fails everywhere: best
pooled per-venue read is +0.02 (skinport, h=3) and every venue is negative
conditional on a move (−0.02 to −0.10) — then the
paired width-at-matched-80% harness with shuffled placebo and supply-shock strata
may run. September accumulation maturing changes nothing about this gate.
