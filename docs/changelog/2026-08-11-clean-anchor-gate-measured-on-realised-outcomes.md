# The clean-anchor gate, measured on outcomes that actually resolved

The gate shipped 2026-08-11 (`ab08a8b`) on the strength of **replay** rank IC +0.13–0.17, and
its own changelog states it does not touch the backtest — so nothing had checked it against
realised outcomes. This does, on the frozen `forecast_outcomes` already scored.

**The stored `anchor_clean` column was not used.** It is NULL on all 310,304 existing rows and
unbackfillable from the table. The mask is recomputed from the archive by
`replay_serving._tied_mask` — the single definition every published tied/deviating number came
from. This is why the two-week wait predicted when the gate shipped does not apply: the mask is
a function of archive prices at the forecast date, so it reconstructs retroactively.

Cohort: `lgbm-v3*` (`served_identity`), `base_price >= $1`, scored rows only — 14,586 rows over
11 forecast dates, of which 3,967 are the 2025-12-01 backdated batch and are reported separately.
Read-only; `--rescore` was never invoked.

## 1. The ordering claim: directionally confirmed, roughly a quarter of the size

Realised rank IC per forecast date (Spearman of `r_hat` against `r_act`), excluding the
backdated batch:

| h | cohort | dates | rows | mean IC | dates > 0 |
|---|---|---|---|---|---|
| 3 | **tied** | 6 | 1,769 | **+0.0368** | 4/6 |
| 3 | deviating | 6 | 3,600 | +0.0206 | 3/6 |
| 7 | **tied** | 5 | 1,001 | **+0.0364** | 4/5 |
| 7 | deviating | 5 | 2,565 | +0.0050 | 3/5 |
| 14 | tied | 2 | 134 | −0.0032 | 1/2 |
| 14 | deviating | 3 | 1,087 | −0.0667 | 0/3 |
| 30 | — | 0 | 0 | no date clears 20 rows | — |

The tied cohort orders better than the deviating one at every readable horizon, which is the
gate's premise. **Two things temper it.** The magnitude is **+0.037 against the replay's +0.13
to +0.17** — a factor of four — and the sign consistency is 4/6 and 4/5, not the 4/4 and 8–10/10
the replay reported. At h=14 the two cohorts do not even share a date set (2 against 3), so that
row is not a paired comparison and is printed only because dropping it silently would be worse.

## 2. The direction claim: no improvement

DA against the **runnable** always-down baseline (`realised_down_rate`; `constant_call` is
hindsight-selected and is not differenced), excluding the backdated batch:

| h | cohort | n | DA% | down-rate% | DA − down |
|---|---|---|---|---|---|
| 3 | tied | 1,773 | 32.2 | 43.5 | **−11.3** |
| 3 | deviating | 3,610 | 37.5 | 51.7 | **−14.2** |
| 7 | tied | 1,001 | 37.0 | 53.4 | **−16.4** |
| 7 | deviating | 2,565 | 44.6 | 61.6 | **−17.0** |
| 14 | tied | 147 | 44.9 | 74.8 | **−29.9** |
| 14 | deviating | 1,087 | 48.9 | 73.8 | **−24.9** |

⚠️ The **`DA − down` column is not a model property** — 56–89% of it is the realised direction of
the 5–7 dates in this panel, and the within-date term is ±1.2pp. See the banner in §3. The
tied-vs-deviating *contrast* is the readable part of this table; its levels are not, and the
decomposition was **not** run per cohort, so how much composition survives the differencing is
unmeasured.

On excess over the runnable baseline the tied cohort is better at 2 of 3 readable horizons and
worse at 1, by 0.6–5.0pp. **The gate improves ordering slightly and direction not at all** —
which is coherent, since it was justified on rank IC and the surface it controls is a ranked
list. Raw DA is *lower* on the tied cohort (32.2 vs 37.5 at h=3) purely because that cohort has
a lower down-rate; differencing the raw column across cohorts would have read that base-rate gap
as a regression.

## 3. The number that outranks both, and is not explained here

> ## ✅ CLOSED the same day — it is composition, and the hypothesis below was right.
> `changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`. Decomposing
> `excess` per (horizon, date) into the call mix costed against a typical market (`side_ref`),
> this window's realised direction (`composition`) and within-date information (`assoc`) gives
> **56% / 78% / 89% composition** at 3/7/14d against `assoc` of **−1.19 / +0.99 / −1.08pp**. On the
> CV label basis the same live item-dates give `assoc` **+2.70 / +0.17 / +0.42**, against the
> published CV **+3.5 / +0.1 / −1.3** — the two measurements agree. Per-date sd of `excess` is
> 14.96–24.39pp, so the live draw is z = −1.0 to −1.6 and **the whole gap is one to two SE.**
> Two corrections to the sentence below: the model is **above** the baseline on the dates where
> the market rose (2026-08-06 +4.1, 08-01 +19.7, 08-02 +6.4, and the backdated batch at
> +1.80/+3.99/+5.00/+5.96), so "every cohort at every horizon" holds of the pooled cells and not
> of the panel; and h=30 has **no live forecast date at all**. **Do not quote `excess` on fewer
> than ~50 dates.** One real defect fell out and is **diagnosed**: 2026-07-19 called `flat` on 64.0%
> of items at h=3 against a **23.1%** realised flat rate in this cohort — a low-hit-rate call at
> 18.3%, not a near-certain miss — and it carries essentially the entire pooled `side_ref` term on
> its own. Cause: `f71ffb4` shipped a global ±0.5% dead band in `predict()` **42 minutes before that
> run**, with no directional classifier in existence yet (`direction_models` was empty until
> `a332c2b`, 07-24). The band is still reachable as `predict()`'s no-classifier fallback. **And the
> arm labels are the reverse of what this repo recorded: 2026-07-19 is the production daily run;
> 2026-07-18's `-global-only` rows are an ablation arm that overwrote that day's production
> forecast** — so 07-18's membership in this panel is itself in question.

**The model is 11–30pp below the always-down call on every cohort at every horizon.** The
published CV figure is +3.5 / +0.1 / −1.3 / +4.4pp against the same baseline
(`2026-08-10-constant-call-is-hindsight-picked.md`). These are not the same measurement — CV
folds against 5–7 production dates — and `da-is-dominated-by-the-market-date` says the down-rate
swings 32.7→76.9% between dates, so a handful of dates can produce this on composition alone.
That is a hypothesis, not a result. It is the largest unexplained gap currently on the board and
it should be resolved before any DA figure from either source is quoted again.

## 4. What is not readable

h=30 has **one** forecast date and no per-date cell clearing 20 rows; h=14 has 2–3. The gate's
weakest evidence (30d, unconfirmed in CI at +0.0536) is also the horizon this cannot test. No
PT test is reported: `directional_test` wants a date panel, and 5–6 dates is below what the
`|t| > 3.0` hurdle was designed for.

## Method notes worth keeping

- The `no_obs` cohort is reported separately from `deviating`. `_tied_mask` maps both to "not
  tied" (`isclose(nan, nan)` is False by design), but an item with **no observation on the
  forecast date** is unknown, not deviating — 436 such rows sit at h=7 on a single date with a
  20.2% down-rate, and pooling them into `deviating` would have moved that cohort's numbers.
- 82 rows (0.6%) carry no archive mask because they fall outside the `is_backfilled` gate, and
  are dropped rather than defaulted.

## Verdict

The gate is **not refuted and not confirmed at its published size.** Keeping it is justified —
it orders better on every readable horizon and costs nothing. Quoting +0.13–0.17 as its realised
effect is not: on outcomes that resolved, the readable figure is **+0.037 at 3d and 7d, on 6 and
5 dates.**
