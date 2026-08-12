# The 12–16pp DA gap is the realised market direction of three to eight anchor dates

**Date:** 2026-08-11
**Amended:** 2026-08-12 — the source file was a 36%-deficient, **biased** view of the scored panel.
Nine published figures change, with sign reversals in three of them: the within-date term at h=14,
two of the three rising-date examples, and the h=7 direction correlation. **The central conclusion
survives.** Read the next section before any figure below. The filename keeps its original
`five-dates` slug because five other docs link it.
**Closes:** §3 of `docs/changelog/2026-08-11-clean-anchor-gate-measured-on-realised-outcomes.md`
— "the largest unexplained gap currently on the board".
**Against:** the published CV figure of **+3.5 / +0.1 / −1.3 / +4.4pp** at 3/7/14/30d
(`2026-08-10-constant-call-is-hindsight-picked.md` §2, CI run `31418286692`).
**Method (as amended):** the panel figures come from a **read-only production Postgres** session
(`set_session(readonly=True)`), read 2026-08-12; `q̄` still comes from read-only DuckDB over
`price-archive/prices-*.parquet`. No `--rescore`, no `--reresolve`, no script run, no write of any
kind, no code changed. The **original** read, 2026-08-11, was in-memory DuckDB over
`price-archive/ops/forecast_outcomes.parquet` — which is the defect.
**Also records:** the diagnosed cause of 2026-07-19's flat calls (a ±0.5% dead band, no classifier),
and a **correction** — 07-19 is the production daily run and 07-18 is the ablation arm, which is the
reverse of what the repo had recorded.

**The gap is not a model failure and not a measurement error. It is composition** — the direction
the market happened to take over three to eight anchor dates, measured against a metric whose
per-date standard deviation is 15–24pp.

## ⚠️ Amended: `ops/forecast_outcomes.parquet` is not a mirror of the scored panel

On the ≥$1 cohort the file holds **14,668 of Postgres' 23,073 rows — missing 8,405 (36%)** across
**11 of 22 (forecast_date, horizon) cells**, including **two whole dates at h=7 that no published
h=7 figure in this entry ever saw**: 2026-07-31 and 2026-08-04.

**The obvious hypothesis is refuted. It is not a sync cutoff.** The missing 2026-07-19 h=14 rows
resolved at **2026-08-05 00:09:44**, well before the file's own last write (**2026-08-11
21:10:28**), and the file already contains rows resolved as late as 2026-08-11 00:19.

**The mechanism is worse than a lag: no resolution batch after 2026-08-02 23:12:46 ever landed in
the mirror.** In every deficient cell, **100% of the rows present have `evaluated_at > resolved_at`**
and all of them share the single timestamp 2026-08-11 21:10:28 — they are there only because a later
verdict-refresh pass rewrote them (`_flush_verdict_refresh`, `backend/scripts/backtest_accuracy.py`
~line 350), and that pass writes **only rows whose stored verdict differed from the re-derived one**
(`_refresh_verdict_columns`: "Only rows whose stored verdict actually differs are written"). Each
surviving subset is therefore **selected on "the verdict changed"**, not a random subsample. That
selection is why 07-19 h=14's 115 mirror rows read DA **29.6%** / down-rate **65.2%** against the
true **39.4% / 51.1%** on 1,052 rows.

Store totals: **Postgres 121,699 rows**, max `resolved_at` 2026-08-12 00:05:52. **Mirror 114,489
rows**, max `resolved_at` 2026-08-11 00:19:05, max `evaluated_at` 2026-08-11 21:10:28. The mirror
also still carries the **21,737 NULL-`base_price` 2025-12-01 rows purged from prod**
(`2026-08-06-stale-null-base-price-outcomes-deleted.md`) — inert here only because the ≥$1 filter
drops them.

**Two independent confirmations that Postgres is the correct side.** It reproduces the CI panel
exactly — 999 + 1,093 + 1,053 + 1,052 = **4,198** at h=14 pre-exclusion, **3,146** post, matching
runs `31548564675` and `31557070748` — and the recomputed `q̄` reproduces this entry's own 07-19
`side_ref` figures to within **0.7pp**.

### The nine figures that change

Corrected values live in their own sections below; this table is the old-versus-new record.

| # | Figure | Published (mirror) | Corrected (Postgres) |
|---|---|---|---|
| 1 | Cohort | 14,668 rows / 11 dates / 1,199 items | **23,073 / 12 / 1,207** |
| 2 | h=3 pooled: dates, n, excess / side_ref / composition / assoc | 7, 5,413, −13.12 / −4.62 / −7.31 / −1.19 | 7, **7,429**, **−11.89 / −3.39 / −6.23 / −2.26** |
| 2 | h=7 pooled | 6, 4,022, −14.35 / −4.20 / −11.14 / +0.99 | **8**, **8,455**, **−12.08 / −2.53 / −9.89 / +0.34** |
| 3 | h=14 pooled | 3, 1,242, −25.52 / −1.65 / −22.80 / −1.08 | 3, **3,198**, **−15.54 / −4.55 / −15.98 / +4.98** |
| 4 | `assoc` at h=14 | −1.08 | **+4.98** — sign reversal |
| 5 | Rising-date excess: 08-01 / 08-02 / 08-06 | +19.7 / +6.4 / +4.1 | **−5.56 / −7.17 / +2.33** — two sign reversals |
| 5 | Dates with negative excess, 3/7/14d | 5/6, 4/6, 3/3 | **6/7, 7/8, 3/3** |
| 6 | `r(q_down, excess)` at 3/7d | −0.87 / −0.83 | **−0.86 / −0.55** |
| 6 | `r(p_down, q_down)` at 3/7d | −0.33 / −0.09 | **−0.33 / +0.31** — h=7 sign flip |
| 7 | Cells called unreadable: h=3 08-04, h=14 07-18 | n=16, n=34 | **n=1,049, n=1,053** |
| 8 | 07-19 counterfactual, h=14 row: served DA / sign rule / Δ / down-rate / ties | 29.6 / 34.8 / +5.2 / 65.2 / 10 | **39.4 / 43.8 / +4.4 / 51.1 / 80** |
| 9 | Pooled served flat, 3/7/14/30d | 10.5 / 7.2 / 1.3 / 0.0% | **8.0 / 3.8 / 6.0 / 0.0%** |
| 9 | Realised class flat | 23.1 / 12.7 / 4.5 / 2.8 | **23.59 / 14.12 / 6.41 / 2.83** |

The two h=7 dates the panel gains are **2026-07-31 (excess −5.62)** and **2026-08-04 (excess
−22.33, `assoc` −5.26)**. The h=3 08-04 cell that was reported as an unreadable n=16 is **n=1,049 at
excess −12.58, `assoc` −6.25** — the *worst* `assoc` cell at h=3, not an unreadable one.

Per date at h=14, with the mirror's row count in parentheses where it differed:

| date | n | DA | q_down | excess | side_ref | composition | assoc |
|---|---|---|---|---|---|---|---|
| 2026-07-17 | 1,093 (1,093) | 51.24 | 75.30 | −24.06 | −0.84 | −22.40 | −0.82 |
| 2026-07-18 | 1,053 (**34**) | 44.06 | 54.61 | −10.54 | −1.39 | −11.09 | **+1.94** |
| 2026-07-19 | 1,052 (**115**) | 39.45 | 51.14 | −11.69 | −11.55 | −5.44 | **+5.30** |
| 2025-12-01 backdated | 1,000 | 38.70 | 33.70 | +5.00 | −0.46 | +6.75 | −1.29 |

### What survives, and it is the conclusion

- **Composition is still the first-order term at every horizon** — the largest single term, and
  **≈100% of `excess` at h=14**.
- **Every live date still has negative excess**: 6/7, 7/8, 3/3 at 3/7/14d.
- **h=30 still has no live forecast date.** Its only rows are 2025-12-01, at excess +5.96.
- **The backdated batch is unaffected** — +1.80 / +3.99 / +5.00 / +5.96 — because those cells are
  complete in both stores.
- **The 07-19 dead-band diagnosis is untouched**, and so is the exclusion shipped as `66b49f4`: both
  rest on 07-19's h=3 and h=7 rows, which are **identical in both stores**.
- **The ~50-date bar on `excess` stands.** It is driven by per-date sd, which this correction does
  not move.

### What this amendment does not establish

- **The CV-basis agreement table's h=14 `assoc` of +0.42 was NOT recomputed** — its target dates
  include days the archive does not cover — so it is **unverified**, not corrected. **The h=3 and
  h=7 legs of that table were not re-derived either.** "CV and production agree" is this entry's
  headline and it now rests on legs measured on the panel this amendment has just corrected.
- Figures use the **stored** `direction_predicted` / `direction_actual`, not a re-derivation from
  the frozen actuals.
- `side_ref` and `composition` depend on the recomputed `q̄`; their **sum** does not, and `excess`
  and `assoc` are `q̄`-free.
- **h=14 ex-07-19 is 07-17 plus 07-18, and 07-18's rows are the `--compare-regime` ablation arm** —
  so that 2-date panel is half a disputed cohort. Its `n` is fine; its membership is not.
- Recorded as an unexplained observation: **something ran a verdict refresh against production at
  2026-08-11 21:10:28 UTC.** That write is the only reason the deficient cells have any rows at all.
  Nothing in this repo's scheduled chain has been tied to it.

## Cohort

`served_identity` matching `lgbm-v3*`, `base_price >= 1.0` (`MIN_SERVED_PRICE_USD`), scored rows
only: **23,073 scorable rows over 12 forecast dates and 1,207 items** (Postgres, 2026-08-12), of
which the 2025-12-01 backdated batch is reported separately throughout. The original read of this
line was **14,668 rows / 11 dates / 1,199 items** off the ops mirror.

⚠️ **The 14,586-row and 82-row universe notes derive from the stale count and were not re-derived.**
§3's 14,586 was described here as this cohort minus **82 rows on 6 slugs** falling outside
`archive_universe_sql_filter` — the same 82 rows §3 dropped rather than defaulted. Both figures are
mirror-era. The `_tied_mask` reproduction of §3's tied/deviating split also happened on the mirror
panel, so "this is the same population as §3" now says only that both reads used the same deficient
source — which is exactly why §3 needs the banner it now carries.

## The decomposition

`excess = DA − q_down` is additive per (horizon, forecast date), with `p_c` the model's served call
share, `q_c` the realised class share on that date, and `q̄_c` the archive-average class share:

- **`side_ref = Σ_c p_c q̄_c − q̄_down`** — the model's call mix costed against a *typical* market.
  What its side selection is worth before this window's direction is known.
- **`composition = (Σ_c p_c q_c − q_down) − side_ref`** — this window's realised direction.
- **`assoc = DA − Σ_c p_c q_c`** — within-date directional information. **The only skill term.**

`q̄_c` is recomputed (2026-08-12) on the corrected cohort's **1,201 slugs** over **4,725 anchor
dates** — voted median across non-bid sources under `archive_universe_sql_filter`, ±0.5% flat band:
`(down, flat, up)` = **`(.450, .108, .442)`** at h=3 and **`(.4832, .0531, .4637)`** at h=14, against
the mirror-era `(.453, .100, .447)` at h=3. **The h=7 and h=30 recomputed vectors were not
tabulated** — the h=7 row's `side_ref` / `composition` split below rests on a recomputed `q̄` this
entry does not print. The superseded read used 1,193 items over 1,282–1,307 anchor dates and gave
`(.487, .033, .480)` at h=30.

Always-down has `assoc ≡ 0` and `side_ref ≡ 0` by construction — it makes one call and that call is
the reference — which is what makes the attribution clean rather than a change of basis.

## The result: composition is first-order; the within-date term is −2.3 / +0.3 / +5.0pp

Live dates pooled, served basis, **corrected 2026-08-12** (components sum to `excess` to within
rounding):

| h | dates | n | excess | side_ref | composition | assoc |
|---|---|---|---|---|---|---|
| 3 | 7 | 7,429 | **−11.89** | −3.39 | **−6.23 (52%)** | **−2.26** |
| 7 | 8 | 8,455 | **−12.08** | −2.53 | **−9.89 (82%)** | **+0.34** |
| 14 | 3 | 3,198 | **−15.54** | −4.55 | **−15.98 (103%)** | **+4.98** |
| 14, excl 07-19 | 2 | 2,146 | **−17.43** | −1.11 | **−19.40 (111%)** | **+3.08** |
| 30 | — | — | no live forecast date exists at all | | | |

Superseded (ops mirror, 2026-08-11): h=3 7 dates / 5,413 / −13.12 / −4.62 / −7.31 / −1.19; h=7 **6**
dates / 4,022 / −14.35 / −4.20 / −11.14 / +0.99; h=14 3 dates / 1,242 / −25.52 / −1.65 / −22.80 /
−1.08. h=7 gains two dates it never had (07-31, 08-04) and h=14's `excess` moves **10pp less
negative**.

52–103% of the gap is the direction of the window — composition is the largest single term at every
horizon and accounts for essentially all of `excess` at h=14. Nothing in the pooled deficit is a
statement about the model's ordering.

⚠️ **The claim "the within-date term is ±1.2pp at every horizon" fails at h=14 — and it fails in the
model's favour.** On the true panel `assoc` is **+4.98** there, not −1.08: the model beats its own
call mix by **3–5pp** at h=14, driven by 2026-07-19 at **+5.30 on 1,052 rows** rather than the
mirror's 115. The h=3 term also roughly doubles in size, to −2.26. So the readable statement is
that `assoc` is within ±2.3pp at 3d and 7d and **positive** at 14d, on three dates.

**The panel mixes three direction rules** — a zero-threshold sign rule, a ±0.5% dead band, and the
classifier, in that order (diagnosed below). Almost all of the `side_ref` column is a single
dead-band date, 2026-07-19. That does not move the conclusion — neither `side_ref` nor `composition`
is skill either way — but it means the `side_ref` column is largely a one-date configuration
artifact rather than a property of the served classifier's call mix.

## The backdated batch differs in exactly one respect, and it flips the sign

Same model, same metric, 2025-12-01 — and these four figures are **unchanged by the 2026-08-12
correction**, because the backdated batch's cells are complete in both stores: excess
**+1.80 / +3.99 / +5.00 / +5.96** at 3/7/14/30d —
**inside the CV band at all four horizons.** Its forward window *rose*, by an up-rate minus
down-rate of +15 to +33pp. That is the only respect in which it differs from the live dates. A
cohort that is 12–16pp below the baseline on falling windows and 2–6pp above it on a rising one is
describing the windows, not the artifact.

## The decisive number: on the CV label basis, the two measurements agree — but the table is now unverified

> ⚠️ **None of this table was recomputed on 2026-08-12, and it is the entry's headline.** Every cell
> was read on the ops-mirror panel now known to be 36% deficient and verdict-selected. The **h=14
> `assoc` of +0.42 cannot be recomputed cheaply** — its target dates include days the archive does
> not cover — so it is **unverified, not corrected**. The h=3 and h=7 legs were not re-derived
> either. Treat "CV and production agree" as an **open claim resting on a corrected panel's
> predecessor**, not as confirmed. What *is* now known is that the served-basis `assoc` moved on the
> same cells: −1.19 → −2.26 at h=3, +0.99 → +0.34 at h=7, −1.08 → **+4.98** at h=14.

Recomputing the *same live item-dates* on the CV label basis — `p[d+h] / p[d]`, `prepare_targets`'
own raw-anchor denominator with `LABEL_SMOOTHED_ANCHOR` off:

| h | `assoc`, CV basis | per-date mean (dates with n ≥ 100) | published CV edge |
|---|---|---|---|
| 3 | **+2.70** | +2.33 ± 0.85 (5 dates) | **+3.5** |
| 7 | **+0.17** | +0.33 ± 1.35 (5 dates) | **+0.1** |
| 14 | **+0.42** | one date | **−1.3** |

On the panel as read on 2026-08-11 they agree, and the reading was that the published CV edge was
never in conflict with production — it was being compared against a pooled `excess` that carries a
composition term CV averages away. **That second sentence survives the correction on its own terms**
(composition is still first-order); the numerical agreement above does not yet, per the banner.

**Basis moves `excess` by +2.6 / −3.5 / +9.5pp, with no consistent sign** — mirror-era, not
re-derived. DA is far less
basis-sensitive than rank IC — where the anchor denominator carried **+0.1398 of a +0.1464** swing,
all in one direction (`2026-08-11-the-gap-is-the-anchor-denominator.md`) — because the ±0.5% flat
band (`DIRECTION_FLAT_TOLERANCE_PCT`) absorbs most of the denominator perturbation before it can
change a class label.

## The model is above always-down on two live cells only — this claim weakened badly

⚠️ **Corrected 2026-08-12, and the correction takes most of this section with it.** Two of the three
worked examples **reverse sign** on the true panel: 2026-08-01 **+19.7 → −5.56** and 2026-08-02
**+6.4 → −7.17**. Only 2026-08-06 holds, and smaller: **+4.1 → +2.33**. The **positive live cells
are two**: 08-06 at h=3 (+2.33) and 07-29 at h=7 (+2.33). Dates with negative excess are **6/7 at
h=3, 7/8 at h=7, 3/3 at h=14** (published 5/6, 4/6, 3/3).

So the statement that stands is much weaker than the one this section made: on the corrected panel,
**16 of 18 live cells are below the runnable baseline**, and the two that are above it clear it by
2.33pp. §3's "on every cohort at every horizon" is still not literally true of the panel, but the
counterexamples are two small positives rather than a rising-market pattern worth a paragraph. What
does the work in this entry is the decomposition, not the exceptions.

Over live dates with n ≥ 20, corrected:

- `r(q_down, excess)` = **−0.86** at h=3 and **−0.55** at h=7 (published −0.87 / −0.83). The
  down-rate still orders `excess` at h=3; at h=7 that relation is materially weaker than published.
- Dates with negative excess: **6/7, 7/8, 3/3**.

Pooling-versus-unweighted, **h=14 only** (the h=3 and h=7 unweighted means and `r(n, q_down)` were
**not re-derived**, so the mirror-era +0.47 / +0.49 and −11.63 / −11.45 stand unverified): pooled
**−15.54** reads against an unweighted per-date mean of **−15.43 ± 7.50**, against a mirror-era
−32.65 ± 7.6. At h=14 the pooling effect is therefore **0.11pp — it has vanished**, where the
mirror panel showed 7pp of it. The published "pooling over-weights the falling dates" reading was
itself partly an artifact of which rows survived into the mirror.

## Sampling: the entire published gap is one to two standard errors

A zero-skill model carrying the live call distribution, evaluated on every archive anchor date with
≥ 200 item-pairs, lands **≤ −11pp on 30.0 / 37.8 / 34.8 / 33.5%** of dates and **≥ +3.5pp on
30.0 / 40.5 / 43.8 / 48.4%**. Per-date sd of `excess` is **14.96 / 22.40 / 21.56 / 24.39pp**, so
the SE of a mean over the handful of dates each horizon actually has is 6.7–15.3pp. The live draw
sits at **z = −1.0 to −1.6**.

⚠️ **Not re-derived on 2026-08-12, in two places.** The simulation's call distribution `p_c` was
taken from the mirror panel, and `z` was computed from the superseded pooled means. The per-date sd
is the part that matters for the ~50-date bar, and it comes from the archive-wide simulation rather
than from this panel, so **the bar itself is unaffected**. The corrected pooled `excess` is smaller
in magnitude at every horizon, so the correction does not move the conclusion against itself; the
exact `z` at each horizon is **unmeasured** — at h=7 the date count changed from 6 to 8 as well, so
it cannot be recovered by arithmetic from what is published here.

An ordinary unlucky sample. A statistic this dispersed produces a reading at or below −11pp on
roughly a third of dates from a model with **no skill at all**, and a reading at or above +3.5pp
about as often.

## The one actionable defect: a ±0.5% dead band shipped 42 minutes before the 07-19 run

**2026-07-19 predicted `flat` for 64.0% of items at h=3** (34.3% at h=7, **24.05%** at h=14 —
corrected from 25.2%; the h=3 and h=7 shares reproduce exactly, because 07-19's cells at those two
horizons are identical in both stores). That date alone carries `side_ref` of
**−22.7 / −14.5 / −11.55pp** — essentially the entire pooled `side_ref` term — and **no other date
in the panel emits a single flat call** (`p_flat` = 0.00% on 2025-12-01, 07-17, 07-18, 07-29,
08-01, 08-02, 08-04, 08-05, 08-06, 08-07 — measured on the mirror panel, so **2026-07-31's flat
share was never read**). Pooled flat across the ≥$1 scored panel is
**8.0 / 3.8 / 6.0 / 0.0%** at 3/7/14/30d — corrected from 10.5 / 7.2 / 1.3 / 0.0 — and **0.00% once
07-19 is dropped**, on the dates that were read.

The corrected quadruple is internally consistent with the corrected cell counts and with 07-19 being
the only flat-emitting date: at h=14, `0.2405 × 1,052 / (3,198 + 1,000) = 6.03%` against the
reported 6.0%. Since the pooled figure would have to exceed that if any of 07-17's or 07-18's
**full** 1,053–1,093 rows carried a flat call, this also extends their `p_flat = 0.00%` from the
mirror's subsets to the whole cells at h=14. It is arithmetic on published counts, not a second
measurement.

**Diagnosed in a parallel read, and the mechanism is unambiguous.** `f71ffb4` ("fix: Correct
accuracy evaluation — pct_error denominator, multi-source voting, direction threshold",
2026-07-19 18:57:46 −0400 = **22:57 UTC**) introduced `DIRECTION_FLAT_TOLERANCE_PCT = 0.5` and
replaced `predict()`'s `up if mid_ret > 0 else down if mid_ret < 0 else flat` with a global ±0.5%
dead band. The 07-19 rows were written at `created_at = 2026-07-19 23:39:12` — **42 minutes
later.** 07-17 (22:04) and 07-18 (18:33) predate the commit, which is exactly why they show no
flat. Reconstructing `mid_ret = price_mid/current_price − 1` and testing every row against a global
±0.5% band with a half-cent rounding allowance is **100% consistent in 12 of 12 (horizon × call)
cells** — 668/668 flat, 151/151 up, 234/234 down at h=3, and the same at 7/14/30d. The apparent
boundary fuzz (flat rows out to ±0.95%, down rows in to −0.376%) is cent-quantisation of
`round(price, 2)`, **not** per-tier thresholds.

**It was not a classifier at all**, so the "confident flat versus three-way near-tie" question does
not arise — there were no class probabilities. `direction_models` was empty until `a332c2b`
(2026-07-24), and `_recenter_on_direction` demonstrably did not run: only 252 of the 668 flat rows
at h=3 have `price_mid = current_price`, and those are cent-rounding.

**Why the band caught the majority: the served mid is shrunk far harder than realised returns are.**
On 07-19 at ≥$1, **62.2%** of `|mid_ret|` fell under 0.5% at h=3 (35.8% at h=7) against a **23.59%**
realised flat rate — the gap to the 64.0% served flat share is the same half-cent rounding
allowance. On 07-17 it was **99.8%** (p25 = p50 = p75 = 0.000): had the band been live that day it
would have called almost everything flat. A symmetric dead band against a shrunk prediction
distribution.

⚠️ **Two flat rates, and they must not be conflated.** `q̄_flat` = **10.8%** at h=3 in the
decomposition above (recomputed; 10.0% on the mirror read) is the *archive average* over 4,725 dates.
**The realised flat rate in this cohort is 23.59% / 14.12% / 6.41% / 2.83% at 3/7/14/30d** —
corrected from 23.1 / 12.7 / 4.5 / 2.8 — and that is the number a flat call on
07-19 was up against. So a flat call there was a **low-hit-rate call — 18.3% hit against a 23.59%
base rate — not "close to a guaranteed miss"**, which an earlier draft of this entry claimed off
the archive-wide figure. One consequence of the identity: `side_ref` costs the call mix at `q̄_flat`
while that date realised more than twice it, so part of the flat penalty sits in that date's
`composition` term rather than in its `side_ref`.

**Counterfactual cost** — the served mid's sign joined to the realised direction, with ties still
counted flat, so these are lower bounds on the call mix's cost:

| h | served DA | zero-threshold sign rule | Δ | realised down-rate | ties |
|---|---|---|---|---|---|
| 3 | 27.6% | 36.3% | **+8.7** | 52.3% | 251 |
| 7 | 29.8% | 38.4% | **+8.6** | 54.4% | 87 |
| 14 | **39.4%** | **43.8%** | **+4.4** | **51.1%** | **80** |

**The h=14 row is corrected (2026-08-12)** from 29.6 / 34.8 / +5.2 / 65.2% / 10 ties: the mirror had
115 of that cell's 1,052 rows, selected on "the verdict changed". **The h=3 and h=7 rows reproduce
exactly**, because 07-19's cells at those horizons are complete in both stores.

So at least +8.7pp of that date's penalty at h=3 is the call *mix*. The remainder is that 07-19 was
wrong-sided anyway — 21.9% down calls into a 52.3% down-rate.

**Blast radius: nothing served today is affected, and the rule is still reachable.** Flat is
**0.0% at every horizon on every classifier-era ≥$1 date** — 07-29 and 08-05 in the
`item_forecasts` mirror, 08-05 / 08-06 / 08-07 in the scored outcomes. Below $1 the classifier
*does* emit flat, on 29.4–33.0% of rows, but that is under `MIN_SERVED_PRICE_USD = 1.0`
(`backend/api/serving_policy.py:26`) — written to the mirror, never served and never scored.
**The dead band remains live as a fallback:** `predict()`'s no-classifier `else` branch
(`backend/models/forecaster.py:6965-6976` as of `67324fb`) reapplies `t_down` / `t_up` defaulting to
±`DIRECTION_FLAT_TOLERANCE_PCT` whenever `self.direction_models` has no entry for a horizon, so an
artifact missing its classifier re-enters this exact mode **silently**.

### Which run each date is — the labels are the reverse of what the repo recorded

⚠️ **`da-is-dominated-by-market-date` and any prose following it have this backwards, and this is a
correction, not a refinement.** At that day's code state (`git show
f71ffb4:backend/scripts/forecast_prices.py`), **run A — the unconditional daily path — writes
`f"{MODEL_VERSION}-regime"`** (`:212-213`), while `-global-only` is written **only** by
`--compare-regime`'s run B (`:217-228`), which writes to the same
`(item_id, forecast_date, horizon_days)` key and therefore **overwrites** run A's rows.

Therefore **2026-07-19 (`lgbm-v3-regime`) is the production daily run**, and **2026-07-18
(`lgbm-v3-global-only`) is an ablation arm's rows that overwrote that day's production forecast.**
Not the other way round. The overwrite-on-one-key mechanism is the same one recorded in
`2026-08-11-model-version-is-not-a-config.md`; what is new is which direction it ran.

**Coverage caveat on that argument.** The `item_forecasts` mirror holds only 6 forecast dates
(2025-12-01 — itself backdated on 07-17 — plus 07-17 / 07-18 / 07-19, 07-29, 08-05) and last synced
2026-08-05, so it carries **no `model_config` column** for these dates. The arm-label reading rests
on the **writer code at that commit**, not on an observed config. The scored panel jumps 07-19 →
07-29, consistent with the known 2026-07-14 → 07-31 CI outage.

## Also measured: no date-level market-direction information at h=3 — and the h=7 leg flipped

`r(p_down, q_down)` = **−0.33** at h=3 and **+0.31** at h=7 (corrected 2026-08-12 from −0.33 /
−0.09). The pooled figure across all readable live cells (**−0.17** over 15 cells) was **not
re-derived** and the cell count has since changed. The model called down on **71.7%** of items into a
25.7% down-rate (2026-08-06) and on **11.0%** into a 54.4% down-rate (2026-07-19 at h=7).

⚠️ **The h=7 correlation changes sign, so this claim now rests on h=3 alone.** A +0.31 at h=7 is the
sign a model with *some* date-level market information would show. On 8 dates it is not a result
either way — but "the model has no date-level market-direction information" is a one-horizon
observation on the corrected panel, not a two-horizon one.

⚠️ One confound, which weakens rather than reverses this: **2026-07-19's call shares are the
dead-band rule's, not a classifier's** (see above), so its `p_down` is not comparable to the other
dates' and it is one of the cells inside these correlations. They were not recomputed with 07-19
excluded.

This is consistent with the closed own-history leg of N2 — the market factor is not forecastable
from its own history (`2026-08-11-market-factor-is-not-forecastable-from-its-own-history.md`,
refuted in both directions by `2026-08-11-mean-reversion-does-not-replicate.md`). It is **not new
evidence against the cross-sectional model**, which is not asked to call the market's level.

## Why the two measurements ever differed: different units of observation

A CV fold validates on `VALIDATION_WINDOW_DAYS = 30` distinct dates
(`_compute_cv_splits`: `val_d = sorted_dates[end:end + val_window]`), which is **15,532–24,979 rows
over ~30 anchor dates** per fold at h=3 (`n_val`, local `meta.json`; fold 1 spans 2023-02-26 →
2023-03-27). So a fold's `realised_down_rate` is *already averaged over a month of market
direction*, and its composition term averages to the archive mean — **−4.6 to −0.2pp**.

A live "date" is **one anchor**. Three to eight of them do not average to anything. The two numbers
were never the same measurement, and §3 was right to say so; what it lacked was the size.

## The consequence, and it is the recommendation

**Never quote `excess` on fewer than ~50 forecast dates. Report `assoc` instead.**

`MIN_FORECAST_DATES = 20` (`backend/backtest/scoring.py:84`) is documented in-source as "a judgement
call, not a derivation", set to span more than one market swing. **It is not calibrated for a
statistic whose per-date sd is 15–24pp**: taking the measured per-date sd at face value, `sd/√20`
is 3.3–5.5pp — arithmetic, not a measurement — which cannot resolve a +3.5pp effect at 20 dates.
Twenty dates fixes the "which two dates did the cohort contain" failure it was written for; it does
not make `excess` a readable number. `assoc` is readable on the dates that exist, because the market
term is differenced out per date — which is also exactly what the PT statistic does, and why the
published headline is PT and not DA.

## What was deliberately not done

- **The `num_only` / `den_only` basis legs were not read.** Stored `base_price` sits a median
  **8.2% / 6.8% / 1.2% / 0.0%** from today's recomputed voted quote at *d*, so mixing one frozen
  leg with one recomputed leg measures the level wedge, not the basis: those legs return a 94.7%
  down-rate on one and an 86.7% up-rate on the other. Reported as artifacts and dropped. (This is
  a different pair from the `base_price`-vs-`current_price` wedge of median 5.70% / p90 37.82% in
  `2026-08-11-actionable-selection-is-the-base-wedge.md`.)
- **The CV figure was not recomputed.** `+3.5 / +0.1 / −1.3 / +4.4` is taken from CI run
  `31418286692` as cited in `2026-08-10-constant-call-is-hindsight-picked.md`. The local
  `meta.json` has `mean_classifier_acc = None` — `CV_DIAGNOSTIC_CLASSIFIER=0` on the production
  path, and the per-fold classifier is 52% of a retrain
  (`2026-08-10-arm64-and-scheduled-diagnostics.md`). **`mean_classifier_acc` is the correct CV
  counterpart**, because `forecast_outcomes.direction_predicted` is the served classifier's call
  (`predict`: "Served signal: the directional classifier"), **not `mean_dir_acc`** — which is the
  q50 sign and reads 39.9 / 42.0 / 43.1 / 48.4 in the local `meta.json`.
- **No `--rescore`, and no `--reresolve`.** `--rescore` refits served bias corrections and is a
  production write. The 2026-08-12 correction is a **read-only** Postgres session
  (`set_session(readonly=True)`), no script invoked, so it neither repairs the mirror nor touches a
  stored verdict. ⚠️ The original entry's claim that "the ops mirror answers this question without
  one" is exactly the claim that turned out to be false.
- **The mirror was not repaired, and its root cause was not traced past the symptom.** What is
  established is that no resolution batch after 2026-08-02 23:12:46 landed in
  `ops/forecast_outcomes.parquet` and that the surviving rows in the gaps came from a verdict
  refresh. **Which step stopped writing, and why, is unmeasured** — this entry did not read
  `aggregator-update.yml` / `backtest-accuracy.yml` run logs for the publish leg.
- **The h=7 and h=30 `q̄` vectors were not tabulated**, so the h=7 `side_ref` / `composition` split
  cannot be re-derived from what is printed here. Their sum, `excess` and `assoc` are unaffected.
- **No PT test.** 3–8 dates is below what the `|t| > 3.0` hurdle and the HAC bandwidth rule were
  designed for — the same reason §4 of the prior entry reported none.
- **No interval on `assoc`.** Sign and size across dates are what is readable here.
- **The 2026-08-10 next-steps ranking was not rewritten.** Only the claims this contradicts are
  annotated — including, on 2026-08-12, its repetition of the corrected decomposition figures.

## What this does not establish

- **The decomposition is new methodology, built for this read.** It is not an existing repo metric,
  has no test, and is not computed by `scoring.py`. **Flag it for review before it becomes the
  reported number.**
- **The CV-basis agreement rested on 5 / 5 / 2 / 0 dates of the mirror panel, and is now
  unverified** — see the banner on that section. Its date counts describe the deficient panel and
  were not re-derived, so even "at 3d and 7d only" is a mirror-era statement.
- **h=30 has no live forecast date.** Its only resolved rows are the backdated batch. **Unchanged by
  the correction.**
- ⚠️ **The list of unreadable cells was largely void, and is corrected.** h=3 2026-08-04 is
  **n=1,049**, not 16 — and at excess **−12.58** / `assoc` **−6.25** it is the *worst* `assoc` cell
  at h=3, not an unreadable one. h=14 2026-07-18 is **n=1,053**, not 34. **h=14 does not rest on the
  2026-07-17 cell alone**; all three of its cells clear the 20-row bar comfortably. What remains true
  is the membership objection: 07-18's rows are the `--compare-regime` ablation arm, so **h=14
  ex-07-19 (07-17 + 07-18) is half a disputed cohort** — its `n` is fine, its membership is not.
  CV basis was unavailable for h=3 08-06 and 08-07, h=7 08-02, and h=14 07-19 — targets or anchors
  beyond the archive's 2026-08-08 max day, or on the missing days 07-27 / 07-30 / 08-02 / 08-03 —
  and that availability list was **not** re-derived against the corrected panel.
- `mean_realised_down_rate` of **46.06 / 48.73 / 50.56 / 48.48** is the **pooled** CV value, not a
  ≥$1-specific one. A minor caveat on the CV side of the comparison, unquantified.
- **The pooled table was not recomputed with 2026-07-19 excluded** (which would leave 6 / 7 / 2
  dates at 3/7/14d on the corrected panel; **h=14 ex-07-19 is now tabulated**, h=3 and h=7 are not),
  so its `side_ref` and `assoc` are mixtures of serving configurations. And it is
  **three** rules, not two: `f71ffb4`'s ±0.5% dead band covers 07-19 only; 2025-12-01, 07-17 and
  07-18 were written by the **zero-threshold sign rule** that commit replaced; the classifier
  (`a332c2b`, 07-24) serves 07-29 onward. Which of those dates lands in which horizon's panel was
  not tabulated, so how much of each pooled cell is classifier-era is **unmeasured**.

## Verification

**Corrected panel (2026-08-12).** Cohort counts, the pooled and per-date decompositions, the
correlations, the flat rates and the counterfactual's h=14 row come from a **read-only production
Postgres** session — `set_session(readonly=True)`, no script run, no `--rescore`, no `--reresolve`,
no write of any kind. `q̄` comes from read-only DuckDB over `price-archive/prices-*.parquet` on the
cohort's 1,201 slugs across 4,725 anchor dates, voted median across non-bid sources under
`archive_universe_sql_filter` with a ±0.5% flat band. Two independent checks that this is the correct
side: the h=14 counts reproduce the CI panel exactly (999 + 1,093 + 1,053 + 1,052 = 4,198
pre-exclusion, 3,146 post — runs `31548564675` and `31557070748`), and the recomputed `q̄` reproduces
this entry's own 07-19 `side_ref` figures to within 0.7pp. The deficiency itself was diagnosed by
comparing the two stores cell by cell and by the `evaluated_at > resolved_at` test described above;
`_flush_verdict_refresh` / `_refresh_verdict_columns` were read at source
(`backend/scripts/backtest_accuracy.py:350` and `:382`) for the write-only-on-difference behaviour.

**Original read (2026-08-11), which is the superseded one.** Cohort counts, `_tied_mask`
reproduction and every figure not re-derived above come from one read-only DuckDB
session over the ops mirror and the archive Parquet; the CV-side numbers (`n_val`, per-fold
`val_start` / `val_end`, `mean_realised_down_rate`, `mean_dir_acc`, `mean_classifier_acc = None`)
were read from `backend/models/saved_models/meta.json` (`trained_at` 2026-08-09). `VALIDATION_WINDOW_DAYS = 30`,
`CV_STEP_DAYS = 150`, `DIRECTION_FLAT_TOLERANCE_PCT = 0.5`, `MIN_FORECAST_DATES = 20` and the
`prepare_targets` denominator were read at source. No tests were added — there is no new code to
test, which is itself a reason the decomposition should not be published from a one-off read.

The 07-19 diagnosis rests on a **separate, parallel read** plus git archaeology, re-verified here at
source: `git show f71ffb4` carries the `DIRECTION_FLAT_TOLERANCE_PCT = 0.5` definition and the
`predict()` line it replaced; `git show -s f71ffb4` gives 2026-07-19 18:57:46 −0400; `a332c2b` is
dated 2026-07-24; `git show f71ffb4:backend/scripts/forecast_prices.py:206-228` is the run-A /
run-B writer that fixes the arm labels; the live fallback branch is `forecaster.py:6965-6976` and
`MIN_SERVED_PRICE_USD = 1.0` is `api/serving_policy.py:26`. The row-level consistency check,
`created_at` timestamps, per-date flat shares and the counterfactual table come from the parallel
read over the same ops mirror — 07-19's h=3 and h=7 cells, which that diagnosis rests on, are
**identical in the mirror and in Postgres**, so none of it is affected by the 2026-08-12 correction.

## Still open

> ## ✅ Both 07-19 recommendations SHIPPED the same day, after this entry was written.
> `37b09f9` guards the fallback branch (`_warn_no_classifier`, reporting the horizon, the
> item count and the flat share once per horizon, silent when the classifier is present).
> `66b49f4` excludes the date, as `EXCLUDED_FORECAST_DATES` in `backtest/scoring.py` applied
> at `_records_from_frozen_outcomes`. Full suite green at each: 1,993 then 2,009 passed.
> **The exclusion is scoring-only** — resolution is untouched, so 07-19's outcomes still
> freeze and only never reach a metric, which a test asserts directly. It is counted at
> WARNING with its reason, and `SCORE_ALL_DATES=1` restores the rows for comparison against
> the pre-exclusion figures while logging that its output is not publishable.
> **Every pooled `side_ref` figure in this entry therefore describes a panel the code no
> longer scores.** The numbers stand as the measurement that motivated the exclusion; the
> next scheduled Backtest Accuracy run is the first to report without the date, and no
> `--rescore` was forced to bring it forward.
> One bug worth recording, because a unit test could not have caught it: `forecast_date`
> arrives as an **ISO string** under SQLite and a `date` under psycopg2, so the first
> version matched only `date` and the exclusion **silently never fired** — dropping nothing
> and reporting nothing. The end-to-end wiring test is what found it.
> **07-18 is still IN, and that is unresolved** — see item 1.

1. **07-18's cohort membership.** Its rows are `--compare-regime`'s run B overwriting that day's
   production forecast, so they are an ablation arm's output — but written under the *ordinary*
   sign rule, which makes it a cohort-membership question rather than an estimator one. The
   2026-07-19 exclusion deliberately does not cover it, and the distinction is recorded in
   `EXCLUDED_FORECAST_DATES`' own comment so the two are not conflated later.
3. **Whether `assoc` should be a stored metric** beside DA and the PT statistic. It is the term that
   is readable on the panel the repo actually has. Raised here as a question; not a decision this
   entry makes.
4. **A production DA headline still needs the calendar** — 8 / 7 / 4 / 1 dates at 3/7/14/30d after
   `served_identity` merged the panel (`2026-08-11-model-version-is-not-a-config.md`). This entry
   says the 20-date bar is the wrong bar for `excess`, not that the panel is now sufficient.
5. **`ops/forecast_outcomes.parquet` is still 36% deficient and still biased**, and nothing has been
   fixed: no resolution batch after 2026-08-02 23:12:46 has landed in it. Every API route and
   analysis that reads Parquet-first is reading that file. Which publish step stopped writing is
   **unmeasured** (see "not done"), and the mirror is a served surface, not only an analysis
   convenience — `db/parquet.py` is read before the DB fallback.
6. **Nothing explains the 2026-08-11 21:10:28 UTC verdict refresh against production.** That single
   `evaluated_at` timestamp is the only reason the deficient cells hold any rows, and it is the
   mirror's own last write. It has not been tied to a scheduled run, a dispatch or a local
   invocation. Recorded as an observation.
7. **The CV-basis agreement table needs re-deriving on the corrected panel** before "CV and
   production agree" is quoted again — it is this entry's headline and it currently rests on the
   deficient panel at all three horizons.

## Related

- `2026-08-11-clean-anchor-gate-measured-on-realised-outcomes.md` — §3 is what this closes. ⚠️ Its §1
  and §2 (rank IC +0.037; the gate does not move direction) were read from the **same deficient
  file over the same period** and now carry a banner saying so. "Untouched", as this entry
  originally put it, is wrong: they are unrecomputed on a panel known to be 36% deficient and
  verdict-selected.
- `.claude/rules/archive-reads.md` and `backend/AGENTS.md` — where the "do not read
  `ops/forecast_outcomes.parquet` for panel work" rule and the per-cell `evaluated_at > resolved_at`
  test now live.
- `2026-08-10-constant-call-is-hindsight-picked.md` — the runnable-baseline correction this builds
  on. `constant_call_accuracy` is still not to be differenced against model DA.
- `2026-08-03-accuracy-is-clustered-by-forecast-date.md` — the clustering that makes composition
  this large; "the down-rate swings 32.7 → 76.9% between dates" is the same fact, now decomposed.
- `2026-08-11-the-gap-is-the-anchor-denominator.md` — the rank-IC basis swing this contrasts with.
