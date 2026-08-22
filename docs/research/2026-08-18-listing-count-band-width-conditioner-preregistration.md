# Pre-registered read: does `log1p(listing_count)` as a band-width conditioner tighten conditional coverage?

> **Status (as of 2026-08-21): SCORED 2026-08-18 — REFUTED, 0 of 3 horizons pass. Do not re-run.**
> `changelog/2026-08-18-listing-count-conditioner-refuted.md`. h=3 cuts dispersion 27% but
> worsens the worst-bucket miss (guardrail fail); h=7/14 worsen dispersion. The decisive finding
> is **structural**: the thin buckets (1–5, 6–15 listings) that are the whole target hold <20
> items in the served ≥$1 cohort and drop out — per-item `sigma` already absorbs listing
> information for the population actually served. **Reopen only if** the iflow backfill
> materially grows the thin-listing ≥$1 cohort; re-check bucket `n` first.


**Date:** 2026-08-18
**Status:** pre-registration. No code changed. Scored later, inline, below.
**Owner ruler:** conditional (per-listing-bucket) coverage on **served/replayed outcomes**, NOT
exceedance rate and NOT marginal coverage.

## Hypothesis and provenance

`research/2026-08-16-listing-count-floor.md` established, read-only, that thin items are wilder
**monotonically and within price tier** (P(|r₇|>5%) runs 0.68 → 0.50 across the 1–5 → >100
listing buckets; the same gradient holds inside every $-tier, so it is not the cheap-item
confound). It rejected a listing-count *floor* — the ≥30 index threshold sits in the flat part of
our payoff curve, costs 69% of the cohort for 0.011 — and recommended feeding listing count to the
**band width** instead: "a conditional-σ statement — thin items need wider intervals, not
exclusion." That doc explicitly deferred the build behind one unanswered question, which this test
answers: **are low-listing rows the ones under-covered today?** That is a coverage question and
needs served outcomes, which the local `cs2_market.db` fixture cannot supply.

Standing band context: the served 80% band **over-covers marginally** (87.2 / 91.8 / 90.6 / 89.0%
at h=3/7/14/30, 19,917 prod outcomes). The expanding window, calibration centre, quiet dates and
the sigma profile are all **refuted** as the cause
(`changelog/2026-08-12-expanding-window-refuted-for-band-width.md`); the residual is regime
non-exchangeability, for which offline marginal remedies are exhausted. **This test does not
target the marginal over-coverage.** It targets the *conditional* miscalibration — per-date
coverage swings 58–99%, and the hypothesis is that a meaningful slice of that swing is ordered by
listing count.

## Why this is a real test and not a re-read

The 2026-08-16 gradient is measured on **exceedance rate** (|r₇| magnitude), which is mechanically
correlated with any width signal and cannot by itself say the band is *mis*-calibrated — a wider
true move is fine if the band already widened for it. The novel, non-circular quantity here is
**coverage residual by listing bucket**: does the *current* band, which has no listing-count term,
systematically miss more often on thin items than on liquid ones? Only that justifies adding the
term. It is measured on realised outcomes the band never saw, not on the same |r| the gradient was
built from.

## Arms, fixed in advance

The band half-width today is `q_hat_h · m_h · sigma_item` where `q_hat_h` is the per-horizon
split-conformal quantile, `m_h` is the served-outcome multiplier
(`ItemForecaster.served_qhat_multiplier`, `models/served_recalibration.factors_from_panel`), and
`sigma_item` is the per-item scale. The conditioner plugs in **at the same site as `m_h`** — a
multiplicative width factor, never a gate, never a cohort filter.

- **CONTROL** — production band, unchanged: `w = q_hat_h · m_h · sigma_item`.
- **TREATMENT** — `w = q_hat_h · m_h · sigma_item · g(L_i)`, with
  `g(L) = exp(β_h · (log1p(L) − c))`, `L` = listing count (**max over the four non-Steam supply
  feeds** `lis_skins / market_csgo / waxpeer / bitskins`, matching `forecaster.py`'s documented max
  rule), `c` = the cohort mean of `log1p(L)` (so `g` is width-neutral on average by construction —
  it *redistributes* width, it does not inflate it), and `β_h ≤ 0` fit per horizon on a
  **calibration split** to equalise per-bucket coverage, then frozen and applied on a disjoint
  **scoring split**. β sign is constrained non-positive a priori (thin ⇒ wider); a fitted β_h > 0
  is a VOID, not a result.

**Missingness policy, fixed in advance:** 16.3% of the served cohort has no listing count (26.2%
in $1-5, 0.8% at ≥$1k; missingness is itself informative). Missing rows take `g = 1` (control
width) — the conservative choice that neither widens nor narrows them, so the term only moves rows
where the signal exists. "Missing = widen" is explicitly NOT this arm; it is a separate future
test.

## Metrics, fixed in advance

Stratify the scoring split into the six listing buckets (`1-5 / 6-15 / 16-30 / 31-50 / 51-100 /
>100`, plus a seventh `missing` cell reported but not scored). Per horizon:

1. **Primary — conditional-coverage dispersion.** `D = stdev over buckets of (coverage_b − 0.80)`.
   Lower is better-calibrated. This is the number the arm is judged on.
2. **Secondary — worst-bucket miss.** `max_b |coverage_b − 0.80|`. The thin-tail cell is the one
   the floor doc predicts is under-covered; this must not get worse.
3. **Guardrail — mean width (`rel_width`).** Cohort-mean half-width. Because `c` centres `g`,
   treatment mean width must stay within ±2% of control. A treatment that tightens `D` by
   *inflating* width is a fail, not a pass.
4. **Reported, not scored** — marginal coverage (should be ≈ unchanged; the term is
   width-neutral) and per-bucket `n`.

## The bar, fixed in advance

Per horizon, TREATMENT passes iff **all** hold:
- `D` drops by **≥ 20% relative** vs control, AND
- worst-bucket miss (metric 2) does not increase, AND
- mean width stays within **±2%** of control (guardrail).

Ship decision needs **≥ 2 of 4 horizons** passing with **no horizon regressing** metric 2 by more
than its MDE. A mixed or null read leaves the band unchanged and closes the listing-count thread
for calibration (the floor was already closed).

**Power precondition.** Before scoring, run `compute_mde.py` on `D` and on `rel_width` for the
chosen anchor(s) to confirm the design can detect a 20% move in `D` at the realised per-bucket
`n`. If the thin-tail buckets (`1-5`, `6-15`) are too small at the anchor to resolve their
coverage (rule of thumb `n_b · 0.2 · 0.8 ≥ 10`, i.e. `n_b ≳ 63`), the test is UNDERPOWERED there
and that bucket is declared unresolvable in advance rather than read as a null.

## Void conditions

- Fitted `β_h > 0` at a horizon (wrong sign) → that horizon VOID.
- Anchor fails `replay_serving.py`'s feed audit (leaky/substituted day) → whole run VOID; pick
  another audited anchor.
- Calibration and scoring splits share dates (leakage) → VOID.
- Listing-count series at the anchor is the zero-filled pre-2023-01-25 region for BUFF, or the
  non-Steam feeds are absent at the replay anchor → VOID (no signal to condition on).

## Declared confounds

- **Price tier.** Listing count skews cheap (median kept $16 vs dropped $7). Coverage is
  re-checked **within tier** in a secondary cut; the primary `D` is computed on the pooled cohort
  but the within-tier read must agree in sign or the pooled result is treated as a price proxy.
- **One regime.** The historical replay window is a single regime; a pass here is necessary, not
  sufficient. Confirmation on the live served-outcome panel (below) is required before enabling on
  the daily path.
- **Venue books.** `L` is a max over four non-Steam venues with structurally smaller books than
  the Steam index the ≥30 literature used — the *ordering* is what is being used, not any absolute
  threshold, so this is a scale concern only, noted not corrected.

## Harness — runnable now, plus the live confirmation

**Primary (offline, runnable today):** `scripts/replay_serving.py` at an **audited** historical
anchor (rewind the serving clock, bound the archive at the anchor, resolve the h-day outcome from
`anchor + h`), producing real served-band-vs-outcome rows the band never trained on. Split the
anchor's item universe deterministically (hash of slug) into calibration (fit `β_h`) and scoring
(measure `D`). This sidesteps the fixture problem — the archive, not `cs2_market.db`, is the
source of truth, and `replay_serving` is the same harness that validated the iflow expansion.
Because coverage needs many items per bucket, prefer a recent, feed-audited anchor on the ≥$1
cohort (n≈1k/anchor); if underpowered in the thin buckets, aggregate 2–3 audited anchors and
report per-anchor before pooling.

**Confirmation (live, data-gated):** the served-outcome panel that feeds `served_qhat_multiplier`
reaches its 20-date threshold per horizon around **h30 ≈ 2026-09-06** (append-only store, ~1/day;
see the outcomes-store note). Once ≥ ~20 served dates exist, re-fit `β_h` and re-measure `D` on the
**real** served panel stratified by listing bucket. Enabling on the daily path is gated on this
live read agreeing with the offline one, not on the offline read alone.

## Committed interpretation

- **Offline pass + live pass** → ship `g(L)` at the `m_h` site, guardrailed to ±2% mean width;
  update the band docs; the listing-count thread converts from "closed as a floor" to "shipped as
  a conditioner."
- **Offline pass, live null/underpowered** → hold; do not ship on one regime; revisit after more
  served dates.
- **Offline null** → close the listing-count thread for calibration. The gradient is real on
  exceedance but does not correspond to a coverage defect the band can exploit; the honest reading
  is that the current per-item `sigma` already absorbs most of the listing-count information.

## Result — scored 2026-08-18: NULL (0/3 horizons pass), target population largely absent

Offline replay, anchor **2026-06-16** (h=3/7/14; h=30 spans the 2026-07-09 collector cutover and
was refused by the feed audit), listing `L` = max over the 4 non-Steam venues from the single
**2026-08-06** snapshot (~51d stale — declared confound), deterministic slug-hash cal/score split,
`β_h` grid-fit on cal. Served ≥$1 cohort, score split **n≈523**.

| h | β | dispersion D (ctrl→treat) | worst-bucket miss | mean width | marg cov | verdict |
|---|---|---|---|---|---|---|
| 3 | −0.225 | 0.0375 → 0.0274 (−26.9%) | 0.071 → **0.114** | ×0.995 | 0.853→0.853 | **fail** (guardrail 2) |
| 7 | −0.150 | 0.0259 → **0.0418** (+61.5%) | 0.082 → 0.141 | ×0.990 | 0.855→0.845 | fail |
| 14 | −0.075 | 0.0344 → **0.0439** (+27.5%) | 0.053 → 0.066 | ×0.991 | 0.767→0.761 | fail |

**0 of 3 horizons pass.** h=3 improves the primary dispersion metric but worsens the worst-bucket
miss (guardrail 2 fail); h=7/14 worsen dispersion — the cal-fit β does not generalize to the score
split (overfit).

**The decisive finding is structural, not the metric.** The thin buckets (`1-5`, `6-15`) — the
entire hypothesis's target — have **< MIN_BUCKET_N items in the served ≥$1 cohort** and dropped out
at every horizon; the surviving buckets are `16-30 / 31-50 / 51-100 / 101+`, with `101+` holding
372 of 523 rows. Among those liquid buckets control coverage is **already flat** (0.77–0.88), so
there is no listing-ordered coverage defect left to condition on. Reading per the committed
interpretation and the floor doc's caveat #1: the thin tail is **unresolvable** here (underpowered,
not null), and among the resolvable liquid cohort the conditioner has **nothing to fix** — the
per-item `sigma` already carries the listing-count information for the population actually served.

**Decision:** do NOT ship. This is one stale-snapshot anchor (n≈523), so not a hard kill, but the
target population (thin, ≥$1) barely exists in the served set — a gap the live served-panel
confirmation cannot close. **Listing count is closed for band-width calibration** the same way it
was closed as a floor; the surviving band-calibration lever is the dormant served-outcome `q_hat`
feedback (self-activating ~2026-09-06), not a listing-count term. Reopen only if the served ≥$1
universe expansion (iflow) materially grows the thin-listing cohort — re-check bucket `n` there
before spending another run.
