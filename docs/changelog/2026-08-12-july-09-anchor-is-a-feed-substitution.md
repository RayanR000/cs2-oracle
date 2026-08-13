# The 2026-07-09 anchor is a feed substitution, not a market date

**Date:** 2026-08-12
**Answers:** the open thread in `2026-08-12-learned-band-scale-measured.md` — "four horizons all
reversing on one date is a property of that date, not of the model".
**Instruments:** the archive (DuckDB, `db/archive.py::prices_relation`) and the stored logs of runs
`31649561391` (control) / `31649571169` (`learned_scale=true`), commit `5b39760`.
**Status:** 🔑 diagnosed. `2026-07-09` must leave the replay anchor set. **No published verdict
flips** — both band flags stay off — but three published figures move, and one of them changes sign.

## The cause

On every ordinary day of that week the archive holds one source, `aggregator_steam_17mafo`, at
~26,170 items and a median price of $2.97–3.00. **On 2026-07-09 that source is absent.** The only
source that day is `aggregator_sync`, at 5,502 items and a median of $0.13 archive-wide.

It is the **only day in 2026** where the day's source set differs from its neighbours' while the
collector still reported. (The four whole-day gaps — 07-27, 07-30, 08-02, 08-03 — are absences, not
substitutions; 03-22–03-25 and 07-11–07-16 are the two documented regime changes, where the item
count steps by 9.3× and 3.9× respectively.)

The substitute feed still reaches the cohort production serves — **1,068 of 1,084** served ≥$1 items
get a quote — but at a **different basis**:

| ratio, per item, served ≥$1 cohort | median | p10 | p90 | share > +10% |
|---|---:|---:|---:|---:|
| 07-09 / 07-08 — the substitute steps **in** | **1.287** | 1.046 | 1.414 | 85.3% |
| 07-10 / 07-09 — and steps back **out** | **0.773** | 0.710 | 0.952 | 1.5% |
| 07-10 / 07-08 — the two ordinary days | **1.000** | 0.979 | 1.016 | 1.5% |

A synthetic **+28.7% one-day spike that fully reverts the next day**, on 85% of the cohort, with the
two days either side of it agreeing to within 2%. Nothing happened in the market.

## What it did to the replay

Everything the anchor touches inherits the spike. Per-anchor, from the two stored runs:

| anchor | tied | down% 3/7/14/30 | DA% 3/7/14/30 |
|---|---:|---|---|
| 2026-04-15 | 301/1100 (27.4%) | 61.6 / 60.6 / 68.5 / 71.9 | 49.3 / 52.5 / 57.6 / 73.0 |
| 2026-05-16 | 598/1061 (56.4%) | 61.5 / 60.9 / 80.6 / 49.7 | 45.3 / 45.0 / 47.1 / 45.5 |
| 2026-06-16 | 445/1071 (41.6%) | 57.2 / 69.1 / 78.2 / 92.2 | 41.1 / 51.4 / 53.2 / 52.9 |
| **2026-07-09** | **26/1032 (2.5%)** | **77.1 / 85.2 / 87.4 / 86.1** | **63.2 / 66.2 / 69.7 / 69.3** |

- **The tied share collapses** to 2.5%, an order of magnitude below the others. An item is "tied" when
  its anchor-day quote equals its own local median; a cohort-wide +28.7% displacement makes almost
  nobody tied. This statistic was flagged as an outlier **twice** —
  `2026-08-11-clean-anchor-signal-replicates.md` ("Something about that date's collection") and
  `2026-08-11-clean-anchor-confirmed-in-ci.md` — and never diagnosed. It was the symptom.
- **The down-rate is the highest of the four anchors at 3 of 4 horizons**, because returns are
  measured *from* the inflated anchor. The decline is arithmetic, not market.
- **DA is also the highest of the four** (63–70% against 41–73% elsewhere). The model calls down into
  a guaranteed decline. That is not skill and it must not be read as any.
- **The band's misses flip sides.** At h=3 the control misses **21.71% above** the band against
  12.60% below — the only such cell in eight. Everywhere else the misses are predominantly below.
- **The learned scale widens the band there while narrowing it everywhere else** — served half-width
  15.77 / 18.42 / 26.23 / 29.18% against the control's 8.22 / 12.11 / 16.96 / 25.64%. This is the
  reversal, and the mechanism is direct: a +28.7% anchor-day move enters the volatility features the
  scale model reads, so it predicts a large residual for the whole cohort at once. The fixed `sigma`
  band, which is not a function of the anchor day's own jump to nearly the same degree, does not move.

## What moves when it is dropped

Means over the four anchors as published, against the three ordinary ones:

| h | arm | cov% 4a → 3a | ramp 4a → 3a | halfw% 4a → 3a |
|---|---|---|---|---|
| 3 | control | 77.80 → **81.83** | +15.66 → **+8.81** | 9.80 → 10.33 |
| 3 | learned | 72.31 → 68.97 | −10.02 → **−22.69** | 8.63 → 6.26 |
| 7 | control | 84.94 → 87.80 | +9.94 → +3.60 | 14.56 → 15.38 |
| 7 | learned | 78.88 → 77.30 | −11.41 → **−23.26** | 11.87 → 9.68 |
| 14 | control | 84.76 → 85.94 | +8.10 → +9.86 | 20.87 → 22.17 |
| 14 | learned | 76.63 → 73.91 | **+0.65 → −9.59** | 17.07 → 14.01 |
| 30 | control | 87.07 → 88.71 | +9.13 → +11.55 | 31.32 → 33.22 |
| 30 | learned | 85.68 → 86.75 | +7.19 → +0.75 | 24.32 → 22.70 |

Three corrections to the record, and one of them is a sign:

1. ⚠️ **`sigma` over-covers at 4 of 4 horizons, not 3 of 4.** The published served marginal of
   **77.9%** at h=3 — the one figure that read as *under*-coverage and made h=3 "a regression" for the
   exponent arm — is **81.83%** without 07-09. `sigma`'s displacement is one-directional:
   81.8 / 87.8 / 85.9 / 88.7 against an 80% target.
2. ⚠️ **The learned scale's "flat" 14d ramp of +0.65 is gone**, as that entry already warned: it is
   **−9.59**, and the arm overcorrects at 3 of 4 horizons (−22.69 / −23.26 / −9.59 / +0.75). 30d
   remains its one genuine success.
3. ✅ **The control's tilt survives.** The ramp stays positive at 4 of 4 (+8.81 / +3.60 / +9.86 /
   +11.55), so `2026-08-12-served-sigma-profile.md`'s central finding — the band *is* tilted in
   `sigma` where it is served — does not depend on the bad anchor. Neither does the conclusion that
   the width variable is not the lever: three scales still calibrate to exactly 80% and land at
   81.8 / 87.8 / 85.9 / 88.7 (`sigma`) and 69.0 / 77.3 / 73.9 / 86.8 (learned) when served.

**Both flags stay off.** Dropping 07-09 strengthens the case against `LEARNED_SCALE` rather than
weakening it.

## Scope of the contamination

Every read that used `replay_anchors=2026-04-15,2026-05-16,2026-06-16,2026-07-09` carries one
contaminated anchor in four: `2026-08-12-learned-band-scale-measured.md`,
`2026-08-12-served-sigma-profile.md`, `2026-08-12-sigma-exponent-paired-read.md`,
`2026-08-11-label-smoothed-anchor.md` and `2026-08-11-smoothed-anchor-label-measured.md` (whose own
text notes 07-09 "carries a disproportionate share of the effect" — now explained), plus the
ten-anchor `2026-08-11-clean-anchor-signal-replicates.md`.

Two things it does **not** contaminate. The `sigma`-tilt panel of
`2026-08-12-the-band-is-tilted-in-sigma.md` and the marginal-coverage attribution merely *end* on
2026-07-09 — one date of 731, immaterial to a pooled elasticity. And no production forecast was made
on 07-09; the daily chain writes no row for a day the aggregator did not deliver.

## The guard found a second bad anchor: 2026-04-15

Written as `audit_anchor_feed` (below) and run over the archive, the check refuses **12 of the 187
dates in 2026 with a resolvable h=30** — and **two of the four published anchors are among them.**

`2026-04-15` is the **last day of the three-source era**: 04-08–04-15 hold
`buff163 + csfloat + youpin` at ~32,420 items, and from **04-16** the archive holds
`steam_17mafo` alone at ~25,010. So the anchor is quoted on one collection and **every one of its
horizons resolves on another**. The level effect is far milder than 07-09's — median 0.979 across the
boundary — but the cross-section is scrambled: **30.1% of served items move more than ±10%**, against
3.9–7.0% on an ordinary adjacent pair.

That leaves **two** clean anchors of the four, which is what the band series' "four-anchor mean"
actually rested on:

| h | arm | cov% 4a | cov% **2 clean** | ramp 4a | ramp **2 clean** |
|---|---|---:|---:|---:|---:|
| 3 | control | 77.80 | **85.78** | +15.66 | +11.18 |
| 3 | learned | 72.31 | 78.54 | −10.02 | **−25.18** |
| 7 | control | 84.94 | **90.20** | +9.94 | +6.77 |
| 7 | learned | 78.88 | 83.86 | −11.41 | −16.93 |
| 14 | control | 84.76 | 84.14 | +8.10 | +10.24 |
| 14 | learned | 76.63 | 74.05 | +0.65 | **−12.34** |
| 30 | control | 87.07 | 86.24 | +9.13 | +12.09 |
| 30 | learned | 85.68 | 86.67 | +7.19 | +1.58 |

Same three conclusions, all of them stronger. The over-coverage is **larger** than published
(85.8 / 90.2 / 84.1 / 86.2 against 80%, i.e. **+4.1 to +10.2pp**, where the published set read
−2.2 to +7.1pp); the control's tilt is positive at 4 of 4 and bigger; and the learned scale
overcorrects at 3 of 4 rather than looking flat at 14d. **Both flags stay off, and the level of the
over-coverage is now the only open quantity.**

**Where the clean anchors are.** 175 of 187 dates pass, in four stretches:
`2026-01-01..03-19`, `03-24..03-27`, `04-01..04-13`, `04-18..07-06`. (July 07-09 onward is excluded
from the pool by the h=30 requirement rather than by the audit — the archive ends 08-08.) For the
next dispatch: **`2026-04-22,2026-05-16,2026-06-16,2026-07-06`** — four audited dates, ~30 days
apart, keeping the two anchors that were already clean so the new read is comparable to the old one.

## A smaller, separate basis step

While measuring the above: the single-feed → 11-source transition on 07-11–07-14 moves the served
cohort by a median **0.964×** (p10 0.855), so a replay whose *outcome* window crosses it resolves at
a slightly different basis than its anchor. `2026-06-16` at h=30 is exactly that case — its outcome
lands 07-16 — and it prints the highest down-rate in the table at 92.18%. An order of magnitude
smaller than 07-09's 28.7% and not a reversal, but it is the same class of defect and it is
un-instrumented.

## The guard, as shipped

`audit_anchor_feed` and `_feed_profile` in `scripts/replay_serving.py`, called from `main()` **before
the artifact loads** — a bad anchor otherwise costs ~4 minutes of replay to find out. Ten tests in
`tests/test_serving_replay.py`.

- **Two columns, one aggregate query**, through `prices_relation` and `archive_universe_sql_filter`
  (`backend/AGENTS.md` invariants 1 and 2): which sources wrote each day within ±3, and how many
  items each covered. ±3 rather than ±1 because the archive drops whole calendar days and a tighter
  window can find nothing to compare against.
- **The test is about collection, never about prices.** A level check would have to pick a
  denominator, and every denominator available here is one of the arms — the trap that cost two
  cancelled dispatches on 2026-08-12. Which feeds wrote a day is a property of the archive alone.
- **The two sides are checked separately, because they fail differently.** Differing from the days
  *before* means the anchor's own features read a synthetic jump across the change (2026-03-22's
  shape). Differing from the days *after* means the outcome resolves on a basis the anchor was never
  quoted on (2026-04-15's). 2026-07-09 is both at once, which is why it reversed four horizons
  instead of tilting them.
- **The reference is the modal neighbour set, not the union or the intersection.** The union widens by
  any single neighbour inside a transition and calls an ordinary anchor deficient; the intersection
  lets a substitution through as a subset.
- **It fails closed** — no neighbour, no rows, or a dropped calendar day all refuse, with the dropped
  day named as one rather than reported as a substitution.
- `ALLOW_DIRTY_ANCHOR=1` proceeds anyway, printing the audit as a WARNING plus an explicit "these
  numbers describe the archive's collection, not the model". Replaying 07-09 deliberately is a
  diagnosis, and the flag keeps that possible without keeping it quiet.

**What it does not do.** The window is local. An anchor 30 days before a collection change still
resolves its h=30 outcome across it — §"A smaller, separate basis step" above, where `2026-06-16` at
h=30 crosses the 07-11 transition and prints a 92.18% down-rate — and no ±3 check can see that. The
tied share reported per anchor remains the downstream symptom to read.

`model-diagnostics.yml`'s `replay_anchors` input carries the same warning and the replacement set, so
the next person to dispatch it does not have to find this entry first.
