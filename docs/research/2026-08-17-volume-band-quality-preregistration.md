# Pre-registered read: does the repaired volume feed improve band quality (not DA)?

**Written 2026-08-17 before the A/B is run on the iflow-repaired volume feed.** No coverage or
width delta for these arms has been scored. The bar, the placebo, the gate, and the void conditions
below are fixed here and bind whatever the numbers turn out to be. This doc exists because the
volume lead's only prior evidence is **directional accuracy**, which `backend/AGENTS.md` invariant 4
forbids as a shippable claim — so DA is demoted to a reported diagnostic here and **band quality is
the sole gate**.

## Hypothesis and provenance

The 13 shelved volume features (`NEW_PRIMITIVES` in `scripts/ab_test_volume_features.py`) measured
**+1.50 / +1.88 / +1.44 pp DA** at 3/7/14d, placebo-clean at 4/4 horizons
(`changelog/2026-08-15-volume-features-remeasured.md`), but on a **dead Kaggle feed** that goes
identically 0 from 2026-07 on. That is a train/serve gap, not a lack of signal. The iflow backfill
(`2026-08-16-refutation-power-tiers-and-iflow-backfill.md`) supplies a real 24h trade-count series
(`count_in_24`, stored as `steam_volume` in `buff-iflow-staging/price-archive/iflow-liquidity-*.parquet`,
2022-04→2026-05, already durable on data-repo branch `buff-iflow-backfill`). This test asks the one
question that can ship:

> **If the volume feed is repaired from `count_in_24`, do the volume features tighten the served
> interval without breaking coverage — i.e. earn a place on band quality, not DA?**

**Standing caveat, stated up front so a null is a real close.** Volume is *linearly* refuted at
|r| < 0.002 across 4.47M rows (Tier A, well-powered, `changelog/2026-08-08-volume-null-not-zero-and-skinport-sales.md`).
This test only survives as a **nonlinear/conditional** effect on interval width. A null here retires
the volume lead as *well-powered*, not underpowered.

## Why this is a real test and not a re-read

- The volume column was 0 in every late fold of the shelved harness; here it is real through
  2026-05-20. `VOLUME_LIVE_THROUGH` is raised `2026-04-30 → 2026-05-20` and the panel is extended to
  2022–2026, multiplying non-overlapping episodes ~10× — the power every prior directional volume
  test lacked.
- The endpoint is switched from DA to **interval width subject to coverage**, which no prior volume
  run scored. A DA-positive/width-null outcome is therefore *new information*, not a replication.

## Arms, fixed in advance

Three arms on the **same** walk-forward folds, same universe, same embargo
(`_purge_overlapping_train_rows`, `embargo_days(h) = h + 13`), same `MIN_MEDIAN_PRICE = 1.0` cohort
(the production ≥$1 cohort; sub-$1 flat-return contamination is excluded by construction):

- **baseline** — drop the 13 volume columns.
- **treatment** — all features including the 13 volume columns, sourced from
  `iflow-liquidity-*.parquet` (`steam_volume` → the archive `volume` column the feature engineering
  reads).
- **placebo** — the 13 volume columns column-shuffled within fold (capacity-inflation guard).

Horizons: 3, 7, 14. (h=30 excluded: the 21-day validation window empties under embargo at this
panel depth; adding it changes power, so it is out by rule, not by result.)

## Metrics, fixed in advance

Per validation row the harness already emits interval endpoints (`walkforward_records.py:224`). Two
paired metrics are added to the existing `paired_records` call and contrasted with
`paired_arm_contrasts` (fold-clustered, `cluster_key="fold_id"`):

- **`rel_width`** = `(high − low) / anchor_price` — interval sharpness. **Lower is better**
  (`higher_is_better=False`).
- **`in_interval`** = `1[low ≤ actual ≤ high]` — realised coverage against `NOMINAL_COVERAGE = 0.80`.

**`direction_correct`** is also emitted but is **diagnostic only** — reported beside
`realised_down_rate` per invariant 4, never differenced against `constant_call_accuracy`, and it
does **not** gate.

## The bar, fixed in advance

**Primary (must pass to ship).** On `rel_width`, treatment vs baseline, the paired fold-clustered
mean difference is **negative** (tighter) and its 95% CI **excludes 0**.

**Coverage gate (must also hold).** Treatment realised coverage (mean `in_interval`) must be
**≥ 0.80 at every scored horizon** — width bought by dropping below nominal is disqualified, not
credited. Equivalently, the `in_interval` treatment−baseline CI must not sit entirely below the
level that would carry coverage under 0.80.

**Placebo gate (must also hold).** The identical `rel_width` contrast on the **placebo** arm must be
**null** — CI includes 0, *or* |mean diff| < ½ the treatment effect. A significant placebo width
gain is capacity inflation and voids the read.

**Power precondition.** Before scoring, `compute_mde.py` is run on `rel_width` to confirm the design
can resolve the target effect (the price arm's MDE machinery reported 0.056pp on its endpoint; the
width MDE is computed fresh here). If MDE > the smallest width delta worth shipping, the result is
reported as **underpowered**, not as a null.

**Ship** ⇔ primary passes **and** coverage gate holds **and** placebo null **and** powered.
Anything else = **do not ship**; DA movement alone never overrides this (invariant 4).

## Void conditions

- Any arm shares no rows with baseline → `no_shared_rows`, reported not swallowed.
- `steam_volume`→`volume` join coverage on the ≥$1 cohort < 80% of item-days → the feed is not
  actually repaired for this cohort; report join coverage and **void**, do not score.
- Fewer than ~14 non-overlapping fold clusters survive at a horizon → that horizon is dropped for
  lack of clustered power, reported explicitly.

## Declared confounds

- **Survivorship.** The iflow cohort is liquidity-selected — history exists only for items liquid at
  the time. Width gains concentrated in the always-liquid subset may not transfer to thin items;
  reported as a within-cohort split, never hidden in the pooled number.
- **Regime.** 2022–2026 spans the 2023 Steam economy and the 2025-10 volume spike. A width gain that
  is entirely one regime is noted (the volume-spike exceedance feature already died this way —
  `2026-08-16-wash-trade-screen-and-volume-spike-exceedance.md`).

## Committed interpretation

- **Pass** → the volume features ship as **band-width conditioners** (the shippable axis), and the
  volume feed repair (`count_in_24`) is promoted from staging to durable ingest. DA is mentioned as
  corroboration, not as the claim.
- **Fail, powered** → the volume lead is retired as a **well-powered null on band quality**,
  upgrading the prior underpowered/dead-feed status. The linear null (|r|<0.002) then stands as the
  whole story and no nonlinear rescue remains open on this data.
- **Underpowered** → not a null; the panel depth or clustering is the limit, recorded as such.

## Query / harness

`scripts/ab_test_volume_features.py` with volume sourced from `iflow-liquidity-*.parquet`,
`VOLUME_LIVE_THROUGH=2026-05-20`, endpoints `rel_width` + `in_interval` added to the paired
contrast. Power check: `scripts/compute_mde.py` on `rel_width`. Read-only against the staging
archive; ships to durable only on a full pass.

## Result — scored 2026-08-17

Run on the iflow era 2022-04-18 → 2026-05-20 (285,752 price rows, ≥$1 cohort, join
coverage **80.9%** — clears the void gate; absolute interval coverage ~80–81%). **8
non-overlapping fold clusters at every horizon.**

| h | rel_width (ship gate, lower better) | coverage Δ | placebo width | DA (diagnostic) |
|---|---|---|---|---|
| 3 | null +0.048 [−0.174, +0.221] | null | clean | +1.41pp positive |
| 7 | null −0.076 [−0.259, +0.097] | null | +0.027 [+0.004,+0.049], < ½ treatment | +1.77pp positive |
| 14 | **positive −0.204 [−0.351, −0.062]** | null (holds) | null −0.021 | +1.32pp null |

**Verdict: no clean ship.** The ship gate (width tighter, CI excludes 0, coverage held,
placebo null) is met **only at h=14** — where, notably, DA also goes null, so it is a genuine
band-quality effect and not a DA artifact. Width point estimates trend tighter with horizon
(+0.048 → −0.076 → −0.204).

**But the preregistered ≥14-cluster power floor fails at every horizon (8 < 14).** Restricting
to the iflow era to make the volume feature live everywhere costs episodes (2013–2026 gave 26
folds; 2022–2026 gives 8). By the void condition fixed above, all three horizons are **dropped
for lack of clustered power** — including the h=14 pass. The floor is **not** rewritten post-hoc.

**Interpretation.** The volume→band-quality lead is **neither confirmed nor cleanly refuted** on
this data — the iflow era is cluster-starved at these horizons. The h=14 tightening is
real-shaped and worth remembering, but it is a single horizon at the power boundary. A clean
verdict needs either more history (none exists — iflow starts 2022-04) or a re-scoped fold
scheme, which requires a fresh preregistration. Do **not** ship on the h=14 result alone.

**Correction to the prereg's premise.** The "~10× episodes" claim was directionally wrong: the
iflow era is *shorter* than the existing 2013–2026 panel, so it yields *fewer* non-overlapping
folds, not more. The ~10× held only against a 2026-only baseline, which this harness never used.
