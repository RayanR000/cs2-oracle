# Pre-registered read #2: re-fold the volume band-quality test to reach the ≥14-cluster floor

> **Status (as of 2026-08-21): SCORED 2026-08-17 — PASSES at h=14, and then the thread CLOSED
> anyway.** `changelog/2026-08-17-volume-band-quality-passes-on-refold.md`: `rel_width` −0.233
> [−0.370, −0.104] at h=14, coverage holds ~81%, placebo null, both robustness checks survive
> (the effect *strengthens* without the 2025-10 spike regime); null at h=3/h=7.
> **But the pass is on the retired q10/q90 band, not the served one.** Volume routed into the
> production learned scale is **net-negative** — it degrades the dominant `sigma`-tilt axis —
> so the served-band question closed the same day
> (`changelog/2026-08-17-volume-in-scale-is-net-negative.md`). **Do not build the ingest
> integration on the strength of the pass above.**


**Written 2026-08-17, after the first run
(`2026-08-17-volume-band-quality-preregistration.md`) came back cluster-starved — 8 folds at
every horizon, below the ≥14 floor that prereg fixed. No re-folded result has been scored.** The
fold scheme, the bar, the placebo, the horizon roles, and the robustness checks below are fixed
here and bind whatever the numbers turn out to be. This supersedes **only the fold-scheme and
power clauses** of prereg #1; its endpoint, arms, cohort, and shippability logic are inherited
unchanged.

## What run #1 found, and why a re-fold (not a rewrite)

Run #1 scored the iflow-repaired volume feed on band quality. Result: `rel_width` tightened
**only at h=14** (−0.204 [−0.351, −0.062], coverage held ~81%, placebo clean, DA null there — a
genuine band-quality effect, not a DA artifact); null at h=3/h=7. **Every horizon had 8
non-overlapping folds, < the preregistered 14**, so prereg #1's own void condition dropped all
three horizons, including the h=14 pass. The floor was not rewritten post-hoc.

The 8-fold starvation is a fold-scheme artifact, not a data limit: the harness sets
`split_idx = 2/3 · len(dates)`, so folds cover only the **last third** of the era. The iflow era
holds **1,494 distinct dates** (2022-04-18 → 2026-05-20); two-thirds of them are spent as pure
initial training and never validated. Starting validation earlier recovers folds **without**
slicing windows thinner or overlapping them.

## The new fold scheme, fixed in advance

Expanding-window walk-forward, same as production's grain:

- `split_fraction = 0.50` — validation begins at the era midpoint (initial train ≈ 747 dates,
  ~2 years; each later fold's train expands). Was 0.667.
- `VAL_WINDOW_DAYS = 21` — **unchanged** from run #1.
- `step = 45` — was 60. The gap between consecutive validation windows is `45 − 21 = 24` days,
  which **exceeds the longest tested horizon (14d)** plus margin, so a window's forward-resolved
  labels cannot bleed into the next window. Folds stay mutually disjoint *and* label-independent.
- Train embargo unchanged: `_purge_overlapping_train_rows`, `embargo_days(h) = h + 13`.

**Expected folds ≈ 17** (`(1494 − 747)/45`). The scheme is fixed by these three numbers; the fold
count is whatever they produce, not a target to tune toward. If the realized count is < 14 the
run is **void** (same floor as prereg #1).

## What the re-fold does and does not buy — stated up front

**It buys resolution, not new regimes.** Denser folds sample the *same* 2022–2026 era more
finely; they do not add independent market episodes. The AGENTS.md constraint — the binding
limit is the count of *independent* episodes, not fold count — still holds. The honest ruler
remains the **fold-clustered CI**, which already widens for within-fold correlation; I will not
claim "well-powered" beyond "meets the preregistered cluster count, read the cluster-robust CI."

**It is not an independent replication.** iflow ends 2026-05-20; no fresh holdout exists. Having
seen h=14 win in run #1, re-testing h=14 on denser folds of the same data is a
garden-of-forking-paths risk. The mitigations are the pre-declared horizon roles and the
robustness checks below; a "pass" here **strengthens** the run-#1 signal, it does not fully
validate it absent new data. This limitation is part of the committed interpretation, not a
footnote.

## Horizon roles, fixed in advance (anti-forking)

- **h=14 is the single primary horizon** — declared here *because* run #1 flagged it, so the
  re-fold cannot be accused of re-picking the winner after the fact.
- **h=3 and h=7 are secondary**, reported for the gradient (run #1 showed width trending tighter
  with horizon: +0.048 → −0.076 → −0.204). A ship claim may **not** rest on newly promoting a
  secondary horizon.

## The bar, inherited and fixed

Ship ⇔ **all** hold at h=14:

1. **Primary — `rel_width`:** treatment−baseline paired fold-clustered mean **negative**, 95% CI
   **excludes 0**.
2. **Coverage gate — `in_interval`:** treatment coverage does **not** fall below baseline
   coverage (Δ CI not entirely negative); absolute coverage stays ≥ ~nominal on this harness's
   uncalibrated band (~0.80±).
3. **Placebo gate:** the same `rel_width` contrast on the shuffled arm is **null** — CI includes
   0, *or* |placebo| < ½ |treatment|.
4. **Power:** realized clusters ≥ 14.

## Robustness, fixed in advance (gating, not decorative)

The h=14 effect must be **broad**, not carried by a few folds:

- **Leave-one-fold-out:** dropping any single fold must not flip the `rel_width` CI to include 0.
  Report the min-across-drops CI.
- **Leave-out-2025-10:** drop every fold whose validation window intersects 2025-10 (the volume
  spike that reduced the exceedance feature to one episode,
  `2026-08-16-wash-trade-screen-and-volume-spike-exceedance.md`). If the width effect vanishes
  without those folds, it is one regime, not signal → **do not ship**, regardless of the pooled
  CI.

## Void conditions (carried from prereg #1)

- iflow→`volume` join coverage on the ≥$1 era cohort < 80% → void.
- Realized fold clusters < 14 → void.
- Any arm shares no rows with baseline → `no_shared_rows`, reported not swallowed.

## Committed interpretation

- **Pass (all gates + both robustness checks)** → the h=14 volume band-tightening is **robust on
  the iflow era**. Promote volume to a candidate **band-width conditioner** and build the served
  path (conformal/served-band wiring) — earns a build, not an automatic ship, since this is not
  an independent-data replication.
- **Fail** → the volume → band-quality lead is **retired as this era's verdict**: no shippable
  band effect survives a properly-powered fold scheme. The linear null (|r|<0.002) then stands as
  the whole story on the data that exists.
- **Void** → the era cannot answer even re-folded; record and stop, do not slice thinner (step <
  `VAL_WINDOW_DAYS + horizon` would overlap labels and is off the table).

## Query / harness

`scripts/archive/ab_test_volume_features.py` with `IFLOW_VOLUME=1` and the new fold scheme exposed as
run params (`SPLIT_FRACTION=0.50`, `STEP_DAYS=45`, `VAL_WINDOW_DAYS=21`), fingerprinted into the
frame/verdict cache. Endpoints and arms unchanged from prereg #1. Read-only against the staging
archive; ships to durable only on a full pass, and even then only after the served-band build.

## Result — scored 2026-08-17

Re-fold `SPLIT_FRACTION=0.50 STEP_DAYS=45 VAL_WINDOW_DAYS=21`, iflow era, ≥$1 cohort, join
coverage 80.9%. **16 fold clusters at every horizon (≥14 floor cleared).**

| h | role | rel_width (ship gate) | coverage Δ | placebo width | verdict |
|---|---|---|---|---|---|
| **14** | **primary** | **−0.233 [−0.370, −0.104]** | null −0.184 (holds ~81.3%) | null +0.015 | **PASS** |
| 3 | secondary | null +0.097 [−0.086, +0.224] | +0.68 (over-covers) | null | no effect |
| 7 | secondary | null −0.095 [−0.285, +0.074] | null +0.33 | null −0.001 | no effect (trends tighter) |

The width point estimates are monotone in horizon — **+0.097 → −0.095 → −0.233** — with only the
primary h=14 clearing significance, exactly the gradient run #1 flagged.

**Primary horizon h=14 passes all four ship gates**, and both **gating robustness checks
SURVIVE**:

- **Leave-one-fold-out:** worst drop (fold 1299) → −0.197 [−0.320, −0.075] — CI still excludes 0.
- **Leave-out-2025-10:** dropped 2,427 spike rows → −0.251 [−0.382, −0.115] — the effect
  *strengthens* without the spike regime, so it is not a 2025-10 artifact.

DA is +1.37pp (diagnostic, positive) at h=14 but does not gate. The width gradient run #1 flagged
holds: null at short horizons, tightening at h=14. This **confirms run #1's h=14 tightening on a
properly-powered fold scheme.**

**Verdict: PASS on the iflow era.** Per the committed interpretation, this earns a **build** —
promote volume to a candidate band-width conditioner and wire the served/conformal path — **not
an automatic ship**, because this is a power re-scope on the same 2022–2026 era, not an
independent-data replication (iflow ends 2026-05; no holdout exists). The claim is: *the h=14
volume band-tightening is robust on this era*, not *validated out of sample*.

## Erratum (2026-09-30, path-only)

The 2026-09-22 cleanup moved the scripts this document cites into `backend/scripts/archive/`. The cited paths were updated in place to match. No design, threshold, bar or result text was touched; the original paths remain in the file history.
