# Volume feed repair PASSES band quality at h=14 on a re-folded scheme

*2026-08-17. Follows `2026-08-17-volume-band-quality-cluster-starved.md`, which found the same
effect but at 8 folds < the preregistered ≥14 floor.*

Re-folded the volume band-quality A/B under a fresh prereg
(`docs/research/2026-08-17-volume-band-quality-refold-preregistration.md`) to reach the cluster
floor honestly, then re-scored. **The h=14 band-tightening confirms and passes every gate.**

## The re-fold (fold-scheme params, not a rewrite of the bar)
Exposed the walk-forward scheme as run params in `scripts/ab_test_volume_features.py`
(`SPLIT_FRACTION`, `STEP_DAYS`, `VAL_WINDOW_DAYS`; defaults reproduce the old hardcoded
2/3 · 60 · 21 scheme exactly). The 8-fold starvation was an artifact of `split_idx = 2/3`, which
validates only the last third of the era. New scheme: **`SPLIT_FRACTION=0.50 STEP_DAYS=45`** →
**16 folds**. The `45 − 21 = 24`-day gap between validation windows exceeds h=14, so folds stay
disjoint and label-independent — resolution recovered without overlapping windows.

## Result (iflow era 2022–2026, ≥$1 cohort, join 80.9%, 16 clusters)

| h | role | rel_width (lower=better) | coverage Δ | placebo width | verdict |
|---|---|---|---|---|---|
| **14** | primary | **−0.233 [−0.370, −0.104]** | null (holds ~81.3%) | null +0.015 | **PASS** |
| 3 | secondary | null +0.097 | over-covers +0.68 | null | no effect |
| 7 | secondary | null −0.095 [−0.285, +0.074] | null | null | no effect (trends tighter) |

Width is monotone in horizon (+0.097 → −0.095 → −0.233); only h=14 clears significance.

**Gating robustness — both survive:**
- **Leave-one-fold-out:** worst drop → −0.197 [−0.320, −0.075], CI excludes 0.
- **Leave-out-2025-10:** −0.251 [−0.382, −0.115] — *strengthens* without the spike regime, so
  the effect is not a 2025-10 artifact (unlike the volume-spike exceedance feature).

DA at h=14 is +1.37pp (diagnostic only, invariant 4). The tightening is a genuine band-quality
effect, present only at the long horizon.

## What this earns, and what it does not
Per the prereg's committed interpretation: **a build, not an automatic ship.** Volume becomes a
candidate **band-width conditioner** at h=14, and the served/conformal path can be wired. But this
is a power re-scope on the *same* 2022–2026 era, not an independent-data replication — iflow ends
2026-05-20 and no holdout exists. The claim is *robust on this era*, not *validated out of sample*.
Ship only after the served-band build and, ideally, once fresh data extends the panel.

The wiring (`IFLOW_VOLUME`, iflow-era restriction, `rel_width`/`in_interval` endpoints, fold
params, `_width_robustness`) all live behind flags whose defaults preserve the original harness
behavior.
