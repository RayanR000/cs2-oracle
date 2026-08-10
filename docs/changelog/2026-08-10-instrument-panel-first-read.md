# The first read of the three gated instruments, 2026-08-10

Four arms, one commit, one basis. **The cross-sectional rank transform (`C1`) clears the
`−return_1d` bar at all four horizons. `init_score` (`N1`) does not, and as shipped it cannot
move the served signal at all.** That inverts the ranking in
`docs/research/2026-08-10-next-steps.md`, which put Track N first and carried `C1` forward
below it.

## What was run

| Arm | Run | Commit |
|---|---|---|
| control | `31430866207` | `b610f9c` |
| `xs_rank` (`CROSS_SECTIONAL_RANK=1`) | `31430874845` | `b610f9c` |
| `naive_init` (`NAIVE_INIT_SCORE=1`) | `31430882892` | `b610f9c` |
| `tier_lead` (`TIER_LEAD_FEATURE=1`) | `31425490196` | `c9a5b1e` |

`model-diagnostics.yml`, `horizons=matrix`, one job per horizon, ~9 min per run. No arm can
promote an artifact: the workflow restores the `forecast-models-` cache and deliberately never
saves to it.

**The `tier_lead` arm is on the earlier commit and is still comparable**, because the control
reproduced **bit-for-bit** across the two commits — 0.1774 / 0.1267 / 0.1023 / 0.0967 rank IC at
3/7/14/30d, identical to `c9a5b1e`'s control at every digit. That reproducibility is the control
the single-dispatch design rests on (`2026-08-10-served-classifier-scored.md` §1), and it held
across two commits and two hours. The two commits between them (`50c5105`, `b610f9c`) are gated
off, and the panel confirms they are inert when off.

## Rank IC edge vs `−return_1d`

The bar is `rank_ic_edge_vs_naive ≥ 0`. Positive means the model beats the one-line baseline.

| Arm | 3d | 7d | 14d | 30d |
|---|---|---|---|---|
| control | −0.0159 | −0.0371 | −0.0433 | −0.0091 |
| **`xs_rank`** | **+0.0556** | **+0.0561** | **+0.0371** | **+0.0316** |
| `naive_init` | +0.0090 | −0.0115 | +0.0131 | −0.0167 |
| `tier_lead` | −0.0081 | −0.0276 | −0.0422 | −0.0099 |

Absolute rank IC, same folds (`naive` is the same number in every row by construction —
0.1933 / 0.1638 / 0.1456 / 0.1058):

| Arm | 3d | 7d | 14d | 30d |
|---|---|---|---|---|
| control | 0.1774 | 0.1267 | 0.1023 | 0.0967 |
| **`xs_rank`** | **0.2489** | **0.2199** | **0.1827** | **0.1374** |
| `naive_init` | 0.2023 | 0.1523 | 0.1587 | 0.0891 |

`xs_rank` adds +0.0715 / +0.0932 / +0.0804 / +0.0407 rank IC over the control. It is the largest
single-arm movement measured on this metric in the project.

## Served signal (`Invariant #4`, SERVED classifier)

Read `PT excess` and its `t`, not `edge vs constant call` — that baseline is hindsight-selected
(`2026-08-10-constant-call-is-hindsight-picked.md`).

| Arm | 3d | 7d | 14d | 30d |
|---|---|---|---|---|
| control PT excess | 4.819pp (t=13.18) | 3.207pp (t=8.00) | 2.393pp (t=5.40) | 2.333pp (t=4.31) |
| **`xs_rank`** | **7.933pp (t=18.93)** | **6.274pp (t=11.90)** | **4.220pp (t=8.03)** | **3.394pp (t=9.60)** |
| `naive_init` | 4.819pp (t=13.18) | 3.207pp (t=8.00) | 2.393pp (t=5.40) | 2.333pp (t=4.31) |

`xs_rank` raises served PT excess by 45–96% and the verdict stays `skill` everywhere. The gain
is not confined to the quantile head.

## N1's served numbers are identical to the control, to the digit

Every served-classifier figure in the `naive_init` row above is byte-identical to the control's,
at all four horizons. Only the `quantile-sign` line moved. **`NAIVE_INIT_SCORE` reaches the q50
boosters and not the directional classifier**, and production serves direction from the
classifier — so N1 as shipped cannot change served DA whatever it does to the q50. This is
narrower than the caveat the instrument shipped with, which anticipated the effect being
*partly overwritten* by `_recenter_on_direction` downstream (`F1`). The classifier is not
downstream of the offset; it never sees it.

Not a wiring bug — the instrument's changelog scopes it to the quantile `Dataset` seams — but it
means the N1 result cannot be read as "no effect on serving". It was never connected to serving.

## N1 does not floor the model at the baseline

`init_score` was argued to make "loses to the one-liner" structurally impossible. It does not:
the edge is negative at 7d (−0.0115) and 30d (−0.0167) even starting from the offset. The
boosted model fits its way back below its own starting point, which is exactly the failure the
instrument's changelog flagged as possible and the action list's algebraic argument did not
allow for. **The "empirical, not algebraic" correction is now measured, not hypothetical.**

Against the control, though, N1 is a clear improvement at 3/7/14d (+0.0249 / +0.0256 / +0.0564
rank IC) and a small loss at 30d (−0.0076). The offset helps; it just does not guarantee.

## `tier_lead` closes the gap nowhere

+0.0078 / +0.0095 / +0.0011 / −0.0008 against the control. It halves the deficit at 3d and 7d,
does nothing at 14d, and is flat-to-negative at 30d. `tier_lead_return_1d` does land in the
top-5 features at 14d, so it is being used — it is simply not worth much. The expensive→cheap
lead-lag is real (z=9.1, Granger R² 9.0%) and still does not translate into forecast skill.

## The `xs_rank` void run, explained

Run `31424689196` returned `rank_ic=None` at all four horizons. Cause: the transform was applied
to **33/33** features including `price_tier`, which is the ≥$1 cohort gate — rank-transforming it
empties the cohort the metric is computed on. `50c5105` added
`RANK_TRANSFORM_EXCLUDED = {"price_tier"}`; this run logs **32/32** transformed with `price_tier`
carried at its raw scale, and the metric returns. The fix is confirmed by this panel.

## Caveats that bind the size, not the sign

1. **Cached `tuned_params`, selected against untransformed features.** Every arm restored the
   same HP on purpose, which is what makes the arms comparable, and it is also why the
   *magnitude* is provisional. Confirm with `FORCE_HP_SEARCH=1` before quoting +0.05.
2. **One dispatch per arm.** Justified only by the exact control reproduction above. It is not a
   paired harness and it has no interval.
3. **The rank-IC gain does not reach the served mid until `F1`.** `predict` recentres the mid on
   the classifier's call after all return-space corrections. `xs_rank` does improve the
   classifier itself (table above), which N1 does not, so more of its gain should survive — but
   "should" is an argument, not a measurement.
4. **Serving `xs_rank` requires the same per-date cross-section at predict time.** The transform
   is within-date over the cohort; if the served cohort differs in size or composition from the
   CV folds, the feature values are not the ones the model was fit on. This is a deployment
   question the CV read does not answer.
5. **All four arms share one label basis** (post-`873148b` re-vote). Do not compare any number
   here to a stored verdict predating 2026-08-09.

## The HP confirm, run `31432640723` (`9d47768`, `xs_rank+hpsearch`)

Caveat 1 above is now discharged at three horizons of four. **Re-tuning does not collapse the
effect — it enlarges it.**

| Horizon | cached HP | re-tuned | Optuna | served PT excess (cached → re-tuned) |
|---|---|---|---|---|
| 3d | +0.0556 | +0.0556 | **not run** | 7.933pp → 7.933pp |
| 7d | +0.0561 | +0.0561 | 111.0s | 6.274pp → 6.274pp |
| 14d | +0.0371 | **+0.0472** | 38.1s | 4.220pp → 4.464pp |
| 30d | +0.0316 | **+0.0446** | 241.6s | 3.394pp → 4.068pp |

Every cell is still positive, and every served PT excess still beats the control's
4.819 / 3.207 / 2.393 / 2.333pp. The provisional +0.05 was not an artefact of hyperparameters
tuned for a different feature scale.

**3d cannot be confirmed this way, and that is a config fact rather than a result.**
`ItemForecaster.SKIP_HP_HORIZONS = [3]` skips the search unconditionally, so `FORCE_HP_SEARCH=1`
is a no-op at 3d and the run logs `skipped - SKIP_HP_HORIZONS`. Its numbers are identical to the
cached read *by construction*, not by reproduction. The "confirm with `FORCE_HP_SEARCH=1`"
instruction that both instrument changelogs carry silently does not apply at 3d — worth knowing
before it is relied on again.

**7d re-tuned to the same fold rank IC to four decimals** (0.2199) after a real 111.0s search
that logged `best loss=-0.126100`. Consistent with a seeded search re-finding an equivalent
optimum. The chosen params were not compared against the cached ones — only the fold metric was —
so "re-found the same params" is an inference here, not a measurement.

## Not done

- No confirm at 3d, and none is possible without changing `SKIP_HP_HORIZONS`.
- No combined `xs_rank + naive_init` arm. They are independent seams and the panel does not say
  whether they add.
- N1 was not extended to the classifier's `Dataset`. That is the change that would make the N1
  question answerable on the served signal, and it is not obviously worth making given the
  `xs_rank` result.
- The `beta * -return_1d` scale variant the N1 changelog names as the next move if long horizons
  came back negative. They did (7d and 30d). Not attempted.
