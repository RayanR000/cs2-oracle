# The `sigma` scale is one exponent per horizon — shrinkage is a no-op and flexibility buys nothing

**Date:** 2026-08-12
**Instrument:** `backend/scripts/design_sigma_scale.py` (new, offline, read-only). Selection rule
pinned in its docstring and committed **before** the run.
**Settles:** the open half of `2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md` §"Next" item 2 —
*"a decision for 14d/30d that is not the same constant"*
**Status:** ✅ decided. ⚠️ **The pinned selection rule MISFIRED and picked production at 4 of 4** —
reported below rather than replaced, with the substantive read labelled post-hoc. **Nothing shipped.**

## What was open

Held out on CV folds, a single global exponent removed **76% / 62%** of the conditional error at
3d/7d but only **26% / 12%** at 14d/30d. Two explanations need opposite fixes:

- **(a) `beta` is estimated noisily and drifts** → shrink it toward 1.
- **(b) the log-log line is the wrong shape** → drop the parametric form.

Four arms, walk-forward over 507–588 test dates, refit every **14 days** — production's age-based
retrain cadence, so each arm is scored on a scale fitted from stale history the way it would actually
be served: `P0` production (`÷ sigma`), `P1` one fitted exponent, `P2` that exponent shrunk toward 1
by its own sampling noise, `P3` a non-parametric binned scale free to bend, `P4` Mondrian conformal
with a separate `q_hat` per `sigma` decile.

## (a) is refuted: `beta` is not noisy, and shrinkage does nothing

`beta` refitted on six contiguous time blocks of the selection history:

| h | `beta` per block | sd | se of the mean | `betâ − 1` | shrinkage keeps |
|---|---|---:|---:|---:|---:|
| 3 | 0.415 0.401 0.476 0.389 0.435 0.248 | 0.078 | 0.032 | −0.570 | **λ = 0.997** |
| 7 | 0.370 0.406 0.415 0.415 0.489 0.338 | 0.051 | 0.021 | −0.550 | **λ = 0.999** |
| 14 | 0.246 0.410 0.311 0.438 0.476 0.399 | 0.085 | 0.035 | −0.571 | **λ = 0.996** |
| 30 | 0.209 0.388 0.247 0.361 0.471 0.408 | 0.100 | 0.041 | −0.636 | **λ = 0.996** |

The departure from 1.0 is **6–14× the standard error of the estimate**, so there is nothing for
shrinkage to shrink. `P2` reproduces `P1` to two decimals on every metric at every horizon.
**Do not implement shrinkage.**

## (b) is refuted too: flexibility does not pay for itself

Held-out `sigma`-decile error, level-matched so only the tilt is compared:

| h | `P0` production | **`P1` exponent** | `P3` binned scale | `P4` per-bin `q_hat` |
|---|---:|---:|---:|---:|
| 3 | 9.60pp | **1.10** | 0.98 | 2.06 |
| 7 | 8.86pp | **0.51** | 0.55 | 1.56 |
| 14 | 8.35pp | **1.30** | 1.02 | 1.84 |
| 30 | 8.99pp | **2.44** | **1.47** | 2.00 |

`P4` is worse than the exponent at 3 of 4. `P3` is better by **0.04–0.28pp** at 3d/14d — inside the
noise of this statistic — and worse on marginal coverage at **4 of 4** (4.2/5.2/5.5/4.8pp against
`P1`'s 4.1/5.0/5.1/3.9pp). Neither justifies persisting twenty bin edges and their medians instead
of one float.

**The one real exception is 30d**, where the binned scale cuts the residual tilt from 2.44 to
**1.47pp** (−40%). That is the only place the parametric form visibly leaves something on the table,
and it is a follow-up, not a blocker: `P1` already beats production by 73% there.

## So the answer is `P1`, and it works at all four horizons

Held out, against production:

| h | tilt (level-matched) | | width | marginal `|cov − 80%|` |
|---|---|---:|---:|---|
| 3 | 9.60 → **1.10pp** | −89% | **0.87×** | 5.5 → 4.1pp |
| 7 | 8.86 → **0.51pp** | −94% | **0.86×** | 7.0 → 5.0pp |
| 14 | 8.35 → **1.30pp** | −84% | **0.84×** | 7.7 → 5.1pp |
| 30 | 8.99 → **2.44pp** | −73% | **0.77×** | 8.7 → 3.9pp |

The raw decile profile at 7d goes from `70 78 82 85 89 90 91 93 94 97` to
`86 85 85 85 86 85 85 86 83 85`. **Bands are 13–23% narrower.**

⚠️ **This contradicts the 26% / 12% at 14d/30d on record**, and the disagreement is not fully
attributable. Two things differ: that read used real OOF residuals and this uses the model-free
`r̂ = 0` panel; and that read fitted on all folds but one and scored **one** fold, where 14d/30d
windows overlap heavily, against **37–43 independent refits** over 507–588 dates here. The
evaluation axis favours this read and the estimator axis favours that one. The confirm dispatch
settles it, and until then "the exponent reaches only 3d/7d" should not be repeated as settled.

## ⚠️ The pinned selection rule misfired, and it picked production at 4 of 4

The rule gated on `|marginal − 80%| ≤ 2pp` **on the selection period** and then compared tilts.
Production is the only arm that passed the gate at every horizon (80.4 / 81.1 / 81.1 / 78.8%), so it
won by default — while the exponent arms were excluded for landing at **74–77%**.

**Why the rule has no power:** marginal coverage is not stationary. Between the two periods it moves
**80.4 → 85.5%** for production and **76.8 → 84.1%** for the exponent at h=3. A gate applied in one
period cannot select for behaviour in another when the level drifts 5–10pp for every arm. Production
passed the gate in the selection period and is the **worst** arm on the same statistic held out.

The comparison above is therefore **post-hoc** and labelled as such in the script, the CSV and here.
It is the same level-matching the published `sigma`-tilt entry used, for the same reason.

## What is stable, and what is not

**Stable across both periods, 4/4:** the tilt reduction (selection 8.12 → 1.14 / 7.43 → 2.15 /
7.58 → 2.68 / 9.35 → 1.40pp) and the width reduction (0.84–0.88× on the selection period too).

⚠️ **NOT stable: the marginal-coverage win.** On the selection period production is *closer* to 80%
than the exponent at all four horizons. The exponent's held-out marginal advantage is the level
drifting into it, not a property of the scheme. This is the same conclusion the attribution read
reached from the other side — the `sigma` mix is 36–68% of the marginal defect and the residual law
is the rest — and **nothing measured here fixes the level.**

**Consequence, and it is a product risk to state before shipping:** the exponent narrows the band
12–23%, and on a calm period the arms here cover **74–77%** against a stated 80%. Shipping the
exponent improves the tilt unconditionally and leaves the level unresolved in **both** directions.
The served `confidence` tag was already withdrawn (`2026-08-12-served-confidence-withdrawn.md`),
which is what keeps this from being a published-number regression.

## Next

`docs/superpowers/specs/2026-08-12-sigma-exponent-design.md` — the implementation, which is one
persisted float per horizon reaching four call sites, gated, plus a paired dispatch on real OOF
residuals to settle the 14d/30d disagreement above.

## Reproducing

```
cd backend && venv/bin/python -m scripts.design_sigma_scale --horizons 3,7,14,30
```

Read-only, ~4 minutes, no dispatch. Refits every 14 days by design; a per-date refit would change
the arms' variance and is not what production does.
