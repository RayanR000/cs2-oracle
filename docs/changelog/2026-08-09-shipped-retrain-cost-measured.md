# Shipped retrain measured: 872s, and the CV diagnostic costs twice what we thought

**Date:** 2026-08-09
**Type:** measurement — no code changed
**Supersedes cost numbers in:** `docs/research/2026-08-08-model-review.md` §1/§4,
`docs/architecture/model-optimization.md`, `docs/architecture/model.md`

## Why

Every cost figure attached to the fixed-rounds change was estimated, contended, or
measured on a config that never shipped. The 1544s run overlapped another job (30d Optuna
read 42.7s against 8.2s for identical work). The clean 1098s run used 7d = 1000 rounds;
the shipped constant is 750. The shipped config was projected to ~950–1000s by scaling
that run's phase timings — an estimate standing in for a measurement, on the number that
decides whether `SKIP_REGIMES=1` is worth taking.

## Method

Two cold `--train-only` runs, sequential (never concurrent), on an uncontended 10-core Mac,
into scratch `FORECAST_MODEL_DIR`s so the deployed artifact was untouched. Both hit the
same warm voted cache — 918 items / 986,065 rows, identical to the 1098s run. Shipped
constants confirmed *after the fact* from the saved boosters' tree counts rather than from
the source, so the measurement cannot silently describe a different config:
3d 300 / 7d 750 / 14d 150 / 30d 1000, plus 8 regime boosters.

| Arm | Config | Total |
|---|---|---|
| `ship_ci` | exactly what `price-forecast.yml` runs (`CV_DIAGNOSTIC_CLASSIFIER=0`, regimes on) | **872s (14.5 min)** |
| `ship_lfl` | same, CV diagnostic classifier on (the local/research default) | **1804s (30.1 min)** |

| Phase | `ship_ci` | Share |
|---|---|---|
| Conformal CV (33 folds × 4 horizons) | 439.3s | 50.4% |
| q50 ensemble | 158.4s | 18.2% |
| Regime models | 95.4s | 10.9% |
| Direction classifier | 92.3s | 10.6% |
| Remainder (targets, splits, medians, save) | 52.0s | 6.0% |
| Optuna | 35.0s | 4.0% |

## Findings

**1. The weekly retrain is 872s, and early stopping cost +79%, not +125%.** The 1098s
figure was never the shipped config. Against the 487s early-stopping baseline, the
accuracy fix (+23–88% rank IC at 7/14/30d) costs +385s.

**2. The CV diagnostic classifier is 52% of a classifier-on retrain, not 32–37%.**
Measured directly as the difference between the two arms: 932s. The review benchmarked
this at 69% of a fold's fit cost *under early stopping*, which was collapsing the
classifier to ~3 trees. Fixed rounds let it train full-length at 3 trees per round, so its
share roughly doubled. CI already gates it off, so nothing needs changing — but the local
and research default is still on, which puts a research retrain at 30.1 min and over the
project's 30-minute run cap.

**3. Regime models cost 95.4s / 10.9%, not the ~126s previously quoted.** That number came
from the contended and 1000-round runs. The share is unchanged from the 11.1% measured
under early stopping, so `SKIP_REGIMES=1` is still a ~1/9 lever. The cost case for cutting
it is *weaker* than last reported, and the argument has to rest on the degenerate 1-tree
and 3-tree regime boosters plausibly harming serving.

**4. Per-phase timings are not reliable to better than ±25%.** 3d conformal CV read 93.9s
and 68.3s across two uncontended runs at identical rounds. Read totals. Do not size a lever
from a single phase in a single run — several of the levers in `model-optimization.md` are
smaller than this noise band.

## Method note worth keeping

`ship_lfl` was originally intended as a like-for-like bridge to the 1098s run. It is not
one: checking `meta.json` showed `mean_classifier_acc: None` in that run, i.e. it had the
diagnostic already gated off, so 1098s and 872s differ only by the 7d round count. The
bridge arm turned out to measure something more useful than the comparison it was built
for. Artifacts on disk answered a question the logs could not.

## Open

Unchanged by this: §5's quoting-artifact question still gates further feature work, and all
15 A/B harnesses still early-stop on their own fold windows — the same defect fixed in the
trainer, untouched, so every stored A/B result predates the fix.
