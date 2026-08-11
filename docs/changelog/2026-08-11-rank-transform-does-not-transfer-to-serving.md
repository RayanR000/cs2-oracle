# The rank transform's CV gain does not appear at serving, 2026-08-11

Paired dispatch of `model-diagnostics.yml` on `ee6fc11`: control run `31451647680`, arm run
`31452618359` (`cross_sectional_rank=true`). Same commit, same runner image, same restored HP
cache, same replay anchors. Both runs report CV rank IC **and** a serving replay, so the two
layers are measured on the same artifact for the first time.

## The result

**CV, cross-sectional rank IC on the ≥ $1 cohort** — the arm is positive at all four horizons,
reproducing the earlier read:

| h | control | arm | Δ | naive (`-return_1d`) |
|---|---|---|---|---|
| 3 | 0.1794 | 0.2480 | **+0.0686** | 0.1956 |
| 7 | 0.1293 | 0.2176 | **+0.0883** | 0.1653 |
| 14 | 0.1015 | 0.1770 | **+0.0755** | 0.1478 |
| 30 | 0.0907 | 0.1390 | **+0.0483** | 0.1087 |

The control loses to `-return_1d` at all four (edge −0.0162 / −0.0360 / −0.0463 / −0.0180, each
logged with the "the ML stack is subtracting from its own best feature" warning). The arm beats it
at all four (+0.0524 / +0.0523 / +0.0292 / +0.0303). On CV alone this is the strongest arm the
project has produced.

**Serving replay, same artifacts, anchors 2026-06-01 and 2026-07-05** — the sign reverses at three
of four horizons:

| h | anchor | control | arm | Δ |
|---|---|---|---|---|
| 3 | 06-01 | −0.1871 | −0.2015 | −0.0144 |
| 3 | 07-05 | −0.1755 | −0.1963 | −0.0208 |
| 7 | 06-01 | +0.0345 | −0.1166 | −0.1511 |
| 7 | 07-05 | −0.0551 | +0.0004 | +0.0555 |
| 14 | 06-01 | −0.0401 | −0.0847 | −0.0446 |
| 14 | 07-05 | +0.0898 | −0.0182 | −0.1080 |
| 30 | 06-01 | −0.0622 | +0.0326 | +0.0948 |
| 30 | 07-05 | +0.1811 | +0.1756 | −0.0055 |

Mean Δ by horizon: **−0.0176 / −0.0478 / −0.0763 / +0.0447**. Worse in **6 of 8 cells**, mean
−0.024, against a CV Δ of +0.05 to +0.09 everywhere.

## What is not in dispute

- **The transform reaches serving and does what it says.** Every arm replay logs
  `xs_rank=True floor=1.0 skipped=[]` and `32/32 features` transformed — the cohort guard from
  `34091fd` and the skip-set guard from `cfff8ae` both hold at serve time.
- **The CV gain is real and reproducible.** It has now been measured three times: runs
  `31430874845`, `31438314051`, and this control/arm pair, the last of which is the first to carry
  its own control on the same commit.
- **The two layers disagree on the same artifact.** This is not a comparison against a stored
  number (hazard 6) and not a comparison across commits.

## What this does not establish

**It does not refute C1.** Two anchors is two dates, and the per-cell spread (−0.1511 to +0.0948)
is several times the mean effect. There is no interval here and no way to compute one from two
dates. The honest reading is: the CV gain has not been shown to reach the served forecast, and
there is now evidence it may not.

Four further limits:

- **The control loses to the naive baseline at serving in 7 of 8 cells** and the arm in 6 of 8.
  The whole panel sits near or below a one-line baseline, so the arm-vs-control difference is a
  comparison between two things that are both underperforming.
- **HP were selected without the feature**, from the restored cache, for both arms. That is what
  makes them comparable and is also why the size is provisional. `force_hp_search=true` would size
  it, at the cost of comparability with these two runs.
- **The anchor-basis question is unresolved** (`2026-08-10-serving-replay.md`). It cancels in a
  paired comparison at fixed anchors, which is why this pairing is readable at all, but it means
  no absolute number in the serving table should be quoted on its own.
- **Each matrix job trains one horizon beside restored production boosters.** The replay scores
  only that horizon (`--horizons`), and control and arm share the structure, so the pairing is
  sound — but neither run is a coherent four-horizon artifact.

## The larger read

The serving panel's mean rank IC is near zero across the whole control run while CV reports 0.09
to 0.18 on the same artifacts. Whatever the arm does or does not add, **the dominant effect is
that CV rank IC does not survive `predict()`** — the prior-day blend, the tier bias,
`_recenter_on_direction` and the conformal band all sit between the two measurements, and none of
them was ever scored. That makes serving-path work (`F1` in `2026-08-10-next-steps.md`) prior to
any further feature arm: a feature evaluated on CV rank IC is being selected on a quantity that
does not reach the product.

## Next

1. More anchors before any verdict on C1 — the effect needs an interval, and anchors are cheap
   (~90s each in CI on top of a run that already trains).
2. `F1`, and specifically **which of the four serving transforms costs the signal**. The replay can
   answer that directly by disabling them one at a time at a fixed anchor, which nothing could do
   before today.
