# 2026-09-15 — q_hat fold-design dispersion measured: NULL, close Item 3

**Orders:** `docs/specs/2026-09-14-wait-window-workplan.md` Item 3.
**Prereg:** `docs/research/2026-09-14-qhat-dispersion-fold-design-preregistration.md`
(bars quoted from it; no second prereg was opened, no extra arm ran).
**Instrument:** `backend/scripts/measure_qhat_dispersion.py`
(`CV_ROW_SEED` placebo lever; SKIP_HP defaults common to all arms —
see `changelog/2026-09-15-qhat-dispersion-harness.md` for the two
declared deltas). Summary: `/tmp/qhat_dispersion_all.csv` (local);
per-arm OOF + folds JSON under `/tmp/qhat_dispersion/<arm>/`.

## Verdict: valid probe, no winner — estimator variance is irreducible by grid choice

| h | control (9 folds) | placebo | dense (17) | sparse (5) | bar 1 |
|---|---|---|---|---|---|
| 3 | cv 0.294, 3.21× | 0.293 | 0.257 (0.87×) | 0.392 (1.33×) | fail |
| 7 | cv 0.378, 3.70× | 0.378 | 0.349 (0.92×) | 0.436 (1.15×) | fail |
| 14 | cv 0.324, 3.42× | 0.323 | 0.330 (1.02×) | 0.397 (1.23×) | fail |
| 30 | cv 0.306, 3.84× | 0.306 | 0.327 (1.07×) | 0.413 (1.35×) | fail |

- **Bar 2 holds:** control-vs-placebo CV gap 0.0005–0.004 at every
  horizon (bar: <0.15). The grid is seed-stable; the probe is valid, not void.
- **Bar 1 fails everywhere:** dense wins 0/4 (best 0.87×, bar 0.70×),
  sparse is worse at 4/4. No Phase B replay — a 1-2 pass buys one, and
  there is none.
- **Mechanism, read off the folds:** max/min is IDENTICAL across all
  three grids at every horizon (3.21/3.70/3.42/3.84×) — the extremes are
  the same validation windows (2025-10 highs, 2026-08/09 low), merely
  sampled 5, 9, or 17 times. CV falls monotonically with fold count
  (sparse > control > dense) but 2× the folds buys −13% at best. The
  spread is the calibration window's volatility regime, not the grid.

## What this closes

Per the prereg: close Item 3 and serve the feedback multiplier on the
noisy width as planned. The fold-design family joins the conditioning
family as measured dead ends — the eighth and ninth refutations of
"fix q_hat's variance upstream" (bagging, Mondrian, expanding window,
late folds, and now grid density in both directions). The 20-date served
gate (~mid-Oct) stays the correction path; nothing here changes the band
formula, and no grid change ships.
