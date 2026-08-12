# Pre-registered read: C1 on the clean-anchor cohort

**Written 2026-08-11 23:52 UTC (19:52 EDT), while runs `31547391395` (control) and `31547400215`
(arm, `cross_sectional_rank=true`) were still in progress. No number from either was
visible when this was written.** Both on `ab08a8b`, matrix over 3/7/14/30d, replay anchors
2026-04-15, 05-16, 06-16, 07-09.

## Why re-read an arm that was already refuted

`2026-08-11-rank-transform-does-not-transfer-to-serving.md` ruled that C1's large CV gain
(+0.069/+0.088/+0.076/+0.048) does not reach serving. Both legs of that ruling were measured
on the pooled basis, and the same day's work put the label-denominator wedge at +0.14 rank IC
— larger than the effect being judged. The verdict is therefore unreadable, not wrong. This
run re-reads it where the wedge is 1.

## The bar, fixed in advance

**Primary (CV):** `rank_ic_edge_vs_naive_tied`, arm against control on the same commit and
the same restored HP cache.
- C1 survives if the arm's edge is **≥ 0** and **arm − control > 0** at **3 of 4** horizons.

**Decisive (serving):** the replay's **tied** row, `served_ic` arm − control, per anchor.
- C1 becomes a candidate only if that delta is **positive at ≥ 3 of the 4 anchors** at the
  same horizons that cleared the CV bar. This is the leg that failed on the pooled basis; a
  CV-only pass repeats the finding already on file and changes nothing.

**Void conditions, checked before reading any IC:**
- `tied_dates < 4` at a horizon → that horizon is unreadable, not null. The `min_rows=20`
  bar bites hardest on the tied third, and one 2026-08-11 anchor had 26 tied items of 669.
- `n` differing between the two runs on the same tied cell → that cell is void. The tied
  mask is arm-invariant by construction, so a differing count means something moved that
  should not have.
- 30d alone is not a pass. The clean-anchor signal at 30d is unconfirmed in CI (+0.0536,
  3 of 4 anchors, against a local +0.1423), and its embargo exceeds its validation window.

## What this read cannot produce

A magnitude. Four anchors and eight folds are replications, not power; there is no interval
here and `paired_mde` is not being run. The readable statistic is **sign consistency**. Any
size quoted from this run is provisional for a second reason as well: HP were tuned without
the arm's features, so a positive needs `force_hp_search=1` to confirm before it is believed.

## The two outcomes, and what each means

- **CV positive, serving flat or negative** → the 08-11 refutation survives on a clean basis.
  C1 stays shelved, and the interesting question becomes why a within-date rank transform
  gains in CV and not in `predict`.
- **CV positive, serving positive** → the refutation was a measurement artifact of the
  contaminated denominator, C1 is the first arm to clear both layers, and the next step is a
  `force_hp_search` confirm rather than a ship.

Read with `python -m scripts.compare_diagnostics --control 31547391395 --arm 31547400215`.
