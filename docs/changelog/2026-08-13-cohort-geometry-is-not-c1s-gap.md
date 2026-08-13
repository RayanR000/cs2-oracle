# The cohort geometry is real, is 0.8%, and is not C1's gap — the explanation list is now empty

**Date:** 2026-08-13
**Opened by:** `docs/changelog/2026-08-13-serving-transforms-do-not-explain-the-cv-gap.md` — *"what
remains is the cohort geometry the two paths score over … and that is a code-reading question, not a
dispatch."*
**Method:** a code read of both call sites, plus two confirmatory counts straight from the archive
(read-only, no DB, no `predict`, no dispatch, nothing written). No bars, so no pre-registration: this
is a diagnosis, not an arm.
**Verdict:** ❌ **the asymmetry exists and is far too small.** 0.7–0.9% of items. **C1 should be
closed.**

## The asymmetry, stated precisely

Both paths gate the rank transform's reference population on "items whose **median price** clears
$1", and `_reference_cohort_mask`'s docstring says it matches `_filter_by_median_price` *"exactly"*.
**That is true of the statistic and false of its support:**

| | window the item median is taken over |
|---|---|
| training (`build_training_data` → `_filter_by_median_price`) | **1460 days** |
| serving (`predict` → `_reference_cohort_mask`) | the predict frame: `PREDICT_FETCH_DAYS` = **730**, then `_tail_predict_frame` cuts to `PREDICT_TAIL_ITEM_DAYS` = **240 observed item-days** |

So the two cohorts are the same rule evaluated on supports that differ by ~6×, and the ordering in
`predict` puts `_reference_cohort_mask` (`forecaster.py:7363`) **after** the tail
(`forecaster.py:7286`). This is a genuine train/serve difference and it is worth the docstring fix.

**Why it was the right last place to look.** The control does not rank anything, so its served
features cannot depend on the reference population at all. **Only the arm is exposed to this** — the
exact shape of asymmetry needed to explain a CV gain that does not reach serving.

## It measures 0.7–0.9%

Both cohort definitions, evaluated on the archive at C1's six audited anchors:

| anchor | train | serve | both | train-only | serve-only | Jaccard | symmetric diff |
|---|---:|---:|---:|---:|---:|---:|---|
| 2026-04-18 | 22,616 | 22,726 | 22,586 | 30 | 140 | 0.993 | 170 (0.7%) |
| 2026-04-28 | 22,621 | 22,738 | 22,588 | 33 | 150 | 0.992 | 183 (0.8%) |
| 2026-05-08 | 22,629 | 22,751 | 22,598 | 31 | 153 | 0.992 | 184 (0.8%) |
| 2026-05-19 | 22,674 | 22,804 | 22,643 | 31 | 161 | 0.992 | 192 (0.8%) |
| 2026-05-29 | 22,787 | 22,925 | 22,756 | 31 | 169 | 0.991 | 200 (0.9%) |
| 2026-06-08 | 23,208 | 23,350 | 23,177 | 31 | 173 | 0.991 | 204 (0.9%) |

A percentile is a smooth function of its reference set: perturbing 0.8% of the population moves each
rank by order 0.8% of the [−1, 1] range. **That cannot erase +0.0684 / +0.0759 / +0.0588 / +0.0408
rank IC.**

🔑 **One thing here is not noise and is worth keeping.** The disagreement is **asymmetric and
directional** — serving admits **140–173** items training excludes, against **30–33** the other way,
at every anchor. A shorter, more recent window catches items that have *risen* through $1 lately, so
the served reference population is slightly broader and tilted toward recent risers. It is 0.8% now.
It is the direction that would grow in a rising market, and nothing currently measures it.

## The other two training-only filters contribute nothing

Training also drops dead and corrupt items after the floor; serving drops neither. Both terms are
null on this cohort:

- **`_filter_dead_items` is null by construction** — it flags items whose *max* price is ≤ $0.05,
  which cannot clear a $1 median floor. Zero overlap, provable rather than measured.
- **`_flag_corrupt_items` catches 0, 0 and 1 cohort item** at 2026-04-18 / 05-19 / 06-08 (>10 daily
  jumps over 500%). 0.00% at every anchor.

## The plumbing is correct, and that is worth recording too

The read found no defect in the gating, which is the other way this could have gone:

- `_cross_sectional_rank_served` takes **the artifact over the environment** (`forecaster.py:2316`),
  so a flag set at replay time cannot make a raw-fitted booster serve ranks or the reverse.
- `predict` **raises** rather than degrades if the artifact carries no `train_min_median_price`
  (7345) or no `xs_rank_skipped_cols` (7354) — the second is why serving no longer re-derives the
  date-constant skip and ranks 31 of 32 columns against training's 32.
- Training ranks with `reference_mask=None` **after** the price floor has already been applied
  (4592 before 4657), so "every row is the cohort" is accurate there rather than a shortcut.

⚠️ The served-cohort guard at 7371 fires at a **>25%** move. That is ~30× the disagreement measured
here, so it would never have surfaced this — correctly, since the disagreement does not matter. It
is a frame-sanity check, not a percentile-fidelity check, and should not be read as the latter.

## What this closes

Four explanations have now been offered for C1 gaining +0.04–0.08 rank IC in CV and not reaching
serving. **All four are spent:**

1. **The label denominator** — refuted 2026-08-11 on the tied cohort.
2. **The anchor set** — refuted 2026-08-13 on six audited anchors, where 30d *reversed*.
3. **The serving transforms** — refuted 2026-08-13; removing all three leaves the gap.
4. **The cohort geometry** — refuted here, at 0.8%.

**Recommendation: close C1.** It is CV-positive at 4 of 4, serving-positive at 1 of 4, worse on the
deviating cohort in 19 of 24 cells, and there is no longer a named candidate for the difference. What
would be left is an unnamed one, and the honest way to spend the next dispatch is not on finding it.

## Caveat on the counts

`backfilled_only=True` needs the database, and `backend/.env` points at production, so both counts
are taken on the **archive-wide** ≥$1 universe (~22,600 items) rather than the ~916-item backfilled
training cohort. The boundary-crossing *rate* is a property of items sitting near $1 and should
carry; the absolute counts are not the served cohort's. At 916 items, 0.8% is about **7 items**. The
per-item-day vote is reproduced as the median across surviving sources — `archive_universe_sql_filter`
already excludes the bid and trailing-window feeds — which is `_apply_multi_source_voting` without
its 2σ mask.
