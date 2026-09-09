# The exceedance head does not want item metadata either — `EXCEEDANCE_META` stays off

**Date:** 2026-09-09
**Change:** New `scripts/exceedance_meta_ab.py` and `tests/test_exceedance_meta_ab.py`
(10 tests). New `TestExceedanceMetaGate` in `tests/test_bymykel_metadata_wiring.py`
(6 tests) covering the previously untested `EXCEEDANCE_META` gate. **`EXCEEDANCE_META: "1"`
REMOVED from `.github/workflows/price-forecast.yml`** — it was live in the retrain env
(added in `4949714`) and would have shipped at the next full retrain. An earlier note in
this entry's working session claimed it was wired into no workflow; that was a truncated
grep, and the correction is the reason this changelog carries a production change.
**Follows:** `2026-08-06-bymykel-metadata-refuted.md`, which refuted the same nine
columns against the **centre**.

`EXCEEDANCE_META=1` already existed in `forecaster.py` — the gate
(`exceedance_meta_enabled()`, :1509), the widened matrix
(`_exceedance_feature_matrix`, :1521), the median extension (:6214-6221) and
self-describing serving via `head.feature_name()` (:8125). It had never been measured
and had no tests.

The argument for measuring it anyway: the 2026-08-06 refutation scored a **signed**
target, and `centre-shrinkage-lambda-is-zero` plus
`composite-centre-ranks-cannot-scale` say that target is unpredictable here at any
scale — so a null there is weak evidence about an **unsigned** magnitude head, which is
the one signal on this panel that has ever measured real. Rarity, StatTrak, float range
and crate/collection are plausible magnitude priors while being useless for sign.

## The verdict

**Refuted, and at h=7 actively harmful.** Three arms (baseline / treatment / placebo),
paired per-fold on 25-26 walk-forward folds per horizon, scored on production's own
`_fit_exceedance_classifier` against `target_exceed_{h}d`, served cohort (>=$1). Deltas
are treatment − baseline; `*` marks a 95% interval excluding zero.

| h | held-out AUC | held-out log loss | trained AUC | trained log loss |
|---|---|---|---|---|
| 3d  | +0.010 | +0.0009 | **+0.020*** | −0.0008 |
| 7d  | **−0.029*** | **+0.0019*** | +0.021 | −0.0000 |
| 14d | +0.018 | **+0.0027*** | +0.004 | +0.0000 |
| 30d | +0.003 | **+0.0136*** | +0.011 | **+0.0088*** |

Log loss is worse (positive) at **every** horizon and significantly so at 3 of 4. AUC
never improves out-of-sample at any horizon, and at h=7 it significantly *degrades*
(−0.029, winning only 8 of 26 folds). Baseline is not a weak opponent: trained-cohort
mean AUC is 0.774 at h=7, so the head already ranks well on price technicals alone.

## Why the trained-cohort AUC gains are not a result

The pattern is the memorisation signature, not a prior:

- **trained** AUC rises (significantly at h=3, +0.020) while **held-out** log loss
  simultaneously degrades (significantly at h=7/14/30). The columns help items the
  model trained on and hurt items it has never seen.
- `treatment_vs_placebo` reproduces exactly that split — trained AUC **+0.017*** at
  h=3, held-out log loss **+0.014*** at h=30 — so it is not model capacity. Both arms
  carry the same nine columns; only treatment's are unpermuted.
- `type_meta_crate_id` and `type_meta_collection_id` carry ~110-300 levels over ~870
  items. That is enough to address an item, and a per-item base exceedance rate is
  exactly what a tree would bank through them.

**The placebo arm is null everywhere** (every one of 16 cells' CIs spans zero, |mean|
<= 0.004 AUC), which is the harness's own sanity check: permuting the columns removes
the effect, so the effect is in the values and not in the widening.

## What is held fixed

Fold machinery, item split, row budget and the train-side purge are imported wholesale
from `ab_test_item_metadata` rather than re-derived. This harness family has already
been wrong twice from a second implementation of a shared rule — the train universe
(2026-08-08) and the embargo (2026-08-07) — and a paired read only holds if both arms
see identical rows. The model is production's `_fit_exceedance_classifier`, so the
served-cohort reweighting and NaN-label drop are the deployed ones. Arms differ **only**
in which columns enter `X`.

## Reproduce

```
python -m scripts.exceedance_meta_ab --build-cache-only \
  --metadata-parquet ../price-archive/item-metadata-bymykel.parquet \
  --frame-cache /tmp/exc_meta_frame.parquet
python -m scripts.exceedance_meta_ab --horizon 7 \
  --metadata-parquet ../price-archive/item-metadata-bymykel.parquet \
  --frame-cache /tmp/exc_meta_frame.parquet --out /tmp/exc_meta_h7.json
```

Frame is 1,460,099 rows; metadata coverage is 99.0% on rarity/StatTrak/souvenir,
92.9% on item age, 73.6-80.4% on float and crate/collection — so this is not a
coverage-limited null.

## The `static_only` arm closes the obvious reopen condition

The natural objection to the table above is that two high-cardinality ID columns could
be drowning three columns that do have a mechanism. So a fourth arm carries **rarity +
StatTrak + souvenir only** (33 features), dropping `type_meta_crate_id` and
`type_meta_collection_id` — the addressing channel — while keeping the magnitude
priors. Deltas are static_only − baseline:

| h | held-out AUC | held-out log loss | trained AUC | trained log loss |
|---|---|---|---|---|
| 7d  | +0.003 | −0.0003 | **+0.016*** | **−0.0009*** |
| 30d | +0.005 | +0.0040 | +0.011 | +0.0024 |

Dropping the ID columns does remove the harm — nothing is significantly *worse*, and at
h=7 the trained cohort improves on both metrics. But **every held-out cell spans zero at
both horizons**, and at h=30 all four cells do. The gain is confined to items the model
trained on, which is the same trained-only pattern as `treatment`, just smaller. A
feature that helps only the items it has already seen is not a prior; production serves
items in and out of the training set, and the held-out read is the one that says whether
the column carries transferable information.

`treatment` vs `static_only` at h=7 makes the same point from the other side: trained
AUC **+0.016*** for the full bundle over the static three, held-out AUC null. The two ID
columns add trained-cohort separation and no generalisation — exactly what memorisation
looks like.

## Verdict

**The bundle is refuted against both halves of the model.** The centre was settled
2026-08-06; the exceedance head is settled here. `EXCEEDANCE_META` stays off, is wired
into no workflow, and should be read as dead rather than pending. The gate and its
tests stay in the tree because the code is already written, correct and cheap to keep —
not because a future run is expected to reverse this.

Anyone reopening it needs a *new* column family with a magnitude mechanism, not a
different subset of these nine. Note also that `_fit_anomaly_classifier`
(`forecaster.py:7712`) already exists unmeasured — that is the better next question,
since it asks about a 2-sigma own-history exceedance rather than a cost threshold.
