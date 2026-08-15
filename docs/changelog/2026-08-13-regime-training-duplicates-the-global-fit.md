# Regime training is a byte-identical duplicate of the global fit, and has been since the allowlist landed

**Date:** 2026-08-13
**Status:** ⚠️ **Diagnosis only — no code changed.** CI is already unaffected
(`SKIP_REGIMES=1`, `price-forecast.yml:198`); every local retrain, research retrain and A/B
harness still pays for it.
**Instrument:** `md5` of the shipped boosters in `backend/models/saved_models/` (artifact trained
2026-08-09) plus a code read of `_skipped_feature_groups` → `_assign_regime_labels`.
**Corrects:** the rationale comment at `.github/workflows/price-forecast.yml:185-186`.

## The measurement

```
$ for h in 3 7 14 30; do
    md5 -q models/saved_models/lgb_${h}d_q50_e0.txt
    md5 -q models/saved_models/lgb_${h}d_q50_range_e0.txt
  done
3   e3a4e6d82986bd6898d028a58de107da   e3a4e6d82986bd6898d028a58de107da
7   5de15e7bf87dec2343add8f35ca2da62   5de15e7bf87dec2343add8f35ca2da62
14  7fce65063835422fc1dee46b6db563fc   7fce65063835422fc1dee46b6db563fc
30  8ca8f97eeb824915520e2d4448295464   8ca8f97eeb824915520e2d4448295464
```

**Byte-identical at 4 of 4 horizons.** `meta.json` carries `trained_regimes: ["range"]` — `bear`
and `bull` never clear `MIN_REGIME_TRAIN = 500` / `MIN_REGIME_VAL = 50` (`forecaster.py:5327-5331`).

## The mechanism

Three constants compose into a guaranteed no-op:

1. `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` (`forecaster.py:400`).
2. `_skipped_feature_groups` (`:5954-5973`) derives the skip set from that allowlist, so
   `engineer_features` **does not compute the `cross_sectional` group at all** — including
   `market_return_30d`.
3. `_assign_regime_labels` (`:1435-1436`): *"Rows without `market_return_30d` get `range`."*

So `train_set["_regime"]` is the constant `"range"`, `r_train` at `:5325` **is** `train_set`, and
the regime booster is the global booster refitted on identical rows with identical params and seed.
`_detect_current_regime` (`:1448-1449`) takes the same fallback, so `predict` prefers a model that
is bit-for-bit what it would have used anyway.

**This has held since the allowlist landed**, not since `SKIP_REGIMES=1`. Any artifact trained
without `SKIP_REGIMES=1` while `cross_sectional` was off the allowlist carries the same duplicate.

## The workflow comment is wrong on its central number

`price-forecast.yml:185-186` justifies `SKIP_REGIMES=1` with:

> *"(`range` is the only trained regime and it covers 818K-893K of ~985K rows, so the
> "specialist" branch was serving ~90% of the cohort)."*

It covers **100%** of the rows, not 83–91%. The conclusion the comment reaches is right and the
flag should stay; the figure behind it is not, and it understates the case. ⚠️ Per the
no-retroactive-edit rule the workflow comment is left as written — this entry is the correction.

Two further claims that this measurement bears on and that should not be carried forward
unqualified:

- `forecaster.py:5300-5307` — *"an artifact with no regime models serves a different mid — dropping
  them is a change to the forecast, not a cost saving."* **False under the current allowlist.**
  The mid is identical to the byte. It would become true again the moment C4 re-admits
  `cross_sectional`.
- `docs/research/2026-08-10-training-cost-levers.md`'s corrections banner — *"the ~183s regime-model
  half changes the served mid, because `predict` prefers regime models over the global one."* Same
  correction, same condition.

## Cost

The measured share is **95.4s of an 872s cold retrain (10.9%)**
(`2026-08-09-shipped-retrain-cost-measured.md`), quoted elsewhere as ~183s of a ~1120s warm CI
retrain. CI pays none of it. What still pays it: every documented local retrain that does not set
the flag, and every `ab_test_*` harness — none of the thirteen sets `SKIP_REGIMES`.

## Recommendation

**Delete the regime branch** (`forecaster.py:5298-5382`, `REGIMES`/`REGIME_RETURN_THRESHOLD_*` at
`:359-361`, the save/load/orphan-sweep legs at `:8668-8728` and `:9053-9082`, the `predict`
preference at `:7472-7527`) rather than leaving it gated. It is ~200 lines of branch that, under
every configuration production has run since the allowlist, can only reproduce the global model,
and it is a live footgun for C4: re-admitting `cross_sectional` silently reactivates an untested
three-way model split at the same time as the feature change being measured.

⚠️ **If it is kept instead of deleted, it must be gated on the allowlist, not on an env flag** —
the current code will start doing something real the moment C4 lands, inside whatever A/B is
measuring C4.

⚠️ Deleting it strands `lgb_{3,7,14,30}d_q50_range_e0.txt` in the local `saved_models/` and in the
CI Actions cache. The orphan sweep at `:8709-8723` iterates `self.REGIMES`, so removing the
constant removes the sweep that would clean them; delete the files in the same change.
