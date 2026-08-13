# All thirteen A/B harnesses now train the model production trains — and the leak's *size* did not reproduce

**Date:** 2026-08-13
**Change:** `backend/scripts/ab_test_*.py` (13 files), `backend/tests/test_ab_harness_trainer.py`
(new, 52 tests). No production code, no retrain, no knob moved, nothing dispatched.
**Bears on:** every stored A/B verdict in this repo, and `C7` in
`docs/research/2026-08-10-next-steps.md`.
**Verdict:** the defect is real and is fixed. ⚠️ **Its magnitude is NOT established, and the
"+1.5–2.7pp selection leak" framing this change was opened on is only half supported.**

## The defect

Thirteen of the fifteen harnesses train LightGBM. Until now all thirteen did it the same way:

```python
dval  = lgb.Dataset(X_val, y_val, reference=dtrain)
model = lgb.train(params, dtrain, num_boost_round=100,
                  valid_sets=[dval],
                  callbacks=[lgb.early_stopping(15, verbose=False)])
pred  = model.predict(X_val)          # the same rows
```

`lgb.early_stopping` picks `best_iteration` by minimising the metric on `valid_sets`; the harness
then reads its verdict off exactly those rows. Two things are wrong with that, and they are
independent:

1. **It is not production's trainer.** Production removed early stopping on 2026-08-08
   (`FIXED_BOOST_ROUNDS` / `CV_FIXED_BOOST_ROUNDS`, `_train_ensemble_member`) because the trailing
   validation window has an effective sample of ~a dozen dates, which had left 9 of 33 CV folds
   fitting a single tree. So every stored verdict describes an estimator that is not served.
2. **The iteration is selected on the scored window.** A leak on principle, whatever it measures.

`ab_test_item_metadata.py:582-586` already carried a comment saying so — *"That is a leak every arm
shares -- except that an arm holding a date proxy can split the validation window out on its own, so
the leak is worth MORE to it than to baseline"* — which is the asymmetry that makes it a verdict
problem rather than a level problem.

## ⚠️ The correction: the size is not decomposed, and this change could not reproduce it

`2026-08-09-breadth-curve-at-1p2m-budget.md` measured **early stopping minus fixed rounds at +1.5 to
+2.7pp DA** on real folds. That figure stands. What does **not** stand is attributing it to
selection: the changelog itself calls it *"the pathology production removed ... **plus** a direct
selection leak"*, i.e. two channels at once, and an attempt here to separate them synthetically
resolved **neither**.

40 seeds, 500/200/200 iid splits, one informative feature among 40, patience 15:

| component | how it was isolated | measured |
|---|---|---:|
| **selection** | one model, scored on its own early-stopping window vs a fresh iid one | **−0.26pp, t = −0.40** |
| **trainer** | early stopping vs fixed rounds, same scored window | **+0.11pp, t = +0.72** |

Both null, and the selection term has the *wrong sign*. A first attempt on a pure-noise frame failed
for a separate reason worth recording: with no signal, `best_iteration` is **1**, so there is no
interior optimum to select and the comparison is degenerate.

**The honest reading is that a synthetic frame this small has no power against a fold-structured
effect** — not that the breadth harness's +1.5–2.7pp is wrong. The size of this defect is a property
of the real archive, it is reproducible only by re-running the family, and it is therefore left as
the open question rather than asserted. The rule that shipped rests only on the two grounds above,
neither of which needs a magnitude.

## What shipped

One mechanism, replacing three ad-hoc ones. Every training harness now calls production's own
helper with production's own round table and production's own env var:

```python
model = ItemForecaster._train_ensemble_member(
    params, dtrain, dval,
    num_boost_round=ItemForecaster._boost_rounds(horizon, cv=True),
    early_stopping=ItemForecaster._early_stopping_enabled(),
)
```

`dval` is still passed and is simply not attached as a `valid_set` unless early stopping is on —
production's behaviour, unchanged. **`EARLY_STOPPING=1` reproduces the old arm**, which is what makes
the re-read of the stored verdicts a paired read rather than a fresh measurement.

**Three opt-in flags are gone**, each of which defaulted *to* the leak: `--no-early-stop` on
`csfloat_basis` and `item_metadata`, `--fixed-rounds` on `training_breadth`. Those three are exactly
the harnesses re-run on 2026-08-08/09 — the defect was noticed three times and patched three
different ways, none of them the default.

## Two harnesses lose a diagnostic, and it is not a regression

`q50_sampling` and `interval_sampling` **report** `best_iter` per fold, and their motivations are
written around it (*"7d q50 saves 1-2 trees per ensemble member (best_iteration ≈ 1)"*, *"3d q10
saved 11 / 1 / 5 trees"*). Under fixed rounds that field is now the fixed count, so those
observations reproduce only under `EARLY_STOPPING=1`. Noted at each site.

`recency_weights` is the sharper case: its entire stated motivation is that *"its early-stopping
round is noise-determined: 32/16/94 across the three ensemble seeds"*. Production removed the
mechanism, so **that premise now describes history**. The weight question the harness actually tests
is unaffected.

## What this does not do

**It does not re-read anything.** No harness was run; no stored verdict has moved. Every published
refutation — the six price primitives, CSFloat basis, ByMykel metadata, supply side, volume features,
breadth, regime, ensemble — still stands exactly as published, now measured under a trainer that no
longer exists in this repo. `C7` is unblocked, not done.

⚠️ **And the stacked defects are not all one fix.** A stored verdict from before 2026-08-13 sits
downstream of *at least* five separate changes: the walkforward embargo (2026-08-08, code fixed and
nothing re-run), date-clustered `paired_mde` (2026-08-07, three published refutations unresolved),
the bid exclusion (2026-08-07) and trailing-window exclusion (2026-08-09) which both moved the
labels, the ≥$1 training universe (2026-08-08), and now the trainer. Re-running one harness settles
one harness.

## Tests

`backend/tests/test_ab_harness_trainer.py`, 52 tests, suite **2142 → 2194**.

- `TRAINS` / `NO_TRAIN` are **derived from the source**, matching both `lgb.train(` and
  `_train_ensemble_member(`. Matching only the literal would have dropped each file out of the sweep
  at the moment it was fixed — which is what happened on the first run, and is why the detector is
  pinned by `test_the_family_is_covered`.
- Per harness: early stopping is not unconditional, the round count comes from `_boost_rounds`, and
  no `num_boost_round=<literal>` survives beside it.
- The escape hatch is characterised rather than assumed (`valid_sets` attached only when on,
  `best_iteration` unset when off, env var the whole interface) — because a hatch that silently
  stopped working would make the paired re-read report "no change" for the wrong reason.
- `TestTheTwoTrainersAreNotInterchangeable` replaces the magnitude claim with the part that is
  checkable: the two trainers do not produce the same model, so a verdict measured under one does not
  describe the other. If they ever coincided, this change would be cosmetic.
