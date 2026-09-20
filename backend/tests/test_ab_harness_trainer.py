"""An A/B harness must not choose its boosting iteration on the rows it scores.

Thirteen of the fifteen `ab_test_*` harnesses train LightGBM, and until now all
thirteen did it the same way::

    dval  = lgb.Dataset(X_val, y_val, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=100,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(15, verbose=False)])
    pred  = model.predict(X_val)          # <- the same rows

`lgb.early_stopping` picks `best_iteration` by minimising the metric on
`valid_sets`, and the harness then reports its verdict on exactly those rows.
`docs/changelog/2026-08-09-breadth-curve-at-1p2m-budget.md` measured the total
early-stopping-minus-fixed-rounds difference on real folds at **+1.5 to +2.7pp
directional accuracy** — against candidate effects of **0.5-1.5pp** and an
item-level MDE of **2.21-3.69pp**, so it is larger than every effect the family
was built to detect.

⚠️ **That figure is not decomposed, and this file does not try to reproduce it.**
The changelog calls it "the pathology production removed ... plus a direct
selection leak on the scored rows", i.e. two channels at once, and an attempt to
separate them synthetically (40 seeds, 500/200/200 iid splits, a single
informative feature, patience 15) resolved **neither**: the selection component
measured **-0.26pp, t = -0.40** on DA — same model, scored window versus a fresh
iid one — and the trainer component **+0.11pp, t = +0.72**. A frame that small
has no power against a fold-structured effect, so the *size* of this defect is a
property of the real archive and is not pinnable here. The rule below therefore
rests on the two things that need no magnitude claim:

  - the harnesses train a model production does not train, so their verdicts
    describe an estimator that is not served; and
  - selecting the iteration on the rows the verdict is read from is a leak on
    principle, whatever it measures on any one dataset.

Production removed the same channel on 2026-08-08 for a different reason — the
trailing validation window has an effective sample size of ~a dozen dates, so
stopping on it is noise (`FIXED_BOOST_ROUNDS`, `_train_booster`). The harnesses
never followed, so every stored verdict in this repo was produced by a trainer
production no longer uses **and** read off rows that trainer had already seen.

Three harnesses (`csfloat_basis`, `item_metadata`, `training_breadth`) grew an
*opt-in* `--no-early-stop` / `--fixed-rounds` flag when they were re-run on
2026-08-08/09, each spelled differently and each defaulting to the leak. This
file replaces all three with production's own mechanism: `_boost_rounds(h,
cv=True)` by default, `EARLY_STOPPING=1` to reproduce the leak deliberately.

See `docs/changelog/2026-08-09-breadth-curve-at-1p2m-budget.md` and
`backend/AGENTS.md` ("every A/B result stored in this repo predates ...").
"""

from __future__ import annotations

import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from models.forecaster import ItemForecaster

pytestmark = pytest.mark.slow

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "archive"

#: Every harness that trains a booster, and so has a round count to choose.
#:
#: `direction_labels` and `frozen_runs` train nothing — the first sweeps label
#: definitions over a pre-built frame, the second delegates to
#: `walkforward_backtest.run_walkforward`. Membership here is derived from the
#: source rather than hardcoded, so a new harness is covered the day it lands.
#:
#: Both spellings count. A harness that has been converted no longer contains a
#: literal `lgb.train(` — it calls production's `_train_ensemble_member` — and
#: matching only the literal would quietly drop each file out of the sweep at
#: the moment it was fixed, leaving the checks below asserting nothing.
_TRAINS_RE = re.compile(r"\blgb\.train\s*\(|\b_train_ensemble_member\s*\(")
TRAINS = sorted(p.stem for p in SCRIPTS.glob("ab_test_*.py") if _TRAINS_RE.search(p.read_text()))
NO_TRAIN = sorted(p.stem for p in SCRIPTS.glob("ab_test_*.py") if not _TRAINS_RE.search(p.read_text()))


def _informative_split(seed: int = 20260813):
    """One informative feature among 40, so the validation curve has an
    interior minimum for early stopping to find.

    On a pure-noise frame `best_iteration` is 1 and there is no selection to
    speak of, which is the first reason the synthetic reproduction failed.
    """
    rng = np.random.default_rng(seed)
    n, p = 900, 40
    X = rng.standard_normal((n, p))
    y = 0.05 * X[:, 0] + rng.standard_normal(n) * 0.05
    return (X[:500], y[:500]), (X[500:700], y[500:700]), (X[700:], y[700:])


_PARAMS = {
    "objective": "quantile",
    "alpha": 0.5,
    "metric": "quantile",
    "num_leaves": 31,
    "max_depth": 5,
    "min_data_in_leaf": 15,
    "learning_rate": 0.05,
    "verbosity": -1,
    "seed": 7,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
}


class TestTheTwoTrainersAreNotInterchangeable:
    """What is actually pinnable: a stored verdict does not transfer.

    The magnitude claim is disclaimed in the module docstring. What survives
    without it is the part that matters for reading the archive of verdicts —
    the two trainers do not produce the same model, so a number measured under
    one does not describe the other. If they ever did coincide, re-running the
    family would be unnecessary and this whole change would be cosmetic.
    """

    def test_early_stopping_selects_an_interior_iteration(self):
        """The premise of the leak: `best_iteration` is chosen, not fixed.

        Stated separately so that a LightGBM whose early stopping never fires
        cannot make the comparison below pass vacuously.
        """
        (Xtr, ytr), (Xva, yva), _ = _informative_split()
        dtrain = lgb.Dataset(Xtr, ytr)
        dval = lgb.Dataset(Xva, yva, reference=dtrain)
        model = lgb.train(
            _PARAMS,
            dtrain,
            num_boost_round=200,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
        )
        assert 1 < model.best_iteration < 200, f"nothing was selected: best_iteration={model.best_iteration}"

    def test_the_two_trainers_disagree_on_the_same_data(self):
        (Xtr, ytr), (Xva, yva), _ = _informative_split()
        dtrain = lgb.Dataset(Xtr, ytr)
        dval = lgb.Dataset(Xva, yva, reference=dtrain)
        stopped = lgb.train(
            _PARAMS,
            dtrain,
            num_boost_round=200,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
        )
        fixed = lgb.train(
            _PARAMS,
            dtrain,
            num_boost_round=ItemForecaster._boost_rounds(14, cv=True),
            callbacks=[lgb.log_evaluation(0)],
        )
        assert stopped.num_trees() != fixed.num_trees()
        assert not np.allclose(stopped.predict(Xva), fixed.predict(Xva)), (
            "the two trainers agree, which would make every stored verdict transferable and this change cosmetic"
        )


class TestProductionSuppliesTheRoundCount:
    """The harnesses must not carry a second copy of the round table.

    Same reasoning as `BID_SOURCES` having one definition: the bid exclusion
    needed three separate fixes because three loaders each had their own.
    """

    def test_the_cv_table_covers_every_horizon(self):
        for h in (3, 7, 14, 30):
            assert ItemForecaster._boost_rounds(h, cv=True) > 0

    def test_it_is_the_cv_table_and_not_the_production_one(self):
        assert ItemForecaster._boost_rounds(14, cv=True) != ItemForecaster._boost_rounds(14, cv=False)

    def test_an_unknown_horizon_does_not_train_zero_rounds(self):
        assert ItemForecaster._boost_rounds(None, cv=True) > 0

    def test_the_escape_hatch_reproduces_the_legacy_cap(self, monkeypatch):
        """`EARLY_STOPPING=1` is how the leak gets reproduced deliberately —
        the paired re-read of every stored verdict needs both arms."""
        monkeypatch.setenv("EARLY_STOPPING", "1")
        assert ItemForecaster._early_stopping_enabled()
        assert ItemForecaster._boost_rounds(14, cv=True) == 200


class TestTheEscapeHatchStillReproducesTheOldArm:
    """Characterisation, not TDD: `_train_ensemble_member` already behaved this
    way, and 13 new call sites now depend on it.

    Pinned because the re-read of every stored verdict is a *paired* read — the
    fixed-rounds arm against the early-stopping one — so if this hatch silently
    stopped attaching `valid_sets`, the two arms would become identical and the
    re-read would report "no change" for the wrong reason.
    """

    def _fit(self, early_stopping: bool):
        (Xtr, ytr), (Xva, yva), _ = _informative_split()
        dtrain = lgb.Dataset(Xtr, ytr)
        dval = lgb.Dataset(Xva, yva, reference=dtrain)
        return ItemForecaster._train_ensemble_member(
            _PARAMS,
            dtrain,
            dval,
            num_boost_round=200,
            early_stopping=early_stopping,
        )

    def test_off_by_default_it_runs_every_round(self):
        model = self._fit(early_stopping=False)
        assert model.num_trees() == 200
        assert not model.best_iteration, "best_iteration is set, so something was still selected"

    def test_on_it_stops_early_and_records_the_iteration(self):
        model = self._fit(early_stopping=True)
        assert 0 < model.best_iteration < 200

    def test_the_two_modes_are_reachable_from_the_env_var_alone(self, monkeypatch):
        """The harnesses pass `_early_stopping_enabled()`, so the env var is the
        whole interface — no per-harness flag survives."""
        monkeypatch.delenv("EARLY_STOPPING", raising=False)
        assert ItemForecaster._early_stopping_enabled() is False
        monkeypatch.setenv("EARLY_STOPPING", "1")
        assert ItemForecaster._early_stopping_enabled() is True
        monkeypatch.setenv("EARLY_STOPPING", "0")
        assert ItemForecaster._early_stopping_enabled() is False


@pytest.mark.parametrize("name", TRAINS)
class TestNoHarnessSelectsOnTheRowsItScores:
    def test_early_stopping_is_not_unconditional(self, name):
        """A bare `lgb.early_stopping(...)` in a callback list is the defect.

        Allowed only behind `_early_stopping_enabled()`, which reads
        production's `EARLY_STOPPING` env var and is off by default.
        """
        src = (SCRIPTS / f"{name}.py").read_text()
        offenders = [
            line.strip()
            for line in src.splitlines()
            if "lgb.early_stopping(" in line and not line.strip().startswith("#")
        ]
        assert not offenders, (
            f"{name} still selects its iteration on the rows it scores: "
            f"{offenders}. Use ItemForecaster._boost_rounds(h, cv=True)."
        )

    def test_it_sources_the_round_count_from_production(self, name):
        src = (SCRIPTS / f"{name}.py").read_text()
        assert "_boost_rounds" in src, (
            f"{name} trains LightGBM without asking production how many "
            f"rounds; a private num_boost_round is a second copy of "
            f"CV_FIXED_BOOST_ROUNDS"
        )

    def test_it_does_not_hardcode_a_round_count_beside_it(self, name):
        """`num_boost_round=100` was the family's shared literal. Once the
        count comes from `_boost_rounds`, a surviving literal is a second
        trainer that the fix missed."""
        src = (SCRIPTS / f"{name}.py").read_text()
        literals = [
            line.strip()
            for line in src.splitlines()
            if re.search(r"num_boost_round\s*=\s*\d+", line) and not line.strip().startswith("#")
        ]
        assert not literals, f"{name} still hardcodes a round count: {literals}"


def test_the_ad_hoc_opt_out_flags_are_gone():
    """Three harnesses grew three spellings of the same knob, each defaulting
    to the leak. One mechanism replaces them, and it defaults the other way."""
    for name in TRAINS:
        src = (SCRIPTS / f"{name}.py").read_text()
        for dead in ("--no-early-stop", "no_early_stop", "--fixed-rounds", "fixed_rounds"):
            assert dead not in src, (
                f"{name} still carries {dead}; fixed rounds are the default "
                f"now and EARLY_STOPPING=1 is the only opt-out"
            )


@pytest.mark.parametrize("name", NO_TRAIN)
def test_the_non_training_harnesses_really_train_nothing(name):
    """An exemption must be earned. `TRAINS` is derived from the source, so a
    harness that grows a trainer joins the checks above automatically — this
    asserts the derivation, not a hand-maintained list."""
    src = (SCRIPTS / f"{name}.py").read_text()
    assert not _TRAINS_RE.search(src)
    assert "lgb.early_stopping(" not in src


def test_the_family_is_covered():
    """Guards against the parametrize silently going empty — a `glob` typo or
    a rename would make every test above pass by not running."""
    assert len(TRAINS) == 13, f"expected 13 training harnesses, found {TRAINS}"
    assert set(NO_TRAIN) == {"ab_test_direction_labels", "ab_test_frozen_runs"}
