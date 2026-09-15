"""Plumbing for scripts/compute_mde.py: the fold seed and fold step must be
passable as call arguments (not module globals a caller has to rebind), and
both must default to the existing module constants so omitting them leaves
production's behaviour byte-identical.

These tests are cheap (no walkforward run): they pin signatures and the
source line that consumes the new parameter, since actually exercising two
multi-minute walkforward runs is out of scope for unit tests.
"""

from __future__ import annotations

import inspect
import re

import scripts.archive.walkforward_backtest as wf
from models.forecaster import ItemForecaster


def test_run_walkforward_accepts_step_days_and_fold_seed_with_module_defaults():
    sig = inspect.signature(wf.run_walkforward)
    params = sig.parameters

    assert "step_days" in params, (
        "run_walkforward must accept step_days so compute_mde.py can pin "
        "--step-days without editing the module constant between runs"
    )
    assert params["step_days"].default == wf.STEP_DAYS, (
        "step_days must default to the module constant so a caller that omits it gets the old fold cadence"
    )

    assert "fold_seed" in params, (
        "run_walkforward must accept fold_seed so compute_mde.py can vary "
        "the LightGBM seed across its two runs without a global rebind"
    )
    assert params["fold_seed"].default == wf.FOLD_SEED, (
        "fold_seed must default to the module constant (42) so a caller "
        "that omits it reproduces production's fixed seed"
    )


def test_fit_direction_classifier_accepts_random_state_defaulting_to_42():
    sig = inspect.signature(ItemForecaster._fit_direction_classifier)
    params = sig.parameters

    assert "random_state" in params, (
        "_fit_direction_classifier must accept random_state so the gate can "
        "thread FOLD_SEED into the classifier, matching the quantile models"
    )
    assert params["random_state"].default == 42, (
        "random_state must default to 42 -- this is the production-parity "
        "guarantee: omitting the argument must reproduce the old hardcoded "
        "behaviour exactly"
    )


def test_fold_loop_steps_by_the_step_days_parameter_not_the_module_constant():
    """The fold loop's range(...) step must read the step_days parameter.

    If this regressed to the module constant, compute_mde.py's --step-days
    (and Task 12's arms) would silently run at STEP_DAYS=60 regardless of
    what was passed on the command line -- exactly the bug Step 1b exists to
    prevent. We pin the actual range(...) call line rather than merely
    asserting step_days appears somewhere in the function, since the
    parameter could be accepted but unused.
    """
    src = inspect.getsource(wf.run_walkforward)
    loop_lines = [line for line in src.splitlines() if re.search(r"for\s+window_end\s+in\s+range\(", line)]
    assert len(loop_lines) == 1, f"expected exactly one fold-loop range(...) call, found {len(loop_lines)}"
    loop_line = loop_lines[0]

    assert "step_days" in loop_line, f"fold loop must step by the step_days parameter, got: {loop_line!r}"
    assert not re.search(r"\bSTEP_DAYS\b", loop_line), (
        f"fold loop must not reference the module constant directly, got: {loop_line!r}"
    )
