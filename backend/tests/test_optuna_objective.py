"""Optuna must select on the metric the project uses, not the one it discarded.

FIXED_BOOST_ROUNDS replaced early stopping in training and CV on 2026-08-08
because the trailing validation window carries ~a dozen effective observations.
The Optuna objective was not touched: it still scored
`model.best_score["valid_0"]["quantile"]` under `lgb.early_stopping(20)`. The
FIXED_BOOST_ROUNDS comment records the conflict directly -- at 14d and 30d the
val-loss optimum is 25 rounds while rank IC peaks at 500-750.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from models.forecaster import ItemForecaster


def _code(func) -> str:
    """Source of `func` with comments and docstrings stripped.

    These assertions are about what the objective DOES. The method documents
    the criterion it replaced, so a raw `inspect.getsource` grep matches the
    prose explaining why early stopping is gone and fails on the fix.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)  # ast.unparse never emits comments


def test_optuna_does_not_early_stop():
    assert "early_stopping" not in _code(ItemForecaster._optuna_search_params), (
        "Optuna still early-stops on the window it scores; that is the criterion FIXED_BOOST_ROUNDS replaced."
    )


def test_optuna_does_not_prune_on_the_old_metric():
    src = _code(ItemForecaster._optuna_search_params)
    assert "LightGBMPruningCallback" not in src, (
        "the pruner prunes on LightGBM's reported `quantile` metric, not on "
        "the returned objective, so it re-introduces the replaced criterion."
    )


def test_optuna_scores_within_date_rank_ic():
    src = _code(ItemForecaster._optuna_search_params)
    assert "_within_date_rank_ic" in src
    assert "best_score" not in src


def test_optuna_takes_val_dates():
    sig = inspect.signature(ItemForecaster._optuna_search_params)
    assert "val_dates" in sig.parameters, (
        "rank IC is within-date; a pooled Spearman would re-introduce the market factor the PT test exists to reject."
    )


def test_optuna_tunes_at_cv_rounds_not_production_rounds():
    src = _code(ItemForecaster._optuna_search_params)
    assert "cv=True" in src, (
        "tuning at production rounds and evaluating at CV rounds selects "
        "params that only pay off at a depth CV never reaches."
    )


def test_artifact_version_bumped():
    assert ItemForecaster.MODEL_ARTIFACT_VERSION >= 6, (
        "a cached meta.json would otherwise supply params chosen under the old criterion."
    )
