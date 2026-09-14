"""An A/B harness must verdict on a fold-clustered interval, not a win count.

Ten of the thirteen `ab_test_*` harnesses had no significance test of any kind
until 2026-08-08. They decided in one of three ways, and none of them is a test:

  - **fold win-counts** (`interval_sampling`, `q50_sampling`,
    `recency_weights`): "the arm wins on at least half the folds". Two arms
    differing only by a LightGBM seed clear that half the time by construction,
    and at the ~8 folds these run, 6/8 is unremarkable under the binomial;
  - **a pooled delta against a ±0.5pp threshold** (`volume_features`,
    `price_primitives`, `feature_contribution`, `supply_side`,
    `direction_labels`), against a measured item-level MDE of **2.21–3.69pp** —
    so every green tick was inside the noise floor by a factor of five;
  - **a bare `a > b`** (`ensemble`, `regime`), which is a coin toss.

The replacement is the one the other three already used: pair on
`(item_id, forecast_date)`, resample on `fold_id`, and report the interval. The
generic `paired_metric_difference` was factored out of `paired_da_difference`
for the pinball arms, whose metric is not a rate.

See `docs/changelog/2026-08-08-embargo-and-harness-hygiene.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from backtest.paired_mde import (
    format_paired,
    paired_arm_contrasts,
    paired_da_difference,
    paired_metric_difference,
    verdict,
)
from backtest.walkforward_records import (
    fold_level_records,
    paired_records,
    without_records,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _arm(offset, *, folds=6, rows=40, seed=0, metric="pinball"):
    """One arm's records: `offset` shifts every row by a constant."""
    rng = np.random.default_rng(seed)
    out = []
    for f in range(folds):
        out += paired_records(
            item_ids=np.array([f"item-{i}" for i in range(rows)]),
            forecast_dates=np.array([f"2026-01-{f + 1:02d}"] * rows),
            fold_id=f,
            **{metric: rng.normal(1.0, 0.05, rows) + offset},
        )
    return out


class TestPairedRecords:
    def test_it_carries_the_pairing_key_and_the_cluster_key(self):
        rec = paired_records(item_ids=["ak"], forecast_dates=["2026-01-01"], fold_id=7, direction_correct=[True])[0]
        assert set(rec) == {"item_id", "forecast_date", "fold_id", "direction_correct"}
        assert rec["fold_id"] == 7

    def test_the_date_is_stringified(self):
        """A datetime64 on one arm against a datetime.date on the other pairs
        zero rows, silently — the failure mode `_write_records` documents."""
        rec = paired_records(
            item_ids=["ak"],
            forecast_dates=np.array(["2026-01-01"], dtype="datetime64[D]"),
            fold_id=0,
            direction_correct=[True],
        )[0]
        assert isinstance(rec["forecast_date"], str)

    def test_values_survive_a_json_round_trip(self):
        """np.bool_ has no `-`, which is exactly how the difference is taken,
        and np.float32 does not serialise."""
        recs = paired_records(
            item_ids=["ak"],
            forecast_dates=["2026-01-01"],
            fold_id=0,
            direction_correct=np.array([True]),
            pinball=np.array([0.5], dtype=np.float32),
        )
        json.dumps(recs)
        assert recs[0]["direction_correct"] - False == 1

    def test_keep_narrows_every_array_together(self):
        recs = paired_records(
            item_ids=["a", "b", "c"],
            forecast_dates=["2026-01-01"] * 3,
            fold_id=0,
            keep=np.array([True, False, True]),
            direction_correct=np.array([True, True, False]),
        )
        assert [r["item_id"] for r in recs] == ["a", "c"]
        assert [r["direction_correct"] for r in recs] == [True, False]

    def test_a_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            paired_records(
                item_ids=["a", "b"], forecast_dates=["2026-01-01"], fold_id=0, direction_correct=[True, False]
            )

    def test_at_least_one_metric_is_required(self):
        with pytest.raises(ValueError, match="at least one metric"):
            paired_records(item_ids=["a"], forecast_dates=["2026-01-01"], fold_id=0)


class TestPairedMetricDifference:
    def test_it_measures_a_known_shift(self):
        a, b = _arm(0.0, seed=1), _arm(0.0, seed=1)
        b = [dict(r, pinball=r["pinball"] - 0.10) for r in b]
        out = paired_metric_difference(a, b, value_key="pinball")
        assert out["mean_diff"] == pytest.approx(-0.10, abs=1e-9)
        assert out["value_key"] == "pinball"

    def test_a_constant_shift_has_a_zero_width_interval(self):
        """Every fold's difference is identical, so there is no between-fold
        variance to resample. The interval collapsing onto the estimate is the
        correct answer, not a degenerate one."""
        a = _arm(0.0, seed=2)
        b = [dict(r, pinball=r["pinball"] - 0.10) for r in a]
        out = paired_metric_difference(a, b, value_key="pinball")
        assert out["mde"] == pytest.approx(0.0, abs=1e-6)

    def test_it_still_refuses_a_missing_cluster_key(self):
        a = _arm(0.0, seed=3)
        b = [{k: v for k, v in r.items() if k != "fold_id"} for r in a]
        with pytest.raises(ValueError, match="cluster key"):
            paired_metric_difference(a, b, value_key="pinball")

    def test_it_still_refuses_arms_that_share_no_rows(self):
        a = _arm(0.0, seed=4)
        b = [dict(r, item_id="other-" + str(r["item_id"])) for r in a]
        with pytest.raises(ValueError, match="no paired records"):
            paired_metric_difference(a, b, value_key="pinball")

    def test_the_da_wrapper_is_arithmetically_unchanged(self):
        """`paired_da_difference` was factored through the generic function on
        2026-08-08. Stored A/B results are quoted in its `_pp` keys, so the
        names and the numbers both have to survive."""
        a = _arm(0.0, seed=5, metric="direction_correct")
        b = _arm(0.30, seed=5, metric="direction_correct")
        da = paired_da_difference(a, b)
        generic = paired_metric_difference(a, b, value_key="direction_correct", scale=100.0)

        assert set(da) == {
            "mean_diff_pp",
            "n_paired",
            "n_dates",
            "n_clusters",
            "cluster_key",
            "ci_lower_pp",
            "ci_upper_pp",
            "mde_pp",
        }
        assert da["mean_diff_pp"] == generic["mean_diff"]
        assert da["ci_lower_pp"] == generic["ci_lower"]
        assert da["mde_pp"] == generic["mde"]


class TestVerdict:
    def test_an_interval_clear_of_zero_is_an_effect(self):
        assert verdict({"ci_lower": 0.5, "ci_upper": 2.0}) == "positive"
        assert verdict({"ci_lower": -2.0, "ci_upper": -0.5}) == "negative"

    def test_an_interval_spanning_zero_is_null(self):
        assert verdict({"ci_lower": -1.0, "ci_upper": 2.0}) == "null"

    def test_lower_is_better_flips_the_label_not_the_test(self):
        """A pinball loss that went DOWN is a win. The interval is the same;
        only the word changes, so the direction cannot be read off the sign."""
        assert verdict({"ci_lower": -2.0, "ci_upper": -0.5}, higher_is_better=False) == "positive"
        assert verdict({"ci_lower": 0.5, "ci_upper": 2.0}, higher_is_better=False) == "negative"

    def test_no_interval_is_unresolved_and_not_null(self):
        """ "The design could not answer" and "there is no effect" are different
        statements, and printing the first as the second is how a starved run
        gets read as a refutation."""
        assert verdict({"ci_lower": None, "ci_upper": None}) == "unresolved"
        assert verdict({}) == "unresolved"


class TestArmContrasts:
    def test_the_base_arm_is_not_contrasted_with_itself(self):
        arms = {"base": _arm(0.0, seed=6), "treat": _arm(-0.2, seed=6)}
        out = paired_arm_contrasts(arms, base="base", value_key="pinball", scale=1.0, higher_is_better=False)
        assert set(out) == {"treat"}
        assert out["treat"]["verdict"] == "positive"

    def test_an_arm_sharing_no_rows_is_reported_not_swallowed(self):
        base = _arm(0.0, seed=7)
        stray = [dict(r, item_id="other-" + str(r["item_id"])) for r in base]
        out = paired_arm_contrasts({"base": base, "stray": stray}, base="base", value_key="pinball", scale=1.0)
        assert out["stray"]["verdict"] == "no_shared_rows"

    def test_bookkeeping_keys_are_skipped(self):
        arms = {"base": _arm(0.0, seed=8), "_paired_vs_base": []}
        out = paired_arm_contrasts(arms, base="base", value_key="pinball")
        assert out == {}


class TestFoldLevelRecords:
    def test_the_fold_is_the_observation_and_the_cluster(self):
        recs = fold_level_records([0, 1, 2], [1.0, 2.0, 3.0], metric="pinball")
        assert [r["fold_id"] for r in recs] == [0, 1, 2]
        assert [r["item_id"] for r in recs] == [0, 1, 2]
        assert recs[0]["pinball"] == 1.0

    def test_it_pairs_and_resamples_at_fold_grain(self):
        a = fold_level_records(range(8), [1.0] * 8, metric="pinball")
        b = fold_level_records(range(8), [0.9] * 8, metric="pinball")
        out = paired_metric_difference(a, b, value_key="pinball")
        assert out["n_paired"] == 8
        assert out["n_clusters"] == 8
        assert out["mean_diff"] == pytest.approx(-0.1)

    def test_a_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="values has"):
            fold_level_records([0, 1], [1.0], metric="pinball")


class TestWithoutRecords:
    def test_it_strips_records_at_every_depth(self):
        obj = {"7": {"arm": {"dir_acc": 51.0, "records": [1, 2, 3]}, "list": [{"records": [4]}]}}
        assert without_records(obj) == {"7": {"arm": {"dir_acc": 51.0}, "list": [{}]}}

    def test_it_leaves_everything_else_alone(self):
        obj = {"a": [1, {"b": None}], "c": "records"}
        assert without_records(obj) == obj


# Every harness, and the module-level name whose presence proves it now runs a
# paired interval. The three fixed on 2026-08-07 call `paired_da_difference`
# directly; the ten fixed on 2026-08-08 go through the shared contrast step.
VERDICT_WIRING = {
    "ab_test_csfloat_basis": "paired_da_difference",
    "ab_test_item_metadata": "paired_da_difference",
    "ab_test_training_breadth": "paired_da_difference",
    "ab_test_volume_features": "paired_arm_contrasts",
    "ab_test_price_primitives": "paired_arm_contrasts",
    "ab_test_feature_contribution": "paired_arm_contrasts",
    "ab_test_supply_side": "paired_arm_contrasts",
    "ab_test_ensemble": "paired_arm_contrasts",
    "ab_test_regime": "paired_arm_contrasts",
    "ab_test_direction_labels": "paired_arm_contrasts",
    "ab_test_interval_sampling": "paired_arm_contrasts",
    "ab_test_q50_sampling": "paired_arm_contrasts",
    "ab_test_recency_weights": "paired_arm_contrasts",
}


@pytest.mark.parametrize("name,helper", sorted(VERDICT_WIRING.items()))
def test_every_harness_computes_a_paired_interval(name, helper):
    src = (SCRIPTS / f"{name}.py").read_text()
    assert helper in src, (
        f"{name} does not compute a paired, fold-clustered interval; a pooled delta or a fold win-count is not a test"
    )


@pytest.mark.parametrize("name", sorted(VERDICT_WIRING))
def test_no_harness_gates_on_a_fold_win_count(name):
    """The win count may still be PRINTED as context — it is cheap and it says
    something about consistency. It must not be a gate condition."""
    src = (SCRIPTS / f"{name}.py").read_text()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "context, not a gate" in stripped:
            continue
        assert not (("folds_won >=" in stripped or "won >= len(" in stripped) and "=" in stripped.split("#")[0]), (
            f"{name} still gates on a fold win count: {stripped!r}"
        )


def test_the_pinball_harnesses_score_lower_as_better():
    """Their metric is a loss. Passing `higher_is_better=True` would invert
    every verdict they print while leaving the interval correct — the kind of
    error no interval catches."""
    for name in ("ab_test_interval_sampling", "ab_test_q50_sampling", "ab_test_recency_weights"):
        src = (SCRIPTS / f"{name}.py").read_text()
        assert "higher_is_better=False" in src, name


class TestTheGuardsFoundInReview:
    """Four defects the 2026-08-08 code review found in this change itself."""

    def test_a_missing_cluster_key_is_not_reported_as_no_shared_rows(self):
        """`paired_arm_contrasts` caught bare `ValueError`, and
        `paired_metric_difference` raises it for two different things. A
        harness that forgot to thread `fold_id` got a confident, wrong
        "not measured on the same folds" — with the real message discarded by
        `format_paired`. The cluster-key guard has to stay loud; it is the
        2026-08-07 under-dispersion bug it protects against.
        """
        from backtest.paired_mde import NoPairedRows

        base = _arm(0.0, seed=20)
        unclustered = [{k: v for k, v in r.items() if k != "fold_id"} for r in base]
        with pytest.raises(ValueError, match="cluster key"):
            paired_arm_contrasts({"base": base, "broken": unclustered}, base="base", value_key="pinball")
        assert issubclass(NoPairedRows, ValueError)

    def test_no_shared_rows_is_still_caught(self):
        base = _arm(0.0, seed=21)
        stray = [dict(r, item_id="other-" + str(r["item_id"])) for r in base]
        out = paired_arm_contrasts({"base": base, "stray": stray}, base="base", value_key="pinball")
        assert out["stray"]["verdict"] == "no_shared_rows"

    def test_a_nan_bound_is_unresolved_not_null(self):
        """A NaN fails both `> 0` and `< 0` and fell through to `null` —
        printing "no effect" for a comparison that produced no number. One NaN
        row is enough: it propagates through `np.percentile` to both bounds."""
        assert verdict({"ci_lower": float("nan"), "ci_upper": float("nan")}) == "unresolved"
        assert verdict({"ci_lower": float("nan"), "ci_upper": 1.0}) == "unresolved"

    def test_a_nan_metric_does_not_read_as_a_null_result(self):
        a = fold_level_records(range(8), [1.0] * 8, metric="pinball")
        b = fold_level_records(range(8), [0.9] * 7 + [float("nan")], metric="pinball")
        out = paired_metric_difference(a, b, value_key="pinball")
        assert verdict(out, higher_is_better=False) == "unresolved"

    def test_a_short_keep_mask_raises_rather_than_mispairing(self):
        """`keep` is derived from the prediction arrays, which are different
        objects from `item_ids`. A short mask would silently score a subset
        while `ids[i]` labelled rows the mask never described."""
        with pytest.raises(ValueError, match="keep has 2, expected 3"):
            paired_records(
                item_ids=["a", "b", "c"],
                forecast_dates=["2026-01-01"] * 3,
                fold_id=0,
                keep=np.array([True, False]),
                direction_correct=np.array([True, True, False]),
            )


def test_the_gate_versions_its_series_on_the_embargo():
    """The purge flip is a discontinuity by the module's own docstring, so
    purged rows must not append to the un-purged `lgbm-v3-clustered` series:
    the dashboard trend and `backtest-triage` would read the step as a model
    regression with nothing stored to say otherwise."""
    src = (SCRIPTS / "walkforward_backtest.py").read_text()
    assert '"lgbm-v4-embargoed" if purge' in src
    assert '"purge": bool(purge),' in src, (
        "the persisted metrics must say which way the run went, not just the in-memory report"
    )


def test_the_win_count_harnesses_no_longer_headline_it():
    """`regime` and `ensemble` printed `Winner: X` as the last line an operator
    reads, and persisted only the boolean."""
    for name, boolean in (("ab_test_regime", "regime_wins"), ("ab_test_ensemble", "ens3_wins")):
        src = (SCRIPTS / f"{name}.py").read_text()
        assert "paired_verdict" in src, f"{name} persists no paired verdict"
        assert "not a test" in src, f"{name} still presents {boolean} without saying what it is"


class TestMergedShardsStillGetAVerdict:
    """`--arm` shards one arm per process, so no shard can contrast in-process.

    Before 2026-08-08 the merge step could only recompute the fold win count
    this change refutes. It now pairs at fold grain off `per_fold`, keyed on
    `val_start` — the window's own first date, which identifies the same fold
    in every arm where a positional index would not.
    """

    @staticmethod
    def _arm_folds(accs):
        return {"per_fold": [{"val_start": f"2026-{i + 1:02d}-01", "dir_acc": a} for i, a in enumerate(accs)]}

    def test_a_clear_effect_is_resolved_across_shards(self):
        from scripts.merge_price_primitives_ab import paired_verdicts

        merged = {
            7: {
                "baseline": self._arm_folds([50, 51, 49, 50, 52, 48]),
                "treatment": self._arm_folds([55, 56, 54, 55, 57, 53]),
                "placebo": self._arm_folds([50, 51, 49, 50, 52, 48]),
            }
        }
        out = paired_verdicts(merged)
        assert out[7]["treatment"]["verdict"] == "positive"
        assert out[7]["treatment"]["mean_diff"] == pytest.approx(5.0)
        assert out[7]["placebo"]["verdict"] == "null"

    def test_one_shared_fold_is_unresolved_not_null(self):
        from scripts.merge_price_primitives_ab import paired_verdicts

        merged = {30: {a: self._arm_folds([50]) for a in ("baseline", "treatment", "placebo")}}
        out = paired_verdicts(merged)
        assert out[30]["treatment"]["verdict"] == "unresolved"

    def test_a_horizon_missing_an_arm_is_skipped_not_half_reported(self):
        """The ship rule is treatment-vs-baseline AND treatment-vs-placebo;
        half of it is not a weaker version of it."""
        from scripts.merge_price_primitives_ab import paired_verdicts

        assert paired_verdicts({30: {"baseline": self._arm_folds([50, 51, 52])}}) == {}

    def test_format_paired_does_not_print_a_nan_estimate(self):
        assert "nan" not in format_paired({"verdict": "unresolved", "n_clusters": 1})
