"""The walkforward A/B harnesses must embargo the train side, like production.

The `ab_test_*` walkforward family built its folds as `train = every date <
val_start` with no purge gap. A row dated `d` is labelled with the price at
`d + horizon`, so every training row in the last `horizon` days before the
boundary carried a target resolved from *inside* the validation window.
Production never had this bug: `_compute_cv_splits(..., purge_days=...)`
and `_purge_overlapping_train_rows` both purge, and
`ab_test_direction_labels.py` / `ab_test_recency_weights.py` call the former.
The walkforward family had simply diverged.

The overlap is symmetric across arms, which is why it went unnoticed -- but only
an arm that can *locate* the overlapping rows exploits it, and only a date-level
or date-proxy column can. The often-quoted size of that -- an event-calendar arm
at h=30 going +12.1pp unpurged -> +6.1pp purged -- comes from the external
review (docs/research/2026-08-07-cs2-forecasting-research.md) and has **never
been replicated in this repo**; `2026-08-07-next-steps.md` says so explicitly,
and three harness docstrings still present it as a local measurement. It is a
reason to embargo, not evidence about any harness here.

Three harnesses were fixed 2026-08-07 and the remaining six on 2026-08-08,
along with two changes to the rule itself: the band widened from `horizon` to
`embargo_days(horizon)`, and `walkforward_backtest.py` -- the published gate --
went from opt-in to embargoed by default.

Three invariants are locked down here:
  1. the embargo removes exactly the `embargo_days(horizon)` band before
     `val_start`, so every surviving train row's label lands strictly before the
     window opens -- and the 13 days on top of the horizon are DERIVED from the
     resolution constants, not typed in;
  2. the VALIDATION window is untouched. Purging the val side would shrink the
     21-day window and, at horizon=30, empty it outright;
  3. every harness that builds folds is wired to one of the two production
     helpers, and none of them re-derives the rule locally.
"""

from __future__ import annotations

import inspect
import re
from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
import scripts.archive.ab_test_csfloat_basis as cf
import scripts.archive.ab_test_direction_labels as dirlab
import scripts.archive.ab_test_ensemble as ensemble
import scripts.archive.ab_test_feature_contribution as featcon
import scripts.archive.ab_test_interval_sampling as interval
import scripts.archive.ab_test_item_metadata as meta
import scripts.archive.ab_test_price_primitives as primitives
import scripts.archive.ab_test_q50_sampling as q50
import scripts.archive.ab_test_recency_weights as recency
import scripts.archive.ab_test_regime as regime
import scripts.archive.ab_test_supply_side as supply
import scripts.archive.ab_test_training_breadth as breadth
import scripts.archive.ab_test_volume_features as volume
import scripts.archive.walkforward_backtest as wf
from models.forecaster import ItemForecaster, embargo_days

# The three fixed 2026-08-07. They share a code shape the assertions below
# match on literally, which the six fixed 2026-08-08 do not.
AB_HARNESSES = pytest.mark.parametrize(
    "module",
    [cf, meta, breadth],
    ids=["csfloat_basis", "item_metadata", "training_breadth"],
)

# Every harness that rolls its own `split_idx`-based folds, and therefore has
# to call `_purge_overlapping_train_rows` itself. Nine of the thirteen.
ROLLED_FOLD_HARNESSES = pytest.mark.parametrize(
    "module",
    [cf, meta, breadth, volume, primitives, featcon, supply, regime, ensemble],
    ids=[
        "csfloat_basis",
        "item_metadata",
        "training_breadth",
        "volume_features",
        "price_primitives",
        "feature_contribution",
        "supply_side",
        "regime",
        "ensemble",
    ],
)

# The four that take their folds from production's `_compute_cv_splits`, which
# embargoes on the `purge_days` it is handed. They must hand it the full
# `embargo_days(horizon)` -- passing the bare horizon is the old, too-narrow
# band, and it is silent.
CV_SPLIT_HARNESSES = pytest.mark.parametrize(
    "module",
    [interval, q50, recency, dirlab],
    ids=["interval_sampling", "q50_sampling", "recency_weights", "direction_labels"],
)

HORIZONS = [3, 7, 14, 30]


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(
        db_session=MagicMock(),
        model_dir=str(tmp_path_factory.mktemp("saved_models")),
    )


def _daily_frame(start="2024-01-01", days=400, items=(1, 2)):
    """One row per item per calendar day -- the geometry the harnesses assume."""
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame([{"item_id": i, "date": d.date(), "price": 10.0} for i in items for d in dates])


class TestTheEmbargoRemovesTheRightRows:
    """Semantics, at the harnesses' own fold geometry (21-day val window)."""

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_train_stops_an_embargo_band_before_the_window(self, forecaster, horizon):
        df = _daily_frame()
        dates = sorted(df["date"].unique())
        window_end = len(dates) * 2 // 3 + 1
        val_dates = dates[window_end : window_end + cf.VAL_WINDOW_DAYS]
        val_start = val_dates[0]

        unpurged = df[df["date"] <= dates[window_end - 1]]
        purged = forecaster._purge_overlapping_train_rows(unpurged, val_start, horizon)

        # Exactly the `embargo_days(horizon)` calendar days before val_start
        # are gone — `horizon` for the label's nominal date plus 13 for the
        # resolved anchor's carry behind it.
        band = embargo_days(horizon)
        dropped_days = set(pd.to_datetime(unpurged["date"]).dt.date) - set(pd.to_datetime(purged["date"]).dt.date)
        assert len(dropped_days) == band
        assert max(dropped_days) == val_start - timedelta(days=1)
        assert min(dropped_days) == val_start - timedelta(days=band)

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_every_surviving_label_lands_before_the_window_opens(self, forecaster, horizon):
        df = _daily_frame()
        dates = sorted(df["date"].unique())
        window_end = len(dates) * 2 // 3 + 1
        val_start = dates[window_end]

        unpurged = df[df["date"] <= dates[window_end - 1]]
        leaking = [d for d in unpurged["date"] if d + timedelta(days=horizon) >= val_start]
        assert len(leaking) > 0, "precondition: the un-purged split leaks"

        purged = forecaster._purge_overlapping_train_rows(unpurged, val_start, horizon)
        assert not purged.empty
        assert all(d + timedelta(days=horizon) < val_start for d in purged["date"])

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_the_validation_window_width_is_unchanged(self, forecaster, horizon):
        """The purge is train-side only. At h=30 a val-side purge would empty a
        21-day window entirely; the width must not move at any horizon."""
        df = _daily_frame()
        dates = sorted(df["date"].unique())
        window_end = len(dates) * 2 // 3 + 1
        val_dates = dates[window_end : window_end + cf.VAL_WINDOW_DAYS]
        val_df = df[df["date"].isin(val_dates)]

        forecaster._purge_overlapping_train_rows(df[df["date"] <= dates[window_end - 1]], val_dates[0], horizon)

        assert len(val_dates) == cf.VAL_WINDOW_DAYS == 21
        assert val_df["date"].nunique() == 21
        assert len(df[df["date"].isin(val_dates)]) == len(val_df)

    def test_the_purge_is_a_no_op_on_a_gap_wider_than_the_horizon(self, forecaster):
        """Nothing is dropped when the train side already stops early enough."""
        df = _daily_frame(days=60)
        val_start = df["date"].max() + timedelta(days=45)
        purged = forecaster._purge_overlapping_train_rows(df, val_start, 30)
        assert len(purged) == len(df)


class TestTheEmbargoWidthIsDerived:
    """`embargo_days` is `horizon + 13`, and the 13 is derived, not typed.

    Added 2026-08-08. The band used to be the bare horizon, which assumed the
    label at `d + horizon` is a point observation. It is not: both legs resolve
    through `price_resolution.resolve_anchors`, which medians the last
    SMOOTH_WINDOW observations, admits one from up to MAX_WINDOW_SPAN_DAYS
    before the anchor, and pairs against features whose lags reach back
    LAG_TOLERANCE_DAYS. If any of those three constants moves, the embargo has
    to move with it -- which is what makes the derivation, not the number, the
    thing worth locking down.
    """

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_the_band_is_the_horizon_plus_the_carry(self, horizon):
        assert embargo_days(horizon) == horizon + 13

    def test_the_carry_is_the_sum_of_the_three_resolution_constants(self):
        from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS, SMOOTH_WINDOW

        carry = ItemForecaster.LAG_TOLERANCE_DAYS + SMOOTH_WINDOW + MAX_WINDOW_SPAN_DAYS
        assert embargo_days(0) == carry

    def test_at_30d_the_embargo_exceeds_the_validation_window(self):
        """The documented cost, asserted so it is not mistaken for a bug.

        A 30d fold cannot be built out of 30 days of history. That was always
        true; the bare-horizon purge merely hid it.
        """
        assert embargo_days(30) > ItemForecaster.VALIDATION_WINDOW_DAYS


@ROLLED_FOLD_HARNESSES
class TestTheAbHarnessesAreWired:
    """Nine harnesses build folds by hand and must purge by hand.

    Six of them (volume_features, price_primitives, feature_contribution,
    supply_side, regime, ensemble) had no embargo of any kind until
    2026-08-08 -- not a narrow one, none. Their published deltas all predate
    the fix.
    """

    def test_the_harness_calls_productions_purge(self, module):
        src = inspect.getsource(module)
        assert "_purge_overlapping_train_rows(" in src, (
            f"{module.__name__} must reuse "
            "ItemForecaster._purge_overlapping_train_rows rather than "
            "re-deriving the embargo -- a second implementation of the same "
            "rule is how this drifted from production in the first place"
        )

    def test_no_unpurged_train_frame_survives(self, module):
        """The exact defect: a training frame sliced straight off the window
        bound, with nothing between it and the validation set."""
        src = inspect.getsource(module)
        offenders = re.findall(r"^\s*train_df\s*=\s*(?:sub|tdf)\[[^\]]*\]\s*$", src, flags=re.M)
        assert not offenders, f"{module.__name__} still builds an un-embargoed training frame: {offenders}"

    def test_the_purge_is_unconditional(self, module):
        """These are research tools: there is no un-purged mode to fall back to.

        `walkforward_backtest.py` is the one module that keeps an escape hatch,
        because it has a stored series to read back against; it is not in this
        list.
        """
        src = inspect.getsource(module)
        for token in ("--no-purge", "--purge-days", "purge=False", "if purge"):
            assert token not in src, f"{module.__name__} exposes {token!r}; the embargo must be the only behaviour here"

    def test_the_val_side_is_never_purged(self, module):
        src = inspect.getsource(module)
        for line in src.splitlines():
            if "_purge_overlapping_train_rows" in line:
                assert "val_df" not in line, (
                    f"{module.__name__} purges the validation frame; at h=30 that empties the 21-day window"
                )

    def test_the_purge_is_handed_the_windows_first_date(self, module):
        """`val_dates[0]`, not `window_end` or a train-side date. The band is
        measured backwards from where validation opens."""
        src = inspect.getsource(module)
        calls = re.findall(r"_purge_overlapping_train_rows\((.*?)\)\s*$", src, flags=re.S | re.M)
        assert calls, module.__name__
        for call in calls:
            if "val_start" in call or "val_dates[0]" in call:
                continue
            pytest.fail(
                f"{module.__name__} purges against {call.strip()!r} rather than the validation window's first date"
            )


@AB_HARNESSES
class TestTheOriginalThreeKeepTheirShape:
    """Shape assertions that only hold for the 2026-08-07 three."""

    def test_the_val_mask_is_built_off_the_window_bounds(self, module):
        assert re.search(r"sub_days\s*>=\s*dates_dt\[window_end\]", inspect.getsource(module))


@CV_SPLIT_HARNESSES
class TestTheCvSplitHarnessesPassTheFullEmbargo:
    def test_it_does_not_pass_the_bare_horizon(self, module):
        """`purge_days=horizon` is the pre-2026-08-08 band. It is 13 days too
        narrow and nothing about the call site says so."""
        src = inspect.getsource(module)
        assert "purge_days=horizon)" not in src, (
            f"{module.__name__} passes the bare horizon as the embargo; it must pass embargo_days(horizon)"
        )

    def test_it_derives_the_band_from_production(self, module):
        src = inspect.getsource(module)
        assert "embargo_days" in src, (
            f"{module.__name__} must take the band from models.forecaster.embargo_days rather than spelling one"
        )


class TestTheProductionGateEmbargoesByDefault:
    """`walkforward_backtest` is the published Backtest Accuracy gate, and
    since 2026-08-08 the number it publishes is the purged one.

    The flag survives as an opt-OUT so a stored pre-flip run can still be read
    like for like; it is not a mode anyone should publish from.
    """

    def test_purge_is_a_parameter(self):
        assert "purge" in inspect.signature(wf.run_walkforward).parameters

    def test_purge_defaults_to_on(self):
        assert inspect.signature(wf.run_walkforward).parameters["purge"].default is True

    def test_the_cli_exposes_an_opt_out_flag(self):
        src = inspect.getsource(wf.build_parser)
        assert '"--no-purge"' in src and 'action="store_false"' in src
        assert '"--purge"' not in src, (
            "an opt-in --purge alongside the opt-out reads as if the embargo "
            "were still a choice; there is one default and one escape hatch"
        )

    def test_the_cli_default_agrees_with_the_function_default(self):
        """`store_false` without `set_defaults` would still default True, but
        only by argparse convention. The gate's default is load-bearing enough
        to state twice and check once."""
        parser = wf.build_parser()
        assert parser.parse_args([]).purge is True
        assert parser.parse_args(["--no-purge"]).purge is False

    def test_the_gate_uses_productions_purge_helper(self):
        src = inspect.getsource(wf.run_walkforward)
        assert "_purge_overlapping_train_rows(" in src
        # Train side only -- val_df is built before the purge and not touched.
        purge_lines = [l for l in src.splitlines() if "_purge_overlapping_train_rows" in l]
        assert purge_lines and all("val_df" not in l for l in purge_lines)

    def test_the_report_records_which_way_it_ran(self):
        """Two runs of the gate differ only by this flag; the report has to say."""
        assert '"purge": bool(purge)' in inspect.getsource(wf.run_walkforward)


class TestTheGateCanActuallyPersist:
    """The 2026-08-08 run scored all four horizons and stored none of them.

    `run_walkforward` opens a session, closes it right after `fetch_events`,
    then reuses it for the writes at the end. At the default 500 items the fold
    loop between those two points runs ~35 minutes, so the write checks out a
    pooled connection the Supabase pooler dropped long ago and dies with
    `SSL SYSCALL error: EOF detected` — with every number already computed and
    nothing to show for it. `prediction_type='walkforward_backtest'` had zero
    rows in production as a result.
    """

    def test_the_write_block_opens_its_own_session(self):
        src = inspect.getsource(wf.run_walkforward)
        head, _, tail = src.partition("if not skip_db:")
        assert tail, "the write block moved; this test needs updating"
        assert "SessionLocal()" in tail, (
            "the DB write must open a fresh session — the one from the top of "
            "the function was closed before the fold loop ran"
        )

    def test_the_early_session_is_still_closed_promptly(self):
        """Holding it open for the whole run is the other way to 'fix' this,
        and it is worse: an idle transaction against the pooler for 35 minutes.
        Closing early and reopening late is the intended shape."""
        src = inspect.getsource(wf.run_walkforward)
        head, _, _ = src.partition("if not skip_db:")
        assert "events_df = forecaster.fetch_events()" in head
        assert "db.close()" in head
