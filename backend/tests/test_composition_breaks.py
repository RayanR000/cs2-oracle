"""_composition_break_dates: a market-wide switch in which sources vote.

The item-count rule (_collection_shift_dates) cannot see 2026-04-16: every
train-universe item switched from buff163/csfloat/youpin to steam_17mafo with
the item count flat (+0.02%). Nor, on the train universe, 2026-01-01, where the
NULL-sourced legacy series ends. See
docs/specs/2026-09-30-composition-break-calendar-design.md.
"""

from datetime import date, timedelta

import pandas as pd
from models.forecaster import ItemForecaster

D0 = date(2026, 4, 14)


def _frame(n_items: int, sets_by_day: list[list[str]]) -> pd.DataFrame:
    rows = []
    for k, sets in enumerate(sets_by_day):
        for i in range(n_items):
            rows.append({"item_id": f"i{i}", "date": D0 + timedelta(days=k), "price": 1.0, "source_set": sets[i % len(sets)]})
    df = pd.DataFrame(rows)
    df["source_set"] = df["source_set"].astype("category")
    return df


def test_full_switch_fires_on_the_switch_day_only():
    df = _frame(100, [["a|b"], ["a|b"], ["c"], ["c"]])
    assert ItemForecaster._composition_break_dates(df) == frozenset({D0 + timedelta(days=2)})


def test_ordinary_churn_does_not_fire():
    # ~56% of items change set: the highest ordinary day measured on the train universe.
    day1 = ["a|b"] * 100
    day2 = ["a"] * 56 + ["a|b"] * 44
    assert ItemForecaster._composition_break_dates(_frame(100, [day1, day2])) == frozenset()


def test_ninety_percent_is_the_boundary():
    day1 = ["a"] * 100
    at = ["b"] * 90 + ["a"] * 10
    below = ["b"] * 89 + ["a"] * 11
    assert ItemForecaster._composition_break_dates(_frame(100, [day1, at])) == frozenset({D0 + timedelta(days=1)})
    assert ItemForecaster._composition_break_dates(_frame(100, [day1, below])) == frozenset()


def test_small_cross_section_never_fires():
    assert ItemForecaster._composition_break_dates(_frame(24, [["a"], ["b"]])) == frozenset()


def test_null_sourced_history_is_one_composition():
    df = _frame(100, [["<null>"], ["<null>"], ["<null>"]])
    assert ItemForecaster._composition_break_dates(df) == frozenset()


def test_the_null_to_labelled_year_boundary_fires():
    df = _frame(100, [["<null>"], ["<null>"], ["aggregator_sync"]])
    assert ItemForecaster._composition_break_dates(df) == frozenset({D0 + timedelta(days=2)})


def test_new_items_do_not_count_as_changed():
    # Day 2 adds 1,000 items that did not exist on day 1; only the 100 paired items are judged.
    paired = _frame(100, [["a"], ["a"]])
    newcomers = _frame(1000, [["a"], ["b"]])
    newcomers = newcomers[newcomers["date"] == D0 + timedelta(days=1)].assign(item_id=lambda d: "new" + d["item_id"])
    df = pd.concat([paired, newcomers], ignore_index=True)
    assert ItemForecaster._composition_break_dates(df) == frozenset()


def test_missing_column_is_empty_not_an_error():
    df = _frame(100, [["a"], ["b"]]).drop(columns=["source_set"])
    assert ItemForecaster._composition_break_dates(df) == frozenset()


def test_a_price_crash_with_a_stable_source_set_does_not_fire():
    df = _frame(100, [["a|b"], ["a|b"]])
    df.loc[df["date"] == D0 + timedelta(days=1), "price"] = 0.5
    assert ItemForecaster._composition_break_dates(df) == frozenset()
