"""The paired read of two `model-diagnostics.yml` runs.

Every arm verdict in this project has come from hand-grepping two runs' logs, and
the workflow's own `grep -E` dropped the cross-sectional line whenever an arm
worked (`2026-08-11-rank-arms-on-the-clean-anchor-cohort.md`). These tests pin the
parse against the exact format `replay_serving` prints and the exact keys
`_summarise_rank_ic` writes, so a format change fails here rather than silently
producing a table of the wrong rows.
"""

import pytest
from scripts.compare_diagnostics import (
    cv_table_note,
    cv_tied_row,
    pair_replay_rows,
    parse_replay_log,
)

# Verbatim from `replay_serving.py`'s basis-sweep block: the tied/deviating rows
# share the table with the five basis rows and are told apart only by the label.
REPLAY_LOG = """
BAND COVERAGE @ 2026-04-15   (low <= realised <= high; conformal_centre: 3d=served)
   h      cov%   target%
   3      82.1      80.0

LABEL BASIS SWEEP @ 2026-04-15   (same served mids; 'cv' is prepare_targets' own basis)
   h      basis       n   rankIC  vs served
   3     served     669   0.1102
   3         cv     669   0.2456    +0.1354
   3   num_only     669   0.1180    +0.0078
   3       tied     220   0.1321    -0.0206  (cv +0.1115)
   3  deviating     449  -0.2014    +0.4260  (cv +0.2246)

LABEL BASIS SWEEP @ 2026-05-16   (same served mids; 'cv' is prepare_targets' own basis)
   h      basis       n   rankIC  vs served
   3     served     700   0.1000
   3       tied     240   0.1400    -0.0100  (cv +0.1300)
   3  deviating       2   too few
"""


def test_parses_the_tied_row_and_leaves_the_basis_rows_out():
    rows = parse_replay_log(REPLAY_LOG)

    tied = [r for r in rows if r["cohort"] == "tied"]
    assert [(r["anchor"], r["horizon"], r["n"], r["served_ic"], r["cv_ic"]) for r in tied] == [
        ("2026-04-15", 3, 220, 0.1321, 0.1115),
        ("2026-05-16", 3, 240, 0.1400, 0.1300),
    ]
    # `served`, `cv`, `num_only` are label bases, not cohorts. Pooling them into
    # the same table is the contaminated read this harness exists to avoid.
    assert {r["cohort"] for r in rows} == {"tied", "deviating"}


def test_a_too_few_row_is_dropped_rather_than_read_as_zero():
    rows = parse_replay_log(REPLAY_LOG)

    deviating = [r for r in rows if r["cohort"] == "deviating"]
    assert [r["anchor"] for r in deviating] == ["2026-04-15"]


def test_cv_tied_row_is_none_when_the_artifact_predates_the_column():
    meta = {"cv_results": {"7": {"mean_rank_ic": 0.1293}}}

    assert cv_tied_row(meta, 7) is None


def test_cv_tied_row_reads_the_tied_keys_and_never_the_pooled_ones():
    meta = {
        "cv_results": {
            "7": {
                "mean_rank_ic": 0.1293,
                "mean_naive_rank_ic": 0.1653,
                "mean_rank_ic_tied": 0.0812,
                "mean_naive_rank_ic_tied": 0.0640,
                "rank_ic_edge_vs_naive_tied": 0.0172,
                "tied_rows": 41022,
                "tied_dates": 6,
            }
        }
    }

    row = cv_tied_row(meta, 7)

    assert row == {
        "horizon": 7,
        "rank_ic_tied": 0.0812,
        "naive_rank_ic_tied": 0.0640,
        "edge_vs_naive_tied": 0.0172,
        "tied_rows": 41022,
        "tied_dates": 6,
    }


def test_an_empty_cv_table_says_why_rather_than_printing_a_bare_header():
    """An empty table and a broken parse look identical on a terminal.

    Every artifact before 2026-08-11 lacks `mean_rank_ic_tied`, so this is the
    expected state for a re-read of an older pair -- and the one case where
    silence would be read as "the harness is broken".
    """
    assert cv_table_note({}, {}, (3, 7)) is not None
    assert "predate" in cv_table_note({}, {}, (3, 7))
    # A single horizon carrying the column is enough for the table to stand on
    # its own; the note is for the all-missing case only.
    meta = {"cv_results": {"3": {"mean_rank_ic_tied": 0.08}}}
    assert cv_table_note({3: meta}, {}, (3, 7)) is None


def test_pairing_differences_the_arm_against_the_control_on_matching_cells():
    control = [{"anchor": "2026-04-15", "horizon": 3, "cohort": "tied", "n": 220, "served_ic": 0.1321, "cv_ic": 0.1115}]
    arm = [{"anchor": "2026-04-15", "horizon": 3, "cohort": "tied", "n": 220, "served_ic": 0.1500, "cv_ic": 0.1400}]

    paired = pair_replay_rows(control, arm)

    assert len(paired) == 1
    assert paired[0]["delta_served_ic"] == pytest.approx(0.0179)
    assert paired[0]["n_control"] == 220
    assert paired[0]["n_arm"] == 220


def test_pairing_reports_a_cell_the_two_runs_do_not_share_instead_of_dropping_it():
    """A dropped cell is how a paired read silently becomes an unpaired one."""
    control = [{"anchor": "2026-04-15", "horizon": 3, "cohort": "tied", "n": 220, "served_ic": 0.1321, "cv_ic": 0.1115}]
    arm = [{"anchor": "2026-05-16", "horizon": 3, "cohort": "tied", "n": 240, "served_ic": 0.1400, "cv_ic": 0.1300}]

    paired = pair_replay_rows(control, arm)

    unmatched = [r for r in paired if r["delta_served_ic"] is None]
    assert len(unmatched) == 2
    assert {r["anchor"] for r in unmatched} == {"2026-04-15", "2026-05-16"}


def test_pairing_flags_a_cell_whose_cohort_membership_moved():
    """`n` differing on a tied cell means the two runs scored different items.

    The tied mask is arm-invariant by construction, so this should never fire --
    which is exactly why it must be visible if it does, rather than differenced
    into a number that looks paired.
    """
    control = [{"anchor": "2026-04-15", "horizon": 3, "cohort": "tied", "n": 220, "served_ic": 0.1321, "cv_ic": 0.1115}]
    arm = [{"anchor": "2026-04-15", "horizon": 3, "cohort": "tied", "n": 198, "served_ic": 0.1500, "cv_ic": 0.1400}]

    paired = pair_replay_rows(control, arm)

    assert paired[0]["cohort_moved"] is True
