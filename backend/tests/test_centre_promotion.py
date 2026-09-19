from datetime import date, timedelta

BASE = date(2026, 9, 1)


def _dates(n):
    return [BASE + timedelta(days=i) for i in range(n)]


def _rows(dates, items=5, err=1.0, covered=True, width=10.0):
    return [
        {
            "forecast_date": d,
            "item_id": i,
            "abs_error": err,
            "in_interval": covered,
            "width_pct": width,
        }
        for d in dates
        for i in range(items)
    ]


def _evaluate(last, gbm, **overrides):
    from backtest.promotion import evaluate_centre_promotion

    params = dict(
        horizon=7,
        rows_last=last,
        rows_gbm=gbm,
        candidate_versions={"v1"},
        candidate_fingerprints={"a" * 64},
        expected_batches=20,
        observed_batches=20,
        resolution_ok=True,
    )
    params.update(overrides)
    return evaluate_centre_promotion(**params)


def _passing():
    dates = _dates(20)
    return _rows(dates, err=1.0), _rows(dates, err=2.0)


def test_nineteen_dates_is_insufficient():
    dates = _dates(19)
    result = _evaluate(_rows(dates, err=1.0), _rows(dates, err=2.0), expected_batches=19, observed_batches=19)
    assert result.verdict == "INSUFFICIENT_EVIDENCE"


def test_clear_win_with_neutral_coverage_passes():
    last, gbm = _passing()
    result = _evaluate(last, gbm)
    assert result.verdict == "PASS_FOR_MANUAL_PROMOTION"


def test_win_crossing_zero_is_unresolved():
    dates = _dates(20)
    last = _rows(dates[:10], err=0.0) + _rows(dates[10:], err=4.0)
    gbm = _rows(dates, err=2.0)
    result = _evaluate(last, gbm)
    assert result.verdict == "UNRESOLVED"


def test_coverage_regression_blocks_pass():
    dates = _dates(20)
    last = _rows(dates, err=1.0, covered=False)
    gbm = _rows(dates, err=2.0, covered=True)
    result = _evaluate(last, gbm)
    assert result.verdict == "UNRESOLVED"


def test_clear_loss_is_rejected():
    dates = _dates(20)
    result = _evaluate(_rows(dates, err=3.0), _rows(dates, err=1.0))
    assert result.verdict == "REJECTED"


def test_low_overlap_fails_eligibility():
    dates = _dates(20)
    last, gbm = _passing()
    dropped = [r for r in gbm if not (r["forecast_date"] == dates[0] and r["item_id"] < 2)]
    result = _evaluate(last, dropped)
    assert result.verdict == "DATA_INTEGRITY_FAILURE"


def test_low_completeness_fails_eligibility():
    last, gbm = _passing()
    result = _evaluate(last, gbm, observed_batches=15)
    assert result.verdict == "DATA_INTEGRITY_FAILURE"


def test_mixed_fingerprints_fail_eligibility():
    last, gbm = _passing()
    result = _evaluate(last, gbm, candidate_fingerprints={"a" * 64, "b" * 64})
    assert result.verdict == "DATA_INTEGRITY_FAILURE"


def test_resolution_mismatch_fails_eligibility():
    last, gbm = _passing()
    result = _evaluate(last, gbm, resolution_ok=False)
    assert result.verdict == "DATA_INTEGRITY_FAILURE"


def test_thousands_of_rows_still_fail_without_dates():
    dates = _dates(5)
    last = _rows(dates, items=2000, err=1.0)
    gbm = _rows(dates, items=2000, err=2.0)
    result = _evaluate(last, gbm, expected_batches=5, observed_batches=5)
    assert result.verdict == "INSUFFICIENT_EVIDENCE"
