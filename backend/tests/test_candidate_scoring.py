import numpy as np


def test_point_is_the_mean_of_daily_means():
    """One 10,000-row date must not outweigh nineteen 100-row dates."""
    from backtest.candidate_scoring import paired_daily_interval

    a, b, dates = [], [], []
    for d in range(20):
        n = 10_000 if d == 0 else 100
        delta = 5.0 if d == 0 else -1.0
        a.extend([delta] * n)
        b.extend([0.0] * n)
        dates.extend([f"2026-09-{d + 1:02d}"] * n)

    result = paired_daily_interval(a, b, dates)
    assert result.n_dates == 20
    assert result.point == (5.0 + 19 * -1.0) / 20
    # A row-pooled mean would be dominated by the big date (~+4.7).
    assert result.point < 0.0


def test_bootstrap_is_deterministic_under_the_fixed_seed():
    from backtest.candidate_scoring import paired_daily_interval

    rng = np.random.default_rng(7)
    a = list(rng.normal(0.5, 1.0, 500))
    b = list(rng.normal(0.0, 1.0, 500))
    dates = [f"d{i % 20}" for i in range(500)]
    first = paired_daily_interval(a, b, dates)
    second = paired_daily_interval(a, b, dates)
    assert (first.lower, first.upper) == (second.lower, second.upper)


def test_single_date_yields_no_interval():
    from backtest.candidate_scoring import paired_daily_interval

    result = paired_daily_interval([1.0, 2.0], [0.5, 1.5], ["d", "d"])
    assert result.n_dates == 1
    assert result.lower is None and result.upper is None
