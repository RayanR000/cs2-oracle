"""Offline direction benchmark: report shape only.

The heavy main() needs the price archive and is not run here; what is
worth pinning is that the report carries the full metric quartet together,
so a PT verdict can never be quoted without its baselines.
"""


def test_report_format_carries_the_quartet():
    from scripts.direction_benchmark import format_direction_report

    out = format_direction_report(
        {
            "horizon": 7,
            "n_dates": 24,
            "n_rows": 12000,
            "directional_accuracy": 49.6,
            "constant_call_accuracy": 51.7,
            "constant_call_direction": "down",
            "realised_down_rate": 51.7,
            "pt_excess_pp": 0.12,
            "pt_t_stat": 0.9,
            "pt_p_value": 0.37,
            "pt_verdict": "no_skill",
        }
    )
    assert "Pesaran-Timmermann" in out
    assert "directional_accuracy" in out
    assert "realised_down_rate" in out
    assert "constant_call_accuracy" in out


def test_benchmark_module_imports_without_side_effects():
    import scripts.direction_benchmark as m

    assert hasattr(m, "main")
    assert hasattr(m, "format_direction_report")
