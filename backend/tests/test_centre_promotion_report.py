from datetime import date, datetime, timedelta

BASE = date(2026, 9, 1)
T = datetime(2026, 9, 8, 12, 0, 0)


def _dates(n):
    return [BASE + timedelta(days=i) for i in range(n)]


def _frame(n_dates=20, cand_err=1.0, prod_err=2.0, cand_cov=True, prod_cov=True, width_pct=25.0, fps=("a" * 64,)):
    dates = _dates(n_dates)
    production, candidate = [], []
    for d in dates:
        for i in range(5):
            base = 40.0
            actual = base + prod_err
            p_mid = base
            half = p_mid * width_pct / 200
            production.append(
                {
                    "item_id": i,
                    "forecast_date": d,
                    "base_price": base,
                    "actual_price": actual,
                    "resolved_at": T,
                    "mid": p_mid,
                    "low": p_mid - half,
                    "high": p_mid + half,
                    "in_interval": prod_cov,
                }
            )
            c_mid = base + cand_err
            chalf = c_mid * width_pct / 200
            candidate.append(
                {
                    "item_id": i,
                    "forecast_date": d,
                    "base_price": base,
                    "actual_price": actual,
                    "resolved_at": T,
                    "candidate_name": "last_price",
                    "centre_price": c_mid,
                    "low": c_mid - chalf,
                    "high": c_mid + chalf,
                    "version": "v1",
                    "fingerprint": fps[(i % len(fps))],
                }
            )
    return {
        "production": production,
        "candidate": candidate,
        "expected_batches": n_dates,
        "observed_batches": n_dates,
    }


def _run_main(monkeypatch, tmp_path, frame, horizon=7):
    import scripts.centre_promotion_report as rep

    monkeypatch.setattr(rep, "load_horizon_frame", lambda db, h: frame)
    monkeypatch.setattr(rep, "SessionLocal", lambda: object())
    json_out = str(tmp_path / "centre.json")
    md_out = str(tmp_path / "centre.md")
    code = rep.main(["--horizon", str(horizon), "--json-out", json_out, "--markdown-out", md_out])
    import json

    with open(json_out) as f:
        payload = json.load(f)
    with open(md_out) as f:
        markdown = f.read()
    return code, payload, markdown


def test_pass_writes_json_and_markdown(tmp_path, monkeypatch):
    code, payload, markdown = _run_main(monkeypatch, tmp_path, _frame())
    assert code == 0
    assert payload["verdict"] == "PASS_FOR_MANUAL_PROMOTION"
    assert payload["delta_mae"]["upper"] <= 0
    assert "# Centre promotion report" in markdown
    assert "PASS_FOR_MANUAL_PROMOTION" in markdown


def test_unresolved_exits_1(tmp_path, monkeypatch):
    dates = _dates(20)
    frame = _frame()
    for r in frame["candidate"]:
        first_half = r["forecast_date"] < dates[10]
        r["centre_price"] = 40.0 + (0.0 if first_half else 4.0)
    code, payload, _ = _run_main(monkeypatch, tmp_path, frame)
    assert code == 1
    assert payload["verdict"] == "UNRESOLVED"


def test_rejected_exits_3(tmp_path, monkeypatch):
    code, payload, _ = _run_main(monkeypatch, tmp_path, _frame(cand_err=5.0, prod_err=1.0))
    assert code == 3
    assert payload["verdict"] == "REJECTED"


def test_insufficient_exits_1(tmp_path, monkeypatch):
    code, payload, _ = _run_main(monkeypatch, tmp_path, _frame(n_dates=19))
    assert code == 1
    assert payload["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_integrity_exits_2(tmp_path, monkeypatch):
    code, payload, _ = _run_main(monkeypatch, tmp_path, _frame(fps=("a" * 64, "b" * 64)))
    assert code == 2
    assert payload["verdict"] == "DATA_INTEGRITY_FAILURE"


def test_report_never_mutates_champion_config(tmp_path, monkeypatch):
    from models.centre_policy import CENTRE_CHAMPIONS

    before = dict(CENTRE_CHAMPIONS)
    _run_main(monkeypatch, tmp_path, _frame())
    assert dict(CENTRE_CHAMPIONS) == before
