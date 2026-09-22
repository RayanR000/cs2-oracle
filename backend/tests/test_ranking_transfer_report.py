from datetime import date, datetime, timedelta

BASE = date(2026, 9, 1)
T = datetime(2026, 9, 8, 12, 0, 0)


def _dates(n):
    return [BASE + timedelta(days=i) for i in range(n)]


def _frame(n_dates=20, mode="supported", items=100):
    dates = _dates(n_dates)
    rows = []
    for d in dates:
        for i in range(items):
            actual_ret = float(i - items / 2)
            if mode == "supported":
                score = actual_ret
                q50_ret = float((i * 7) % 11 - 5)
            elif mode == "rejected":
                score = -actual_ret
                q50_ret = float((i * 7) % 11 - 5)
            else:
                score = float((i * 13) % 17 - 8)
                q50_ret = float((i * 13) % 17 - 8)
            rows.append(
                {
                    "item_id": i,
                    "forecast_date": d,
                    "base_price": 40.0,
                    "actual_price": 40.0 * (1 + actual_ret / 100),
                    "resolved_at": T,
                    "rank_score": score,
                    "q50_ret": q50_ret,
                    "actual_ret": actual_ret,
                    "version": "v1",
                    "fingerprint": "a" * 64,
                }
            )
    return {"rows": rows, "expected_batches": n_dates, "observed_batches": n_dates}


def _run_main(monkeypatch, tmp_path, frame):
    import scripts.ranking_transfer_report as rep

    monkeypatch.setattr(rep, "load_horizon_frame", lambda db, h: frame)
    monkeypatch.setattr(rep, "SessionLocal", lambda: object())
    json_out = str(tmp_path / "ranking.json")
    md_out = str(tmp_path / "ranking.md")
    code = rep.main(["--horizon", "7", "--json-out", json_out, "--markdown-out", md_out])
    import json

    with open(json_out) as f:
        payload = json.load(f)
    with open(md_out) as f:
        markdown = f.read()
    return code, payload, markdown


def test_supported_transfer(tmp_path, monkeypatch):
    code, payload, markdown = _run_main(monkeypatch, tmp_path, _frame(mode="supported"))
    assert code == 0
    assert payload["verdict"] == "SUPPORTED"
    assert "# Ranking transfer report" in markdown


def test_unresolved_transfer(tmp_path, monkeypatch):
    code, payload, _ = _run_main(monkeypatch, tmp_path, _frame(mode="tied"))
    assert code == 1
    assert payload["verdict"] == "UNRESOLVED"


def test_rejected_transfer(tmp_path, monkeypatch):
    code, payload, _ = _run_main(monkeypatch, tmp_path, _frame(mode="rejected"))
    assert code == 3
    assert payload["verdict"] == "REJECTED"


def test_thin_dates_dropped_and_counted(tmp_path, monkeypatch):
    frame = _frame(mode="supported", items=100)
    frame["rows"] = [r for r in frame["rows"] if not (r["forecast_date"] == BASE and r["item_id"] >= 50)]
    _code, payload, _ = _run_main(monkeypatch, tmp_path, frame)
    assert payload["dropped_dates"] >= 1
    assert payload["shared_dates"] == 19


def test_rank_score_stays_shadow_only():
    from pathlib import Path

    import database

    assert "rank_score" not in set(database.ItemForecast.__table__.columns.keys())
    schemas = (Path(__file__).resolve().parent.parent / "api" / "schemas.py").read_text()
    assert "rank_score" not in schemas


def test_supported_leaves_champion_config_alone(tmp_path, monkeypatch):
    from models.candidate_predictions import CENTRE_CHAMPIONS

    before = dict(CENTRE_CHAMPIONS)
    _run_main(monkeypatch, tmp_path, _frame(mode="supported"))
    assert dict(CENTRE_CHAMPIONS) == before
