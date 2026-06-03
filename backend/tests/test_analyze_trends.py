from __future__ import annotations

import ast
from pathlib import Path


def _load_pure_analyze_trends_symbols():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_trends.py"
    source = module_path.read_text()
    tree = ast.parse(source, filename=str(module_path))

    wanted_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(getattr(target, "id", None) == "DAILY_ANALYSIS_WRITE_COLUMNS" for target in node.targets):
                wanted_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_filter_daily_analysis_row":
            wanted_nodes.append(node)

    subset = ast.Module(body=wanted_nodes, type_ignores=[])
    ast.fix_missing_locations(subset)
    namespace: dict[str, object] = {}
    exec(compile(subset, str(module_path), "exec"), namespace)
    return namespace


def test_daily_analysis_filter_drops_updated_at():
    namespace = _load_pure_analyze_trends_symbols()
    row = {
        "item_id": 123,
        "analysis_date": "2026-06-03",
        "current_price": 10.0,
        "created_at": "2026-06-03T00:00:00",
        "updated_at": "2026-06-03T00:00:00",
        "unexpected": "drop-me",
    }

    filtered = namespace["_filter_daily_analysis_row"](row)

    assert "updated_at" not in filtered
    assert "unexpected" not in filtered
    assert filtered["item_id"] == 123
    assert filtered["created_at"] == "2026-06-03T00:00:00"


def test_daily_analysis_upsert_uses_reflected_table():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_trends.py"
    source = module_path.read_text()

    assert "sa.Table(table.name, sa.MetaData(), autoload_with=bind)" in source
    assert "insert_stmt(target_table).values(filtered_rows)" in source
    assert "updated_at" not in source.split("DAILY_ANALYSIS_WRITE_COLUMNS = (", 1)[1].split(")", 1)[0]


def test_min_required_history_points_remains_accuracy_threshold():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_trends.py"
    source = module_path.read_text()

    assert "MIN_REQUIRED_HISTORY_POINTS = 7" in source
