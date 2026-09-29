"""Every data-repo publish must be a compare-and-swap, never a blind force-push.

Each publisher checks out cs2-oracle-data `main`, rebuilds it as one orphan commit, and
force-pushes. A blind `--force` from a checkout that went stale deletes whatever landed in
between: on 2026-09-20 an item_forecasts publish overwrote the Aggregator's 2026-09-19 prices,
supply, volume and exchange rates 21 seconds after they were pushed
(docs/changelog/2026-09-29-archive-day-0919-restored.md). A lease on the checked-out SHA
turns that silent loss into a failed job.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
ORPHAN_PUBLISHERS = [p for p in sorted(WORKFLOWS.glob("*.yml")) if "git checkout --orphan" in p.read_text()]


def test_there_are_publishers_to_check():
    names = {p.name for p in ORPHAN_PUBLISHERS}
    assert {"aggregator-update.yml", "price-forecast.yml", "backtest-accuracy.yml"} <= names


def test_no_publisher_force_pushes_blind():
    for wf in ORPHAN_PUBLISHERS:
        text = wf.read_text()
        assert not re.search(r"git push --force(?!-with-lease)", text), f"{wf.name}: blind force-push"
        assert 'git push --force-with-lease=main:"$BASE" origin HEAD:main' in text, f"{wf.name}: no lease"


def test_the_lease_base_is_captured_before_the_orphan_checkout():
    for wf in ORPHAN_PUBLISHERS:
        text = wf.read_text()
        for orphan in re.finditer(r"git checkout --orphan", text):
            before = text[: orphan.start()]
            last_base = before.rfind("BASE=$(git rev-parse HEAD)")
            last_step = before.rfind("- name:")
            assert last_base > last_step, f"{wf.name}: BASE must be set in the publish step, before --orphan"
