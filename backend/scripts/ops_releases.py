#!/usr/bin/env python3
"""
Closed ops months as GitHub Releases on cs2-oracle-data: the second and third copies.

docs/specs/2026-10-11-postgres-to-parquet-history-design.md, "Second and third copies".
The data repo is one force-pushed orphan commit, so a bad write replaces the only copy
(archive day 09-19 was lost that way). Release assets live outside the git history.

``publish`` (last step of Backtest Accuracy) looks at every month under
``ops/<table>/YYYY-MM.parquet``. A month is CLOSED once its calendar end + 43 days has
passed (h=30 plus the 13-day embargo carry), so nothing can still mature into it:
September 2026 closes 11-12. For each closed month:

* no release yet: create tag ``ops-YYYY-MM`` with that month's file from each table,
  named ``<table>-YYYY-MM.parquet``, plus ``SHA256SUMS``;
* release exists: compare the tree's files with its ``SHA256SUMS``. Any difference,
  including a file added or removed, FAILS the run: a closed month was rewritten. An
  asset is never replaced.

So the release is both the backup and the immutability check. Calendar end rather than
the last forecast date, so a row added late cannot reopen a month.

``pull`` (the Mac, monthly or by hand) downloads every ``ops-*`` release it lacks into
``--dest/<tag>/`` and checks ``SHA256SUMS`` on everything it holds. It is the one copy
GitHub cannot delete.

Restore one month: ``gh release download ops-YYYY-MM -R RayanR000/cs2-oracle-data``, then
move each ``<table>-YYYY-MM.parquet`` back to ``ops/<table>/YYYY-MM.parquet``.

Usage:
    venv/bin/python scripts/ops_releases.py publish --archive-dir ../archive/price-archive [--dry-run]
    venv/bin/python scripts/ops_releases.py pull --dest ~/personal-projects/cs2-analyzer/ops-releases
"""

import argparse
import calendar
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger("ops_releases")

REPO = "RayanR000/cs2-oracle-data"
TABLES = ("item_forecasts", "forecast_outcomes", "forecast_candidates", "forecast_candidate_outcomes")
#: Longest horizon (30) plus the embargo carry (13): the last day anything matures.
CLOSE_AFTER_DAYS = 43
TAG_PREFIX = "ops-"
SUMS = "SHA256SUMS"
_MONTH = re.compile(r"^(\d{4})-(\d{2})\.parquet$")


def closes_on(month: str) -> date:
    """The first day month *month* (``YYYY-MM``) counts as closed."""
    y, m = map(int, month.split("-"))
    return date(y, m, calendar.monthrange(y, m)[1]) + timedelta(days=CLOSE_AFTER_DAYS)


def month_files(ops_dir: Path) -> dict[str, dict[str, Path]]:
    """{month: {asset name: path}} over every partitioned table present."""
    out: dict[str, dict[str, Path]] = {}
    for table in TABLES:
        d = ops_dir / table
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if m := _MONTH.match(p.name):
                month = f"{m[1]}-{m[2]}"
                out.setdefault(month, {})[f"{table}-{month}.parquet"] = p
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def format_sums(sums: dict[str, str]) -> str:
    """``sha256sum``-compatible, sorted, so ``sha256sum -c SHA256SUMS`` verifies a download."""
    return "".join(f"{sums[n]}  {n}\n" for n in sorted(sums))


def parse_sums(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.lstrip("*")] = digest
    return out


def diff_sums(released: dict[str, str], tree: dict[str, str]) -> list[str]:
    problems = [f"{n}: in the release, missing from the tree" for n in sorted(released.keys() - tree.keys())]
    problems += [f"{n}: in the tree, not in the release" for n in sorted(tree.keys() - released.keys())]
    problems += [f"{n}: sha256 changed" for n in sorted(released.keys() & tree.keys()) if released[n] != tree[n]]
    return problems


# --- gh ----------------------------------------------------------------------


def _gh(*args: str, capture: bool = True) -> str:
    r = subprocess.run(["gh", *args], check=True, capture_output=capture, text=True)
    return r.stdout if capture else ""


def release_tags() -> set[str]:
    rows = json.loads(_gh("release", "list", "-R", REPO, "--limit", "1000", "--json", "tagName"))
    return {r["tagName"] for r in rows if r["tagName"].startswith(TAG_PREFIX)}


def released_sums(tag: str) -> dict[str, str]:
    return parse_sums(_gh("release", "download", tag, "-R", REPO, "-p", SUMS, "-O", "-"))


def create_release(tag: str, files: dict[str, Path], sums_text: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        staged = []
        for name, path in sorted(files.items()):
            shutil.copyfile(path, Path(tmp) / name)
            staged.append(str(Path(tmp) / name))
        (Path(tmp) / SUMS).write_text(sums_text)
        staged.append(str(Path(tmp) / SUMS))
        notes = (
            f"Closed ops month {tag.removeprefix(TAG_PREFIX)}: written once, never rewritten. "
            "Each <table>-YYYY-MM.parquet restores to price-archive/ops/<table>/YYYY-MM.parquet. "
            f"Verify with: sha256sum -c {SUMS}"
        )
        _gh("release", "create", tag, "-R", REPO, "--target", "main", "--title", tag, "--notes", notes, *staged)


# --- commands ----------------------------------------------------------------


def publish(ops_dir: Path, today: date, dry_run: bool) -> int:
    months = month_files(ops_dir)
    closed = sorted(m for m in months if closes_on(m) <= today)
    tags = release_tags() if closed else set()
    failed = False
    for month in closed:
        tag = f"{TAG_PREFIX}{month}"
        tree = {n: sha256(p) for n, p in months[month].items()}
        if tag in tags:
            problems = diff_sums(released_sums(tag), tree)
            if problems:
                failed = True
                for p in problems:
                    logger.error("%s: closed month rewritten: %s", tag, p)
                print(f"::error::{tag}: a closed ops month changed after release ({len(problems)} file(s))")
            else:
                logger.info("%s: matches its release (%d files)", tag, len(tree))
        elif dry_run:
            logger.info("%s: would release %s", tag, ", ".join(sorted(tree)))
        else:
            create_release(tag, months[month], format_sums(tree))
            logger.info("%s: released %d files", tag, len(tree))
    for month in sorted(set(months) - set(closed)):
        logger.info("ops-%s: open until %s", month, closes_on(month))
    return 1 if failed else 0


def pull(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    failed = False
    for tag in sorted(release_tags()):
        target = dest / tag
        if not target.exists():
            tmp = dest / f".{tag}.partial"
            shutil.rmtree(tmp, ignore_errors=True)
            _gh("release", "download", tag, "-R", REPO, "-D", str(tmp), capture=False)
            tmp.rename(target)
            logger.info("%s: downloaded", tag)
        released = parse_sums((target / SUMS).read_text())
        held = {p.name: sha256(p) for p in target.iterdir() if p.name != SUMS}
        problems = diff_sums(released, held)
        if problems:
            failed = True
            for p in problems:
                logger.error("%s: %s", tag, p)
        else:
            logger.info("%s: %d files verified", tag, len(held))
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish", help="Release closed months; fail if a released month changed.")
    p.add_argument("--archive-dir", type=Path, required=True, help="The data checkout's price-archive/ directory.")
    p.add_argument("--today", type=date.fromisoformat, default=date.today())
    p.add_argument("--dry-run", action="store_true", help="Verify existing releases, create none.")
    q = sub.add_parser("pull", help="Download every ops release not held locally and verify all.")
    q.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.cmd == "publish":
        return publish(args.archive_dir / "ops", args.today, args.dry_run)
    return pull(args.dest.expanduser())


if __name__ == "__main__":
    sys.exit(main())
