"""Read two `model-diagnostics.yml` runs as one paired table.

    python -m scripts.compare_diagnostics --control 31547391395 --arm 31547400215

Downloads both runs' `diagnostics-{h}d` artifacts with `gh` and prints the two
numbers an arm is decided on:

  * CV, clean-anchor cohort — `mean_rank_ic_tied` against its own naive bar. The
    pooled `mean_rank_ic` is NOT printed as a verdict: its label divides by the
    raw anchor quote its own features are built from, worth 0.03-0.24 rank IC,
    which is larger than any arm effect measured in this project
    (`2026-08-11-the-gap-is-the-anchor-denominator.md`).
  * Serving replay, tied cohort — the same cohort one layer down, per anchor.

Reads nothing but the two runs' artifacts, writes nothing anywhere.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HORIZONS = (3, 7, 14, 30)

_ANCHOR_RE = re.compile(r"LABEL BASIS SWEEP @ (\d{4}-\d{2}-\d{2})")
# `replay_serving` prints the cohort split into the same table as the five label
# bases; only the label tells them apart. The `(cv ...)` suffix is what a cohort
# row has and a basis row does not, so it anchors the match.
_COHORT_RE = re.compile(
    r"^\s*(\d+)\s+(tied|deviating)\s+(\d+)\s+"
    r"(-?\d+\.\d+)\s+([+-]\d+\.\d+)\s+\(cv\s+([+-]\d+\.\d+)\)\s*$"
)


def parse_replay_log(text: str) -> list[dict]:
    """The tied/deviating rows of every basis sweep in one replay log.

    A `too few` row carries no IC and is dropped rather than defaulted: three
    finite rows is the floor `replay_serving` applies, and a zero there would
    read as a measured null.
    """
    rows: list[dict] = []
    anchor = None
    for line in text.splitlines():
        found = _ANCHOR_RE.search(line)
        if found:
            anchor = found.group(1)
            continue
        m = _COHORT_RE.match(line)
        if m and anchor:
            rows.append(
                {
                    "anchor": anchor,
                    "horizon": int(m.group(1)),
                    "cohort": m.group(2),
                    "n": int(m.group(3)),
                    "served_ic": float(m.group(4)),
                    "cv_ic": float(m.group(6)),
                }
            )
    return rows


def cv_tied_row(meta: dict, horizon: int) -> dict | None:
    """The clean-anchor CV block for one horizon, or None if absent.

    None and never a fallback to the pooled keys: every artifact before
    2026-08-11 lacks the column, and publishing the contaminated number under
    the clean name is worse than publishing no number. Same rule
    `classifier_accuracy_ge1` follows.
    """
    cv = (meta.get("cv_results") or {}).get(str(horizon))
    if not cv or cv.get("mean_rank_ic_tied") is None:
        return None
    return {
        "horizon": horizon,
        "rank_ic_tied": cv["mean_rank_ic_tied"],
        "naive_rank_ic_tied": cv.get("mean_naive_rank_ic_tied"),
        "edge_vs_naive_tied": cv.get("rank_ic_edge_vs_naive_tied"),
        "tied_rows": cv.get("tied_rows"),
        "tied_dates": cv.get("tied_dates"),
    }


def cv_table_note(c_meta: dict, a_meta: dict, horizons=HORIZONS) -> str | None:
    """Why the CV table is empty, when it is — or None when it has rows.

    A table with no rows and a parse that silently failed look the same on a
    terminal, and the all-missing case is the expected state for any pair of
    artifacts trained before 2026-08-11.
    """
    if any(cv_tied_row(c_meta.get(h, {}), h) or cv_tied_row(a_meta.get(h, {}), h) for h in horizons):
        return None
    return (
        "  (no rows: neither artifact carries `mean_rank_ic_tied` — both "
        "predate the 2026-08-11 column. The pooled keys are NOT shown in "
        "its place.)"
    )


def pair_replay_rows(control: list[dict], arm: list[dict]) -> list[dict]:
    """Difference the arm against the control on (anchor, horizon, cohort).

    A cell present in one run and not the other is emitted with a null delta
    rather than dropped -- a silently unpaired read is the failure this harness
    exists to prevent -- and a cell whose `n` moved is flagged, because the tied
    mask is arm-invariant by construction and a differing count means the two
    runs scored different items.
    """

    def key(r):
        return (r["anchor"], r["horizon"], r["cohort"])

    c_by, a_by = {key(r): r for r in control}, {key(r): r for r in arm}
    out = []
    for k in sorted(c_by.keys() | a_by.keys()):
        c, a = c_by.get(k), a_by.get(k)
        anchor, horizon, cohort = k
        out.append(
            {
                "anchor": anchor,
                "horizon": horizon,
                "cohort": cohort,
                "n_control": c["n"] if c else None,
                "n_arm": a["n"] if a else None,
                "served_ic_control": c["served_ic"] if c else None,
                "served_ic_arm": a["served_ic"] if a else None,
                "delta_served_ic": (a["served_ic"] - c["served_ic"] if c and a else None),
                "cohort_moved": bool(c and a and c["n"] != a["n"]),
            }
        )
    return out


def _download(run_id: str, dest: Path) -> Path:
    out = dest / run_id
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gh", "run", "download", run_id, "--dir", str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def _load(run_dir: Path) -> tuple[dict, list[dict]]:
    """meta.json per horizon, and every replay row in the run."""
    metas, rows = {}, []
    for h in HORIZONS:
        d = run_dir / f"diagnostics-{h}d"
        meta = d / "models" / "saved_models" / "meta.json"
        if not meta.exists():  # the artifact flattens differently on some runs
            found = list(d.rglob("meta.json"))
            meta = found[0] if found else None
        if meta:
            metas[h] = json.loads(meta.read_text())
        log = d / f"replay-{h}d.log"
        if not log.exists():
            found = list(d.rglob(f"replay-{h}d.log"))
            log = found[0] if found else None
        if log:
            rows.extend(parse_replay_log(log.read_text()))
    return metas, rows


def _fmt(v, spec="+.4f"):
    return "     --" if v is None else format(v, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", required=True, help="control run id")
    ap.add_argument("--arm", required=True, help="arm run id")
    ap.add_argument("--dir", help="reuse an already-downloaded directory")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(args.dir) if args.dir else Path(tmp)
        c_dir = base / args.control if args.dir else _download(args.control, base)
        a_dir = base / args.arm if args.dir else _download(args.arm, base)
        c_meta, c_rows = _load(c_dir)
        a_meta, a_rows = _load(a_dir)

    print(f"\nCV — CLEAN-ANCHOR COHORT   control {args.control} vs arm {args.arm}")
    print(f"{'h':>4} {'ctl IC':>9} {'arm IC':>9} {'delta':>9} {'ctl edge':>9} {'arm edge':>9} {'rows':>9} {'dates':>6}")
    note = cv_table_note(c_meta, a_meta)
    if note:
        print(note)
    for h in HORIZONS:
        c = cv_tied_row(c_meta.get(h, {}), h)
        a = cv_tied_row(a_meta.get(h, {}), h)
        if not c and not a:
            continue
        delta = (a["rank_ic_tied"] - c["rank_ic_tied"]) if c and a else None
        print(
            f"{h:>4} {_fmt(c and c['rank_ic_tied']):>9} "
            f"{_fmt(a and a['rank_ic_tied']):>9} {_fmt(delta):>9} "
            f"{_fmt(c and c['edge_vs_naive_tied']):>9} "
            f"{_fmt(a and a['edge_vs_naive_tied']):>9} "
            f"{(c or a)['tied_rows']:>9,} {(c or a)['tied_dates']:>6}"
        )

    print("\nSERVING REPLAY — tied cohort (deviating shown for contrast)")
    print(f"{'anchor':>12} {'h':>4} {'cohort':>10} {'n':>7} {'ctl IC':>9} {'arm IC':>9} {'delta':>9}")
    for r in pair_replay_rows(c_rows, a_rows):
        flag = "  <- COHORT MOVED" if r["cohort_moved"] else ""
        if r["delta_served_ic"] is None:
            flag += "  <- UNPAIRED"
        n = r["n_control"] if r["n_control"] is not None else r["n_arm"]
        print(
            f"{r['anchor']:>12} {r['horizon']:>4} {r['cohort']:>10} "
            f"{n:>7,} {_fmt(r['served_ic_control']):>9} "
            f"{_fmt(r['served_ic_arm']):>9} "
            f"{_fmt(r['delta_served_ic']):>9}{flag}"
        )

    print(
        "\nThe bar is the arm's own naive column on the same folds, not a "
        "stored number and not 50%. Four anchors are replications, not power: "
        "read sign consistency, not magnitude."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
