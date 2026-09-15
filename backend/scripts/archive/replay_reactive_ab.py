"""Paired calm-anchor A/B for the regime-reactive climatology band scale.

Replays the CONTROL (static climatology) and the ARM (reactive) artifacts at the
same anchors and prints, per horizon, coverage and served half-width for each so
the two can be differenced on WIDTH and COVERAGE (never on q_hat — matched-pair
rule). The replay reads models/saved_models, so this swaps that path to each
artifact dir in turn. Run from the repo root (dev env).

    backend/venv/bin/python backend/scripts/replay_reactive_ab.py
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SB = ROOT / "backend" / "models" / "saved_models"
CTRL = ROOT / "backend" / "models" / "saved_models_ctrl"
ARM = ROOT / "backend" / "models" / "saved_models_arm"

CALM = ["2026-04-07", "2026-06-16", "2026-07-01"]  # 04-13 crosses a cutover
VOLATILE = ["2026-03-24", "2026-03-27"]

# BAND COVERAGE columns: h  n  cov%  target%  miss<low  miss>high  halfw%
ROW = re.compile(
    r"^\s*(\d+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def _point(dir_path: Path):
    if SB.is_symlink() or SB.exists():
        if SB.is_symlink():
            SB.unlink()
        else:
            # A real directory here is the seed; move it aside once.
            bak = SB.with_name("saved_models_seed_bak")
            if not bak.exists():
                SB.rename(bak)
            else:
                import shutil

                shutil.rmtree(SB)
    SB.symlink_to(dir_path)


def _replay(anchor: str, reactive: bool) -> dict:
    env = dict(os.environ)
    env["REPLAY_ANCHOR"] = anchor
    env["CLIMATOLOGY_SCALE"] = "1"
    env["CLIMATOLOGY_REACTIVE"] = "1" if reactive else "0"
    env["PYTHONPATH"] = str(ROOT / "backend")
    p = subprocess.run(
        [str(ROOT / "backend" / "venv" / "bin" / "python"), "-m", "scripts.replay_serving"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    out = p.stdout + p.stderr
    raw = ROOT / f"ab_raw_{anchor}_{'arm' if reactive else 'ctrl'}.log"
    raw.write_text(out)
    rows, in_band = {}, False
    for line in out.splitlines():
        if "BAND COVERAGE @" in line and "BY SIGMA" not in line:
            in_band = True
            continue
        if in_band:
            m = ROW.match(line)
            if m:
                h = int(m.group(1))
                rows[h] = {
                    "n": int(m.group(2).replace(",", "")),
                    "cov": float(m.group(3)),
                    "target": float(m.group(4)),
                    "halfw": float(m.group(7)),
                }
            elif line.strip() and not line.startswith(" ") and rows:
                in_band = False
    if not rows:
        (ROOT / "scratchpad_replay_fail.log").write_text(out[-4000:])
        print(f"    !! no BAND COVERAGE parsed for {anchor} reactive={reactive} (rc={p.returncode}); tail saved")
    return rows


def main() -> int:
    results = []
    for label, anchors in (("CALM", CALM), ("VOLATILE", VOLATILE)):
        for a in anchors:
            print(f"\n=== {label} anchor {a} ===")
            _point(CTRL)
            ctrl = _replay(a, reactive=False)
            _point(ARM)
            arm = _replay(a, reactive=True)
            for h in sorted(set(ctrl) & set(arm)):
                c, r = ctrl[h], arm[h]
                ratio = r["halfw"] / c["halfw"] if c["halfw"] else float("nan")
                print(
                    f"  h={h:>2}  cov ctrl {c['cov']:.1f}% -> arm {r['cov']:.1f}% "
                    f"| halfw ctrl {c['halfw']:.3f} -> arm {r['halfw']:.3f} "
                    f"(x{ratio:.3f})  n={c['n']}"
                )
                results.append((label, a, h, c["cov"], r["cov"], c["halfw"], r["halfw"], ratio))
    print("\n================ SUMMARY (arm/ctrl width ratio) ================")
    for label in ("CALM", "VOLATILE"):
        for h in (3, 7, 14, 30):
            rr = [x[7] for x in results if x[0] == label and x[2] == h]
            cc = [x[4] - x[3] for x in results if x[0] == label and x[2] == h]
            if rr:
                import numpy as np

                print(
                    f"  {label:>8} h={h:>2}: width x{np.mean(rr):.3f} "
                    f"(n_anchors={len(rr)}), mean cov Δ {np.mean(cc):+.1f}pp"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
