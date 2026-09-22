"""Importing a script module must not mutate os.environ.

2026-09: `scripts/confirm_mondrian_oof.py` set EXCEEDANCE_HEAD=1 (plus
FEATURE_NATIVE_NAN=1) at import time, and three `anomaly_*_ab` scripts set
ANOMALY_GBM=1 the same way. Test modules import these scripts for their pure
helpers, so collection alone flipped feature-engineering and CV gates for
every test file after them alphabetically — 25 failures (KeyError:
target_exceed_3d) that passed in isolation and vanished when run per-file.

The fix moved every such set into main(). This test pins it: each script is
imported in a FRESH interpreter and the environment must come back identical.
Subprocess, not in-process — by the time this file runs, other test modules
have already imported the scripts in this process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Modules that once mutated os.environ at import, plus their neighbours that
# share the pattern. Importing the module must be enough — no function calls.
SCRIPT_MODULES = [
    "scripts.archive.confirm_mondrian_oof",
    "scripts.archive.anomaly_gbm_ab",
    "scripts.archive.anomaly_calibration_ab",
    "scripts.archive.anomaly_band_modulator_ab",
]

PROBE = (
    "import json, os, sys; "
    # Warmup first: importing numpy/LightGBM itself adds interpreter-startup
    # vars (KMP_DUPLICATE_LIB_OK, ...). Snapshot AFTER that so the diff is the
    # script's own effect, not its dependency tree's.
    "import models.forecaster; "
    "before = dict(os.environ); "
    "mod = sys.argv[1]; "
    "__import__(mod); "
    "after = dict(os.environ); "
    "added = {k: after[k] for k in after if k not in before}; "
    "removed = [k for k in before if k not in after]; "
    "changed = {k: (before[k], after[k]) for k in before if k in after and before[k] != after[k]}; "
    "print(json.dumps({'added': added, 'removed': removed, 'changed': changed}))"
)


@pytest.mark.parametrize("module", SCRIPT_MODULES)
def test_script_import_leaves_the_environment_unchanged(module):
    out = subprocess.run(
        [sys.executable, "-c", PROBE, module],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert out.returncode == 0, f"importing {module} failed:\n{out.stderr[-2000:]}"
    diff = json.loads(out.stdout.strip().splitlines()[-1])
    assert diff == {"added": {}, "removed": [], "changed": {}}, (
        f"importing {module} mutated os.environ (this once flipped CV gates for 25 downstream tests): {diff}"
    )
