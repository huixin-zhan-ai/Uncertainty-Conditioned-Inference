#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
subprocess.run(
    [sys.executable, str(ROOT / "code" / "run_corrected_reclassification.py")],
    cwd=ROOT,
    check=True,
)
subprocess.run(
    [sys.executable, str(ROOT / "code" / "sync_corrected_experiments.py")],
    cwd=ROOT,
    check=True,
)
