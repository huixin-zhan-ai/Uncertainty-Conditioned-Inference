#!/usr/bin/env python3
"""Freeze corrected artifacts and fail on cross-layer inconsistencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "outputs" / "corrected"
MANUSCRIPT = PROJECT / "npj_manuscript" / "manuscript.tex"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition, message, checks):
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main():
    checks = []
    real = json.loads((OUT / "reports/corrected_results.json").read_text())
    reclass = json.loads((OUT / "reports/reclassification_results.json").read_text())
    synthetic = json.loads((OUT / "reports/synthetic_generality.json").read_text())
    split = json.loads((OUT / "manifests/locked_split.json").read_text())
    folds = json.loads((OUT / "manifests/reclassification_outer_folds.json").read_text())
    model = json.loads((OUT / "models/path_scorer.json").read_text())
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")

    definitive = set(split["definitive_ids"])
    likely = set(split["likely_validation_ids"])
    vus = set(split["vus_ids"])
    check(not definitive & likely, "Definitive and likely IDs are disjoint", checks)
    check(not definitive & vus, "Definitive and VUS IDs are disjoint", checks)
    check(not likely & vus, "Likely and VUS IDs are disjoint", checks)
    check(
        set(model["training_ids"]).issubset(definitive),
        "Path scorer uses definitive IDs only",
        checks,
    )
    for fold in folds:
        train = set(map(str, fold["train_ids"]))
        validation = set(map(str, fold["validation_ids"]))
        check(not train & validation, f"Fold {fold['fold']} is disjoint", checks)
        check(
            train | validation == definitive,
            f"Fold {fold['fold']} covers definitive set",
            checks,
        )

    check(real["n_variants_loaded"] == 985, "Loaded cohort count is 985", checks)
    check(real["n_variants_analyzed"] == 982, "Analyzed cohort count is 982", checks)
    check(
        reclass["n"]["definitive_training"] == len(definitive) == 121,
        "Definitive training count is consistent",
        checks,
    )
    check(
        len(likely) == 399 and reclass["n"]["locked_validation"] == 398,
        "Locked validation has 399 IDs and 398 valid graph routes",
        checks,
    )
    check(
        len(vus) == 465 and reclass["n"]["vus"] == 463,
        "VUS input has 465 IDs and 463 valid graph routes",
        checks,
    )
    check(
        real["mean_sampler_js_to_exact"] < 0.01,
        "Real-data sampler mean JS is below 0.01",
        checks,
    )
    check(
        synthetic["maximum_p95_sampler_js"] < 0.01,
        "Synthetic sampler p95 JS is below 0.01",
        checks,
    )

    banned = [
        r"0\.25\s*\+\s*1\.4",
        r"10\s*\+\s*\\lfloor\s*15",
        r"Pathogenic/Likely",
        r"complete candidate set.*every valid",
        r"gene--process/pathway--disease--phenotype graph defines",
    ]
    for pattern in banned:
        check(
            re.search(pattern, manuscript, flags=re.IGNORECASE | re.DOTALL) is None,
            f"Removed legacy manuscript pattern: {pattern}",
            checks,
        )

    required_strings = [
        f"mean Jensen--Shannon divergence {real['mean_sampler_js_to_exact']:.4f}",
        f"AUROC point estimate increased from {reclass['models']['Sequence only']['auroc']:.3f} to\n{reclass['models']['Sequence + graph']['auroc']:.3f}",
        f"worsened from {reclass['models']['Sequence only']['brier']:.3f} to {reclass['models']['Sequence + graph']['brier']:.3f}",
        f"largest condition-level\nmean JS error was {synthetic['maximum_mean_sampler_js']:.4f}",
    ]
    for text in required_strings:
        check(text in manuscript, f"Manuscript contains frozen value: {text}", checks)

    included_figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", manuscript)
    for relative in included_figures:
        path = PROJECT / "npj_manuscript" / relative
        check(path.exists(), f"Figure exists: {relative}", checks)

    for csv_path in (OUT / "tables").glob("*.csv"):
        frame = pd.read_csv(csv_path)
        numeric = frame.select_dtypes(include="number")
        check(
            not np.isinf(numeric.to_numpy()).any(),
            f"No infinite numeric values: {csv_path.name}",
            checks,
        )

    artifacts = {}
    for base in (OUT / "tables", OUT / "reports", OUT / "figures", OUT / "models", OUT / "manifests"):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.name != "artifact_manifest.json":
                artifacts[str(path.relative_to(OUT))] = sha256(path)
    artifacts["manuscript/manuscript.tex"] = sha256(MANUSCRIPT)
    artifacts["manuscript/manuscript.pdf"] = sha256(
        PROJECT / "npj_manuscript" / "manuscript.pdf"
    )
    payload = {
        "run_id": "corrected_frozen_2026-09-04",
        "status": "PASS",
        "checks": checks,
        "artifacts_sha256": artifacts,
    }
    (OUT / "manifests/artifact_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"PASS: {len(checks)} consistency checks")


if __name__ == "__main__":
    main()
