#!/usr/bin/env python3
"""Materialize self-documented corrected experiment folders from frozen outputs."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "corrected"
DEST = ROOT / "experiments_corrected"

EXPERIMENTS = {
    "00_locked_inputs": {
        "description": "Definitive-only signal calibration and frozen split IDs.",
        "runner": "run_corrected_study.py",
        "files": [
            "tables/variants_locked_signals.csv",
            "manifests/locked_split.json",
            "manifests/run_manifest.json",
            "manifests/artifact_manifest.json",
        ],
    },
    "01_graph_routes_and_learning": {
        "description": "Valid route audit, 14-feature model, and learned weights.",
        "runner": "run_corrected_study.py",
        "files": [
            "tables/path_audit.csv",
            "models/path_scorer.json",
            "figures/figure1_framework_corrected.pdf",
        ],
    },
    "02_diffusion_and_propagation": {
        "description": "Hastings-corrected trajectory diagnostics and uncertainty propagation.",
        "runner": "run_corrected_study.py",
        "files": [
            "tables/diffusion_diagnostics.csv",
            "tables/propagation.csv",
            "tables/propagation_summary.csv",
            "figures/figure3_diffusion_and_propagation_corrected.pdf",
        ],
    },
    "03_ranking_sensitivity": {
        "description": "Endpoint-ranking response to H plus or minus 0.30.",
        "runner": "run_corrected_study.py",
        "files": [
            "tables/ranking.csv",
            "tables/ranking_summary.csv",
            "figures/figure4_ranking_sensitivity_corrected.pdf",
        ],
    },
    "04_alternative_uncertainty": {
        "description": "Isotonic-versus-kNN uncertainty sensitivity.",
        "runner": "run_corrected_study.py",
        "files": [
            "tables/alternative_uncertainty.csv",
            "tables/alternative_class_summary.csv",
            "tables/alternative_gene_summary.csv",
            "figures/figure5_alternative_uncertainty_corrected.pdf",
        ],
    },
    "05_baselines_and_branching": {
        "description": "Uniform and fixed-temperature baselines plus feature ablations.",
        "runner": "run_corrected_study.py",
        "files": [
            "tables/ablations.csv",
            "tables/branching.csv",
            "figures/figure6_graph_ablations_corrected.pdf",
        ],
    },
    "06_locked_reclassification": {
        "description": "Nested definitive training and locked likely-label validation.",
        "runner": "run_corrected_reclassification.py",
        "files": [
            "tables/locked_validation_predictions.csv",
            "tables/reclassification_features.csv",
            "tables/selective_classification.csv",
            "tables/vus_calls_sequence_only.csv",
            "tables/vus_calls_sequence_plus_graph.csv",
            "reports/reclassification_results.json",
            "manifests/reclassification_outer_folds.json",
            "figures/figure7_reclassification_corrected.pdf",
        ],
    },
    "07_synthetic_generality": {
        "description": "Synthetic sampler scaling across 53 to 1,024 states.",
        "runner": "run_synthetic_generality.py",
        "files": [
            "tables/synthetic_generality.csv",
            "tables/synthetic_generality_summary.csv",
            "reports/synthetic_generality.json",
            "figures/figure8_synthetic_generality.pdf",
        ],
    },
}


def runner_text(script):
    return f"""#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
subprocess.run(
    [sys.executable, str(ROOT / "code" / "{script}")],
    cwd=ROOT,
    check=True,
)
subprocess.run(
    [sys.executable, str(ROOT / "code" / "sync_corrected_experiments.py")],
    cwd=ROOT,
    check=True,
)
"""


def main():
    for name, specification in EXPERIMENTS.items():
        directory = DEST / name
        output = directory / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        (directory / "README.md").write_text(
            f"""# {name.replace('_', ' ').title()}

{specification['description']}

Run `python run.py` from this directory. The runner refits the complete shared
pipeline needed for this experiment and then refreshes this frozen output copy.
All fitted upstream components obey `../../CORRECTED_STUDY_SPEC.md`.
""",
            encoding="utf-8",
        )
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "seed": 42,
                    "runner": specification["runner"],
                    "normative_specification": "../../CORRECTED_STUDY_SPEC.md",
                    "source_files": specification["files"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (directory / "run.py").write_text(
            runner_text(specification["runner"]), encoding="utf-8"
        )
        for relative in specification["files"]:
            source = SOURCE / relative
            if not source.exists():
                raise FileNotFoundError(source)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    (DEST / "README.md").write_text(
        """# Corrected experiments

These folders correspond to the frozen corrected manuscript. The older
`experiments/` directory is retained only for provenance and must not be used to
regenerate manuscript claims. Shared code lives in `../code`; each folder has a
runner, configuration, documentation, and a frozen copy of its outputs.
""",
        encoding="utf-8",
    )
    print(DEST)


if __name__ == "__main__":
    main()
