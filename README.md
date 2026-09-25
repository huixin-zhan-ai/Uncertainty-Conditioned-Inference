# Uncertainty-conditioned path-energy inference over biomedical knowledge graphs

Code and frozen result tables for the paper *From Sequence Uncertainty to
Phenotype Hypotheses: Uncertainty-Conditioned Inference over Biomedical
Knowledge Graphs* (under review at ICLR 2027). This repository is anonymized
for review.

The method takes a genomic sequence model's calibrated prediction `p` and
uncertainty `H` for a variant, enumerates the admissible
gene → [gene] → disease → phenotype paths in a knowledge graph, scores them
with an interpretable learned energy `U`, and defines the target
`π_H(path) ∝ exp{U/θ(H)}` with `θ(H) = 0.35 + 0.90 H`. Phenotype hypotheses are
the aggregated path mass. Because the support is enumerable the target is
normalized exactly, and a Hastings-corrected revision operator is verified
against it.

## What is included

| Path | Contents |
|---|---|
| `code/corrected_core.py` | The method: calibration, path enumeration, 14-feature energy, exact target, Hastings-corrected sampler, diagnostics. |
| `code/run_corrected_study.py` | AlzKB study driver (locked scorer, all per-variant distributions, ablations, perturbation, figures, manifests). |
| `code/run_corrected_reclassification.py` | Five-fold variant-classification diagnostic with the locked-label firewall. |
| `code/run_synthetic_generality.py` | Synthetic sampler check on planted energies. |
| `code/run_operator_reasoning_experiments*.py`, `run_alzkb_uncertainty_redesign.py`, `learned_operator.py`, `learning_pipeline.py` | Graph loading, Evo 2 score handling and earlier utilities imported by `corrected_core`; kept for import compatibility. |
| `code/notebooks/` | Upstream notebook that produced the Evo 2 score table. |
| `experiments_corrected/00–07` | Thin wrappers that call the drivers above in paper order. |
| `experiments_corrected/08_baselines` … `11_bootstrap_cis` | Uniform/PageRank/beam baselines, mixing curve and 1/N chain test, synthetic energy recovery with restart / initialization diagnostics, gene-cluster bootstrap intervals. |
| `experiments_corrected/12_hetionet/` | Hetionet in the AlzKB schema (`hetionet_common.py`), breadth replication (`run_12a`), masked-edge recovery and DistMult reference (`run_12b`), worked query (`run_12c_example`), matched controls (`run_12d_controls`), gene hold-out across mask seeds (`run_12e_geneholdout`). |
| `outputs/corrected/tables/` | **Frozen result tables** (65 CSV/JSON) from which every number in the paper is generated. |
| `outputs/corrected/manifests/`, `models/`, `reports/` | Split IDs, outer-fold IDs, hashes, the frozen path scorers, and run reports. |
| `manuscript/verify_manuscript_numbers.py` | Recomputes 249 quantities from the frozen tables and asserts the exact strings appear in the paper source. |
| `manuscript/figures/make_*.py` | Regenerate Figures 1–8 and the appendix panels from the frozen tables (`make_framework_figure.py` draws the Fig. 1 schematic into the current directory). |
| `tests/test_corrected_core.py` | Path validity, feature dimension, detailed balance, empirical stationarity, full proposal support, empty/singleton supports, deterministic seeds, mass conservation. |
| `config/default.json` | Paper-level constants (the live schedules are `revision_temperature` and the step rule in `corrected_core.revision_steps`). |
| `CORRECTED_STUDY_SPEC.md`, `CORRECTION_LOG.md`, `REPRODUCIBILITY_GAPS.md` | Normative decisions, what was corrected relative to an earlier pipeline, and known gaps. |

Raw data (≈430 MB) are **not** included; see below.

## Quick start (no data needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m unittest tests/test_corrected_core.py          # method-level tests
python3 manuscript/figures/make_c3_figure.py . figures      # regenerate the recovery figure from frozen tables
python3 manuscript/figures/make_iclr_figures.py . figures
python3 manuscript/figures/make_hetionet_figure.py . figures
python3 manuscript/figures/make_recovery_example_figure.py . figures
```

To check the paper against the tables, place the paper's `main.tex` in a
directory `<paper>` and run

```bash
python3 manuscript/verify_manuscript_numbers.py . <paper>
```

It prints `249/249 checks passed` for the submitted version.

## Full reproduction (data required)

1. **AlzKB v2.0.0** — download the Memgraph CYPHERL dump from the AlzKB
   release page and place it at `data/raw/alzkb/alzkb-v2-0-0_memgraph.cypherl`.
2. **Hetionet v1.0** — download `hetionet-v1.0.json.bz2` from the Hetionet
   repository into `data/raw/hetionet/`.
3. **ClinVar / Evo 2 scores** — `data/raw/evo2/ad_all_scores_v3_vcf.csv` is the
   variant table with Evo 2 wild-type/mutant likelihoods produced by
   `code/notebooks/train_cm_evo2_head_v3_vcf.ipynb`; the ClinVar GRCh38
   source tables go in `data/raw/clinvar/`. SHA-256 checksums of every raw
   file are in `data/manifest.json`.

Then run the experiments in order (each writes to `outputs/corrected/`):

```bash
for d in experiments_corrected/0[0-7]_*; do python3 $d/run.py; done
for d in 08_baselines 09_mixing_curve 10_synthetic_recovery 10b_restart_diagnostic \
         10c_init_scale 10d_optimizer_fix 11_bootstrap_cis; do
  python3 experiments_corrected/$d/run.py; done
cd experiments_corrected/12_hetionet
python3 run_12a.py && python3 run_12b.py && python3 run_12c_example.py \
  && python3 run_12d_controls.py && python3 run_12e_geneholdout.py
```

Seeds are fixed (`random_seed: 42`; mask seeds 42, 7, 123 in `run_12e`), and
`outputs/corrected/manifests/run_manifest.json` records package versions and
artifact hashes of the frozen run.

## Data firewall

Only the 121 definitive Pathogenic/Benign ClinVar variants enter sequence
calibration, energy learning, or the classifier. Likely-label and VUS
records are held in `outputs/corrected/manifests/locked_split.json` and are
never fitted; `corrected_core.validate_split_disjointness` and the outer-fold
manifest enforce this.

## License

MIT (see `LICENSE`). AlzKB, Hetionet and ClinVar are subject to their own
licenses.
