# Corrected frozen real-data run

## Locked design
- Sequence calibration and path-scorer training use only Pathogenic versus Benign variants.
- Likely pathogenic and likely benign variants are untouched locked validation data.
- Candidate paths are contiguous directed graph paths with zero or one gene-gene hop, followed by disease and phenotype.
- BiologicalProcess and Pathway nodes are used as gene annotations, not falsely described as traversed path nodes.
- One temperature schedule, theta(H) = 0.35 + 0.9 H, defines both the exact learned target and the Metropolis acceptance ratio.
- Proposals mix 90% learned mass at twice the target temperature with 10%
  uniform mass; the full Hastings ratio corrects proposal asymmetry.

## Cohort and path audit
- Loaded: 985 variants.
- Analyzed: 982; excluded for no valid phenotype path: 3.
- Path scorer training examples: 121, all definitive labels.
- Candidate route range: 0–20; candidate path range: 0–1060.

## Learned diffusion validation
- Mean acceptance rate: 0.507.
- Mean JS divergence between 512-chain final-state distribution and exact target: 0.0031.
- Median JS divergence: 0.0029.
- Spearman H versus unique final routes: 0.403.

## Experiment 1: uncertainty propagation
uncertainty_stratum  phenotype_entropy_mean  effective_groups_mean  top1_mass_mean   n
             Q1 low                0.585336               1.536816        0.863921 349
                 Q2                1.813622               3.705145        0.585838 236
                 Q3                2.370408               5.256658        0.434814 170
            Q4 high                2.630858               6.254840        0.356596 227

Spearman H versus grouped phenotype entropy: 0.951.

## Experiment 2: uncertainty perturbation
uncertainty_stratum  mean_js  mean_top1_switch  mean_top5_overlap   n
             Q1 low 0.035332          0.004298           0.946036 349
                 Q2 0.031708          0.059322           0.966808 236
                 Q3 0.013355          0.138235           0.993137 170
            Q4 high 0.007399          0.149780           1.000000 227

## Experiment 3: alternative uncertainty
 clinical_significance  mean_js  mean_top1_switch  mean_top5_overlap   n
                Benign 0.098890          0.073171           0.967480  41
         Likely_benign 0.020123          0.033537           0.983740 328
     Likely_pathogenic 0.008994          0.071429           0.952381  70
            Pathogenic 0.057283          0.075000           0.900000  80
Uncertain_significance 0.015006          0.047516           0.971202 463

The most sensitive gene represented by at least five variants is APP
(mean JS 0.028, n=372). Singleton genes are not ranked
as stable gene-level findings.

## Experiment 4: branching and ablation
Spearman H versus raw-endpoint entropy: 0.890.
See `ablations.csv` for uniform, fixed-temperature, no-confidence, no-H, and
no-mechanism comparisons.

## Interpretation boundary
Uncertainty broadening is partly structural by design because H controls the
target temperature. The ablations quantify this designed effect; they are not
presented as independent biological validation. AlzKB contains phenotype links
for a shared disease layer, so gene specificity arises through valid gene-gene
routes and route-level process/pathway annotations, not distinct endpoint sets.
