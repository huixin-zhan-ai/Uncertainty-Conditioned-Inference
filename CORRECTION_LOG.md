# Correction log

## Resolved

1. **Mixed output versions**: manuscript values are now injected from
   `outputs/corrected/reports/*.json` by `write_corrected_manuscript.py`.
2. **Sampler not connected to reported analysis**: learned-energy chain
   diagnostics are reported for every analyzed variant and compared with the
   exact finite target.
3. **Invalid proposal/acceptance pairing**: the learned proposal now uses the
   complete Hastings correction; detailed balance is unit tested.
4. **Insufficient 10–25-step schedule**: rejected after synthetic and real-data
   mixing failures. The final schedule scales with candidate-set size and
   uncertainty.
5. **Shortest-path degeneracy**: candidate states now include valid directed
   gene-gene routes before the disease/phenotype layer.
6. **False mechanistic traversal claim**: Pathway and BiologicalProcess nodes
   are explicitly annotations because they do not connect onward to phenotype
   in this AlzKB slice.
7. **Likely-label leakage**: sequence calibration, path scoring, graph-feature
   generation, and classifier training use definitive labels only; outer folds
   refit the complete pipeline.
8. **Missing comparisons**: uniform, fixed-temperature, no-confidence,
   fixed-temperature/no-confidence, and no-mechanism comparisons are included.
9. **Unmeasured learned trajectory diversity**: final-route diversity is now
   measured from the learned chain.
10. **Unsupported gene ranking**: headline alternative-metric gene rankings
    require at least five variants.
11. **Missing calibration analysis**: locked Brier score, expected calibration
    error, and reliability curves are reported.
12. **No synthetic check**: a new 53/256/1,024-state experiment validates
    sampler scaling and target broadening.

## Remaining scientific limitations

- Locked validation is internal to ClinVar and is not an external cohort.
- The graph-augmented model improves ranking discrimination but worsens
  probability calibration.
- VUS assignments have no outcome labels and are exploratory.
- The shared AlzKB disease-to-symptom layer limits endpoint-set specificity.
- Training uses only 121 definitive variants.
