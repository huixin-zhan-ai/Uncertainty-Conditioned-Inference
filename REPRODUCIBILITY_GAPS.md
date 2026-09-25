# Reproducibility gaps requiring a frozen scientific decision

The suite preserves and runs the current implementation, but the
following points should be resolved before claiming exact end-to-end
reproduction of every manuscript sentence.

1. **Distribution versus sampled diffusion trajectory.** The experiment
   tables use the learned path-softmax distribution directly. The
   manuscript also describes a Metropolis revision chain with a separate
   temperature. It does not specify whether reported phenotype mass is
   the direct softmax, the final chain state over repeated runs, or
   empirical mass over visited states. The current results correspond to
   the direct learned softmax.
2. **Branch entropy has two operational levels.** Experiment 4 uses
   entropy over grouped phenotype mass, while VUS reclassification uses
   entropy over raw endpoint mass. The manuscript should name these
   separately or choose one definition.
3. **Published-number drift.** Some polished manuscript values differ
   from the latest CSV summaries (including uncertainty-propagation
   quartiles and the most alternative-metric-sensitive gene). The CSVs
   copied into each experiment folder are the executable run artifacts;
   manuscript values must be reconciled against a frozen run before
   submission.
4. **Upstream Evo2 regeneration.** The score table and its producing
   notebook are included. Regenerating scores from sequence requires the
   Evo2 model/GPU environment and GRCh38 reference resources, which are
   not bundled because of their size and licensing/distribution terms.

These are scientific specification issues, not missing-file errors.
They should be resolved by freezing one run manifest and updating either
the implementation or manuscript text to match it.
