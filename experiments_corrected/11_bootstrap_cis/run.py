#!/usr/bin/env python3
"""E5 -- Uncertainty intervals for every reported association, and a
kNN-uncertainty robustness recomputation.

For each Spearman rho reported in the paper we give (i) a naive IID variant
bootstrap CI and (ii) a gene-cluster bootstrap CI (resampling the 15 genes
with replacement, 1000 replicates). Three genes carry 94% of variants, so
cluster CIs are expected to be much wider; both are reported so the
dependence structure is visible rather than hidden.

kNN robustness: the exact target is RECOMPUTED with the kNN-calibrated
entropy in place of the isotonic entropy (same frozen scorer, same support),
and rho(H_knn, grouped entropy) is reported with the same intervals.
Outputs are NEW tables only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
import corrected_core as core  # noqa: E402

TABLES = ROOT / "outputs" / "corrected" / "tables"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
BOOT = 1000


def boot_rho(x, y, clusters, seed=core.RNG_SEED):
    x = np.asarray(x, float); y = np.asarray(y, float); clusters = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[ok], y[ok], clusters[ok]
    point = spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    n = len(x)
    iid = []
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        iid.append(spearmanr(x[idx], y[idx]).correlation)
    uniq = np.unique(clusters); by = {c: np.where(clusters == c)[0] for c in uniq}
    clu = []
    for _ in range(BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[c] for c in pick])
        r = spearmanr(x[idx], y[idx]).correlation
        if np.isfinite(r):
            clu.append(r)
    iid = np.array(iid); clu = np.array(clu)
    return dict(rho=float(point), n=int(n), n_genes=int(len(uniq)),
                iid_ci_low=float(np.nanpercentile(iid, 2.5)), iid_ci_high=float(np.nanpercentile(iid, 97.5)),
                cluster_ci_low=float(np.percentile(clu, 2.5)), cluster_ci_high=float(np.percentile(clu, 97.5)),
                cluster_n_valid=int(len(clu)))


def main():
    t0 = time.time()
    prop = pd.read_csv(TABLES / "propagation.csv")
    branch = pd.read_csv(TABLES / "branching.csv")
    diag = pd.read_csv(TABLES / "diffusion_diagnostics.csv")
    rank = pd.read_csv(TABLES / "ranking.csv")
    abl = pd.read_csv(TABLES / "ablations.csv")
    rows = []

    def add(name, frame, ycol, xcol="isotonic_entropy"):
        r = boot_rho(frame[xcol], frame[ycol], frame["gene_id"]); r.update(quantity=name, x=xcol, y=ycol); rows.append(r)

    add("grouped phenotype entropy", prop, "phenotype_entropy_bits")
    add("top-1 group mass", prop, "top1_mass")
    add("effective groups", prop, "effective_groups")
    add("raw endpoint entropy", branch, "raw_endpoint_entropy_bits")
    add("unique final paths (512 chains)", diag, "unique_final_paths")
    add("unique final routes (512 chains)", diag, "unique_final_routes")
    add("acceptance rate", diag, "acceptance_rate")
    add("perturbation JS (+/-0.30)", rank, "mean_js")
    add("perturbation top-1 switch", rank, "mean_top1_switch")
    add("perturbation top-5 overlap", rank, "mean_top5_overlap")
    for model, g in abl.groupby("model"):
        add(f"ablation:{model} grouped entropy", g, "phenotype_entropy_bits")

    # kNN robustness: recompute the exact target with H_knn
    graph, _, _ = core.load_graph_and_raw_variants()
    rf = core.relation_frequencies(graph)
    scorer = core.LearnedPathScorer.load()
    sig = pd.read_csv(TABLES / "variants_locked_signals.csv"); sig["variant_id"] = sig["variant_id"].astype(str)
    sig = sig[sig["variant_id"].isin(set(diag["variant_id"].astype(str)))].reset_index(drop=True)
    knn_rows = []
    for row in sig.itertuples(index=False):
        cands = core.enumerate_candidate_paths(graph, row.gene_id)
        if not cands:
            continue
        f, groups = core.feature_matrix(graph, cands, rf, row.pathogenicity_prob, row.knn_entropy)
        u = scorer.utilities(f)
        dist = core.aggregate_distribution(groups, core.exact_path_probabilities(u, row.knn_entropy))
        p = np.asarray(list(dist.values()))
        knn_rows.append(dict(variant_id=row.variant_id, gene_id=row.gene_id, knn_entropy=row.knn_entropy,
                             isotonic_entropy=row.isotonic_entropy, phenotype_entropy_bits_knn=core.entropy_from_prob(p),
                             top1_mass_knn=float(p.max())))
    knn = pd.DataFrame(knn_rows)
    knn.to_csv(TABLES / "propagation_knn_recomputed.csv", index=False)
    add("grouped entropy, target recomputed with H_knn", knn, "phenotype_entropy_bits_knn", xcol="knn_entropy")
    add("top-1 mass, target recomputed with H_knn", knn, "top1_mass_knn", xcol="knn_entropy")
    r = boot_rho(sig.isotonic_entropy, sig.knn_entropy, sig.gene_id); r.update(quantity="H_iso vs H_knn agreement", x="isotonic_entropy", y="knn_entropy"); rows.append(r)

    out = pd.DataFrame(rows)[["quantity", "x", "y", "rho", "iid_ci_low", "iid_ci_high", "cluster_ci_low", "cluster_ci_high", "n", "n_genes"]]
    out.to_csv(TABLES / "bootstrap_cis.csv", index=False)
    out.to_csv(OUT / "bootstrap_cis.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(dict(rows=out.to_dict("records"), runtime_s=time.time() - t0), indent=2))
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
