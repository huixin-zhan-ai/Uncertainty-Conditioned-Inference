#!/usr/bin/env python3
"""E1/E2 -- Non-learned and commitment baselines on the identical support.

Baselines (all evaluated on the same enumerated candidate support and the same
phenotype-group pushforward as the full model):
  uniform_paths      : uniform mass over candidate paths (paper's existing control)
  uniform_endpoints  : uniform mass over reachable endpoints (path multiplicity removed)
  ppr                : personalized PageRank from the start gene on the retained
                       AlzKB graph (alpha=0.85), restricted to the reachable
                       endpoints and renormalized. Ignores (p, H) by construction.
  beam_k (k=1,5,20)  : autoregressive-style commitment: top-k paths under the
                       learned utility U (which still sees p and 1-H through the
                       interaction features) with uniform mass, i.e. no
                       uncertainty-scaled temperature.
  full               : exact uncertainty-conditioned target (reference).

Distribution-level metrics per variant: grouped entropy, effective groups,
top-1 mass, AD-core mass, JS to the full model; and the H-response
Spearman rho(H, entropy) per model.

Downstream single-split diagnostic (clearly NOT the paper's 5-fold protocol):
logistic regression on standardized [p, H] (+ four distribution features of a
model) fit on the definitive locked variants and evaluated on the locked
likely-label variants; AUROC with gene-cluster bootstrap (15 genes) and the
paired difference versus sequence-only. Inputs: frozen signals table, frozen
scorer, frozen split manifest. Outputs are NEW tables only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
import corrected_core as core  # noqa: E402

TABLES = ROOT / "outputs" / "corrected" / "tables"
MANIFESTS = ROOT / "outputs" / "corrected" / "manifests"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
BEAM_KS = [1, 5, 20]
BOOT = 1000


def metrics(dist):
    p = np.asarray(list(dist.values()), dtype=float)
    return dict(
        phenotype_entropy_bits=core.entropy_from_prob(p),
        effective_groups=core.effective_support(p),
        top1_mass=float(p.max()) if len(p) else np.nan,
        ad_core_mass=float(sum(dist.get(g, 0.0) for g in core.AD_CORE_GROUPS)),
    )


def ppr_endpoint_distribution(graph, gene, endpoints, alpha=0.85):
    scores = nx.pagerank(graph, alpha=alpha, personalization={gene: 1.0}, max_iter=200, tol=1e-10)
    mass = np.array([scores.get(e, 0.0) for e in endpoints], dtype=float)
    if mass.sum() <= 0:
        mass = np.ones(len(endpoints))
    return mass / mass.sum()


def cluster_bootstrap_auc(y, prob, clusters, seed, y2=None, prob2=None):
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    aucs, diffs = [], []
    for _ in range(BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        a = roc_auc_score(y[idx], prob[idx])
        aucs.append(a)
        if prob2 is not None:
            diffs.append(a - roc_auc_score(y[idx], prob2[idx]))
    aucs = np.array(aucs)
    out = dict(auroc_ci_low=float(np.percentile(aucs, 2.5)), auroc_ci_high=float(np.percentile(aucs, 97.5)),
               n_boot_valid=int(len(aucs)))
    if diffs:
        d = np.array(diffs)
        out.update(diff_ci_low=float(np.percentile(d, 2.5)), diff_ci_high=float(np.percentile(d, 97.5)),
                   diff_p_gt0=float((d > 0).mean()))
    return out


def main():
    t0 = time.time()
    graph, _, _ = core.load_graph_and_raw_variants()
    relation_frequency = core.relation_frequencies(graph)
    scorer = core.LearnedPathScorer.load()
    variants = pd.read_csv(TABLES / "variants_locked_signals.csv")
    variants["variant_id"] = variants["variant_id"].astype(str)
    diag = pd.read_csv(TABLES / "diffusion_diagnostics.csv")
    valid = set(diag["variant_id"].astype(str))
    variants = variants[variants["variant_id"].isin(valid)].reset_index(drop=True)
    print(f"loaded {len(variants)} valid variants in {time.time()-t0:.0f}s", flush=True)

    # per-gene non-learned baselines
    gene_cache = {}
    for gene in variants["gene_id"].unique():
        candidates = core.enumerate_candidate_paths(graph, gene)
        if not candidates:
            continue
        groups = [core.endpoint_group(graph, p[-1]) for p in candidates]
        endpoints = sorted({p[-1] for p in candidates})
        ep_groups = [core.endpoint_group(graph, e) for e in endpoints]
        ppr = ppr_endpoint_distribution(graph, gene, endpoints)
        gene_cache[gene] = dict(
            groups=groups, endpoints=endpoints,
            uniform_paths=core.aggregate_distribution(groups, np.ones(len(groups)) / len(groups)),
            uniform_endpoints=core.aggregate_distribution(ep_groups, np.ones(len(endpoints)) / len(endpoints)),
            ppr=core.aggregate_distribution(ep_groups, ppr),
            ppr_endpoint_top1=float(ppr.max()),
        )
        print(f"  PPR done for {gene} ({len(endpoints)} endpoints)  {time.time()-t0:.0f}s", flush=True)

    rows = []
    for row in variants.itertuples(index=False):
        cache = gene_cache.get(row.gene_id)
        if cache is None:
            continue
        candidates = core.enumerate_candidate_paths(graph, row.gene_id)
        features, groups = core.feature_matrix(graph, candidates, relation_frequency,
                                               row.pathogenicity_prob, row.isotonic_entropy)
        utilities = scorer.utilities(features)
        full = core.aggregate_distribution(groups, core.exact_path_probabilities(utilities, row.isotonic_entropy))
        models = {"full": full, "uniform_paths": cache["uniform_paths"],
                  "uniform_endpoints": cache["uniform_endpoints"], "ppr": cache["ppr"]}
        order = np.argsort(-utilities)
        for k in BEAM_KS:
            top = order[:k]
            models[f"beam_{k}"] = core.aggregate_distribution([groups[i] for i in top], np.ones(len(top)) / len(top))
        for name, dist in models.items():
            m = metrics(dist)
            m.update(model=name, variant_id=row.variant_id, gene_id=row.gene_id, gene_symbol=row.gene_symbol,
                     clinical_significance=row.clinical_significance, pathogenicity_prob=row.pathogenicity_prob,
                     isotonic_entropy=row.isotonic_entropy, js_to_full=core.js_divergence(dist, full))
            rows.append(m)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "baselines.csv", index=False)
    frame.to_csv(TABLES / "baselines.csv", index=False)

    # distribution-level summary
    summ = []
    for name, g in frame.groupby("model"):
        rho = spearmanr(g.isotonic_entropy, g.phenotype_entropy_bits).correlation
        rho_top1 = spearmanr(g.isotonic_entropy, g.top1_mass).correlation
        summ.append(dict(model=name, entropy_mean=g.phenotype_entropy_bits.mean(), top1_mean=g.top1_mass.mean(),
                         effective_groups_mean=g.effective_groups.mean(), js_to_full_mean=g.js_to_full.mean(),
                         rho_H_entropy=rho, rho_H_top1=rho_top1, n=len(g)))
    summary = pd.DataFrame(summ)

    # downstream single-split diagnostic
    split = json.loads((MANIFESTS / "locked_split.json").read_text())
    def_ids = set(map(str, split["definitive_ids"]))
    likely_ids = set(map(str, split["likely_validation_ids"]))
    base = variants.set_index("variant_id")
    y_all = base["clinical_significance"].isin(core.DEFINITIVE_POS | core.LIKELY_POS).astype(int)
    diag_rows = []
    seq_prob_cache = None
    for name in ["sequence_only", "full", "ppr", "uniform_paths", "beam_5"]:
        feats = base[["pathogenicity_prob", "isotonic_entropy"]].copy()
        if name != "sequence_only":
            d = frame[frame.model == name].set_index("variant_id")[
                ["phenotype_entropy_bits", "effective_groups", "top1_mass", "ad_core_mass"]]
            feats = feats.join(d, how="inner")
        tr = feats.index.isin(def_ids); te = feats.index.isin(likely_ids)
        scaler = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(feats[tr]), y_all.loc[feats.index[tr]])
        prob = clf.predict_proba(scaler.transform(feats[te]))[:, 1]
        y = y_all.loc[feats.index[te]].to_numpy()
        clusters = base.loc[feats.index[te], "gene_id"].to_numpy()
        auc = roc_auc_score(y, prob)
        if name == "sequence_only":
            seq_prob_cache = (feats.index[te], prob)
            ci = cluster_bootstrap_auc(y, prob, clusters, core.RNG_SEED)
            diff = np.nan
        else:
            ref = pd.Series(seq_prob_cache[1], index=seq_prob_cache[0]).loc[feats.index[te]].to_numpy()
            ci = cluster_bootstrap_auc(y, prob, clusters, core.RNG_SEED, prob2=ref)
            diff = auc - roc_auc_score(y, ref)
        diag_rows.append(dict(model=name, n_train=int(tr.sum()), n_test=int(te.sum()), auroc=auc,
                              diff_vs_sequence=diff, **ci))
    downstream = pd.DataFrame(diag_rows)
    summary.to_csv(TABLES / "baselines_summary.csv", index=False)
    downstream.to_csv(TABLES / "baselines_downstream_single_split.csv", index=False)
    out = dict(distribution_summary=summary.to_dict("records"), downstream=downstream.to_dict("records"),
               n_genes=int(len(gene_cache)), runtime_s=time.time() - t0)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(summary.round(4).to_string(index=False)); print(downstream.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
