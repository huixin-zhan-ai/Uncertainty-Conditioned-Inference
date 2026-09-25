#!/usr/bin/env python3
"""H2-C -- Gene-held-out masked-edge recovery across prespecified mask seeds (C3 panel C).

Prespecified design (fixed before any result was inspected):
  mask seeds   : 42 (the H2 seed), 7, 123   -- 20% of disease->symptom edges each
  gene folds   : F1 = {APP}, F2 = {PSEN1}, F3 = {PSEN2}, F4 = all remaining study genes.
                 Folds are defined by the anchor distribution: APP/PSEN1/PSEN2 carry
                 118 of the 121 definitive anchors; every gene is held out exactly once.
  per split    : (1) mask edges -> G';  (2) refit the sequence-score calibrators
                 (Platt / isotonic / kNN) on definitive variants of TRAINING genes only;
                 (3) refit the path energy on TRAINING-gene anchors on G';
                 (4) evaluate HELD-OUT genes on the filtered top-k protocol with the
                 same candidates for every method.
  methods      : full theta(H), fixed theta, uniform paths, PPR, popularity (G' degree),
                 random (analytic), DistMult (trained on G' edges once per seed; it uses
                 no variant labels, so gene folds do not apply to it)
  reporting    : primary recall@10; recall@5/25, MRR; per-gene means; gene-balanced
                 mean per seed; pooled variant-level mean with gene-cluster 95% CI;
                 paired differences full - other with gene-cluster CIs; spread across seeds.
Held-out genes stay in the graph as nodes; only their labels are withheld from fitting
and the tested associations are masked.  Outputs: tables/hetionet_geneholdout*.csv/json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hetionet_common as hc  # noqa: E402
import run_12a as h1  # noqa: E402
import run_12b as h2  # noqa: E402
import run_12d_controls as hd  # noqa: E402

core = hc.core
ROOT = hc.ROOT; TABLES = hc.TABLES
SEEDS = (42, 7, 123)
FOLDS = {"F1_APP": ["APP"], "F2_PSEN1": ["PSEN1"], "F3_PSEN2": ["PSEN2"]}  # F4 = rest, built at runtime
KS = hd.KS; PRIMARY = hd.PRIMARY
DEFPOS = core.DEFINITIVE_POS; DEFNEG = core.DEFINITIVE_NEG


def distmult_scorer(Gp, bases, seed):
    assoc_genes = {u for u, v, a in Gp.edges(data=True) if a["relation"] == "GENEASSOCIATESWITHDISEASE"} | set(bases)
    ents = {}
    def eid(x):
        if x not in ents:
            ents[x] = len(ents)
        return ents[x]
    triples = []
    for u, v, a in Gp.edges(data=True):
        rel = a["relation"]
        if rel == "GENEASSOCIATESWITHDISEASE":
            triples.append((eid(u), 0, eid(v)))
        elif rel == "DISEASEMANIFESTSSYMPTOM":
            triples.append((eid(u), 1, eid(v)))
        elif rel in ("GENEINTERACTSWITHGENE", "GENEREGULATESGENE", "GENECOVARIESWITHGENE") and (u in bases or v in bases) and u in assoc_genes and v in assoc_genes:
            triples.append((eid(u), 2, eid(v)))
    E, R = h2.train_distmult(np.array(triples), len(ents), 3, seed=seed)
    diseases_all = [d for d, a in Gp.nodes(data=True) if a["type"] == "Disease" and d in ents]
    def score(g, cands):
        gi = ents[g]; pd_ = np.array([expit(h2.distmult_score(E, R, gi, 0, ents[d])) for d in diseases_all]); out = np.zeros(len(cands))
        for j, s in enumerate(cands):
            if s in ents:
                out[j] = float(np.max(pd_ * np.array([expit(h2.distmult_score(E, R, ents[d], 1, ents[s])) for d in diseases_all])))
        return out
    return score


def main():
    t0 = time.time(); RNG = np.random.default_rng(core.RNG_SEED)
    G = hc.build_graph(); variants0 = hc.study_variants(G)
    dps = [(u, v) for u, v, a in G.edges(data=True) if a["relation"] == "DISEASEMANIFESTSSYMPTOM"]
    all_genes = sorted(variants0.gene_symbol.unique()); folds = dict(FOLDS); folds["F4_rest"] = [g for g in all_genes if g not in {"APP", "PSEN1", "PSEN2"}]
    rows = []; split_log = []
    for seed in SEEDS:
        rs = np.random.default_rng(seed); idx = rs.permutation(len(dps)); held = {dps[i] for i in idx[: int(h2.HOLDOUT * len(dps))]}
        Gp = G.copy(); Gp.remove_edges_from(held); hc.reset_caches(); rf = core.relation_frequencies(Gp)
        bases = {g: h1.base_features(Gp, g, rf) for g in variants0.gene_id.unique()}; bases = {g: b for g, b in bases.items() if b is not None}
        valid0 = variants0[variants0.gene_id.isin(bases)].reset_index(drop=True)
        disease_of = {g: [d for d in G.successors(g) if G.nodes[d]["type"] == "Disease"] for g in bases}
        per_gene = {}
        for g, b in bases.items():
            ends = set(b["endpoints"]); known = {s for d in disease_of[g] for s in Gp.successors(d) if Gp.nodes[s]["type"] == "Phenotype"}
            heldpos = {s for d in disease_of[g] for (dd, s) in held if dd == d}; cand = sorted(ends - known); pos = sorted(heldpos & set(cand))
            per_gene[g] = dict(cand=cand, pos=pos, n_all=len(heldpos))
        eval_genes = {g for g, v in per_gene.items() if v["pos"] and len(v["cand"]) > len(v["pos"])}
        deg = {}
        for u, v, a in Gp.edges(data=True):
            if a["relation"] == "DISEASEMANIFESTSSYMPTOM":
                deg[v] = deg.get(v, 0) + 1
        ppr = {g: dict(zip(sorted(set(b["endpoints"])), h1.ppr_endpoint_mass(Gp, g, sorted(set(b["endpoints"]))))) for g, b in bases.items() if g in eval_genes}
        dm = distmult_scorer(Gp, bases, seed); dm_cache = {g: dm(g, per_gene[g]["cand"]) for g in eval_genes}
        print(f"seed {seed}: G' built, {len(eval_genes)} eval genes, DistMult trained  {time.time()-t0:.0f}s", flush=True)
        for fold, held_genes in folds.items():
            train_mask = ~valid0.gene_symbol.isin(held_genes)
            train_def = valid0[train_mask & valid0.clinical_significance.isin(DEFPOS | DEFNEG)]
            models = core.fit_signal_models(valid0[train_mask], training_ids=train_def.variant_id.astype(str).tolist())
            valid = core.apply_signal_models(valid0, models)  # recalibrated p, H for everyone; held-out genes never entered the fit
            samples = core.build_training_samples(Gp, valid, rf, train_def.variant_id.astype(str).tolist())
            sc_d = core.train_path_scorer(samples, l2=1e-2, seed=core.RNG_SEED); sc_i = h1.train_informed(samples)
            scorer = sc_d if h1.objective_value(sc_d, samples) <= h1.objective_value(sc_i, samples) else sc_i
            split_log.append(dict(seed=seed, fold=fold, held_out_genes=";".join(held_genes), n_train_anchors=len(samples), n_calib_pos=int(train_def.clinical_significance.isin(DEFPOS).sum()), n_calib_neg=int(train_def.clinical_significance.isin(DEFNEG).sum()), objective=float(h1.objective_value(scorer, samples))))
            for r in valid[valid.gene_symbol.isin(held_genes)].itertuples(index=False):
                if r.gene_id not in eval_genes:
                    continue
                b = bases[r.gene_id]; pg = per_gene[r.gene_id]; cand = pg["cand"]; pos = pg["pos"]
                phi = h1.features_for(b, r.pathogenicity_prob, r.isotonic_entropy); u = scorer.utilities(phi); ends = b["endpoints"]
                def em(probs):
                    d = core.aggregate_distribution(ends, probs); return np.array([d.get(s, 0.0) for s in cand])
                scores = {"full": em(core.exact_path_probabilities(u, r.isotonic_entropy)), "fixed_temperature": em(core.exact_path_probabilities(u, r.isotonic_entropy, fixed_temperature=True)),
                          "uniform_paths": em(np.ones(len(u)) / len(u)), "ppr": np.array([ppr[r.gene_id].get(s, 0.0) for s in cand]),
                          "popularity": np.array([deg.get(s, 0) for s in cand]) + RNG.random(len(cand)) * 1e-6, "distmult": dm_cache[r.gene_id]}
                base = dict(seed=seed, fold=fold, variant_id=str(r.variant_id), gene_id=r.gene_id, gene_symbol=r.gene_symbol, isotonic_entropy=float(r.isotonic_entropy), n_candidates=len(cand), n_positives=len(pos))
                for name, s in scores.items():
                    rows.append(dict(base, model=name, **hd.rank_metrics(s, cand, pos)))
                rows.append(dict(base, model="random", **hd.random_expectation(len(cand), len(pos))))
            print(f"  seed {seed} {fold}: held-out {held_genes if len(held_genes) < 4 else str(len(held_genes)) + ' genes'} done  {time.time()-t0:.0f}s", flush=True)
    res = pd.DataFrame(rows); res.to_csv(TABLES / "hetionet_geneholdout.csv", index=False)
    pd.DataFrame(split_log).to_csv(TABLES / "hetionet_geneholdout_splits.csv", index=False)

    # ---------------- summaries
    mets = [f"recall@{k}" for k in KS] + ["mrr"]
    per_gene_tab = res.groupby(["seed", "model", "gene_symbol"])[mets].mean().reset_index(); per_gene_tab.to_csv(TABLES / "hetionet_geneholdout_by_gene.csv", index=False)
    summ = []
    for (seed, model), g in res.groupby(["seed", "model"]):
        row = dict(seed=seed, model=model, n=len(g), n_genes=g.gene_id.nunique())
        for met in mets:
            m, lo, hi = hd.cluster_mean_ci(g[met], g.gene_id, RNG); row[met] = m; row[f"{met}_lo"] = lo; row[f"{met}_hi"] = hi
            row[f"{met}_gene_balanced"] = float(g.groupby("gene_id")[met].mean().mean())
        summ.append(row)
    summ = pd.DataFrame(summ); summ.to_csv(TABLES / "hetionet_geneholdout_summary.csv", index=False)
    # pooled over seeds (each variant appears once per seed): cluster on gene
    pooled = []
    for model, g in res.groupby("model"):
        row = dict(model=model, n=len(g), n_genes=g.gene_id.nunique())
        for met in mets:
            m, lo, hi = hd.cluster_mean_ci(g[met], g.gene_id, RNG); row[met] = m; row[f"{met}_lo"] = lo; row[f"{met}_hi"] = hi
            row[f"{met}_gene_balanced"] = float(g.groupby("gene_id")[met].mean().mean())
            row[f"{met}_seed_min"] = float(summ[summ.model == model][f"{met}_gene_balanced"].min()); row[f"{met}_seed_max"] = float(summ[summ.model == model][f"{met}_gene_balanced"].max())
        pooled.append(row)
    pooled = pd.DataFrame(pooled); pooled.to_csv(TABLES / "hetionet_geneholdout_pooled.csv", index=False)
    paired = {}
    for met in [f"recall@{k}" for k in KS]:
        piv = res.pivot_table(index=["seed", "variant_id"], columns="model", values=met); cl = res.drop_duplicates(["seed", "variant_id"]).set_index(["seed", "variant_id"]).loc[piv.index, "gene_id"].to_numpy()
        for other in ["fixed_temperature", "distmult", "uniform_paths", "ppr", "popularity", "random"]:
            m, lo, hi = hd.cluster_mean_ci((piv["full"] - piv[other]).to_numpy(), cl, RNG); paired[f"full_minus_{other}|{met}"] = dict(mean=m, lo=lo, hi=hi)
    out = dict(seeds=list(SEEDS), folds={k: v for k, v in folds.items()}, n_rows=len(res), n_genes=int(res.gene_id.nunique()), pooled=pooled.to_dict("records"), by_seed=summ.to_dict("records"), paired=paired,
               splits=split_log, runtime_s=time.time() - t0)
    (TABLES / "hetionet_geneholdout_results.json").write_text(json.dumps(out, indent=2, default=str))
    pd.set_option("display.width", 240); print(pooled[["model", "recall@5", "recall@10", "recall@10_lo", "recall@10_hi", "recall@10_gene_balanced", "recall@10_seed_min", "recall@10_seed_max", "recall@25", "mrr"]].round(3).to_string(index=False))
    print(summ[summ.model == "full"][["seed", "n", "n_genes", "recall@10", "recall@10_lo", "recall@10_hi", "recall@10_gene_balanced"]].round(3).to_string(index=False))
    print(json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in paired.items() if "recall@10" in k}, indent=1))
    print(per_gene_tab[per_gene_tab.model == "full"].pivot(index="gene_symbol", columns="seed", values="recall@10").round(2).to_string())
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
