#!/usr/bin/env python3
"""H2 -- Held-out link prediction on Hetionet with a trained baseline.

Task. 20% of Disease->Symptom edges are held out (seed 42), giving G'. For each
study variant (gene g with frozen p, H), the candidate set is the reachable
endpoint set on G'. Known positives (symptoms of g's own diseases still linked
in G') are filtered out of the ranking (standard filtered protocol). Held-out
positives are symptoms of g's own diseases whose edge was hidden but which are
still reachable on G' through another disease or a gene relay. Methods rank
the filtered candidates; we report recall@10/25 and MRR per variant, and
stratify by H to test whether the uncertainty coupling changes downstream
behaviour on real held-out labels.

Methods: full (theta(H)), fixed theta, uniform paths, PPR on G', beam-5, and a
trained DistMult knowledge-graph embedding (dim 32, 5 negatives, Adagrad) on
G' edges (all disease-gene, remaining disease-symptom, and gene-gene edges
among disease-associated genes), scored by two-hop composition
max_d sigma(f(g,assoc,d)) * sigma(f(d,presents,s)). DistMult sees no (p, H).
Outputs are NEW tables prefixed hetionet_linkpred_*.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hetionet_common as hc  # noqa: E402
import run_12a as h1  # noqa: E402  (base_features, features_for, ppr_endpoint_mass, train_informed, objective_value)
core = hc.core
ROOT = hc.ROOT; TABLES = hc.TABLES
OUT = HERE / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
MANIFESTS = ROOT / "outputs" / "corrected" / "manifests"
RNG = np.random.default_rng(core.RNG_SEED)
HOLDOUT = 0.20
KS = (10, 25)


# ----------------------------------------------------------------- DistMult (numpy)
def train_distmult(triples, n_ent, n_rel, dim=32, epochs=80, negs=3, lr=0.1, l2=1e-4, seed=42):
    """DistMult with logistic loss, uniform negatives, Adagrad; sparse-matrix gradient accumulation."""
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.1, (n_ent, dim)); R = rng.normal(0, 0.1, (n_rel, dim))
    gE = np.full_like(E, 1e-8); gR = np.full_like(R, 1e-8)
    h0, r0, t0 = triples[:, 0], triples[:, 1], triples[:, 2]; n = len(triples)
    for ep in range(epochs):
        hh = np.concatenate([h0] + [h0] * negs + [rng.integers(0, n_ent, n) for _ in range(negs)])
        tt = np.concatenate([t0] + [rng.integers(0, n_ent, n) for _ in range(negs)] + [t0] * negs)
        rr = np.concatenate([r0] * (1 + 2 * negs)); lab = np.concatenate([np.ones(n), np.zeros(2 * negs * n)])
        s = np.einsum("ij,ij,ij->i", E[hh], R[rr], E[tt]); g = expit(s) - lab
        m = len(hh)
        Sh = sp.csr_matrix((np.ones(m), (hh, np.arange(m))), shape=(n_ent, m))
        St = sp.csr_matrix((np.ones(m), (tt, np.arange(m))), shape=(n_ent, m))
        Sr = sp.csr_matrix((np.ones(m), (rr, np.arange(m))), shape=(n_rel, m))
        dE = (Sh @ (g[:, None] * R[rr] * E[tt]) + St @ (g[:, None] * R[rr] * E[hh])) / n + l2 * E
        dR = (Sr @ (g[:, None] * E[hh] * E[tt])) / n + l2 * R
        gE += dE ** 2; gR += dR ** 2; E -= lr * dE / np.sqrt(gE); R -= lr * dR / np.sqrt(gR)
    return E, R


def distmult_score(E, R, h, r, t):
    return float(np.einsum("j,j,j->", E[h], R[r], E[t]))


# ----------------------------------------------------------------- main
def main():
    t0 = time.time()
    G = hc.build_graph(); hc.reset_caches()
    variants = hc.study_variants(G)
    # ---- hold out 20% of disease->symptom edges
    dps = [(u, v) for u, v, a in G.edges(data=True) if a["relation"] == "DISEASEMANIFESTSSYMPTOM"]
    idx = RNG.permutation(len(dps)); held = {dps[i] for i in idx[: int(HOLDOUT * len(dps))]}
    Gp = G.copy(); Gp.remove_edges_from(held)
    print(f"held out {len(held)} of {len(dps)} disease->symptom edges  {time.time()-t0:.0f}s", flush=True)
    rf = core.relation_frequencies(Gp)

    # ---- supports on G', retrain scorer on G' (same 121 anchors)
    bases = {g: h1.base_features(Gp, g, rf) for g in variants.gene_id.unique()}
    bases = {g: b for g, b in bases.items() if b is not None}
    valid = variants[variants.gene_id.isin(bases)].reset_index(drop=True)
    split = json.loads((MANIFESTS / "locked_split.json").read_text())
    def_ids = [i for i in map(str, split["definitive_ids"]) if i in set(valid.variant_id)]
    samples = core.build_training_samples(Gp, valid, rf, def_ids)
    sc_d = core.train_path_scorer(samples, l2=1e-2, seed=core.RNG_SEED); sc_i = h1.train_informed(samples)
    scorer = sc_d if h1.objective_value(sc_d, samples) <= h1.objective_value(sc_i, samples) else sc_i
    print(f"scorer on G' trained (objective {h1.objective_value(scorer, samples):.4f})  {time.time()-t0:.0f}s", flush=True)

    # ---- positives / filtered candidates per gene
    disease_of = {g: [d for d in G.successors(g) if G.nodes[d]["type"] == "Disease"] for g in bases}
    per_gene = {}
    for g, b in bases.items():
        ends = set(b["endpoints"])
        known = {s for d in disease_of[g] for s in Gp.successors(d) if Gp.nodes[s]["type"] == "Phenotype"}
        heldpos = {s for d in disease_of[g] for (dd, s) in held if dd == d}
        cand = sorted(ends - known)                       # filtered candidates
        pos = sorted(heldpos & set(cand))                 # recoverable held-out positives
        per_gene[g] = dict(cand=cand, pos=pos, n_known=len(known), n_heldout_total=len(heldpos), n_unrecoverable=len(heldpos - ends))
    audit = pd.DataFrame([dict(gene_id=g, gene_symbol=valid[valid.gene_id == g].gene_symbol.iloc[0], n_candidates_filtered=len(v["cand"]), n_positives=len(v["pos"]),
                               n_known_filtered=v["n_known"], n_heldout_total=v["n_heldout_total"], n_unrecoverable=v["n_unrecoverable"]) for g, v in per_gene.items()])
    audit.to_csv(TABLES / "hetionet_linkpred_audit.csv", index=False); print(audit.to_string(index=False), flush=True)
    eval_genes = {g for g, v in per_gene.items() if v["pos"] and len(v["cand"]) > len(v["pos"])}

    # ---- PPR on G'
    ppr = {g: dict(zip(sorted(set(b["endpoints"])), h1.ppr_endpoint_mass(Gp, g, sorted(set(b["endpoints"]))))) for g, b in bases.items()}
    print(f"PPR on G' done {time.time()-t0:.0f}s", flush=True)

    # ---- DistMult on G'
    assoc_genes = {u for u, v, a in Gp.edges(data=True) if a["relation"] == "GENEASSOCIATESWITHDISEASE"} | set(bases)
    ents = {}; rels = {"assoc": 0, "presents": 1, "gg": 2}
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
    triples = np.array(triples); print(f"DistMult: {len(ents)} entities, {len(triples)} triples", flush=True)
    E, R = train_distmult(triples, len(ents), 3); print(f"DistMult trained {time.time()-t0:.0f}s", flush=True)
    diseases_all = [d for d, a in Gp.nodes(data=True) if a["type"] == "Disease" and d in ents]
    def dm_scores(g, cands):
        gi = ents[g]; out = np.zeros(len(cands))
        pd_ = np.array([expit(distmult_score(E, R, gi, 0, ents[d])) for d in diseases_all])
        for j, s in enumerate(cands):
            if s not in ents:
                continue
            ps = np.array([expit(distmult_score(E, R, ents[d], 1, ents[s])) for d in diseases_all])
            out[j] = float(np.max(pd_ * ps))
        return out
    dm_cache = {g: dm_scores(g, per_gene[g]["cand"]) for g in eval_genes}
    print(f"DistMult scoring done {time.time()-t0:.0f}s", flush=True)

    # ---- evaluate per variant
    def rank_metrics(scores, cand, pos):
        order = np.argsort(-scores, kind="stable"); ranked = [cand[i] for i in order]; posset = set(pos)
        ranks = [i + 1 for i, s in enumerate(ranked) if s in posset]
        out = {f"recall@{k}": float(sum(r <= k for r in ranks) / len(pos)) for k in KS}
        out["mrr"] = float(np.mean([1.0 / r for r in ranks])); return out
    rows = []
    for i, r in enumerate(valid.itertuples(index=False)):
        if r.gene_id not in eval_genes:
            continue
        b = bases[r.gene_id]; pg = per_gene[r.gene_id]; cand = pg["cand"]; pos = pg["pos"]
        phi = h1.features_for(b, r.pathogenicity_prob, r.isotonic_entropy); u = scorer.utilities(phi); ends = b["endpoints"]
        def endpoint_mass(probs):
            d = core.aggregate_distribution(ends, probs); return np.array([d.get(s, 0.0) for s in cand])
        scores = {"full": endpoint_mass(core.exact_path_probabilities(u, r.isotonic_entropy)),
                  "fixed_temperature": endpoint_mass(core.exact_path_probabilities(u, r.isotonic_entropy, fixed_temperature=True)),
                  "uniform_paths": endpoint_mass(np.ones(len(u)) / len(u)),
                  "ppr": np.array([ppr[r.gene_id].get(s, 0.0) for s in cand]),
                  "distmult": dm_cache[r.gene_id]}
        top = np.argsort(-u)[:5]; scores["beam_5"] = endpoint_mass(np.bincount(top, minlength=len(u)) / 5.0)
        for name, s in scores.items():
            m = rank_metrics(s, cand, pos)
            m.update(model=name, variant_id=r.variant_id, gene_id=r.gene_id, gene_symbol=r.gene_symbol, isotonic_entropy=r.isotonic_entropy,
                     pathogenicity_prob=r.pathogenicity_prob, n_candidates=len(cand), n_positives=len(pos), mass_on_positives=float(s[[cand.index(p) for p in pos]].sum()) if name not in ("distmult",) else np.nan)
            rows.append(m)
    res = pd.DataFrame(rows); res.to_csv(TABLES / "hetionet_linkpred.csv", index=False)

    # ---- summaries: per model (variant-level mean, gene-cluster CI) and by H stratum
    def cluster_mean_ci(vals, cl):
        vals = np.asarray(vals, float); cl = np.asarray(cl); u = np.unique(cl); by = {c: vals[cl == c] for c in u}; b = []
        for _ in range(1000):
            b.append(np.concatenate([by[c] for c in RNG.choice(u, len(u), replace=True)]).mean())
        return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    summ = []
    for name, g in res.groupby("model"):
        row = dict(model=name, n=len(g), n_genes=g.gene_id.nunique())
        for met in ["recall@10", "recall@25", "mrr"]:
            m, lo, hi = cluster_mean_ci(g[met], g.gene_id); row[met] = m; row[f"{met}_lo"] = lo; row[f"{met}_hi"] = hi
        row["rho_H_recall10"] = float(spearmanr(g.isotonic_entropy, g["recall@10"]).correlation)
        row["rho_H_mrr"] = float(spearmanr(g.isotonic_entropy, g["mrr"]).correlation)
        summ.append(row)
    summ = pd.DataFrame(summ); summ.to_csv(TABLES / "hetionet_linkpred_summary.csv", index=False)
    strat = res.assign(stratum=pd.qcut(res.isotonic_entropy, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")).groupby(["model", "stratum"], observed=False).agg(
        recall10=("recall@10", "mean"), recall25=("recall@25", "mean"), mrr=("mrr", "mean"), n=("variant_id", "size")).reset_index()
    strat.to_csv(TABLES / "hetionet_linkpred_by_stratum.csv", index=False)
    # paired comparison full vs fixed theta (gene-cluster bootstrap of the paired difference)
    piv = res.pivot(index="variant_id", columns="model", values="recall@10"); cl = res.drop_duplicates("variant_id").set_index("variant_id").loc[piv.index, "gene_id"].to_numpy()
    paired = {}
    for other in ["fixed_temperature", "uniform_paths", "ppr", "beam_5", "distmult"]:
        d = (piv["full"] - piv[other]).to_numpy(); m, lo, hi = cluster_mean_ci(d, cl); paired[f"full_minus_{other}"] = dict(mean=m, lo=lo, hi=hi)
    out = dict(held_out_edges=len(held), total_dps=len(dps), n_eval_genes=len(eval_genes), n_eval_variants=int(res.variant_id.nunique()), summary=summ.to_dict("records"),
               by_stratum=strat.to_dict("records"), paired_recall10=paired, runtime_s=time.time() - t0)
    (OUT / "summary_12b.json").write_text(json.dumps(out, indent=2, default=str)); (TABLES / "hetionet_linkpred_results.json").write_text(json.dumps(out, indent=2, default=str))
    pd.set_option("display.width", 220); print(summ.round(3).to_string(index=False)); print(strat.round(3).to_string(index=False)); print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main()
