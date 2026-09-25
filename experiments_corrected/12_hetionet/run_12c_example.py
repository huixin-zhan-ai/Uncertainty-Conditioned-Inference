#!/usr/bin/env python3
"""H2 illustration -- one concrete masked-edge recovery example (no new metrics).

Reproduces the H2 hold-out exactly (same seed, same order of operations as
run_12b.py), retrains the scorer on G', and extracts ONE worked example for the
evaluation-illustration figure:
  * a study gene g and one of its variants (selection rule below),
  * a hidden disease->symptom edge (d, s) whose symptom s is still reachable on G',
  * every candidate path on G' that ends at s (the alternative routes),
  * the filtered ranking of candidates under full theta(H), fixed theta and DistMult,
    with the rank of s and its endpoint mass.
Selection: per gene the VUS whose H is closest to the median H of all study
variants. The illustrated pair is fixed by name (EXAMPLE_GENE / EXAMPLE_SYMPTOM):
APP -> Aphasia, chosen because gene and hidden symptom are both
Alzheimer's-relevant and the hidden edge is recoverable through several gene
relays. It is an illustration, not a representative case: the ranks of EVERY
(gene, held-out positive) pair at these variants are written to
hetionet_linkpred_example_all_pairs.csv, and the aggregate numbers are Table 2.
Note (structural): under the filtered protocol a held-out symptom that is still
linked to ANOTHER disease of the same gene is a known positive and is filtered
out, so every recoverable held-out positive is reached only through gene relays
g -> g' -> d' -> s (the all-pairs table has n_direct_routes == 0 throughout).
Writes tables/hetionet_linkpred_example.json only.
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

core = hc.core
ROOT = hc.ROOT; TABLES = hc.TABLES
MANIFESTS = ROOT / "outputs" / "corrected" / "manifests"
EXAMPLE_GENE = "APP"; EXAMPLE_SYMPTOM = "Aphasia"


def main():
    t0 = time.time()
    RNG = np.random.default_rng(core.RNG_SEED)  # identical stream to run_12b
    G = hc.build_graph(); hc.reset_caches()
    variants = hc.study_variants(G)
    dps = [(u, v) for u, v, a in G.edges(data=True) if a["relation"] == "DISEASEMANIFESTSSYMPTOM"]
    idx = RNG.permutation(len(dps)); held = {dps[i] for i in idx[: int(h2.HOLDOUT * len(dps))]}
    Gp = G.copy(); Gp.remove_edges_from(held)
    assert len(held) == 671, len(held)
    rf = core.relation_frequencies(Gp)
    bases = {g: h1.base_features(Gp, g, rf) for g in variants.gene_id.unique()}
    bases = {g: b for g, b in bases.items() if b is not None}
    valid = variants[variants.gene_id.isin(bases)].reset_index(drop=True)
    split = json.loads((MANIFESTS / "locked_split.json").read_text())
    def_ids = [i for i in map(str, split["definitive_ids"]) if i in set(valid.variant_id)]
    samples = core.build_training_samples(Gp, valid, rf, def_ids)
    sc_d = core.train_path_scorer(samples, l2=1e-2, seed=core.RNG_SEED); sc_i = h1.train_informed(samples)
    scorer = sc_d if h1.objective_value(sc_d, samples) <= h1.objective_value(sc_i, samples) else sc_i
    print(f"G' rebuilt, scorer objective {h1.objective_value(scorer, samples):.4f}  {time.time()-t0:.0f}s", flush=True)

    disease_of = {g: [d for d in G.successors(g) if G.nodes[d]["type"] == "Disease"] for g in bases}
    per_gene = {}
    for g, b in bases.items():
        ends = set(b["endpoints"])
        known = {s for d in disease_of[g] for s in Gp.successors(d) if Gp.nodes[s]["type"] == "Phenotype"}
        heldpos = {(dd, s) for d in disease_of[g] for (dd, s) in held if dd == d}
        cand = sorted(ends - known)
        pos = sorted({s for _, s in heldpos} & set(cand))
        per_gene[g] = dict(cand=cand, pos=pos, hidden=sorted(heldpos))
    eval_genes = [g for g, v in per_gene.items() if v["pos"] and len(v["cand"]) > len(v["pos"])]

    # ---- variant per gene: VUS with H closest to the global median H
    medH = float(valid.isotonic_entropy.median())
    name = lambda n: G.nodes[n]["name"]

    def mass_full(g, r):
        b = bases[g]; phi = h1.features_for(b, r.pathogenicity_prob, r.isotonic_entropy); u = scorer.utilities(phi)
        return b, u, core.aggregate_distribution(b["endpoints"], core.exact_path_probabilities(u, r.isotonic_entropy))

    # ---- every (gene, held-out positive) pair at the per-gene median-H VUS; then the named example
    rows = []; chosen = None
    for g in eval_genes:
        vg = valid[(valid.gene_id == g) & (valid.clinical_significance.str.upper().str.contains("UNCERTAIN"))]
        if vg.empty:
            continue
        r = vg.iloc[(vg.isotonic_entropy - medH).abs().argsort().iloc[0]]
        b, u, m = mass_full(g, r); cand = per_gene[g]["cand"]
        sc_ = np.array([m.get(c, 0.0) for c in cand]); ranked = [cand[i] for i in np.argsort(-sc_, kind="stable")]
        for s in per_gene[g]["pos"]:
            routes = [c for c in b["cands"] if c[-1] == s]
            rows.append(dict(gene=name(g), variant_id=str(r.variant_id), H=float(r.isotonic_entropy), symptom=name(s), hidden_from=";".join(name(d) for d, ss in per_gene[g]["hidden"] if ss == s),
                             rank_full=ranked.index(s) + 1, n_candidates=len(cand), n_direct_routes=sum(len(c) == 3 for c in routes), n_relay_routes=sum(len(c) == 4 for c in routes), mass_full=float(m.get(s, 0.0))))
            if name(g) == EXAMPLE_GENE and name(s) == EXAMPLE_SYMPTOM:
                chosen = (g, r, s, routes)
    allp = pd.DataFrame(rows); allp.to_csv(TABLES / "hetionet_linkpred_example_all_pairs.csv", index=False)
    assert (allp.n_direct_routes == 0).all(), "filtered protocol should leave relay routes only"
    print(f"{len(allp)} (gene, held-out positive) pairs; recovered in top-10 by full: {(allp.rank_full <= 10).sum()}", flush=True)
    g, r, s, routes = chosen
    b, u, m = mass_full(g, r)
    hidden_d = [d for (d, ss) in per_gene[g]["hidden"] if ss == s]
    print(f"selected {name(g)} / {name(s)} : {len(routes)} routes, mass {m.get(s, 0.0):.5f}, hidden from {[name(d) for d in hidden_d]}", flush=True)

    # ---- rankings: full, fixed theta, DistMult (same construction as run_12b)
    cand = per_gene[g]["cand"]; pos = per_gene[g]["pos"]
    m_fix = core.aggregate_distribution(b["endpoints"], core.exact_path_probabilities(u, r.isotonic_entropy, fixed_temperature=True))
    assoc_genes = {u_ for u_, v_, a in Gp.edges(data=True) if a["relation"] == "GENEASSOCIATESWITHDISEASE"} | set(bases)
    ents = {}
    def eid(x):
        if x not in ents:
            ents[x] = len(ents)
        return ents[x]
    triples = []
    for u_, v_, a in Gp.edges(data=True):
        rel = a["relation"]
        if rel == "GENEASSOCIATESWITHDISEASE":
            triples.append((eid(u_), 0, eid(v_)))
        elif rel == "DISEASEMANIFESTSSYMPTOM":
            triples.append((eid(u_), 1, eid(v_)))
        elif rel in ("GENEINTERACTSWITHGENE", "GENEREGULATESGENE", "GENECOVARIESWITHGENE") and (u_ in bases or v_ in bases) and u_ in assoc_genes and v_ in assoc_genes:
            triples.append((eid(u_), 2, eid(v_)))
    E, R = h2.train_distmult(np.array(triples), len(ents), 3)
    diseases_all = [d for d, a in Gp.nodes(data=True) if a["type"] == "Disease" and d in ents]
    pd_ = np.array([expit(h2.distmult_score(E, R, ents[g], 0, ents[d])) for d in diseases_all])
    dm = np.zeros(len(cand))
    for j, c in enumerate(cand):
        if c in ents:
            dm[j] = float(np.max(pd_ * np.array([expit(h2.distmult_score(E, R, ents[d], 1, ents[c])) for d in diseases_all])))
    print(f"DistMult done {time.time()-t0:.0f}s", flush=True)

    def ranking(score):
        order = np.argsort(-np.asarray(score), kind="stable")
        return [cand[i] for i in order], [float(score[i]) for i in order]
    out = {}
    for nm, sc in [("full", [m.get(c, 0.0) for c in cand]), ("fixed_temperature", [m_fix.get(c, 0.0) for c in cand]), ("distmult", dm)]:
        ranked, vals = ranking(sc)
        out[nm] = dict(rank_of_target=ranked.index(s) + 1, score_of_target=vals[ranked.index(s)],
                       top10=[dict(symptom=name(c), score=v, held_out_positive=(c in pos)) for c, v in zip(ranked[:10], vals[:10])])
    # alternative routes with path probability under full
    pf = core.exact_path_probabilities(u, r.isotonic_entropy)
    alt = []
    for c, p_ in zip(b["cands"], pf):
        if c[-1] == s:
            alt.append(dict(path=[name(x) for x in c], types=[G.nodes[x]["type"] for x in c], prob=float(p_)))
    alt.sort(key=lambda z: -z["prob"])
    # the hidden edge's own path(s) would be g -> d -> s on the full graph; record whether d still links to g on G'
    ex = dict(gene=name(g), gene_id=g, variant_id=str(r.variant_id), clinvar_class=str(r.clinical_significance), p=float(r.pathogenicity_prob), H=float(r.isotonic_entropy),
              theta=float(core.theta(r.isotonic_entropy)), target_symptom=name(s), hidden_from_diseases=[name(d) for d in hidden_d],
              n_candidates_filtered=len(cand), n_held_out_positives=len(pos), n_candidate_paths=len(b["cands"]), n_alternative_routes=len(alt),
              alternative_routes=alt, rankings=out, selection_rule="per gene: VUS with H nearest global median H; illustrated pair fixed by name (APP, Aphasia); all pairs in hetionet_linkpred_example_all_pairs.csv",
              all_pairs_n=int(len(allp)), all_pairs_top10=int((allp.rank_full <= 10).sum()), this_gene_positive_ranks=sorted(int(x) for x in allp[allp.gene == EXAMPLE_GENE].rank_full),
              held_out_edges=len(held))
    (TABLES / "hetionet_linkpred_example.json").write_text(json.dumps(ex, indent=2))
    print(json.dumps({k: v for k, v in ex.items() if k not in ("alternative_routes",)}, indent=1)); print(f"{len(alt)} alternative routes; top 5:")
    for z in alt[:5]:
        print(f"  {z['prob']:.4f}  " + " -> ".join(z["path"]))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
