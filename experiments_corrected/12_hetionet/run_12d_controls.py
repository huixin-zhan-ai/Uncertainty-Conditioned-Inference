#!/usr/bin/env python3
"""H2-B -- Matched-control recovery on the H2 hold-out (C3 panel B).

Same hold-out as run_12b.py (seed 42, 671 hidden disease->symptom edges), same
filtered candidates and held-out positives per gene, same full-model endpoint
probabilities. Four conditions are compared on IDENTICAL candidates:

  full            actual endpoint mass m(s) from the full model (theta(H))
  random          chance retrieval over exactly those candidates (analytic
                  expectation: recall@k = min(k, n)/n, MRR = H_n / n)
  popularity      rank by symptom in-degree (DISEASEMANIFESTSSYMPTOM edges) in
                  the reduced training graph G'; random tie-break averaged over
                  N_REPS draws
  degree_perm     degree-matched permutation of the full model's endpoint
                  probabilities: candidates of a gene are binned by training-
                  graph degree (quantile bins of >= MIN_BIN candidates) and the
                  probability values are shuffled WITHIN bins; the same
                  permutation is applied to every variant of the gene; N_REPS
                  permutations give a reference distribution.

Prespecified analysis (fixed before looking at results):
  * primary metric recall@10; recall@5 and recall@25 complementary; MRR reported
  * paired difference full - control per variant, gene-cluster 95% bootstrap CI
  * permutation p-value for degree_perm: fraction of permutations whose
    gene-balanced mean recall@10 >= the full model's
Protocol checks written alongside: leakage audit of the masked edges and the
coverage of held-out positives (reachable fraction; unconditional recall).
Outputs: tables/hetionet_controls*.csv/json  (new files only).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hetionet_common as hc  # noqa: E402
import run_12a as h1  # noqa: E402
import run_12b as h2  # noqa: E402

core = hc.core
ROOT = hc.ROOT; TABLES = hc.TABLES
MANIFESTS = ROOT / "outputs" / "corrected" / "manifests"
N_REPS = 200; MIN_BIN = 8; KS = (5, 10, 25); PRIMARY = "recall@10"


def rank_metrics(scores, cand, pos):
    order = np.argsort(-np.asarray(scores, float), kind="stable"); ranked = [cand[i] for i in order]; posset = set(pos)
    ranks = [i + 1 for i, s in enumerate(ranked) if s in posset]
    out = {f"recall@{k}": float(sum(r <= k for r in ranks) / len(pos)) for k in KS}
    out["mrr"] = float(np.mean([1.0 / r for r in ranks])); return out


def random_expectation(n, m):
    out = {f"recall@{k}": float(min(k, n) / n) for k in KS}
    out["mrr"] = float(np.sum(1.0 / np.arange(1, n + 1)) / n); return out


def cluster_mean_ci(vals, cl, rng, B=1000):
    vals = np.asarray(vals, float); cl = np.asarray(cl); u = np.unique(cl); by = {c: vals[cl == c] for c in u}; b = []
    for _ in range(B):
        b.append(np.concatenate([by[c] for c in rng.choice(u, len(u), replace=True)]).mean())
    return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def degree_bins(degs, min_bin=MIN_BIN):
    """Quantile bins over candidate degrees with at least min_bin members each; returns bin id per candidate."""
    n = len(degs); nb = max(1, n // min_bin); order = np.argsort(np.asarray(degs, float), kind="stable")
    bins = np.zeros(n, int)
    for j, chunk in enumerate(np.array_split(order, nb)):
        bins[chunk] = j
    return bins


def main():
    t0 = time.time(); RNG = np.random.default_rng(core.RNG_SEED)
    G = hc.build_graph(); hc.reset_caches(); variants = hc.study_variants(G)
    dps = [(u, v) for u, v, a in G.edges(data=True) if a["relation"] == "DISEASEMANIFESTSSYMPTOM"]
    idx = RNG.permutation(len(dps)); held = {dps[i] for i in idx[: int(h2.HOLDOUT * len(dps))]}
    Gp = G.copy(); Gp.remove_edges_from(held); assert len(held) == 671
    rf = core.relation_frequencies(Gp)
    bases = {g: h1.base_features(Gp, g, rf) for g in variants.gene_id.unique()}; bases = {g: b for g, b in bases.items() if b is not None}
    valid = variants[variants.gene_id.isin(bases)].reset_index(drop=True)
    split = json.loads((MANIFESTS / "locked_split.json").read_text()); def_ids = [i for i in map(str, split["definitive_ids"]) if i in set(valid.variant_id)]
    samples = core.build_training_samples(Gp, valid, rf, def_ids)
    sc_d = core.train_path_scorer(samples, l2=1e-2, seed=core.RNG_SEED); sc_i = h1.train_informed(samples)
    scorer = sc_d if h1.objective_value(sc_d, samples) <= h1.objective_value(sc_i, samples) else sc_i
    print(f"G' rebuilt {time.time()-t0:.0f}s", flush=True)

    # ---------------- protocol check 1: leakage audit
    leak = dict(n_masked=len(held), forward_present=0, inverse_present=0, other_relation_between_pair=0, in_any_candidate_path=0, in_training_graph_relation_counts=int(rf.get("DISEASEMANIFESTSSYMPTOM", 0)),
                total_dps_original=len(dps), caches_reset=True)
    for d, s in held:
        if Gp.has_edge(d, s):
            leak["forward_present"] += 1
        if Gp.has_edge(s, d):
            leak["inverse_present"] += 1
    heldset = set(held)
    for g, b in bases.items():
        for path in b["cands"]:
            if any((path[i], path[i + 1]) in heldset for i in range(len(path) - 1)):
                leak["in_any_candidate_path"] += 1
    leak["relation_count_check"] = (len(dps) - len(held) == leak["in_training_graph_relation_counts"])
    leak["pass"] = leak["forward_present"] == 0 and leak["inverse_present"] == 0 and leak["in_any_candidate_path"] == 0 and leak["relation_count_check"]
    (TABLES / "hetionet_leak_check.json").write_text(json.dumps(leak, indent=2)); print("leak check:", leak, flush=True)

    # ---------------- candidates / positives / coverage
    disease_of = {g: [d for d in G.successors(g) if G.nodes[d]["type"] == "Disease"] for g in bases}
    per_gene = {}; cov_rows = []
    for g, b in bases.items():
        ends = set(b["endpoints"]); known = {s for d in disease_of[g] for s in Gp.successors(d) if Gp.nodes[s]["type"] == "Phenotype"}
        heldpos_all = {s for d in disease_of[g] for (dd, s) in held if dd == d}
        cand = sorted(ends - known); pos = sorted(heldpos_all & set(cand))
        per_gene[g] = dict(cand=cand, pos=pos, n_all=len(heldpos_all))
        cov_rows.append(dict(gene_symbol=valid[valid.gene_id == g].gene_symbol.iloc[0], n_heldout_symptoms=len(heldpos_all), n_reachable=len(pos), n_candidates=len(cand)))
    cov = pd.DataFrame(cov_rows); cov["reachable_fraction"] = cov.n_reachable / cov.n_heldout_symptoms.replace(0, np.nan)
    cov.to_csv(TABLES / "hetionet_coverage.csv", index=False)
    eval_genes = [g for g, v in per_gene.items() if v["pos"] and len(v["cand"]) > len(v["pos"])]
    # symptom degree in G' (training graph)
    deg = {}
    for u, v, a in Gp.edges(data=True):
        if a["relation"] == "DISEASEMANIFESTSSYMPTOM":
            deg[v] = deg.get(v, 0) + 1

    # ---------------- per-gene permutation structures (same permutation for all variants of a gene)
    gene_perms = {}
    for g in eval_genes:
        cand = per_gene[g]["cand"]; degs = np.array([deg.get(c, 0) for c in cand]); bins = degree_bins(degs)
        perms = []
        for r in range(N_REPS):
            p = np.arange(len(cand))
            for bidx in np.unique(bins):
                members = np.where(bins == bidx)[0]; p[members] = RNG.permutation(members)
            perms.append(p)
        pop_ties = [RNG.random(len(cand)) * 1e-6 for _ in range(N_REPS)]
        gene_perms[g] = dict(perms=perms, degs=degs, bins=bins, pop_ties=pop_ties, n_bins=int(len(np.unique(bins))))

    # ---------------- evaluate
    rows = []; perm_rows = []
    for r in valid.itertuples(index=False):
        if r.gene_id not in eval_genes:
            continue
        b = bases[r.gene_id]; pg = per_gene[r.gene_id]; cand = pg["cand"]; pos = pg["pos"]; gp = gene_perms[r.gene_id]
        phi = h1.features_for(b, r.pathogenicity_prob, r.isotonic_entropy); u = scorer.utilities(phi)
        dist = core.aggregate_distribution(b["endpoints"], core.exact_path_probabilities(u, r.isotonic_entropy))
        full = np.array([dist.get(c, 0.0) for c in cand])
        base = dict(variant_id=str(r.variant_id), gene_id=r.gene_id, gene_symbol=r.gene_symbol, isotonic_entropy=float(r.isotonic_entropy), n_candidates=len(cand), n_positives=len(pos))
        m_full = rank_metrics(full, cand, pos); rows.append(dict(base, model="full", **m_full))
        rows.append(dict(base, model="random", **random_expectation(len(cand), len(pos))))
        pop = [rank_metrics(gp["degs"] + tie, cand, pos) for tie in gp["pop_ties"]]
        rows.append(dict(base, model="popularity", **{k: float(np.mean([m[k] for m in pop])) for k in pop[0]}))
        dp = [rank_metrics(full[p], cand, pos) for p in gp["perms"]]
        rows.append(dict(base, model="degree_perm", **{k: float(np.mean([m[k] for m in dp])) for k in dp[0]}))
        perm_rows.append(dict(variant_id=str(r.variant_id), gene_id=r.gene_id, full=m_full[PRIMARY], perms=[m[PRIMARY] for m in dp]))
    res = pd.DataFrame(rows); res.to_csv(TABLES / "hetionet_controls.csv", index=False)
    print(f"evaluated {res.variant_id.nunique()} variants, {res.gene_id.nunique()} genes  {time.time()-t0:.0f}s", flush=True)

    # ---------------- summaries: variant-level mean with cluster CI; gene-balanced mean
    summ = []
    for name, g in res.groupby("model"):
        row = dict(model=name, n=len(g), n_genes=g.gene_id.nunique())
        for met in [f"recall@{k}" for k in KS] + ["mrr"]:
            m, lo, hi = cluster_mean_ci(g[met], g.gene_id, RNG); row[met] = m; row[f"{met}_lo"] = lo; row[f"{met}_hi"] = hi
            row[f"{met}_gene_balanced"] = float(g.groupby("gene_id")[met].mean().mean())
        summ.append(row)
    summ = pd.DataFrame(summ); summ.to_csv(TABLES / "hetionet_controls_summary.csv", index=False)
    paired = {}
    piv = res.pivot(index="variant_id", columns="model", values=PRIMARY); cl = res.drop_duplicates("variant_id").set_index("variant_id").loc[piv.index, "gene_id"].to_numpy()
    for other in ["random", "popularity", "degree_perm"]:
        for met in [f"recall@{k}" for k in KS]:
            pv = res.pivot(index="variant_id", columns="model", values=met).loc[piv.index]
            m, lo, hi = cluster_mean_ci((pv["full"] - pv[other]).to_numpy(), cl, RNG); paired[f"full_minus_{other}|{met}"] = dict(mean=m, lo=lo, hi=hi)
    # permutation p-value on gene-balanced mean recall@10
    pr = pd.DataFrame(perm_rows); full_gb = pr.groupby("gene_id").full.mean().mean()
    perm_gb = np.array([pr.assign(v=[p[i] for p in pr.perms]).groupby("gene_id").v.mean().mean() for i in range(N_REPS)])
    pval = float((perm_gb >= full_gb).mean()); pval = max(pval, 1.0 / (N_REPS + 1))
    out = dict(n_eval_variants=int(res.variant_id.nunique()), n_eval_genes=int(res.gene_id.nunique()), n_reps=N_REPS, min_bin=MIN_BIN, primary=PRIMARY,
               bins_per_gene={valid[valid.gene_id == g].gene_symbol.iloc[0]: gene_perms[g]["n_bins"] for g in eval_genes},
               summary=summ.to_dict("records"), paired=paired,
               degree_perm_reference=dict(full_gene_balanced=float(full_gb), perm_mean=float(perm_gb.mean()), perm_p975=float(np.percentile(perm_gb, 97.5)), perm_max=float(perm_gb.max()), p_value_upper_bound=pval),
               coverage=dict(n_heldout_symptoms=int(cov.n_heldout_symptoms.sum()), n_reachable=int(cov.n_reachable.sum()), reachable_fraction=float(cov.n_reachable.sum() / cov.n_heldout_symptoms.sum()),
                             eval_gene_reachable_fraction=float(cov[cov.gene_symbol.isin(valid[valid.gene_id.isin(eval_genes)].gene_symbol.unique())].n_reachable.sum() / cov[cov.gene_symbol.isin(valid[valid.gene_id.isin(eval_genes)].gene_symbol.unique())].n_heldout_symptoms.sum())),
               leak_check=leak, runtime_s=time.time() - t0)
    # unconditional recall: conditional x per-gene reachable fraction, variant-weighted
    covg = cov.set_index("gene_symbol").reachable_fraction
    for met in [f"recall@{k}" for k in KS]:
        fullr = res[res.model == "full"]; uncond = (fullr[met] * fullr.gene_symbol.map(covg)).mean(); out[f"unconditional_{met}_full"] = float(uncond)
    (TABLES / "hetionet_controls_results.json").write_text(json.dumps(out, indent=2, default=str))
    pd.set_option("display.width", 220); print(summ[["model", "recall@5", "recall@10", "recall@10_lo", "recall@10_hi", "recall@25", "mrr", "recall@10_gene_balanced"]].round(3).to_string(index=False))
    print(json.dumps(paired, indent=1)); print("degree_perm reference:", out["degree_perm_reference"]); print("coverage:", out["coverage"]); print({k: v for k, v in out.items() if k.startswith("unconditional")})
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
