#!/usr/bin/env python3
"""H1 -- Second real graph (Hetionet v1.0), same upstream signal, same method.

Everything below uses corrected_core unchanged on a Hetionet graph expressed
in the AlzKB schema (hetionet_common.build_graph). The 985 Evo2/ClinVar
variants keep their frozen (p, H_iso, H_knn) and labels; only the graph changes.

Stages: graph summary and per-gene support audit; scorer training on the same
121 definitive anchors (with collapse check and informed-init fallback, both
reported); exact-target distributions for full / fixed-theta / uniform /
PPR / beam; sampler fidelity; gene-cluster CIs; single-split downstream
diagnostic. Outputs are NEW tables prefixed hetionet_*.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hetionet_common as hc  # noqa: E402
core = hc.core
ROOT = hc.ROOT; TABLES = hc.TABLES
OUT = HERE / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
MANIFESTS = ROOT / "outputs" / "corrected" / "manifests"
BOOT = 1000
RNG = np.random.default_rng(core.RNG_SEED)


# ----------------------------------------------------------------- helpers
def base_features(graph, gene, rf):
    """Per-gene base feature matrix (p=0, H=0); interaction columns filled per variant."""
    cands = core.enumerate_candidate_paths(graph, gene)
    if not cands:
        return None
    phi0, groups = core.feature_matrix(graph, cands, rf, 0.0, 0.0)
    return dict(cands=cands, phi0=phi0, groups=groups, endpoints=[c[-1] for c in cands])


def features_for(base, p, H):
    phi = base["phi0"].copy()
    ad_core, mech, spec = phi[:, 6], phi[:, 7], phi[:, 5]
    conf = 1.0 - float(np.clip(H, 0, 1))
    phi[:, 10] = p * ad_core; phi[:, 11] = p * mech; phi[:, 12] = conf * spec; phi[:, 13] = conf * mech
    return phi


def metrics(dist):
    pr = np.asarray(list(dist.values()), float)
    return dict(entropy_bits=core.entropy_from_prob(pr), effective=core.effective_support(pr), top1=float(pr.max()) if len(pr) else np.nan)


def cluster_rho(x, y, cl):
    x = np.asarray(x, float); y = np.asarray(y, float); cl = np.asarray(cl); u = np.unique(cl); by = {c: np.where(cl == c)[0] for c in u}; r = []
    for _ in range(BOOT):
        idx = np.concatenate([by[c] for c in RNG.choice(u, len(u), replace=True)]); v = spearmanr(x[idx], y[idx]).correlation
        if np.isfinite(v):
            r.append(v)
    return float(spearmanr(x, y).correlation), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def objective_value(scorer, samples, l2=1e-2, readout_l2=1e-3):
    loss = 0.0
    for s in samples:
        u = scorer.utilities(s["features"]); z = (u - u.max()) / s["theta"]; pr = np.exp(z); pr /= pr.sum()
        m = float(s["ad_core"] @ pr); pred = float(np.clip(expit(scorer.alpha * m + scorer.beta), 1e-9, 1 - 1e-9))
        loss += -(s["label"] * np.log(pred) + (1 - s["label"]) * np.log(1 - pred))
    return loss / len(samples) + 0.5 * l2 * float(scorer.weights @ scorer.weights) + 0.5 * readout_l2 * scorer.alpha ** 2


def train_informed(samples, l2=1e-2, seed=core.RNG_SEED):
    """Verbatim objective of core.train_path_scorer with the first-order informed initialization (E4d)."""
    pooled = np.vstack([s["features"] for s in samples]); mean = pooled.mean(0); std = pooled.std(0); std[std < 1e-8] = 1.0
    prep = [{**s, "features": (s["features"] - mean) / std} for s in samples]; dim = core.N_FEATURES; rl2 = 1e-3

    def obj(t):
        w = t[:dim]; a, b = t[dim:]; loss = 0.0; gw = np.zeros(dim); ga = gb = 0.0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for s in prep:
                phi = s["features"]; c = s["ad_core"]; tau = s["theta"]; y = s["label"]; u = phi @ w
                if not np.all(np.isfinite(u)):
                    return 1e100, np.zeros_like(t)
                pr = np.exp((u - u.max()) / tau); pr /= pr.sum(); m = float(c @ pr)
                pred = float(np.clip(expit(a * m + b), 1e-9, 1 - 1e-9)); loss += -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
                gz = pred - y; ga += gz * m; gb += gz; gw += phi.T @ (gz * a * pr * (c - m) / tau)
        n = len(prep)
        return loss / n + 0.5 * l2 * float(w @ w) + 0.5 * rl2 * a * a, np.concatenate([gw / n + l2 * w, [ga / n + rl2 * a, gb / n]])

    ybar = np.mean([s["label"] for s in prep]); d = np.zeros(dim)
    for s in prep:
        c = s["ad_core"].astype(bool)
        if c.any() and (~c).any():
            d += (s["label"] - ybar) * (s["features"][c].mean(0) - s["features"][~c].mean(0))
    w0 = 0.5 * d / np.linalg.norm(d) if np.linalg.norm(d) > 0 else np.random.default_rng(seed).normal(0, 0.01, dim)
    res = minimize(obj, np.concatenate([w0, [1.0, 0.0]]), jac=True, method="L-BFGS-B",
                   bounds=[(-5.0, 5.0)] * dim + [(-20.0, 20.0), (-20.0, 20.0)], options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-12, "maxls": 50})
    return core.LearnedPathScorer(weights=res.x[:dim], feature_mean=mean, feature_std=std, alpha=float(res.x[dim]), beta=float(res.x[dim + 1]),
                                  l2=l2, n_train=len(prep), training_ids=[s["variant_id"] for s in prep])


def ppr_endpoint_mass(graph, gene, endpoints, alpha=0.85):
    sc = nx.pagerank(graph, alpha=alpha, personalization={gene: 1.0}, max_iter=200, tol=1e-10)
    m = np.array([sc.get(e, 0.0) for e in endpoints]); m = m if m.sum() > 0 else np.ones(len(endpoints))
    return m / m.sum()


# ----------------------------------------------------------------- main
def main():
    t0 = time.time()
    graph = hc.build_graph()
    summary = hc.graph_summary(graph); print("graph:", summary, f"{time.time()-t0:.0f}s", flush=True)
    hc.reset_caches()
    rf = core.relation_frequencies(graph)
    variants = hc.study_variants(graph)

    # support audit
    bases = {}; audit = []
    for gene, g in variants.groupby("gene_id"):
        b = base_features(graph, gene, rf)
        if b is None:
            audit.append(dict(gene_id=gene, gene_symbol=g.gene_symbol.iloc[0], n_variants=len(g), candidate_paths=0, candidate_routes=0, reachable_endpoints=0, direct_diseases=0)); continue
        bases[gene] = b
        routes = {tuple(c[:-2]) for c in b["cands"]}; endpoints = set(b["endpoints"])
        direct = [v for v in graph.successors(gene) if graph.nodes[v]["type"] == "Disease"]
        audit.append(dict(gene_id=gene, gene_symbol=g.gene_symbol.iloc[0], n_variants=len(g), candidate_paths=len(b["cands"]), candidate_routes=len(routes),
                          reachable_endpoints=len(endpoints), direct_diseases=len(direct), mean_gene_hops=float(np.mean([len(c) - 3 for c in b["cands"]]))))
    audit = pd.DataFrame(audit).sort_values("n_variants", ascending=False); audit.to_csv(TABLES / "hetionet_path_audit.csv", index=False)
    print(audit.to_string(index=False), flush=True)
    valid = variants[variants.gene_id.isin(bases)].reset_index(drop=True)
    print(f"valid variants {len(valid)} in {valid.gene_id.nunique()} genes  {time.time()-t0:.0f}s", flush=True)

    # training on the same 121 definitive anchors
    split = json.loads((MANIFESTS / "locked_split.json").read_text())
    def_ids = [i for i in map(str, split["definitive_ids"]) if i in set(valid.variant_id)]
    samples = core.build_training_samples(graph, valid, rf, def_ids)
    ybar = np.mean([s["label"] for s in samples]); const_loss = -(ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar))
    sc_default = core.train_path_scorer(samples, l2=1e-2, seed=core.RNG_SEED); obj_default = objective_value(sc_default, samples)
    sc_informed = train_informed(samples); obj_informed = objective_value(sc_informed, samples)
    scorer = sc_default if obj_default <= obj_informed else sc_informed
    train_info = dict(n_train=len(samples), constant_predictor_loss=const_loss, objective_default=obj_default, objective_informed=obj_informed,
                      selected="default" if scorer is sc_default else "informed", weight_cosine_default_vs_informed=float(sc_default.weights @ sc_informed.weights / (np.linalg.norm(sc_default.weights) * np.linalg.norm(sc_informed.weights))))
    # anchor AUROC of the selected scorer
    preds, ys = [], []
    for s in samples:
        u = scorer.utilities(s["features"]); pr = np.exp((u - u.max()) / s["theta"]); pr /= pr.sum(); preds.append(expit(scorer.alpha * float(s["ad_core"] @ pr) + scorer.beta)); ys.append(s["label"])
    train_info["anchor_auroc"] = float(roc_auc_score(ys, preds))
    scorer.save(ROOT / "outputs" / "corrected" / "models" / "hetionet_path_scorer.json")
    print("training:", train_info, flush=True)

    # per-gene PPR (non-trivial now)
    ppr = {}
    for gene, b in bases.items():
        ep = sorted(set(b["endpoints"])); ppr[gene] = dict(zip(ep, ppr_endpoint_mass(graph, gene, ep)))
    print(f"PPR done {time.time()-t0:.0f}s", flush=True)

    # distributions per variant
    rows = []; fid = []
    for i, r in enumerate(valid.itertuples(index=False)):
        b = bases[r.gene_id]; phi = features_for(b, r.pathogenicity_prob, r.isotonic_entropy); u = scorer.utilities(phi)
        ends = b["endpoints"]; groups = b["groups"]
        full_p = core.exact_path_probabilities(u, r.isotonic_entropy)
        full_e = core.aggregate_distribution(ends, full_p); full_g = core.aggregate_distribution(groups, full_p)
        models = {"full": full_e,
                  "fixed_temperature": core.aggregate_distribution(ends, core.exact_path_probabilities(u, r.isotonic_entropy, fixed_temperature=True)),
                  "uniform_paths": core.aggregate_distribution(ends, np.ones(len(ends)) / len(ends)),
                  "ppr": {e: ppr[r.gene_id][e] for e in set(ends)}}
        order = np.argsort(-u)
        for k in (5, 20):
            top = order[:k]; models[f"beam_{k}"] = core.aggregate_distribution([ends[j] for j in top], np.ones(len(top)) / len(top))
        for name, dist in models.items():
            m = metrics(dist); m.update(model=name, variant_id=r.variant_id, gene_id=r.gene_id, gene_symbol=r.gene_symbol, clinical_significance=r.clinical_significance,
                                        pathogenicity_prob=r.pathogenicity_prob, isotonic_entropy=r.isotonic_entropy, knn_entropy=r.knn_entropy,
                                        js_to_full=core.js_divergence(dist, full_e), grouped_entropy_bits=(core.entropy_from_prob(np.asarray(list(full_g.values()))) if name == "full" else np.nan),
                                        ad_core_mass=float(sum(full_g.get(gr, 0.0) for gr in core.AD_CORE_GROUPS)) if name == "full" else np.nan)
            rows.append(m)
        # sampler fidelity (grouped, as in the paper) on this larger support
        final, acc, nsteps = core.metropolis_sample(u, r.isotonic_entropy, seed=core.RNG_SEED + i)
        emp = core.aggregate_distribution([groups[j] for j in final], np.ones(len(final)) / len(final))
        emp_e = core.aggregate_distribution([ends[j] for j in final], np.ones(len(final)) / len(final))
        fid.append(dict(variant_id=r.variant_id, gene_id=r.gene_id, isotonic_entropy=r.isotonic_entropy, n_states=len(u), revision_steps=nsteps, acceptance_rate=acc,
                        js_grouped=core.js_divergence(emp, full_g), js_endpoint=core.js_divergence(emp_e, full_e), unique_final_paths=int(len(set(final.tolist())))))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(valid)} {time.time()-t0:.0f}s", flush=True)
    dist = pd.DataFrame(rows); dist.to_csv(TABLES / "hetionet_distributions.csv", index=False)
    fid = pd.DataFrame(fid); fid.to_csv(TABLES / "hetionet_sampler_fidelity.csv", index=False)

    # summaries with cluster CIs
    summ = []
    for name, g in dist.groupby("model"):
        rho, lo, hi = cluster_rho(g.isotonic_entropy, g.entropy_bits, g.gene_id)
        summ.append(dict(model=name, entropy_mean=g.entropy_bits.mean(), top1_mean=g.top1.mean(), effective_mean=g.effective.mean(), js_to_full_mean=g.js_to_full.mean(),
                         rho_H_entropy=rho, rho_ci_low=lo, rho_ci_high=hi, n=len(g)))
    summ = pd.DataFrame(summ); summ.to_csv(TABLES / "hetionet_summary.csv", index=False)
    full = dist[dist.model == "full"]
    extra = {}
    for nm, col in [("rho_H_grouped_entropy", "grouped_entropy_bits"), ("rho_H_top1", "top1")]:
        extra[nm] = dict(zip(["rho", "lo", "hi"], cluster_rho(full.isotonic_entropy, full[col], full.gene_id)))
    extra["rho_Hknn_entropy_full"] = dict(zip(["rho", "lo", "hi"], cluster_rho(full.knn_entropy, full.entropy_bits, full.gene_id)))
    extra["rho_H_unique_final_paths"] = dict(zip(["rho", "lo", "hi"], cluster_rho(fid.isotonic_entropy, fid.unique_final_paths, fid.gene_id)))
    extra["fidelity"] = dict(js_grouped_mean=float(fid.js_grouped.mean()), js_grouped_median=float(fid.js_grouped.median()), js_endpoint_mean=float(fid.js_endpoint.mean()),
                             acceptance_mean=float(fid.acceptance_rate.mean()), n_states_mean=float(fid.n_states.mean()), n_states_max=int(fid.n_states.max()))
    q = full.assign(stratum=pd.qcut(full.isotonic_entropy, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")).groupby("stratum", observed=False).agg(
        entropy_mean=("entropy_bits", "mean"), top1_mean=("top1", "mean"), n=("variant_id", "size")).reset_index()
    q.to_csv(TABLES / "hetionet_propagation_summary.csv", index=False)

    # single-split downstream diagnostic
    likely_ids = set(map(str, split["likely_validation_ids"])); base_v = valid.set_index("variant_id")
    y_all = base_v.clinical_significance.isin(core.DEFINITIVE_POS | core.LIKELY_POS).astype(int)
    diag_rows = []; seq_ref = None
    for name in ["sequence_only", "full", "fixed_temperature", "ppr", "beam_5"]:
        feats = base_v[["pathogenicity_prob", "isotonic_entropy"]].copy()
        if name != "sequence_only":
            d = dist[dist.model == name].set_index("variant_id")[["entropy_bits", "effective", "top1"]]
            if name == "full":
                d = d.join(dist[dist.model == "full"].set_index("variant_id")[["ad_core_mass"]])
            feats = feats.join(d, how="inner")
        tr = feats.index.isin(set(def_ids)); te = feats.index.isin(likely_ids)
        scaler = StandardScaler().fit(feats[tr]); clf = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(feats[tr]), y_all.loc[feats.index[tr]])
        prob = clf.predict_proba(scaler.transform(feats[te]))[:, 1]; y = y_all.loc[feats.index[te]].to_numpy(); cl = base_v.loc[feats.index[te], "gene_id"].to_numpy()
        auc = roc_auc_score(y, prob)
        u = np.unique(cl); by = {c: np.where(cl == c)[0] for c in u}; aucs = []; diffs = []
        ref = None if seq_ref is None else pd.Series(seq_ref[1], index=seq_ref[0]).loc[feats.index[te]].to_numpy()
        for _ in range(BOOT):
            idx = np.concatenate([by[c] for c in RNG.choice(u, len(u), replace=True)])
            if len(np.unique(y[idx])) < 2:
                continue
            a = roc_auc_score(y[idx], prob[idx]); aucs.append(a)
            if ref is not None:
                diffs.append(a - roc_auc_score(y[idx], ref[idx]))
        row = dict(model=name, n_train=int(tr.sum()), n_test=int(te.sum()), auroc=auc, auroc_ci_low=float(np.percentile(aucs, 2.5)), auroc_ci_high=float(np.percentile(aucs, 97.5)))
        if ref is not None:
            row.update(diff_vs_sequence=auc - roc_auc_score(y, ref), diff_ci_low=float(np.percentile(diffs, 2.5)), diff_ci_high=float(np.percentile(diffs, 97.5)), diff_p_gt0=float(np.mean(np.array(diffs) > 0)))
        else:
            seq_ref = (feats.index[te], prob)
        diag_rows.append(row)
    downstream = pd.DataFrame(diag_rows); downstream.to_csv(TABLES / "hetionet_downstream_single_split.csv", index=False)

    out = dict(graph=summary, n_valid=int(len(valid)), n_genes=int(valid.gene_id.nunique()), training=train_info, summary=summ.to_dict("records"), extra=extra,
               strata=q.to_dict("records"), downstream=downstream.to_dict("records"), runtime_s=time.time() - t0)
    (OUT / "summary_12a.json").write_text(json.dumps(out, indent=2, default=str)); (TABLES / "hetionet_results.json").write_text(json.dumps(out, indent=2, default=str))
    pd.set_option("display.width", 220); print(summ.round(3).to_string(index=False)); print(json.dumps(extra, indent=1)); print(q.to_string(index=False)); print(downstream.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
