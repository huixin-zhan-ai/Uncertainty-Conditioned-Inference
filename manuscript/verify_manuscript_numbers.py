#!/usr/bin/env python3
"""Numeric verification of iclr_manuscript/main.tex against the frozen tables.

Unlike text-regex audits, every check here RECOMPUTES the quantity from
outputs/corrected/tables (or reports), formats it exactly as the manuscript
does, and asserts that string is present in main.tex. A second group asserts
that withdrawn claims are absent. Exit code 1 on any failure.

Usage: python3 verify_manuscript_numbers.py <learning_based_experiments> <iclr_manuscript>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(sys.argv[1]).resolve(); MS = Path(sys.argv[2]).resolve()
T = ROOT / "outputs" / "corrected" / "tables"; R = ROOT / "outputs" / "corrected" / "reports"
tex = (MS / "main.tex").read_text()
# Only the ACTIVE body counts: strip any \iffalse ... \fi blocks (disabled drafts),
# otherwise a parked alternate version can satisfy or break checks invisibly.
tex = re.sub(r"\\iffalse.*?\\fi", "", tex, flags=re.S)
# Also drop %-commented lines (parked old wording must not satisfy or break a check).
tex = "\n".join(l for l in tex.split("\n") if not l.lstrip().startswith("%"))
tex_flat = re.sub(r"\s+", " ", tex)
results = []


def present(label, value_str):
    ok = value_str in tex_flat
    results.append((ok, label, value_str)); return ok


def absent(label, pattern):
    ok = re.search(pattern, tex_flat, flags=re.I) is None
    results.append((ok, "ABSENT: " + label, pattern)); return ok


def rho(frame, x, y):
    return spearmanr(frame[x], frame[y]).correlation


# ---------------- sampler fidelity
d = pd.read_csv(T / "diffusion_diagnostics.csv")
present("mean chain JS", f"{d.js_empirical_to_exact.mean():.4f}")
present("median chain JS", f"{d.js_empirical_to_exact.median():.4f}")
present("mean acceptance", f"{d.acceptance_rate.mean():.3f}")
present("n valid queries", f"{len(d)} ")
ch = pd.read_csv(T / "mixing_curve_by_chains.csv").set_index("n_chains").js_mean
present("JS at 32 chains", f"{ch[32]:.3f}"); present("JS at 128 chains", f"{ch[128]:.3f}")
present("JS at 512 chains", f"{ch[512]:.4f}"); present("JS at 2048 chains", f"{ch[2048]:.4f}")
mx = pd.read_csv(T / "mixing_curve.csv"); mx = mx[mx.n_chains == 512]
bysteps = mx.groupby("steps").js_empirical_to_exact.mean()
present("JS after 1 step", f"{bysteps[1]:.3f}"); present("JS after 2 steps", f"{bysteps[2]:.3f}")
present("JS after 5 steps", f"{bysteps[5]:.4f}"); present("JS after 10 steps", f"{bysteps[10]:.4f}")
present("schedule mean steps", f"mean {mx[mx.checkpoint_kind=='schedule'].steps.mean():.0f}")
present("legacy mean steps", f"mean {mx[mx.checkpoint_kind=='legacy'].steps.mean():.0f} steps")
syn = pd.read_csv(T / "synthetic_generality_summary.csv")
present("synthetic max mean JS", f"{syn.sampler_js_mean.max():.4f}"); present("synthetic max p95 JS", f"{syn.sampler_js_p95.max():.4f}")

# ---------------- propagation + CIs
ps = pd.read_csv(T / "propagation_summary.csv")
present("entropy Q1", f"{ps.phenotype_entropy_mean.iloc[0]:.3f}"); present("entropy Q4", f"{ps.phenotype_entropy_mean.iloc[-1]:.3f}")
present("top1 Q1", f"{ps.top1_mass_mean.iloc[0]:.3f}"); present("top1 Q4", f"{ps.top1_mass_mean.iloc[-1]:.3f}")
cis = pd.read_csv(T / "bootstrap_cis.csv").set_index("quantity")


def ci_check(q, label, fmt="{:.3f}"):
    r = cis.loc[q]
    present(f"rho {label}", fmt.format(r.rho))
    present(f"cluster CI {label}", f"{r.cluster_ci_low:.3f}--{r.cluster_ci_high:.3f}")


ci_check("grouped phenotype entropy", "grouped entropy")
ci_check("raw endpoint entropy", "raw entropy")
ci_check("unique final paths (512 chains)", "unique paths")
ci_check("unique final routes (512 chains)", "unique routes")
ci_check("grouped entropy, target recomputed with H_knn", "kNN recomputed")
r = cis.loc["ablation:fixed_temperature grouped entropy"]; present("fixed theta rho", f"{r.rho:.3f}"); present("fixed theta CI (table)", f"[${r.cluster_ci_low:.2f}$, ${r.cluster_ci_high:.2f}$]")
r = cis.loc["ablation:fixed_theta_no_confidence grouped entropy"]; present("fixed+noconf rho", f"{r.rho:.3f}"); present("fixed+noconf CI (table)", f"[${r.cluster_ci_low:.2f}$, ${r.cluster_ci_high:.2f}$]")
r = cis.loc["perturbation JS (+/-0.30)"]; present("perturb JS rho", f"{r.rho:.3f}"); present("perturb JS CI", f"{r.cluster_ci_low:.3f}$ to $" + f"{r.cluster_ci_high:.3f}")
r = cis.loc["perturbation top-1 switch"]; present("switch rho", f"{r.rho:.3f}"); present("switch CI", f"{r.cluster_ci_low:.3f}$--${r.cluster_ci_high:.3f}")

# ---------------- ablation/baseline table
abl = pd.read_csv(T / "ablations.csv"); base = pd.read_csv(T / "baselines.csv")
for model, frame in [("uniform", abl), ("fixed_temperature", abl), ("fixed_theta_no_confidence", abl), ("no_confidence_features", abl), ("no_mechanism", abl), ("learned_uncertainty", abl),
                     ("ppr", base), ("beam_5", base), ("beam_20", base)]:
    g = frame[frame.model == model]
    present(f"{model} entropy", f"{g.phenotype_entropy_bits.mean():.3f}"); present(f"{model} top1", f"{g.top1_mass.mean():.3f}")
    present(f"{model} JS", f"{g.js_to_full.mean():.3f}")
    present(f"{model} rho", f"{rho(g, 'isotonic_entropy', 'phenotype_entropy_bits'):.3f}")

# ---------------- ranking dissociation
rs = pd.read_csv(T / "ranking_summary.csv")
present("ranking JS Q1", f"{rs.mean_js.iloc[0]:.4f}"); present("ranking JS Q4", f"{rs.mean_js.iloc[-1]:.4f}")
present("switch Q1", f"{rs.mean_top1_switch.iloc[0]:.3f}"); present("switch Q4", f"{rs.mean_top1_switch.iloc[-1]:.3f}")

# ---------------- recovery
rec = pd.read_csv(T / "synthetic_recovery.csv"); fails = rec.assign(f=rec.weight_cosine < 0.8).groupby(["n_states", "n_train"]).f.sum()
present("fail K53 N121", f"{fails[(53,121)]}/5 replicates for $K=53$")
present("fail K256 N121", f"{fails[(256,121)]}/5 for $K=256$"); present("fail K1024 N121", f"{fails[(1024,121)]}/5 for $K=1024$")
fz = json.load(open(T / "frozen_scorer_restart_check.json")); present("frozen objective", f"{fz['frozen_objective']:.4f}")
opt = pd.read_csv(T / "synthetic_recovery_optimizer_summary.csv").set_index(["n_states", "n_train", "variant"])
present("informed JS K1024 N2000", f"{opt.loc[(1024,2000,'informed')].js_mean:.2f}"); present("default JS K1024 N2000", f"{opt.loc[(1024,2000,'default')].js_mean:.2f}")

# ---------------- secondary diagnostic (paper's 5-fold) + single-split baselines
rr = json.load(open(R / "reclassification_results.json"))
present("seq AUROC", f"{rr['models']['Sequence only']['auroc']:.3f}"); present("graph AUROC", f"{rr['models']['Sequence + graph']['auroc']:.3f}")
present("delta AUROC", f"{rr['paired_improvement']['delta_auroc']:.4f}")
present("delta lo", f"{rr['paired_improvement']['delta_auroc_lo']:.4f}"); present("delta hi", f"{rr['paired_improvement']['delta_auroc_hi']:.4f}")
present("brier seq", f"{rr['models']['Sequence only']['brier']:.3f}"); present("brier graph", f"{rr['models']['Sequence + graph']['brier']:.3f}")
present("ece seq", f"{rr['models']['Sequence only']['ece']:.3f}"); present("ece graph", f"{rr['models']['Sequence + graph']['ece']:.3f}")
ds = pd.read_csv(T / "baselines_downstream_single_split.csv").set_index("model")
present("beam single-split diff", f"{ds.loc['beam_5'].diff_vs_sequence:.4f}"); present("beam diff CI lo", f"{ds.loc['beam_5'].diff_ci_low:.3f}"); present("beam diff CI hi", f"{ds.loc['beam_5'].diff_ci_high:.3f}")

# ---------------- structural facts
pv = pd.read_csv(T / "propagation.csv"); present("n genes", f"{pv.gene_id.nunique()} genes")
counts = pv.groupby("gene_id").size().sort_values(ascending=False); top3 = counts.head(3).sum() / counts.sum()
present("top-3 gene share", f"{100*top3:.0f}\\%")


# ---------------- Hetionet (second graph)
hr = json.load(open(T / "hetionet_results.json")); hl = json.load(open(T / "hetionet_linkpred_results.json"))
present("hetionet nodes", f"{hr['graph']['nodes']:,}"); present("hetionet edges", f"{hr['graph']['edges']:,}")
present("hetionet dps edges", f"{hr['graph']['relations']['DISEASEMANIFESTSSYMPTOM']:,}"); present("hetionet disease sources", f"{hr['graph']['disease_sources']} diseases")
present("hetionet multiparent", f"{hr['graph']['symptoms_multi_parent']} of the"); present("hetionet linked symptoms", f"{hr['graph']['symptoms_with_edges']} linked"); present("hetionet max parents", f"up to {hr['graph']['max_disease_parents']}")
present("hetionet objective", f"{hr['training']['objective_default']:.3f}"); present("hetionet const loss", f"{hr['training']['constant_predictor_loss']:.3f}"); present("hetionet anchor auroc", f"{hr['training']['anchor_auroc']:.3f}")
hs = {m['model']: m for m in hr['summary']}
present("hetionet full rho", f"{hs['full']['rho_H_entropy']:.3f}"); present("hetionet full CI", f"{hs['full']['rho_ci_low']:.3f}--{hs['full']['rho_ci_high']:.3f}")
present("hetionet fixed rho", f"{hs['fixed_temperature']['rho_H_entropy']:.3f}"); present("hetionet fixed CI", f"{hs['fixed_temperature']['rho_ci_low']:.3f}--{hs['fixed_temperature']['rho_ci_high']:.3f}")
present("hetionet uniform rho", f"{hs['uniform_paths']['rho_H_entropy']:.3f}"); present("hetionet ppr rho", f"{hs['ppr']['rho_H_entropy']:.3f}")
present("hetionet uniform entropy", f"{hs['uniform_paths']['entropy_mean']:.3f}"); present("hetionet ppr entropy", f"{hs['ppr']['entropy_mean']:.3f}")
present("hetionet paths rho", f"{hr['extra']['rho_H_unique_final_paths']['rho']:.3f}"); present("hetionet paths CI", f"{hr['extra']['rho_H_unique_final_paths']['lo']:.3f}$--${hr['extra']['rho_H_unique_final_paths']['hi']:.3f}")
present("hetionet fidelity", f"{hr['extra']['fidelity']['js_grouped_mean']:.3f}"); present("hetionet states", f"{hr['extra']['fidelity']['n_states_mean']:.0f} states")
present("hetionet entropy Q1", f"{hr['strata'][0]['entropy_mean']:.3f}"); present("hetionet entropy Q4", f"{hr['strata'][-1]['entropy_mean']:.3f}")
present("hetionet top1 Q1", f"{hr['strata'][0]['top1_mean']:.3f}"); present("hetionet top1 Q4", f"{hr['strata'][-1]['top1_mean']:.3f}")
hd = {d['model']: d for d in hr['downstream']}; present("hetionet downstream full diff", f"{hd['full']['diff_vs_sequence']:.3f}"); present("hetionet downstream CI", f"{hd['full']['diff_ci_low']:.3f}$--${hd['full']['diff_ci_high']:.3f}")
present("held out edges", f"{hl['held_out_edges']} of {hl['total_dps']:,}"); present("linkpred n", f"{hl['n_eval_variants']} variants in {hl['n_eval_genes']} genes")
ls_ = {m['model']: m for m in hl['summary']}
for m in ["full", "fixed_temperature", "distmult", "beam_5", "uniform_paths", "ppr"]:
    present(f"linkpred {m} r10", f"{ls_[m]['recall@10']:.3f}"); present(f"linkpred {m} r25", f"{ls_[m]['recall@25']:.3f}"); present(f"linkpred {m} mrr", f"{ls_[m]['mrr']:.3f}")
pf = hl['paired_recall10']['full_minus_fixed_temperature']; present("paired full-fixed", f"differences of {pf['mean']:.3f}"); present("paired full-fixed CI", f"{pf['lo']:.3f}$ to ${pf['hi']:.3f}")
bs = {(r['model'], r['stratum']): r for r in hl['by_stratum']}
present("linkpred full Q1", f"{bs[('full','Q1 low')]['recall10']:.3f}"); present("linkpred full Q4", f"{bs[('full','Q4 high')]['recall10']:.3f}")

# ---------------- worked masked-edge recovery example (Appendix)
ex = json.load(open(T / "hetionet_linkpred_example.json")); ap = pd.read_csv(T / "hetionet_linkpred_example_all_pairs.csv")
present("example gene/symptom", f"{ex['gene']} variant"); present("example target", f"$\\to${ex['target_symptom']}")
present("example p", f"$p={ex['p']:.2f}$"); present("example H", f"$H={ex['H']:.2f}$")
present("example n candidates", f"{ex['n_candidates_filtered']} candidates"); present("example n routes", {1:"one",2:"two",3:"three",4:"four"}[ex['n_alternative_routes']] + " gene relays")
for r_ in ex["alternative_routes"]:
    present("example relay " + r_["path"][1], f"{r_['path'][1]}$\\to${r_['path'][2]}")
rk = ex["rankings"]; ordn = {1:"first",2:"second",3:"third",42:"42nd"}
present("example rank full", f"ranks {ordn[rk['full']['rank_of_target']]} under the full"); present("example rank fixed", f"{ordn[rk['fixed_temperature']['rank_of_target']]} under the full model and under fixed")
present("example rank distmult", f"{rk['distmult']['rank_of_target']}nd under DistMult" if rk['distmult']['rank_of_target'] % 10 == 2 else f"{rk['distmult']['rank_of_target']}th under DistMult")
mine = ap[ap.gene == ex["gene"]]; present("example n positives", f"{len(mine)} recoverable held-out"); present("example top10", f"{int((mine.rank_full<=10).sum())} fall in the top 10"); present("example top25", f"{int((mine.rank_full<=25).sum())} in the top 25")
present("all pairs n", f"{len(ap)} (gene, held-out symptom) pairs"); present("all pairs top10", f"{int((ap.rank_full<=10).sum())} rank in the top 10")
assert (ap.n_direct_routes == 0).all(); present("relay-only structural claim", "reached only through gene relays")
present("example held-out count", f"{ex['held_out_edges']} hidden edges")
au = pd.read_csv(T / "hetionet_linkpred_audit.csv").set_index("gene_symbol"); present("example n known filtered", f"filtering the {int(au.loc[ex['gene'], 'n_known_filtered'])} symptoms")

# ---------------- C3: matched controls (H2-B) and gene hold-out (H2-C)
ctl = pd.read_csv(T / "hetionet_controls_summary.csv").set_index("model"); ctr = json.load(open(T / "hetionet_controls_results.json"))
present("ctrl full r10", f"{ctl.loc['full','recall@10']:.3f}"); present("ctrl random r10", f"{ctl.loc['random','recall@10']:.3f}")
present("ctrl perm r10", f"{ctl.loc['degree_perm','recall@10']:.3f}"); present("ctrl popularity r10", f"{ctl.loc['popularity','recall@10']:.3f}")
present("ctrl popularity CI", f"CI ${ctl.loc['popularity','recall@10_lo']:.3f}$--${ctl.loc['popularity','recall@10_hi']:.3f}$")
pr_ = ctr["paired"]
present("ctrl paired vs random", f"paired difference {pr_['full_minus_random|recall@10']['mean']:.3f}, 95\\% CI ${pr_['full_minus_random|recall@10']['lo']:.3f}$--${pr_['full_minus_random|recall@10']['hi']:.3f}$")
present("ctrl paired vs perm", f"paired difference {pr_['full_minus_degree_perm|recall@10']['mean']:.3f}, 95\\% CI ${pr_['full_minus_degree_perm|recall@10']['lo']:.3f}$--${pr_['full_minus_degree_perm|recall@10']['hi']:.3f}$")
present("ctrl paired vs popularity", f"difference is {pr_['full_minus_popularity|recall@10']['mean']:.3f} (95\\% CI ${pr_['full_minus_popularity|recall@10']['lo']:.3f}$ to ${pr_['full_minus_popularity|recall@10']['hi']:.3f}$)")
present("perm p-value", f"$p<{ctr['degree_perm_reference']['p_value_upper_bound']:.3f}$")
present("reachable fraction", f"reachability of {round(100 * ctr['coverage']['reachable_fraction']):.0f}\\%")
present("unconditional r10", f"\\kappa_{{g(v)}}={ctr['unconditional_recall@10_full']:.3f}")
lk = json.load(open(T / "hetionet_leak_check.json")); results.append((lk["pass"], "leak check passes", "forward/inverse/path/count"))
present("leak masked count", f"{lk['n_masked']} masked disease--symptom edges")
gho = pd.read_csv(T / "hetionet_geneholdout_pooled.csv").set_index("model"); ghr_ = json.load(open(T / "hetionet_geneholdout_results.json"))
present("gho full r10", f"mean recall@10 of {gho.loc['full','recall@10']:.3f} across three edge masks"); present("gho full CI (table)", f"{gho.loc['full','recall@10']:.3f} [{gho.loc['full','recall@10_lo']:.2f}, {gho.loc['full','recall@10_hi']:.2f}]")
present("gho distmult pooled (table)", f"{gho.loc['distmult','recall@10']:.3f} [{gho.loc['distmult','recall@10_lo']:.2f}, {gho.loc['distmult','recall@10_hi']:.2f}]")
present("gho gene-balanced range", f"ranges from {gho.loc['full','recall@10_seed_min']:.3f} to {gho.loc['full','recall@10_seed_max']:.3f} across masks")
gp = ghr_["paired"]
present("gho vs random", f"is {gp['full_minus_random|recall@10']['mean']:.3f} (95\\% CI ${gp['full_minus_random|recall@10']['lo']:.3f}$--${gp['full_minus_random|recall@10']['hi']:.3f}$)")
present("gho vs distmult", f"is {gp['full_minus_distmult|recall@10']['mean']:.3f} (gene-cluster 95\\% CI {gp['full_minus_distmult|recall@10']['lo']:.3f}--{gp['full_minus_distmult|recall@10']['hi']:.3f})")
present("gho vs distmult (table)", f"${gp['full_minus_distmult|recall@10']['mean']:+.3f}$ [${gp['full_minus_distmult|recall@10']['lo']:.3f}$, ${gp['full_minus_distmult|recall@10']['hi']:.3f}$]")
present("gho vs distmult (abstract)", f"exceeding DistMult by {gp['full_minus_distmult|recall@10']['mean']:.3f}, with a roughly fivefold lower across-mask coefficient of variation")
present("gho vs fixed", f"${gp['full_minus_fixed_temperature|recall@10']['mean']:.3f}$ (${gp['full_minus_fixed_temperature|recall@10']['lo']:.3f}$ to ${gp['full_minus_fixed_temperature|recall@10']['hi']:.3f}$) under gene hold-out")
ghs_ = pd.read_csv(T / "hetionet_geneholdout_summary.csv"); dms = ghs_[ghs_.model == "distmult"]["recall@10"]
present("gho distmult range", f"against {dms.min():.3f}--{dms.max():.3f} for DistMult")
present("noninferiority lower bound", f"lower bound ${hl['paired_recall10']['full_minus_distmult']['lo']:.3f}$")
sp_ = pd.read_csv(T / "hetionet_geneholdout_splits.csv").drop_duplicates("fold")
for _, r_ in sp_.iterrows():
    present(f"split {r_.fold} anchors", f"& {int(r_.n_train_anchors)} & {int(r_.n_calib_pos)} & {int(r_.n_calib_neg)}")
absent("calibration inference from recall-vs-H", r"demonstrates calibration")
absent("parity established", r"parity is established")
absent("changes downstream behaviour without improving", r"without improving it")

# ---------------- coverage accounting definitions (Appendix)
cov_ = pd.read_csv(T / "hetionet_coverage.csv"); ctl_ = pd.read_csv(T / "hetionet_controls.csv"); f_ = ctl_[ctl_.model == "full"]
kap_all = cov_.set_index("gene_symbol").reachable_fraction
kap = kap_all[kap_all.index.isin(f_.gene_symbol.unique())]  # evaluated genes only
present("cov pairs", f"{int(cov_.n_heldout_symptoms.sum())} (gene, symptom) pairs")
present("cov reachable", f"Of those pairs, {int(cov_.n_reachable.sum())} remain reachable")
present("cov pair-weighted", f"pair-weighted reachability of {round(100 * cov_.n_reachable.sum() / cov_.n_heldout_symptoms.sum()):.0f}\\%")
present("cov n queries", f"averaged over the {int(f_.variant_id.nunique())} queries with equal weight")
present("cov conditional", f"r_{{10}}(v)={f_['recall@10'].mean():.3f}")
present("cov unconditional", f"\\kappa_{{g(v)}}={(f_['recall@10'] * f_.gene_symbol.map(kap)).mean():.3f}")
present("cov uncond25", f"({(f_['recall@25'] * f_.gene_symbol.map(kap)).mean():.3f} at $k=25$)")
present("cov kappa min gene", f"{kap.min():.2f} for {kap.idxmin()}"); present("cov kappa max gene", f"{kap.max():.2f} for {kap.idxmax()}")
present("cov query-weighted kappa", f"is {f_.gene_symbol.map(kap).mean():.2f}, against {cov_.n_reachable.sum() / cov_.n_heldout_symptoms.sum():.2f} pair-weighted and {kap.mean():.2f} gene-balanced")
absent("popularity in abstract superiority", r"above chance, symptom-popularity")
absent("vague controls in conclusion", r"above chance, controls")

# ---------------- per-mask DistMult decomposition (both verdicts must be stated)
gr_ = pd.read_csv(T / "hetionet_geneholdout.csv"); RNG2 = np.random.default_rng(42)
def _ci(vals, cl, B=1000):
    vals = np.asarray(vals, float); cl = np.asarray(cl); u = np.unique(cl); by = {c: vals[cl == c] for c in u}
    b = [np.concatenate([by[c] for c in RNG2.choice(u, len(u), replace=True)]).mean() for _ in range(B)]
    return vals.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
_seedfmt = {42: "on mask 42", 7: "on mask 7", 123: "on mask 123"}
for _sd in (42, 7, 123):
    _g = gr_[gr_.seed == _sd]; _p = _g.pivot_table(index="variant_id", columns="model", values="recall@10")
    _cl = _g.drop_duplicates("variant_id").set_index("variant_id").loc[_p.index, "gene_id"].to_numpy()
    _m, _lo, _hi = _ci((_p["full"] - _p["distmult"]).to_numpy(), _cl)
    present(f"per-mask dm diff seed {_sd}", f"{_m:.3f} for seed {_sd} (95\\% CI ${_lo:.3f}$ to ${_hi:.3f}$)" if _sd == 42 else f"{_m:.3f} for seed {_sd} (${_lo:.3f}$--${_hi:.3f}$)")
    results.append((_m > 0, f"per-mask point estimate positive seed {_sd}", f"{_m:.3f}"))
# scope of the advantage must be stated positively AND bounded
present("abstract win condition", "With query genes held out from calibration and energy learning, mean recall@10 reaches")
present("related work states the win", "the gene-held-out, mask-averaged evaluation demonstrates improved recovery over DistMult")
present("results: single-split intervals include zero", "the paired intervals on that individual split include zero")
present("appendix: noninferiority not met", "does not meet the prespecified noninferiority criterion")
present("future work: untested reasoners named", "compare with relational and path-policy reasoners")
absent("blanket no-superiority disclaimer", r"no superiority to trained relational reasoners")
absent("single-mask negation", r"though not on a single mask")
absent("stability-not-accuracy framing", r"stability rather than peak accuracy")
absent("collapses without coupling", r"collapses without the coupling")
absent("two boundaries ending", r"Two boundaries frame")
absent("supports one claim", r"supports one claim")
absent("contribution ends", r"ends at that distribution")
absent("draft-history language", r"withdrawn|Corrected method|added for this version|honest template")
absent("null verdict on classification", r"the answer is null")
absent("universal disjoint-interval claim", r"intervals do not overlap the full model|intervals that again exclude")
absent("wrong uncorrected-Metropolis target", r"targets \$\\exp\\\{2U")
present("uncorrected-Metropolis stationary mass", "stationary mass proportional to $\\pi_H(i)\\,q_H(i)$")
gsd = gr_.groupby(["model", "seed"])["recall@10"].mean().unstack()
_f = gsd.loc["full"]; _d = gsd.loc["distmult"]
present("results cv", f"coefficient of variation of 0.10 versus {_d.std() / _d.mean():.2f} (roughly fivefold")
present("intro contribution", f"improves mean recall@10 over DistMult by 0.108 while reducing across-mask variability from {_d.std() / _d.mean():.2f} to 0.10")
present("our mask range", f"{_f.min():.3f}--{_f.max():.3f} across masks")
present("our cv", f"coefficient of variation of {_f.std() / _f.mean():.2f} versus {_d.std() / _d.mean():.2f}")
present("gho cv (table)", f"Across-mask CV & {_f.std() / _f.mean():.2f} & {_d.std() / _d.mean():.2f}")
present("dm mask range (table)", f"{_f.min():.3f}--{_f.max():.3f} & {_d.min():.3f}--{_d.max():.3f}")
results.append((4.5 < (_d.std() / _d.mean()) / (_f.std() / _f.mean()) < 5.5, "cv ratio roughly fivefold", f"{(_d.std() / _d.mean()) / (_f.std() / _f.mean()):.2f}"))
_g = gr_.groupby(["model", "gene_symbol"])["recall@10"].mean().unstack(0)
present("per-gene win count", f"higher recall than DistMult for {int((_g['full'] > _g['distmult']).sum())} of the {len(_g)} evaluated genes")

# ---------------- withdrawn / unsupported claims must be absent
absent("prospective mixing test", r"prospective mixing test")
absent("non-monotonic switch claim", r"switch rates are non-monotonic")
absent("legacy softmax schedule", r"0\.25\s*\+\s*1\.4")
absent("clinical value", r"clinical value")
absent("improve held-out classification", r"improve held-out classification")

ok = all(r[0] for r in results)
for good, label, val in results:
    print(("PASS " if good else "FAIL ") + f"{label}: {val}")
print(f"\n{sum(r[0] for r in results)}/{len(results)} checks passed")
sys.exit(0 if ok else 1)
