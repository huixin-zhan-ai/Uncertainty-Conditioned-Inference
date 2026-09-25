#!/usr/bin/env python3
"""ICLR figure set, regenerated entirely from outputs/corrected/tables.

Design rules: 5.5-inch column (panels 2-wide), never a bare line through a
few means -- every trend shows binned means with 95% intervals over lightly
jittered points; gene-cluster bootstrap where the unit of dependence is the
gene; y-axes start at zero for divergences; framework palette; Helvetica-like
type. Nothing in these figures is computed outside the frozen tables except
binning/bootstrapping of those tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(sys.argv[1]).resolve()            # learning_based_experiments
FIGDIR = Path(sys.argv[2]).resolve()          # iclr_manuscript/figures
T = ROOT / "outputs" / "corrected" / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)

NAVY = "#1F5066"; ORANGE = "#E0851E"; TEAL = "#3C7F79"; GOLD = "#CF9A2B"; PURPLE = "#7E3F98"
GREEN = "#2E8B6F"; GREY = "#8B9096"; INK = "#242830"; MUTED = "#6B7178"; LIGHT = "#D9DDE2"
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.7, "pdf.fonttype": 42, "axes.grid": True,
                     "grid.color": "#EEF0F2", "grid.linewidth": 0.6, "mathtext.default": "regular"})
RNG = np.random.default_rng(42)


def label(ax, s):
    """Merge a bold panel letter with the existing centered title into one left title."""
    t = ax.get_title(loc="center"); ax.set_title("", loc="center")
    ax.set_title(r"$\bf{" + s + r"}$   " + t, loc="left", fontsize=7.4, pad=3)


GENE_COL = {"APP": "#4C78A8", "PSEN1": "#B279A2", "PSEN2": "#72B7B2", "other": "#B0B6BC"}


def gene_points(ax, x, y, genes, legend=False):
    """Per-query points coloured by gene; jittered in x because H_iso is piecewise constant."""
    x = np.asarray(x, float); y = np.asarray(y, float); genes = np.asarray(genes)
    key = np.where(np.isin(genes, ["APP", "PSEN1", "PSEN2"]), genes, "other")
    for g in ["other", "PSEN2", "PSEN1", "APP"]:
        m = key == g
        if m.any():
            ax.scatter(x[m] + RNG.normal(0, 0.012, m.sum()), y[m], s=2.6, color=GENE_COL[g], alpha=0.35, lw=0, zorder=1, rasterized=True, label=(g if legend else None))


def binned(ax, x, y, color, clusters=None, nbins=8, label_=None, points=True, ls="-", marker="o", genes=None, gene_legend=False):
    """Quantile-binned means with 95% bootstrap CI (gene-cluster if clusters given)."""
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    if points and genes is not None:
        gene_points(ax, x[ok], y[ok], np.asarray(genes)[ok], legend=gene_legend)
    x, y = x[ok], y[ok]
    clusters = None if clusters is None else np.asarray(clusters)[ok]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    xs, ms, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x <= b) if b == edges[-1] else (x >= a) & (x < b)
        if sel.sum() < 3:
            continue
        yy = y[sel]; boots = []
        if clusters is not None:
            cl = clusters[sel]; u = np.unique(cl); by = {c: yy[cl == c] for c in u}
            for _ in range(400):
                pick = RNG.choice(u, len(u), replace=True); boots.append(np.concatenate([by[c] for c in pick]).mean())
        else:
            for _ in range(400):
                boots.append(RNG.choice(yy, len(yy), replace=True).mean())
        xs.append(x[sel].mean()); ms.append(yy.mean()); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
    ax.plot(xs, ms, ls, color=color, lw=1.4, marker=marker, ms=3.2, zorder=3, label=label_)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)


def cluster_rho_ci(x, y, clusters, n=1000):
    x = np.asarray(x, float); y = np.asarray(y, float); clusters = np.asarray(clusters)
    u = np.unique(clusters); by = {c: np.where(clusters == c)[0] for c in u}; r = []
    for _ in range(n):
        idx = np.concatenate([by[c] for c in RNG.choice(u, len(u), replace=True)])
        v = spearmanr(x[idx], y[idx]).correlation
        if np.isfinite(v):
            r.append(v)
    return spearmanr(x, y).correlation, np.percentile(r, 2.5), np.percentile(r, 97.5)


# ------------------------------------------------------------------ load
diag = pd.read_csv(T / "diffusion_diagnostics.csv")
prop = pd.read_csv(T / "propagation.csv")
abl = pd.read_csv(T / "ablations.csv")
rank = pd.read_csv(T / "ranking.csv")
base = pd.read_csv(T / "baselines.csv")
mix_steps = pd.read_csv(T / "mixing_curve.csv")
mix_chains = pd.read_csv(T / "mixing_curve_by_chains.csv")
syn = pd.read_csv(T / "synthetic_generality.csv")
rec = pd.read_csv(T / "synthetic_recovery_optimizer.csv")
audit = pd.read_csv(T / "path_audit.csv")
cis = pd.read_csv(T / "bootstrap_cis.csv")

# ================================================================== Fig 2
fig, ax = plt.subplots(2, 2, figsize=(5.5, 4.6))
a, b, c, d = ax.ravel()
# (a) ECDF of sampler JS + chain scaling inset
js = np.sort(diag.js_empirical_to_exact.values); a.plot(js, np.arange(1, len(js) + 1) / len(js), color=NAVY, lw=1.6)
for q, lab, col in [(np.median(js), "median", NAVY), (np.percentile(js, 95), "95th pct", ORANGE)]:
    a.axvline(q, color=col, ls="--", lw=0.9); a.text(q, 0.05, f" {lab}\n {q:.4f}", color=col, fontsize=6.2, va="bottom")
a.set_xlim(0, max(0.012, js.max() * 1.05)); a.set_ylim(0, 1); a.set_xlabel("JS(chain empirical, exact target)"); a.set_ylabel("Fraction of 982 queries")
a.set_title("Sampler fidelity (paper schedule, 512 chains)", fontsize=7.3, pad=2)
ins = a.inset_axes([0.55, 0.28, 0.42, 0.42]); ins.loglog(mix_chains.n_chains, mix_chains.js_mean, "o-", color=NAVY, ms=3, lw=1.2, label="mean JS")
ref = mix_chains.js_mean.iloc[0] * mix_chains.n_chains.iloc[0] / mix_chains.n_chains; ins.loglog(mix_chains.n_chains, ref, ":", color=GREY, lw=1, label="1/N")
ins.set_xlabel("chains N", fontsize=6); ins.set_ylabel("JS", fontsize=6); ins.tick_params(labelsize=5.5); ins.legend(fontsize=5.2, frameon=False); ins.grid(False)
label(a, "a")
# (b) mixing curve
ms = mix_steps[mix_steps.n_chains == 512].groupby("steps").js_empirical_to_exact
fixed = ms.agg(["mean", lambda s: np.percentile(s, 95)]); fixed.columns = ["mean", "p95"]; fixed = fixed.reset_index()
fixed = fixed[fixed.steps.isin([1, 2, 5, 10, 25, 50, 100])]
b.plot(fixed.steps, fixed["mean"], "o-", color=NAVY, lw=1.4, ms=3.2, label="mean JS, 512 chains")
b.fill_between(fixed.steps, 0, fixed.p95, color=NAVY, alpha=0.12, lw=0, label="up to 95th pct")
floor = mix_chains.loc[mix_chains.n_chains == 512, "js_mean"].item(); b.axhline(floor, color=GREY, ls=":", lw=1); b.text(14, floor * 0.60, "Monte-Carlo floor (512 chains)", fontsize=5.8, color=MUTED)
sched = mix_steps[(mix_steps.checkpoint_kind == "schedule") & (mix_steps.n_chains == 512)]
leg = mix_steps[(mix_steps.checkpoint_kind == "legacy") & (mix_steps.n_chains == 512)]
b.errorbar(sched.steps.mean(), sched.js_empirical_to_exact.mean(), xerr=sched.steps.std(), fmt="s", color=ORANGE, ms=4, capsize=2, label=f"paper schedule ({sched.steps.mean():.0f}±{sched.steps.std():.0f} steps)")
b.errorbar(leg.steps.mean(), leg.js_empirical_to_exact.mean(), xerr=leg.steps.std(), fmt="D", color=TEAL, ms=4, capsize=2, label=f"10+round(15H) ({leg.steps.mean():.0f}±{leg.steps.std():.0f} steps)")
b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel("Revision steps"); b.set_ylabel("JS(chain empirical, exact)"); b.set_title("Mixing: JS vs. step budget", fontsize=7.3, pad=2); b.legend(frameon=False, loc="upper right", fontsize=5.8)
label(b, "b")
# (c) exploration: unique final paths and routes vs H
binned(c, diag.isotonic_entropy, diag.unique_final_paths, PURPLE, label_="unique final paths", points=True, genes=diag.gene_symbol, gene_legend=True)
c.set_xlabel("Upstream uncertainty H"); c.set_ylabel("Unique final paths (512 chains)", color=PURPLE)
r1 = cis[cis.quantity.str.startswith("unique final paths")].iloc[0]; r2 = cis[cis.quantity.str.startswith("unique final routes")].iloc[0]
c2 = c.twinx(); c2.grid(False); c2.spines["right"].set_visible(True)
binned(c2, diag.isotonic_entropy, diag.unique_final_routes, GOLD, points=False, ls="--", marker="^", label_="unique final routes")
c2.set_ylabel("Unique final routes (≤ 20)", color=GOLD)
c.set_title(f"Exploration vs. H  (paths ρ={r1.rho:.2f}, routes ρ={r2.rho:.2f})", fontsize=7.3, pad=2); c.set_xlim(-0.03, 1.03)
hc_, lc_ = c.get_legend_handles_labels(); keep = [(h_, l_) for h_, l_ in zip(hc_, lc_) if l_ not in GENE_COL]
gene_handles = [(h_, l_) for h_, l_ in zip(hc_, lc_) if l_ in GENE_COL]
ser_h = [h_ for h_, _ in keep] + c2.get_legend_handles_labels()[0]; ser_l = [l_ for _, l_ in keep] + c2.get_legend_handles_labels()[1]
c.legend(ser_h, ser_l, frameon=False, loc="lower right", fontsize=5.6, handletextpad=0.4)
label(c, "c")
# (d) target entropy vs H: full vs fixed theta vs uniform
for model, col, nm, ls in [("learned_uncertainty", NAVY, "full model θ(H)", "-"), ("fixed_temperature", ORANGE, "fixed θ = 0.80", "--"), ("uniform", GREY, "uniform paths", ":")]:
    g = abl[abl.model == model]
    binned(d, g.isotonic_entropy, g.phenotype_entropy_bits, col, label_=nm, points=(model == "learned_uncertainty"), ls=ls, genes=g.gene_symbol)
rf = cis[cis.quantity == "ablation:learned_uncertainty grouped entropy"].iloc[0]; rx = cis[cis.quantity == "ablation:fixed_temperature grouped entropy"].iloc[0]
d.set_xlabel("Upstream uncertainty H"); d.set_ylabel("Exact phenotype-group entropy (bits)"); d.legend(frameon=False, loc="lower right")
d.set_title(f"Target breadth vs. H  (ρ={rf.rho:.2f}; fixed θ: {rx.rho:.2f})", fontsize=7.3, pad=2); d.set_xlim(-0.03, 1.03)
label(d, "d")
fig.tight_layout(w_pad=1.6, h_pad=1.4, rect=(0, 0.045, 1, 1))
order = ["APP", "PSEN1", "PSEN2", "other"]; gh = {l_: h_ for h_, l_ in gene_handles}
fig.legend([gh[g] for g in order if g in gh], [f"{g} ({int((diag.gene_symbol == g).sum())})" if g != "other" else f"other genes ({int((~diag.gene_symbol.isin(['APP','PSEN1','PSEN2'])).sum())})" for g in order if g in gh],
           loc="lower center", ncol=5, frameon=False, fontsize=6.0, markerscale=2.4, handletextpad=0.3, columnspacing=1.2, title="Points in (c),(d): individual queries, coloured by gene", title_fontsize=6.0, bbox_to_anchor=(0.5, -0.005))
fig.savefig(FIGDIR / "fig2_sampler_uncertainty.pdf", bbox_inches="tight"); plt.close(fig)

# ================================================================== Fig 3
fig = plt.figure(figsize=(5.5, 4.3)); gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.15, 1.0])
a = fig.add_subplot(gs[0, :]); b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])
# (a) H-entropy rho with cluster CI: ablations + baselines
rows = []
for model, nm in [("learned_uncertainty", "Full model"), ("no_confidence_features", "No (1−H) interactions"), ("no_mechanism", "No mechanism features"),
                  ("fixed_temperature", "Fixed θ"), ("fixed_theta_no_confidence", "Fixed θ + no (1−H)")]:
    g = abl[abl.model == model]; rows.append((nm, *cluster_rho_ci(g.isotonic_entropy, g.phenotype_entropy_bits, g.gene_id), "ablation"))
for model, nm in [("beam_20", "Beam k=20 (no θ(H))"), ("beam_5", "Beam k=5 (no θ(H))"), ("ppr", "Personalized PageRank"), ("uniform_paths", "Uniform paths")]:
    g = base[base.model == model]; rows.append((nm, *cluster_rho_ci(g.isotonic_entropy, g.phenotype_entropy_bits, g.gene_id), "baseline"))
rows = rows[::-1]
for i, (nm, r, lo, hi, kind) in enumerate(rows):
    col = NAVY if kind == "ablation" else GREY
    a.plot([lo, hi], [i, i], color=col, lw=1.4); a.plot(r, i, "o", color=col, ms=4)
a.axvline(0, color=LIGHT, lw=1); a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows]); a.set_xlim(-1.05, 1.05)
a.set_xlabel("Spearman ρ(H, group entropy) with gene-cluster 95% CI"); a.set_title("Uncertainty coupling by model", fontsize=7.5, pad=2); label(a, "a"); a.grid(axis="y", visible=False)
# (b) JS to full model
order = [("uniform_paths", "Uniform"), ("ppr", "PPR"), ("beam_20", "Beam-20"), ("beam_5", "Beam-5")]
for i, (m, nm) in enumerate(order):
    v = base[base.model == m].js_to_full.values; b.boxplot(v, positions=[i], widths=0.55, showfliers=False, patch_artist=True,
        boxprops=dict(facecolor=GREY, alpha=0.35, lw=0.7), medianprops=dict(color=INK, lw=1), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
for j, (m, nm) in enumerate([("no_mechanism", "No mech."), ("fixed_temperature", "Fixed θ"), ("no_confidence_features", "No (1−H)")]):
    v = abl[abl.model == m].js_to_full.values; i = len(order) + j; b.boxplot(v, positions=[i], widths=0.55, showfliers=False, patch_artist=True,
        boxprops=dict(facecolor=NAVY, alpha=0.35, lw=0.7), medianprops=dict(color=INK, lw=1), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
b.set_xticks(range(7)); b.set_xticklabels([n for _, n in order] + ["No mech.", "Fixed θ", "No (1−H)"], rotation=35, ha="right"); b.set_ylim(0)
b.set_ylabel("JS to full model (per query)"); b.set_title("Distance from the learned target", fontsize=7.5, pad=2); label(b, "b")
# (c) topology audit
top = audit.sort_values("candidate_routes", ascending=True).tail(10)
c.barh(top.gene_symbol, top.candidate_routes, color=TEAL, height=0.65); c.axvline(20, color=ORANGE, ls="--", lw=0.9); c.text(20.3, 0.2, "route cap", color=ORANGE, fontsize=6, rotation=90, va="bottom")
c.set_xlabel("Retained valid routes"); c.set_title("Valid routes per gene", fontsize=7.5, pad=2); c.grid(axis="y", visible=False); label(c, "c")
fig.tight_layout(w_pad=1.4, h_pad=1.6); fig.savefig(FIGDIR / "fig3_ablations_baselines.pdf"); plt.close(fig)

# ================================================================== Fig 4 (synthetic)
fig, ax = plt.subplots(2, 2, figsize=(5.5, 4.4)); a, b, c, d = ax.ravel()
cols = {53: NAVY, 256: TEAL, 1024: PURPLE}
for K, g in syn.groupby("n_states"):
    s = g.groupby("uncertainty").js_empirical_to_exact.agg(["mean", lambda x: np.percentile(x, 95)]); s.columns = ["mean", "p95"]
    a.plot(s.index, s["mean"], "o-", color=cols[K], ms=3, lw=1.3, label=f"{K} states"); a.fill_between(s.index, 0, s.p95, color=cols[K], alpha=0.10, lw=0)
a.set_ylim(0, 0.012); a.set_xlabel("Uncertainty H"); a.set_ylabel("JS(chain empirical, exact)"); a.set_title("Sampler fidelity (mean; shaded to 95th pct)", fontsize=7.3, pad=2); a.legend(frameon=False); label(a, "a")
for K, g in syn.groupby("n_states"):
    s = g.groupby("uncertainty").acceptance_rate.agg(["mean", "std"]); b.plot(s.index, s["mean"], "o-", color=cols[K], ms=3, lw=1.3); b.fill_between(s.index, s["mean"] - 1.96 * s["std"] / np.sqrt(30), s["mean"] + 1.96 * s["std"] / np.sqrt(30), color=cols[K], alpha=0.18, lw=0)
b.set_xlabel("Uncertainty H"); b.set_ylabel("Acceptance rate"); b.set_title("Acceptance vs. H (30 replicates, 95% CI)", fontsize=7.3, pad=2); label(b, "b")
for K, g in syn.groupby("n_states"):
    s = g.groupby("uncertainty")[["exact_group_entropy", "fixed_group_entropy"]].mean(); c.plot(s.index, s.exact_group_entropy, "o-", color=cols[K], ms=3, lw=1.3, label=f"{K} states, θ(H)"); c.plot(s.index, s.fixed_group_entropy, ":", color=cols[K], lw=1.2)
c.plot([], [], ":", color=GREY, label="fixed θ (dotted)"); c.set_xlabel("Uncertainty H"); c.set_ylabel("Target group entropy (bits)"); c.set_title("Target broadening vs. fixed θ", fontsize=7.3, pad=2); c.legend(frameon=False, fontsize=5.8); label(c, "c")
# (d) recovery: cosine by K / N for default vs informed init
sub = rec[rec.variant.isin(["default", "informed"])].copy(); sub["cond"] = sub.n_states.astype(str) + "/" + sub.n_train.astype(str)
conds = ["53/121", "256/121", "1024/121", "1024/500", "1024/2000"]; x = np.arange(len(conds))
for off, (var, col, nm) in zip([-0.18, 0.18], [("default", NAVY, "default init"), ("informed", ORANGE, "informed init")]):
    for i, cnd in enumerate(conds):
        v = sub[(sub.variant == var) & (sub.cond == cnd)].weight_cosine.values
        d.scatter(np.full(len(v), x[i] + off) + RNG.normal(0, 0.03, len(v)), v, s=14, color=col, alpha=0.75, lw=0, label=nm if i == 0 else None)
d.axhline(0.8, color=LIGHT, lw=1); d.set_xticks(x); d.set_xticklabels([f"K={c.split('/')[0]}\nN={c.split('/')[1]}" for c in conds], fontsize=6.2)
d.set_ylabel("cos(learned w, planted w*)"); d.set_ylim(-0.6, 1.05); d.set_title("Energy recovery (5 replicates)", fontsize=7.3, pad=2); d.legend(frameon=False, loc="lower left"); label(d, "d")
fig.tight_layout(w_pad=1.4, h_pad=1.4); fig.savefig(FIGDIR / "fig4_synthetic.pdf"); plt.close(fig)

# ================================================================== Fig 5 (perturbation dissociation)
fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3)); a, b = ax
binned(a, rank.isotonic_entropy, rank.mean_js, NAVY); r = cis[cis.quantity == "perturbation JS (+/-0.30)"].iloc[0]
a.set_xlabel("Upstream uncertainty H"); a.set_ylabel("JS after H ± 0.30"); a.set_ylim(0); a.set_title(f"Distributional sensitivity falls with H (ρ={r.rho:.2f})", fontsize=7.2, pad=2); a.set_xlim(-0.03, 1.03); label(a, "a")
binned(b, rank.isotonic_entropy, rank.mean_top1_switch, ORANGE); r = cis[cis.quantity == "perturbation top-1 switch"].iloc[0]
b.set_xlabel("Upstream uncertainty H"); b.set_ylabel("Top-1 switch rate"); b.set_ylim(0); b.set_title(f"Leader switching rises with H (ρ={r.rho:.2f})", fontsize=7.2, pad=2); b.set_xlim(-0.03, 1.03); label(b, "b")
fig.tight_layout(w_pad=1.6); fig.savefig(FIGDIR / "fig5_perturbation.pdf"); plt.close(fig)
print("figures written:", sorted(p.name for p in FIGDIR.glob("fig[2-5]_*.pdf")))
