#!/usr/bin/env python3
"""Fig 6 (Hetionet): non-degenerate second graph, same upstream signal.
All panels from outputs/corrected/tables/hetionet_*.csv."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]).resolve(); FIGDIR = Path(sys.argv[2]).resolve(); T = ROOT / "outputs" / "corrected" / "tables"
NAVY = "#1F5066"; ORANGE = "#E0851E"; TEAL = "#3C7F79"; GOLD = "#CF9A2B"; PURPLE = "#7E3F98"; GREY = "#8B9096"; INK = "#242830"; MUTED = "#6B7178"; LIGHT = "#D9DDE2"
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.3, "xtick.labelsize": 6.6,
                     "ytick.labelsize": 6.6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "pdf.fonttype": 42, "axes.grid": True,
                     "grid.color": "#EEF0F2", "grid.linewidth": 0.6, "mathtext.default": "regular"})
RNG = np.random.default_rng(42)


def label(ax, s):
    t = ax.get_title(loc="center"); ax.set_title(""); ax.set_title(r"$\bf{" + s + r"}$   " + t, loc="left", fontsize=7.4, pad=3)


GENE_COL = {"APP": "#4C78A8", "PSEN1": "#B279A2", "PSEN2": "#72B7B2", "other": "#B0B6BC"}


def binned(ax, x, y, color, nbins=8, label_=None, points=True, ls="-", marker="o", genes=None):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    if points and genes is not None:
        gg = np.asarray(genes)[ok]; key = np.where(np.isin(gg, ["APP", "PSEN1", "PSEN2"]), gg, "other")
        for g_ in ["other", "PSEN2", "PSEN1", "APP"]:
            m = key == g_
            if m.any():
                ax.scatter(x[ok][m] + RNG.normal(0, 0.012, m.sum()), y[ok][m], s=2.6, color=GENE_COL[g_], alpha=0.35, lw=0, zorder=1, rasterized=True, label=g_)
    x, y = x[ok], y[ok]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    xs, ms, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x <= b) if b == edges[-1] else (x >= a) & (x < b)
        if sel.sum() < 3:
            continue
        yy = y[sel]; boots = [RNG.choice(yy, len(yy), replace=True).mean() for _ in range(300)]
        xs.append(x[sel].mean()); ms.append(yy.mean()); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
    ax.plot(xs, ms, ls, color=color, lw=1.4, marker=marker, ms=3.0, zorder=3, label=label_); ax.fill_between(xs, lo, hi, color=color, alpha=0.16, lw=0, zorder=2)


audit = pd.read_csv(T / "hetionet_path_audit.csv"); dist = pd.read_csv(T / "hetionet_distributions.csv"); fid = pd.read_csv(T / "hetionet_sampler_fidelity.csv")
summ = pd.read_csv(T / "hetionet_summary.csv").set_index("model"); lp = pd.read_csv(T / "hetionet_linkpred_summary.csv").set_index("model")
lps = pd.read_csv(T / "hetionet_linkpred_by_stratum.csv"); res = json.load(open(T / "hetionet_results.json"))

fig, ax = plt.subplots(1, 3, figsize=(5.5, 2.35)); a, b, c = ax.ravel()
# (a) support audit
au = audit[audit.candidate_paths > 0].sort_values("reachable_endpoints")
a.barh(au.gene_symbol, au.reachable_endpoints, color=TEAL, height=0.7); a.axvline(53, color=ORANGE, ls="--", lw=0.9); a.text(a.get_xlim()[1] * 0.98 if False else 250, 1.0, "AlzKB: 53\nfor every gene", color=ORANGE, fontsize=5.4, va="bottom", ha="left")
a.set_xlabel("Reachable symptom endpoints"); a.set_title("Gene-dependent support"); a.grid(axis="y", visible=False); a.tick_params(axis="y", labelsize=5.6); label(a, "a")
# (b) entropy vs H
for model, col, nm, ls, pts in [("full", NAVY, "full θ(H)", "-", True), ("fixed_temperature", ORANGE, "fixed θ", "--", False), ("ppr", TEAL, "PPR", ":", False), ("uniform_paths", GREY, "uniform", ":", False)]:
    g = dist[dist.model == model]; binned(b, g.isotonic_entropy, g.entropy_bits, col, label_=nm, points=pts, ls=ls, genes=g.gene_symbol)
b.set_xlabel("Upstream uncertainty H"); b.set_ylabel("Endpoint entropy (bits)"); b.set_xlim(-0.03, 1.03)
b.set_title(f"Coupling: ρ={summ.loc['full'].rho_H_entropy:.2f} vs fixed θ {summ.loc['fixed_temperature'].rho_H_entropy:.2f}"); hb_, lb_ = b.get_legend_handles_labels(); gene_handles = {l_: h_ for h_, l_ in zip(hb_, lb_) if l_ in GENE_COL}
b.legend([h_ for h_, l_ in zip(hb_, lb_) if l_ not in GENE_COL], [l_ for l_ in lb_ if l_ not in GENE_COL], frameon=False, loc="lower right", fontsize=5.6); label(b, "b")
# (c) sampler fidelity vs state-space size
c.scatter(fid.n_states, fid.js_grouped, s=6, color=NAVY, alpha=0.5, lw=0, label=f"grouped, mean {fid.js_grouped.mean():.4f}"); c.scatter(fid.n_states, fid.js_endpoint, s=6, color=GOLD, alpha=0.5, lw=0, label=f"endpoint, mean {fid.js_endpoint.mean():.3f}")
c.set_xscale("log"); c.set_yscale("log"); c.set_ylim(max(1e-4, 0.7 * min(fid.js_grouped.min(), fid.js_endpoint.min())), 1.5 * max(fid.js_grouped.max(), fid.js_endpoint.max()))
c.set_xlabel("Candidate paths per query"); c.set_ylabel("JS(chain, exact)"); c.set_title("Sampler fidelity"); c.legend(frameon=False, fontsize=5.2, loc="lower left", handletextpad=0.2); label(c, "c")
fig.tight_layout(w_pad=1.2, rect=(0, 0.09, 1, 1))
order = ["APP", "PSEN1", "PSEN2", "other"]; gf = dist[dist.model == "full"]
fig.legend([gene_handles[g] for g in order if g in gene_handles], [f"{g} ({int((gf.gene_symbol == g).sum())})" if g != "other" else f"other genes ({int((~gf.gene_symbol.isin(['APP','PSEN1','PSEN2'])).sum())})" for g in order if g in gene_handles],
           loc="lower center", ncol=4, frameon=False, fontsize=6.0, markerscale=2.4, handletextpad=0.3, columnspacing=1.2, title="Points in (b): individual queries, coloured by gene", title_fontsize=6.0, bbox_to_anchor=(0.5, -0.005))
fig.savefig(FIGDIR / "fig6_hetionet.pdf", bbox_inches="tight"); print("saved fig6_hetionet.pdf")
