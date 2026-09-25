#!/usr/bin/env python3
"""C3 figure: does the propagated distribution recover associations absent from its input graph?
(a) held-out recovery (H2)  (b) matched controls (H2-B)  (c) gene-held-out across masks (H2-C)  (d) paired differences.
All from outputs/corrected/tables/hetionet_*.csv/json. Also writes the appendix panel recall-vs-H (old Fig 5e)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]).resolve(); FIGDIR = Path(sys.argv[2]).resolve(); T = ROOT / "outputs" / "corrected" / "tables"
NAVY = "#1F5066"; ORANGE = "#E0851E"; TEAL = "#3C7F79"; GOLD = "#CF9A2B"; PURPLE = "#7E3F98"; GREY = "#8B9096"; INK = "#242830"; MUTED = "#6B7178"; LIGHT = "#D9DDE2"; RED = "#B8352E"
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.3, "xtick.labelsize": 6.6,
                     "ytick.labelsize": 6.6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "pdf.fonttype": 42, "axes.grid": True,
                     "grid.color": "#EEF0F2", "grid.linewidth": 0.6, "mathtext.default": "regular"})


def label(ax, s, t):
    ax.set_title(r"$\bf{" + s + r"}$   " + t, loc="left", fontsize=7.4, pad=3)


lp = pd.read_csv(T / "hetionet_linkpred_summary.csv").set_index("model"); lpr = json.load(open(T / "hetionet_linkpred_results.json"))
cs = pd.read_csv(T / "hetionet_controls_summary.csv").set_index("model"); cr = json.load(open(T / "hetionet_controls_results.json"))
gh = pd.read_csv(T / "hetionet_geneholdout_pooled.csv").set_index("model"); ghs = pd.read_csv(T / "hetionet_geneholdout_summary.csv"); ghr = json.load(open(T / "hetionet_geneholdout_results.json"))
lps = pd.read_csv(T / "hetionet_linkpred_by_stratum.csv")

fig, ax = plt.subplots(2, 2, figsize=(5.5, 4.7)); a, b, c, d = ax.ravel()

# ---------------- (a) held-out recovery, H2 (kept from former Fig 5d)
order = ["full", "fixed_temperature", "distmult", "beam_5", "uniform_paths", "ppr"]; names = ["Full θ(H)", "Fixed θ", "DistMult (trained)", "Beam-5", "Uniform", "PPR"]
for i, (m, nm) in enumerate(zip(order, names)):
    r = lp.loc[m]
    a.plot([r["recall@25_lo"], r["recall@25_hi"]], [i + 0.15, i + 0.15], color=GOLD, lw=1.2); a.plot(r["recall@25"], i + 0.15, "s", color=GOLD, ms=3.4, label="recall@25" if i == 0 else None)
    a.plot([r["recall@10_lo"], r["recall@10_hi"]], [i - 0.15, i - 0.15], color=NAVY, lw=1.2); a.plot(r["recall@10"], i - 0.15, "o", color=NAVY, ms=3.4, label="recall@10" if i == 0 else None)
a.set_yticks(range(len(order))); a.set_yticklabels(names, fontsize=6); a.invert_yaxis(); a.set_xlim(0, 0.7); a.set_xlabel("Held-out symptom recall (gene-cluster 95% CI)")
a.legend(frameon=False, fontsize=5.6, loc="lower right"); a.grid(axis="y", visible=False); label(a, "a", "Held-out recovery (964 queries, 15 genes)")

# ---------------- (b) matched controls, H2-B
conds = ["full", "degree_perm", "popularity", "random"]; cn = ["Full θ(H)", "Degree-\nmatched perm.", "Popularity\n(G′ degree)", "Random"]
cols = {"full": NAVY, "degree_perm": PURPLE, "popularity": TEAL, "random": GREY}
x = np.arange(len(conds)); w = 0.25
for j, k in enumerate([5, 10, 25]):
    vals = [cs.loc[m, f"recall@{k}"] for m in conds]
    bars = b.bar(x + (j - 1) * w, vals, width=w, color=[cols[m] for m in conds], alpha=[0.45, 0.8, 1.0][j], edgecolor="none", label=f"recall@{k}")
    if k == 10:
        for i, m in enumerate(conds):
            b.plot([x[i], x[i]], [cs.loc[m, "recall@10_lo"], cs.loc[m, "recall@10_hi"]], color=INK, lw=0.8)
b.set_xticks(x); b.set_xticklabels(cn, fontsize=5.8); b.set_ylabel("recall"); b.set_ylim(0, 0.52)
pv = cr["degree_perm_reference"]["p_value_upper_bound"]; b.text(0.98, 0.80, f"perm. p < {pv:.3f}", transform=b.transAxes, fontsize=5.6, ha="right", va="top", color=PURPLE)
hb = [plt.Rectangle((0, 0), 1, 1, color=INK, alpha=al) for al in [0.35, 0.65, 1.0]]; b.legend(hb, ["recall@5", "recall@10", "recall@25"], frameon=False, fontsize=5.6, loc="upper center", ncol=3, handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))
b.grid(axis="x", visible=False); label(b, "b", "Matched controls (H2 hold-out)")

# ---------------- (c) gene-held-out across masks, H2-C
ordc = ["full", "fixed_temperature", "distmult", "uniform_paths", "ppr", "popularity", "random"]; nc = ["Full θ(H)", "Fixed θ", "DistMult", "Uniform", "PPR", "Popularity", "Random"]
colc = {"full": NAVY, "fixed_temperature": ORANGE, "distmult": PURPLE, "uniform_paths": GREY, "ppr": TEAL, "popularity": TEAL, "random": GREY}
mk = {42: "o", 7: "^", 123: "s"}
for i, m in enumerate(ordc):
    r = gh.loc[m]
    c.plot([i, i], [r["recall@10_lo"], r["recall@10_hi"]], color=colc[m], lw=1.4, alpha=0.9); c.plot(i, r["recall@10"], "_", color=colc[m], ms=9, mew=1.6)
    for seed in (42, 7, 123):
        v = ghs[(ghs.model == m) & (ghs.seed == seed)]["recall@10_gene_balanced"].iloc[0]
        c.plot(i + 0.28, v, mk[seed], color=colc[m], ms=3.2, mfc="white", mew=0.8, label=f"seed {seed}" if i == 0 else None)
c.set_xticks(range(len(ordc))); c.set_xticklabels(nc, fontsize=5.8, rotation=25, ha="right"); c.set_ylabel("recall@10, held-out genes"); c.set_ylim(0, 0.5)
c.legend(frameon=False, fontsize=5.4, loc="upper right", title="gene-balanced, per mask", title_fontsize=5.4, ncol=3, columnspacing=0.6, handletextpad=0.2)
c.grid(axis="x", visible=False); label(c, "c", "Gene-held-out recovery across masks")

# ---------------- (d) paired differences, recall@10
items = []
pr = lpr["paired_recall10"]
for k, nm in [("full_minus_fixed_temperature", "vs fixed θ"), ("full_minus_distmult", "vs DistMult"), ("full_minus_beam_5", "vs beam-5"), ("full_minus_uniform_paths", "vs uniform"), ("full_minus_ppr", "vs PPR")]:
    items.append(("H2  " + nm, pr[k], NAVY, "H2"))
for k, nm in [("full_minus_degree_perm|recall@10", "vs degree-matched perm."), ("full_minus_popularity|recall@10", "vs popularity"), ("full_minus_random|recall@10", "vs random")]:
    items.append(("ctrl  " + nm, cr["paired"][k], PURPLE, "controls"))
for k, nm in [("full_minus_fixed_temperature|recall@10", "vs fixed θ"), ("full_minus_distmult|recall@10", "vs DistMult"), ("full_minus_popularity|recall@10", "vs popularity"), ("full_minus_random|recall@10", "vs random")]:
    items.append(("GHO  " + nm, ghr["paired"][k], TEAL, "gene-held-out"))
for i, (nm, v, col, grp) in enumerate(items):
    d.plot([v["lo"], v["hi"]], [i, i], color=col, lw=1.3); d.plot(v["mean"], i, "o", color=col, ms=3.4)
d.axvline(0, color=LIGHT, lw=1); d.set_yticks(range(len(items))); d.set_yticklabels([it[0] for it in items], fontsize=5.6); d.invert_yaxis(); d.set_xlabel("Δ recall@10 (full − other), gene-cluster 95% CI")
for tl, (nm, v, col, grp) in zip(d.get_yticklabels(), items):
    tl.set_color(col)
d.axhspan(4.5, 7.5, color="#F6F7F8", zorder=0); d.grid(axis="y", visible=False); label(d, "d", "Paired differences")
fig.tight_layout(w_pad=1.4, h_pad=1.2); fig.savefig(FIGDIR / "fig_c3_recovery.pdf"); print("saved fig_c3_recovery.pdf")

# ---------------- appendix: recall by uncertainty stratum (former Fig 5e), descriptive only
fig2, e = plt.subplots(figsize=(2.6, 1.9)); x = np.arange(4); labels = ["Q1 low", "Q2", "Q3", "Q4 high"]
for m, col, nm, off in [("full", NAVY, "full θ(H)", -0.25), ("fixed_temperature", ORANGE, "fixed θ", 0.0), ("distmult", PURPLE, "DistMult", 0.25)]:
    s = lps[lps.model == m].set_index("stratum").reindex(labels); e.bar(x + off, s.recall10, width=0.25, color=col, label=nm)
e.set_xticks(x); e.set_xticklabels(labels, fontsize=6); e.set_ylabel("recall@10"); e.set_title("Recall by uncertainty stratum (descriptive)", fontsize=7); e.legend(frameon=False, fontsize=5.6); e.grid(axis="x", visible=False)
fig2.tight_layout(); fig2.savefig(FIGDIR / "figA_recall_by_H.pdf"); print("saved figA_recall_by_H.pdf")
