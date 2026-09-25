#!/usr/bin/env python3
"""Fig 7: worked masked-edge recovery example (APP -> Aphasia on Hetionet G').
All content from outputs/corrected/tables/hetionet_linkpred_example.json and
hetionet_linkpred_example_all_pairs.csv (written by run_12c_example.py)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]).resolve(); FIGDIR = Path(sys.argv[2]).resolve(); T = ROOT / "outputs" / "corrected" / "tables"
NAVY = "#1F5066"; ORANGE = "#E0851E"; TEAL = "#3C7F79"; GOLD = "#CF9A2B"; PURPLE = "#7E3F98"; GREY = "#8B9096"; INK = "#242830"; MUTED = "#6B7178"; LIGHT = "#D9DDE2"; RED = "#B8352E"
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.3, "xtick.labelsize": 6.6,
                     "ytick.labelsize": 6.6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "pdf.fonttype": 42, "mathtext.default": "regular"})
ex = json.load(open(T / "hetionet_linkpred_example.json")); allp = pd.read_csv(T / "hetionet_linkpred_example_all_pairs.csv")
gene = ex["gene"]; target = ex["target_symptom"]; routes = ex["alternative_routes"]; rk = ex["rankings"]
mine = allp[allp.gene == gene].sort_values("rank_full"); ncand = ex["n_candidates_filtered"]


def label(ax, s, t):
    ax.set_title(r"$\bf{" + s + r"}$   " + t, loc="left", fontsize=7.4, pad=3)


fig = plt.figure(figsize=(5.5, 4.5)); gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.05], width_ratios=[1.35, 0.5, 0.9], wspace=0.55, hspace=0.12)
a = fig.add_subplot(gs[0, :]); b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1]); d = fig.add_subplot(gs[1, 2])

# ---------------- (a) schematic of the protocol on G'
a.set_xlim(0, 12); a.set_ylim(0.4, 10.4); a.axis("off"); a.set_aspect("auto")
def node(x, y, text, color, w=2.0, h=0.8, fs=6.0, fc="white", ls="-"):
    a.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=color, lw=0.9, ls=ls))
    a.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK)
def arrow(p0, p1, color, ls="-", lw=0.9, txt=None, tpos=0.5, toff=(0, 0.25), tcol=None, fs=5.3):
    a.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, ls=ls, shrinkA=1, shrinkB=1))
    if txt:
        a.text(p0[0] + tpos * (p1[0] - p0[0]) + toff[0], p0[1] + tpos * (p1[1] - p0[1]) + toff[1], txt, fontsize=fs, color=tcol or color, ha="center", va="center")
GX, RX, DX, SX = 1.35, 4.6, 7.6, 10.7
node(GX, 5.0, f"{gene}\np={ex['p']:.2f}, H={ex['H']:.2f}", NAVY, w=2.3, h=1.0, fc="#EEF3F6")
# own disease (top row), hidden edge, one kept symptom
node(DX, 8.9, "Alzheimer's disease", TEAL, w=2.6, h=0.62, fs=5.8)
arrow((GX + 1.15, 5.45), (DX - 1.3, 8.75), TEAL, txt="associates", tpos=0.4, toff=(-0.55, 0.15), tcol=MUTED)
node(SX, 9.55, "Amnesia", GREY, w=1.6, h=0.55, fs=5.6, fc="#F3F4F5")
arrow((DX + 1.3, 9.0), (SX - 0.8, 9.5), GREY, txt="kept edge → known → filtered out", tpos=0.5, toff=(0.85, -0.5), tcol=MUTED, fs=5.0)
node(SX, 5.0, f"{target}", RED, w=1.6, h=0.62, fs=6.2, fc="#FBEFEE"); a.text(SX, 4.35, "held-out positive", fontsize=5.4, color=RED, ha="center", va="center")
arrow((DX + 0.6, 8.55), (SX - 0.3, 5.35), RED, ls=(0, (2.2, 1.6)), txt="masked edge\n(absent from G′)", tpos=0.42, toff=(0.95, 0.1))
# relay routes (real, from JSON)
ys = [6.8, 5.0, 3.2]
for (r, y) in zip(routes, ys):
    _, relay, dis, _ = r["path"]
    node(RX, y, relay, GOLD, w=1.15, h=0.55, fs=5.8); node(DX, y, dis, TEAL, w=2.2, h=0.55, fs=5.6)
    arrow((GX + 1.15, 5.0 + (y - 5.0) * 0.3), (RX - 0.58, y), GOLD)
    arrow((RX + 0.58, y), (DX - 1.1, y), TEAL)
    arrow((DX + 1.1, y), (SX - 0.8, 5.0 + (y - 5.0) * 0.3), TEAL)
    a.text(RX, y - 0.52, f"P(π)={r['prob']:.4f}", fontsize=5.0, color=MUTED, ha="center", va="center")
a.text(RX - 0.6, 2.05, "relay routes on G′ (gene→gene→disease→symptom); m(s)=ΣP(π)", fontsize=5.4, color=GOLD, ha="left")
a.text(0.0, 1.35, f"G′ masks 671 of 3,357 disease→symptom edges (seed 42); {ncand} filtered candidates for {gene};", fontsize=5.3, color=MUTED, ha="left")
a.text(0.0, 0.75, "the query never sees the masked edge; its symptom is reachable only by relay.", fontsize=5.3, color=MUTED, ha="left")
label(a, "a", "Masked-edge recovery protocol: one query on G′")

# ---------------- (b) filtered ranking, top 10 under full theta(H)
top = rk["full"]["top10"]; names = [t["symptom"] for t in top]; vals = np.array([t["score"] for t in top]); pos = [t["held_out_positive"] for t in top]
y = np.arange(len(top)); b.barh(y, vals * 1e3, color=[RED if p else LIGHT for p in pos], height=0.72, edgecolor="none")
b.set_yticks(y); b.set_yticklabels([n if len(n) <= 22 else n[:20] + "…" for n in names], fontsize=5.6); b.invert_yaxis()
b.set_xlabel("endpoint mass m(s) ×10³  (full θ(H))"); b.grid(axis="x", color="#EEF0F2", lw=0.6); b.set_axisbelow(True)
for yi, p, v in zip(y, pos, vals):
    if p:
        b.text(v * 1e3 + 0.04, yi, "held-out ✓", fontsize=5.4, color=RED, va="center")
b.text(0.98, 0.02, f"rank of {target}:\nfull θ(H) {rk['full']['rank_of_target']}   fixed θ {rk['fixed_temperature']['rank_of_target']}\nDistMult {rk['distmult']['rank_of_target']}  (of {ncand})",
       transform=b.transAxes, fontsize=5.6, ha="right", va="bottom", color=INK, bbox=dict(fc="white", ec=LIGHT, lw=0.6, boxstyle="round,pad=0.25"))
label(b, "b", "Filtered ranking (top 10)")

# ---------------- (c) ranks of all held-out positives of this query
r = mine.rank_full.to_numpy(); c.scatter(np.zeros(len(r)) + 0.5 + np.linspace(-0.12, 0.12, len(r)), r, s=14, color=RED, zorder=3, lw=0)
for k, col in [(10, NAVY), (25, GOLD)]:
    c.axhline(k, color=col, lw=0.8, ls="--"); c.text(0.97, k, f"@{k}", fontsize=5.6, color=col, va="bottom", ha="right")
c.set_ylim(ncand + 5, 0); c.set_xlim(0, 1); c.set_xticks([]); c.set_ylabel(f"rank among {ncand} candidates")
c.text(0.5, 60, f"{len(r)} held-out\npositives:\n{(r <= 10).sum()} in top-10\n{(r <= 25).sum()} in top-25", fontsize=5.6, ha="center", va="top", color=INK)
c.spines["bottom"].set_visible(False); label(c, "c", f"All positives, {gene}")

d.axis("off")
steps = [("1", "Mask 20% of Hetionet disease→symptom edges → G′ (671 hidden)."),
         ("2", "On G′: enumerate routes; retrain energy U = wᵀφ on the same 121 anchors."),
         ("3", "Query (g, p, H): P(π) = softmax(U/θ(H)); m(s) = sum of P(π) over paths ending at s."),
         ("4", "Filter out symptoms still linked to g's diseases on G′; rank the rest by m(s)."),
         ("5", "Positives = hidden-edge symptoms still reachable on G′. Report recall@10/25, MRR with gene-cluster CIs; same candidates for fixed θ, DistMult, PPR, uniform.")]
yy = 0.98
for k, t in steps:
    d.text(0.0, yy, k, transform=d.transAxes, fontsize=6.2, color=NAVY, fontweight="bold", va="top")
    d.text(0.12, yy, t, transform=d.transAxes, fontsize=5.5, color=INK, va="top", wrap=True)
    yy -= 0.19
label(d, "d", "Protocol")
fig.savefig(FIGDIR / "fig7_recovery_example.pdf", bbox_inches="tight"); print("saved fig7_recovery_example.pdf")
