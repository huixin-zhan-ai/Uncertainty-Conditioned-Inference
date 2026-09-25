#!/usr/bin/env python3
"""Figure 0 - framework overview (AlzKB diffusion-uncertainty study).
Panel a embeds the real gene->phenotype mechanism illustration (mech_panel.png);
panels b-f are a clean, borderless schematic of the pipeline. Nature/npj style.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "Nimbus Sans", "font.size": 8,
    "mathtext.default": "regular", "pdf.fonttype": 42, "ps.fonttype": 42,
})

NAVY="#1f5066"; ORANGE="#e0851e"; TEAL="#3c7f79"; GREEN="#2e8b6f"
GOLD="#cf9a2b"; GREY="#8b9096"; INK="#242830"; MUTED="#6b7178"
SOFT="#eef1f4"; LINE="#565b62"

fig = plt.figure(figsize=(7.6, 9.7), dpi=400)

from pathlib import Path as _P
img = plt.imread(str(_P(__file__).resolve().parent / "assets" / "mech_panel.png"))
ax_img = fig.add_axes([0.045, 0.605, 0.915, 0.360])
ax_img.imshow(img); ax_img.axis("off")

ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def letter(x, y, s, title=None):
    ax.text(x, y, s, weight="bold", fontsize=11.5, color=INK, ha="left", va="top", zorder=6)
    if title:
        ax.text(x+3.4, y-0.25, title, weight="bold", fontsize=8.0, color=INK,
                ha="left", va="top", zorder=6)

def box(x, y, w, h, text, fc="white", ec=INK, tc=INK, fs=7.0, weight="normal", lw=0.9, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))
    if text:
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", color=tc,
                fontsize=fs, weight=weight, zorder=z+1)

def chip(x, y, w, h, text, fc, tc="white", fs=7.0, weight="bold", z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=0, facecolor=fc, zorder=z))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", color=tc,
            fontsize=fs, weight=weight, zorder=z+1)

def arrow(x1, y1, x2, y2, color=LINE, lw=1.4, ms=9, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=ms, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=3))

letter(4.2, 98.0, "a", "One AD gene reaches many phenotypes (AlzKB substrate)")
ax.add_line(Line2D([4, 96], [59.0, 59.0], color="#e4e6e9", lw=0.9, zorder=2))

# ---- geometry: 2x2 + full-width bottom ----
LX, RX, W = 4.0, 53.0, 43.0
Y1, H1 = 38.0, 17.0     # row 1 (b,c)
Y2, H2 = 18.5, 16.0     # row 2 (d,e)
Y3, H3 = 2.5, 12.0      # row 3 (f)

# ================= b : genomic language model =================
letter(LX, Y1+H1+1.6, "b", "Genomic language model")
x0, y0 = LX, Y1
seg = [NAVY, TEAL, NAVY, TEAL, NAVY, TEAL, NAVY]
def strip(x, y, hi=None):
    for i in range(7):
        c = seg[i]
        if hi == i: c = ORANGE
        ax.add_patch(Rectangle((x+i*1.5, y), 1.3, 1.9, facecolor=c,
                     edgecolor="white", linewidth=0.4, zorder=4))
ax.text(x0+1.0, y0+H1-2.0, "variant window", fontsize=6.0, color=MUTED, ha="left", va="center")
ax.text(x0+1.0, y0+H1-5.0, "WT", fontsize=6.6, weight="bold", color=INK, ha="left", va="center")
strip(x0+6.0, y0+H1-6.0)
ax.text(x0+1.0, y0+H1-8.4, "MUT", fontsize=6.6, weight="bold", color=INK, ha="left", va="center")
strip(x0+6.0, y0+H1-9.4, hi=4)
box(x0+2.0, y0+2.6, 11.0, 4.2, "Evo 2", fc="white", ec=NAVY, tc=NAVY, weight="bold", fs=8.2, lw=1.2)
arrow(x0+13.3, y0+4.7, x0+16.6, y0+4.7, ms=8)
chip(x0+18.0, y0+7.3, 9.0, 4.0, "p", NAVY, fs=8.5)
chip(x0+18.0, y0+2.5, 9.0, 4.0, "H", GOLD, fs=8.5)
ax.text(x0+28.0, y0+9.3, "pathogenicity", fontsize=5.9, color=MUTED, ha="left", va="center")
ax.text(x0+28.0, y0+4.5, "uncertainty", fontsize=5.9, color=MUTED, ha="left", va="center")
ax.text(x0+1.0, y0+0.2, r"$\Delta=\mathrm{score(WT)}-\mathrm{score(MUT)}$", fontsize=6.0, color=MUTED, ha="left")

# ================= c : candidate paths & features =================
letter(RX, Y1+H1+1.6, "c", "Candidate paths & features")
x0, y0 = RX, Y1
gx, gy = x0+4.0, y0+8.5
ax.add_patch(Circle((gx, gy), 1.8, facecolor=NAVY, edgecolor="white", lw=0.7, zorder=4))
ax.text(gx, gy+3.0, "gene", fontsize=5.8, color=INK, ha="center")
mids = [(x0+10, y0+12.5, TEAL), (x0+10, y0+8.5, "#57a08f"), (x0+10, y0+4.5, TEAL)]
ends = [(x0+15.5, y0+13.8, ORANGE), (x0+15.5, y0+10.2, GOLD),
        (x0+15.5, y0+6.6, GREEN), (x0+15.5, y0+3.0, GREY)]
for mx, my, mc in mids:
    ax.add_line(Line2D([gx, mx], [gy, my], color="#c3c7cc", lw=0.8, zorder=2))
    ax.add_patch(Circle((mx, my), 1.3, facecolor=mc, edgecolor="white", lw=0.6, zorder=4))
for i, (ex, ey, ec) in enumerate(ends):
    mx, my, _ = mids[min(i, len(mids)-1)]
    ax.add_line(Line2D([mx, ex], [my, ey], color="#c3c7cc", lw=0.8, zorder=2))
    ax.add_patch(Circle((ex, ey), 1.15, facecolor=ec, edgecolor="white", lw=0.6, zorder=4))
ax.text(gx-0.5, y0+0.3, r"paths $\pi$, depth $\leq 6$", fontsize=5.9, color=MUTED, ha="left")
fx = x0+21.0
ax.text(fx, y0+H1-2.2, r"$\phi(\pi)\in\mathbb{R}^{14}$", fontsize=8.0, weight="bold",
        color=INK, ha="left", va="center")
for i, t in enumerate(["structure, coverage, length",
                       "endpoint specificity $s$",
                       "mechanism affinity $a_g$",
                       "AD-core, generic, crowding",
                       r"$p\cdot,\ (1{-}H)\cdot$ interactions"]):
    ax.text(fx, y0+H1-5.2-i*2.4, t, fontsize=5.9, color=INK, ha="left", va="center")

# ================= d : learned scoring & diffusion =================
letter(LX, Y2+H2+1.6, "d", "Learned scoring & diffusion")
x0, y0 = LX, Y2
ax.text(x0+1.0, y0+H2-2.2, r"$U(\pi)=w^{\top}\tilde\phi(\pi)$", fontsize=8.2, color=INK, ha="left", va="center")
ax.text(x0+1.0, y0+H2-5.6, r"$a_i=\mathrm{softmax}(U/\theta),\ \theta=0.25+1.4H$",
        fontsize=6.8, color=INK, ha="left", va="center")
ax.text(x0+1.0, y0+H2-8.7, r"Metropolis revision, $n=10+\lfloor 15H\rceil$",
        fontsize=6.1, color=MUTED, ha="left", va="center")
cy = y0+1.8
xs = [x0+1.5, x0+9.5, x0+17.5, x0+25.5]
for i, x in enumerate(xs):
    hl = (i == 2)
    box(x, cy, 6.6, 3.9, r"$\pi_{%d}$" % i, fc=(GOLD if hl else "white"),
        ec=(GOLD if hl else NAVY), tc=("white" if hl else INK), fs=7.2, weight="bold")
    if i < 3:
        arrow(x+6.7, cy+2.0, xs[i+1]-0.1, cy+2.0, ms=7, lw=1.1)
ax.text(x0+34.5, cy+2.0, "learned\nend-to-end", fontsize=5.8, color=MUTED, ha="left", va="center")

# ================= e : phenotype distribution =================
letter(RX, Y2+H2+1.6, "e", "Phenotype distribution & coupling")
x0, y0 = RX, Y2
ax.text(x0+1.0, y0+H2-2.4, r"$P(\mathrm{endpoint})=\sum_{i:\,\pi_i\to\mathrm{endpoint}} a_i$",
        fontsize=7.0, color=INK, ha="left", va="center")
def minibars(bx, byy, heights, color, title):
    aw, ah = 14.0, 7.5
    ax.add_line(Line2D([bx, bx], [byy, byy+ah], color=MUTED, lw=0.8))
    ax.add_line(Line2D([bx, bx+aw], [byy, byy], color=MUTED, lw=0.8))
    n = len(heights); bw = aw/(n*1.5)
    for i, h in enumerate(heights):
        ax.add_patch(Rectangle((bx+0.9+i*(aw/n), byy), bw, h*ah, facecolor=color,
                     edgecolor="white", lw=0.4, zorder=4))
    ax.text(bx+aw/2, byy+ah+1.4, title, fontsize=6.1, color=INK, ha="center", weight="bold")
minibars(x0+2.0,  y0+2.2, [0.95,0.20,0.11,0.07,0.05,0.04], NAVY, "low H → sharp")
minibars(x0+22.0, y0+2.2, [0.30,0.26,0.22,0.20,0.18,0.16], GOLD, "high H → broad")

# ================= f : VUS reclassification (full width) =================
letter(LX, Y3+H3+1.6, "f", "Clinical readout: VUS reclassification")
x0, y0 = LX, Y3
chip(x0+2.0, y0+4.6, 16.0, 4.4, "sequence + graph\nfeatures", SOFT, tc=NAVY, fs=6.4)
arrow(x0+18.5, y0+6.8, x0+21.5, y0+6.8, ms=8)
box(x0+22.0, y0+4.6, 14.0, 4.4, "logistic\nmodel", fc="white", ec=INK, fs=6.6)
arrow(x0+36.5, y0+6.8, x0+39.5, y0+6.8, ms=8)
chip(x0+40.0, y0+4.6, 14.0, 4.4, "benign-like",     GREEN,  fs=6.4)
chip(x0+55.0, y0+4.6, 13.0, 4.4, "uncertain",       GREY,   fs=6.4)
chip(x0+69.0, y0+4.6, 16.5, 4.4, "pathogenic-like", ORANGE, fs=6.4)
ax.text(x0+40.0, y0+2.0, "held-out AUROC 0.95 → 0.97; largest gains at high H",
        fontsize=6.2, color=MUTED, ha="left", va="center")

# ================= flow arrows (clean central spine + within-row) =================
arrow(50, 58.3, 50, Y1+H1+0.3, color=LINE, lw=1.7, ms=11)   # a -> method
arrow(LX+W-0.3, Y1+H1/2, RX-0.3, Y1+H1/2, lw=1.5, ms=9)     # b -> c
arrow(50, Y1-0.3, 50, Y2+H2+0.4, color=LINE, lw=1.6, ms=10) # row1 -> row2
arrow(LX+W-0.3, Y2+H2/2, RX-0.3, Y2+H2/2, lw=1.5, ms=9)     # d -> e
arrow(50, Y2-0.3, 50, Y3+H3+0.4, color=LINE, lw=1.6, ms=10) # row2 -> row3

fig.savefig("fig1_framework.pdf", dpi=400, bbox_inches="tight", pad_inches=0.04)
fig.savefig("fig1_framework.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
print("saved")
