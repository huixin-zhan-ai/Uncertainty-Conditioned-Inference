#!/usr/bin/env python3
"""Shared Nature / npj-style plotting helpers.

Centralizes typography, palette, panel labels, and box+strip rendering so every
figure in the project has a consistent, journal-quality look.
"""
from __future__ import annotations

from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Palette (cohesive, print-friendly). VUS is the focal amber accent.
# ---------------------------------------------------------------------------
CATEGORY_ORDER = [
    "Pathogenic",
    "Likely_pathogenic",
    "Uncertain_significance",
    "Likely_benign",
    "Benign",
    "Conflicting_interpretations",
]
CATEGORY_LABELS = {
    "Pathogenic": "Pathogenic",
    "Likely_pathogenic": "Likely\npathogenic",
    "Uncertain_significance": "VUS",
    "Likely_benign": "Likely\nbenign",
    "Benign": "Benign",
    "Conflicting_interpretations": "Conflicting",
}
CATEGORY_COLORS = {
    "Pathogenic": "#365F70",           # deep blue
    "Likely_pathogenic": "#7093A3",    # mid blue
    "Uncertain_significance": "#D1A66D",  # amber (focal)
    "Likely_benign": "#90B7AF",        # sage
    "Benign": "#5F9079",               # green
    "Conflicting_interpretations": "#A987B7",  # purple
}

# Accent colors for lines / scatters
ACCENT_BLUE = "#365F70"
ACCENT_AMBER = "#D1A66D"
ACCENT_PURPLE = "#806F96"
ACCENT_GREEN = "#5F9079"
ACCENT_SLATE = "#365F70"
INK = "#4D5961"
MUTED = "#6F7880"
GRID = "#DCE1E4"


def _sans() -> List[str]:
    return ["Helvetica", "Arial", "DejaVu Sans"]


def apply() -> None:
    """Apply the shared journal style to matplotlib rcParams."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _sans(),
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "regular",
            "axes.titlepad": 8.0,
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "grid.color": GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def present_categories(values: Iterable[str]) -> List[str]:
    present = set(values)
    ordered = [c for c in CATEGORY_ORDER if c in present]
    extras = sorted(present - set(CATEGORY_ORDER))
    return ordered + extras


def label_for(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, cat.replace("_", " "))


def color_for(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, MUTED)


def soft_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.15,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def _jitter(n: int, center: float, width: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + n + int(center * 100))
    return center + rng.uniform(-width, width, size=n)


def box_strip(
    ax,
    df: pd.DataFrame,
    categories: List[str],
    value_col: str,
    ylabel: str,
    title: str,
    seed: int = 42,
    title_size: float = 10.5,
) -> None:
    """Refined box + jittered-strip plot with a mean marker."""
    data = [df.loc[df["clinical_significance"] == c, value_col].dropna().to_numpy() for c in categories]
    positions = np.arange(len(categories))
    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.56,
        showfliers=False,
        showcaps=False,
        medianprops={"color": INK, "linewidth": 1.3},
        whiskerprops={"color": MUTED, "linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )
    for patch, cat in zip(bp["boxes"], categories):
        patch.set_facecolor(color_for(cat))
        patch.set_alpha(0.22)
        patch.set_edgecolor(color_for(cat))
        patch.set_linewidth(1.2)
    for idx, (cat, vals) in enumerate(zip(categories, data)):
        if len(vals) == 0:
            continue
        ax.scatter(
            _jitter(len(vals), idx, 0.11, seed),
            vals,
            s=9,
            color=color_for(cat),
            alpha=0.30,
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            [idx],
            [vals.mean()],
            marker="D",
            s=34,
            color=color_for(cat),
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([label_for(c) for c in categories])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=title_size)
    soft_grid(ax, "y")


def save(fig, out_dir, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
