#!/usr/bin/env python3
"""Synthetic state-space check for sampler correctness and scaling."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import corrected_core as core
import pubstyle as pub

OUT = core.ROOT / "outputs" / "corrected"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
for directory in (TABLES, FIGURES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)


def run():
    rows = []
    state_sizes = [53, 256, 1024]
    uncertainty_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    for n_states in state_sizes:
        for replicate in range(30):
            rng = np.random.default_rng(
                core.RNG_SEED + 10000 * n_states + replicate
            )
            # Ten endpoint groups with heterogeneous latent route preferences.
            groups = np.arange(n_states) % 10
            group_effect = rng.normal(0, 0.65, 10)
            route_effect = rng.normal(0, 0.70, n_states)
            utilities = group_effect[groups] + route_effect
            for uncertainty in uncertainty_levels:
                exact = core.exact_path_probabilities(utilities, uncertainty)
                exact_groups = core.aggregate_distribution(
                    groups.astype(str), exact
                )
                final, acceptance, steps = core.metropolis_sample(
                    utilities,
                    uncertainty,
                    n_chains=512,
                    seed=core.RNG_SEED + replicate,
                )
                empirical = core.aggregate_distribution(
                    groups[final].astype(str),
                    np.ones(len(final)) / len(final),
                )
                fixed = core.exact_path_probabilities(
                    utilities, uncertainty, fixed_temperature=True
                )
                fixed_groups = core.aggregate_distribution(
                    groups.astype(str), fixed
                )
                rows.append(
                    {
                        "n_states": n_states,
                        "replicate": replicate,
                        "uncertainty": uncertainty,
                        "steps": steps,
                        "acceptance_rate": acceptance,
                        "unique_final_states": len(set(final.tolist())),
                        "js_empirical_to_exact": core.js_divergence(
                            empirical, exact_groups
                        ),
                        "exact_group_entropy": core.entropy_from_prob(
                            exact_groups.values()
                        ),
                        "fixed_group_entropy": core.entropy_from_prob(
                            fixed_groups.values()
                        ),
                        "uncertainty_effect_vs_fixed_js": core.js_divergence(
                            exact_groups, fixed_groups
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "synthetic_generality.csv", index=False)
    summary = frame.groupby(["n_states", "uncertainty"]).agg(
        acceptance_mean=("acceptance_rate", "mean"),
        sampler_js_mean=("js_empirical_to_exact", "mean"),
        sampler_js_p95=("js_empirical_to_exact", lambda x: np.percentile(x, 95)),
        unique_states_mean=("unique_final_states", "mean"),
        exact_entropy_mean=("exact_group_entropy", "mean"),
    ).reset_index()
    summary.to_csv(TABLES / "synthetic_generality_summary.csv", index=False)

    pub.apply()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    palette = [pub.ACCENT_BLUE, pub.ACCENT_GREEN, pub.ACCENT_PURPLE]
    for size, color in zip(state_sizes, palette):
        subset = summary[summary.n_states == size]
        axes[0].plot(subset.uncertainty, subset.sampler_js_mean, marker="o", color=color, label=f"{size} states")
        axes[1].plot(subset.uncertainty, subset.acceptance_mean, marker="o", color=color)
        axes[2].plot(subset.uncertainty, subset.exact_entropy_mean, marker="o", color=color)
    axes[0].set(xlabel="Uncertainty H", ylabel="Mean JS(empirical, exact)", title="Sampler fidelity")
    axes[1].set(xlabel="Uncertainty H", ylabel="Mean acceptance rate", title="Exploration increases with H")
    axes[2].set(xlabel="Uncertainty H", ylabel="Target group entropy (bits)", title="Target broadening")
    axes[0].legend()
    for label, ax in zip("abc", axes):
        pub.panel_label(ax, label)
        pub.soft_grid(ax, "both")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure8_synthetic_generality.png", dpi=400, facecolor="white")
    fig.savefig(FIGURES / "figure8_synthetic_generality.pdf", facecolor="white")
    plt.close(fig)

    correlations = {
        str(size): {
            "H_vs_acceptance": float(
                spearmanr(
                    frame.loc[frame.n_states == size, "uncertainty"],
                    frame.loc[frame.n_states == size, "acceptance_rate"],
                ).statistic
            ),
            "H_vs_target_entropy": float(
                spearmanr(
                    frame.loc[frame.n_states == size, "uncertainty"],
                    frame.loc[frame.n_states == size, "exact_group_entropy"],
                ).statistic
            ),
        }
        for size in state_sizes
    }
    report = {
        "design": {
            "state_sizes": state_sizes,
            "uncertainty_levels": uncertainty_levels,
            "replicates": 30,
            "chains_per_condition": 512,
            "groups": 10,
            "proposal": "90% learned proposal at twice target temperature plus 10% uniform, with Hastings correction",
        },
        "correlations": correlations,
        "maximum_mean_sampler_js": float(summary.sampler_js_mean.max()),
        "maximum_p95_sampler_js": float(summary.sampler_js_p95.max()),
    }
    (REPORTS / "synthetic_generality.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
