#!/usr/bin/env python3
"""Run the corrected real-data study from one locked, leakage-controlled model."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn

import corrected_core as core
import pubstyle as pub

OUT = core.ROOT / "outputs" / "corrected"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
MANIFESTS = OUT / "manifests"
for directory in (TABLES, FIGURES, REPORTS, MANIFESTS):
    directory.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name: str):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=400, facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def stable_seed(variant_id: str) -> int:
    digest = hashlib.sha256(str(variant_id).encode("utf-8")).hexdigest()
    return core.RNG_SEED + int(digest[:8], 16)


def distribution_metrics(distribution):
    probabilities = np.asarray(list(distribution.values()), dtype=float)
    return {
        "phenotype_entropy_bits": core.entropy_from_prob(probabilities),
        "effective_groups": core.effective_support(probabilities),
        "top1_mass": float(probabilities.max()) if len(probabilities) else np.nan,
        "ad_core_mass": float(
            sum(distribution.get(group, 0.0) for group in core.AD_CORE_GROUPS)
        ),
    }


def uncertainty_strata(frame: pd.DataFrame, column: str = "isotonic_entropy"):
    result = frame.copy()
    result["uncertainty_stratum"] = pd.qcut(
        result[column],
        q=4,
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
        duplicates="drop",
    )
    return result


def train_locked_model(graph, variants, relation_frequency):
    training_ids = variants.loc[
        variants["clinical_significance"].isin(
            core.DEFINITIVE_POS | core.DEFINITIVE_NEG
        ),
        "variant_id",
    ].astype(str).tolist()
    samples = core.build_training_samples(
        graph, variants, relation_frequency, training_ids
    )
    scorer = core.train_path_scorer(samples, l2=1e-2, seed=core.RNG_SEED)
    scorer.save()
    core.validate_split_disjointness(scorer, variants)
    return scorer


def path_audit(graph, variants):
    rows = []
    for (gene_id, gene_symbol), group in variants.groupby(["gene_id", "gene_symbol"]):
        paths = core.enumerate_candidate_paths(graph, gene_id)
        route_lengths = [len(path) - 3 for path in paths]
        rows.append(
            {
                "gene_id": gene_id,
                "gene_symbol": gene_symbol,
                "n_variants": len(group),
                "candidate_paths": len(paths),
                "candidate_routes": len({tuple(path[:-2]) for path in paths}),
                "reachable_endpoints": len({path[-1] for path in paths}),
                "mean_gene_hops": float(np.mean(route_lengths)) if paths else np.nan,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["n_variants", "gene_symbol"], ascending=[False, True])
    frame.to_csv(TABLES / "path_audit.csv", index=False)
    return frame


def compute_primary_results(graph, variants, scorer, relation_frequency):
    propagation = []
    ranking = []
    alternative = []
    branching = []
    ablations = []
    diffusion = []
    invalid = []
    ablation_specs = {
        "learned_uncertainty": (False, ()),
        "fixed_temperature": (True, ()),
        "no_confidence_features": (
            False,
            ("confidence_x_specificity", "confidence_x_route_mechanism"),
        ),
        "fixed_theta_no_confidence": (
            True,
            ("confidence_x_specificity", "confidence_x_route_mechanism"),
        ),
        "no_mechanism": (
            False,
            (
                "route_mechanism_affinity",
                "pathogenicity_x_route_mechanism",
                "confidence_x_route_mechanism",
            ),
        ),
    }
    for row in variants.itertuples(index=False):
        common = {
            "variant_id": str(row.variant_id),
            "gene_id": row.gene_id,
            "gene_symbol": row.gene_symbol,
            "clinical_significance": row.clinical_significance,
            "pathogenicity_prob": float(row.pathogenicity_prob),
            "isotonic_entropy": float(row.isotonic_entropy),
        }
        full, ranked = core.exact_distribution(
            graph,
            scorer,
            relation_frequency,
            row.gene_id,
            row.pathogenicity_prob,
            row.isotonic_entropy,
            aggregate_groups=True,
        )
        if not full:
            invalid.append(common)
            continue
        propagation.append({**common, **distribution_metrics(full)})

        low_h = max(0.0, row.isotonic_entropy - 0.30)
        high_h = min(1.0, row.isotonic_entropy + 0.30)
        low, _ = core.exact_distribution(
            graph, scorer, relation_frequency, row.gene_id,
            row.pathogenicity_prob, low_h, aggregate_groups=True,
        )
        high, _ = core.exact_distribution(
            graph, scorer, relation_frequency, row.gene_id,
            row.pathogenicity_prob, high_h, aggregate_groups=True,
        )
        ranking.append(
            {
                **common,
                "js_low": core.js_divergence(full, low),
                "js_high": core.js_divergence(full, high),
                "top1_switch_low": int(core.top_key(full) != core.top_key(low)),
                "top1_switch_high": int(core.top_key(full) != core.top_key(high)),
                "top5_overlap_low": core.topk_overlap(full, low),
                "top5_overlap_high": core.topk_overlap(full, high),
            }
        )

        knn, _ = core.exact_distribution(
            graph, scorer, relation_frequency, row.gene_id,
            row.pathogenicity_prob, row.knn_entropy, aggregate_groups=True,
        )
        alternative.append(
            {
                **common,
                "knn_entropy": float(row.knn_entropy),
                "entropy_gap": abs(float(row.knn_entropy - row.isotonic_entropy)),
                "js_iso_knn": core.js_divergence(full, knn),
                "top1_switch": int(core.top_key(full) != core.top_key(knn)),
                "top5_overlap": core.topk_overlap(full, knn),
            }
        )

        raw, _ = core.exact_distribution(
            graph, scorer, relation_frequency, row.gene_id,
            row.pathogenicity_prob, row.isotonic_entropy, aggregate_groups=False,
        )
        branching.append(
            {
                **common,
                "reachable_endpoints": len(raw),
                "raw_endpoint_entropy_bits": core.entropy_from_prob(raw.values()),
                "effective_raw_endpoints": core.effective_support(raw.values()),
                "raw_top1_mass": max(raw.values()),
            }
        )

        baseline = core.uniform_distribution(graph, row.gene_id, aggregate_groups=True)
        ablations.append(
            {
                **common,
                "model": "uniform",
                **distribution_metrics(baseline),
                "js_to_full": core.js_divergence(full, baseline),
            }
        )
        for name, (fixed_temperature, ablate) in ablation_specs.items():
            distribution, _ = core.exact_distribution(
                graph, scorer, relation_frequency, row.gene_id,
                row.pathogenicity_prob, row.isotonic_entropy,
                aggregate_groups=True,
                fixed_temperature=fixed_temperature,
                ablate=ablate,
            )
            ablations.append(
                {
                    **common,
                    "model": name,
                    **distribution_metrics(distribution),
                    "js_to_full": core.js_divergence(full, distribution),
                }
            )

        empirical, diagnostic = core.diffusion_diagnostics(
            graph, scorer, relation_frequency, row.gene_id,
            row.pathogenicity_prob, row.isotonic_entropy,
            seed=stable_seed(row.variant_id),
        )
        diffusion.append({**common, **diagnostic, **{
            "empirical_entropy_bits": core.entropy_from_prob(empirical.values()),
            "empirical_top1_mass": max(empirical.values()),
        }})

    frames = {
        "propagation": uncertainty_strata(pd.DataFrame(propagation)),
        "ranking": uncertainty_strata(pd.DataFrame(ranking)),
        "alternative_uncertainty": pd.DataFrame(alternative),
        "branching": pd.DataFrame(branching),
        "ablations": pd.DataFrame(ablations),
        "diffusion_diagnostics": pd.DataFrame(diffusion),
        "excluded_variants": pd.DataFrame(invalid),
    }
    frames["ranking"]["mean_js"] = (
        frames["ranking"]["js_low"] + frames["ranking"]["js_high"]
    ) / 2
    frames["ranking"]["mean_top1_switch"] = (
        frames["ranking"]["top1_switch_low"]
        + frames["ranking"]["top1_switch_high"]
    ) / 2
    frames["ranking"]["mean_top5_overlap"] = (
        frames["ranking"]["top5_overlap_low"]
        + frames["ranking"]["top5_overlap_high"]
    ) / 2
    for name, frame in frames.items():
        frame.to_csv(TABLES / f"{name}.csv", index=False)
    return frames


def summarize(frames, scorer, path_frame):
    propagation = frames["propagation"]
    ranking = frames["ranking"]
    alternative = frames["alternative_uncertainty"]
    branching = frames["branching"]
    diffusion = frames["diffusion_diagnostics"]
    prop_summary = propagation.groupby("uncertainty_stratum", observed=False).agg(
        phenotype_entropy_mean=("phenotype_entropy_bits", "mean"),
        effective_groups_mean=("effective_groups", "mean"),
        top1_mass_mean=("top1_mass", "mean"),
        n=("variant_id", "count"),
    ).reset_index()
    rank_summary = ranking.groupby("uncertainty_stratum", observed=False).agg(
        mean_js=("mean_js", "mean"),
        mean_top1_switch=("mean_top1_switch", "mean"),
        mean_top5_overlap=("mean_top5_overlap", "mean"),
        n=("variant_id", "count"),
    ).reset_index()
    alt_class = alternative.groupby("clinical_significance").agg(
        mean_js=("js_iso_knn", "mean"),
        mean_top1_switch=("top1_switch", "mean"),
        mean_top5_overlap=("top5_overlap", "mean"),
        n=("variant_id", "count"),
    ).reset_index()
    alt_gene = alternative.groupby(["gene_id", "gene_symbol"]).agg(
        mean_js=("js_iso_knn", "mean"),
        median_js=("js_iso_knn", "median"),
        n=("variant_id", "count"),
    ).reset_index()
    # Singleton genes are retained but never called the "most sensitive gene".
    alt_gene["eligible_n_ge_5"] = alt_gene["n"] >= 5
    alt_gene = alt_gene.sort_values(
        ["eligible_n_ge_5", "mean_js", "n"], ascending=[False, False, False]
    )
    correlations = {
        "H_vs_group_entropy": float(
            spearmanr(
                propagation["isotonic_entropy"],
                propagation["phenotype_entropy_bits"],
            ).statistic
        ),
        "H_vs_raw_endpoint_entropy": float(
            spearmanr(
                branching["isotonic_entropy"],
                branching["raw_endpoint_entropy_bits"],
            ).statistic
        ),
        "H_vs_unique_final_routes": float(
            spearmanr(
                diffusion["isotonic_entropy"],
                diffusion["unique_final_routes"],
            ).statistic
        ),
        "H_vs_acceptance_rate": float(
            spearmanr(
                diffusion["isotonic_entropy"],
                diffusion["acceptance_rate"],
            ).statistic
        ),
    }
    prop_summary.to_csv(TABLES / "propagation_summary.csv", index=False)
    rank_summary.to_csv(TABLES / "ranking_summary.csv", index=False)
    alt_class.to_csv(TABLES / "alternative_class_summary.csv", index=False)
    alt_gene.to_csv(TABLES / "alternative_gene_summary.csv", index=False)
    summary = {
        "n_variants_loaded": int(
            len(propagation) + len(frames["excluded_variants"])
        ),
        "n_variants_analyzed": int(len(propagation)),
        "n_variants_excluded_no_path": int(len(frames["excluded_variants"])),
        "n_genes": int(path_frame["gene_id"].nunique()),
        "n_path_scorer_training": scorer.n_train,
        "path_scorer_training_classes": ["Pathogenic", "Benign"],
        "locked_validation_classes": ["Likely_pathogenic", "Likely_benign"],
        "correlations": correlations,
        "mean_sampler_js_to_exact": float(diffusion["js_empirical_to_exact"].mean()),
        "median_sampler_js_to_exact": float(diffusion["js_empirical_to_exact"].median()),
        "mean_acceptance_rate": float(diffusion["acceptance_rate"].mean()),
        "propagation_by_stratum": prop_summary.to_dict(orient="records"),
        "ranking_by_stratum": rank_summary.to_dict(orient="records"),
        "alternative_by_class": alt_class.to_dict(orient="records"),
        "alternative_by_gene": alt_gene.to_dict(orient="records"),
    }
    (REPORTS / "corrected_results.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def make_figures(variants, frames, summary, scorer, path_frame):
    pub.apply()
    colors = pub.CATEGORY_COLORS
    order = [category for category in pub.CATEGORY_ORDER if category in set(variants.clinical_significance)]

    # Figure 2: locked sequence signals.
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))
    counts = variants.clinical_significance.value_counts().reindex(order)
    axes[0].bar(range(len(order)), counts, color=[colors[x] for x in order])
    axes[0].set_xticks(range(len(order)), [pub.label_for(x) for x in order])
    axes[0].set_ylabel("Variants")
    axes[0].set_title("Locked cohort")
    pub.panel_label(axes[0], "a")
    pub.box_strip(axes[1], variants, order, "pathogenicity_prob", "Pathogenicity probability", "Definitive-only Platt calibration")
    pub.panel_label(axes[1], "b")
    pub.box_strip(axes[2], variants, order, "isotonic_entropy", "Sequence entropy (bits)", "Definitive-only isotonic uncertainty")
    pub.panel_label(axes[2], "c")
    save_figure(fig, "figure2_sequence_landscape_corrected")

    # Figure 3: actual sampler + propagation.
    diffusion = frames["diffusion_diagnostics"]
    propagation = frames["propagation"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0))
    axes[0, 0].scatter(diffusion.isotonic_entropy, diffusion.js_empirical_to_exact, s=12, alpha=.3, color=pub.ACCENT_BLUE)
    axes[0, 0].set(xlabel="Sequence uncertainty H", ylabel="JS(empirical, exact)", title="Metropolis sampler fidelity")
    pub.panel_label(axes[0, 0], "a")
    axes[0, 1].scatter(diffusion.isotonic_entropy, diffusion.acceptance_rate, s=12, alpha=.3, color=pub.ACCENT_GREEN)
    axes[0, 1].set(xlabel="Sequence uncertainty H", ylabel="Acceptance rate", title="Uncertainty-dependent exploration")
    pub.panel_label(axes[0, 1], "b")
    axes[1, 0].scatter(diffusion.isotonic_entropy, diffusion.unique_final_routes, s=12, alpha=.3, color=pub.ACCENT_PURPLE)
    axes[1, 0].set(xlabel="Sequence uncertainty H", ylabel=f"Unique final routes ({core.N_CHAINS} chains)", title="Learned trajectory diversity")
    pub.panel_label(axes[1, 0], "c")
    axes[1, 1].scatter(propagation.isotonic_entropy, propagation.phenotype_entropy_bits, s=12, alpha=.3, color=pub.ACCENT_AMBER)
    rho = summary["correlations"]["H_vs_group_entropy"]
    axes[1, 1].set(xlabel="Sequence uncertainty H", ylabel="Phenotype-group entropy (bits)", title=f"Target spread (Spearman ρ={rho:.2f})")
    pub.panel_label(axes[1, 1], "d")
    for ax in axes.flat: pub.soft_grid(ax, "both")
    save_figure(fig, "figure3_diffusion_and_propagation_corrected")

    # Figure 4: perturbation/ranking.
    ranking = frames["ranking"]
    rank_summary = pd.read_csv(TABLES / "ranking_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    x = np.arange(len(rank_summary))
    axes[0].plot(x, rank_summary.mean_js, marker="o", color=pub.ACCENT_BLUE)
    axes[0].set_xticks(x, rank_summary.uncertainty_stratum)
    axes[0].set(ylabel="Mean JS divergence", title="Distribution shift under H ± 0.30")
    pub.panel_label(axes[0], "a")
    axes[1].plot(x, rank_summary.mean_top1_switch, marker="o", color=pub.ACCENT_AMBER)
    axes[1].set_xticks(x, rank_summary.uncertainty_stratum)
    axes[1].set(ylabel="Mean top-1 switch rate", title="Top-group instability")
    pub.panel_label(axes[1], "b")
    axes[2].plot(x, rank_summary.mean_top5_overlap, marker="o", color=pub.ACCENT_GREEN)
    axes[2].set_xticks(x, rank_summary.uncertainty_stratum)
    axes[2].set(ylabel="Mean top-5 Jaccard overlap", title="Ranked-set stability")
    pub.panel_label(axes[2], "c")
    for ax in axes: pub.soft_grid(ax, "y")
    save_figure(fig, "figure4_ranking_sensitivity_corrected")

    # Figure 5: alternative uncertainty, with sample-size-aware gene display.
    alternative = frames["alternative_uncertainty"]
    alt_class = pd.read_csv(TABLES / "alternative_class_summary.csv")
    alt_gene = pd.read_csv(TABLES / "alternative_gene_summary.csv")
    eligible = alt_gene[alt_gene.n >= 5].head(8).sort_values("mean_js")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    axes[0].scatter(alternative.isotonic_entropy, alternative.knn_entropy, s=12, alpha=.3, color=pub.ACCENT_BLUE)
    axes[0].plot([0,1],[0,1],"--",color=pub.MUTED,lw=1)
    axes[0].set(xlabel="Isotonic entropy", ylabel="kNN entropy", title="Alternative calibration")
    pub.panel_label(axes[0], "a")
    display_order = [x for x in order if x in set(alt_class.clinical_significance)]
    vals = alt_class.set_index("clinical_significance").reindex(display_order)
    axes[1].bar(range(len(vals)), vals.mean_js, color=[colors[x] for x in display_order])
    axes[1].set_xticks(range(len(vals)), [pub.label_for(x) for x in display_order])
    axes[1].set(ylabel="Mean JS divergence", title="Distribution sensitivity by class")
    pub.panel_label(axes[1], "b")
    axes[2].barh(eligible.gene_symbol, eligible.mean_js, color=pub.ACCENT_PURPLE)
    axes[2].set(xlabel="Mean JS divergence", title="Genes represented by ≥5 variants")
    pub.panel_label(axes[2], "c")
    for ax in axes: pub.soft_grid(ax, "both")
    save_figure(fig, "figure5_alternative_uncertainty_corrected")

    # Figure 6: graph audit + ablations + branching.
    ablations = frames["ablations"]
    branching = frames["branching"]
    model_order = ["uniform", "fixed_theta_no_confidence", "fixed_temperature", "no_confidence_features", "no_mechanism", "learned_uncertainty"]
    means = ablations.groupby("model").agg(
        entropy=("phenotype_entropy_bits","mean"),
        js=("js_to_full","mean"),
    ).reindex(model_order)
    top_paths = path_frame.sort_values("n_variants", ascending=False).head(10).sort_values("candidate_routes")
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
    axes[0].barh(top_paths.gene_symbol, top_paths.candidate_routes, color=pub.ACCENT_BLUE)
    axes[0].set(xlabel="Retained valid routes", title="Non-degenerate route sets")
    pub.panel_label(axes[0], "a")
    axes[1].bar(range(len(means)), means.js, color=[pub.MUTED, pub.ACCENT_SLATE, pub.ACCENT_GREEN, pub.ACCENT_PURPLE, pub.ACCENT_AMBER, pub.ACCENT_BLUE])
    axes[1].set_xticks(range(len(means)), ["Uniform","No H","Fixed θ","No H features","No mechanism","Full"], rotation=30, ha="right")
    axes[1].set(ylabel="Mean JS divergence from full model", title="Baselines and ablations")
    pub.panel_label(axes[1], "b")
    axes[2].scatter(branching.isotonic_entropy, branching.raw_endpoint_entropy_bits, s=12, alpha=.3, color=pub.ACCENT_PURPLE)
    rho = summary["correlations"]["H_vs_raw_endpoint_entropy"]
    axes[2].set(xlabel="Sequence uncertainty H", ylabel="Raw-endpoint entropy (bits)", title=f"Endpoint branching (ρ={rho:.2f})")
    pub.panel_label(axes[2], "c")
    for ax in axes: pub.soft_grid(ax, "both")
    save_figure(fig, "figure6_graph_ablations_corrected")

    # Figure 1: corrected framework (no false pathway/process traversal claim).
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
    panels = [
        ("a", "Sequence signals", "WT / MUT Evo 2 score\nDefinitive-only calibration\n$p$ and $H$"),
        ("b", "Valid graph routes", "$\\mathrm{Gene}\\rightarrow\\mathrm{Gene}$ (0--1 hop)\n$\\rightarrow\\mathrm{Disease}\\rightarrow\\mathrm{Phenotype}$\nProcess/pathway annotations"),
        ("c", "Learned energy", "14 route + endpoint features\n$U(\\pi)=w^{\\mathsf{T}}\\widetilde{\\phi}(\\pi)$\nPathogenic/benign training only"),
        ("d", "Metropolis revision", "Tempered learned proposals\nHastings-corrected acceptance\nExact target validation"),
    ]
    fills = ["#E8F1FA","#E4EEE9","#E4E0F9","#F0ECE7"]
    for ax, (letter,title,text),fill in zip(axes,panels,fills):
        ax.set_facecolor(fill)
        ax.text(.04,.95,letter,transform=ax.transAxes,va="top",fontweight="bold",fontsize=14)
        ax.text(.5,.72,title,transform=ax.transAxes,ha="center",fontweight="bold",fontsize=11,color=pub.INK)
        ax.text(.5,.40,text,transform=ax.transAxes,ha="center",va="center",fontsize=9.5,linespacing=1.6,color=pub.INK)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(wspace=.08)
    fig.savefig(FIGURES / "figure1_framework_corrected.png",dpi=400,facecolor="white",bbox_inches="tight")
    fig.savefig(FIGURES / "figure1_framework_corrected.pdf",facecolor="white",bbox_inches="tight")
    plt.close(fig)


def write_report(summary, scorer, path_frame, frames):
    prop = pd.read_csv(TABLES / "propagation_summary.csv")
    rank = pd.read_csv(TABLES / "ranking_summary.csv")
    alt_class = pd.read_csv(TABLES / "alternative_class_summary.csv")
    alt_gene = pd.read_csv(TABLES / "alternative_gene_summary.csv")
    eligible = alt_gene[alt_gene.n >= 5].iloc[0]
    text = f"""# Corrected frozen real-data run

## Locked design
- Sequence calibration and path-scorer training use only Pathogenic versus Benign variants.
- Likely pathogenic and likely benign variants are untouched locked validation data.
- Candidate paths are contiguous directed graph paths with zero or one gene-gene hop, followed by disease and phenotype.
- BiologicalProcess and Pathway nodes are used as gene annotations, not falsely described as traversed path nodes.
- One temperature schedule, theta(H) = {core.THETA_INTERCEPT} + {core.THETA_SLOPE} H, defines both the exact learned target and the Metropolis acceptance ratio.
- Proposals mix 90% learned mass at twice the target temperature with 10%
  uniform mass; the full Hastings ratio corrects proposal asymmetry.

## Cohort and path audit
- Loaded: {summary['n_variants_loaded']} variants.
- Analyzed: {summary['n_variants_analyzed']}; excluded for no valid phenotype path: {summary['n_variants_excluded_no_path']}.
- Path scorer training examples: {scorer.n_train}, all definitive labels.
- Candidate route range: {int(path_frame.candidate_routes.min())}–{int(path_frame.candidate_routes.max())}; candidate path range: {int(path_frame.candidate_paths.min())}–{int(path_frame.candidate_paths.max())}.

## Learned diffusion validation
- Mean acceptance rate: {summary['mean_acceptance_rate']:.3f}.
- Mean JS divergence between {core.N_CHAINS}-chain final-state distribution and exact target: {summary['mean_sampler_js_to_exact']:.4f}.
- Median JS divergence: {summary['median_sampler_js_to_exact']:.4f}.
- Spearman H versus unique final routes: {summary['correlations']['H_vs_unique_final_routes']:.3f}.

## Experiment 1: uncertainty propagation
{prop.to_string(index=False)}

Spearman H versus grouped phenotype entropy: {summary['correlations']['H_vs_group_entropy']:.3f}.

## Experiment 2: uncertainty perturbation
{rank.to_string(index=False)}

## Experiment 3: alternative uncertainty
{alt_class.to_string(index=False)}

The most sensitive gene represented by at least five variants is {eligible.gene_symbol}
(mean JS {eligible.mean_js:.3f}, n={int(eligible.n)}). Singleton genes are not ranked
as stable gene-level findings.

## Experiment 4: branching and ablation
Spearman H versus raw-endpoint entropy: {summary['correlations']['H_vs_raw_endpoint_entropy']:.3f}.
See `ablations.csv` for uniform, fixed-temperature, no-confidence, no-H, and
no-mechanism comparisons.

## Interpretation boundary
Uncertainty broadening is partly structural by design because H controls the
target temperature. The ablations quantify this designed effect; they are not
presented as independent biological validation. AlzKB contains phenotype links
for a shared disease layer, so gene specificity arises through valid gene-gene
routes and route-level process/pathway annotations, not distinct endpoint sets.
"""
    (REPORTS / "corrected_real_data_report.md").write_text(text, encoding="utf-8")


def write_manifest(mode, scorer):
    def digest(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    manifest = {
        "run_id": "corrected_frozen_2026-09-04",
        "seed": core.RNG_SEED,
        "graph_mode": mode,
        "data": {
            "evo2_scores_sha256": digest(core.SCORE_PATH),
            "alzkb_sha256": digest(
                core.ROOT / "data/raw/alzkb/alzkb-v2-0-0_memgraph.cypherl"
            ),
        },
        "model_sha256": digest(core.MODEL_PATH),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "parameters": {
            "theta": f"{core.THETA_INTERCEPT} + {core.THETA_SLOPE} H",
            "proposal": "0.9 softmax(U/(2 theta)) + 0.1 uniform; full Hastings correction",
            "fixed_theta_ablation": core.FIXED_THETA,
            "max_gene_hops": core.MAX_GENE_HOPS,
            "max_routes": core.MAX_ROUTES,
            "metropolis_chains": core.N_CHAINS,
            "revision_steps": "50 + 10 ceil(log2 N_paths) + round(25 H)",
            "l2": scorer.l2,
        },
    }
    (MANIFESTS / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main():
    graph, variants, _, mode = core.locked_variants()
    relation_frequency = core.relation_frequencies(graph)
    scorer = train_locked_model(graph, variants, relation_frequency)
    path_frame = path_audit(graph, variants)
    frames = compute_primary_results(
        graph, variants, scorer, relation_frequency
    )
    summary = summarize(frames, scorer, path_frame)
    make_figures(variants, frames, summary, scorer, path_frame)
    write_report(summary, scorer, path_frame, frames)
    write_manifest(mode, scorer)
    variants.to_csv(TABLES / "variants_locked_signals.csv", index=False)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
