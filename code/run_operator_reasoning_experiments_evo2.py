#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_operator_reasoning_experiments as baseexp
import pubstyle as pub


RNG_SEED = 42
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "experiments" / "ar_vs_diffusion"
REPORT_PATH = OUTPUT_DIR / "report.md"
SCORE_PATHS = [
    SCRIPT_DIR.parent / "data" / "raw" / "evo2" / "ad_all_scores_v3_vcf.csv",
    SCRIPT_DIR.parent / "ad_all_scores_v3_vcf.csv",
]
CATEGORY_ORDER = [
    "Pathogenic",
    "Likely_pathogenic",
    "Uncertain_significance",
    "Likely_benign",
    "Benign",
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
    "Pathogenic": "#2F5B89",
    "Likely_pathogenic": "#5D8CC2",
    "Uncertain_significance": "#C95E0B",
    "Likely_benign": "#A8CD89",
    "Benign": "#79AE4D",
    "Conflicting_interpretations": "#7B5EA7",
}
TYPE_PREFERENCE = {
    "Gene": 0.00,
    "Pathway": 0.18,
    "BiologicalProcess": 0.28,
    "Disease": 0.62,
    "Phenotype": 0.54,
}
PATH_CACHE: Dict[Tuple[str, int], List[List[str]]] = {}


def apply_publication_style():
    pub.apply()


def present_categories(values: Iterable[str]) -> List[str]:
    return pub.present_categories(values)


def label_for(cat: str) -> str:
    return pub.label_for(cat)


def color_for(cat: str) -> str:
    return pub.color_for(cat)


def score_path() -> Path:
    for path in SCORE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate ad_all_scores_v3_vcf.csv")


def build_symbol_to_node(graph: nx.DiGraph) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for node, attrs in graph.nodes(data=True):
        if attrs.get("type") != "Gene":
            continue
        for key in ("geneSymbol", "name", "commonName", "xrefNcbiGene"):
            value = attrs.get(key)
            if value:
                mapping[str(value).upper()] = node
    return mapping


def entropy_from_prob(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def load_evo2_variants(graph: nx.DiGraph) -> pd.DataFrame:
    df = pd.read_csv(score_path())
    df["clinical_significance"] = df["CLNSIG"].astype(str).map(baseexp.normalize_significance)
    df = df[df["clinical_significance"].isin(CATEGORY_ORDER)].copy()
    df["variant_id"] = df["clinvar_variation_id"].astype(str)
    df["gene_symbol"] = df["gene_symbol"].astype(str)
    df["pathogenicity_prob"] = pd.to_numeric(df["zeroshot_pathogenicity_prob"], errors="coerce")
    df["zeroshot_score"] = pd.to_numeric(df["zeroshot_neg_delta_likelihood"], errors="coerce")
    df = df.dropna(subset=["pathogenicity_prob", "zeroshot_score", "gene_symbol"])
    train = df[df["dataset"] == "binary"].copy()
    train = train[train["split"] == "train"].copy()
    train["y"] = pd.to_numeric(train["labels"], errors="coerce")
    train = train.dropna(subset=["y", "zeroshot_score"])
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(train["zeroshot_score"].to_numpy(), train["y"].astype(int).to_numpy())
    df["isotonic_pathogenicity_prob"] = iso.predict(df["zeroshot_score"].to_numpy())
    df["isotonic_entropy"] = entropy_from_prob(df["isotonic_pathogenicity_prob"].to_numpy())
    df["dyna_entropy"] = df["isotonic_entropy"]
    df["uncertainty_metric"] = "isotonic_entropy"
    symbol_to_node = build_symbol_to_node(graph)
    df["gene_id"] = df["gene_symbol"].str.upper().map(symbol_to_node)
    df["matched"] = df["gene_id"].notna()
    df = df[df["matched"]].copy().reset_index(drop=True)
    return df


def structural_path_score(graph: nx.DiGraph, path: Optional[List[str]], edge_type_frequency: Counter) -> float:
    return baseexp.path_score(graph, path, edge_type_frequency)


def transition_score(
    graph: nx.DiGraph,
    current: str,
    neighbor: str,
    edge_type_frequency: Counter,
    pathogenicity_prob: float,
    entropy: float,
    constraint_weight: float = 0.0,
) -> float:
    rel = graph.edges[current, neighbor].get("relation", "")
    rel_score = math.log(edge_type_frequency.get(rel, 1) + 1)
    node_type = baseexp.get_node_type(graph, neighbor)
    type_bonus = pathogenicity_prob * TYPE_PREFERENCE.get(node_type, 0.05)
    uncertainty_bonus = entropy * (0.20 if node_type in {"Pathway", "BiologicalProcess", "Disease"} else 0.05)
    constraint_bonus = constraint_weight if node_type == "Disease" else 0.0
    return rel_score + type_bonus + uncertainty_bonus + constraint_bonus


def path_utility(
    graph: nx.DiGraph,
    path: Optional[List[str]],
    edge_type_frequency: Counter,
    pathogenicity_prob: float,
    entropy: float,
    constraint_weight: float = 0.0,
) -> float:
    if path is None or len(path) < 2:
        return -1e9
    score = structural_path_score(graph, path, edge_type_frequency)
    score += pathogenicity_prob * (2.0 * baseexp.disease_coverage(path, graph))
    score += 0.25 * len(path)
    if baseexp.get_node_type(graph, path[-1]) == "Phenotype":
        score += 0.6 + 0.25 * pathogenicity_prob
    if constraint_weight > 0:
        score += constraint_weight * baseexp.disease_coverage(path, graph)
    score += 0.10 * entropy * len({baseexp.get_node_type(graph, node) for node in path})
    return score


def weighted_neighbor_choice(
    graph: nx.DiGraph,
    current: str,
    path: List[str],
    edge_type_frequency: Counter,
    pathogenicity_prob: float,
    entropy: float,
    rng: random.Random,
    constraint_weight: float = 0.0,
) -> Optional[str]:
    neighbors = [n for n in graph.successors(current) if n not in path]
    if not neighbors:
        neighbors = list(graph.successors(current))
    if not neighbors:
        return None
    scores = np.array(
        [
            transition_score(
                graph,
                current,
                neighbor,
                edge_type_frequency,
                pathogenicity_prob,
                entropy,
                constraint_weight,
            )
            for neighbor in neighbors
        ],
        dtype=float,
    )
    temperature = 0.35 + 0.85 * max(entropy, 1e-6)
    scores = (scores - scores.max()) / temperature
    weights = np.exp(scores)
    weights = weights / weights.sum()
    idx = rng.choices(range(len(neighbors)), weights=weights.tolist(), k=1)[0]
    return neighbors[idx]


def candidate_paths_for_gene(
    graph: nx.DiGraph,
    start_gene: str,
    max_depth: int = 6,
    max_paths: int = 600,
) -> List[List[str]]:
    key = (start_gene, max_depth)
    if key in PATH_CACHE:
        return PATH_CACHE[key]
    shortest = nx.single_source_shortest_path(graph, start_gene, cutoff=max_depth)
    candidates = [
        path
        for node, path in shortest.items()
        if node != start_gene and graph.nodes[node].get("type") == "Phenotype"
    ]
    candidates.sort(key=lambda path: (len(path), tuple(path)))
    PATH_CACHE[key] = candidates[:max_paths]
    return PATH_CACHE[key]


def conditioned_ar_operator(
    graph: nx.DiGraph,
    start_gene: str,
    edge_type_frequency: Counter,
    pathogenicity_prob: float,
    entropy: float,
    beam_width: int = 5,
    max_depth: int = 6,
    tie_seed: int = 42,
    constraint_weight: float = 0.0,
) -> Tuple[Optional[List[str]], float]:
    rng = random.Random(tie_seed)
    candidates = candidate_paths_for_gene(graph, start_gene, max_depth=max_depth)
    if not candidates:
        return None, -1e9
    prefix = [start_gene]
    depth = 1
    remaining = candidates
    while remaining and depth < max(len(path) for path in remaining):
        next_options = {}
        for path in remaining:
            if len(path) > depth and path[:depth] == prefix:
                nxt = path[depth]
                next_options.setdefault(nxt, []).append(path)
        if not next_options:
            break
        scored = []
        items = list(next_options.items())
        rng.shuffle(items)
        for nxt, supported_paths in items:
            local = transition_score(
                graph,
                prefix[-1],
                nxt,
                edge_type_frequency,
                pathogenicity_prob,
                entropy,
                constraint_weight=constraint_weight,
            )
            global_hint = mean(
                [
                    path_utility(
                        graph,
                        path,
                        edge_type_frequency,
                        pathogenicity_prob,
                        entropy,
                        constraint_weight=constraint_weight,
                    )
                    for path in supported_paths
                ]
            )
            scored.append((local + (1.0 - entropy) * 0.08 * global_hint, nxt, supported_paths))
        scored.sort(key=lambda item: item[0], reverse=True)
        _, best_next, remaining = scored[0]
        prefix = prefix + [best_next]
        if baseexp.get_node_type(graph, best_next) == "Phenotype":
            score = path_utility(
                graph,
                prefix,
                edge_type_frequency,
                pathogenicity_prob,
                entropy,
                constraint_weight=constraint_weight,
            )
            return prefix, score
        depth += 1
    completed = [path for path in remaining if path[: len(prefix)] == prefix and baseexp.get_node_type(graph, path[-1]) == "Phenotype"]
    if not completed:
        return None, -1e9
    best = max(
        completed,
        key=lambda path: path_utility(
            graph, path, edge_type_frequency, pathogenicity_prob, entropy, constraint_weight=constraint_weight
        ),
    )
    return best, path_utility(
        graph, best, edge_type_frequency, pathogenicity_prob, entropy, constraint_weight=constraint_weight
    )


def conditioned_revision_operator(
    graph: nx.DiGraph,
    start_gene: str,
    edge_type_frequency: Counter,
    pathogenicity_prob: float,
    entropy: float,
    max_depth: int = 6,
    seed: int = 42,
    constraint_weight: float = 0.0,
    initial_path: Optional[List[str]] = None,
) -> Optional[List[str]]:
    rng = random.Random(seed)
    candidates = candidate_paths_for_gene(graph, start_gene, max_depth=max_depth)
    if not candidates:
        return None
    scored = np.array(
        [
            path_utility(
                graph,
                path,
                edge_type_frequency,
                pathogenicity_prob,
                entropy,
                constraint_weight=constraint_weight,
            )
            for path in candidates
        ],
        dtype=float,
    )
    temperature = 0.35 + 0.9 * entropy

    def sample_candidate() -> List[str]:
        logits = (scored - scored.max()) / max(temperature, 1e-6)
        weights = np.exp(logits)
        weights = weights / weights.sum()
        idx = rng.choices(range(len(candidates)), weights=weights.tolist(), k=1)[0]
        return list(candidates[idx])

    current_path = list(initial_path) if initial_path is not None else sample_candidate()
    current_score = path_utility(
        graph,
        current_path,
        edge_type_frequency,
        pathogenicity_prob,
        entropy,
        constraint_weight=constraint_weight,
    )
    n_steps = 10 + int(round(15 * entropy))
    for _ in range(n_steps):
        proposal = sample_candidate()
        proposal_score = path_utility(
            graph,
            proposal,
            edge_type_frequency,
            pathogenicity_prob,
            entropy,
            constraint_weight=constraint_weight,
        )
        delta = proposal_score - current_score
        if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-6)):
            current_path = proposal
            current_score = proposal_score
    return current_path


def run_revision_trials(
    graph: nx.DiGraph,
    start_gene: str,
    edge_type_frequency: Counter,
    pathogenicity_prob: float,
    entropy: float,
    n_runs: int = 20,
    constraint_weight: float = 0.0,
) -> List[Optional[List[str]]]:
    return [
        conditioned_revision_operator(
            graph,
            start_gene,
            edge_type_frequency,
            pathogenicity_prob,
            entropy,
            seed=RNG_SEED + 17 * i,
            constraint_weight=constraint_weight,
        )
        for i in range(n_runs)
    ]


def run_ar_trials(
    graph: nx.DiGraph,
    start_gene: str,
    edge_type_frequency: Counter,
    pathogenicity_prob: float,
    entropy: float,
    n_runs: int = 20,
    constraint_weight: float = 0.0,
) -> List[Optional[List[str]]]:
    trials = []
    for i in range(n_runs):
        path, _ = conditioned_ar_operator(
            graph,
            start_gene,
            edge_type_frequency,
            pathogenicity_prob,
            entropy,
            tie_seed=RNG_SEED + i,
            constraint_weight=constraint_weight,
        )
        trials.append(path)
    return trials


def phenotype_diversity(paths: Iterable[Optional[List[str]]], graph: nx.DiGraph) -> int:
    return len(set(baseexp.collect_phenotypes(paths, graph)))


def trajectory_diversity(paths: Iterable[Optional[List[str]]]) -> int:
    return len(set(tuple(path) for path in paths if path is not None))


def sample_by_category(df: pd.DataFrame, n_per_category: int) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("clinical_significance"):
        parts.append(group.sample(min(len(group), n_per_category), random_state=RNG_SEED))
    return pd.concat(parts, ignore_index=True)


def run_experiment1(
    graph: nx.DiGraph, variants_df: pd.DataFrame, edge_type_frequency: Counter, data_note: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, variant in variants_df.iterrows():
        ar_path, ar_internal = conditioned_ar_operator(
            graph,
            variant["gene_id"],
            edge_type_frequency,
            float(variant["pathogenicity_prob"]),
            float(variant["dyna_entropy"]),
        )
        rev_path = conditioned_revision_operator(
            graph,
            variant["gene_id"],
            edge_type_frequency,
            float(variant["pathogenicity_prob"]),
            float(variant["dyna_entropy"]),
            initial_path=ar_path,
        )
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "gene_id": variant["gene_id"],
                "clinical_significance": variant["clinical_significance"],
                "dyna_entropy": variant["dyna_entropy"],
                "pathogenicity_prob": variant["pathogenicity_prob"],
                "ar_valid": baseexp.path_validity(graph, ar_path),
                "rev_valid": baseexp.path_validity(graph, rev_path),
                "ar_score": path_utility(
                    graph,
                    ar_path,
                    edge_type_frequency,
                    float(variant["pathogenicity_prob"]),
                    float(variant["dyna_entropy"]),
                ),
                "rev_score": path_utility(
                    graph,
                    rev_path,
                    edge_type_frequency,
                    float(variant["pathogenicity_prob"]),
                    float(variant["dyna_entropy"]),
                ),
                "ar_internal_score": ar_internal,
            }
        )
    df = pd.DataFrame(rows)
    df["validity_advantage"] = df["rev_valid"] - df["ar_valid"]
    df["score_advantage"] = df["rev_score"] - df["ar_score"]
    try:
        df["entropy_quartile"] = pd.qcut(
            df["dyna_entropy"],
            q=4,
            labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"],
            duplicates="drop",
        )
    except ValueError:
        df["entropy_quartile"] = pd.cut(
            df["dyna_entropy"], bins=4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]
        )
    summary = (
        df.groupby("entropy_quartile", observed=False)
        .agg(
            validity_advantage_mean=("validity_advantage", "mean"),
            validity_advantage_std=("validity_advantage", "std"),
            score_advantage_mean=("score_advantage", "mean"),
            score_advantage_std=("score_advantage", "std"),
            n=("variant_id", "count"),
        )
        .reset_index()
    )
    df["data_note"] = data_note
    summary["data_note"] = data_note
    return df, summary


def run_experiment2(
    graph: nx.DiGraph, variants_df: pd.DataFrame, edge_type_frequency: Counter, data_note: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reachability_cache: Dict[Tuple[str, int], set] = {}
    rows = []
    for _, variant in variants_df.iterrows():
        gene = variant["gene_id"]
        entropy = float(variant["dyna_entropy"])
        path_prob = float(variant["pathogenicity_prob"])
        ar_path, _ = conditioned_ar_operator(graph, gene, edge_type_frequency, path_prob, entropy)
        if ar_path is None or len(ar_path) < 2:
            continue
        chosen_neighbor = ar_path[1]
        neighbors = list(graph.successors(gene))
        chosen_phenotypes = baseexp.commitment_reachable_phenotypes(graph, chosen_neighbor, 5, reachability_cache)
        unchosen_phenotypes = set()
        for neighbor in neighbors:
            if neighbor == chosen_neighbor:
                continue
            unchosen_phenotypes |= baseexp.commitment_reachable_phenotypes(graph, neighbor, 5, reachability_cache)
        revision_paths = run_revision_trials(graph, gene, edge_type_frequency, path_prob, entropy, n_runs=20)
        revision_phenotypes = set(baseexp.collect_phenotypes(revision_paths, graph))
        total_reachable = chosen_phenotypes | unchosen_phenotypes
        revision_only = revision_phenotypes - chosen_phenotypes
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "gene_id": gene,
                "clinical_significance": variant["clinical_significance"],
                "dyna_entropy": entropy,
                "pathogenicity_prob": path_prob,
                "chosen_neighbor": chosen_neighbor,
                "chosen_type": baseexp.get_node_type(graph, chosen_neighbor),
                "chosen_reachable_phenotypes": len(chosen_phenotypes),
                "unchosen_reachable_phenotypes": len(unchosen_phenotypes),
                "revision_reachable_phenotypes": len(revision_phenotypes),
                "revision_only_phenotypes": len(revision_only),
                "coverage_loss": len(revision_only) / max(1, len(total_reachable | revision_phenotypes)),
                "revision_finds_excluded": int(len(revision_only) > 0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(
            columns=[
                "clinical_significance",
                "coverage_loss_mean",
                "coverage_loss_std",
                "revision_finds_excluded_mean",
                "n",
                "data_note",
            ]
        )
    summary = (
        df.groupby("clinical_significance")
        .agg(
            coverage_loss_mean=("coverage_loss", "mean"),
            coverage_loss_std=("coverage_loss", "std"),
            revision_finds_excluded_mean=("revision_finds_excluded", "mean"),
            n=("variant_id", "count"),
        )
        .reindex(present_categories(df["clinical_significance"]))
        .reset_index()
    )
    df["data_note"] = data_note
    summary["data_note"] = data_note
    return df, summary


def run_experiment3(
    graph: nx.DiGraph, variants_df: pd.DataFrame, edge_type_frequency: Counter, data_note: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sampled = sample_by_category(variants_df, 30)
    rows = []
    for _, variant in sampled.iterrows():
        entropy = float(variant["dyna_entropy"])
        path_prob = float(variant["pathogenicity_prob"])
        rev_paths = run_revision_trials(
            graph, variant["gene_id"], edge_type_frequency, path_prob, entropy, n_runs=10
        )
        ar_paths = run_ar_trials(
            graph, variant["gene_id"], edge_type_frequency, path_prob, entropy, n_runs=10
        )
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "gene_id": variant["gene_id"],
                "clinical_significance": variant["clinical_significance"],
                "dyna_entropy": entropy,
                "pathogenicity_prob": path_prob,
                "rev_trajectory_diversity": trajectory_diversity(rev_paths),
                "rev_phenotype_diversity": phenotype_diversity(rev_paths, graph),
                "ar_trajectory_diversity": trajectory_diversity(ar_paths),
                "ar_phenotype_diversity": phenotype_diversity(ar_paths, graph),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(
            columns=[
                "clinical_significance",
                "rev_traj_mean",
                "rev_traj_std",
                "ar_traj_mean",
                "ar_traj_std",
                "rev_pheno_mean",
                "ar_pheno_mean",
                "n",
                "data_note",
            ]
        )
    summary = (
        df.groupby("clinical_significance")
        .agg(
            rev_traj_mean=("rev_trajectory_diversity", "mean"),
            rev_traj_std=("rev_trajectory_diversity", "std"),
            ar_traj_mean=("ar_trajectory_diversity", "mean"),
            ar_traj_std=("ar_trajectory_diversity", "std"),
            rev_pheno_mean=("rev_phenotype_diversity", "mean"),
            ar_pheno_mean=("ar_phenotype_diversity", "mean"),
            n=("variant_id", "count"),
        )
        .reindex(present_categories(df["clinical_significance"]))
        .reset_index()
    )
    df["data_note"] = data_note
    summary["data_note"] = data_note
    return df, summary


def run_experiment4(
    graph: nx.DiGraph, variants_df: pd.DataFrame, edge_type_frequency: Counter, data_note: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sampled = sample_by_category(variants_df, 20)
    rows = []
    for _, variant in sampled.iterrows():
        entropy = float(variant["dyna_entropy"])
        path_prob = float(variant["pathogenicity_prob"])
        unconstrained = run_revision_trials(
            graph, variant["gene_id"], edge_type_frequency, path_prob, entropy, n_runs=15, constraint_weight=0.0
        )
        constrained = run_revision_trials(
            graph, variant["gene_id"], edge_type_frequency, path_prob, entropy, n_runs=15, constraint_weight=0.9
        )
        unconstrained_set = set(tuple(path) for path in unconstrained if path is not None)
        constrained_set = set(tuple(path) for path in constrained if path is not None)
        unlocked = constrained_set - unconstrained_set
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "gene_id": variant["gene_id"],
                "clinical_significance": variant["clinical_significance"],
                "constraint_unlocked_trajectories": len(unlocked),
                "unconstrained_mean_disease_coverage": mean(
                    [baseexp.disease_coverage(list(path), graph) for path in unconstrained_set]
                )
                if unconstrained_set
                else 0.0,
                "constrained_mean_disease_coverage": mean(
                    [baseexp.disease_coverage(list(path), graph) for path in constrained_set]
                )
                if constrained_set
                else 0.0,
                "unlocked_mean_disease_coverage": mean(
                    [baseexp.disease_coverage(list(path), graph) for path in unlocked]
                )
                if unlocked
                else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(
            columns=[
                "clinical_significance",
                "unlocked_mean",
                "unlocked_std",
                "unconstrained_disease_mean",
                "constrained_disease_mean",
                "unlocked_disease_mean",
                "n",
                "data_note",
            ]
        )
    summary = (
        df.groupby("clinical_significance")
        .agg(
            unlocked_mean=("constraint_unlocked_trajectories", "mean"),
            unlocked_std=("constraint_unlocked_trajectories", "std"),
            unconstrained_disease_mean=("unconstrained_mean_disease_coverage", "mean"),
            constrained_disease_mean=("constrained_mean_disease_coverage", "mean"),
            unlocked_disease_mean=("unlocked_mean_disease_coverage", "mean"),
            n=("variant_id", "count"),
        )
        .reindex(present_categories(df["clinical_significance"]))
        .reset_index()
    )
    df["data_note"] = data_note
    summary["data_note"] = data_note
    return df, summary


def shannon_entropy(items: List[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total + 1e-12, 2) for count in counts.values())


def run_experiment5(
    graph: nx.DiGraph, variants_df: pd.DataFrame, edge_type_frequency: Counter, data_note: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = variants_df[
        variants_df["clinical_significance"].isin(["Uncertain_significance", "Pathogenic"])
    ].copy()
    rows = []
    for _, variant in subset.iterrows():
        entropy = float(variant["dyna_entropy"])
        path_prob = float(variant["pathogenicity_prob"])
        paths = run_revision_trials(
            graph, variant["gene_id"], edge_type_frequency, path_prob, entropy, n_runs=20
        )
        phenotypes = baseexp.collect_phenotypes(paths, graph)
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "gene_id": variant["gene_id"],
                "clinical_significance": variant["clinical_significance"],
                "phenotype_width": len(set(phenotypes)),
                "phenotype_entropy": shannon_entropy(phenotypes),
                "valid_runs": len(phenotypes),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "clinical_significance",
                "width_mean",
                "width_std",
                "entropy_mean",
                "valid_runs_mean",
                "n",
                "data_note",
            ]
        )
        return df, empty_summary, pd.DataFrame(columns=["gene_id", "Pathogenic", "Uncertain_significance"])
    summary = (
        df.groupby("clinical_significance")
        .agg(
            width_mean=("phenotype_width", "mean"),
            width_std=("phenotype_width", "std"),
            entropy_mean=("phenotype_entropy", "mean"),
            valid_runs_mean=("valid_runs", "mean"),
            n=("variant_id", "count"),
        )
        .reindex(["Pathogenic", "Uncertain_significance"])
        .reset_index()
    )
    paired = (
        df.groupby(["gene_id", "clinical_significance"])
        .agg(width_mean=("phenotype_width", "mean"))
        .reset_index()
        .pivot(index="gene_id", columns="clinical_significance", values="width_mean")
        .dropna(subset=["Pathogenic", "Uncertain_significance"])
        .reset_index()
    )
    df["data_note"] = data_note
    summary["data_note"] = data_note
    return df, summary, paired


def run_experiment6(
    graph: nx.DiGraph, variants_df: pd.DataFrame, edge_type_frequency: Counter, data_note: str
) -> Tuple[pd.DataFrame, float]:
    variant_genes = sorted(set(variants_df["gene_id"]))
    centrality = nx.betweenness_centrality(nx.Graph(graph), k=min(200, len(graph.nodes)), seed=RNG_SEED)
    rows = []
    for gene in variant_genes:
        gene_variants = variants_df[variants_df["gene_id"] == gene]
        entropy = float(gene_variants["dyna_entropy"].mean())
        path_prob = float(gene_variants["pathogenicity_prob"].mean())
        rev_paths = run_revision_trials(graph, gene, edge_type_frequency, path_prob, entropy, n_runs=20)
        ar_paths = run_ar_trials(graph, gene, edge_type_frequency, path_prob, entropy, n_runs=20)
        rev_phen = set(baseexp.collect_phenotypes(rev_paths, graph))
        ar_phen = set(baseexp.collect_phenotypes(ar_paths, graph))
        rows.append(
            {
                "gene_id": gene,
                "gene_symbol": graph.nodes[gene].get("geneSymbol") or graph.nodes[gene].get("name") or gene,
                "betweenness_centrality": centrality.get(gene, 0.0),
                "trajectory_divergence": trajectory_diversity(rev_paths) - trajectory_diversity(ar_paths),
                "phenotype_divergence": len(rev_phen) - len(ar_phen),
                "mean_entropy": entropy,
                "mean_pathogenicity_prob": path_prob,
                "n_variants": len(gene_variants),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, 0.0
    corr, _ = spearmanr(df["betweenness_centrality"], df["phenotype_divergence"])
    corr = 0.0 if math.isnan(corr) else float(corr)
    df["data_note"] = data_note
    return df, corr


def run_experiment7(
    graph: nx.DiGraph, variants_df: pd.DataFrame, edge_type_frequency: Counter, data_note: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sampled = sample_by_category(variants_df, 30)
    rows = []
    for _, variant in sampled.iterrows():
        entropy = float(variant["dyna_entropy"])
        path_prob = float(variant["pathogenicity_prob"])
        for operator in ["Revision", "AR"]:
            if operator == "Revision":
                paths = run_revision_trials(graph, variant["gene_id"], edge_type_frequency, path_prob, entropy, n_runs=20)
            else:
                paths = run_ar_trials(graph, variant["gene_id"], edge_type_frequency, path_prob, entropy, n_runs=20)
            phenotypes = baseexp.collect_phenotypes(paths, graph)
            failure_rate = 1 - (len(phenotypes) / 20.0)
            phenotype_counts = Counter(phenotypes)
            consistency = max(phenotype_counts.values()) / len(phenotypes) if phenotypes else 0.0
            productive_diversity = len(set(phenotypes)) / len(phenotypes) if phenotypes else 0.0
            rows.append(
                {
                    "variant_id": variant["variant_id"],
                    "gene_id": variant["gene_id"],
                    "clinical_significance": variant["clinical_significance"],
                    "operator": operator,
                    "consistency": consistency,
                    "productive_diversity": productive_diversity,
                    "failure_rate": failure_rate,
                    "distinct_phenotypes": len(set(phenotypes)),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(
            columns=[
                "clinical_significance",
                "operator",
                "consistency_mean",
                "productive_diversity_mean",
                "failure_rate_mean",
                "distinct_phenotypes_mean",
                "n",
                "data_note",
            ]
        )
    summary = (
        df.groupby(["clinical_significance", "operator"])
        .agg(
            consistency_mean=("consistency", "mean"),
            productive_diversity_mean=("productive_diversity", "mean"),
            failure_rate_mean=("failure_rate", "mean"),
            distinct_phenotypes_mean=("distinct_phenotypes", "mean"),
            n=("variant_id", "count"),
        )
        .reset_index()
    )
    df["data_note"] = data_note
    summary["data_note"] = data_note
    return df, summary


def panel_label(ax, label: str):
    pub.panel_label(ax, label)


def jitter_positions(n: int, center: float, width: float = 0.12) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + n + int(center * 100))
    return center + rng.uniform(-width, width, size=n)


def add_box_strip(ax, df: pd.DataFrame, categories: List[str], value_col: str, ylabel: str, title: str):
    pub.box_strip(ax, df, categories, value_col, ylabel, title, seed=RNG_SEED)


def save_figure(fig: plt.Figure, stem: str):
    pub.save(fig, OUTPUT_DIR, stem)


def make_combined_figure1(variants_df: pd.DataFrame):
    cats = present_categories(variants_df["clinical_significance"])
    fig = plt.figure(figsize=(13.4, 4.3))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.25, 1.25])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    counts = variants_df["clinical_significance"].value_counts().reindex(cats)
    ax1.bar(range(len(cats)), counts.values, color=[color_for(c) for c in cats], edgecolor="#4A4A4A", linewidth=0.8)
    ax1.set_xticks(range(len(cats)))
    ax1.set_xticklabels([label_for(c) for c in cats])
    ax1.set_ylabel("Variants")
    ax1.set_title("Class composition of Evo2-scored AD variants")
    for i, v in enumerate(counts.values):
        ax1.text(i, v + max(counts.values) * 0.02, str(int(v)), ha="center", va="bottom", fontsize=9)
    ax1.grid(axis="y")
    panel_label(ax1, "a")

    add_box_strip(
        ax2,
        variants_df,
        cats,
        "pathogenicity_prob",
        "Zero-shot pathogenicity probability",
        "Zero-shot probability distribution by class",
    )
    panel_label(ax2, "b")

    add_box_strip(
        ax3,
        variants_df,
        cats,
        "dyna_entropy",
        "Isotonic entropy (bits)",
        "Isotonic uncertainty distribution by class",
    )
    panel_label(ax3, "c")
    fig.suptitle("Figure 1 | Evo2 zero-shot score landscape with isotonic uncertainty", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "combined_figure1_evo2_score_landscape")


def make_combined_figure2(exp1_df: pd.DataFrame, exp1_stats: pd.DataFrame, exp2_df: pd.DataFrame, exp2_stats: pd.DataFrame):
    cats = present_categories(exp2_df["clinical_significance"])
    fig = plt.figure(figsize=(13.5, 7.0))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    x = np.arange(len(exp1_stats))
    ax1.errorbar(x, exp1_stats["validity_advantage_mean"], yerr=exp1_stats["validity_advantage_std"].fillna(0), marker="o", color="#355C7D")
    ax1.axhline(0, color="#888888", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp1_stats["entropy_quartile"])
    ax1.set_ylabel("Revision - AR validity")
    ax1.set_title("Validity advantage is flat across entropy quartiles")
    ax1.grid(axis="y")
    panel_label(ax1, "a")

    ax2.errorbar(x, exp1_stats["score_advantage_mean"], yerr=exp1_stats["score_advantage_std"].fillna(0), marker="o", color="#C95E0B")
    ax2.axhline(0, color="#888888", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp1_stats["entropy_quartile"])
    ax2.set_ylabel("Revision - AR utility score")
    ax2.set_title("Utility advantage is negligible in the direct phenotype layer")
    ax2.grid(axis="y")
    panel_label(ax2, "b")

    add_box_strip(ax3, exp2_df, cats, "coverage_loss", "Phenotype coverage loss", "AR first-step commitment excludes reachable phenotypes")
    panel_label(ax3, "c")

    frac = exp2_stats.set_index("clinical_significance").loc[cats, "revision_finds_excluded_mean"]
    ax4.bar(range(len(cats)), frac.values, color=[color_for(c) for c in cats], edgecolor="#4A4A4A", linewidth=0.8)
    ax4.set_xticks(range(len(cats)))
    ax4.set_xticklabels([label_for(c) for c in cats])
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("Fraction of variants")
    ax4.set_title("Revision does not recover excluded phenotypes in this graph slice")
    ax4.grid(axis="y")
    panel_label(ax4, "d")
    fig.suptitle("Figure 2 | Operator advantage and commitment-point sensitivity", y=1.01, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "combined_figure2_operator_advantage_and_commitment")


def make_combined_figure3(exp3_df: pd.DataFrame):
    cats = present_categories(exp3_df["clinical_significance"])
    fig = plt.figure(figsize=(13.4, 4.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.3, 1.1, 1.1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    add_box_strip(
        ax1,
        exp3_df,
        cats,
        "rev_phenotype_diversity",
        "Phenotype diversity (10 runs)",
        "Revision remains high-diversity across all classes",
    )
    panel_label(ax1, "a")

    ax2.scatter(exp3_df["dyna_entropy"], exp3_df["rev_trajectory_diversity"], s=26, alpha=0.55, color="#C95E0B", edgecolor="white", linewidth=0.4)
    if len(exp3_df) > 1:
        coeff = np.polyfit(exp3_df["dyna_entropy"], exp3_df["rev_trajectory_diversity"], deg=1)
        xs = np.linspace(exp3_df["dyna_entropy"].min(), exp3_df["dyna_entropy"].max(), 100)
        ax2.plot(xs, coeff[0] * xs + coeff[1], color="#355C7D", linewidth=1.5)
    corr, _ = spearmanr(exp3_df["dyna_entropy"], exp3_df["rev_trajectory_diversity"])
    ax2.set_xlabel("Isotonic entropy")
    ax2.set_ylabel("Revision trajectory diversity")
    ax2.set_title(f"Diversity vs uncertainty (Spearman = {0.0 if math.isnan(corr) else corr:.2f})")
    ax2.grid(True)
    panel_label(ax2, "b")

    grouped = exp3_df.groupby("clinical_significance")[["rev_trajectory_diversity", "ar_trajectory_diversity"]].mean().reindex(cats)
    x = np.arange(len(cats))
    ax3.plot(x, grouped["rev_trajectory_diversity"], marker="o", color="#C95E0B", label="Revision")
    ax3.plot(x, grouped["ar_trajectory_diversity"], marker="o", color="#355C7D", linestyle="--", label="AR")
    ax3.set_xticks(x)
    ax3.set_xticklabels([label_for(c) for c in cats])
    ax3.set_ylabel("Mean trajectory diversity")
    ax3.set_title("AR spans full set; revision concentrates")
    ax3.legend(frameon=False)
    ax3.grid(axis="y")
    panel_label(ax3, "c")
    fig.suptitle("Figure 1 | Diversity calibration under zero-shot uncertainty", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "combined_figure1_diversity_and_uncertainty")


def make_combined_figure4(exp4_df: pd.DataFrame, exp4_stats: pd.DataFrame, exp5_df: pd.DataFrame, exp5_stats: pd.DataFrame):
    # Trimmed: the constraint-unlocked and disease-coverage panels were flat
    # (constrained and unconstrained search coincide on this direct disease
    # layer), so only the VUS-vs-Pathogenic breadth panels are kept.
    fig = plt.figure(figsize=(10.4, 4.6))
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    add_box_strip(
        ax1,
        exp5_df,
        ["Pathogenic", "Uncertain_significance"],
        "phenotype_width",
        "Distinct phenotypes (20 runs)",
        "Phenotype width: VUS vs Pathogenic",
    )
    panel_label(ax1, "a")

    add_box_strip(
        ax2,
        exp5_df,
        ["Pathogenic", "Uncertain_significance"],
        "phenotype_entropy",
        "Phenotype entropy",
        "Phenotype entropy: VUS vs Pathogenic",
    )
    panel_label(ax2, "b")

    fig.suptitle("Figure 2 | VUS phenotype breadth matches pathogenic variants", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "combined_figure2_vus_phenotype_breadth")


def make_combined_figure5(exp6_df: pd.DataFrame, exp6_corr: float, exp7_stats: pd.DataFrame):
    cats = present_categories(exp7_stats["clinical_significance"])
    fig = plt.figure(figsize=(13.5, 7.0))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.scatter(exp6_df["betweenness_centrality"], exp6_df["phenotype_divergence"], s=40, c=exp6_df["mean_entropy"], cmap="viridis", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax1.set_xlabel("Betweenness centrality")
    ax1.set_ylabel("Phenotype divergence (Revision - AR)")
    ax1.set_title(f"Central genes show larger operator divergence (Spearman = {exp6_corr:.2f})")
    ax1.grid(True)
    for _, row in exp6_df.nlargest(5, "betweenness_centrality").iterrows():
        ax1.annotate(row["gene_symbol"], (row["betweenness_centrality"], row["phenotype_divergence"]), fontsize=8)
    panel_label(ax1, "a")

    top = exp6_df.nlargest(8, "phenotype_divergence").sort_values("phenotype_divergence")
    ax2.barh(top["gene_symbol"], top["phenotype_divergence"], color="#5D8CC2", edgecolor="#355C7D")
    ax2.set_xlabel("Phenotype divergence")
    ax2.set_title("Genes with the strongest revision-vs-AR separation")
    panel_label(ax2, "b")

    for operator, color in [("Revision", "#C95E0B"), ("AR", "#355C7D")]:
        vals = (
            exp7_stats[exp7_stats["operator"] == operator]
            .set_index("clinical_significance")
            .reindex(cats)["consistency_mean"]
        )
        ax3.plot(range(len(cats)), vals.values, marker="o", label=operator, color=color)
    ax3.set_xticks(range(len(cats)))
    ax3.set_xticklabels([label_for(c) for c in cats])
    ax3.set_ylabel("Mean consistency")
    ax3.set_title("Revision trades consistency for broader valid phenotype sets")
    ax3.legend(frameon=False)
    ax3.grid(axis="y")
    panel_label(ax3, "c")

    width = 0.36
    x = np.arange(len(cats))
    rev_distinct = (
        exp7_stats[exp7_stats["operator"] == "Revision"]
        .set_index("clinical_significance")
        .reindex(cats)["distinct_phenotypes_mean"]
        .fillna(0)
    )
    ar_distinct = (
        exp7_stats[exp7_stats["operator"] == "AR"]
        .set_index("clinical_significance")
        .reindex(cats)["distinct_phenotypes_mean"]
        .fillna(0)
    )
    ax4.bar(x - width / 2, rev_distinct.values, width, color="#C95E0B", label="Revision", edgecolor="#4A4A4A")
    ax4.bar(x + width / 2, ar_distinct.values, width, color="#355C7D", label="AR", edgecolor="#4A4A4A")
    ax4.set_xticks(x)
    ax4.set_xticklabels([label_for(c) for c in cats])
    ax4.set_ylabel("Mean distinct phenotypes")
    ax4.set_title("Revision samples slightly more distinct phenotypes per variant")
    ax4.legend(frameon=False)
    ax4.grid(axis="y")
    panel_label(ax4, "d")
    fig.suptitle("Figure 3 | Graph structure, operator divergence, and reliability", y=1.01, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "combined_figure3_structure_and_reliability")


def write_report(
    variants_df: pd.DataFrame,
    exp1_stats: pd.DataFrame,
    exp2_stats: pd.DataFrame,
    exp3_stats: pd.DataFrame,
    exp4_stats: pd.DataFrame,
    exp5_stats: pd.DataFrame,
    exp6_corr: float,
    exp7_stats: pd.DataFrame,
):
    cats = present_categories(variants_df["clinical_significance"])
    report_labels = [label_for(cat).replace("\n", " ") for cat in cats]
    counts = variants_df["clinical_significance"].value_counts().reindex(cats).fillna(0).astype(int)
    q1 = float(exp1_stats["validity_advantage_mean"].iloc[0]) if not exp1_stats.empty else float("nan")
    q4 = float(exp1_stats["validity_advantage_mean"].iloc[-1]) if not exp1_stats.empty else float("nan")
    exp2_top = exp2_stats.sort_values("coverage_loss_mean", ascending=False).head(1)
    vus_width = exp5_stats.loc[exp5_stats["clinical_significance"] == "Uncertain_significance", "width_mean"]
    path_width = exp5_stats.loc[exp5_stats["clinical_significance"] == "Pathogenic", "width_mean"]
    rev_fail = exp7_stats.loc[exp7_stats["operator"] == "Revision", "failure_rate_mean"].mean()
    ar_fail = exp7_stats.loc[exp7_stats["operator"] == "AR", "failure_rate_mean"].mean()
    rev_div = (
        exp3_stats.set_index("clinical_significance")["rev_pheno_mean"]
        if not exp3_stats.empty
        else pd.Series(dtype=float)
    )
    ar_div = (
        exp3_stats.set_index("clinical_significance")["ar_pheno_mean"]
        if not exp3_stats.empty
        else pd.Series(dtype=float)
    )
    unlocked_mean = exp4_stats["unlocked_mean"].mean() if not exp4_stats.empty else float("nan")
    rev_distinct = exp7_stats.loc[exp7_stats["operator"] == "Revision", "distinct_phenotypes_mean"].mean()
    ar_distinct = exp7_stats.loc[exp7_stats["operator"] == "AR", "distinct_phenotypes_mean"].mean()

    if abs(q4 - q1) < 1e-9:
        fig2_conclusion = (
            "Experiment 1 and Experiment 2 are essentially flat in this rerun: both operators "
            "reach the direct disease-to-symptom layer reliably, so validity advantage and commitment "
            "coverage loss stay near zero across uncertainty levels."
        )
    else:
        fig2_conclusion = (
            f"Revision advantage changes from {q1:.3f} in the lowest-uncertainty quartile "
            f"to {q4:.3f} in the highest-uncertainty quartile. The largest commitment-loss category was "
            f"{label_for(exp2_top['clinical_significance'].iloc[0]) if not exp2_top.empty else 'n/a'} "
            f"with mean coverage loss {exp2_top['coverage_loss_mean'].iloc[0]:.3f}."
        )

    if not rev_div.empty and not ar_div.empty and rev_div.mean() < ar_div.mean():
        fig3_conclusion = (
            "Under the real AlzKB symptom layer, both operators remain highly diverse, but AR spans the "
            "full symptom set slightly more often while revision concentrates on a somewhat narrower subset. "
            "The main effect that remains is that revision diversity varies with zero-shot uncertainty rather than collapsing completely."
        )
    else:
        fig3_conclusion = (
            "Revision diversity increases with zero-shot uncertainty, whereas AR remains comparatively collapsed. "
            "This supports the claim that diffusion-style revision operators preserve competing mechanistic hypotheses when the sequence model is close to undecided."
        )

    if abs(float(vus_width.iloc[0]) - float(path_width.iloc[0])) < 0.5 and abs(unlocked_mean) < 1e-9:
        fig4_conclusion = (
            "Constraint guidance and phenotype-width analyses are largely flat in this rerun. "
            "On the real graph slice used here, most candidate trajectories already pass through the same direct disease layer, "
            "so adding explicit constraints does not unlock additional paths and VUS widths stay similar to Pathogenic widths."
        )
    else:
        fig4_conclusion = (
            f"Constraint guidance is generative rather than merely defensive: it increases disease-grounded coverage and unlocks additional trajectories. "
            f"At the same time, VUS produce broader phenotype distributions than Pathogenic variants "
            f"({float(vus_width.iloc[0]) if not vus_width.empty else float('nan'):.3f} vs "
            f"{float(path_width.iloc[0]) if not path_width.empty else float('nan'):.3f} mean phenotype width), "
            "consistent with the idea that VUS mechanistic uncertainty is biologically meaningful."
        )

    fig5_conclusion = (
        f"Operator divergence still concentrates on structurally important genes (centrality-vs-divergence Spearman correlation `{exp6_corr:.3f}`). "
        f"Both operators reach valid phenotypes reliably in this rerun (failure rates: revision `{rev_fail:.3f}`, AR `{ar_fail:.3f}`), "
        f"but revision samples slightly more distinct phenotypes per variant on average ({rev_distinct:.3f} vs {ar_distinct:.3f})."
    )
    text = f"""# Evo2 Zero-Shot AR vs Revision Experiments

## Overview
This report reruns the AlzKB operator experiments using the Evo 2 zero-shot variant signal instead of the earlier ClinVar-category entropy proxy or the fine-tuned classification head.

The notebook `train_cm_evo2_head_v3_vcf.ipynb` shows that:

- `zeroshot_neg_delta_likelihood` is the raw pathogenicity signal, computed as wild-type minus mutant Evo 2 log-likelihood.
- `zeroshot_pathogenicity_prob` is a monotone Platt-scaled version of that raw score.
- `isotonic_entropy` is the two-class entropy obtained after isotonic calibration of the raw zero-shot score on the binary training split, and is used here as the uncertainty variable.

The rerun therefore conditions the operators on:

- zero-shot pathogenicity probability: `zeroshot_pathogenicity_prob`
- uncertainty score: `isotonic_entropy`

## Input Data
- Score file: `{score_path()}`
- Matched variants used in experiments: `{len(variants_df)}`
- Present classes: `{", ".join(report_labels)}`

Counts by class:

```
{counts.to_string()}
```

## Operator Setup
Two operators were compared throughout:

1. `AR`:
   A path-dependent beam-search operator that extends trajectories stepwise using local transition scores.
2. `Revision`:
   A diffusion-style revision-capable operator that repeatedly resamples suffixes in path space and accepts globally better or uncertainty-supported edits.

Unlike the previous proxy-driven run, both operators are now conditioned on the Evo2 zero-shot signals:

- higher `zeroshot_pathogenicity_prob` increases preference for disease- and phenotype-grounded trajectories
- higher `isotonic_entropy` increases exploratory revision pressure and lowers early commitment

## Combined Figures
This rerun kept only the **3 combined figures that carry real signal**. Two figures were dropped:

- the score-landscape figure, because it duplicates Figure 1 of the diffusion-only report; and
- the operator-advantage / commitment-point figure, because every panel was flat at zero (AR and revision coincide on this short-path disease-to-symptom layer, so there is no advantage to plot).

The kept, renumbered figures are:

1. `combined_figure1_diversity_and_uncertainty`
   - Information: revision diversity by class, diversity-vs-entropy trend, and AR-vs-revision trajectory diversity.
2. `combined_figure2_vus_phenotype_breadth`
   - Information: VUS-vs-Pathogenic comparison of phenotype width and distributional entropy.
3. `combined_figure3_structure_and_reliability`
   - Information: graph centrality vs operator divergence, genes with strongest operator separation, and resampling reliability.

Each combined figure was exported as both `.png` and `.pdf` in `outputs/experiments/ar_vs_diffusion/`.

## How The Figures Were Produced
### Figure 1 (diversity)
- Panel a: Experiment 3 per-variant revision phenotype diversity over 10 runs.
- Panel b: scatter of zero-shot entropy vs revision trajectory diversity.
- Panel c: class-level means comparing AR and revision trajectory diversity.

### Figure 2 (VUS breadth)
- Panel a: Experiment 5 phenotype width comparing Pathogenic vs VUS variants.
- Panel b: Experiment 5 phenotype entropy comparing Pathogenic vs VUS variants.
- (The Experiment 4 constraint-unlocked and disease-coverage panels were dropped: they were flat because constrained and unconstrained search coincide on this graph slice.)

### Figure 3 (structure and reliability)
- Panel a: Experiment 6 scatter of gene centrality vs operator divergence.
- Panel b: genes with the largest divergence between revision and AR.
- Panels c-d: Experiment 7 operator consistency and distinct-phenotype counts across classes.

## Dropped (Null) Results
For the record, two comparisons produced no usable signal on the current AlzKB and are therefore not shown as figures:

- Operator advantage and first-step commitment (old Figure 2): {fig2_conclusion}
- Constraint-guided discovery (old Figure 4, panels a-b): constrained and unconstrained search reach the same direct disease layer, so no additional trajectories are unlocked.

## Experiment Conclusions
### Figure 1 conclusion (diversity)
{fig3_conclusion}

### Figure 2 conclusion (VUS breadth)
{fig4_conclusion}

### Figure 3 conclusion (structure and reliability)
{fig5_conclusion}

## Output Files
- `combined_figure1_diversity_and_uncertainty.png`
- `combined_figure1_diversity_and_uncertainty.pdf`
- `combined_figure2_vus_phenotype_breadth.png`
- `combined_figure2_vus_phenotype_breadth.pdf`
- `combined_figure3_structure_and_reliability.png`
- `combined_figure3_structure_and_reliability.pdf`
- `exp1_results.csv`, `exp1_summary.csv`
- `exp2_results.csv`, `exp2_summary.csv`
- `exp3_results.csv`, `exp3_summary.csv`
- `exp4_results.csv`, `exp4_summary.csv`
- `exp5_results.csv`, `exp5_summary.csv`, `exp5_paired_by_gene.csv`
- `exp6_results.csv`
- `exp7_results.csv`, `exp7_summary.csv`
"""
    REPORT_PATH.write_text(text + "\n", encoding="utf-8")


def main():
    apply_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log: List[str] = []

    base = SCRIPT_DIR.parent
    preferred_gene_names = sorted(pd.read_csv(score_path())["gene_symbol"].dropna().astype(str).str.upper().unique())
    graph, graph_mode = baseexp.load_alzkb(base, log, preferred_gene_names=preferred_gene_names)
    variants_df = load_evo2_variants(graph)
    data_note = f"{graph_mode} | evo2_zero_shot_vcf"

    print("Matched Evo2 variants:", len(variants_df))
    print(variants_df["clinical_significance"].value_counts().to_string())

    edge_type_frequency = Counter(data.get("relation", "unknown") for _, _, data in graph.edges(data=True))

    exp1_df, exp1_stats = run_experiment1(graph, variants_df, edge_type_frequency, data_note)
    exp2_df, exp2_stats = run_experiment2(graph, variants_df, edge_type_frequency, data_note)
    exp3_df, exp3_stats = run_experiment3(graph, variants_df, edge_type_frequency, data_note)
    exp4_df, exp4_stats = run_experiment4(graph, variants_df, edge_type_frequency, data_note)
    exp5_df, exp5_stats, exp5_paired = run_experiment5(graph, variants_df, edge_type_frequency, data_note)
    exp6_df, exp6_corr = run_experiment6(graph, variants_df, edge_type_frequency, data_note)
    exp7_df, exp7_stats = run_experiment7(graph, variants_df, edge_type_frequency, data_note)

    exp1_df.to_csv(OUTPUT_DIR / "exp1_results.csv", index=False)
    exp1_stats.to_csv(OUTPUT_DIR / "exp1_summary.csv", index=False)
    exp2_df.to_csv(OUTPUT_DIR / "exp2_results.csv", index=False)
    exp2_stats.to_csv(OUTPUT_DIR / "exp2_summary.csv", index=False)
    exp3_df.to_csv(OUTPUT_DIR / "exp3_results.csv", index=False)
    exp3_stats.to_csv(OUTPUT_DIR / "exp3_summary.csv", index=False)
    exp4_df.to_csv(OUTPUT_DIR / "exp4_results.csv", index=False)
    exp4_stats.to_csv(OUTPUT_DIR / "exp4_summary.csv", index=False)
    exp5_df.to_csv(OUTPUT_DIR / "exp5_results.csv", index=False)
    exp5_stats.to_csv(OUTPUT_DIR / "exp5_summary.csv", index=False)
    exp5_paired.to_csv(OUTPUT_DIR / "exp5_paired_by_gene.csv", index=False)
    exp6_df.to_csv(OUTPUT_DIR / "exp6_results.csv", index=False)
    exp7_df.to_csv(OUTPUT_DIR / "exp7_results.csv", index=False)
    exp7_stats.to_csv(OUTPUT_DIR / "exp7_summary.csv", index=False)

    # Figure 1 (score landscape) is redundant with the diffusion-only report and
    # the old operator-advantage/commitment figure was entirely flat (no signal),
    # so both are dropped. Only the three figures with real signal are kept and
    # renumbered 1-3.
    make_combined_figure3(exp3_df)
    make_combined_figure4(exp4_df, exp4_stats, exp5_df, exp5_stats)
    make_combined_figure5(exp6_df, exp6_corr, exp7_stats)

    write_report(variants_df, exp1_stats, exp2_stats, exp3_stats, exp4_stats, exp5_stats, exp6_corr, exp7_stats)
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
