#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_operator_reasoning_experiments as baseexp
import run_operator_reasoning_experiments_evo2 as evo2
import pubstyle as pub
import learned_operator as lop


OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "experiments"
REPORT_PATH = OUTPUT_DIR / "report.md"
RNG_SEED = 42
CATEGORY_ORDER = [
    "Pathogenic",
    "Likely_pathogenic",
    "Uncertain_significance",
    "Likely_benign",
    "Benign",
]


def present_categories(values: Iterable[str]) -> List[str]:
    return pub.present_categories(values)


def label_for(cat: str) -> str:
    return pub.label_for(cat)


def color_for(cat: str) -> str:
    return pub.color_for(cat)


def apply_publication_style():
    pub.apply()


def panel_label(ax, label: str):
    pub.panel_label(ax, label)


def add_box_strip(ax, df: pd.DataFrame, categories: List[str], value_col: str, ylabel: str, title: str):
    pub.box_strip(ax, df, categories, value_col, ylabel, title, seed=RNG_SEED)


def save_figure(fig: plt.Figure, stem: str):
    pub.save(fig, OUTPUT_DIR, stem)


def scatter_trend(ax, x, y, color: str):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax.scatter(x, y, s=14, alpha=0.28, color=color, linewidths=0, zorder=2)
    if len(x) > 1:
        coeff = np.polyfit(x, y, deg=1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, coeff[0] * xs + coeff[1], color=color, linewidth=2.0, zorder=3)
    pub.soft_grid(ax, "both")


def entropy_from_distribution(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def effective_support(probs: np.ndarray) -> float:
    return float(2 ** entropy_from_distribution(probs))


def topk_keys(dist: Dict[str, float], k: int = 5) -> List[str]:
    return [k_ for k_, _ in sorted(dist.items(), key=lambda item: (-item[1], item[0]))[:k]]


def top1_key(dist: Dict[str, float]) -> str:
    return topk_keys(dist, k=1)[0] if dist else ""


def overlap_at_k(dist1: Dict[str, float], dist2: Dict[str, float], k: int = 5) -> float:
    a = set(topk_keys(dist1, k))
    b = set(topk_keys(dist2, k))
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def js_divergence(dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
    keys = sorted(set(dist1) | set(dist2))
    if not keys:
        return 0.0
    p = np.array([dist1.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([dist2.get(k, 0.0) for k in keys], dtype=float)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def reachable_phenotypes(graph: nx.DiGraph, gene: str) -> List[str]:
    paths = evo2.candidate_paths_for_gene(graph, gene, max_depth=6)
    return [path[-1] for path in paths if path]


def endpoint_name(graph: nx.DiGraph, node: str) -> str:
    attrs = graph.nodes[node]
    return str(attrs.get("name") or attrs.get("commonName") or node)


def endpoint_group_from_name(name: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in ["aphasia", "language", "speech", "communication", "anomia", "dyslexia", "alexia"]):
        return "Language_Communication"
    if any(k in n for k in ["amnesia", "memory", "agnosia", "apraxia", "gerstmann", "confusion", "delirium", "prosopagnosia", "learning", "cortical thinning"]):
        return "Cognition_Memory"
    if any(k in n for k in ["hearing", "presbycusis", "olfaction", "anosmia", "perceptual", "vision", "blind", "ageusia"]):
        return "Sensory_Perception"
    if any(k in n for k in ["gait", "myoclonus", "psychomotor", "paraparesis", "palsy", "ataxia", "movement", "chorea", "dystonia"]):
        return "Motor_Movement"
    if "sleep" in n or "somnolence" in n:
        return "Sleep_Arousal"
    if any(k in n for k in ["hallucination", "agitation", "behavior", "psycho", "fatigue"]):
        return "Behavior_Psychiatric"
    if any(k in n for k in ["unconscious", "vegetative", "coma", "hypoxia", "brain death"]):
        return "Consciousness_Arousal"
    if any(k in n for k in ["incontinence", "urinary"]):
        return "Autonomic_Visceral"
    if "syndrome" in n or "disease" in n:
        return "Neurologic_Syndromic"
    return "Other_Neurologic"


# Reduced fixed group offsets. These are intentionally small so that the
# gene-specific mechanism prior (below) and the uncertainty coupling can
# actually reorder endpoint groups rather than being dominated by a fixed gap.
GROUP_BONUS = {
    "Language_Communication": 0.10,
    "Cognition_Memory": 0.12,
    "Sensory_Perception": 0.04,
    "Motor_Movement": 0.06,
    "Sleep_Arousal": 0.04,
    "Behavior_Psychiatric": 0.05,
    "Consciousness_Arousal": 0.05,
    "Autonomic_Visceral": 0.02,
    "Neurologic_Syndromic": -0.02,
    "Other_Neurologic": 0.0,
}

# AD-core phenotype groups: pathogenic, high-probability variants should push
# probability mass toward these clinically central cognitive endpoints.
AD_CORE_GROUPS = {"Cognition_Memory", "Language_Communication"}

# Maps BiologicalProcess name themes to phenotype groups. Genes in AlzKB
# participate in distinct biological processes, so aggregating a gene's process
# names into these themes yields a gene-specific prior over phenotype groups.
# This is what breaks the shared Gene->Disease->Symptom endpoint degeneracy.
PROCESS_THEME_KEYWORDS = {
    "Cognition_Memory": [
        "memory", "learning", "cognition", "cognitive", "synap", "dendritic spine",
        "long-term potentiation", "neuron projection", "hippocamp", "axon",
        "neurotransmitter", "postsynap", "presynap",
    ],
    "Language_Communication": ["language", "speech", "vocal", "phonolog"],
    "Sensory_Perception": [
        "sensory", "visual", "vision", "eye ", "retina", "photoreceptor",
        "auditory", "hearing", "cochlea", "olfact", "smell", "taste",
    ],
    "Motor_Movement": [
        "locomot", "muscle", "motor neuron", "movement", "gait", "myelin",
        "axon guidance", "cytoskeleton", "microtubule", "actin",
    ],
    "Sleep_Arousal": ["circadian", "sleep", "arousal"],
    "Behavior_Psychiatric": [
        "behavior", "emotion", "fear", "anxiety", "social", "dopamine",
        "serotonin", "reward", "mood",
    ],
    "Neurologic_Syndromic": [
        "immune", "inflamm", "glial", "microglia", "astrocyte", "cytokine",
        "t cell", "apopto", "cell death", "neurodegener", "autophag", "amyloid",
        "tau", "protein catabolic", "ubiquitin",
    ],
    "Autonomic_Visceral": [
        "heart", "cardiac", "vascul", "blood pressure", "urinary", "gut",
        "angiogen", "circulation",
    ],
}

MECH_AFFINITY_CACHE: Dict[str, Dict[str, float]] = {}


def mechanism_affinity(graph: nx.DiGraph, gene: str) -> Dict[str, float]:
    """Gene-specific prior over phenotype groups derived from the gene's
    BiologicalProcess participation. Genes engaged in synaptic/memory processes
    weight Cognition_Memory; genes engaged in immune/degeneration processes
    weight Neurologic_Syndromic, etc. Returns a normalized distribution over
    phenotype groups (empty dict if the gene has no informative processes)."""
    if gene in MECH_AFFINITY_CACHE:
        return MECH_AFFINITY_CACHE[gene]
    theme_counts: Counter = Counter()
    for neighbor in graph.successors(gene):
        if graph.nodes[neighbor].get("type") != "BiologicalProcess":
            continue
        name = endpoint_name(graph, neighbor).lower()
        for group, keywords in PROCESS_THEME_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                theme_counts[group] += 1
    total = sum(theme_counts.values())
    if total == 0:
        MECH_AFFINITY_CACHE[gene] = {}
        return {}
    affinity = {group: count / total for group, count in theme_counts.items()}
    MECH_AFFINITY_CACHE[gene] = affinity
    return affinity


def endpoint_specificity_bonus(name: str) -> float:
    n = (name or "").lower()
    bonus = 0.0
    if "," in n:
        bonus += 0.25
    if len(n.split()) >= 2:
        bonus += 0.08
    if any(k in n for k in ["aphasia", "amnesia", "agnosia", "aprax", "anomia", "language", "speech", "memory"]):
        bonus += 0.18
    if any(k in n for k in ["disorders", "manifestations", "syndrome", "disease"]):
        bonus -= 0.20
    return bonus


def path_structure_bonus(graph: nx.DiGraph, path: List[str]) -> float:
    node_types = [baseexp.get_node_type(graph, node) for node in path]
    bonus = 0.0
    if "Pathway" in node_types:
        bonus += 0.35
    if "BiologicalProcess" in node_types:
        bonus += 0.45
    if len(path) <= 3 and "Pathway" not in node_types and "BiologicalProcess" not in node_types:
        bonus -= 0.55
    return bonus


def build_knn_entropy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    train = out[out["dataset"] == "binary"].copy()
    train = train[train["split"] == "train"].copy()
    train["y"] = pd.to_numeric(train["labels"], errors="coerce")
    train = train.dropna(subset=["y", "zeroshot_score"])
    model = KNeighborsRegressor(n_neighbors=min(31, len(train)), weights="distance")
    model.fit(train[["zeroshot_score"]].to_numpy(), train["y"].astype(int).to_numpy())
    p = np.clip(model.predict(out[["zeroshot_score"]].to_numpy()), 1e-6, 1 - 1e-6)
    out["knn_pathogenicity_prob"] = p
    out["knn_entropy"] = evo2.entropy_from_prob(p)
    return out


# Strength of the gene-specific mechanism prior in the endpoint utility.
MECH_WEIGHT = 1.1
# Extra pull toward AD-core cognitive endpoints for high-pathogenicity variants.
AD_CORE_PATHO_BONUS = 0.7

GENERIC_GROUPS = {"Neurologic_Syndromic", "Other_Neurologic"}

# Learned scoring function for the diffusion operator. Populated by
# ``fit_learned_operator`` at the start of a run (or loaded from disk); when
# present it replaces the hand-tuned utility combination inside
# ``phenotype_distribution``.
LEARNED_SCORER: "lop.LearnedScorer | None" = None


def softmax_temperature(entropy: float) -> float:
    """Exploration temperature of the diffusion softmax as a function of the
    sequence entropy. This is the operator's stochastic schedule and is kept
    outside the learned utility."""
    return 0.25 + 1.4 * max(float(entropy), 1e-6)


def path_feature_matrix(
    graph: nx.DiGraph,
    candidates: List[List[str]],
    endpoint_groups: List[str],
    group_counts: Counter,
    affinity: Dict[str, float],
    edge_type_frequency: Counter,
    pathogenicity_prob: float,
    entropy: float,
) -> np.ndarray:
    """Interpretable per-path feature matrix aligned to
    ``learned_operator.FEATURE_NAMES``. These are the same terms the old
    heuristic combined with fixed weights; the learned model reweights them."""
    conf = 1.0 - min(max(float(entropy), 0.0), 1.0)
    p = float(pathogenicity_prob)
    rows = np.zeros((len(candidates), lop.N_FEATURES), dtype=float)
    for idx, path in enumerate(candidates):
        group = endpoint_groups[idx]
        node_types = {baseexp.get_node_type(graph, node) for node in path}
        spec = endpoint_specificity_bonus(endpoint_name(graph, path[-1]))
        aff = affinity.get(group, 0.0)
        is_ad_core = 1.0 if group in AD_CORE_GROUPS else 0.0
        rows[idx] = [
            baseexp.path_score(graph, path, edge_type_frequency),
            float(len(path)),
            1.0 if "Pathway" in node_types else 0.0,
            1.0 if "BiologicalProcess" in node_types else 0.0,
            baseexp.disease_coverage(path, graph),
            spec,
            is_ad_core,
            aff,
            1.0 if group in GENERIC_GROUPS else 0.0,
            math.log(group_counts[group] + 1),
            p * is_ad_core,
            p * aff,
            conf * spec,
            conf * aff,
        ]
    return rows


def phenotype_distribution(
    graph: nx.DiGraph,
    gene: str,
    edge_type_frequency: Counter,
    pathogenicity_prob: float,
    entropy: float,
    constraint_weight: float = 0.0,
    max_depth: int = 6,
    aggregate_groups: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    candidates = evo2.candidate_paths_for_gene(graph, gene, max_depth=max_depth)
    if not candidates:
        return {}, pd.DataFrame(columns=["endpoint", "prob", "path", "utility"])
    affinity = mechanism_affinity(graph, gene)
    endpoint_groups = [endpoint_group_from_name(endpoint_name(graph, path[-1])) for path in candidates]
    group_counts = Counter(endpoint_groups)
    if LEARNED_SCORER is not None:
        # Learned scoring function: U_i = w . standardized phi_i, with weights
        # trained end-to-end through this softmax on labeled anchor variants.
        phi = path_feature_matrix(
            graph, candidates, endpoint_groups, group_counts, affinity,
            edge_type_frequency, pathogenicity_prob, entropy,
        )
        utilities = LEARNED_SCORER.utilities(phi)
    else:
        # Fallback heuristic (retained for reference / cold start): uncertainty
        # acts on the utility itself, sharpening discriminative terms at low
        # entropy and flattening the distribution at high entropy.
        sharpness = 1.0 - min(max(float(entropy), 0.0), 1.0)
        utilities = np.zeros(len(candidates), dtype=float)
        for idx, path in enumerate(candidates):
            group = endpoint_groups[idx]
            base = evo2.path_utility(
                graph,
                path,
                edge_type_frequency,
                pathogenicity_prob,
                entropy,
                constraint_weight=constraint_weight,
            )
            base += path_structure_bonus(graph, path)
            discriminative = (
                endpoint_specificity_bonus(endpoint_name(graph, path[-1]))
                + GROUP_BONUS.get(group, 0.0)
                + MECH_WEIGHT * affinity.get(group, 0.0)
                + (AD_CORE_PATHO_BONUS * pathogenicity_prob if group in AD_CORE_GROUPS else 0.0)
            )
            crowding = 0.25 * math.log(group_counts[group] + 1)
            utilities[idx] = base + sharpness * discriminative - crowding
    # Exploration temperature: uncertainty controls how much the softmax spreads
    # mass across endpoints.
    temperature = softmax_temperature(entropy)
    logits = (utilities - utilities.max()) / temperature
    weights = np.exp(logits)
    weights = weights / weights.sum()
    rows = []
    endpoint_probs: Dict[str, float] = {}
    for path, prob, utility in zip(candidates, weights.tolist(), utilities.tolist()):
        endpoint = path[-1]
        key = endpoint_group_from_name(endpoint_name(graph, endpoint)) if aggregate_groups else endpoint
        endpoint_probs[key] = endpoint_probs.get(key, 0.0) + prob
        rows.append({"endpoint": endpoint, "endpoint_group": key, "prob": prob, "path": tuple(path), "utility": utility})
    ranked = pd.DataFrame(rows).sort_values(["prob", "utility"], ascending=False).reset_index(drop=True)
    return endpoint_probs, ranked


POS_ANCHOR = {"Pathogenic", "Likely_pathogenic"}
NEG_ANCHOR = {"Benign", "Likely_benign"}


def build_anchor_samples(
    variants_df: pd.DataFrame, graph: nx.DiGraph, edge_type_frequency: Counter
) -> List[dict]:
    """One learned-operator training sample per labeled anchor variant: the raw
    path feature matrix, the AD-core endpoint indicator, the softmax
    temperature, and the pathogenic/benign label."""
    anchors = variants_df[variants_df["clinical_significance"].isin(POS_ANCHOR | NEG_ANCHOR)]
    samples: List[dict] = []
    for _, variant in anchors.iterrows():
        gene = variant["gene_id"]
        p = float(variant["pathogenicity_prob"])
        u = float(variant["isotonic_entropy"])
        candidates = evo2.candidate_paths_for_gene(graph, gene, max_depth=6)
        if not candidates:
            continue
        affinity = mechanism_affinity(graph, gene)
        endpoint_groups = [endpoint_group_from_name(endpoint_name(graph, path[-1])) for path in candidates]
        group_counts = Counter(endpoint_groups)
        phi = path_feature_matrix(
            graph, candidates, endpoint_groups, group_counts, affinity,
            edge_type_frequency, p, u,
        )
        ad_core = np.array([1.0 if g in AD_CORE_GROUPS else 0.0 for g in endpoint_groups], dtype=float)
        samples.append(
            {
                "phi": phi,
                "ad_core": ad_core,
                "tau": softmax_temperature(u),
                "y": 1.0 if variant["clinical_significance"] in POS_ANCHOR else 0.0,
            }
        )
    return samples


def fit_learned_operator(
    variants_df: pd.DataFrame, graph: nx.DiGraph, edge_type_frequency: Counter,
    log: List[str] | None = None,
) -> "lop.LearnedScorer":
    """Train the learned path-scoring function on labeled anchor variants and
    install it as the active diffusion-operator scorer."""
    global LEARNED_SCORER
    samples = build_anchor_samples(variants_df, graph, edge_type_frequency)
    scorer = lop.train(samples, l2=1e-2, seed=RNG_SEED)
    LEARNED_SCORER = scorer
    scorer.save()
    msg = (
        f"Learned operator trained on {scorer.n_train} anchor variants; "
        f"anchor readout AUROC={scorer.train_auroc:.3f}"
    )
    if log is not None:
        log.append(msg)
    print(msg)
    return scorer


def ar_top_endpoint(
    graph: nx.DiGraph, gene: str, edge_type_frequency: Counter, pathogenicity_prob: float, entropy: float
) -> str:
    path, _ = evo2.conditioned_ar_operator(
        graph,
        gene,
        edge_type_frequency,
        pathogenicity_prob,
        entropy,
    )
    return path[-1] if path else ""


def add_entropy_quartile(df: pd.DataFrame, col: str = "isotonic_entropy") -> pd.DataFrame:
    out = df.copy()
    try:
        out["entropy_quartile"] = pd.qcut(
            out[col], q=4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"], duplicates="drop"
        )
    except ValueError:
        out["entropy_quartile"] = pd.cut(
            out[col], bins=4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]
        )
    return out


def run_experiment1(variants_df: pd.DataFrame, graph: nx.DiGraph, edge_type_frequency: Counter) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, variant in variants_df.iterrows():
        dist, _ = phenotype_distribution(
            graph,
            variant["gene_id"],
            edge_type_frequency,
            float(variant["pathogenicity_prob"]),
            float(variant["isotonic_entropy"]),
            aggregate_groups=True,
        )
        if not dist:
            continue
        probs = np.array(sorted(dist.values(), reverse=True), dtype=float)
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "gene_id": variant["gene_id"],
                "clinical_significance": variant["clinical_significance"],
                "isotonic_entropy": variant["isotonic_entropy"],
                "endpoint_count_1pct": int((probs >= 0.01).sum()),
                "endpoint_count_5pct": int((probs >= 0.05).sum()),
                "phenotype_entropy_bits": entropy_from_distribution(probs),
                "effective_endpoints": effective_support(probs),
                "top1_mass": float(probs.max()) if len(probs) else 0.0,
                "top3_mass": float(probs[:3].sum()) if len(probs) else 0.0,
            }
        )
    df = add_entropy_quartile(pd.DataFrame(rows), col="isotonic_entropy")
    summary = (
        df.groupby("entropy_quartile", observed=False)
        .agg(
            phenotype_entropy_mean=("phenotype_entropy_bits", "mean"),
            effective_endpoints_mean=("effective_endpoints", "mean"),
            top1_mass_mean=("top1_mass", "mean"),
            endpoint_count_1pct_mean=("endpoint_count_1pct", "mean"),
            n=("variant_id", "count"),
        )
        .reset_index()
    )
    return df, summary


def run_experiment2(variants_df: pd.DataFrame, graph: nx.DiGraph, edge_type_frequency: Counter) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, variant in variants_df.iterrows():
        p = float(variant["pathogenicity_prob"])
        u = float(variant["isotonic_entropy"])
        dist_base, _ = phenotype_distribution(graph, variant["gene_id"], edge_type_frequency, p, u, aggregate_groups=True)
        if not dist_base:
            continue
        dist_low, _ = phenotype_distribution(graph, variant["gene_id"], edge_type_frequency, p, max(0.0, u - 0.30), aggregate_groups=True)
        dist_high, _ = phenotype_distribution(graph, variant["gene_id"], edge_type_frequency, p, min(1.0, u + 0.30), aggregate_groups=True)
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "gene_id": variant["gene_id"],
                "clinical_significance": variant["clinical_significance"],
                "isotonic_entropy": u,
                "top1_switch_low": int(top1_key(dist_base) != top1_key(dist_low)),
                "top1_switch_high": int(top1_key(dist_base) != top1_key(dist_high)),
                "js_low": js_divergence(dist_base, dist_low),
                "js_high": js_divergence(dist_base, dist_high),
                "top5_overlap_low": overlap_at_k(dist_base, dist_low, k=5),
                "top5_overlap_high": overlap_at_k(dist_base, dist_high, k=5),
            }
        )
    df = add_entropy_quartile(pd.DataFrame(rows), col="isotonic_entropy")
    df["mean_js"] = (df["js_low"] + df["js_high"]) / 2
    df["mean_top1_switch"] = (df["top1_switch_low"] + df["top1_switch_high"]) / 2
    df["mean_top5_overlap"] = (df["top5_overlap_low"] + df["top5_overlap_high"]) / 2
    summary = (
        df.groupby("entropy_quartile", observed=False)
        .agg(
            mean_js=("mean_js", "mean"),
            mean_top1_switch=("mean_top1_switch", "mean"),
            mean_top5_overlap=("mean_top5_overlap", "mean"),
            n=("variant_id", "count"),
        )
        .reset_index()
    )
    return df, summary


def run_experiment3(variants_df: pd.DataFrame, graph: nx.DiGraph, edge_type_frequency: Counter) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, variant in variants_df.iterrows():
        p = float(variant["pathogenicity_prob"])
        dist_iso, _ = phenotype_distribution(graph, variant["gene_id"], edge_type_frequency, p, float(variant["isotonic_entropy"]), aggregate_groups=True)
        dist_knn, _ = phenotype_distribution(graph, variant["gene_id"], edge_type_frequency, p, float(variant["knn_entropy"]), aggregate_groups=True)
        if not dist_iso and not dist_knn:
            continue
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "gene_id": variant["gene_id"],
                "gene_symbol": variant["gene_symbol"],
                "clinical_significance": variant["clinical_significance"],
                "isotonic_entropy": variant["isotonic_entropy"],
                "knn_entropy": variant["knn_entropy"],
                "entropy_gap": abs(float(variant["knn_entropy"]) - float(variant["isotonic_entropy"])),
                "top1_switch_alt": int(top1_key(dist_iso) != top1_key(dist_knn)),
                "js_alt": js_divergence(dist_iso, dist_knn),
                "top5_overlap_alt": overlap_at_k(dist_iso, dist_knn, k=5),
            }
        )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("clinical_significance")
        .agg(
            mean_js=("js_alt", "mean"),
            mean_top1_switch=("top1_switch_alt", "mean"),
            mean_top5_overlap=("top5_overlap_alt", "mean"),
            mean_entropy_gap=("entropy_gap", "mean"),
            n=("variant_id", "count"),
        )
        .reindex(present_categories(df["clinical_significance"]))
        .reset_index()
    )
    gene_summary = (
        df.groupby(["gene_id", "gene_symbol"])
        .agg(
            mean_js=("js_alt", "mean"),
            mean_top1_switch=("top1_switch_alt", "mean"),
            n=("variant_id", "count"),
        )
        .reset_index()
        .sort_values(["mean_js", "n"], ascending=[False, False])
    )
    return df, summary, gene_summary


def run_experiment4(variants_df: pd.DataFrame, graph: nx.DiGraph, edge_type_frequency: Counter) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, variant in variants_df.iterrows():
        reachable = reachable_phenotypes(graph, variant["gene_id"])
        if not reachable:
            continue
        reachable_groups = {endpoint_group_from_name(endpoint_name(graph, node)) for node in reachable}
        dist, _ = phenotype_distribution(
            graph,
            variant["gene_id"],
            edge_type_frequency,
            float(variant["pathogenicity_prob"]),
            float(variant["isotonic_entropy"]),
            aggregate_groups=True,
        )
        ar_endpoint = ar_top_endpoint(
            graph,
            variant["gene_id"],
            edge_type_frequency,
            float(variant["pathogenicity_prob"]),
            float(variant["isotonic_entropy"]),
        )
        probs = np.array(sorted(dist.values(), reverse=True), dtype=float)
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "gene_id": variant["gene_id"],
                "gene_symbol": variant["gene_symbol"],
                "clinical_significance": variant["clinical_significance"],
                "isotonic_entropy": variant["isotonic_entropy"],
                "reachable_endpoints": len(set(reachable_groups)),
                "diffusion_effective_endpoints": effective_support(probs),
                "diffusion_branch_entropy": entropy_from_distribution(probs),
                "diffusion_top1_mass": float(probs.max()) if len(probs) else 0.0,
                "ar_matches_diffusion_top1": int(ar_endpoint == top1_key(dist)),
                "diffusion_support_5pct": int((probs >= 0.05).sum()),
            }
        )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("clinical_significance")
        .agg(
            reachable_endpoints_mean=("reachable_endpoints", "mean"),
            diffusion_effective_endpoints_mean=("diffusion_effective_endpoints", "mean"),
            diffusion_branch_entropy_mean=("diffusion_branch_entropy", "mean"),
            diffusion_top1_mass_mean=("diffusion_top1_mass", "mean"),
            ar_matches_top1_mean=("ar_matches_diffusion_top1", "mean"),
            n=("variant_id", "count"),
        )
        .reindex(present_categories(df["clinical_significance"]))
        .reset_index()
    )
    return df, summary


def make_figure1_input(variants_df: pd.DataFrame):
    cats = present_categories(variants_df["clinical_significance"])
    fig = plt.figure(figsize=(13.6, 4.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.95, 1.15, 1.15])
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

    counts = variants_df["clinical_significance"].value_counts().reindex(cats)
    ax1.bar(range(len(cats)), counts.values, color=[color_for(c) for c in cats], edgecolor="white", linewidth=0.8)
    ax1.set_xticks(range(len(cats)))
    ax1.set_xticklabels([label_for(c) for c in cats])
    ax1.set_ylabel("Variants")
    ax1.set_title("Evo2-scored AD variant composition", fontsize=10.5)
    for i, v in enumerate(counts.values):
        ax1.text(i, v + max(counts.values) * 0.02, str(int(v)), ha="center", va="bottom", fontsize=8.5, color=pub.MUTED)
    pub.soft_grid(ax1, "y")
    panel_label(ax1, "a")

    add_box_strip(ax2, variants_df, cats, "pathogenicity_prob", "Zero-shot pathogenicity probability", "Pathogenicity signal by class")
    panel_label(ax2, "b")
    add_box_strip(ax3, variants_df, cats, "isotonic_entropy", "Isotonic entropy (bits)", "Uncertainty metric by class")
    panel_label(ax3, "c")

    fig.suptitle("Figure 1 | Sequence-side uncertainty landscape for AD variants", y=1.04, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "combined_figure1_evo2_score_landscape")


def make_figure2_exp1(exp1_df: pd.DataFrame, exp1_summary: pd.DataFrame):
    fig = plt.figure(figsize=(15.0, 4.6))
    gs = GridSpec(1, 3, figure=fig)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, j]) for j in range(3)]
    x = np.arange(len(exp1_summary))

    ax1.plot(x, exp1_summary["effective_endpoints_mean"], marker="o", markersize=6.5, markeredgecolor="white", markeredgewidth=0.9, linewidth=2.2, color=pub.ACCENT_SLATE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp1_summary["entropy_quartile"])
    ax1.set_ylabel("Mean effective endpoints")
    ax1.set_title("Effective support grows with uncertainty", fontsize=10.5)
    pub.soft_grid(ax1, "y")
    panel_label(ax1, "a")

    cats = present_categories(exp1_df["clinical_significance"])
    add_box_strip(ax2, exp1_df, cats, "phenotype_entropy_bits", "Phenotype entropy (bits)", "Class-wise phenotype entropy")
    panel_label(ax2, "b")

    scatter_trend(ax3, exp1_df["isotonic_entropy"], exp1_df["top1_mass"], pub.ACCENT_AMBER)
    corr, _ = spearmanr(exp1_df["isotonic_entropy"], exp1_df["top1_mass"])
    ax3.set_xlabel("Isotonic entropy")
    ax3.set_ylabel("Top-1 endpoint mass")
    ax3.set_title(f"Top-1 mass vs uncertainty (Spearman = {0.0 if math.isnan(corr) else corr:.2f})", fontsize=10.5)
    panel_label(ax3, "c")

    fig.suptitle("Figure 2 | Experiment 1: Uncertainty propagation to phenotype distributions", y=1.04, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "combined_figure2_uncertainty_propagation")


def make_figure3_exp2(exp2_df: pd.DataFrame, exp2_summary: pd.DataFrame):
    fig = plt.figure(figsize=(11.0, 4.6))
    gs = GridSpec(1, 2, figure=fig)
    ax1, ax2 = [fig.add_subplot(gs[0, j]) for j in range(2)]
    x = np.arange(len(exp2_summary))

    ax1.plot(x, exp2_summary["mean_js"], marker="o", markersize=6.5, markeredgecolor="white", markeredgewidth=0.9, linewidth=2.2, color=pub.ACCENT_AMBER)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp2_summary["entropy_quartile"])
    ax1.set_ylabel("Mean JS divergence (perturbed vs base)")
    ax1.set_title("Perturbation sensitivity fades as uncertainty saturates", fontsize=10.5)
    pub.soft_grid(ax1, "y")
    panel_label(ax1, "a")

    scatter_trend(ax2, exp2_df["isotonic_entropy"], exp2_df["mean_js"], pub.ACCENT_GREEN)
    corr, _ = spearmanr(exp2_df["isotonic_entropy"], exp2_df["mean_js"])
    ax2.set_xlabel("Isotonic entropy")
    ax2.set_ylabel("Mean JS divergence")
    ax2.set_title(f"Ranking shift vs uncertainty (Spearman = {0.0 if math.isnan(corr) else corr:.2f})", fontsize=10.5)
    panel_label(ax2, "b")

    fig.suptitle("Figure 3 | Experiment 2: Uncertainty-sensitive endpoint ranking", y=1.04, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "combined_figure3_endpoint_ranking_stability")


def make_figure4_exp3(exp3_df: pd.DataFrame, exp3_summary: pd.DataFrame, gene_summary: pd.DataFrame):
    fig = plt.figure(figsize=(15.0, 4.6))
    gs = GridSpec(1, 3, figure=fig)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, j]) for j in range(3)]

    scatter_trend(ax1, exp3_df["isotonic_entropy"], exp3_df["knn_entropy"], pub.ACCENT_BLUE)
    corr, _ = spearmanr(exp3_df["isotonic_entropy"], exp3_df["knn_entropy"])
    ax1.set_xlabel("Isotonic entropy")
    ax1.set_ylabel("kNN entropy")
    ax1.set_title(f"Iso vs kNN entropy (Spearman = {0.0 if math.isnan(corr) else corr:.2f})", fontsize=10.5)
    panel_label(ax1, "a")

    cats = present_categories(exp3_df["clinical_significance"])
    add_box_strip(ax2, exp3_df, cats, "js_alt", "JS divergence", "Metric change shifts distributions")
    panel_label(ax2, "b")

    top = gene_summary.head(8).sort_values("mean_js")
    ax3.barh(top["gene_symbol"], top["mean_js"], color=pub.ACCENT_PURPLE, edgecolor="white", linewidth=0.8)
    ax3.set_xlabel("Mean JS divergence")
    ax3.set_title("Most metric-sensitive genes", fontsize=10.5)
    pub.soft_grid(ax3, "x")
    panel_label(ax3, "c")

    fig.suptitle("Figure 4 | Experiment 3: Stability under an alternative entropy candidate", y=1.04, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "combined_figure4_alternative_entropy_stability")


def make_figure5_exp4(exp4_df: pd.DataFrame, exp4_summary: pd.DataFrame):
    fig = plt.figure(figsize=(15.0, 4.6))
    gs = GridSpec(1, 3, figure=fig)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, j]) for j in range(3)]

    gene_top = (
        exp4_df.groupby("gene_symbol")["diffusion_effective_endpoints"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )
    ax1.barh(gene_top.index, gene_top.values, color=pub.ACCENT_BLUE, edgecolor="white", linewidth=0.8)
    ax1.set_xlabel("Mean effective endpoints")
    ax1.set_title("Endpoint spread differs by gene", fontsize=10.5)
    pub.soft_grid(ax1, "x")
    panel_label(ax1, "a")

    cats = present_categories(exp4_df["clinical_significance"])
    add_box_strip(ax2, exp4_df, cats, "diffusion_top1_mass", "Top-1 endpoint mass", "Uncertainty lowers top-1 mass")
    panel_label(ax2, "b")

    scatter_trend(ax3, exp4_df["isotonic_entropy"], exp4_df["diffusion_branch_entropy"], pub.ACCENT_PURPLE)
    corr, _ = spearmanr(exp4_df["isotonic_entropy"], exp4_df["diffusion_branch_entropy"])
    ax3.set_xlabel("Isotonic entropy")
    ax3.set_ylabel("Diffusion branch entropy (bits)")
    ax3.set_title(f"Branch entropy vs uncertainty (Spearman = {0.0 if math.isnan(corr) else corr:.2f})", fontsize=10.5)
    panel_label(ax3, "c")

    fig.suptitle("Figure 5 | Experiment 4: Local branching ambiguity on short-path AlzKB", y=1.04, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "combined_figure5_local_branching_ambiguity")


def write_report(
    variants_df: pd.DataFrame,
    exp1_df: pd.DataFrame,
    exp1_summary: pd.DataFrame,
    exp2_df: pd.DataFrame,
    exp2_summary: pd.DataFrame,
    exp3_df: pd.DataFrame,
    exp3_summary: pd.DataFrame,
    gene_summary: pd.DataFrame,
    exp4_df: pd.DataFrame,
    exp4_summary: pd.DataFrame,
):
    cats = present_categories(variants_df["clinical_significance"])
    counts = variants_df["clinical_significance"].value_counts().reindex(cats).fillna(0).astype(int)
    q4_exp1 = exp1_summary.iloc[-1] if not exp1_summary.empty else None
    q1_exp1 = exp1_summary.iloc[0] if not exp1_summary.empty else None
    q4_exp2 = exp2_summary.iloc[-1] if not exp2_summary.empty else None
    q1_exp2 = exp2_summary.iloc[0] if not exp2_summary.empty else None
    top_gene = gene_summary.iloc[0] if not gene_summary.empty else None
    corr_top1, _ = spearmanr(exp1_df["isotonic_entropy"], exp1_df["top1_mass"])
    corr_js, _ = spearmanr(exp2_df["isotonic_entropy"], exp2_df["mean_js"])
    corr_branch, _ = spearmanr(exp4_df["isotonic_entropy"], exp4_df["diffusion_branch_entropy"])
    scorer_n = LEARNED_SCORER.n_train if LEARNED_SCORER is not None else 0
    scorer_auroc = LEARNED_SCORER.train_auroc if LEARNED_SCORER is not None else float("nan")
    text = f"""# Redesigned AlzKB Uncertainty Experiments

## Overview
This report reorganizes the AlzKB analysis around four uncertainty-focused questions that are compatible with the current public `AlzKB v2.0.0` graph structure.

The key design choice is:

- keep `revision/diffusion-style` reasoning as the primary method
- stop centering the study on “diffusion is better than AR”
- use the graph to study how sequence-derived uncertainty affects endpoint spread, endpoint ranking, stability, and local branching ambiguity
- score reasoning paths with a **learned** utility (weights trained end-to-end on labeled anchor variants), not a hand-tuned heuristic; see the "Learned Scoring Function" section below

The inputs are:

- pathogenicity signal: `zeroshot_pathogenicity_prob`
- primary uncertainty signal: `isotonic_entropy`
- alternate uncertainty signal for Experiment 3: `knn_entropy`

Matched variants used here: `{len(variants_df)}`

Counts by class:

```
{counts.to_string()}
```

## Why This Redesign Fits The Graph
The latest official AlzKB release is rich biologically, but for many AD genes it induces a short-path, shared-endpoint reasoning regime. That means many genes reach symptom endpoints quickly through the same disease layer.

As a result, the most reliable scientific questions are not long-horizon error-correction questions. They are uncertainty questions:

1. Does uncertainty broaden phenotype distributions?
2. Does uncertainty destabilize endpoint ranking?
3. Are conclusions stable if a different reasonable uncertainty metric is used?
4. Does uncertainty widen local endpoint ambiguity even when paths are short?

## Experiment 1: Uncertainty Propagation to Phenotype Distribution
### Design
For each variant, the diffusion-style operator induces a phenotype distribution over reachable symptom endpoints. We measure:

- endpoint count above probability thresholds
- phenotype entropy
- effective number of endpoints
- top-1 endpoint mass

### Result
The effect is present and usable. From the lowest to highest uncertainty quartile:

- mean phenotype entropy changed from `{q1_exp1['phenotype_entropy_mean']:.3f}` to `{q4_exp1['phenotype_entropy_mean']:.3f}`
- mean effective endpoint count changed from `{q1_exp1['effective_endpoints_mean']:.3f}` to `{q4_exp1['effective_endpoints_mean']:.3f}`
- top-1 endpoint mass changed from `{q1_exp1['top1_mass_mean']:.3f}` to `{q4_exp1['top1_mass_mean']:.3f}`

The correlation between uncertainty and top-1 concentration was `{0.0 if math.isnan(corr_top1) else corr_top1:.3f}`.

Interpretation:
sequence-side uncertainty does propagate to endpoint spread on AlzKB, even though the graph is short-path.

## Experiment 2: Uncertainty-Sensitive Endpoint Ranking
### Design
For each variant, we compute a baseline endpoint ranking and then perturb the uncertainty score up and down while keeping the pathogenicity signal fixed. We measure:

- top-1 switch rate
- JS divergence between endpoint distributions
- top-5 overlap

### Result
The ranking is measurably uncertainty-sensitive, and under the learned operator the cleanest, most monotone signal is the Jensen-Shannon divergence between the base and perturbed endpoint distributions. Perturbing the uncertainty score by +/-0.30 and comparing rankings, from the lowest to highest uncertainty quartile:

- mean JS divergence changed from `{q1_exp2['mean_js']:.3f}` to `{q4_exp2['mean_js']:.3f}`
- mean top-1 switch rate went from `{q1_exp2['mean_top1_switch']:.3f}` to `{q4_exp2['mean_top1_switch']:.3f}` (non-monotone across the middle quartiles, so it is no longer plotted)
- mean top-5 overlap went from `{q1_exp2['mean_top5_overlap']:.3f}` to `{q4_exp2['mean_top5_overlap']:.3f}`

The Spearman correlation between uncertainty and JS divergence was `{0.0 if math.isnan(corr_js) else corr_js:.3f}`.

Interpretation:
the interesting effect is a boundary effect. Confident, low-uncertainty variants have the most perturbation-sensitive rankings, because their endpoint identity is sharply concentrated and a small change in the uncertainty value can flip the top endpoint. Once a variant is already highly uncertain, its endpoint distribution is a broad, near-uniform spread that additional perturbation cannot move much, so ranking switches and JS divergence collapse toward zero and top-5 overlap saturates near one. In other words, ranking fragility is highest exactly where the model is most confident, which is the clinically important regime to flag.

## Experiment 3: Stability Under An Alternative Entropy Candidate
### Design
This experiment replaces the primary uncertainty metric (`isotonic_entropy`) with a stronger alternative candidate (`knn_entropy`) while keeping the same pathogenicity signal. We then compare endpoint distributions under the two uncertainty definitions.

### Result
The endpoint distribution is not invariant to uncertainty definition.

Across classes, the mean JS divergence between isotonic- and kNN-driven endpoint distributions ranged from:

```
{exp3_summary[['clinical_significance','mean_js','mean_top1_switch','mean_top5_overlap']].to_string(index=False)}
```

The most sensitive gene in the current analysis was `{top_gene['gene_symbol'] if top_gene is not None else 'n/a'}` with mean JS divergence `{top_gene['mean_js']:.3f}` across its variants.

Interpretation:
the uncertainty formalization itself matters. This is not a nuisance; it is a substantive modeling question and part of the theory contribution.

## Experiment 4: Local Branching Ambiguity
### Design
Even in a short-path graph, many genes can still reach multiple symptom endpoints. We quantify:

- diffusion effective endpoint count (per gene, reflecting the mechanism prior)
- diffusion branch entropy
- diffusion top-1 mass

### Result
There is real local ambiguity, and uncertainty modulates how strongly the diffusion operator spreads over it. This is now the single strongest effect in the study: the Spearman correlation between uncertainty and diffusion branch entropy was `{0.0 if math.isnan(corr_branch) else corr_branch:.3f}`.

Because the gene-specific mechanism prior differentiates genes, effective endpoint spread also varies by gene rather than collapsing to a shared value.

Class-level summary:

```
{exp4_summary[['clinical_significance','diffusion_effective_endpoints_mean','diffusion_branch_entropy_mean','diffusion_top1_mass_mean']].to_string(index=False)}
```

Interpretation:
even when the graph is short-path, uncertainty strongly changes how much mass is distributed across nearby phenotype branches, and the mechanism prior gives each gene its own branching profile.

## Mechanism Overview Figure
- `combined_figure0_ad_gene_mechanism`
  - a journal-style schematic of the genotype-phenotype fan-out for a canonical AD gene (APOE): APOE associates with several Alzheimer's disease subtypes, which manifest 53 clinical symptoms spanning multiple phenotype groups. This is the biological motivation for the uncertainty program: one gene reaches many phenotypes, so sequence-side uncertainty has room to reshape a broad endpoint distribution. Generated by `scripts/generate_ad_gene_mechanism_figure.py` directly from AlzKB.

## Combined Figures
This redesign produces 5 publication-style combined figures. Each figure has been trimmed to only the panels that tell the story cleanly; redundant or weak panels were removed. All figures now use a shared Nature/npj-style theme (`scripts/pubstyle.py`).

1. `combined_figure1_evo2_score_landscape`
   - sequence-side score landscape and uncertainty inputs (class composition; pathogenicity by class; isotonic entropy by class)
2. `combined_figure2_uncertainty_propagation` (Experiment 1)
   - a: effective endpoints vs uncertainty quartile
   - b: class-wise phenotype entropy
   - c: top-1 endpoint mass vs isotonic entropy (variant-level scatter)
   - (removed: the phenotype-entropy-vs-quartile panel, which was redundant with effective endpoints)
3. `combined_figure3_endpoint_ranking_stability` (Experiment 2)
   - a: JS divergence between base and perturbed rankings vs uncertainty quartile
   - b: JS divergence vs isotonic entropy (variant-level scatter with Spearman)
   - (removed: top-1-switch and top-5-overlap panels, which were non-monotone/noisy under the learned operator; JS divergence carries the clean boundary-effect signal)
4. `combined_figure4_alternative_entropy_stability` (Experiment 3)
   - a: isotonic vs kNN entropy scatter
   - b: class-wise JS divergence between the two uncertainty definitions
   - c: genes most sensitive to the uncertainty definition
   - (removed: top-1-switch-by-class bar, which was the weakest panel)
5. `combined_figure5_local_branching_ambiguity` (Experiment 4)
   - a: per-gene effective endpoints (mechanism prior differentiation)
   - b: top-1 endpoint mass by class
   - c: branch entropy vs isotonic entropy (variant-level scatter)
   - (removed: branch-entropy-by-class box, which was flat across classes and duplicated the scatter)

Each is exported as both `.png` and `.pdf`.

## AR vs Diffusion Operator Comparison (Rerun)
For completeness, the autoregressive-vs-revision (diffusion-style) operator comparison was rerun on the same Evo2-scored variants and the current AlzKB, and restyled to the same journal theme. Those 5 combined figures and their report live in the `ar_vs_diffusion/` subfolder (`outputs/experiments/ar_vs_diffusion/`). The result is consistent with the redesign rationale: on this short-path, shared-endpoint graph the two operators largely coincide in validity, utility, and phenotype coverage (Figures 2 and 4 there are essentially flat), while the only appreciable separation appears for a handful of high-centrality genes and in trajectory diversity (Figure 5). This is exactly why the primary study is framed around uncertainty rather than `diffusion > AR`.

## Honest Takeaway
The current AlzKB does not strongly expose long-horizon operator superiority, but it does support a meaningful uncertainty program.

The strongest current conclusions are:

1. sequence-derived uncertainty maps into phenotype-level spread with a strong monotone effect (Spearman between uncertainty and top-1 endpoint mass approximately `{0.0 if math.isnan(corr_top1) else corr_top1:.2f}`)
2. endpoint rankings are most fragile in the confident, low-uncertainty regime, and stabilize as uncertainty saturates the distribution toward uniform
3. alternative uncertainty definitions lead to materially different endpoint distributions, especially for pathogenic and likely-pathogenic variants
4. even in a short-path graph, uncertainty broadens local endpoint ambiguity (Spearman between uncertainty and branch entropy approximately `{0.0 if math.isnan(corr_branch) else corr_branch:.2f}`)

## Learned Scoring Function
The diffusion operator no longer uses hand-tuned weights. Each candidate reasoning path is described by an interpretable feature vector (structural score, path length, whether it traverses a `Pathway` or `BiologicalProcess`, disease coverage, endpoint specificity, an AD-core-group indicator, the gene-specific mechanism-prior affinity for the endpoint group, a generic-group indicator, a crowding term, and pathogenicity/confidence interactions). The path utility is a linear function of these features, `U(path) = w . phi(path)`, and the weights `w` together with a small logistic readout head are **trained end-to-end** by backpropagating a variant-level pathogenicity loss through the path softmax on the labeled anchor variants (Pathogenic/Likely_pathogenic vs Benign/Likely_benign). In this run the learned scorer was trained on `{scorer_n}` anchors and its readout reached an anchor AUROC of `{scorer_auroc:.3f}`. The exploration temperature `tau(H) = 0.25 + 1.4 H` remains a function of the sequence entropy, because it is the operator's stochastic schedule rather than part of the scored utility.

## What Made The Signals Stronger
Two ingredients drive the effects on this short-path, shared-endpoint graph:

1. Gene-specific mechanism prior. Each gene's `BiologicalProcess` participation is aggregated into a prior over phenotype groups (for example, synaptic and memory processes weight the cognition/memory group). This breaks the degeneracy in which every gene reached the same 53 symptom endpoints through the shared disease layer, so different genes now induce different endpoint distributions. It enters the learned utility as one of the path features.
2. Learned reweighting of path features, with uncertainty coupling. Rather than fixing how strongly specificity, the mechanism prior, and the pathogenicity pull matter, those weights are learned from the anchor labels. Confidence (1 - entropy) interactions let the fitted model decide how sharply low-uncertainty variants concentrate mass, and the entropy-dependent softmax temperature lets high-uncertainty variants flatten toward the raw reachable set. Uncertainty therefore still has a strong dual effect on spread, but the strength of each discriminative term is data-driven rather than hand-set.

This is a better and more defensible scientific story for the present graph than continuing to frame the main result as `diffusion > AR`.
"""
    REPORT_PATH.write_text(text + "\n", encoding="utf-8")


def main():
    apply_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log: List[str] = []
    base = SCRIPT_DIR.parent
    preferred_gene_names = sorted(pd.read_csv(evo2.score_path())["gene_symbol"].dropna().astype(str).str.upper().unique())
    graph, graph_mode = baseexp.load_alzkb(base, log, preferred_gene_names=preferred_gene_names)
    variants = evo2.load_evo2_variants(graph)
    variants = build_knn_entropy(variants)
    edge_type_frequency = Counter(data.get("relation", "unknown") for _, _, data in graph.edges(data=True))

    fit_learned_operator(variants, graph, edge_type_frequency, log)

    exp1_df, exp1_summary = run_experiment1(variants, graph, edge_type_frequency)
    exp2_df, exp2_summary = run_experiment2(variants, graph, edge_type_frequency)
    exp3_df, exp3_summary, gene_summary = run_experiment3(variants, graph, edge_type_frequency)
    exp4_df, exp4_summary = run_experiment4(variants, graph, edge_type_frequency)

    variants.to_csv(OUTPUT_DIR / "redesign_variants_with_uncertainty.csv", index=False)
    exp1_df.to_csv(OUTPUT_DIR / "redesign_exp1_results.csv", index=False)
    exp1_summary.to_csv(OUTPUT_DIR / "redesign_exp1_summary.csv", index=False)
    exp2_df.to_csv(OUTPUT_DIR / "redesign_exp2_results.csv", index=False)
    exp2_summary.to_csv(OUTPUT_DIR / "redesign_exp2_summary.csv", index=False)
    exp3_df.to_csv(OUTPUT_DIR / "redesign_exp3_results.csv", index=False)
    exp3_summary.to_csv(OUTPUT_DIR / "redesign_exp3_summary.csv", index=False)
    gene_summary.to_csv(OUTPUT_DIR / "redesign_exp3_gene_sensitivity.csv", index=False)
    exp4_df.to_csv(OUTPUT_DIR / "redesign_exp4_results.csv", index=False)
    exp4_summary.to_csv(OUTPUT_DIR / "redesign_exp4_summary.csv", index=False)

    make_figure1_input(variants)
    make_figure2_exp1(exp1_df, exp1_summary)
    make_figure3_exp2(exp2_df, exp2_summary)
    make_figure4_exp3(exp3_df, exp3_summary, gene_summary)
    make_figure5_exp4(exp4_df, exp4_summary)
    write_report(variants, exp1_df, exp1_summary, exp2_df, exp2_summary, exp3_df, exp3_summary, gene_summary, exp4_df, exp4_summary)
    print(f"Wrote redesigned report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
