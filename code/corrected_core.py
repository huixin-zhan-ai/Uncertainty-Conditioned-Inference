#!/usr/bin/env python3
"""Leakage-controlled learned path-energy model and Metropolis sampler."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsRegressor

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import run_operator_reasoning_experiments as graph_io
import run_operator_reasoning_experiments_evo2 as legacy_evo2
import run_alzkb_uncertainty_redesign as legacy_analysis

RNG_SEED = 42
SCORE_PATH = ROOT / "data" / "raw" / "evo2" / "ad_all_scores_v3_vcf.csv"
MODEL_PATH = ROOT / "outputs" / "corrected" / "models" / "path_scorer.json"
SPLIT_PATH = ROOT / "outputs" / "corrected" / "manifests" / "locked_split.json"

CLASSES = [
    "Pathogenic",
    "Likely_pathogenic",
    "Uncertain_significance",
    "Likely_benign",
    "Benign",
]
DEFINITIVE_POS = {"Pathogenic"}
DEFINITIVE_NEG = {"Benign"}
LIKELY_POS = {"Likely_pathogenic"}
LIKELY_NEG = {"Likely_benign"}
AD_CORE_GROUPS = {"Cognition_Memory", "Language_Communication"}
GENERIC_GROUPS = {"Neurologic_Syndromic", "Other_Neurologic"}
GENE_RELATIONS = {
    "GENEREGULATESGENE",
    "GENEINTERACTSWITHGENE",
    "GENECOVARIESWITHGENE",
}

# One schedule defines both the exact target law and its Metropolis sampler.
THETA_INTERCEPT = 0.35
THETA_SLOPE = 0.90
FIXED_THETA = 0.80
MAX_GENE_HOPS = 1
MAX_ROUTES = 24
N_CHAINS = 512
PROPOSAL_LEARNED_MIX = 0.90

FEATURE_NAMES = [
    "structural_score",
    "path_length",
    "has_regulation",
    "has_interaction_or_covariation",
    "disease_coverage",
    "endpoint_specificity",
    "is_ad_core",
    "route_mechanism_affinity",
    "generic_group",
    "group_crowding",
    "pathogenicity_x_ad_core",
    "pathogenicity_x_route_mechanism",
    "confidence_x_specificity",
    "confidence_x_route_mechanism",
]
N_FEATURES = len(FEATURE_NAMES)


def theta(entropy: float, fixed: bool = False) -> float:
    if fixed:
        return FIXED_THETA
    return THETA_INTERCEPT + THETA_SLOPE * float(np.clip(entropy, 0.0, 1.0))


def revision_steps(entropy: float, n_states: int) -> int:
    """Mixing-aware schedule; grows with state-space size and uncertainty."""
    return (
        50
        + 10 * int(math.ceil(math.log2(max(2, n_states))))
        + int(round(25 * float(np.clip(entropy, 0.0, 1.0))))
    )


def entropy_from_prob(probabilities: Iterable[float]) -> float:
    p = np.asarray(list(probabilities), dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def effective_support(probabilities: Iterable[float]) -> float:
    return float(2.0 ** entropy_from_prob(probabilities))


def js_divergence(left: Dict[str, float], right: Dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    p = np.array([left.get(key, 0.0) for key in keys], dtype=float)
    q = np.array([right.get(key, 0.0) for key in keys], dtype=float)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def top_key(distribution: Dict[str, float]) -> str:
    if not distribution:
        return ""
    return max(distribution.items(), key=lambda item: (item[1], item[0]))[0]


def topk_overlap(left: Dict[str, float], right: Dict[str, float], k: int = 5) -> float:
    a = {x[0] for x in sorted(left.items(), key=lambda x: (-x[1], x[0]))[:k]}
    b = {x[0] for x in sorted(right.items(), key=lambda x: (-x[1], x[0]))[:k]}
    return len(a & b) / max(1, len(a | b))


@dataclass
class SignalModels:
    platt: LogisticRegression
    isotonic: IsotonicRegression
    knn: KNeighborsRegressor
    training_ids: List[str]


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def load_graph_and_raw_variants():
    raw = pd.read_csv(SCORE_PATH)
    preferred = sorted(raw["gene_symbol"].dropna().astype(str).str.upper().unique())
    log: List[str] = []
    graph, mode = graph_io.load_alzkb(
        ROOT / "data" / "raw", log, preferred_gene_names=preferred
    )
    if not mode.startswith("real_"):
        raise RuntimeError("Corrected paper runs require the real AlzKB graph.")
    raw["clinical_significance"] = raw["CLNSIG"].astype(str).map(
        graph_io.normalize_significance
    )
    raw = raw[raw["clinical_significance"].isin(CLASSES)].copy()
    raw["variant_id"] = raw["clinvar_variation_id"].astype(str)
    raw["gene_symbol"] = raw["gene_symbol"].astype(str)
    raw["zeroshot_score"] = pd.to_numeric(
        raw["zeroshot_neg_delta_likelihood"], errors="coerce"
    )
    raw = raw.dropna(subset=["zeroshot_score", "gene_symbol"])
    symbol_to_node = legacy_evo2.build_symbol_to_node(graph)
    raw["gene_id"] = raw["gene_symbol"].str.upper().map(symbol_to_node)
    raw = raw[raw["gene_id"].notna()].copy().reset_index(drop=True)
    return graph, raw, mode


def fit_signal_models(
    variants: pd.DataFrame, training_ids: Sequence[str] | None = None
) -> SignalModels:
    definitive = variants[
        variants["clinical_significance"].isin(DEFINITIVE_POS | DEFINITIVE_NEG)
    ].copy()
    if training_ids is not None:
        wanted = set(map(str, training_ids))
        definitive = definitive[definitive["variant_id"].isin(wanted)]
    if definitive["clinical_significance"].nunique() != 2:
        raise ValueError("Signal calibration requires both definitive classes.")
    x = definitive[["zeroshot_score"]].to_numpy()
    y = definitive["clinical_significance"].isin(DEFINITIVE_POS).astype(int).to_numpy()
    platt = LogisticRegression(
        C=1e6, max_iter=2000, random_state=RNG_SEED
    ).fit(x, y)
    isotonic = IsotonicRegression(
        y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip", increasing=True
    ).fit(x[:, 0], y)
    knn = KNeighborsRegressor(
        n_neighbors=min(31, len(definitive)), weights="distance"
    ).fit(x, y)
    return SignalModels(
        platt=platt,
        isotonic=isotonic,
        knn=knn,
        training_ids=definitive["variant_id"].astype(str).tolist(),
    )


def apply_signal_models(variants: pd.DataFrame, models: SignalModels) -> pd.DataFrame:
    out = variants.copy()
    x = out[["zeroshot_score"]].to_numpy()
    out["pathogenicity_prob"] = models.platt.predict_proba(x)[:, 1]
    iso_probability = np.clip(models.isotonic.predict(x[:, 0]), 1e-6, 1 - 1e-6)
    knn_probability = np.clip(models.knn.predict(x), 1e-6, 1 - 1e-6)
    out["isotonic_pathogenicity_prob"] = iso_probability
    out["isotonic_entropy"] = binary_entropy(iso_probability)
    out["knn_pathogenicity_prob"] = knn_probability
    out["knn_entropy"] = binary_entropy(knn_probability)
    return out


def locked_variants():
    graph, raw, mode = load_graph_and_raw_variants()
    definitive_ids = raw.loc[
        raw["clinical_significance"].isin(DEFINITIVE_POS | DEFINITIVE_NEG),
        "variant_id",
    ].astype(str).tolist()
    likely_ids = raw.loc[
        raw["clinical_significance"].isin(LIKELY_POS | LIKELY_NEG),
        "variant_id",
    ].astype(str).tolist()
    vus_ids = raw.loc[
        raw["clinical_significance"] == "Uncertain_significance", "variant_id"
    ].astype(str).tolist()
    models = fit_signal_models(raw, definitive_ids)
    variants = apply_signal_models(raw, models)
    payload = {
        "seed": RNG_SEED,
        "signal_calibration_training": "Pathogenic vs Benign only",
        "path_scorer_training": "Pathogenic vs Benign only",
        "locked_validation": "Likely_pathogenic vs Likely_benign",
        "definitive_ids": sorted(definitive_ids),
        "likely_validation_ids": sorted(likely_ids),
        "vus_ids": sorted(vus_ids),
    }
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return graph, variants, models, mode


def _stable_key(route: Tuple[Tuple[str, ...], str]) -> str:
    return hashlib.sha256(
        ("|".join(route[0]) + "|" + route[1]).encode("utf-8")
    ).hexdigest()


def _gene_successors(graph: nx.DiGraph, gene: str):
    for neighbor in graph.successors(gene):
        if (
            graph.nodes[neighbor].get("type") == "Gene"
            and graph.edges[gene, neighbor].get("relation") in GENE_RELATIONS
        ):
            yield neighbor


def _phenotype_diseases(graph: nx.DiGraph, gene: str):
    for disease in graph.successors(gene):
        if graph.nodes[disease].get("type") != "Disease":
            continue
        endpoints = tuple(
            sorted(
                endpoint
                for endpoint in graph.successors(disease)
                if graph.nodes[endpoint].get("type") == "Phenotype"
            )
        )
        if endpoints:
            yield disease, endpoints


def _select_routes(
    graph: nx.DiGraph,
    routes: List[Tuple[Tuple[str, ...], str, Tuple[str, ...]]],
    max_routes: int,
):
    direct = sorted(
        [route for route in routes if len(route[0]) == 1],
        key=lambda route: (route[1], route[0]),
    )
    selected = direct[:max_routes]
    if len(selected) >= max_routes:
        return selected
    strata: Dict[str, List] = defaultdict(list)
    for route in routes:
        genes, disease, _ = route
        if len(genes) == 1:
            continue
        first_relation = graph.edges[genes[0], genes[1]].get("relation", "unknown")
        strata[first_relation].append(route)
    for key in strata:
        strata[key].sort(key=lambda route: _stable_key((route[0], route[1])))
    stratum_names = sorted(strata)
    while len(selected) < max_routes and any(strata.values()):
        for name in stratum_names:
            if strata[name] and len(selected) < max_routes:
                selected.append(strata[name].pop(0))
    return selected


PATH_CACHE: Dict[Tuple[str, int, int], List[Tuple[str, ...]]] = {}


def enumerate_candidate_paths(
    graph: nx.DiGraph,
    start_gene: str,
    max_gene_hops: int = MAX_GENE_HOPS,
    max_routes: int = MAX_ROUTES,
) -> List[Tuple[str, ...]]:
    """Enumerate valid gene-hop→disease→phenotype paths with route-stratified cap."""
    key = (start_gene, max_gene_hops, max_routes)
    if key in PATH_CACHE:
        return PATH_CACHE[key]
    queue: List[Tuple[str, ...]] = [(start_gene,)]
    routes: Dict[Tuple[Tuple[str, ...], str], Tuple[str, ...]] = {}
    for depth in range(max_gene_hops + 1):
        next_queue: List[Tuple[str, ...]] = []
        for genes in queue:
            current = genes[-1]
            for disease, endpoints in _phenotype_diseases(graph, current):
                routes[(genes, disease)] = endpoints
            if depth < max_gene_hops:
                for neighbor in _gene_successors(graph, current):
                    if neighbor not in genes:
                        next_queue.append(genes + (neighbor,))
        queue = next_queue
    route_rows = [
        (genes, disease, endpoints)
        for (genes, disease), endpoints in routes.items()
    ]
    chosen = _select_routes(graph, route_rows, max_routes)
    candidates = [
        genes + (disease, endpoint)
        for genes, disease, endpoints in chosen
        for endpoint in endpoints
    ]
    candidates.sort(key=lambda path: (len(path), path))
    PATH_CACHE[key] = candidates
    return candidates


MECHANISM_CACHE: Dict[str, Dict[str, float]] = {}


def gene_mechanism_affinity(graph: nx.DiGraph, gene: str) -> Dict[str, float]:
    if gene not in MECHANISM_CACHE:
        MECHANISM_CACHE[gene] = legacy_analysis.mechanism_affinity(graph, gene)
    return MECHANISM_CACHE[gene]


def route_mechanism_affinity(
    graph: nx.DiGraph, path: Sequence[str], group: str
) -> float:
    genes = [node for node in path if graph.nodes[node].get("type") == "Gene"]
    if not genes:
        return 0.0
    return float(
        np.mean([gene_mechanism_affinity(graph, gene).get(group, 0.0) for gene in genes])
    )


def endpoint_name(graph: nx.DiGraph, node: str) -> str:
    attrs = graph.nodes[node]
    return str(attrs.get("name") or attrs.get("commonName") or node)


def endpoint_group(graph: nx.DiGraph, endpoint: str) -> str:
    return legacy_analysis.endpoint_group_from_name(endpoint_name(graph, endpoint))


def feature_matrix(
    graph: nx.DiGraph,
    candidates: Sequence[Sequence[str]],
    relation_frequencies: Counter,
    pathogenicity: float,
    uncertainty: float,
) -> Tuple[np.ndarray, List[str]]:
    groups = [endpoint_group(graph, path[-1]) for path in candidates]
    group_counts = Counter(groups)
    confidence = 1.0 - float(np.clip(uncertainty, 0.0, 1.0))
    rows = np.zeros((len(candidates), N_FEATURES), dtype=float)
    for index, (path, group) in enumerate(zip(candidates, groups)):
        relations = [
            graph.edges[path[i], path[i + 1]].get("relation", "unknown")
            for i in range(len(path) - 1)
        ]
        structural = sum(
            math.log(relation_frequencies.get(relation, 0) + 1)
            for relation in relations
        )
        specificity = legacy_analysis.endpoint_specificity_bonus(
            endpoint_name(graph, path[-1])
        )
        mechanism = route_mechanism_affinity(graph, path, group)
        ad_core = float(group in AD_CORE_GROUPS)
        generic = float(group in GENERIC_GROUPS)
        disease_coverage = np.mean(
            [graph.nodes[node].get("type") == "Disease" for node in path]
        )
        rows[index] = [
            structural,
            float(len(path) - 1),
            float("GENEREGULATESGENE" in relations),
            float(
                "GENEINTERACTSWITHGENE" in relations
                or "GENECOVARIESWITHGENE" in relations
            ),
            float(disease_coverage),
            specificity,
            ad_core,
            mechanism,
            generic,
            math.log(group_counts[group] + 1),
            pathogenicity * ad_core,
            pathogenicity * mechanism,
            confidence * specificity,
            confidence * mechanism,
        ]
    return rows, groups


@dataclass
class LearnedPathScorer:
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    alpha: float
    beta: float
    l2: float
    n_train: int
    training_ids: List[str]

    def utilities(self, features: np.ndarray, ablate: Sequence[str] = ()) -> np.ndarray:
        if not np.all(np.isfinite(features)):
            raise ValueError("Non-finite path feature encountered.")
        standardized = (features - self.feature_mean) / self.feature_std
        for feature in ablate:
            standardized[:, FEATURE_NAMES.index(feature)] = 0.0
        # Some Accelerate/BLAS builds leave floating-status flags set after
        # optimized matmul despite finite operands and result. Validate values
        # explicitly before and after rather than promoting those stale flags.
        with np.errstate(all="ignore"):
            utilities = standardized @ self.weights
        if not np.all(np.isfinite(utilities)):
            raise ValueError("Non-finite learned path utility encountered.")
        return utilities

    def save(self, path: Path = MODEL_PATH) -> Path:
        payload = {
            "feature_names": FEATURE_NAMES,
            "weights": self.weights.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "alpha": self.alpha,
            "beta": self.beta,
            "l2": self.l2,
            "n_train": self.n_train,
            "training_ids": self.training_ids,
            "theta": {
                "intercept": THETA_INTERCEPT,
                "slope": THETA_SLOPE,
                "fixed_ablation": FIXED_THETA,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["feature_names"] != FEATURE_NAMES:
            raise ValueError("Saved feature order does not match corrected feature order.")
        return cls(
            weights=np.asarray(payload["weights"], dtype=float),
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_std=np.asarray(payload["feature_std"], dtype=float),
            alpha=float(payload["alpha"]),
            beta=float(payload["beta"]),
            l2=float(payload["l2"]),
            n_train=int(payload["n_train"]),
            training_ids=list(map(str, payload["training_ids"])),
        )


def build_training_samples(
    graph: nx.DiGraph,
    variants: pd.DataFrame,
    relation_frequencies: Counter,
    training_ids: Sequence[str],
):
    wanted = set(map(str, training_ids))
    train = variants[variants["variant_id"].isin(wanted)].copy()
    train = train[
        train["clinical_significance"].isin(DEFINITIVE_POS | DEFINITIVE_NEG)
    ]
    samples = []
    for row in train.itertuples(index=False):
        candidates = enumerate_candidate_paths(graph, row.gene_id)
        if not candidates:
            continue
        features, groups = feature_matrix(
            graph,
            candidates,
            relation_frequencies,
            row.pathogenicity_prob,
            row.isotonic_entropy,
        )
        samples.append(
            {
                "variant_id": str(row.variant_id),
                "features": features,
                "ad_core": np.asarray(
                    [group in AD_CORE_GROUPS for group in groups], dtype=float
                ),
                "theta": theta(row.isotonic_entropy),
                "label": float(row.clinical_significance in DEFINITIVE_POS),
            }
        )
    return samples


def train_path_scorer(
    samples: List[dict], l2: float = 1e-2, seed: int = RNG_SEED
) -> LearnedPathScorer:
    if not samples:
        raise ValueError("No valid definitive training samples.")
    pooled = np.vstack([sample["features"] for sample in samples])
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std[std < 1e-8] = 1.0
    prepared = [
        {
            **sample,
            "features": (sample["features"] - mean) / std,
        }
        for sample in samples
    ]
    dimension = N_FEATURES
    readout_l2 = 1e-3

    def objective(parameters):
        if not np.all(np.isfinite(parameters)):
            return 1e100, np.zeros_like(parameters)
        weights = parameters[:dimension]
        alpha, beta = parameters[dimension:]
        loss = 0.0
        grad_w = np.zeros(dimension)
        grad_alpha = 0.0
        grad_beta = 0.0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for sample in prepared:
                phi = sample["features"]
                c = sample["ad_core"]
                tau = sample["theta"]
                label = sample["label"]
                utilities = phi @ weights
                if not np.all(np.isfinite(utilities)):
                    return 1e100, np.zeros_like(parameters)
                logits = (utilities - utilities.max()) / tau
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum()
                mass = float(c @ probabilities)
                prediction = float(expit(alpha * mass + beta))
                prediction = float(np.clip(prediction, 1e-9, 1 - 1e-9))
                loss += -(label * np.log(prediction) + (1 - label) * np.log(1 - prediction))
                grad_z = prediction - label
                grad_alpha += grad_z * mass
                grad_beta += grad_z
                grad_utility = grad_z * alpha * probabilities * (c - mass) / tau
                grad_w += phi.T @ grad_utility
        count = len(prepared)
        loss = (
            loss / count
            + 0.5 * l2 * float(weights @ weights)
            + 0.5 * readout_l2 * alpha * alpha
        )
        gradient = np.concatenate(
            [
                grad_w / count + l2 * weights,
                [grad_alpha / count + readout_l2 * alpha, grad_beta / count],
            ]
        )
        return loss, gradient

    rng = np.random.default_rng(seed)
    initial = np.concatenate([rng.normal(0, 0.01, dimension), [1.0, 0.0]])
    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0)] * dimension + [(-20.0, 20.0), (-20.0, 20.0)],
        options={"maxiter": 1000, "ftol": 1e-10, "maxls": 50},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Path-scorer optimization failed: {result.message}")
    return LearnedPathScorer(
        weights=result.x[:dimension],
        feature_mean=mean,
        feature_std=std,
        alpha=float(result.x[dimension]),
        beta=float(result.x[dimension + 1]),
        l2=l2,
        n_train=len(prepared),
        training_ids=[sample["variant_id"] for sample in prepared],
    )


def relation_frequencies(graph: nx.DiGraph) -> Counter:
    return Counter(
        data.get("relation", "unknown") for _, _, data in graph.edges(data=True)
    )


def exact_path_probabilities(
    utilities: np.ndarray, uncertainty: float, fixed_temperature: bool = False
) -> np.ndarray:
    tau = theta(uncertainty, fixed_temperature)
    logits = (np.asarray(utilities, dtype=float) - np.max(utilities)) / tau
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def proposal_probabilities(utilities: np.ndarray, uncertainty: float) -> np.ndarray:
    """Full-support learned/uniform mixture corrected by Hastings ratio."""
    utilities = np.asarray(utilities, dtype=float)
    if utilities.size == 0:
        return np.array([], dtype=float)
    tau_proposal = 2.0 * theta(uncertainty)
    logits = (utilities - np.max(utilities)) / tau_proposal
    learned = np.exp(logits)
    learned /= learned.sum()
    probabilities = (
        PROPOSAL_LEARNED_MIX * learned
        + (1.0 - PROPOSAL_LEARNED_MIX) / len(utilities)
    )
    probabilities /= probabilities.sum()
    return probabilities


def aggregate_distribution(
    keys: Sequence[str], probabilities: Sequence[float]
) -> Dict[str, float]:
    distribution: Dict[str, float] = defaultdict(float)
    for key, probability in zip(keys, probabilities):
        distribution[key] += float(probability)
    total = sum(distribution.values())
    return {key: value / total for key, value in distribution.items()} if total else {}


def exact_distribution(
    graph: nx.DiGraph,
    scorer: LearnedPathScorer,
    relation_frequency: Counter,
    gene: str,
    pathogenicity: float,
    uncertainty: float,
    aggregate_groups: bool = True,
    fixed_temperature: bool = False,
    ablate: Sequence[str] = (),
):
    candidates = enumerate_candidate_paths(graph, gene)
    if not candidates:
        return {}, pd.DataFrame()
    features, groups = feature_matrix(
        graph, candidates, relation_frequency, pathogenicity, uncertainty
    )
    utilities = scorer.utilities(features, ablate=ablate)
    probabilities = exact_path_probabilities(
        utilities, uncertainty, fixed_temperature=fixed_temperature
    )
    keys = groups if aggregate_groups else [path[-1] for path in candidates]
    distribution = aggregate_distribution(keys, probabilities)
    rows = pd.DataFrame(
        {
            "path": [json.dumps(path) for path in candidates],
            "endpoint": [path[-1] for path in candidates],
            "endpoint_group": groups,
            "utility": utilities,
            "probability": probabilities,
        }
    ).sort_values("probability", ascending=False)
    return distribution, rows


def uniform_distribution(
    graph: nx.DiGraph, gene: str, aggregate_groups: bool = True
) -> Dict[str, float]:
    candidates = enumerate_candidate_paths(graph, gene)
    if not candidates:
        return {}
    keys = (
        [endpoint_group(graph, path[-1]) for path in candidates]
        if aggregate_groups
        else [path[-1] for path in candidates]
    )
    return aggregate_distribution(keys, np.ones(len(keys)) / len(keys))


def metropolis_sample(
    utilities: np.ndarray,
    uncertainty: float,
    n_chains: int = N_CHAINS,
    seed: int = RNG_SEED,
    fixed_temperature: bool = False,
):
    """Vectorized independence-MH chains with a learned tempered proposal."""
    utilities = np.asarray(utilities, dtype=float)
    if utilities.size == 0:
        return np.array([], dtype=int), 0.0, 0
    tau = theta(uncertainty, fixed_temperature)
    n_steps = revision_steps(uncertainty, len(utilities))
    # Candidate paths are sorted shortest-first, making index zero the
    # deterministic initial shortest path required by the protocol.
    initial = 0
    rng = np.random.default_rng(seed)
    current = np.full(n_chains, initial, dtype=int)
    if fixed_temperature:
        proposal_tau = 2.0 * tau
        proposal_logits = (utilities - utilities.max()) / proposal_tau
        learned = np.exp(proposal_logits)
        learned /= learned.sum()
        proposal_probability = (
            PROPOSAL_LEARNED_MIX * learned
            + (1.0 - PROPOSAL_LEARNED_MIX) / len(utilities)
        )
        proposal_probability /= proposal_probability.sum()
    else:
        proposal_probability = proposal_probabilities(utilities, uncertainty)
    log_proposal = np.log(np.clip(proposal_probability, 1e-300, 1.0))
    accepted = 0
    attempted = n_chains * n_steps
    for _ in range(n_steps):
        proposal = rng.choice(
            len(utilities), size=n_chains, p=proposal_probability
        )
        log_acceptance = (
            (utilities[proposal] - utilities[current]) / tau
            + log_proposal[current]
            - log_proposal[proposal]
        )
        accept = (log_acceptance >= 0) | (
            np.log(rng.random(n_chains)) < log_acceptance
        )
        current[accept] = proposal[accept]
        accepted += int(accept.sum())
    return current, accepted / max(1, attempted), n_steps


def diffusion_diagnostics(
    graph: nx.DiGraph,
    scorer: LearnedPathScorer,
    relation_frequency: Counter,
    gene: str,
    pathogenicity: float,
    uncertainty: float,
    seed: int,
):
    candidates = enumerate_candidate_paths(graph, gene)
    if not candidates:
        return {}, {}
    features, groups = feature_matrix(
        graph, candidates, relation_frequency, pathogenicity, uncertainty
    )
    utilities = scorer.utilities(features)
    exact_prob = exact_path_probabilities(utilities, uncertainty)
    exact_group = aggregate_distribution(groups, exact_prob)
    final, acceptance, n_steps = metropolis_sample(
        utilities, uncertainty, seed=seed
    )
    empirical = aggregate_distribution(
        [groups[index] for index in final],
        np.ones(len(final)) / max(1, len(final)),
    )
    routes = [tuple(candidates[index][:-2]) for index in final]
    diagnostics = {
        "acceptance_rate": acceptance,
        "revision_steps": n_steps,
        "unique_final_paths": int(len(set(final.tolist()))),
        "unique_final_routes": int(len(set(routes))),
        "js_empirical_to_exact": js_divergence(empirical, exact_group),
    }
    return empirical, diagnostics


def validate_split_disjointness(scorer: LearnedPathScorer, variants: pd.DataFrame):
    held_out = set(
        variants.loc[
            variants["clinical_significance"].isin(LIKELY_POS | LIKELY_NEG),
            "variant_id",
        ].astype(str)
    )
    overlap = held_out & set(map(str, scorer.training_ids))
    if overlap:
        raise AssertionError(f"Leakage: {len(overlap)} locked validation IDs in scorer.")

