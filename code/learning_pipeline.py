#!/usr/bin/env python3
"""Shared setup that guarantees the learned scorer is active."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import learned_operator as lop
import pubstyle
import run_operator_reasoning_experiments as graph_io
import run_operator_reasoning_experiments_evo2 as evo2
import run_alzkb_uncertainty_redesign as analysis

MODEL_PATH = ROOT / "outputs" / "models" / "learned_operator_weights.json"
SCORE_PATH = ROOT / "data" / "raw" / "evo2" / "ad_all_scores_v3_vcf.csv"


@dataclass
class Context:
    graph: object
    variants: pd.DataFrame
    edge_type_frequency: Counter
    scorer: lop.LearnedScorer | None
    graph_mode: str


def load_inputs(include_knn: bool = True):
    pubstyle.apply()
    if not SCORE_PATH.exists():
        raise FileNotFoundError(f"Missing Evo2 score table: {SCORE_PATH}")
    preferred = sorted(
        pd.read_csv(SCORE_PATH, usecols=["gene_symbol"])["gene_symbol"]
        .dropna().astype(str).str.upper().unique()
    )
    log = []
    graph, graph_mode = graph_io.load_alzkb(
        ROOT / "data" / "raw", log, preferred_gene_names=preferred
    )
    if not graph_mode.startswith("real_"):
        raise RuntimeError(
            "The real AlzKB dump was not loaded; refusing a paper run."
        )
    variants = evo2.load_evo2_variants(graph)
    if include_knn:
        variants = analysis.build_knn_entropy(variants)
    edge_frequency = Counter(
        data.get("relation", "unknown")
        for _, _, data in graph.edges(data=True)
    )
    return graph, variants, edge_frequency, graph_mode


def load_learning_context(
    include_knn: bool = True, force_retrain: bool = False
) -> Context:
    graph, variants, edge_frequency, graph_mode = load_inputs(include_knn)
    if MODEL_PATH.exists() and not force_retrain:
        scorer = lop.LearnedScorer.load(MODEL_PATH)
    else:
        samples = analysis.build_anchor_samples(
            variants, graph, edge_frequency
        )
        scorer = lop.train(samples, l2=1e-2, seed=42)
        scorer.save(MODEL_PATH)
    analysis.LEARNED_SCORER = scorer
    if analysis.LEARNED_SCORER is None:
        raise RuntimeError("Learned scorer is required; no heuristic fallback.")
    return Context(
        graph, variants, edge_frequency, scorer, graph_mode
    )
