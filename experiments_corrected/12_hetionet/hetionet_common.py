#!/usr/bin/env python3
"""Hetionet v1.0 in the AlzKB schema, so corrected_core runs unchanged.

Node types: Gene, Disease, Phenotype (Hetionet Symptom), BiologicalProcess,
Pathway.  Edge relations (AlzKB names): GENEASSOCIATESWITHDISEASE
(Disease-associates-Gene), DISEASEMANIFESTSSYMPTOM (Disease-presents-Symptom),
GENEINTERACTSWITHGENE (both directions), GENEREGULATESGENE (forward),
GENECOVARIESWITHGENE (both), GENEPARTICIPATESINBIOLOGICALPROCESS, GENEINPATHWAY.
Anatomy/Compound/Side-effect/etc. are dropped (not on the reasoning path).
"""
from __future__ import annotations

import bz2
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
import corrected_core as core  # noqa: E402

RAW = ROOT / "data" / "raw" / "hetionet" / "hetionet-v1.0.json.bz2"
CACHE = ROOT / "data" / "processed" / "hetionet_alzkb_schema.pkl"
TABLES = ROOT / "outputs" / "corrected" / "tables"

REL = {
    ("Disease", "associates", "Gene"): ("GENEASSOCIATESWITHDISEASE", "reverse"),
    ("Disease", "presents", "Symptom"): ("DISEASEMANIFESTSSYMPTOM", "forward"),
    ("Gene", "interacts", "Gene"): ("GENEINTERACTSWITHGENE", "both"),
    ("Gene", "regulates", "Gene"): ("GENEREGULATESGENE", "forward"),
    ("Gene", "covaries", "Gene"): ("GENECOVARIESWITHGENE", "both"),
    ("Gene", "participates", "Biological Process"): ("GENEPARTICIPATESINBIOLOGICALPROCESS", "forward"),
    ("Gene", "participates", "Pathway"): ("GENEINPATHWAY", "forward"),
}
TYPE = {"Gene": "Gene", "Disease": "Disease", "Symptom": "Phenotype", "Biological Process": "BiologicalProcess", "Pathway": "Pathway"}


def node_id(kind, ident):
    return f"H{TYPE[kind]}:{ident}"


def build_graph(force=False) -> nx.DiGraph:
    if CACHE.exists() and not force:
        return pickle.load(open(CACHE, "rb"))
    d = json.load(bz2.open(RAW, "rt"))
    g = nx.DiGraph()
    for n in d["nodes"]:
        if n["kind"] in TYPE:
            g.add_node(node_id(n["kind"], n["identifier"]), type=TYPE[n["kind"]], name=n["name"])
    for e in d["edges"]:
        key = (e["source_id"][0], e["kind"], e["target_id"][0])
        if key not in REL:
            continue
        rel, direction = REL[key]
        u = node_id(e["source_id"][0], e["source_id"][1]); v = node_id(e["target_id"][0], e["target_id"][1])
        if direction == "reverse":
            g.add_edge(v, u, relation=rel)
        elif direction == "forward":
            g.add_edge(u, v, relation=rel)
        else:
            g.add_edge(u, v, relation=rel); g.add_edge(v, u, relation=rel)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(g, open(CACHE, "wb"))
    return g


def study_variants(graph: nx.DiGraph) -> pd.DataFrame:
    """Frozen per-variant signals (p, H_iso, H_knn, labels) mapped to Hetionet gene ids."""
    sym2id = {a.get("name", "").upper(): n for n, a in graph.nodes(data=True) if a.get("type") == "Gene"}
    v = pd.read_csv(TABLES / "variants_locked_signals.csv")
    v["variant_id"] = v["variant_id"].astype(str)
    v["alzkb_gene_id"] = v["gene_id"]
    v["gene_id"] = v["gene_symbol"].str.upper().map(sym2id)
    v = v[v["gene_id"].notna()].reset_index(drop=True)
    return v


def reset_caches():
    core.PATH_CACHE.clear(); core.MECHANISM_CACHE.clear()
    try:
        import run_alzkb_uncertainty_redesign as legacy
        legacy.MECH_AFFINITY_CACHE.clear()
    except Exception:
        pass


def graph_summary(graph: nx.DiGraph) -> dict:
    from collections import Counter
    types = Counter(a["type"] for _, a in graph.nodes(data=True))
    rels = Counter(a["relation"] for _, _, a in graph.edges(data=True))
    parents = Counter()
    for u, v, a in graph.edges(data=True):
        if a["relation"] == "DISEASEMANIFESTSSYMPTOM":
            parents[v] += 1
    return dict(nodes=graph.number_of_nodes(), edges=graph.number_of_edges(), node_types=dict(types), relations=dict(rels),
                symptoms_with_edges=len(parents), symptoms_multi_parent=int(sum(c > 1 for c in parents.values())),
                max_disease_parents=int(max(parents.values())) if parents else 0,
                disease_sources=len({u for u, v, a in graph.edges(data=True) if a["relation"] == "DISEASEMANIFESTSSYMPTOM"}))
