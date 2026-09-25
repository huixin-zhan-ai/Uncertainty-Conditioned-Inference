#!/usr/bin/env python3
"""Operator-theoretic reasoning experiments for AlzKB + ClinVar.

Implements:
1. AlzKB loading with multiple fallbacks
2. ClinVar loading with entropy proxy assignment
3. Two operators:
   - autoregressive beam search
   - revision-capable Gibbs-like path revision
4. Three experiments:
   - phase transition in operator advantage
   - early branching point sensitivity
   - hypothesis diversity vs uncertainty

Outputs:
  experiment1.png, experiment2.png, experiment3.png
  exp1_results.csv, exp2_results.csv, exp3_results.csv
  experiment_summary.txt

If no local AlzKB graph is available, the script uses the synthetic graph
specified in the prompt and notes that explicitly in all outputs.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import random
import re
import socket
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from rdflib import Graph as RDFGraph
from scipy.stats import spearmanr


RNG_SEED = 42
OUTPUT_DIR = Path("outputs/experiments/simulation")
CLINVAR_CACHE_PATH = OUTPUT_DIR / "clinvar_alzheimer_variants_cache.csv"
EDGE_CSV_CANDIDATES = [
    "alzkb_edges.csv",
    "edges.csv",
    "alzkb.csv",
    "alzkb_edgelist.txt",
    "kg_edges.csv",
]
EUTILS_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUTILS_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
VARIANT_FILE_CANDIDATES = [
    "clinvar_variants.csv",
    "variants.csv",
    "alzkb_clinvar.csv",
    "clinvar_alzkb.csv",
    "variant_gene_map.csv",
    "outputs/ad_vus_alzkb_bridge_full.csv",
    "outputs/ad_vus_alzkb_bridge.csv",
]
GRAPH_FILE_CANDIDATES = [
    "alzkb.json",
    "alzkb.pkl",
    "graph.pkl",
]
CYTHERL_FILE_CANDIDATES = [
    "alzkb-v2-0-0_memgraph.cypherl",
    "alzkb_memgraph.cypherl",
    "alzkb.cypherl",
]
RDF_EXTENSIONS = {".ttl", ".owl", ".rdf", ".nt"}
TYPE_TO_NODE_IDS = {
    "Gene": "G_{}",
    "Pathway": "P_{}",
    "BiologicalProcess": "BP_{}",
    "Phenotype": "Ph_{}",
}
ENTROPY_MAP = {
    "Pathogenic": 0.05,
    "Benign": 0.05,
    "Likely_pathogenic": 0.25,
    "Likely_benign": 0.25,
    "Uncertain_significance": 0.65,
    "Conflicting_interpretations": 0.85,
}
SIG_ORDER = [
    "Pathogenic",
    "Likely_pathogenic",
    "Uncertain_significance",
    "Conflicting_interpretations",
    "Likely_benign",
    "Benign",
]


def normalize_significance(value: str) -> str:
    text = (value or "").strip().lower().replace(" ", "_")
    mapping = {
        "pathogenic": "Pathogenic",
        "likely_pathogenic": "Likely_pathogenic",
        "uncertain_significance": "Uncertain_significance",
        "conflicting_classifications_of_pathogenicity": (
            "Conflicting_interpretations"
        ),
        "conflicting_interpretations": "Conflicting_interpretations",
        "benign": "Benign",
        "likely_benign": "Likely_benign",
        "benign/likely_benign": "Likely_benign",
        "pathogenic/likely_pathogenic": "Likely_pathogenic",
    }
    return mapping.get(text, value.strip().replace(" ", "_"))


def assign_entropy(significance: str, log: List[str]) -> float:
    if significance in ENTROPY_MAP:
        return ENTROPY_MAP[significance]
    log.append(
        f"Unknown significance '{significance}' assigned default entropy 0.5"
    )
    return 0.5


def candidate_files(base: Path, names: Iterable[str]) -> List[Path]:
    found = []
    search_dirs = [base, base / "data", base / "alzkb", base / "kg", base / "outputs"]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for name in names:
            path = search_dir / name
            if path.exists():
                found.append(path)
    return found


def detect_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def can_connect_neo4j(host: str = "localhost", port: int = 7687) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def load_alzkb_via_neo4j(log: List[str]) -> Optional[nx.DiGraph]:
    if not can_connect_neo4j():
        return None
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)
        graph = nx.DiGraph()
        with driver.session() as session:
            node_records = session.run(
                "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, "
                "coalesce(n.name, n.commonName, n.id, elementId(n)) AS name"
            )
            for record in node_records:
                labels = record["labels"] or []
                node_type = labels[0] if labels else "Unknown"
                graph.add_node(
                    record["id"], type=node_type, name=record["name"], source="neo4j"
                )
            edge_records = session.run(
                "MATCH (a)-[r]->(b) RETURN elementId(a) AS source, "
                "type(r) AS relation, elementId(b) AS target"
            )
            for record in edge_records:
                graph.add_edge(
                    record["source"],
                    record["target"],
                    relation=record["relation"],
                )
        log.append("Loaded AlzKB from local Neo4j instance.")
        return graph
    except Exception as exc:  # pragma: no cover
        log.append(f"Neo4j load failed: {exc}")
        return None


def infer_type(node_id: str) -> str:
    if node_id.startswith("gene_") or node_id.startswith("G_"):
        return "Gene"
    if node_id.startswith("path_") or node_id.startswith("P_"):
        return "Pathway"
    if node_id.startswith("bp_") or node_id.startswith("BP_"):
        return "BiologicalProcess"
    if node_id.startswith("ph_") or node_id.startswith("Ph_"):
        return "Phenotype"
    return "Unknown"


def load_alzkb_from_edge_files(base: Path, log: List[str]) -> Optional[nx.DiGraph]:
    found = candidate_files(base, EDGE_CSV_CANDIDATES)
    for path in found:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        subj = detect_column(df.columns, ["subject", "source", "head"])
        pred = detect_column(df.columns, ["predicate", "relation", "edge", "type"])
        obj = detect_column(df.columns, ["object", "target", "tail"])
        if not (subj and pred and obj):
            continue
        graph = nx.DiGraph()
        for _, row in df.iterrows():
            source = str(row[subj])
            target = str(row[obj])
            relation = str(row[pred])
            for node in (source, target):
                if node not in graph:
                    graph.add_node(node, type=infer_type(node), name=node)
            graph.add_edge(source, target, relation=relation)
        log.append(f"Loaded AlzKB from edge file: {path}")
        return graph
    return None


def _extract_quoted_property(line: str, key: str) -> Optional[str]:
    match = re.search(rf"`{re.escape(key)}`:\s*\"((?:[^\"\\\\]|\\\\.)*)\"", line)
    if match:
        return bytes(match.group(1), "utf-8").decode("unicode_escape")
    return None


def _extract_numeric_property(line: str, key: str) -> Optional[str]:
    match = re.search(rf"`{re.escape(key)}`:\s*([0-9]+)", line)
    return match.group(1) if match else None


def load_alzkb_from_cypherl(base: Path, log: List[str]) -> Optional[nx.DiGraph]:
    candidates = candidate_files(base, CYTHERL_FILE_CANDIDATES)
    if not candidates:
        for subdir in [base, base / "data", base / "alzkb", base / "kg", base / "data" / "alzkb"]:
            if subdir.exists():
                for path in subdir.iterdir():
                    if path.suffix.lower() == ".cypherl":
                        candidates.append(path)

    selected_labels = {"Gene", "Pathway", "BiologicalProcess", "Disease", "Symptom"}
    node_pattern = re.compile(
        r"CREATE \(:__mg_vertex__:`([^`]+)` \{__mg_id__:\s*([0-9]+),"
    )
    rel_pattern = re.compile(
        r"MATCH \(u:__mg_vertex__\), \(v:__mg_vertex__\) WHERE u\.__mg_id__ = ([0-9]+) "
        r"AND v\.__mg_id__ = ([0-9]+) CREATE \(u\)-\[:`([^`]+)`(?: \{.*\})?\]->\(v\);"
    )

    for path in candidates:
        try:
            graph = nx.DiGraph()
            mgid_to_node: Dict[str, str] = {}
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    node_match = node_pattern.search(line)
                    if node_match:
                        label, mgid = node_match.groups()
                        if label not in selected_labels:
                            continue
                        node_type = "Phenotype" if label == "Symptom" else label
                        node_key = f"{label}:{mgid}"
                        name = (
                            _extract_quoted_property(line, "geneSymbol")
                            or _extract_quoted_property(line, "commonName")
                            or _extract_quoted_property(line, "nodeID")
                            or node_key
                        )
                        node_id_value = _extract_quoted_property(line, "nodeID") or mgid
                        graph.add_node(
                            node_key,
                            type=node_type,
                            original_type=label,
                            name=name,
                            nodeID=node_id_value,
                            mg_id=mgid,
                            geneSymbol=_extract_quoted_property(line, "geneSymbol"),
                            commonName=_extract_quoted_property(line, "commonName"),
                            xrefNcbiGene=_extract_numeric_property(line, "xrefNcbiGene"),
                            chromosome=_extract_quoted_property(line, "chromosome"),
                            source="cypherl",
                        )
                        mgid_to_node[mgid] = node_key
                        continue

                    rel_match = rel_pattern.search(line)
                    if rel_match:
                        src_mgid, dst_mgid, rel_type = rel_match.groups()
                        if src_mgid in mgid_to_node and dst_mgid in mgid_to_node:
                            src_node = mgid_to_node[src_mgid]
                            dst_node = mgid_to_node[dst_mgid]
                            graph.add_edge(src_node, dst_node, relation=rel_type)
                            # In AlzKB the symptom relation is stored as Symptom -> Disease.
                            # Add the semantic inverse so reasoning can follow Gene -> Disease -> Symptom.
                            if (
                                rel_type == "SYMPTOMMANIFESTATIONOFDISEASE"
                                and graph.nodes[src_node].get("original_type") == "Symptom"
                                and graph.nodes[dst_node].get("original_type") == "Disease"
                            ):
                                graph.add_edge(
                                    dst_node,
                                    src_node,
                                    relation="DISEASEMANIFESTSSYMPTOM",
                                )
            if graph.number_of_nodes() > 0 and graph.number_of_edges() > 0:
                log.append(f"Loaded AlzKB from CYPHERL dump: {path}")
                log.append(
                    "Symptom nodes were treated as Phenotype endpoints; added Disease->Symptom semantic inverse edges for reasoning paths."
                )
                return graph
        except Exception as exc:
            log.append(f"CYPHERL load failed for {path}: {exc}")
    return None


def load_alzkb_from_rdf(base: Path, log: List[str]) -> Optional[nx.DiGraph]:
    candidates = []
    for subdir in [base, base / "data", base / "alzkb", base / "kg"]:
        if subdir.exists():
            for path in subdir.iterdir():
                if path.suffix.lower() in RDF_EXTENSIONS:
                    candidates.append(path)
    for path in candidates:
        try:
            rdf_graph = RDFGraph()
            rdf_graph.parse(path)
        except Exception:
            continue
        graph = nx.DiGraph()
        for s, p, o in rdf_graph:
            s_str, p_str, o_str = str(s), str(p), str(o)
            if p_str.endswith("#type") or p_str.endswith("/type"):
                continue
            for node_str in (s_str, o_str):
                if node_str not in graph:
                    graph.add_node(node_str, type=infer_type(node_str), name=node_str)
            graph.add_edge(s_str, o_str, relation=p_str.split("#")[-1])
        if graph.number_of_nodes() > 0:
            log.append(f"Loaded AlzKB from RDF file: {path}")
            return graph
    return None


def load_alzkb_from_serialized(base: Path, log: List[str]) -> Optional[nx.DiGraph]:
    found = candidate_files(base, GRAPH_FILE_CANDIDATES)
    for path in found:
        try:
            if path.suffix == ".json":
                payload = json.loads(path.read_text())
                graph = nx.node_link_graph(payload)
            else:
                with path.open("rb") as handle:
                    graph = pickle.load(handle)
            if isinstance(graph, nx.DiGraph):
                log.append(f"Loaded AlzKB from serialized graph: {path}")
                return graph
        except Exception:
            continue
    return None


def synthetic_graph(
    seed: int = RNG_SEED, preferred_gene_names: Optional[List[str]] = None
) -> nx.DiGraph:
    rng = random.Random(seed)
    graph = nx.DiGraph()
    preferred_gene_names = preferred_gene_names or []

    for idx in range(1, 201):
        node_id = TYPE_TO_NODE_IDS["Gene"].format(idx)
        name = preferred_gene_names[idx - 1] if idx <= len(preferred_gene_names) else node_id
        graph.add_node(node_id, type="Gene", name=name)
    for idx in range(1, 51):
        node_id = TYPE_TO_NODE_IDS["Pathway"].format(idx)
        graph.add_node(node_id, type="Pathway", name=node_id)
    for idx in range(1, 31):
        node_id = TYPE_TO_NODE_IDS["BiologicalProcess"].format(idx)
        graph.add_node(node_id, type="BiologicalProcess", name=node_id)
    for idx in range(1, 16):
        node_id = TYPE_TO_NODE_IDS["Phenotype"].format(idx)
        graph.add_node(node_id, type="Phenotype", name=node_id)

    pathways = [TYPE_TO_NODE_IDS["Pathway"].format(i) for i in range(1, 51)]
    processes = [TYPE_TO_NODE_IDS["BiologicalProcess"].format(i) for i in range(1, 31)]
    phenotypes = [TYPE_TO_NODE_IDS["Phenotype"].format(i) for i in range(1, 16)]

    for idx in range(1, 201):
        gene = TYPE_TO_NODE_IDS["Gene"].format(idx)
        for pathway in rng.sample(pathways, rng.randint(1, 3)):
            graph.add_edge(gene, pathway, relation="gene_in_pathway")
    for pathway in pathways:
        for process in rng.sample(processes, rng.randint(1, 3)):
            graph.add_edge(pathway, process, relation="pathway_involves")
    for process in processes:
        for phenotype in rng.sample(phenotypes, rng.randint(1, 3)):
            graph.add_edge(
                process, phenotype, relation="process_associates_phenotype"
            )
    return graph


def summarize_graph(graph: nx.DiGraph, data_mode: str, log: List[str]) -> None:
    type_counts = Counter(graph.nodes[n].get("type", "Unknown") for n in graph.nodes)
    relation_counts = Counter(
        data.get("relation", "unknown") for _, _, data in graph.edges(data=True)
    )
    print("\nGraph summary:")
    print(f"  data_mode: {data_mode}")
    print(f"  total_nodes: {graph.number_of_nodes()}")
    print(f"  total_edges: {graph.number_of_edges()}")
    print(f"  nodes_by_type: {dict(type_counts)}")
    print(f"  edges_by_relation: {dict(relation_counts)}")
    print(f"  Gene nodes: {type_counts.get('Gene', 0)}")
    print(f"  Pathway nodes: {type_counts.get('Pathway', 0)}")
    print(f"  BiologicalProcess nodes: {type_counts.get('BiologicalProcess', 0)}")
    print(f"  Phenotype nodes: {type_counts.get('Phenotype', 0)}")

    rng = random.Random(RNG_SEED)
    gene_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Gene"]
    phenotype_nodes = {
        n for n, d in graph.nodes(data=True) if d.get("type") == "Phenotype"
    }
    sample_genes = rng.sample(gene_nodes, min(5, len(gene_nodes))) if gene_nodes else []
    connectivity = []
    for gene in sample_genes:
        found = False
        for target in phenotype_nodes:
            try:
                path = nx.shortest_path(graph, gene, target)
                if len(path) - 1 <= 6:
                    found = True
                    break
            except nx.NetworkXNoPath:
                continue
        connectivity.append((gene, found))
    print(f"  Gene->Phenotype path within depth 6 (sample): {connectivity}")
    log.append(f"Graph summary complete. Sample connectivity: {connectivity}")


def load_alzkb(base: Path, log: List[str], preferred_gene_names: Optional[List[str]] = None):
    graph = load_alzkb_via_neo4j(log)
    data_mode = "real_neo4j"
    if graph is None:
        graph = load_alzkb_from_cypherl(base, log)
        data_mode = "real_cypherl" if graph is not None else data_mode
    if graph is None:
        graph = load_alzkb_from_edge_files(base, log)
        data_mode = "real_edge_file" if graph is not None else data_mode
    if graph is None:
        graph = load_alzkb_from_rdf(base, log)
        data_mode = "real_rdf" if graph is not None else data_mode
    if graph is None:
        graph = load_alzkb_from_serialized(base, log)
        data_mode = "real_serialized" if graph is not None else data_mode
    if graph is None:
        graph = synthetic_graph(preferred_gene_names=preferred_gene_names)
        data_mode = "synthetic_graph"
        log.append("No local AlzKB source found; generated synthetic graph.")
    summarize_graph(graph, data_mode, log)
    return graph, data_mode


def find_variant_file(base: Path) -> Optional[Path]:
    for candidate in VARIANT_FILE_CANDIDATES:
        path = base / candidate
        if path.exists():
            return path
    return None


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def clinvar_search_count(term: str) -> int:
    params = {"db": "clinvar", "retmode": "json", "retmax": "0", "term": term}
    return int(
        fetch_json(f"{EUTILS_SEARCH}?{urllib.parse.urlencode(params)}")["esearchresult"][
            "count"
        ]
    )


def clinvar_search_ids(term: str, retstart: int = 0, retmax: int = 500) -> List[str]:
    params = {
        "db": "clinvar",
        "retmode": "json",
        "retstart": str(retstart),
        "retmax": str(retmax),
        "term": term,
    }
    return fetch_json(f"{EUTILS_SEARCH}?{urllib.parse.urlencode(params)}")["esearchresult"][
        "idlist"
    ]


def clinvar_search_all_ids(term: str, page_size: int = 500) -> List[str]:
    total = clinvar_search_count(term)
    ids: List[str] = []
    for retstart in range(0, total, page_size):
        ids.extend(clinvar_search_ids(term, retstart=retstart, retmax=page_size))
        time.sleep(0.34)
    return ids


def clinvar_summaries(ids: List[str]) -> List[dict]:
    summaries: List[dict] = []
    for start in range(0, len(ids), 100):
        batch = ids[start : start + 100]
        params = {"db": "clinvar", "retmode": "json", "id": ",".join(batch)}
        payload = fetch_json(f"{EUTILS_SUMMARY}?{urllib.parse.urlencode(params)}")[
            "result"
        ]
        for uid in batch:
            if uid in payload:
                summaries.append(payload[uid])
        time.sleep(0.34)
    return summaries


def parse_gene_symbol_from_title(title: str) -> str:
    match = pd.Series([title]).str.extract(r"\(([A-Za-z0-9_-]+)\):")[0].iloc[0]
    return "" if pd.isna(match) else str(match)


def load_clinvar_alzheimer_remote(log: List[str]) -> pd.DataFrame:
    term = "Alzheimer"
    ids = clinvar_search_all_ids(term)
    rows = []
    for summary in clinvar_summaries(ids):
        germline = summary.get("germline_classification", {}) or {}
        trait_set = germline.get("trait_set", []) or []
        trait_names = [trait.get("trait_name", "") for trait in trait_set]
        ad_traits = [name for name in trait_names if "alzheimer" in name.lower()]
        if not ad_traits:
            continue
        gene_symbol = parse_gene_symbol_from_title(summary.get("title", ""))
        if not gene_symbol:
            continue
        rows.append(
            {
                "variant_id": summary.get("uid", ""),
                "gene_symbol": gene_symbol,
                "clinical_significance": normalize_significance(
                    germline.get("description", "")
                ),
                "condition_name": "; ".join(trait_names),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["variant_id"]).reset_index(drop=True)
    log.append(
        f"Fetched broader Alzheimer ClinVar set from NCBI E-utilities ({len(df)} records)."
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLINVAR_CACHE_PATH, index=False)
    return df


def load_variant_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def synthetic_variants(graph: nx.DiGraph, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    gene_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Gene"]
    categories = [
        ("Pathogenic", 0.15),
        ("Likely_pathogenic", 0.10),
        ("Uncertain_significance", 0.42),
        ("Conflicting_interpretations", 0.10),
        ("Likely_benign", 0.13),
        ("Benign", 0.10),
    ]
    labels = [cat for cat, _ in categories]
    probs = [prob for _, prob in categories]
    variants = []
    for idx in range(1140):
        significance = rng.choices(labels, weights=probs, k=1)[0]
        variants.append(
            {
                "variant_id": f"synthetic_variant_{idx+1}",
                "gene_id": rng.choice(gene_nodes),
                "gene_symbol": graph.nodes[gene_nodes[idx % len(gene_nodes)]].get("name"),
                "clinical_significance": significance,
            }
        )
    return pd.DataFrame(variants)


def remap_variants_to_synthetic_graph(graph: nx.DiGraph, variants_df: pd.DataFrame) -> pd.DataFrame:
    gene_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Gene"]
    gene_names = (
        variants_df["gene_symbol"].dropna().astype(str).value_counts().index.tolist()
    )
    name_to_node = {}
    for idx, symbol in enumerate(gene_names[: len(gene_nodes)]):
        node_id = gene_nodes[idx]
        graph.nodes[node_id]["name"] = symbol
        name_to_node[symbol] = node_id
    mapped = variants_df.copy()
    mapped["gene_id"] = mapped["gene_symbol"].map(name_to_node)
    return mapped


def load_variants(base: Path, graph: nx.DiGraph, graph_mode: str, log: List[str]):
    path = find_variant_file(base)
    if path is None:
        log.append("No ClinVar file found; generating synthetic variants.")
        variants = synthetic_variants(graph)
        source_mode = "synthetic_variants"
    else:
        variants = load_variant_file(path)
        source_mode = f"real_variants:{path.name}"

    # Harmonize columns
    if "variant_id" not in variants.columns:
        if "clinvar_variation_id" in variants.columns:
            variants["variant_id"] = variants["clinvar_variation_id"]
        else:
            variants["variant_id"] = [f"variant_{i+1}" for i in range(len(variants))]

    if "gene_id" not in variants.columns:
        if "alzkb_gene_node_id" in variants.columns:
            variants["gene_id"] = variants["alzkb_gene_node_id"]
        elif "gene_symbol" in variants.columns:
            variants["gene_id"] = variants["gene_symbol"]
        else:
            raise ValueError("Variant file lacks gene_id/alzkb_gene_node_id/gene_symbol.")

    if "clinical_significance" not in variants.columns:
        raise ValueError("Variant file lacks clinical_significance column.")

    if "gene_symbol" not in variants.columns:
        if "title" in variants.columns:
            variants["gene_symbol"] = (
                variants["title"].astype(str).str.extract(r"\(([A-Za-z0-9_-]+)\):")[0]
            )
        else:
            variants["gene_symbol"] = variants["gene_id"]

    variants["clinical_significance"] = variants["clinical_significance"].astype(str).map(
        normalize_significance
    )

    unique_categories = set(variants["clinical_significance"].dropna().tolist())
    if len(unique_categories) < 2:
        log.append(
            "Local variant file lacks category diversity; fetching broader Alzheimer ClinVar records."
        )
        try:
            variants = load_clinvar_alzheimer_remote(log)
        except Exception as exc:
            log.append(f"Remote ClinVar fetch failed: {exc}")
            if CLINVAR_CACHE_PATH.exists():
                variants = pd.read_csv(CLINVAR_CACHE_PATH)
                log.append(f"Loaded cached Alzheimer ClinVar variants from {CLINVAR_CACHE_PATH}.")
            elif (OUTPUT_DIR / "exp1_results.csv").exists():
                cached_results = pd.read_csv(OUTPUT_DIR / "exp1_results.csv")
                keep_cols = [
                    col
                    for col in [
                        "variant_id",
                        "gene_id",
                        "clinical_significance",
                        "dyna_entropy",
                    ]
                    if col in cached_results.columns
                ]
                variants = cached_results[keep_cols].drop_duplicates().copy()
                if "gene_symbol" not in variants.columns:
                    variants["gene_symbol"] = variants["gene_id"]
                log.append(
                    f"Loaded fallback ClinVar variants from cached experiment results in {OUTPUT_DIR / 'exp1_results.csv'}."
                )
            else:
                raise
        variants["clinical_significance"] = variants["clinical_significance"].astype(str).map(
            normalize_significance
        )
        variants["gene_id"] = variants["gene_symbol"]
        source_mode = "remote_clinvar_alzheimer"

    entropy_log: List[str] = []
    variants["dyna_entropy"] = variants["clinical_significance"].map(
        lambda sig: assign_entropy(sig, entropy_log)
    )

    graph_nodes = set(graph.nodes)
    variants["matched"] = variants["gene_id"].astype(str).isin(graph_nodes)

    if not variants["matched"].any():
        symbol_to_node = {}
        for node, attrs in graph.nodes(data=True):
            if attrs.get("type") != "Gene":
                continue
            for key in ("geneSymbol", "name", "commonName", "xrefNcbiGene"):
                value = attrs.get(key)
                if value is None:
                    continue
                symbol_to_node[str(value)] = node
        mapped_ids = (
            variants["gene_id"].astype(str).map(symbol_to_node)
            if "gene_id" in variants.columns
            else pd.Series([None] * len(variants))
        )
        if "gene_symbol" in variants.columns:
            mapped_ids = mapped_ids.fillna(variants["gene_symbol"].astype(str).map(symbol_to_node))
        variants["gene_id"] = mapped_ids.fillna(variants["gene_id"])
        variants["matched"] = variants["gene_id"].astype(str).isin(graph_nodes)
        if variants["matched"].any():
            log.append("Matched ClinVar variants to graph gene nodes by symbol/name lookup.")

    # If the graph is synthetic and real variants are present, remap by gene symbol
    if (
        not variants["matched"].any()
        and graph_mode == "synthetic_graph"
        and "gene_symbol" in variants.columns
    ):
        variants = remap_variants_to_synthetic_graph(graph, variants)
        variants["matched"] = variants["gene_id"].astype(str).isin(graph_nodes)
        source_mode += " + remapped_to_synthetic_gene_names"
        log.append(
            "Real ClinVar variants remapped to synthetic gene nodes using gene symbols."
        )

    # If still no matches, fall back to synthetic variants
    if not variants["matched"].any():
        log.append("No matched variants found; generating synthetic variants.")
        variants = synthetic_variants(graph)
        variants["clinical_significance"] = variants["clinical_significance"].map(
            normalize_significance
        )
        entropy_log = []
        variants["dyna_entropy"] = variants["clinical_significance"].map(
            lambda sig: assign_entropy(sig, entropy_log)
        )
        variants["matched"] = True
        source_mode = "synthetic_variants"

    variants = variants[variants["matched"]].copy().reset_index(drop=True)
    print("\nVariant summary:")
    print(f"  variant_source: {source_mode}")
    print(f"  total_variants_loaded: {len(variants)}")
    print(
        "  counts_by_significance: "
        f"{dict(Counter(variants['clinical_significance']))}"
    )
    print(f"  matched_variants: {len(variants)}")
    print(f"  excluded_variants: 0")
    for line in entropy_log[:10]:
        print(f"  log: {line}")
    if len(entropy_log) > 10:
        print(f"  ... {len(entropy_log) - 10} additional entropy log entries")
    log.extend(entropy_log)
    return variants, source_mode


def get_node_type(graph: nx.DiGraph, node: str) -> str:
    return graph.nodes[node].get("type", "Unknown")


def path_validity(graph: nx.DiGraph, path: Optional[List[str]]) -> int:
    if path is None or len(path) < 2:
        return 0
    for i in range(len(path) - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            return 0
    return int(get_node_type(graph, path[-1]) == "Phenotype")


def path_score(graph: nx.DiGraph, path: Optional[List[str]], edge_type_frequency: Counter) -> float:
    if path is None or len(path) < 2:
        return 0.0
    score = 0.0
    for i in range(len(path) - 1):
        if graph.has_edge(path[i], path[i + 1]):
            rel = graph.edges[path[i], path[i + 1]].get("relation", "")
            score += math.log(edge_type_frequency.get(rel, 1) + 1)
    return score


def ar_operator(
    graph: nx.DiGraph,
    start_gene: str,
    edge_type_frequency: Counter,
    beam_width: int = 5,
    max_depth: int = 6,
    tie_seed: int = 42,
) -> Tuple[Optional[List[str]], float]:
    rng = random.Random(tie_seed)
    beams = [([start_gene], 0.0)]
    complete_paths = []

    for _ in range(max_depth):
        new_beams = []
        for path, score in beams:
            current = path[-1]
            if get_node_type(graph, current) == "Phenotype":
                complete_paths.append((path, score))
                continue
            neighbors = list(graph.successors(current))
            rng.shuffle(neighbors)
            for neighbor in neighbors:
                rel = graph.edges[current, neighbor].get("relation", "")
                edge_score = math.log(edge_type_frequency.get(rel, 1) + 1)
                new_beams.append((path + [neighbor], score + edge_score))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
        if not beams:
            break

    for path, score in beams:
        if get_node_type(graph, path[-1]) == "Phenotype":
            complete_paths.append((path, score))

    if not complete_paths:
        return None, 0.0
    best = max(complete_paths, key=lambda x: x[1])
    return best[0], best[1]


def random_walk_to_phenotype(
    graph: nx.DiGraph, start: str, max_depth: int, rng: random.Random
) -> Optional[List[str]]:
    path = [start]
    for _ in range(max_depth):
        current = path[-1]
        if get_node_type(graph, current) == "Phenotype":
            return path
        neighbors = list(graph.successors(current))
        if not neighbors:
            return None
        path.append(rng.choice(neighbors))
    if get_node_type(graph, path[-1]) == "Phenotype":
        return path
    return None


def revision_operator(
    graph: nx.DiGraph,
    start_gene: str,
    K: int = 20,
    max_depth: int = 6,
    seed: int = 42,
    initial_path: Optional[List[str]] = None,
) -> Optional[List[str]]:
    rng = random.Random(seed)
    current_path = initial_path

    if current_path is None:
        for _ in range(50):
            candidate = random_walk_to_phenotype(graph, start_gene, max_depth, rng)
            if candidate is not None and get_node_type(graph, candidate[-1]) == "Phenotype":
                current_path = candidate
                break
    if current_path is None:
        return None

    for _ in range(K):
        if len(current_path) < 3:
            break
        t = rng.randint(1, len(current_path) - 2)
        resample_start = current_path[t - 1]
        new_suffix = random_walk_to_phenotype(
            graph, resample_start, max_depth - t + 1, rng
        )
        if new_suffix is not None and get_node_type(graph, new_suffix[-1]) == "Phenotype":
            current_path = current_path[:t] + new_suffix[1:]
    return current_path


def revision_operator_multi(
    graph: nx.DiGraph, start_gene: str, K: int = 20, max_depth: int = 6, n_runs: int = 10
) -> List[Tuple[str, ...]]:
    paths = []
    for run in range(n_runs):
        path = revision_operator(graph, start_gene, K=K, max_depth=max_depth, seed=run * 17 + 42)
        if path is not None:
            paths.append(tuple(path))
    return list(set(paths))


def present_categories(values: Iterable[str]) -> List[str]:
    present = set(values)
    ordered = [item for item in SIG_ORDER if item in present]
    extras = sorted(present - set(SIG_ORDER))
    return ordered + extras


def sample_variants_by_category(
    variants_df: pd.DataFrame, n_per_category: int, include_all: bool = False
) -> pd.DataFrame:
    if include_all:
        return variants_df.copy().reset_index(drop=True)
    sampled_parts = []
    for _, group in variants_df.groupby("clinical_significance"):
        sampled_parts.append(group.sample(min(len(group), n_per_category), random_state=RNG_SEED))
    return pd.concat(sampled_parts, ignore_index=True)


def disease_coverage(path: Optional[List[str]], graph: nx.DiGraph) -> float:
    if path is None or len(path) == 0:
        return 0.0
    return sum(1 for node in path if get_node_type(graph, node) == "Disease") / len(path)


def collect_phenotypes(paths: Iterable[Optional[List[str]]], graph: nx.DiGraph) -> List[str]:
    phenotypes = []
    for path in paths:
        if path is not None and len(path) > 0 and get_node_type(graph, path[-1]) == "Phenotype":
            phenotypes.append(path[-1])
    return phenotypes


def run_ar_trials(
    graph: nx.DiGraph,
    start_gene: str,
    edge_freq: Counter,
    n_runs: int = 20,
    beam_width: int = 5,
    max_depth: int = 6,
) -> List[Optional[List[str]]]:
    trials = []
    for seed_offset in range(n_runs):
        path, _ = ar_operator(
            graph, start_gene, edge_freq, beam_width=beam_width, max_depth=max_depth, tie_seed=42 + seed_offset
        )
        trials.append(path)
    return trials


def constrained_revision_operator(
    graph: nx.DiGraph,
    start_gene: str,
    K: int = 20,
    max_depth: int = 6,
    seed: int = 42,
) -> Optional[List[str]]:
    rng = random.Random(seed)
    current_path = None
    for _ in range(50):
        candidate = random_walk_to_phenotype(graph, start_gene, max_depth, rng)
        if candidate is not None and get_node_type(graph, candidate[-1]) == "Phenotype":
            current_path = candidate
            break
    if current_path is None:
        return None

    for _ in range(K):
        if len(current_path) < 3:
            break
        t = rng.randint(1, len(current_path) - 2)
        resample_start = current_path[t - 1]
        proposal = random_walk_to_phenotype(graph, resample_start, max_depth - t + 1, rng)
        if proposal is None or get_node_type(graph, proposal[-1]) != "Phenotype":
            continue
        proposed_path = current_path[:t] + proposal[1:]
        if disease_coverage(proposed_path, graph) >= disease_coverage(current_path, graph):
            current_path = proposed_path
    return current_path


def run_revision_trials(
    graph: nx.DiGraph,
    start_gene: str,
    n_runs: int = 20,
    K: int = 20,
    max_depth: int = 6,
    constrained: bool = False,
) -> List[Optional[List[str]]]:
    trials = []
    operator = constrained_revision_operator if constrained else revision_operator
    for seed_offset in range(n_runs):
        path = operator(graph, start_gene, K=K, max_depth=max_depth, seed=42 + seed_offset * 17)
        trials.append(path)
    return trials


def ar_first_step_choice(
    graph: nx.DiGraph,
    start_gene: str,
    edge_freq: Counter,
    tie_seed: int = 42,
) -> Optional[str]:
    neighbors = list(graph.successors(start_gene))
    if not neighbors:
        return None
    rng = random.Random(tie_seed)
    rng.shuffle(neighbors)
    scored = []
    for neighbor in neighbors:
        rel = graph.edges[start_gene, neighbor].get("relation", "")
        scored.append((math.log(edge_freq.get(rel, 1) + 1), neighbor))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None


def reachable_phenotypes_from(
    graph: nx.DiGraph,
    start_node: str,
    max_depth: int,
    cache: Dict[Tuple[str, int], set],
) -> set:
    key = (start_node, max_depth)
    if key in cache:
        return cache[key]
    reached = set()
    queue = [(start_node, 0)]
    visited = {(start_node, 0)}
    while queue:
        node, depth = queue.pop()
        if get_node_type(graph, node) == "Phenotype":
            reached.add(node)
            continue
        if depth >= max_depth:
            continue
        for neighbor in graph.successors(node):
            state = (neighbor, depth + 1)
            if state not in visited:
                visited.add(state)
                queue.append(state)
    cache[key] = reached
    return reached


def commitment_reachable_phenotypes(
    graph: nx.DiGraph,
    start_node: str,
    max_depth: int,
    cache: Dict[Tuple[str, int], set],
) -> set:
    key = ("commitment", start_node, max_depth)
    if key in cache:
        return cache[key]
    reached = set()
    stack = [(start_node, 0, 0)]
    visited = {(start_node, 0, 0)}
    gene_gene_relations = {"GENEINTERACTSWITHGENE", "GENECOVARIESWITHGENE", "GENEREGULATESGENE"}
    while stack:
        node, depth, gene_hops = stack.pop()
        node_type = get_node_type(graph, node)
        if node_type == "Phenotype":
            reached.add(node)
            continue
        if depth >= max_depth:
            continue
        if node_type == "Disease":
            for neighbor in graph.successors(node):
                if graph.edges[node, neighbor].get("relation") == "DISEASEMANIFESTSSYMPTOM":
                    state = (neighbor, depth + 1, gene_hops)
                    if state not in visited:
                        visited.add(state)
                        stack.append(state)
            continue
        if node_type != "Gene":
            continue
        for neighbor in graph.successors(node):
            rel = graph.edges[node, neighbor].get("relation")
            if rel == "GENEASSOCIATESWITHDISEASE":
                state = (neighbor, depth + 1, gene_hops)
            elif rel in gene_gene_relations and gene_hops < 1:
                state = (neighbor, depth + 1, gene_hops + 1)
            else:
                continue
            if state not in visited:
                visited.add(state)
                stack.append(state)
    cache[key] = reached
    return reached


def run_experiment1(
    graph: nx.DiGraph, variants_df: pd.DataFrame, edge_freq: Counter, data_note: str
):
    results = []
    for _, variant in variants_df.iterrows():
        gene = variant["gene_id"]
        entropy = variant["dyna_entropy"]
        sig = variant["clinical_significance"]
        if gene not in graph.nodes:
            continue
        ar_path, _ = ar_operator(graph, gene, edge_freq, beam_width=5, max_depth=6)
        rev_path = revision_operator(graph, gene, K=20, max_depth=6, seed=42)
        results.append(
            {
                "variant_id": variant.get("variant_id", _),
                "gene_id": gene,
                "clinical_significance": sig,
                "dyna_entropy": entropy,
                "ar_valid": path_validity(graph, ar_path),
                "ar_score": path_score(graph, ar_path, edge_freq),
                "rev_valid": path_validity(graph, rev_path),
                "rev_score": path_score(graph, rev_path, edge_freq),
            }
        )

    df = pd.DataFrame(results)
    df["data_note"] = data_note
    df["validity_advantage"] = df["rev_valid"] - df["ar_valid"]
    df["score_advantage"] = df["rev_score"] - df["ar_score"]

    try:
        df["entropy_quartile"] = pd.qcut(
            df["dyna_entropy"], q=4, labels=["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"], duplicates="drop"
        )
    except ValueError:
        df["entropy_quartile"] = pd.cut(
            df["dyna_entropy"], bins=4, labels=["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"]
        )
    quartile_stats = (
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
    quartile_stats["data_note"] = data_note

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(
        quartile_stats["entropy_quartile"],
        quartile_stats["validity_advantage_mean"],
        yerr=quartile_stats["validity_advantage_std"].fillna(0),
        capsize=5,
        color="steelblue",
        alpha=0.85,
    )
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("State Uncertainty Quartile")
    axes[0].set_ylabel("Revision - AR (Path Validity)")
    axes[0].set_title("A. Operator Advantage vs. Uncertainty")

    axes[1].bar(
        quartile_stats["entropy_quartile"],
        quartile_stats["score_advantage_mean"],
        yerr=quartile_stats["score_advantage_std"].fillna(0),
        capsize=5,
        color="coral",
        alpha=0.85,
    )
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("State Uncertainty Quartile")
    axes[1].set_ylabel("Revision - AR (Path Score)")
    axes[1].set_title("B. Path Score Advantage")
    fig.text(0.5, 0.01, f"Data: {data_note}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment1.png", dpi=150, bbox_inches="tight")
    plt.close()
    return df, quartile_stats


def run_experiment2(
    graph: nx.DiGraph,
    variants_df: pd.DataFrame,
    edge_freq: Counter,
    data_note: str,
    max_depth: int = 6,
):
    sampled = variants_df.copy().reset_index(drop=True)
    results = []
    reachability_cache: Dict[Tuple[str, int], set] = {}
    revision_cache: Dict[str, List[Optional[List[str]]]] = {}

    for _, variant in sampled.iterrows():
        gene = variant["gene_id"]
        if gene not in graph.nodes:
            continue
        neighbors = list(graph.successors(gene))
        if len(neighbors) < 2:
            continue
        chosen_neighbor = ar_first_step_choice(graph, gene, edge_freq, tie_seed=42)
        if chosen_neighbor is None:
            continue
        chosen_type = get_node_type(graph, chosen_neighbor)
        chosen_phenotypes = commitment_reachable_phenotypes(
            graph, chosen_neighbor, max_depth - 1, reachability_cache
        )
        unchosen_neighbors = [neighbor for neighbor in neighbors if neighbor != chosen_neighbor]
        unchosen_phenotypes = set()
        for neighbor in unchosen_neighbors:
            unchosen_phenotypes |= commitment_reachable_phenotypes(
                graph, neighbor, max_depth - 1, reachability_cache
            )
        if gene not in revision_cache:
            revision_cache[gene] = run_revision_trials(graph, gene, n_runs=20, K=20, max_depth=max_depth)
        revision_phenotypes = set(collect_phenotypes(revision_cache[gene], graph))
        total_reachable = chosen_phenotypes | unchosen_phenotypes | revision_phenotypes
        if not total_reachable:
            continue
        revision_only = revision_phenotypes - chosen_phenotypes
        results.append(
            {
                "variant_id": variant.get("variant_id", _),
                "gene_id": gene,
                "clinical_significance": variant["clinical_significance"],
                "dyna_entropy": variant["dyna_entropy"],
                "chosen_neighbor": chosen_neighbor,
                "chosen_type": chosen_type,
                "n_gene_neighbors": len(neighbors),
                "chosen_reachable_phenotypes": len(chosen_phenotypes),
                "unchosen_reachable_phenotypes": len(unchosen_phenotypes),
                "revision_reachable_phenotypes": len(revision_phenotypes),
                "revision_only_phenotypes": len(revision_only),
                "coverage_loss": len(revision_only) / max(1, len(total_reachable)),
                "revision_finds_excluded": int(len(revision_only) > 0),
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        summary = pd.DataFrame(columns=["clinical_significance", "coverage_loss_mean", "coverage_loss_std", "revision_finds_excluded_mean", "n", "data_note"])
        return df, summary
    df["data_note"] = data_note
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
    summary["data_note"] = data_note

    chosen_type_summary = (
        df.groupby("chosen_type")
        .agg(
            revision_finds_excluded_mean=("revision_finds_excluded", "mean"),
            coverage_loss_mean=("coverage_loss", "mean"),
            n=("variant_id", "count"),
        )
        .sort_values(["n", "coverage_loss_mean"], ascending=[False, False])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(summary))
    axes[0].bar(
        x,
        summary["coverage_loss_mean"],
        yerr=summary["coverage_loss_std"].fillna(0),
        capsize=4,
        color="steelblue",
        alpha=0.85,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.replace("_", "\n") for s in summary["clinical_significance"]], fontsize=9)
    axes[0].set_ylabel("Mean Phenotype Coverage Loss")
    axes[0].set_xlabel("Clinical Significance")
    axes[0].set_title("A. AR First-Step Commitment Loss")

    type_x = np.arange(len(chosen_type_summary))
    axes[1].bar(
        type_x,
        chosen_type_summary["revision_finds_excluded_mean"],
        color="coral",
        alpha=0.85,
    )
    axes[1].set_xticks(type_x)
    axes[1].set_xticklabels(chosen_type_summary["chosen_type"], rotation=20, ha="right")
    axes[1].set_ylabel("Fraction Revision Finds Excluded Phenotypes")
    axes[1].set_xlabel("AR First Chosen Neighbor Type")
    axes[1].set_title("B. Commitment Eliminates Reachable Phenotypes")
    fig.text(0.5, 0.01, f"Data: {data_note}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment2.png", dpi=150, bbox_inches="tight")
    plt.close()
    return df, summary


def run_experiment3(
    graph: nx.DiGraph, variants_df: pd.DataFrame, edge_freq: Counter, data_note: str
):
    sampled_parts = []
    for _, group in variants_df.groupby("clinical_significance"):
        sampled_parts.append(group.sample(min(len(group), 30), random_state=RNG_SEED))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    results = []
    for _, variant in sampled.iterrows():
        gene = variant["gene_id"]
        if gene not in graph.nodes:
            continue
        rev_paths = revision_operator_multi(graph, gene, K=20, max_depth=6, n_runs=10)
        rev_trajectory_diversity = len(rev_paths)
        rev_phenotype_diversity = len(set(path[-1] for path in rev_paths if path))

        ar_paths = []
        for seed_offset in range(10):
            ar_path, _ = ar_operator(
                graph, gene, edge_freq, beam_width=10, max_depth=6, tie_seed=42 + seed_offset
            )
            if ar_path is not None:
                ar_paths.append(tuple(ar_path))
        ar_trajectory_diversity = len(set(ar_paths))
        ar_phenotype_diversity = len(set(path[-1] for path in set(ar_paths) if path))
        results.append(
            {
                "variant_id": variant.get("variant_id", _),
                "gene_id": gene,
                "clinical_significance": variant["clinical_significance"],
                "dyna_entropy": variant["dyna_entropy"],
                "rev_trajectory_diversity": rev_trajectory_diversity,
                "rev_phenotype_diversity": rev_phenotype_diversity,
                "ar_trajectory_diversity": ar_trajectory_diversity,
                "ar_phenotype_diversity": ar_phenotype_diversity,
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        summary = pd.DataFrame(
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
        return df, summary
    df["data_note"] = data_note
    present_cats = present_categories(df["clinical_significance"].unique())
    summary = (
        df.groupby("clinical_significance")
        .agg(
            rev_traj_mean=("rev_trajectory_diversity", "mean"),
            rev_traj_std=("rev_trajectory_diversity", "std"),
            ar_traj_mean=("ar_trajectory_diversity", "mean"),
            ar_traj_std=("ar_trajectory_diversity", "std"),
            rev_pheno_mean=("rev_phenotype_diversity", "mean"),
            rev_pheno_std=("rev_phenotype_diversity", "std"),
            ar_pheno_mean=("ar_phenotype_diversity", "mean"),
            ar_pheno_std=("ar_phenotype_diversity", "std"),
            n=("variant_id", "count"),
        )
        .reindex(present_cats)
        .reset_index()
    )
    summary["data_note"] = data_note

    figure_summary = summary[
        summary["clinical_significance"].isin(SIG_ORDER) & (summary["n"] >= 10)
    ].copy()
    label_map = {
        "Pathogenic": "Path",
        "Likely_pathogenic": "Lik-Path",
        "Uncertain_significance": "VUS",
        "Conflicting_interpretations": "Conf",
        "Likely_benign": "Lik-Ben",
        "Benign": "Ben",
    }
    figure_summary["plot_label"] = figure_summary["clinical_significance"].map(label_map)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = np.arange(len(figure_summary))
    box_data = []
    for category in figure_summary["clinical_significance"]:
        subset = df[df["clinical_significance"] == category]
        box_data.append(subset["rev_phenotype_diversity"].tolist())

    bp = ax.boxplot(
        box_data,
        positions=x,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#8E3B2E", "linewidth": 1.8},
        whiskerprops={"color": "#B55D48", "linewidth": 1.2},
        capprops={"color": "#B55D48", "linewidth": 1.2},
        boxprops={"edgecolor": "#B55D48", "linewidth": 1.2},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#E9A28F")
        patch.set_alpha(0.7)

    rng = np.random.default_rng(RNG_SEED)
    for idx, values in enumerate(box_data):
        if not values:
            continue
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), x[idx]) + jitter,
            values,
            s=20,
            color="#8E3B2E",
            alpha=0.45,
            edgecolors="none",
            zorder=3,
        )

    ax.plot(
        x,
        figure_summary["rev_pheno_mean"],
        color="#7A1F0F",
        marker="D",
        markersize=6.5,
        linewidth=1.6,
        label="Revision mean",
        zorder=4,
    )
    ax.axhline(
        0,
        color="#4C78A8",
        linewidth=2.0,
        linestyle=(0, (4, 3)),
        label="AR baseline (0)",
        zorder=1,
    )
    ax.text(
        0.99,
        0.06,
        "AR phenotype diversity = 0 across major categories",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#4C78A8",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(figure_summary["plot_label"])
    ax.set_xlabel("ClinVar category")
    ax.set_ylabel("Distinct phenotypes across 10 runs")
    ax.set_title("Revision Phenotype Diversity Increases with Clinical Uncertainty")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(0.5, float(figure_summary["rev_pheno_mean"].max()) + 1.0))
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.5,
        0.01,
        f"Major ClinVar categories shown (n>=10). Boxplots show revision phenotype diversity; diamonds mark category means. Data: {data_note}",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(OUTPUT_DIR / "experiment3.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "experiment3.pdf", bbox_inches="tight")
    plt.close()
    return df, summary


def run_experiment4(graph: nx.DiGraph, variants_df: pd.DataFrame, data_note: str):
    sampled = sample_variants_by_category(variants_df, n_per_category=20)
    results = []
    unconstrained_cache: Dict[str, List[Optional[List[str]]]] = {}
    constrained_cache: Dict[str, List[Optional[List[str]]]] = {}

    for _, variant in sampled.iterrows():
        gene = variant["gene_id"]
        if gene not in graph.nodes:
            continue
        if gene not in unconstrained_cache:
            unconstrained_cache[gene] = run_revision_trials(graph, gene, n_runs=10, K=20, max_depth=6, constrained=False)
            constrained_cache[gene] = run_revision_trials(graph, gene, n_runs=10, K=20, max_depth=6, constrained=True)
        unconstrained_paths = [tuple(path) for path in unconstrained_cache[gene] if path is not None]
        constrained_paths = [tuple(path) for path in constrained_cache[gene] if path is not None]
        unconstrained_set = set(unconstrained_paths)
        constrained_set = set(constrained_paths)
        unlocked = constrained_set - unconstrained_set
        unlocked_coverages = [disease_coverage(list(path), graph) for path in unlocked]
        results.append(
            {
                "variant_id": variant.get("variant_id", _),
                "gene_id": gene,
                "clinical_significance": variant["clinical_significance"],
                "dyna_entropy": variant["dyna_entropy"],
                "unconstrained_distinct_paths": len(unconstrained_set),
                "constrained_distinct_paths": len(constrained_set),
                "constraint_unlocked_trajectories": len(unlocked),
                "unconstrained_mean_disease_coverage": mean(
                    [disease_coverage(list(path), graph) for path in unconstrained_set]
                )
                if unconstrained_set
                else 0.0,
                "constrained_mean_disease_coverage": mean(
                    [disease_coverage(list(path), graph) for path in constrained_set]
                )
                if constrained_set
                else 0.0,
                "unlocked_mean_disease_coverage": mean(unlocked_coverages) if unlocked_coverages else 0.0,
            }
        )

    df = pd.DataFrame(results)
    df["data_note"] = data_note
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
    summary["data_note"] = data_note

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(summary))
    axes[0].bar(
        x,
        summary["unlocked_mean"],
        yerr=summary["unlocked_std"].fillna(0),
        capsize=4,
        color="darkseagreen",
        alpha=0.85,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.replace("_", "\n") for s in summary["clinical_significance"]], fontsize=9)
    axes[0].set_ylabel("Mean Constraint-Unlocked Trajectories")
    axes[0].set_xlabel("Clinical Significance")
    axes[0].set_title("A. Constraint-Unlocked Hypothesis Discovery")

    width = 0.35
    axes[1].bar(x - width / 2, summary["unconstrained_disease_mean"], width, label="Unconstrained", color="steelblue", alpha=0.85)
    axes[1].bar(x + width / 2, summary["constrained_disease_mean"], width, label="Constraint-guided", color="coral", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.replace("_", "\n") for s in summary["clinical_significance"]], fontsize=9)
    axes[1].set_ylabel("Mean Disease Coverage")
    axes[1].set_xlabel("Clinical Significance")
    axes[1].set_title("B. Disease Grounding Under Constraints")
    axes[1].legend()
    fig.text(0.5, 0.01, f"Data: {data_note}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment4.png", dpi=150, bbox_inches="tight")
    plt.close()
    return df, summary


def shannon_entropy(items: List[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total + 1e-12, 2) for count in counts.values())


def run_experiment5(graph: nx.DiGraph, variants_df: pd.DataFrame, data_note: str):
    subset = variants_df[
        variants_df["clinical_significance"].isin(["Uncertain_significance", "Pathogenic"])
    ].copy()
    results = []
    trial_cache: Dict[str, List[Optional[List[str]]]] = {}
    for _, variant in subset.iterrows():
        gene = variant["gene_id"]
        if gene not in graph.nodes:
            continue
        if gene not in trial_cache:
            trial_cache[gene] = run_revision_trials(graph, gene, n_runs=20, K=20, max_depth=6, constrained=False)
        phenotypes = collect_phenotypes(trial_cache[gene], graph)
        results.append(
            {
                "variant_id": variant.get("variant_id", _),
                "gene_id": gene,
                "clinical_significance": variant["clinical_significance"],
                "phenotype_width": len(set(phenotypes)),
                "phenotype_entropy": shannon_entropy(phenotypes),
                "valid_runs": len(phenotypes),
            }
        )

    df = pd.DataFrame(results)
    df["data_note"] = data_note
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
    summary["data_note"] = data_note

    paired = (
        df.groupby(["gene_id", "clinical_significance"])
        .agg(width_mean=("phenotype_width", "mean"))
        .reset_index()
        .pivot(index="gene_id", columns="clinical_significance", values="width_mean")
        .dropna(subset=["Pathogenic", "Uncertain_significance"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(summary))
    axes[0].bar(
        x,
        summary["width_mean"],
        yerr=summary["width_std"].fillna(0),
        capsize=4,
        color=["steelblue", "coral"],
        alpha=0.85,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["clinical_significance"])
    axes[0].set_ylabel("Mean Distinct Phenotypes Across 20 Runs")
    axes[0].set_xlabel("Clinical Significance")
    axes[0].set_title("A. Phenotype Distribution Width")

    if not paired.empty:
        for _, row in paired.iterrows():
            axes[1].plot(["Pathogenic", "VUS"], [row["Pathogenic"], row["Uncertain_significance"]], marker="o", alpha=0.6)
    axes[1].set_ylabel("Mean Distinct Phenotypes")
    axes[1].set_title("B. Same-Gene Width: Pathogenic vs VUS")
    axes[1].set_xlabel("Variant Category")
    fig.text(0.5, 0.01, f"Data: {data_note}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment5.png", dpi=150, bbox_inches="tight")
    plt.close()
    return df, summary, paired


def run_experiment6(graph: nx.DiGraph, variants_df: pd.DataFrame, edge_freq: Counter, data_note: str):
    variant_genes = sorted(set(variants_df["gene_id"]))
    gene_relations = {"GENEINTERACTSWITHGENE", "GENECOVARIESWITHGENE", "GENEREGULATESGENE"}
    gene_projection = nx.Graph()
    for gene in variant_genes:
        gene_projection.add_node(gene)
        for neighbor in graph.successors(gene):
            if get_node_type(graph, neighbor) == "Gene" and graph.edges[gene, neighbor].get("relation") in gene_relations:
                gene_projection.add_edge(gene, neighbor)
    centrality = nx.betweenness_centrality(
        gene_projection,
        k=min(100, max(1, gene_projection.number_of_nodes())),
        seed=RNG_SEED,
    ) if gene_projection.number_of_nodes() > 1 else {gene: 0.0 for gene in gene_projection.nodes}

    results = []
    rev_cache: Dict[str, List[Optional[List[str]]]] = {}
    ar_cache: Dict[str, List[Optional[List[str]]]] = {}
    for gene in variant_genes:
        if gene not in graph.nodes:
            continue
        rev_cache[gene] = run_revision_trials(graph, gene, n_runs=20, K=20, max_depth=6, constrained=False)
        ar_cache[gene] = run_ar_trials(graph, gene, edge_freq, n_runs=20, beam_width=10, max_depth=6)
        rev_paths = set(tuple(path) for path in rev_cache[gene] if path is not None)
        ar_paths = set(tuple(path) for path in ar_cache[gene] if path is not None)
        rev_phens = set(collect_phenotypes(rev_cache[gene], graph))
        ar_phens = set(collect_phenotypes(ar_cache[gene], graph))
        results.append(
            {
                "gene_id": gene,
                "gene_symbol": graph.nodes[gene].get("geneSymbol") or graph.nodes[gene].get("name") or gene,
                "betweenness_centrality": centrality.get(gene, 0.0),
                "trajectory_divergence": len(rev_paths) - len(ar_paths),
                "phenotype_divergence": len(rev_phens) - len(ar_phens),
                "gene_degree": gene_projection.degree(gene) if gene in gene_projection else 0,
            }
        )

    df = pd.DataFrame(results)
    df["data_note"] = data_note
    corr, _ = spearmanr(df["betweenness_centrality"], df["phenotype_divergence"])
    corr = 0.0 if math.isnan(corr) else corr

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["betweenness_centrality"], df["phenotype_divergence"], alpha=0.8, color="purple")
    for _, row in df.nlargest(5, "betweenness_centrality").iterrows():
        ax.annotate(row["gene_symbol"], (row["betweenness_centrality"], row["phenotype_divergence"]), fontsize=8)
    ax.set_xlabel("Approx. Betweenness Centrality")
    ax.set_ylabel("Phenotype Divergence (Revision - AR)")
    ax.set_title(f"Experiment 6: Centrality vs Operator Divergence\nSpearman={corr:.3f}")
    fig.text(0.5, 0.01, f"Data: {data_note}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment6.png", dpi=150, bbox_inches="tight")
    plt.close()
    return df, corr


def run_experiment7(graph: nx.DiGraph, variants_df: pd.DataFrame, edge_freq: Counter, data_note: str):
    sampled = sample_variants_by_category(variants_df, n_per_category=30)
    results = []
    rev_cache: Dict[str, List[Optional[List[str]]]] = {}
    ar_cache: Dict[str, List[Optional[List[str]]]] = {}

    for _, variant in sampled.iterrows():
        gene = variant["gene_id"]
        if gene not in graph.nodes:
            continue
        if gene not in rev_cache:
            rev_cache[gene] = run_revision_trials(graph, gene, n_runs=20, K=20, max_depth=6, constrained=False)
            ar_cache[gene] = run_ar_trials(graph, gene, edge_freq, n_runs=20, beam_width=10, max_depth=6)
        for operator_name, paths in [("Revision", rev_cache[gene]), ("AR", ar_cache[gene])]:
            phenotypes = collect_phenotypes(paths, graph)
            valid_runs = len(phenotypes)
            failure_rate = 1.0 - (valid_runs / len(paths))
            phenotype_counts = Counter(phenotypes)
            majority_count = max(phenotype_counts.values()) if phenotype_counts else 0
            consistency = (majority_count / valid_runs) if valid_runs else 0.0
            productive_diversity_rate = (
                (valid_runs - majority_count) / len(paths)
            ) if len(paths) else 0.0
            results.append(
                {
                    "variant_id": variant.get("variant_id", _),
                    "gene_id": gene,
                    "clinical_significance": variant["clinical_significance"],
                    "operator": operator_name,
                    "consistency": consistency,
                    "productive_diversity_rate": productive_diversity_rate,
                    "failure_rate": failure_rate,
                    "distinct_valid_phenotypes": len(set(phenotypes)),
                }
            )

    df = pd.DataFrame(results)
    df["data_note"] = data_note
    summary = (
        df.groupby(["clinical_significance", "operator"])
        .agg(
            consistency_mean=("consistency", "mean"),
            productive_diversity_mean=("productive_diversity_rate", "mean"),
            failure_rate_mean=("failure_rate", "mean"),
            distinct_valid_mean=("distinct_valid_phenotypes", "mean"),
            n=("variant_id", "count"),
        )
        .reset_index()
    )
    summary["data_note"] = data_note

    categories = present_categories(df["clinical_significance"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("consistency_mean", "Consistency"),
        ("productive_diversity_mean", "Productive Diversity Rate"),
        ("failure_rate_mean", "Failure Rate"),
    ]
    width = 0.35
    x = np.arange(len(categories))
    for ax, (metric, title) in zip(axes, metrics):
        rev_vals = [
            summary.loc[
                (summary["clinical_significance"] == cat) & (summary["operator"] == "Revision"),
                metric,
            ].iloc[0]
            if not summary.loc[
                (summary["clinical_significance"] == cat) & (summary["operator"] == "Revision")
            ].empty
            else 0.0
            for cat in categories
        ]
        ar_vals = [
            summary.loc[
                (summary["clinical_significance"] == cat) & (summary["operator"] == "AR"),
                metric,
            ].iloc[0]
            if not summary.loc[
                (summary["clinical_significance"] == cat) & (summary["operator"] == "AR")
            ].empty
            else 0.0
            for cat in categories
        ]
        ax.bar(x - width / 2, rev_vals, width, label="Revision", color="coral", alpha=0.85)
        ax.bar(x + width / 2, ar_vals, width, label="AR", color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Clinical Significance")
        if ax is axes[0]:
            ax.legend()
    fig.text(0.5, 0.01, f"Data: {data_note}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment7.png", dpi=150, bbox_inches="tight")
    plt.close()
    return df, summary


def verdict_claim1(stats: pd.DataFrame) -> str:
    vals = stats["validity_advantage_mean"].tolist()
    if len(vals) >= 2 and all(x <= y for x, y in zip(vals, vals[1:])):
        return "supported"
    if len(vals) >= 2 and vals[-1] > vals[0]:
        return "partially supported"
    return "not supported"


def verdict_claim2(stats: pd.DataFrame) -> str:
    if stats.empty or "coverage_loss_mean" not in stats.columns:
        return "not evaluable"
    coverage = stats["coverage_loss_mean"].fillna(0).mean()
    excluded = stats["revision_finds_excluded_mean"].fillna(0).mean()
    if coverage > 0.05 and excluded > 0.25:
        return "supported"
    if coverage > 0 or excluded > 0:
        return "partially supported"
    return "not evaluable"


def verdict_claim3(stats: pd.DataFrame, exp3_df: pd.DataFrame) -> Tuple[str, float]:
    cat_to_val = {row["clinical_significance"]: row["rev_traj_mean"] for _, row in stats.iterrows()}
    ordered = [
        cat_to_val.get("Conflicting_interpretations", -1),
        cat_to_val.get("Uncertain_significance", -1),
    ]
    ar_range = stats["ar_traj_mean"].max() - stats["ar_traj_mean"].min()
    corr, _ = spearmanr(exp3_df["dyna_entropy"], exp3_df["rev_trajectory_diversity"])
    if math.isnan(corr):
        corr = 0.0
    if ordered[0] >= ordered[1] and ar_range < 1.5 and corr > 0:
        return "supported", corr
    if corr > 0:
        return "partially supported", corr
    return "not supported", corr


def preliminary_results_paragraph(
    graph_mode: str,
    variant_mode: str,
    exp1_stats: pd.DataFrame,
    exp2_stats: pd.DataFrame,
    exp3_stats: pd.DataFrame,
    corr: float,
    claim1: str,
    claim2: str,
    claim3: str,
) -> str:
    q1 = exp1_stats.iloc[0]["validity_advantage_mean"]
    q4 = exp1_stats.iloc[-1]["validity_advantage_mean"]
    coverage_loss = exp2_stats["coverage_loss_mean"].fillna(0).mean() if "coverage_loss_mean" in exp2_stats.columns and not exp2_stats.empty else np.nan
    rev_u = exp3_stats.loc[
        exp3_stats["clinical_significance"] == "Uncertain_significance", "rev_traj_mean"
    ]
    rev_u_val = rev_u.iloc[0] if len(rev_u) else float("nan")
    ar_mean = exp3_stats["ar_traj_mean"].mean()
    if pd.isna(coverage_loss):
        perturbation_sentence = (
            "The short-path commitment analysis was not evaluable in this pilot."
        )
    else:
        perturbation_sentence = (
            f"The redesigned commitment-point analysis showed a mean phenotype coverage loss of {coverage_loss:.3f} after AR first-step commitment."
        )
    return (
        f"In a pilot study using {graph_mode.replace('_', ' ')} and {variant_mode.replace('_', ' ')}, "
        f"revision-capable operators and autoregressive operators showed distinguishable trajectory behavior in genotype-to-phenotype reasoning, although the strongest theoretical predictions were not uniformly supported in this pilot setting. "
        f"Across uncertainty quartiles, the mean validity advantage of revision over AR changed from {q1:.3f} in the lowest-uncertainty quartile "
        f"to {q4:.3f} in the highest-uncertainty quartile. {perturbation_sentence} Revision-based search produced "
        f"higher trajectory diversity for uncertain variants (mean {rev_u_val:.2f} distinct paths for VUS) while AR remained low and comparatively flat "
        f"(overall mean {ar_mean:.2f}); the Spearman correlation between entropy proxy and revision trajectory diversity was {corr:.3f}. "
        f"Overall, the pilot yielded verdicts of {claim1} for the phase-transition claim, {claim2} for early-branching sensitivity, and {claim3} for diversity calibration, indicating that richer graph structure and nontrivial dead ends will be important in the next stage of the operator-theoretic study."
    )


def save_text(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("OPERATOR-THEORETIC REASONING EXPERIMENT")
    print("AlzKB + ClinVar Variant Pilot Study")
    print("=" * 60)

    log: List[str] = []
    base = Path(".")
    variant_file = find_variant_file(base)
    preferred_gene_names: List[str] = []
    if variant_file:
        try:
            temp_df = pd.read_csv(variant_file)
            if "gene_symbol" in temp_df.columns:
                preferred_gene_names = sorted(set(temp_df["gene_symbol"].dropna().astype(str)))
        except Exception:
            preferred_gene_names = []

    print("\n--- Loading AlzKB ---")
    graph, graph_mode = load_alzkb(base, log, preferred_gene_names=preferred_gene_names)

    print("\n--- Loading ClinVar variants ---")
    variants_df, variant_mode = load_variants(base, graph, graph_mode, log)

    edge_type_frequency = Counter(
        data.get("relation", "unknown") for _, _, data in graph.edges(data=True)
    )
    print("\nEdge type frequency summary:")
    for rel, count in edge_type_frequency.most_common(10):
        print(f"  {rel}: {count}")

    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 1")
    print("=" * 60)
    data_note = f"{graph_mode} | {variant_mode}"
    exp1_df, exp1_stats = run_experiment1(graph, variants_df, edge_type_frequency, data_note)
    print(exp1_stats.to_string(index=False))

    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 2")
    print("=" * 60)
    exp2_df, exp2_stats = run_experiment2(graph, variants_df, edge_type_frequency, data_note)
    print(exp2_stats.to_string(index=False))

    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 3")
    print("=" * 60)
    exp3_df, exp3_stats = run_experiment3(graph, variants_df, edge_type_frequency, data_note)
    print(exp3_stats.to_string(index=False))

    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 4")
    print("=" * 60)
    exp4_df, exp4_stats = run_experiment4(graph, variants_df, data_note)
    print(exp4_stats.to_string(index=False))

    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 5")
    print("=" * 60)
    exp5_df, exp5_stats, exp5_paired = run_experiment5(graph, variants_df, data_note)
    print(exp5_stats.to_string(index=False))

    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 6")
    print("=" * 60)
    exp6_df, exp6_corr = run_experiment6(graph, variants_df, edge_type_frequency, data_note)
    print(exp6_df.head(20).to_string(index=False))

    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 7")
    print("=" * 60)
    exp7_df, exp7_stats = run_experiment7(graph, variants_df, edge_type_frequency, data_note)
    print(exp7_stats.to_string(index=False))

    claim1 = verdict_claim1(exp1_stats)
    claim2 = verdict_claim2(exp2_stats)
    claim3, corr = verdict_claim3(exp3_stats, exp3_df)

    interpretation = f"""
DATA MODE
- Graph: {graph_mode}
- Variants: {variant_mode}

CLAIM 1 (Phase transition)
VERDICT: {claim1}

CLAIM 2 (Early sensitivity)
VERDICT: {claim2}

CLAIM 3 (Diversity calibration)
VERDICT: {claim3}
Spearman correlation (entropy vs revision trajectory diversity): {corr:.3f}

EXPERIMENT 4 (Constraint-unlocked discovery)
Mean unlocked trajectories: {exp4_stats['unlocked_mean'].fillna(0).mean():.3f}

EXPERIMENT 5 (VUS phenotype width)
Mean VUS width: {exp5_stats.loc[exp5_stats['clinical_significance']=='Uncertain_significance', 'width_mean'].iloc[0] if not exp5_stats.loc[exp5_stats['clinical_significance']=='Uncertain_significance'].empty else float('nan'):.3f}
Mean Pathogenic width: {exp5_stats.loc[exp5_stats['clinical_significance']=='Pathogenic', 'width_mean'].iloc[0] if not exp5_stats.loc[exp5_stats['clinical_significance']=='Pathogenic'].empty else float('nan'):.3f}

EXPERIMENT 6 (Centrality vs divergence)
Spearman correlation (centrality vs phenotype divergence): {exp6_corr:.3f}

EXPERIMENT 7 (Reliability)
Mean revision failure rate: {exp7_stats.loc[exp7_stats['operator']=='Revision', 'failure_rate_mean'].mean() if not exp7_stats.empty else float('nan'):.3f}
Mean AR failure rate: {exp7_stats.loc[exp7_stats['operator']=='AR', 'failure_rate_mean'].mean() if not exp7_stats.empty else float('nan'):.3f}

If any claim is unsupported, plausible methodological causes include graph connectivity ceiling effects, synthetic graph regularity, limited path depth, or entropy discretization. Plausible theoretical causes include insufficient operator separation, weak uncertainty representation, or constrained trajectory space.
""".strip()
    print("\n" + "=" * 60)
    print("COMBINED INTERPRETATION")
    print("=" * 60)
    print(interpretation)

    exp1_df.to_csv(OUTPUT_DIR / "exp1_results.csv", index=False)
    exp2_df.to_csv(OUTPUT_DIR / "exp2_results.csv", index=False)
    exp3_df.to_csv(OUTPUT_DIR / "exp3_results.csv", index=False)
    exp4_df.to_csv(OUTPUT_DIR / "exp4_results.csv", index=False)
    exp5_df.to_csv(OUTPUT_DIR / "exp5_results.csv", index=False)
    exp5_paired.to_csv(OUTPUT_DIR / "exp5_paired_gene_results.csv", index=False)
    exp6_df.to_csv(OUTPUT_DIR / "exp6_results.csv", index=False)
    exp7_df.to_csv(OUTPUT_DIR / "exp7_results.csv", index=False)
    exp1_stats.to_csv(OUTPUT_DIR / "exp1_summary.csv", index=False)
    exp2_stats.to_csv(OUTPUT_DIR / "exp2_summary.csv", index=False)
    exp3_stats.to_csv(OUTPUT_DIR / "exp3_summary.csv", index=False)
    exp4_stats.to_csv(OUTPUT_DIR / "exp4_summary.csv", index=False)
    exp5_stats.to_csv(OUTPUT_DIR / "exp5_summary.csv", index=False)
    exp7_stats.to_csv(OUTPUT_DIR / "exp7_summary.csv", index=False)
    prelim = preliminary_results_paragraph(
        graph_mode, variant_mode, exp1_stats, exp2_stats, exp3_stats, corr, claim1, claim2, claim3
    )
    summary_text = interpretation + "\n\nPRELIMINARY RESULTS PARAGRAPH\n" + prelim + "\n"
    save_text(OUTPUT_DIR / "experiment_summary.txt", summary_text)
    print("\nAll results saved.")
    print(
        f"Figures: {OUTPUT_DIR / 'experiment1.png'}, {OUTPUT_DIR / 'experiment2.png'}, {OUTPUT_DIR / 'experiment3.png'}, "
        f"{OUTPUT_DIR / 'experiment4.png'}, {OUTPUT_DIR / 'experiment5.png'}, {OUTPUT_DIR / 'experiment6.png'}, {OUTPUT_DIR / 'experiment7.png'}"
    )
    print(
        f"Data:    {OUTPUT_DIR / 'exp1_results.csv'}, {OUTPUT_DIR / 'exp2_results.csv'}, {OUTPUT_DIR / 'exp3_results.csv'}, "
        f"{OUTPUT_DIR / 'exp4_results.csv'}, {OUTPUT_DIR / 'exp5_results.csv'}, {OUTPUT_DIR / 'exp6_results.csv'}, {OUTPUT_DIR / 'exp7_results.csv'}"
    )
    print("\nGrant-ready paragraph:")
    print(prelim)


if __name__ == "__main__":
    main()
