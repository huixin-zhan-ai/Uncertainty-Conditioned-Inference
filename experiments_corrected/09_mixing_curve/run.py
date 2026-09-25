#!/usr/bin/env python3
"""E3 -- Mixing curve for the Hastings-corrected sampler.

Question: how quickly does the implemented independence-MH kernel reach its
exact target, and does the paper's step schedule suffice?

Design: for every valid locked variant we run ONE long set of independent
chains (2048 chains, up to max(2x schedule, 100) steps) and record the
empirical group distribution at checkpoints. Chains are Markov, so the state
at step s of the long run IS the state of an s-step run: checkpoints are exact.
Chain-count sensitivity uses the first C of the 2048 independent chains at the
paper's schedule.

Kernel: identical to corrected_core.metropolis_sample (same proposal mixture,
same Hastings-corrected acceptance). A self-test asserts that the paper's
protocol (schedule steps, 512 chains) reproduces diffusion_diagnostics.csv
within Monte-Carlo noise. Inputs are the frozen signals table and frozen
scorer. Outputs are NEW tables; nothing frozen is overwritten.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))

import corrected_core as core  # noqa: E402

TABLES = ROOT / "outputs" / "corrected" / "tables"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

STEP_CHECKPOINTS = [1, 2, 5, 10, 25, 50, 100]
CHAIN_COUNTS = [32, 128, 512, 2048]
N_CHAINS_MAX = 2048


def stable_seed(variant_id: str) -> int:
    digest = hashlib.sha256(str(variant_id).encode("utf-8")).hexdigest()
    return core.RNG_SEED + int(digest[:8], 16)


def legacy_steps(uncertainty: float) -> int:
    return 10 + int(round(15 * float(np.clip(uncertainty, 0.0, 1.0))))


def run_chains_with_checkpoints(utilities, uncertainty, groups, checkpoints, seed):
    utilities = np.asarray(utilities, dtype=float)
    tau = core.theta(uncertainty)
    proposal_probability = core.proposal_probabilities(utilities, uncertainty)
    log_proposal = np.log(np.clip(proposal_probability, 1e-300, 1.0))
    rng = np.random.default_rng(seed)
    current = np.zeros(N_CHAINS_MAX, dtype=int)  # deterministic shortest path
    groups = np.asarray(groups)
    exact = core.aggregate_distribution(
        list(groups), core.exact_path_probabilities(utilities, uncertainty)
    )
    max_steps = max(checkpoints)
    accepted = np.zeros(N_CHAINS_MAX, dtype=int)
    records = []
    for step in range(1, max_steps + 1):
        proposal = rng.choice(len(utilities), size=N_CHAINS_MAX, p=proposal_probability)
        log_acceptance = (
            (utilities[proposal] - utilities[current]) / tau
            + log_proposal[current]
            - log_proposal[proposal]
        )
        accept = (log_acceptance >= 0) | (np.log(rng.random(N_CHAINS_MAX)) < log_acceptance)
        current[accept] = proposal[accept]
        accepted += accept
        if step in checkpoints:
            for n_chains in CHAIN_COUNTS:
                sub = current[:n_chains]
                empirical = core.aggregate_distribution(list(groups[sub]), np.ones(n_chains) / n_chains)
                records.append(
                    {
                        "steps": step,
                        "n_chains": n_chains,
                        "js_empirical_to_exact": core.js_divergence(empirical, exact),
                        "acceptance_rate": float(accepted[:n_chains].sum() / (n_chains * step)),
                        "unique_final_paths": int(len(set(sub.tolist()))),
                    }
                )
    return records


def main():
    t0 = time.time()
    graph, _, _ = core.load_graph_and_raw_variants()
    relation_frequency = core.relation_frequencies(graph)
    scorer = core.LearnedPathScorer.load()
    variants = pd.read_csv(TABLES / "variants_locked_signals.csv")
    variants["variant_id"] = variants["variant_id"].astype(str)
    diag = pd.read_csv(TABLES / "diffusion_diagnostics.csv")
    diag["variant_id"] = diag["variant_id"].astype(str)
    variants = variants[variants["variant_id"].isin(set(diag["variant_id"]))].reset_index(drop=True)
    print(f"loaded graph + {len(variants)} valid variants in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for i, row in enumerate(variants.itertuples(index=False)):
        candidates = core.enumerate_candidate_paths(graph, row.gene_id)
        if not candidates:
            continue
        features, groups = core.feature_matrix(
            graph, candidates, relation_frequency, row.pathogenicity_prob, row.isotonic_entropy
        )
        utilities = scorer.utilities(features)
        schedule = core.revision_steps(row.isotonic_entropy, len(utilities))
        legacy = legacy_steps(row.isotonic_entropy)
        checkpoints = sorted(set(STEP_CHECKPOINTS + [legacy, schedule, 2 * schedule]))
        recs = run_chains_with_checkpoints(
            utilities, row.isotonic_entropy, groups, checkpoints, stable_seed(row.variant_id) + 7
        )
        for r in recs:
            kind = "fixed"
            if r["steps"] == schedule:
                kind = "schedule"
            elif r["steps"] == 2 * schedule:
                kind = "2x_schedule"
            elif r["steps"] == legacy:
                kind = "legacy"
            r.update(
                variant_id=row.variant_id, gene_symbol=row.gene_symbol,
                isotonic_entropy=row.isotonic_entropy, n_states=len(utilities),
                schedule_steps=schedule, legacy_steps=legacy, checkpoint_kind=kind,
            )
            rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(variants)}  {time.time()-t0:.0f}s", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "mixing_curve.csv", index=False)
    frame.to_csv(TABLES / "mixing_curve.csv", index=False)

    # self-test against the paper's protocol (schedule steps, 512 chains)
    paper_like = frame[(frame.checkpoint_kind == "schedule") & (frame.n_chains == 512)]
    merged = paper_like.merge(diag[["variant_id", "js_empirical_to_exact"]], on="variant_id", suffixes=("", "_paper"))
    check = {"n": int(len(merged)),
             "mean_js_here": float(merged.js_empirical_to_exact.mean()),
             "mean_js_paper": float(merged.js_empirical_to_exact_paper.mean())}
    check["abs_diff_of_means"] = abs(check["mean_js_here"] - check["mean_js_paper"])
    assert check["abs_diff_of_means"] < 5e-4, check

    p95 = lambda s: float(np.percentile(s, 95))  # noqa: E731
    fixed = frame[frame.n_chains == 512]
    by_steps = (fixed[fixed.checkpoint_kind == "fixed"].groupby("steps")
                .agg(js_mean=("js_empirical_to_exact", "mean"), js_p95=("js_empirical_to_exact", p95),
                     acceptance_mean=("acceptance_rate", "mean"), n=("variant_id", "size")).reset_index())
    by_kind = (fixed[fixed.checkpoint_kind != "fixed"].groupby("checkpoint_kind")
               .agg(steps_mean=("steps", "mean"), js_mean=("js_empirical_to_exact", "mean"),
                    js_p95=("js_empirical_to_exact", p95), n=("variant_id", "size")).reset_index())
    sched = frame[frame.checkpoint_kind == "schedule"]
    by_chains = (sched.groupby("n_chains")
                 .agg(js_mean=("js_empirical_to_exact", "mean"), js_p95=("js_empirical_to_exact", p95),
                      n=("variant_id", "size")).reset_index())
    strat = sched[sched.n_chains == 512].copy()
    strat["uncertainty_stratum"] = pd.qcut(strat.isotonic_entropy, q=4,
                                           labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")
    by_stratum = (strat.groupby("uncertainty_stratum", observed=False)
                  .agg(js_mean=("js_empirical_to_exact", "mean"), steps_mean=("steps", "mean"),
                       n=("variant_id", "size")).reset_index())
    for name, tbl in [("mixing_curve_by_steps", by_steps), ("mixing_curve_by_schedule", by_kind),
                      ("mixing_curve_by_chains", by_chains), ("mixing_curve_by_stratum", by_stratum)]:
        tbl.to_csv(TABLES / f"{name}.csv", index=False)
    summary = {"self_test": check, "by_steps": by_steps.to_dict("records"),
               "by_schedule_kind": by_kind.to_dict("records"), "by_chains": by_chains.to_dict("records"),
               "by_stratum": by_stratum.to_dict("records"), "runtime_s": time.time() - t0}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
