#!/usr/bin/env python3
"""E4b -- Optimizer-reliability diagnostic for the end-to-end energy learner.

E4 showed that corrected_core.train_path_scorer sometimes converges to a
collapsed solution (weight cosine to the planted rule < 0.8), most often at
small N (121, the real study's size) and large K. Here we (1) measure the
training objective reached by single-seed fits versus best-of-R restarts on
the failing synthetic conditions, and (2) evaluate the FROZEN real scorer's
training objective against best-of-R restarts on the real definitive samples.
The frozen model is NOT modified; if a restart finds a materially lower
objective, that is reported as a decision for the authors.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
import corrected_core as core  # noqa: E402
sys.path.insert(0, str(HERE.parent / "10_synthetic_recovery"))
import run as synth  # noqa: E402  (E4 generator + evaluator)

TABLES = ROOT / "outputs" / "corrected" / "tables"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RESTARTS = 8


def objective(scorer, samples, l2=1e-2, readout_l2=1e-3):
    loss = 0.0
    for s in samples:
        u = scorer.utilities(s["features"])
        z = (u - u.max()) / s["theta"]; p = np.exp(z); p /= p.sum()
        m = float(s["ad_core"] @ p)
        pred = float(np.clip(expit(scorer.alpha * m + scorer.beta), 1e-9, 1 - 1e-9))
        loss += -(s["label"] * np.log(pred) + (1 - s["label"]) * np.log(1 - pred))
    return loss / len(samples) + 0.5 * l2 * float(scorer.weights @ scorer.weights) + 0.5 * readout_l2 * scorer.alpha ** 2


def best_of_restarts(samples, seeds):
    fits = []
    for sd in seeds:
        try:
            sc = core.train_path_scorer(samples, l2=1e-2, seed=sd)
            fits.append((objective(sc, samples), sd, sc))
        except RuntimeError as e:  # optimizer failure is itself a data point
            fits.append((np.inf, sd, None))
    fits.sort(key=lambda t: t[0])
    return fits


def main():
    t0 = time.time(); rows = []
    # (1) synthetic failing conditions
    for K, N in [(256, 121), (1024, 121), (1024, 2000)]:
        for rep in range(3):
            rng = np.random.default_rng(core.RNG_SEED + 100 * rep)
            synth.W_STAR = np.clip(rng.normal(size=synth.D), -3, 3)
            test = [synth.make_query(rng, K, synth.W_STAR) for _ in range(300)]
            train = [synth.make_query(rng, K, synth.W_STAR) for _ in range(N)]
            samples = [dict(variant_id=f"s{i}", features=q["features"], ad_core=q["ad_core"], theta=q["theta"], label=q["label"]) for i, q in enumerate(train)]
            fits = best_of_restarts(samples, [core.RNG_SEED + rep + 1000 * r for r in range(RESTARTS)])
            single = [f for f in fits if f[1] == core.RNG_SEED + rep][0]
            best = fits[0]
            e_single = synth.evaluate(single[2], test) if single[2] else {}
            e_best = synth.evaluate(best[2], test)
            objs = [f[0] for f in fits if np.isfinite(f[0])]
            rows.append(dict(setting="synthetic", n_states=K, n_train=N, replicate=rep,
                             single_seed_objective=single[0], best_objective=best[0],
                             n_restarts=RESTARTS, n_restarts_failed=int(sum(~np.isfinite([f[0] for f in fits]))),
                             objective_spread=float(max(objs) - min(objs)),
                             single_cosine=e_single.get("weight_cosine", np.nan), best_cosine=e_best["weight_cosine"],
                             single_js=e_single.get("target_group_js", np.nan), best_js=e_best["target_group_js"],
                             single_auroc=e_single.get("auroc_learned", np.nan), best_auroc=e_best["auroc_learned"]))
            print(rows[-1], flush=True)
    # (2) frozen real scorer vs restarts on the real definitive samples
    graph, _, _ = core.load_graph_and_raw_variants()
    rf = core.relation_frequencies(graph)
    frozen = core.LearnedPathScorer.load()
    sig = pd.read_csv(TABLES / "variants_locked_signals.csv"); sig["variant_id"] = sig["variant_id"].astype(str)
    samples = core.build_training_samples(graph, sig, rf, frozen.training_ids)
    obj_frozen = objective(frozen, samples)
    fits = best_of_restarts(samples, [core.RNG_SEED + 1000 * r for r in range(RESTARTS)])
    best = fits[0]
    cos_to_frozen = float(best[2].weights @ frozen.weights / (np.linalg.norm(best[2].weights) * np.linalg.norm(frozen.weights)))
    real = dict(setting="real_frozen_vs_restarts", n_train=len(samples), frozen_objective=obj_frozen,
                best_restart_objective=best[0], objectives_all=[f[0] for f in fits],
                best_restart_cosine_to_frozen=cos_to_frozen, frozen_alpha=frozen.alpha, best_alpha=best[2].alpha)
    print(real, flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "synthetic_recovery_restarts.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(dict(synthetic=rows, real=real, runtime_s=time.time() - t0), indent=2, default=str))
    (TABLES / "frozen_scorer_restart_check.json").write_text(json.dumps(real, indent=2, default=str))


if __name__ == "__main__":
    main()
