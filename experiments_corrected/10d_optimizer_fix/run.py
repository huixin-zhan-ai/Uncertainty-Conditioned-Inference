#!/usr/bin/env python3
"""E4d -- Optimizer-termination fix for the plateau at w~0.

Hypothesis: the end-to-end objective is nearly flat at initialization for
large candidate sets (the informative gradient scales like 1/K), so L-BFGS-B
terminates on its default projected-gradient tolerance (gtol=1e-5) before
leaving the plateau. Variants, all with the verbatim objective:
  default   : core settings (init 0.01, gtol 1e-5, maxiter 1000)
  tight     : gtol 1e-12, maxiter 5000 (keep iterating on the plateau)
  informed  : initialize w along the first-order direction
              sum_q (y_q - ybar) * (phibar_core,q - phibar_noncore,q),
              then optimize with tight tolerances
Frozen real model untouched; also reports the real model's objective under
'tight' to confirm it is unchanged.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
import corrected_core as core  # noqa: E402
sys.path.insert(0, str(HERE.parent / "10_synthetic_recovery"))
import run as synth  # noqa: E402

TABLES = ROOT / "outputs" / "corrected" / "tables"
OUT = HERE / "outputs"; OUT.mkdir(parents=True, exist_ok=True)


def make_objective(prepared, l2=1e-2, readout_l2=1e-3):
    dim = core.N_FEATURES

    def objective(theta_):
        w = theta_[:dim]; alpha, beta = theta_[dim:]
        loss = 0.0; gw = np.zeros(dim); ga = 0.0; gb = 0.0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for s in prepared:
                phi = s["features"]; c = s["ad_core"]; tau = s["theta"]; y = s["label"]
                u = phi @ w
                if not np.all(np.isfinite(u)):
                    return 1e100, np.zeros_like(theta_)
                p = np.exp((u - u.max()) / tau); p /= p.sum()
                m = float(c @ p); pred = float(np.clip(expit(alpha * m + beta), 1e-9, 1 - 1e-9))
                loss += -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
                gz = pred - y; ga += gz * m; gb += gz
                gw += phi.T @ (gz * alpha * p * (c - m) / tau)
        n = len(prepared)
        return (loss / n + 0.5 * l2 * float(w @ w) + 0.5 * readout_l2 * alpha * alpha,
                np.concatenate([gw / n + l2 * w, [ga / n + readout_l2 * alpha, gb / n]]))
    return objective


def train_variant(samples, variant, seed=core.RNG_SEED):
    pooled = np.vstack([s["features"] for s in samples]); mean = pooled.mean(0); std = pooled.std(0); std[std < 1e-8] = 1.0
    prepared = [{**s, "features": (s["features"] - mean) / std} for s in samples]
    dim = core.N_FEATURES
    rng = np.random.default_rng(seed)
    w0 = rng.normal(0, 0.01, dim)
    if variant == "informed":
        ybar = np.mean([s["label"] for s in prepared]); d = np.zeros(dim)
        for s in prepared:
            c = s["ad_core"].astype(bool)
            if c.any() and (~c).any():
                d += (s["label"] - ybar) * (s["features"][c].mean(0) - s["features"][~c].mean(0))
        if np.linalg.norm(d) > 0:
            w0 = 0.5 * d / np.linalg.norm(d)
    opts = {"maxiter": 1000, "ftol": 1e-10, "maxls": 50} if variant == "default" else {"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-12, "maxls": 50}
    res = minimize(make_objective(prepared), np.concatenate([w0, [1.0, 0.0]]), jac=True, method="L-BFGS-B",
                   bounds=[(-5.0, 5.0)] * dim + [(-20.0, 20.0), (-20.0, 20.0)], options=opts)
    sc = core.LearnedPathScorer(weights=res.x[:dim], feature_mean=mean, feature_std=std, alpha=float(res.x[dim]), beta=float(res.x[dim + 1]),
                                l2=1e-2, n_train=len(prepared), training_ids=[s["variant_id"] for s in prepared])
    return sc, float(res.fun), int(res.nit)


def main():
    t0 = time.time(); rows = []
    for K, N in [(53, 121), (256, 121), (1024, 121), (1024, 500), (1024, 2000)]:
        for rep in range(5):
            rng = np.random.default_rng(core.RNG_SEED + 100 * rep)
            synth.W_STAR = np.clip(rng.normal(size=synth.D), -3, 3)
            test = [synth.make_query(rng, K, synth.W_STAR) for _ in range(300)]
            train = [synth.make_query(rng, K, synth.W_STAR) for _ in range(N)]
            samples = [dict(variant_id=f"s{i}", features=q["features"], ad_core=q["ad_core"], theta=q["theta"], label=q["label"]) for i, q in enumerate(train)]
            for variant in ["default", "tight", "informed"]:
                t1 = time.time(); sc, obj, nit = train_variant(samples, variant, seed=core.RNG_SEED + rep)
                e = synth.evaluate(sc, test)
                rows.append(dict(n_states=K, n_train=N, replicate=rep, variant=variant, objective=obj, iterations=nit, seconds=time.time() - t1, **e))
            print(f"K={K} N={N} rep{rep}: " + " | ".join(f"{r['variant']}: obj={r['objective']:.3f} it={r['iterations']} cos={r['weight_cosine']:.2f}" for r in rows[-3:]) + f"  ({time.time()-t0:.0f}s)", flush=True)
    f = pd.DataFrame(rows); f.to_csv(TABLES / "synthetic_recovery_optimizer.csv", index=False)
    s = f.groupby(["n_states", "n_train", "variant"]).agg(cos_mean=("weight_cosine", "mean"), cos_min=("weight_cosine", "min"),
        js_mean=("target_group_js", "mean"), auroc=("auroc_learned", "mean"), auroc_oracle=("auroc_oracle", "mean"),
        fail_rate=("weight_cosine", lambda c: float((c < 0.8).mean())), iters=("iterations", "mean"), objective=("objective", "mean")).reset_index()
    s.to_csv(TABLES / "synthetic_recovery_optimizer_summary.csv", index=False)
    # real frozen model under tight tolerances (should be unchanged)
    graph, _, _ = core.load_graph_and_raw_variants(); rf = core.relation_frequencies(graph); frozen = core.LearnedPathScorer.load()
    sig = pd.read_csv(TABLES / "variants_locked_signals.csv"); sig["variant_id"] = sig["variant_id"].astype(str)
    samples = core.build_training_samples(graph, sig, rf, frozen.training_ids)
    sc_t, obj_t, nit_t = train_variant(samples, "tight")
    real = dict(tight_objective=obj_t, tight_iterations=nit_t, cosine_to_frozen=float(sc_t.weights @ frozen.weights / (np.linalg.norm(sc_t.weights) * np.linalg.norm(frozen.weights))))
    (TABLES / "frozen_scorer_tight_check.json").write_text(json.dumps(real, indent=2))
    (OUT / "summary.json").write_text(json.dumps(dict(summary=s.to_dict("records"), real=real, runtime_s=time.time() - t0), indent=2))
    print(s.round(3).to_string(index=False)); print("REAL:", real)


if __name__ == "__main__":
    main()
