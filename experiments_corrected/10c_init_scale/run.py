#!/usr/bin/env python3
"""E4c -- Escaping the w~0 plateau: initialization-scale sweep.

Identical objective, regularization, optimizer and bounds to
corrected_core.train_path_scorer; the ONLY change is the scale of the random
initialization of w (core uses 0.01). Evaluated on the E4 generator at the
failing conditions and at the non-failing ones as a control. This tests a
concrete fix; the frozen real model is not modified.
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
INIT_SCALES = [0.01, 0.3, 1.0]


def train_with_init(samples, init_scale, l2=1e-2, seed=core.RNG_SEED):
    """Verbatim port of corrected_core.train_path_scorer with init_scale exposed."""
    pooled = np.vstack([s["features"] for s in samples]); mean = pooled.mean(0); std = pooled.std(0); std[std < 1e-8] = 1.0
    prepared = [{**s, "features": (s["features"] - mean) / std} for s in samples]
    dim = core.N_FEATURES; readout_l2 = 1e-3

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

    rng = np.random.default_rng(seed)
    x0 = np.concatenate([rng.normal(0, init_scale, dim), [1.0, 0.0]])
    res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                   bounds=[(-5.0, 5.0)] * dim + [(-20.0, 20.0), (-20.0, 20.0)],
                   options={"maxiter": 1000, "ftol": 1e-10, "maxls": 50})
    sc = core.LearnedPathScorer(weights=res.x[:dim], feature_mean=mean, feature_std=std, alpha=float(res.x[dim]),
                                beta=float(res.x[dim + 1]), l2=l2, n_train=len(prepared), training_ids=[s["variant_id"] for s in prepared])
    return sc, float(res.fun), int(res.nit), res.success


def main():
    t0 = time.time(); rows = []
    for K, N in [(53, 121), (256, 121), (1024, 121), (1024, 2000)]:
        for rep in range(5):
            rng = np.random.default_rng(core.RNG_SEED + 100 * rep)
            synth.W_STAR = np.clip(rng.normal(size=synth.D), -3, 3)
            test = [synth.make_query(rng, K, synth.W_STAR) for _ in range(300)]
            train = [synth.make_query(rng, K, synth.W_STAR) for _ in range(N)]
            samples = [dict(variant_id=f"s{i}", features=q["features"], ad_core=q["ad_core"], theta=q["theta"], label=q["label"]) for i, q in enumerate(train)]
            for scale in INIT_SCALES:
                sc, obj, nit, ok = train_with_init(samples, scale, seed=core.RNG_SEED + rep)
                e = synth.evaluate(sc, test)
                rows.append(dict(n_states=K, n_train=N, replicate=rep, init_scale=scale, objective=obj, iterations=nit, success=bool(ok), **e))
            print(f"K={K} N={N} rep{rep}: " + " | ".join(f"init{r['init_scale']}: obj={r['objective']:.3f} it={r['iterations']} cos={r['weight_cosine']:.2f}" for r in rows[-3:]), flush=True)
    f = pd.DataFrame(rows); f.to_csv(TABLES / "synthetic_recovery_init_scale.csv", index=False)
    s = f.groupby(["n_states", "n_train", "init_scale"]).agg(cos_mean=("weight_cosine", "mean"), cos_min=("weight_cosine", "min"),
        js_mean=("target_group_js", "mean"), auroc=("auroc_learned", "mean"), auroc_oracle=("auroc_oracle", "mean"),
        fail_rate=("weight_cosine", lambda c: float((c < 0.8).mean())), iters=("iterations", "mean")).reset_index()
    s.to_csv(TABLES / "synthetic_recovery_init_scale_summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(dict(summary=s.to_dict("records"), runtime_s=time.time() - t0), indent=2))
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
