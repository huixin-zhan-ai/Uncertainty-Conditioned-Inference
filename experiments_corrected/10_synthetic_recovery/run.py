#!/usr/bin/env python3
"""E4 -- Synthetic energy-recovery (learning) experiment.

The paper's synthetic study validates the SAMPLER only. This experiment asks
whether the end-to-end objective (label loss through the uncertainty-
conditioned path softmax, exactly corrected_core.train_path_scorer) recovers a
hidden planted energy rule.

Generator (per query q): K candidate paths with 14-dim standardized features
phi (feature index 6 is a binary 'core-outcome' indicator c, matching the real
model's is_ad_core slot; other dims N(0,1)); each path is assigned one of 10
outcome groups; uncertainty H_q drawn from five levels; theta(H) is the paper's
schedule; the planted target pi* = softmax(w*.phi/theta); core mass
m* = sum_i c_i pi*_i; label y ~ Bernoulli(sigmoid(alpha* m* + beta*)).

Recovery metrics on held-out queries: cosine(w_hat, w*); mean Spearman rank
agreement between learned and planted utilities; mean group-level JS between
learned and planted targets; held-out label AUROC of the learned readout
versus the oracle readout that uses m*. Because the loss observes w only
through the core mass, directions of w that do not move m* are weakly
identified; we therefore report target-level agreement (JS, rank) alongside
the raw weight cosine. Conditions: K in {53,256,1024}, N_train in
{121,500,2000}, 5 replicates. Outputs are NEW tables only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
import corrected_core as core  # noqa: E402

TABLES = ROOT / "outputs" / "corrected" / "tables"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

D = core.N_FEATURES
CORE_IDX = core.FEATURE_NAMES.index("is_ad_core")
H_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]
STATE_SIZES = [53, 256, 1024]
TRAIN_SIZES = [121, 500, 2000]
N_TEST = 500
N_GROUPS = 10
REPLICATES = 5
ALPHA_STAR, BETA_STAR = 6.0, -2.0


def make_query(rng, K, w_star):
    phi = rng.normal(size=(K, D))
    c = (rng.random(K) < 0.3).astype(float)
    phi[:, CORE_IDX] = c
    groups = rng.integers(0, N_GROUPS, K)
    H = float(rng.choice(H_LEVELS))
    tau = core.theta(H)
    u = phi @ w_star
    pi = np.exp((u - u.max()) / tau); pi /= pi.sum()
    m = float(c @ pi)
    y = float(rng.random() < expit(ALPHA_STAR * m + BETA_STAR))
    return dict(features=phi, ad_core=c, theta=tau, label=y, groups=groups, H=H, u_star=u, pi_star=pi, m_star=m)


def evaluate(scorer, test):
    cos = float(scorer.weights @ W_STAR / (np.linalg.norm(scorer.weights) * np.linalg.norm(W_STAR)))
    ranks, jss, yhat, yor, ys = [], [], [], [], []
    for q in test:
        u_hat = scorer.utilities(q["features"])
        pi_hat = core.exact_path_probabilities(u_hat, q["H"])
        g = [str(x) for x in q["groups"]]
        jss.append(core.js_divergence(core.aggregate_distribution(g, pi_hat), core.aggregate_distribution(g, q["pi_star"])))
        ranks.append(spearmanr(u_hat, q["u_star"]).correlation)
        m_hat = float(q["ad_core"] @ pi_hat)
        yhat.append(expit(scorer.alpha * m_hat + scorer.beta)); yor.append(expit(ALPHA_STAR * q["m_star"] + BETA_STAR)); ys.append(q["label"])
    ys = np.array(ys)
    return dict(weight_cosine=cos, utility_rank_rho=float(np.nanmean(ranks)), target_group_js=float(np.mean(jss)),
                auroc_learned=float(roc_auc_score(ys, yhat)), auroc_oracle=float(roc_auc_score(ys, yor)))


def main():
    global W_STAR
    t0 = time.time()
    rows = []
    for rep in range(REPLICATES):
        rng = np.random.default_rng(core.RNG_SEED + 100 * rep)
        W_STAR = np.clip(rng.normal(size=D), -3, 3)
        for K in STATE_SIZES:
            test = [make_query(rng, K, W_STAR) for _ in range(N_TEST)]
            for N in TRAIN_SIZES:
                train = [make_query(rng, K, W_STAR) for _ in range(N)]
                samples = [dict(variant_id=f"s{i}", features=q["features"], ad_core=q["ad_core"], theta=q["theta"], label=q["label"]) for i, q in enumerate(train)]
                t1 = time.time()
                scorer = core.train_path_scorer(samples, l2=1e-2, seed=core.RNG_SEED + rep)
                res = evaluate(scorer, test)
                res.update(replicate=rep, n_states=K, n_train=N, train_positive_rate=float(np.mean([q["label"] for q in train])),
                           fit_seconds=time.time() - t1)
                rows.append(res)
                print(f"rep{rep} K={K} N={N}: cos={res['weight_cosine']:.3f} rank={res['utility_rank_rho']:.3f} "
                      f"JS={res['target_group_js']:.4f} AUROC={res['auroc_learned']:.3f}/{res['auroc_oracle']:.3f}  "
                      f"({res['fit_seconds']:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "synthetic_recovery.csv", index=False); frame.to_csv(OUT / "synthetic_recovery.csv", index=False)
    summ = frame.groupby(["n_states", "n_train"]).agg(
        weight_cosine=("weight_cosine", "mean"), weight_cosine_sd=("weight_cosine", "std"),
        utility_rank_rho=("utility_rank_rho", "mean"), utility_rank_rho_sd=("utility_rank_rho", "std"),
        target_group_js=("target_group_js", "mean"), target_group_js_sd=("target_group_js", "std"),
        auroc_learned=("auroc_learned", "mean"), auroc_oracle=("auroc_oracle", "mean"), n=("replicate", "size")).reset_index()
    summ.to_csv(TABLES / "synthetic_recovery_summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(dict(summary=summ.to_dict("records"), runtime_s=time.time() - t0), indent=2))
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
