#!/usr/bin/env python3
"""Learned scoring function for the diffusion-style reasoning operator.

The diffusion operator turns a variant into a distribution over reachable
phenotype endpoints by placing a softmax over candidate reasoning paths:

    a_i = softmax_i( U(path_i) / tau(H) ),   p(group g) = sum_{i in g} a_i

Historically ``U(path_i)`` was a hand-tuned weighted sum of interpretable path
terms (structural score, endpoint specificity, a gene-specific mechanism prior,
an AD-core pathogenicity pull, a crowding penalty, ...). This module replaces
those fixed weights with a *learned* linear scoring function

    U(path_i) = w . phi(path_i)

whose weights ``w`` (together with a small readout head alpha, beta) are trained
end-to-end by backpropagating a variant-level pathogenicity loss through the
differentiable path softmax. The exploration temperature ``tau(H)`` stays a
function of the sequence entropy (it is the operator's stochastic schedule, not
part of the scored utility).

Supervision. For each labeled anchor variant v (pathogenic-like = 1,
benign-like = 0) the operator induces an AD-core cognitive mass
``m_v = sum_{i: endpoint(i) in AD-core} a_i``. A logistic readout
``sigma(alpha * m_v + beta)`` predicts the anchor label, and the binary
cross-entropy is minimised over ``(w, alpha, beta)``. The scoring function is
therefore learned to route probability mass so that the operator's own output
distribution is diagnostic, rather than being told which terms matter by hand.

This file is deliberately pure math (numpy + scipy). Feature construction lives
in ``run_alzkb_uncertainty_redesign.py`` so that this module has no dependency
on the graph helpers and there is no import cycle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

# Interpretable per-path features. Order is load-bearing: the saved weight
# vector is aligned to this list.
FEATURE_NAMES: List[str] = [
    "struct_score",          # normalized structural path score (edge-type rarity)
    "path_len",              # normalized path length
    "has_pathway",           # path traverses a Pathway node
    "has_bioprocess",        # path traverses a BiologicalProcess node
    "disease_coverage",      # fraction of Disease nodes on the path
    "endpoint_specificity",  # specific (vs generic) endpoint name bonus
    "is_ad_core",            # endpoint group is a core AD cognitive group
    "mech_affinity",         # gene-specific mechanism prior for this group
    "generic_group",         # endpoint group is syndromic/other (generic)
    "crowding",              # log count of candidates sharing this group
    "p_x_ad_core",           # pathogenicity prob x is_ad_core
    "p_x_mech",              # pathogenicity prob x mechanism affinity
    "conf_x_spec",           # (1 - entropy) x endpoint specificity
    "conf_x_mech",           # (1 - entropy) x mechanism affinity
]
N_FEATURES = len(FEATURE_NAMES)

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "models"
    / "learned_operator_weights.json"
)


@dataclass
class LearnedScorer:
    """Trained parameters of the learned path-scoring function."""

    weights: np.ndarray          # (d,) learned linear utility weights
    feature_mean: np.ndarray     # (d,) standardization mean
    feature_std: np.ndarray      # (d,) standardization std
    alpha: float                 # readout slope on AD-core mass
    beta: float                  # readout intercept
    l2: float                    # L2 penalty used in training
    train_auroc: float           # anchor AUROC of the readout at convergence
    n_train: int                 # number of anchor variants used

    def standardize(self, phi: np.ndarray) -> np.ndarray:
        return (phi - self.feature_mean) / self.feature_std

    def utilities(self, phi: np.ndarray) -> np.ndarray:
        """Path utilities U_i = w . standardized(phi_i)."""
        if phi.size == 0:
            return np.zeros(phi.shape[0], dtype=float)
        return self.standardize(phi) @ self.weights

    def to_dict(self) -> Dict:
        return {
            "feature_names": FEATURE_NAMES,
            "weights": self.weights.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "l2": float(self.l2),
            "train_auroc": float(self.train_auroc),
            "n_train": int(self.n_train),
        }

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path is not None else DEFAULT_WEIGHTS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: Dict) -> "LearnedScorer":
        return cls(
            weights=np.asarray(payload["weights"], dtype=float),
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_std=np.asarray(payload["feature_std"], dtype=float),
            alpha=float(payload["alpha"]),
            beta=float(payload["beta"]),
            l2=float(payload.get("l2", 0.0)),
            train_auroc=float(payload.get("train_auroc", float("nan"))),
            n_train=int(payload.get("n_train", 0)),
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LearnedScorer":
        path = Path(path) if path is not None else DEFAULT_WEIGHTS_PATH
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train(
    samples: List[Dict],
    l2: float = 1e-2,
    seed: int = 42,
) -> LearnedScorer:
    """Fit the learned scoring function by backprop through the path softmax.

    ``samples`` is a list, one entry per labeled anchor variant, each with:
      - ``phi``: (n_i, d) raw path feature matrix (unstandardized)
      - ``ad_core``: (n_i,) indicator of AD-core cognitive endpoints
      - ``tau``: scalar softmax temperature for that variant
      - ``y``: 0/1 anchor label
    """
    if not samples:
        raise ValueError("No training samples for the learned operator")

    # Pooled standardization across all candidate paths.
    pooled = np.vstack([s["phi"] for s in samples])
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std[std < 1e-6] = 1.0

    prepared = []
    for s in samples:
        phi = (s["phi"] - mean) / std
        prepared.append(
            {
                "phi": phi,
                "c": np.asarray(s["ad_core"], dtype=float),
                "tau": float(max(s["tau"], 1e-6)),
                "y": float(s["y"]),
            }
        )

    d = N_FEATURES
    n = len(prepared)

    def unpack(theta):
        return theta[:d], theta[d], theta[d + 1]

    def objective(theta):
        w, alpha, beta = unpack(theta)
        loss = 0.0
        grad_w = np.zeros(d)
        grad_alpha = 0.0
        grad_beta = 0.0
        for s in prepared:
            phi = s["phi"]
            c = s["c"]
            tau = s["tau"]
            y = s["y"]
            u = phi @ w
            u = u - u.max()
            e = np.exp(u / tau)
            a = e / e.sum()
            m = float((c * a).sum())
            z = alpha * m + beta
            yhat = _sigmoid(np.array(z))
            yhat = float(np.clip(yhat, 1e-9, 1 - 1e-9))
            loss += -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
            grad_z = yhat - y
            grad_alpha += grad_z * m
            grad_beta += grad_z
            # dm/dU_j = a_j (c_j - m) / tau  (softmax jacobian contracted with c)
            g = grad_z * alpha * (a * (c - m)) / tau
            grad_w += phi.T @ g
        loss /= n
        grad_w /= n
        grad_alpha /= n
        grad_beta /= n
        # L2 on utility weights only.
        loss += 0.5 * l2 * float(w @ w)
        grad_w += l2 * w
        return loss, np.concatenate([grad_w, [grad_alpha, grad_beta]])

    rng = np.random.default_rng(seed)
    theta0 = np.concatenate([rng.normal(0, 0.01, d), [1.0, 0.0]])
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-9})
    w, alpha, beta = unpack(res.x)

    # Anchor AUROC of the readout at the learned parameters.
    ms, ys = [], []
    for s in prepared:
        u = s["phi"] @ w
        u = u - u.max()
        e = np.exp(u / s["tau"])
        a = e / e.sum()
        ms.append(float((s["c"] * a).sum()))
        ys.append(s["y"])
    ms = np.array(ms)
    ys = np.array(ys)
    try:
        auroc = float(roc_auc_score(ys, alpha * ms + beta))
    except ValueError:
        auroc = float("nan")

    return LearnedScorer(
        weights=w,
        feature_mean=mean,
        feature_std=std,
        alpha=float(alpha),
        beta=float(beta),
        l2=float(l2),
        train_auroc=auroc,
        n_train=n,
    )
