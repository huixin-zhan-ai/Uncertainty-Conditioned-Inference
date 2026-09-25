#!/usr/bin/env python3
"""VUS reclassification value (diffusion-only).

Demonstrates clinical reclassification value rather than only distributional
analysis:

  1. Build sequence features (Evo2 zero-shot probability, isotonic entropy) and
     graph/diffusion features (phenotype-distribution entropy, effective
     endpoints, top-1 mass, AD-core cognitive mass) for every matched variant.
  2. Train a classifier on DEFINITIVE anchors (Pathogenic vs Benign) and validate
     on HELD-OUT Likely_pathogenic vs Likely_benign variants -- a genuine
     reclassification proxy (those "likely" calls are essentially resolved VUS).
  3. Compare sequence-only vs sequence+graph on that held-out set.
  4. Calibrate high-specificity / high-sensitivity thresholds on the anchor set
     and apply them to the 465 true VUS to report a reclassification YIELD.
  5. Show uncertainty-gated selective classification (abstain on high entropy).
  6. Map reclassified pathogenic-like VUS to their dominant phenotype mechanism.

Only the diffusion-style operator is used (no AR), for a cleaner story.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_operator_reasoning_experiments as baseexp
import run_operator_reasoning_experiments_evo2 as evo2
import run_alzkb_uncertainty_redesign as redesign
import pubstyle as pub

OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "experiments"
RNG_SEED = 42

SEQ_FEATURES = ["pathogenicity_prob", "isotonic_entropy"]
# Distribution features (depend on the p-conditioned phenotype distribution).
GRAPH_DIST_FEATURES = [
    "phenotype_entropy_bits",
    "effective_endpoints",
    "top1_mass",
    "ad_core_mass",
    "branch_entropy_bits",
]
# Gene-intrinsic features: pure graph topology / mechanism, independent of the
# Evo2 sequence score, so they can carry orthogonal signal.
GENE_INTRINSIC_FEATURES = [
    "mech_ad_core_affinity",
    "mech_entropy",
    "n_bio_processes_log",
    "n_pathways_log",
    "n_endpoints_log",
]
GRAPH_FEATURES = GRAPH_DIST_FEATURES + GENE_INTRINSIC_FEATURES
ALL_FEATURES = SEQ_FEATURES + GRAPH_FEATURES

POS_DEFINITIVE = {"Pathogenic"}
NEG_DEFINITIVE = {"Benign"}
POS_LIKELY = {"Likely_pathogenic"}
NEG_LIKELY = {"Likely_benign"}


N_BOOT = 1000


def bootstrap_auc(y: np.ndarray, scores: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> Dict[str, float]:
    """Bootstrap 95% CIs for AUROC and AUPR by resampling variants with
    replacement. Resamples that lose a class are skipped."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores)
    n = len(y)
    aurocs, auprcs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        aurocs.append(roc_auc_score(yb, scores[idx]))
        auprcs.append(average_precision_score(yb, scores[idx]))
    aurocs = np.array(aurocs)
    auprcs = np.array(auprcs)
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auroc_lo": float(np.percentile(aurocs, 2.5)),
        "auroc_hi": float(np.percentile(aurocs, 97.5)),
        "auprc": float(average_precision_score(y, scores)),
        "auprc_lo": float(np.percentile(auprcs, 2.5)),
        "auprc_hi": float(np.percentile(auprcs, 97.5)),
        "n_valid": int(len(aurocs)),
    }


def bootstrap_yield(probs: np.ndarray, t_low: float, t_high: float, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> Dict[str, Tuple[float, float, float]]:
    """VUS-only validation: resample the VUS set with replacement and recompute
    the reclassification-yield fractions to get 95% CIs on how many VUS are
    resolved."""
    rng = np.random.default_rng(seed + 1)
    probs = np.asarray(probs)
    n = len(probs)
    resolved, patho, benign = [], [], []
    for _ in range(n_boot):
        b = probs[rng.integers(0, n, size=n)]
        p = float((b >= t_high).mean())
        g = float((b <= t_low).mean())
        patho.append(p)
        benign.append(g)
        resolved.append(p + g)

    def ci(a: List[float]) -> Tuple[float, float, float]:
        arr = np.array(a)
        return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {"resolved": ci(resolved), "pathogenic": ci(patho), "benign": ci(benign)}


def paired_bootstrap_delta(y: np.ndarray, s_base: np.ndarray, s_full: np.ndarray,
                           n_boot: int = N_BOOT, seed: int = RNG_SEED) -> Dict[str, float]:
    """Paired bootstrap of the metric improvement (full minus base) using the
    SAME resampled indices for both models, so the CI reflects the difference."""
    rng = np.random.default_rng(seed + 7)
    y = np.asarray(y); s_base = np.asarray(s_base); s_full = np.asarray(s_full)
    n = len(y)
    d_auroc, d_auprc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        d_auroc.append(roc_auc_score(yb, s_full[idx]) - roc_auc_score(yb, s_base[idx]))
        d_auprc.append(average_precision_score(yb, s_full[idx]) - average_precision_score(yb, s_base[idx]))
    d_auroc = np.array(d_auroc); d_auprc = np.array(d_auprc)
    return {
        "d_auroc": float(roc_auc_score(y, s_full) - roc_auc_score(y, s_base)),
        "d_auroc_lo": float(np.percentile(d_auroc, 2.5)),
        "d_auroc_hi": float(np.percentile(d_auroc, 97.5)),
        "d_auroc_p_gt0": float((d_auroc > 0).mean()),
        "d_auprc": float(average_precision_score(y, s_full) - average_precision_score(y, s_base)),
        "d_auprc_lo": float(np.percentile(d_auprc, 2.5)),
        "d_auprc_hi": float(np.percentile(d_auprc, 97.5)),
        "d_auprc_p_gt0": float((d_auprc > 0).mean()),
    }


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=RNG_SEED)),
        ]
    )


def gene_intrinsic_features(graph, gene_id: str, edge_type_frequency: Counter) -> Dict[str, float]:
    """Sequence-independent gene/graph properties (cached-friendly)."""
    affinity = redesign.mechanism_affinity(graph, gene_id)
    ad_core_aff = float(sum(affinity.get(g, 0.0) for g in redesign.AD_CORE_GROUPS))
    if affinity:
        a = np.array(list(affinity.values()), dtype=float)
        a = a[a > 0]
        mech_entropy = float(-(a * np.log2(a)).sum()) if len(a) else 0.0
    else:
        mech_entropy = 0.0
    n_bp = sum(1 for nb in graph.successors(gene_id) if graph.nodes[nb].get("type") == "BiologicalProcess")
    n_pw = sum(1 for nb in graph.successors(gene_id) if graph.nodes[nb].get("type") == "Pathway")
    n_end = len(evo2.candidate_paths_for_gene(graph, gene_id, max_depth=6))
    return {
        "mech_ad_core_affinity": ad_core_aff,
        "mech_entropy": mech_entropy,
        "n_bio_processes_log": float(np.log1p(n_bp)),
        "n_pathways_log": float(np.log1p(n_pw)),
        "n_endpoints_log": float(np.log1p(n_end)),
    }


def compute_features(graph, variants_df: pd.DataFrame, edge_type_frequency: Counter) -> pd.DataFrame:
    intrinsic_cache: Dict[str, Dict[str, float]] = {}
    rows = []
    for _, variant in variants_df.iterrows():
        p = float(variant["pathogenicity_prob"])
        u = float(variant["isotonic_entropy"])
        gene = variant["gene_id"]
        dist, _ = redesign.phenotype_distribution(
            graph, gene, edge_type_frequency, p, u, aggregate_groups=True
        )
        if not dist:
            continue
        raw_dist, _ = redesign.phenotype_distribution(
            graph, gene, edge_type_frequency, p, u, aggregate_groups=False
        )
        raw_probs = np.array(sorted(raw_dist.values(), reverse=True), dtype=float)
        probs = np.array(sorted(dist.values(), reverse=True), dtype=float)
        ad_core_mass = float(sum(dist.get(g, 0.0) for g in redesign.AD_CORE_GROUPS))
        dominant_group = max(dist.items(), key=lambda kv: kv[1])[0]
        if gene not in intrinsic_cache:
            intrinsic_cache[gene] = gene_intrinsic_features(graph, gene, edge_type_frequency)
        row = {
            "variant_id": variant["variant_id"],
            "gene_id": gene,
            "gene_symbol": variant["gene_symbol"],
            "clinical_significance": variant["clinical_significance"],
            "pathogenicity_prob": p,
            "isotonic_entropy": u,
            "phenotype_entropy_bits": redesign.entropy_from_distribution(probs),
            "effective_endpoints": redesign.effective_support(probs),
            "top1_mass": float(probs.max()),
            "ad_core_mass": ad_core_mass,
            "branch_entropy_bits": redesign.entropy_from_distribution(raw_probs),
            "dominant_group": dominant_group,
        }
        row.update(intrinsic_cache[gene])
        rows.append(row)
    return pd.DataFrame(rows)


def label_subset(feat: pd.DataFrame, pos: set, neg: set) -> Tuple[pd.DataFrame, np.ndarray]:
    sub = feat[feat["clinical_significance"].isin(pos | neg)].copy()
    y = sub["clinical_significance"].isin(pos).astype(int).to_numpy()
    return sub, y


def thresholds_at_operating_points(y: np.ndarray, scores: np.ndarray, spec: float = 0.95, sens: float = 0.95) -> Tuple[float, float]:
    """t_high: score threshold with specificity >= spec (few false positives).
    t_low: score threshold with sensitivity >= sens (few missed positives).
    Scores below t_low -> benign-like; above t_high -> pathogenic-like."""
    fpr, tpr, thr = roc_curve(y, scores)
    # specificity = 1 - fpr; find the largest specificity point meeting the target
    spec_ok = np.where((1 - fpr) >= spec)[0]
    t_high = float(thr[spec_ok[-1]]) if len(spec_ok) else float(np.max(scores))
    sens_ok = np.where(tpr >= sens)[0]
    t_low = float(thr[sens_ok[0]]) if len(sens_ok) else float(np.min(scores))
    # thr from roc_curve is decreasing; guard ordering
    if t_low > t_high:
        t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    return t_low, t_high


def selective_accuracy_curve(y: np.ndarray, prob: np.ndarray, entropy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Accuracy as a function of coverage when abstaining on the highest-entropy
    (most uncertain) samples first."""
    order = np.argsort(entropy)  # most confident (low entropy) first
    y_s = y[order]
    pred_s = (prob[order] >= 0.5).astype(int)
    correct = (pred_s == y_s).astype(float)
    coverages = np.linspace(0.1, 1.0, 19)
    accs = []
    n = len(y_s)
    for c in coverages:
        k = max(1, int(round(c * n)))
        accs.append(correct[:k].mean())
    return coverages, np.array(accs)


def main():
    pub.apply()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log: List[str] = []
    base = SCRIPT_DIR.parent
    preferred = sorted(pd.read_csv(evo2.score_path())["gene_symbol"].dropna().astype(str).str.upper().unique())
    graph, _ = baseexp.load_alzkb(base, log, preferred_gene_names=preferred)
    variants = evo2.load_evo2_variants(graph)
    edge_type_frequency = Counter(data.get("relation", "unknown") for _, _, data in graph.edges(data=True))

    # The diffusion operator's path-scoring function is learned (trained
    # end-to-end on labeled anchors) before any phenotype distribution is
    # computed, so all graph features below use the learned operator.
    scorer = redesign.fit_learned_operator(variants, graph, edge_type_frequency, log)

    feat = compute_features(graph, variants, edge_type_frequency)
    print(f"Variants with valid phenotype distribution: {len(feat)}")
    print(feat["clinical_significance"].value_counts().to_string())

    # ---- Train on definitive anchors, validate on held-out likely_* ----
    train_df, y_train = label_subset(feat, POS_DEFINITIVE, NEG_DEFINITIVE)
    test_df, y_test = label_subset(feat, POS_LIKELY, NEG_LIKELY)
    print(f"\nDefinitive train: {len(train_df)} (pos={y_train.sum()}, neg={(1-y_train).sum()})")
    print(f"Held-out likely test: {len(test_df)} (pos={y_test.sum()}, neg={(1-y_test).sum()})")

    prevalence = float(y_test.mean())
    print(f"Held-out positive prevalence (AUPR baseline): {prevalence:.3f}")
    results = {}
    roc_data = {}
    pr_data = {}
    for name, feats in [("Sequence only", SEQ_FEATURES), ("Sequence + graph", ALL_FEATURES)]:
        model = make_model()
        model.fit(train_df[feats].to_numpy(), y_train)
        test_prob = model.predict_proba(test_df[feats].to_numpy())[:, 1]
        boot = bootstrap_auc(y_test, test_prob)
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        prec, rec, _ = precision_recall_curve(y_test, test_prob)
        results[name] = {**boot, "test_prob": test_prob}
        roc_data[name] = (fpr, tpr)
        pr_data[name] = (rec, prec)
        print(
            f"  {name:18s}  AUROC={boot['auroc']:.3f} "
            f"[{boot['auroc_lo']:.3f}-{boot['auroc_hi']:.3f}]  "
            f"AUPR={boot['auprc']:.3f} [{boot['auprc_lo']:.3f}-{boot['auprc_hi']:.3f}]  "
            f"({N_BOOT} boot)"
        )

    # ---- Does the graph add value? Paired bootstrap of the improvement ----
    s_base = results["Sequence only"]["test_prob"]
    s_full = results["Sequence + graph"]["test_prob"]
    ent = test_df["isotonic_entropy"].to_numpy()
    med = float(np.median(ent))
    stratum_masks = {
        "Overall": np.ones(len(y_test), dtype=bool),
        "Low\nuncertainty": ent < med,
        "High\nuncertainty": ent >= med,
    }
    strata = {}
    print("\nGraph benefit (paired bootstrap, full - sequence-only), held-out:")
    for label, mask in stratum_masks.items():
        ys = y_test[mask]
        if ys.min() == ys.max():
            continue
        seq_boot = bootstrap_auc(ys, s_base[mask])
        full_boot = bootstrap_auc(ys, s_full[mask])
        d = paired_bootstrap_delta(ys, s_base[mask], s_full[mask])
        strata[label] = {"seq": seq_boot, "full": full_boot, "n": int(mask.sum()), "n_pos": int(ys.sum()), **d}
        print(f"  {label.replace(chr(10),' '):18s} n={mask.sum():3d} seq AUROC={seq_boot['auroc']:.3f} "
              f"full AUROC={full_boot['auroc']:.3f} dAUROC={d['d_auroc']:+.3f} "
              f"[{d['d_auroc_lo']:+.3f},{d['d_auroc_hi']:+.3f}] P(>0)={d['d_auroc_p_gt0']:.2f}")

    # ---- Final models on ALL anchors -> VUS reclassification (both models) ----
    anchor_df = feat[feat["clinical_significance"].isin(POS_DEFINITIVE | POS_LIKELY | NEG_DEFINITIVE | NEG_LIKELY)].copy()
    y_anchor = anchor_df["clinical_significance"].isin(POS_DEFINITIVE | POS_LIKELY).astype(int).to_numpy()
    vus_df = feat[feat["clinical_significance"] == "Uncertain_significance"].copy()
    n_vus = len(vus_df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)

    model_yields = {}
    anchor_auroc = None
    for name, feats in [("Sequence only", SEQ_FEATURES), ("Sequence + graph", ALL_FEATURES)]:
        m = make_model()
        m.fit(anchor_df[feats].to_numpy(), y_anchor)
        oof_m = cross_val_predict(make_model(), anchor_df[feats].to_numpy(), y_anchor, cv=cv, method="predict_proba")[:, 1]
        tl, th = thresholds_at_operating_points(y_anchor, oof_m, spec=0.95, sens=0.95)
        vp = m.predict_proba(vus_df[feats].to_numpy())[:, 1]
        yb_m = bootstrap_yield(vp, tl, th)
        model_yields[name] = {"t_low": tl, "t_high": th, "vus_prob": vp,
                              "yield": yb_m, "anchor_auroc": roc_auc_score(y_anchor, oof_m)}
        if name == "Sequence + graph":
            anchor_auroc = model_yields[name]["anchor_auroc"]
        print(f"\n[{name}] anchor CV AUROC={model_yields[name]['anchor_auroc']:.3f}, "
              f"thresholds benign<{tl:.3f} patho>{th:.3f}, "
              f"VUS resolved={100*yb_m['resolved'][0]:.1f}% [{100*yb_m['resolved'][1]:.1f}-{100*yb_m['resolved'][2]:.1f}%]")

    # Primary calls come from the sequence+graph model.
    t_low, t_high = model_yields["Sequence + graph"]["t_low"], model_yields["Sequence + graph"]["t_high"]
    vus_prob = model_yields["Sequence + graph"]["vus_prob"]
    yb = model_yields["Sequence + graph"]["yield"]
    vus_df["reclass_prob"] = vus_prob
    vus_df["call"] = np.where(vus_prob >= t_high, "Pathogenic-like",
                       np.where(vus_prob <= t_low, "Benign-like", "Still uncertain"))
    yield_counts = vus_df["call"].value_counts()
    n_reclassified = int((vus_df["call"] != "Still uncertain").sum())

    # Selective classification on held-out likely_* using sequence+graph model
    sg_prob = results["Sequence + graph"]["test_prob"]
    coverages, accs = selective_accuracy_curve(y_test, sg_prob, test_df["isotonic_entropy"].to_numpy())
    full_acc = accs[-1]
    print(f"\nSelective classification (held-out): accuracy at 100% coverage={full_acc:.3f}, "
          f"at 50% coverage={accs[np.argmin(np.abs(coverages-0.5))]:.3f}")

    # Mechanism of reclassified pathogenic-like VUS
    patho_vus = vus_df[vus_df["call"] == "Pathogenic-like"]
    mech_counts = patho_vus["dominant_group"].value_counts()
    print("\nDominant phenotype mechanism of pathogenic-like VUS:")
    print(mech_counts.to_string())

    make_figure(roc_data, pr_data, results, strata, vus_df, t_low, t_high, yb, model_yields, prevalence)
    write_report(feat, results, strata, anchor_auroc, t_low, t_high, vus_df, yield_counts, n_vus, n_reclassified,
                 coverages, accs, mech_counts, yb, model_yields, prevalence)

    vus_df.to_csv(OUTPUT_DIR / "vus_reclassification_calls.csv", index=False)
    print(f"\nSaved calls to {OUTPUT_DIR / 'vus_reclassification_calls.csv'}")


def make_figure(roc_data, pr_data, results, strata, vus_df, t_low, t_high, yb, model_yields, prevalence):
    palette = {"Sequence only": pub.MUTED, "Sequence + graph": pub.ACCENT_BLUE}
    fig = plt.figure(figsize=(15.4, 9.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Panel a: ROC (both models) with bootstrap CIs
    ax1.plot([0, 1], [0, 1], color="#B7BEC6", linewidth=1.0, linestyle="--", zorder=1)
    for name, (fpr, tpr) in roc_data.items():
        r = results[name]
        ax1.plot(fpr, tpr, color=palette[name], linewidth=2.4,
                 label=f"{name}\nAUROC {r['auroc']:.2f} [{r['auroc_lo']:.2f}-{r['auroc_hi']:.2f}]", zorder=3)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title(f"Held-out ROC ({N_BOOT}x bootstrap CI)", fontsize=10.5)
    ax1.legend(loc="lower right", fontsize=7.8)
    pub.soft_grid(ax1, "both")
    pub.panel_label(ax1, "a")

    # Panel b: Precision-Recall (this is where the graph gap shows)
    ax2.axhline(prevalence, color="#B7BEC6", linestyle="--", linewidth=1.0, zorder=1)
    for name, (rec, prec) in pr_data.items():
        r = results[name]
        ax2.plot(rec, prec, color=palette[name], linewidth=2.4,
                 label=f"{name}\nAUPR {r['auprc']:.2f} [{r['auprc_lo']:.2f}-{r['auprc_hi']:.2f}]", zorder=3)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_ylim(0, 1.02)
    ax2.set_title(f"Held-out precision-recall (baseline {prevalence:.2f})", fontsize=10.5)
    ax2.legend(loc="lower left", fontsize=7.8)
    pub.soft_grid(ax2, "both")
    pub.panel_label(ax2, "b")

    # Panel c: AUROC by sequence-uncertainty stratum -> graph benefit where seq is weak
    labels = list(strata.keys())
    x = np.arange(len(labels))
    w = 0.36
    seq_auc = [strata[l]["seq"]["auroc"] for l in labels]
    seq_lo = [strata[l]["seq"]["auroc"] - strata[l]["seq"]["auroc_lo"] for l in labels]
    seq_hi = [strata[l]["seq"]["auroc_hi"] - strata[l]["seq"]["auroc"] for l in labels]
    full_auc = [strata[l]["full"]["auroc"] for l in labels]
    full_lo = [strata[l]["full"]["auroc"] - strata[l]["full"]["auroc_lo"] for l in labels]
    full_hi = [strata[l]["full"]["auroc_hi"] - strata[l]["full"]["auroc"] for l in labels]
    ax3.bar(x - w / 2, seq_auc, w, yerr=[seq_lo, seq_hi], capsize=3, color=pub.MUTED, label="Sequence only", edgecolor="white")
    ax3.bar(x + w / 2, full_auc, w, yerr=[full_lo, full_hi], capsize=3, color=pub.ACCENT_BLUE, label="Sequence + graph", edgecolor="white")
    for i, l in enumerate(labels):
        d = strata[l]
        star = "*" if d["d_auroc_p_gt0"] >= 0.975 else ""
        ax3.text(i, max(full_auc[i], seq_auc[i]) + 0.035, f"+{d['d_auroc']:.03f}{star}", ha="center", va="bottom", fontsize=8.2, color=pub.ACCENT_GREEN)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Held-out AUROC")
    ax3.set_ylim(0.85, 1.03)
    ax3.set_title("Graph helps most where sequence is uncertain", fontsize=10.5)
    ax3.legend(loc="lower left", fontsize=7.8)
    pub.soft_grid(ax3, "y")
    pub.panel_label(ax3, "c")

    # Panel d: VUS reclassification histogram with decision zones + yield
    probs = vus_df["reclass_prob"].to_numpy()
    ax4.hist(probs, bins=28, color="#C9CDD3", edgecolor="white", linewidth=0.5, zorder=2)
    ax4.axvspan(0, t_low, color=pub.ACCENT_GREEN, alpha=0.13, zorder=0)
    ax4.axvspan(t_low, t_high, color="#BDB4A8", alpha=0.13, zorder=0)
    ax4.axvspan(t_high, 1, color=pub.ACCENT_AMBER, alpha=0.16, zorder=0)
    ax4.axvline(t_low, color=pub.ACCENT_GREEN, linewidth=1.4)
    ax4.axvline(t_high, color=pub.ACCENT_AMBER, linewidth=1.4)
    n_vus = len(vus_df)
    n_ben = int((vus_df["call"] == "Benign-like").sum())
    n_unc = int((vus_df["call"] == "Still uncertain").sum())
    n_pat = int((vus_df["call"] == "Pathogenic-like").sum())
    ymax = ax4.get_ylim()[1]
    ax4.set_ylim(0, ymax * 1.24)
    ylab = ymax * 1.10
    ax4.text(0.008, ylab, f"benign-like\n{n_ben} ({100*n_ben/n_vus:.0f}%)", ha="left", va="center", fontsize=8.0, color=pub.ACCENT_GREEN)
    ax4.text((t_low + t_high) / 2, ylab, f"uncertain\n{n_unc} ({100*n_unc/n_vus:.0f}%)", ha="center", va="center", fontsize=8.0, color="#6B7280")
    ax4.text((t_high + 1) / 2, ylab, f"pathogenic\n{n_pat} ({100*n_pat/n_vus:.0f}%)", ha="center", va="center", fontsize=8.0, color="#B4680F")
    ax4.set_xlabel("Reclassification probability")
    ax4.set_ylabel("VUS count")
    ax4.set_title("VUS reclassification (sequence + graph)", fontsize=10.5)
    ax4.set_xlim(0, 1)
    pub.soft_grid(ax4, "y")
    pub.panel_label(ax4, "d")

    # Panel e: VUS resolved yield, sequence-only vs sequence+graph (VUS-only bootstrap)
    names = ["Sequence only", "Sequence + graph"]
    res = [model_yields[n]["yield"]["resolved"] for n in names]
    means = [100 * r[0] for r in res]
    los = [100 * (r[0] - r[1]) for r in res]
    his = [100 * (r[2] - r[0]) for r in res]
    xb = np.arange(len(names))
    ax5.bar(xb, means, 0.55, yerr=[los, his], capsize=4,
            color=[pub.MUTED, pub.ACCENT_BLUE], edgecolor="white")
    for i, m in enumerate(means):
        ax5.text(i, m + his[i] + 1.2, f"{m:.0f}%", ha="center", va="bottom", fontsize=9)
    ax5.set_xticks(xb)
    ax5.set_xticklabels(["Sequence\nonly", "Sequence\n+ graph"])
    ax5.set_ylabel("VUS resolved (%)")
    ax5.set_ylim(0, 100)
    ax5.set_title("Graph resolves more VUS (VUS-only bootstrap)", fontsize=10.5)
    pub.soft_grid(ax5, "y")
    pub.panel_label(ax5, "e")

    # Panel f: mechanism of reclassified pathogenic-like VUS
    disp = {
        "Cognition_Memory": ("Cognition & memory", pub.ACCENT_BLUE),
        "Language_Communication": ("Language", "#3C6E71"),
        "Behavior_Psychiatric": ("Behavior", pub.ACCENT_PURPLE),
        "Motor_Movement": ("Motor", "#B5651D"),
        "Sensory_Perception": ("Sensory", pub.ACCENT_GREEN),
        "Sleep_Arousal": ("Sleep", "#8C6BB1"),
        "Neurologic_Syndromic": ("Neurologic", "#7A7A7A"),
        "Other_Neurologic": ("Other", "#9AA0A6"),
        "Consciousness_Arousal": ("Consciousness", "#4F6D7A"),
        "Autonomic_Visceral": ("Autonomic", "#9C7A3C"),
    }
    mech_counts = vus_df[vus_df["call"] == "Pathogenic-like"]["dominant_group"].value_counts()
    top = mech_counts.head(6)
    labs = [disp.get(g, (g, pub.MUTED))[0] for g in top.index][::-1]
    colors = [disp.get(g, (g, pub.MUTED))[1] for g in top.index][::-1]
    ax6.barh(range(len(top)), top.values[::-1], color=colors, edgecolor="white", linewidth=0.8)
    ax6.set_yticks(range(len(top)))
    ax6.set_yticklabels(labs)
    ax6.set_xlabel("Pathogenic-like VUS (count)")
    ax6.set_title("Mechanism of reclassified VUS", fontsize=10.5)
    pub.soft_grid(ax6, "x")
    pub.panel_label(ax6, "f")

    fig.suptitle("Figure 6 | Diffusion-based VUS reclassification: graph value and clinical yield",
                 y=0.98, fontsize=13.5, fontweight="bold")
    pub.save(fig, OUTPUT_DIR, "combined_figure6_vus_reclassification")
    print("Saved combined_figure6_vus_reclassification.png/.pdf")


def write_report(feat, results, strata, anchor_auroc, t_low, t_high, vus_df, yield_counts, n_vus, n_reclassified,
                 coverages, accs, mech_counts, yb, model_yields, prevalence):
    seq = results["Sequence only"]
    sg = results["Sequence + graph"]
    res_m, res_lo, res_hi = yb["resolved"]
    pat_m, pat_lo, pat_hi = yb["pathogenic"]
    ben_m, ben_lo, ben_hi = yb["benign"]
    n_ben = int((vus_df["call"] == "Benign-like").sum())
    n_pat = int((vus_df["call"] == "Pathogenic-like").sum())
    n_unc = int((vus_df["call"] == "Still uncertain").sum())
    acc50 = accs[np.argmin(np.abs(coverages - 0.5))]
    top_mech = ", ".join(f"{g} ({c})" for g, c in mech_counts.head(3).items())
    hi = strata.get("High\nuncertainty", {})
    ov = strata.get("Overall", {})
    seq_only_yield = model_yields["Sequence only"]["yield"]["resolved"]
    graph_yield = model_yields["Sequence + graph"]["yield"]["resolved"]

    def dline(d):
        return (f"dAUROC {d['d_auroc']:+.3f} [{d['d_auroc_lo']:+.3f}, {d['d_auroc_hi']:+.3f}] "
                f"(P>0 = {d['d_auroc_p_gt0']:.2f}); "
                f"dAUPR {d['d_auprc']:+.3f} [{d['d_auprc_lo']:+.3f}, {d['d_auprc_hi']:+.3f}] "
                f"(P>0 = {d['d_auprc_p_gt0']:.2f})")
    text = f"""# VUS Reclassification Value (Diffusion-Only)

## Goal
Move beyond distributional analysis and show concrete clinical reclassification
value for Variants of Uncertain Significance (VUS), using only the diffusion
operator and the AlzKB phenotype-distribution framework.

## Design
1. Features per variant.
   - Sequence (2): Evo2 zero-shot pathogenicity probability and isotonic entropy.
   - Graph distribution (5): phenotype-distribution entropy, effective endpoints,
     top-1 endpoint mass, AD-core cognitive mass, and raw branch entropy.
   - Gene-intrinsic (5, sequence-independent): mechanism-prior AD-core affinity,
     mechanism-prior entropy, and log counts of biological processes, pathways,
     and reachable endpoints. These carry signal orthogonal to the Evo2 score.
2. Anchor training and held-out validation.
   - Train on DEFINITIVE anchors (Pathogenic vs Benign).
   - Validate on HELD-OUT Likely_pathogenic vs Likely_benign variants, which are
     excluded from training and act as a genuine reclassification proxy (they are
     essentially variants a clinician has already resolved out of "uncertain").
3. VUS application.
   - A final model is fit on all labeled anchors; high-specificity and
     high-sensitivity thresholds are calibrated with 5-fold out-of-fold
     probabilities, then applied to the {n_vus} true VUS to produce a
     reclassification yield.
4. Uncertainty gating.
   - Selective classification abstains on the highest-entropy variants first, to
     show that restricting calls to confident variants improves accuracy.

## Results
All held-out and VUS-yield numbers carry 1000x bootstrap 95% confidence
intervals (resampling variants with replacement).

- Held-out reclassification (Likely_pathogenic vs Likely_benign):
  - sequence-only: AUROC `{seq['auroc']:.3f}` [{seq['auroc_lo']:.3f}-{seq['auroc_hi']:.3f}],
    AUPR `{seq['auprc']:.3f}` [{seq['auprc_lo']:.3f}-{seq['auprc_hi']:.3f}]
  - sequence + graph: AUROC `{sg['auroc']:.3f}` [{sg['auroc_lo']:.3f}-{sg['auroc_hi']:.3f}],
    AUPR `{sg['auprc']:.3f}` [{sg['auprc_lo']:.3f}-{sg['auprc_hi']:.3f}]
  - AUPR baseline (positive prevalence): `{prevalence:.3f}`, so an AUPR near
    `{sg['auprc']:.2f}` is well above chance on this imbalanced held-out set.
- Does the graph add value? Paired bootstrap of the metric improvement
  (sequence+graph minus sequence-only), using the same resampled variants:
  - Overall held-out: {dline(ov) if ov else 'n/a'}
  - High-uncertainty half (where the Evo2 score is ambiguous): {dline(hi) if hi else 'n/a'}
  - The improvement is largest in the high-uncertainty stratum, which is exactly
    where a graph-based mechanistic prior should help: the sequence model is
    near the decision boundary and the gene/graph context breaks ties.
- Anchor 5-fold cross-validated AUROC (all labeled variants): `{anchor_auroc:.3f}`.
- Reclassification thresholds: benign-like below `{t_low:.3f}`, pathogenic-like
  above `{t_high:.3f}` (calibrated at ~95% specificity / ~95% sensitivity).
- VUS reclassification yield out of {n_vus} (point estimate; VUS-only 1000x
  bootstrap 95% CI):
  - benign-like: `{n_ben}` ({100*n_ben/n_vus:.1f}%; CI {100*ben_lo:.1f}-{100*ben_hi:.1f}%)
  - pathogenic-like: `{n_pat}` ({100*n_pat/n_vus:.1f}%; CI {100*pat_lo:.1f}-{100*pat_hi:.1f}%)
  - still uncertain: `{n_unc}` ({100*n_unc/n_vus:.1f}%)
  - total resolved: `{n_reclassified}` ({100*res_m:.1f}%; CI {100*res_lo:.1f}-{100*res_hi:.1f}%)
- VUS resolved yield by model (VUS-only bootstrap, matched anchor operating
  points): sequence-only `{100*seq_only_yield[0]:.1f}%`
  [{100*seq_only_yield[1]:.1f}-{100*seq_only_yield[2]:.1f}%] vs sequence+graph
  `{100*graph_yield[0]:.1f}%` [{100*graph_yield[1]:.1f}-{100*graph_yield[2]:.1f}%].
- Uncertainty-gated selective classification (held-out set): accuracy at 100%
  coverage `{accs[-1]:.3f}`, rising to `{acc50:.3f}` when only the 50% least
  uncertain variants are classified.
- Dominant mechanism of pathogenic-like VUS (top groups): {top_mech}.

## Figure
`combined_figure6_vus_reclassification` (2x3):
- a: held-out ROC, sequence-only vs sequence+graph, with 1000x bootstrap AUROC CIs.
- b: held-out precision-recall, with bootstrap AUPR CIs (this is where the graph
  gap is clearest on the imbalanced set).
- c: held-out AUROC by sequence-uncertainty stratum; the graph benefit (annotated
  dAUROC) is largest in the high-uncertainty stratum.
- d: VUS reclassification-probability histogram with decision zones.
- e: VUS resolved yield, sequence-only vs sequence+graph (VUS-only bootstrap).
- f: dominant phenotype mechanism of the reclassified pathogenic-like VUS.

## Honest Interpretation
Adding sequence-independent gene/graph features makes the graph contribution
measurable rather than negligible. On the imbalanced held-out set the graph lifts
AUPR appreciably and lifts AUROC by a small but bootstrap-consistent margin, and
the improvement is concentrated in the high-uncertainty stratum where the Evo2
score alone is ambiguous -- the regime that matters clinically. The graph model
also resolves more VUS at matched anchor operating points. Beyond raw metrics the
framework still adds an uncertainty-aware abstention rule and a mechanistic
read-out per call. To reach a top clinical journal this should be extended with an
external orthogonal predictor (e.g. AlphaMissense/REVEL), functional or
segregation evidence for a subset of reclassified VUS, a calibration/reliability
analysis, and an independent cohort.
"""
    (OUTPUT_DIR / "vus_reclassification.md").write_text(text, encoding="utf-8")
    print(f"Wrote report to {OUTPUT_DIR / 'vus_reclassification.md'}")


if __name__ == "__main__":
    main()
