#!/usr/bin/env python3
"""End-to-end leakage-free held-out validation and exploratory VUS calls."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import corrected_core as core
import pubstyle as pub

OUT = core.ROOT / "outputs" / "corrected"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
MANIFESTS = OUT / "manifests"
for directory in (TABLES, FIGURES, REPORTS, MANIFESTS):
    directory.mkdir(parents=True, exist_ok=True)

SEQ_FEATURES = ["pathogenicity_prob", "isotonic_entropy"]
GRAPH_FEATURES = [
    "phenotype_entropy_bits",
    "effective_groups",
    "top1_mass",
    "ad_core_mass",
    "raw_endpoint_entropy_bits",
    "mechanism_ad_core_affinity",
    "mechanism_entropy",
    "log_biological_processes",
    "log_pathways",
    "log_candidate_routes",
]
ALL_FEATURES = SEQ_FEATURES + GRAPH_FEATURES
N_BOOT = 1000


def classifier():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0, max_iter=3000, random_state=core.RNG_SEED
                ),
            ),
        ]
    )


def label(frame):
    return frame["clinical_significance"].isin(core.DEFINITIVE_POS | core.LIKELY_POS).astype(int).to_numpy()


def intrinsic_features(graph, gene_id):
    affinity = core.gene_mechanism_affinity(graph, gene_id)
    probabilities = np.asarray(list(affinity.values()), dtype=float)
    biological_processes = sum(
        graph.nodes[node].get("type") == "BiologicalProcess"
        for node in graph.successors(gene_id)
    )
    pathways = sum(
        graph.nodes[node].get("type") == "Pathway"
        for node in graph.successors(gene_id)
    )
    paths = core.enumerate_candidate_paths(graph, gene_id)
    routes = {tuple(path[:-2]) for path in paths}
    return {
        "mechanism_ad_core_affinity": float(
            sum(affinity.get(group, 0.0) for group in core.AD_CORE_GROUPS)
        ),
        "mechanism_entropy": core.entropy_from_prob(probabilities),
        "log_biological_processes": float(np.log1p(biological_processes)),
        "log_pathways": float(np.log1p(pathways)),
        "log_candidate_routes": float(np.log1p(len(routes))),
    }


def build_features(graph, variants, scorer, relation_frequency):
    rows = []
    intrinsic_cache = {}
    for row in variants.itertuples(index=False):
        grouped, _ = core.exact_distribution(
            graph, scorer, relation_frequency, row.gene_id,
            row.pathogenicity_prob, row.isotonic_entropy, aggregate_groups=True,
        )
        raw, _ = core.exact_distribution(
            graph, scorer, relation_frequency, row.gene_id,
            row.pathogenicity_prob, row.isotonic_entropy, aggregate_groups=False,
        )
        if not grouped or not raw:
            continue
        if row.gene_id not in intrinsic_cache:
            intrinsic_cache[row.gene_id] = intrinsic_features(graph, row.gene_id)
        probabilities = np.asarray(list(grouped.values()), dtype=float)
        record = {
            "variant_id": str(row.variant_id),
            "gene_id": row.gene_id,
            "gene_symbol": row.gene_symbol,
            "clinical_significance": row.clinical_significance,
            "pathogenicity_prob": float(row.pathogenicity_prob),
            "isotonic_entropy": float(row.isotonic_entropy),
            "phenotype_entropy_bits": core.entropy_from_prob(probabilities),
            "effective_groups": core.effective_support(probabilities),
            "top1_mass": float(probabilities.max()),
            "ad_core_mass": float(
                sum(grouped.get(group, 0.0) for group in core.AD_CORE_GROUPS)
            ),
            "raw_endpoint_entropy_bits": core.entropy_from_prob(raw.values()),
            "dominant_group": core.top_key(grouped),
        }
        record.update(intrinsic_cache[row.gene_id])
        rows.append(record)
    return pd.DataFrame(rows)


def fit_fold_pipeline(graph, raw, relation_frequency, train_ids, evaluation_ids):
    signal_models = core.fit_signal_models(raw, train_ids)
    signals = core.apply_signal_models(raw, signal_models)
    samples = core.build_training_samples(
        graph, signals, relation_frequency, train_ids
    )
    scorer = core.train_path_scorer(samples, seed=core.RNG_SEED)
    needed = set(map(str, train_ids)) | set(map(str, evaluation_ids))
    features = build_features(
        graph,
        signals[signals["variant_id"].astype(str).isin(needed)].copy(),
        scorer,
        relation_frequency,
    )
    return scorer, features


def cross_fitted_predictions(graph, raw, relation_frequency):
    definitive = raw[
        raw["clinical_significance"].isin(core.DEFINITIVE_POS | core.DEFINITIVE_NEG)
    ].copy().reset_index(drop=True)
    y = definitive["clinical_significance"].isin(core.DEFINITIVE_POS).astype(int).to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=core.RNG_SEED)
    predictions = {
        "Sequence only": np.full(len(definitive), np.nan),
        "Sequence + graph": np.full(len(definitive), np.nan),
    }
    fold_manifest = []
    for fold, (train_index, validation_index) in enumerate(folds.split(definitive, y)):
        train_ids = definitive.iloc[train_index]["variant_id"].astype(str).tolist()
        validation_ids = definitive.iloc[validation_index]["variant_id"].astype(str).tolist()
        scorer, features = fit_fold_pipeline(
            graph, raw, relation_frequency, train_ids, validation_ids
        )
        if set(validation_ids) & set(scorer.training_ids):
            raise AssertionError("Outer-fold leakage into path scorer.")
        indexed = features.set_index("variant_id")
        train_frame = indexed.loc[train_ids]
        validation_frame = indexed.loc[validation_ids]
        y_train = train_frame["clinical_significance"].isin(core.DEFINITIVE_POS).astype(int).to_numpy()
        for name, columns in [
            ("Sequence only", SEQ_FEATURES),
            ("Sequence + graph", ALL_FEATURES),
        ]:
            if not np.isfinite(train_frame[columns].to_numpy()).all():
                raise ValueError(f"Non-finite outer-fold training features: {name}")
            if not np.isfinite(validation_frame[columns].to_numpy()).all():
                raise ValueError(f"Non-finite outer-fold validation features: {name}")
            with np.errstate(all="ignore"):
                model = classifier().fit(train_frame[columns], y_train)
                predictions[name][validation_index] = model.predict_proba(
                    validation_frame[columns]
                )[:, 1]
        fold_manifest.append(
            {
                "fold": fold,
                "train_ids": train_ids,
                "validation_ids": validation_ids,
            }
        )
    if any(np.isnan(values).any() for values in predictions.values()):
        raise RuntimeError("Incomplete cross-fitted predictions.")
    (MANIFESTS / "reclassification_outer_folds.json").write_text(
        json.dumps(fold_manifest, indent=2), encoding="utf-8"
    )
    return definitive, y, predictions


def fit_probability_calibrator(probability, y):
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=2000, random_state=core.RNG_SEED)
    model.fit(logit, y)
    return model


def calibrate(model, probability):
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    return model.predict_proba(logit)[:, 1]


def expected_calibration_error(y, probability, bins=10):
    y = np.asarray(y)
    probability = np.asarray(probability)
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (
            probability <= upper if upper == 1 else probability < upper
        )
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(result)


def cluster_resample_indices(clusters, rng):
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(clusters == cluster) for cluster in sampled])


def bootstrap_metrics(y, probability, clusters, seed=core.RNG_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(N_BOOT):
        indices = cluster_resample_indices(clusters, rng)
        sampled = y[indices]
        if np.unique(sampled).size < 2:
            continue
        rows.append(
            (
                roc_auc_score(sampled, probability[indices]),
                average_precision_score(sampled, probability[indices]),
                brier_score_loss(sampled, probability[indices]),
            )
        )
    values = np.asarray(rows)
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auroc_lo": float(np.percentile(values[:, 0], 2.5)),
        "auroc_hi": float(np.percentile(values[:, 0], 97.5)),
        "aupr": float(average_precision_score(y, probability)),
        "aupr_lo": float(np.percentile(values[:, 1], 2.5)),
        "aupr_hi": float(np.percentile(values[:, 1], 97.5)),
        "brier": float(brier_score_loss(y, probability)),
        "brier_lo": float(np.percentile(values[:, 2], 2.5)),
        "brier_hi": float(np.percentile(values[:, 2], 97.5)),
        "ece": expected_calibration_error(y, probability),
    }


def paired_bootstrap(y, base, full, clusters):
    if np.unique(y).size < 2:
        return {
            "estimable": False,
            "delta_auroc": None,
            "delta_auroc_lo": None,
            "delta_auroc_hi": None,
            "delta_aupr": None,
            "delta_aupr_lo": None,
            "delta_aupr_hi": None,
        }
    rng = np.random.default_rng(core.RNG_SEED + 19)
    rows = []
    for _ in range(N_BOOT):
        indices = cluster_resample_indices(clusters, rng)
        sampled = y[indices]
        if np.unique(sampled).size < 2:
            continue
        rows.append(
            (
                roc_auc_score(sampled, full[indices])
                - roc_auc_score(sampled, base[indices]),
                average_precision_score(sampled, full[indices])
                - average_precision_score(sampled, base[indices]),
            )
        )
    if not rows:
        return {
            "estimable": False,
            "delta_auroc": None,
            "delta_auroc_lo": None,
            "delta_auroc_hi": None,
            "delta_aupr": None,
            "delta_aupr_lo": None,
            "delta_aupr_hi": None,
        }
    values = np.asarray(rows)
    return {
        "estimable": True,
        "delta_auroc": float(roc_auc_score(y, full) - roc_auc_score(y, base)),
        "delta_auroc_lo": float(np.percentile(values[:, 0], 2.5)),
        "delta_auroc_hi": float(np.percentile(values[:, 0], 97.5)),
        "delta_aupr": float(
            average_precision_score(y, full) - average_precision_score(y, base)
        ),
        "delta_aupr_lo": float(np.percentile(values[:, 1], 2.5)),
        "delta_aupr_hi": float(np.percentile(values[:, 1], 97.5)),
    }


def bootstrap_yield(probability, low, high, clusters, seed=core.RNG_SEED + 31):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(N_BOOT):
        indices = cluster_resample_indices(clusters, rng)
        sample = probability[indices]
        benign = float((sample <= low).mean())
        pathogenic = float((sample >= high).mean())
        rows.append((benign, pathogenic, benign + pathogenic))
    values = np.asarray(rows)
    names = ["benign_like", "pathogenic_like", "resolved"]
    return {
        name: {
            "mean": float(values[:, index].mean()),
            "lo": float(np.percentile(values[:, index], 2.5)),
            "hi": float(np.percentile(values[:, index], 97.5)),
        }
        for index, name in enumerate(names)
    }


def thresholds(y, probability, specificity=0.95, sensitivity=0.95):
    fpr, tpr, values = roc_curve(y, probability)
    high_candidates = np.where((1 - fpr) >= specificity)[0]
    low_candidates = np.where(tpr >= sensitivity)[0]
    high = float(values[high_candidates[-1]]) if len(high_candidates) else 1.0
    low = float(values[low_candidates[0]]) if len(low_candidates) else 0.0
    return min(low, high), max(low, high)


def selective_curve(y, probability, uncertainty):
    order = np.argsort(uncertainty)
    rows = []
    for coverage in np.linspace(0.1, 1.0, 19):
        count = max(1, int(round(coverage * len(order))))
        selected = order[:count]
        prediction = (probability[selected] >= 0.5).astype(int)
        rows.append(
            {
                "coverage": coverage,
                "accuracy": float((prediction == y[selected]).mean()),
                "n": count,
            }
        )
    return pd.DataFrame(rows)


def make_figure(results, predictions, test_y, test_features, vus, thresholds_by_model, selective):
    pub.apply()
    colors = {"Sequence only": pub.MUTED, "Sequence + graph": pub.ACCENT_BLUE}
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5))
    for name, probability in predictions.items():
        fpr, tpr, _ = roc_curve(test_y, probability)
        precision, recall, _ = precision_recall_curve(test_y, probability)
        axes[0, 0].plot(fpr, tpr, color=colors[name], lw=2, label=f"{name}: {results[name]['auroc']:.2f}")
        axes[0, 1].plot(recall, precision, color=colors[name], lw=2, label=f"{name}: {results[name]['aupr']:.2f}")
        observed, predicted = calibration_curve(test_y, probability, n_bins=8, strategy="quantile")
        axes[0, 2].plot(predicted, observed, marker="o", color=colors[name], label=f"{name}: Brier {results[name]['brier']:.3f}")
    axes[0, 0].plot([0,1],[0,1],"--",color="#B7BEC6")
    axes[0, 0].set(xlabel="False positive rate",ylabel="True positive rate",title="Locked likely-label validation")
    axes[0, 1].axhline(test_y.mean(),ls="--",color="#B7BEC6")
    axes[0, 1].set(xlabel="Recall",ylabel="Precision",title="Precision–recall")
    axes[0, 2].plot([0,1],[0,1],"--",color="#B7BEC6")
    axes[0, 2].set(xlabel="Mean predicted probability",ylabel="Observed frequency",title="Reliability")
    for ax in axes[0]: ax.legend(fontsize=8)

    for name, group in selective.groupby("model"):
        axes[1, 0].plot(group.coverage, group.accuracy, color=colors[name], marker="o", ms=3, label=name)
    axes[1, 0].set(xlabel="Coverage",ylabel="Accuracy",title="Uncertainty-gated classification")
    axes[1, 0].legend(fontsize=8)

    full_vus = vus["Sequence + graph"]
    low, high = thresholds_by_model["Sequence + graph"]
    axes[1, 1].hist(full_vus.reclass_probability, bins=25, color="#C9CDD3", edgecolor="white")
    axes[1, 1].axvline(low,color=pub.ACCENT_GREEN)
    axes[1, 1].axvline(high,color=pub.ACCENT_AMBER)
    axes[1, 1].set(xlabel="Calibrated probability",ylabel="VUS count",title="Exploratory VUS calls")

    mechanism = full_vus[full_vus.call=="Pathogenic-like"].dominant_group.value_counts().head(6).sort_values()
    axes[1, 2].barh(mechanism.index.str.replace("_"," "),mechanism.values,color=pub.ACCENT_PURPLE)
    axes[1, 2].set(xlabel="VUS count",title="Dominant groups among pathogenic-like calls")
    for label_text, ax in zip("abcdef", axes.flat):
        pub.panel_label(ax, label_text)
        pub.soft_grid(ax, "both")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure7_reclassification_corrected.png",dpi=400,facecolor="white")
    fig.savefig(FIGURES / "figure7_reclassification_corrected.pdf",facecolor="white")
    plt.close(fig)


def main():
    graph, raw, mode = core.load_graph_and_raw_variants()
    relation_frequency = core.relation_frequencies(graph)
    definitive, y_oof, oof_raw = cross_fitted_predictions(
        graph, raw, relation_frequency
    )
    probability_calibrators = {
        name: fit_probability_calibrator(probability, y_oof)
        for name, probability in oof_raw.items()
    }
    oof_calibrated = {
        name: calibrate(probability_calibrators[name], probability)
        for name, probability in oof_raw.items()
    }

    definitive_ids = definitive["variant_id"].astype(str).tolist()
    signal_models = core.fit_signal_models(raw, definitive_ids)
    signals = core.apply_signal_models(raw, signal_models)
    samples = core.build_training_samples(
        graph, signals, relation_frequency, definitive_ids
    )
    scorer = core.train_path_scorer(samples, seed=core.RNG_SEED)
    core.validate_split_disjointness(scorer, signals)
    features = build_features(graph, signals, scorer, relation_frequency)
    features.to_csv(TABLES / "reclassification_features.csv", index=False)

    train = features[
        features["clinical_significance"].isin(
            core.DEFINITIVE_POS | core.DEFINITIVE_NEG
        )
    ].copy()
    test = features[
        features["clinical_significance"].isin(core.LIKELY_POS | core.LIKELY_NEG)
    ].copy()
    vus_features = features[
        features["clinical_significance"] == "Uncertain_significance"
    ].copy()
    train_y = train["clinical_significance"].isin(core.DEFINITIVE_POS).astype(int).to_numpy()
    test_y = test["clinical_significance"].isin(core.LIKELY_POS).astype(int).to_numpy()

    test_predictions = {}
    vus_outputs = {}
    results = {}
    thresholds_by_model = {}
    selective_frames = []
    for name, columns in [
        ("Sequence only", SEQ_FEATURES),
        ("Sequence + graph", ALL_FEATURES),
    ]:
        if not np.isfinite(train[columns].to_numpy()).all():
            raise ValueError(f"Non-finite final training features: {name}")
        with np.errstate(all="ignore"):
            model = classifier().fit(train[columns], train_y)
            test_raw = model.predict_proba(test[columns])[:, 1]
            vus_raw = model.predict_proba(vus_features[columns])[:, 1]
        test_probability = calibrate(probability_calibrators[name], test_raw)
        vus_probability = calibrate(probability_calibrators[name], vus_raw)
        test_predictions[name] = test_probability
        results[name] = bootstrap_metrics(
            test_y, test_probability, test["gene_id"].to_numpy()
        )
        low, high = thresholds(y_oof, oof_calibrated[name])
        thresholds_by_model[name] = (low, high)
        output = vus_features.copy()
        output["reclass_probability"] = vus_probability
        output["call"] = np.where(
            vus_probability >= high,
            "Pathogenic-like",
            np.where(vus_probability <= low, "Benign-like", "Still uncertain"),
        )
        vus_outputs[name] = output
        output.to_csv(
            TABLES / f"vus_calls_{name.lower().replace(' ','_').replace('+','plus')}.csv",
            index=False,
        )
        curve = selective_curve(
            test_y, test_probability, test["isotonic_entropy"].to_numpy()
        )
        curve["model"] = name
        selective_frames.append(curve)

    paired = paired_bootstrap(
        test_y,
        test_predictions["Sequence only"],
        test_predictions["Sequence + graph"],
        test["gene_id"].to_numpy(),
    )
    # Freeze the subgroup boundary from definitive development data.
    median_uncertainty = float(train["isotonic_entropy"].median())
    stratified_improvement = {}
    for name, mask in {
        "low_uncertainty": test["isotonic_entropy"].to_numpy() < median_uncertainty,
        "high_uncertainty": test["isotonic_entropy"].to_numpy() >= median_uncertainty,
    }.items():
        stratified_improvement[name] = {
            "n": int(mask.sum()),
            **paired_bootstrap(
                test_y[mask],
                test_predictions["Sequence only"][mask],
                test_predictions["Sequence + graph"][mask],
                test.loc[mask, "gene_id"].to_numpy(),
            ),
        }
    selective = pd.concat(selective_frames, ignore_index=True)
    selective.to_csv(TABLES / "selective_classification.csv", index=False)
    prediction_table = test[
        ["variant_id", "gene_symbol", "clinical_significance", "isotonic_entropy"]
    ].copy()
    prediction_table["label"] = test_y
    prediction_table["sequence_probability"] = test_predictions["Sequence only"]
    prediction_table["sequence_graph_probability"] = test_predictions["Sequence + graph"]
    prediction_table.to_csv(TABLES / "locked_validation_predictions.csv", index=False)

    payload = {
        "design": {
            "signal_calibration_training": "Pathogenic vs Benign only",
            "path_scorer_training": "Pathogenic vs Benign only",
            "classifier_training": "Pathogenic vs Benign only",
            "locked_validation": "Likely_pathogenic vs Likely_benign",
            "outer_folds_for_calibration": 5,
            "bootstrap_replicates": N_BOOT,
            "bootstrap_unit": "gene cluster",
            "uncertainty_subgroup_boundary": "definitive-training median",
            "vus_interpretation": "exploratory; no gold-standard labels",
        },
        "n": {
            "definitive_training": len(train),
            "locked_validation": len(test),
            "locked_validation_positive": int(test_y.sum()),
            "vus": len(vus_features),
        },
        "models": results,
        "paired_improvement": paired,
        "paired_improvement_by_uncertainty": stratified_improvement,
        "thresholds": {
            name: {"low": values[0], "high": values[1]}
            for name, values in thresholds_by_model.items()
        },
        "vus_calls": {
            name: frame["call"].value_counts().to_dict()
            for name, frame in vus_outputs.items()
        },
        "vus_yield_bootstrap": {
            name: bootstrap_yield(
                frame["reclass_probability"].to_numpy(),
                thresholds_by_model[name][0],
                thresholds_by_model[name][1],
                frame["gene_id"].to_numpy(),
            )
            for name, frame in vus_outputs.items()
        },
        "selective_accuracy": {
            name: {
                "coverage_50": float(
                    selective[
                        (selective["model"] == name)
                        & np.isclose(selective["coverage"], 0.5)
                    ]["accuracy"].iloc[0]
                ),
                "coverage_100": float(
                    selective[
                        (selective["model"] == name)
                        & np.isclose(selective["coverage"], 1.0)
                    ]["accuracy"].iloc[0]
                ),
            }
            for name in ["Sequence only", "Sequence + graph"]
        },
    }
    (REPORTS / "reclassification_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    make_figure(
        results,
        test_predictions,
        test_y,
        test,
        vus_outputs,
        thresholds_by_model,
        selective,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
