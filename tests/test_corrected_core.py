#!/usr/bin/env python3
import math
from pathlib import Path
import sys
import unittest

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import corrected_core as core


class CorrectedCoreTests(unittest.TestCase):
    def setUp(self):
        core.PATH_CACHE.clear()
        core.MECHANISM_CACHE.clear()
        self.graph = nx.DiGraph()
        self.graph.add_node("g0", type="Gene", name="START")
        self.graph.add_node("g1", type="Gene", name="ROUTE")
        self.graph.add_node("bp0", type="BiologicalProcess", name="memory learning")
        self.graph.add_node("d0", type="Disease", name="Disease")
        self.graph.add_node("p0", type="Phenotype", name="memory loss")
        self.graph.add_node("p1", type="Phenotype", name="progressive aphasia")
        self.graph.add_edge("g0", "g1", relation="GENEINTERACTSWITHGENE")
        self.graph.add_edge(
            "g0", "bp0", relation="GENEPARTICIPATESINBIOLOGICALPROCESS"
        )
        self.graph.add_edge("g0", "d0", relation="GENEASSOCIATESWITHDISEASE")
        self.graph.add_edge("g1", "d0", relation="GENEASSOCIATESWITHDISEASE")
        self.graph.add_edge("d0", "p0", relation="DISEASEMANIFESTSSYMPTOM")
        self.graph.add_edge("d0", "p1", relation="DISEASEMANIFESTSSYMPTOM")

    def test_enumerator_returns_only_valid_paths(self):
        paths = core.enumerate_candidate_paths(self.graph, "g0")
        self.assertEqual(len(paths), 4)
        self.assertTrue(any("g1" in path for path in paths))
        for path in paths:
            self.assertEqual(self.graph.nodes[path[-1]]["type"], "Phenotype")
            self.assertTrue(
                all(self.graph.has_edge(path[i], path[i + 1]) for i in range(len(path) - 1))
            )

    def test_feature_vector_has_declared_dimension(self):
        paths = core.enumerate_candidate_paths(self.graph, "g0")
        frequencies = core.relation_frequencies(self.graph)
        features, groups = core.feature_matrix(
            self.graph, paths, frequencies, pathogenicity=0.7, uncertainty=0.2
        )
        self.assertEqual(features.shape, (4, 14))
        self.assertEqual(len(groups), 4)
        self.assertTrue(np.isfinite(features).all())

    def test_metropolis_acceptance_obeys_detailed_balance(self):
        utilities = np.array([-0.4, 0.3, 1.2])
        tau = core.theta(0.4)
        target = core.exact_path_probabilities(utilities, 0.4)
        proposal = core.proposal_probabilities(utilities, 0.4)
        for i in range(len(utilities)):
            for j in range(len(utilities)):
                a_ij = min(
                    1.0,
                    math.exp(
                        (utilities[j] - utilities[i]) / tau
                        + math.log(proposal[i])
                        - math.log(proposal[j])
                    ),
                )
                a_ji = min(
                    1.0,
                    math.exp(
                        (utilities[i] - utilities[j]) / tau
                        + math.log(proposal[j])
                        - math.log(proposal[i])
                    ),
                )
                self.assertAlmostEqual(
                    target[i] * proposal[j] * a_ij,
                    target[j] * proposal[i] * a_ji,
                    places=12,
                )

    def test_sampler_approximates_exact_target(self):
        utilities = np.array([-0.4, 0.3, 1.2])
        expected = core.exact_path_probabilities(utilities, 0.6)
        final, _, _ = core.metropolis_sample(
            utilities, uncertainty=0.6, n_chains=50000, seed=17
        )
        observed = np.bincount(final, minlength=3) / len(final)
        self.assertLess(float(np.abs(observed - expected).sum()), 0.035)

    def test_proposal_has_full_support_under_extreme_utilities(self):
        probability = core.proposal_probabilities(
            np.array([-1e6, 0.0, 1e6]), uncertainty=0.0
        )
        self.assertTrue((probability > 0).all())
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=12)

    def test_empty_and_singleton_support(self):
        empty, acceptance, steps = core.metropolis_sample(
            np.array([]), uncertainty=0.5
        )
        self.assertEqual(len(empty), 0)
        self.assertEqual(acceptance, 0.0)
        self.assertEqual(steps, 0)
        singleton, acceptance, _ = core.metropolis_sample(
            np.array([2.0]), uncertainty=0.5, n_chains=20
        )
        self.assertTrue((singleton == 0).all())
        self.assertEqual(acceptance, 1.0)

    def test_fixed_seed_is_reproducible(self):
        utilities = np.array([-0.2, 0.1, 0.9, 1.3])
        left = core.metropolis_sample(
            utilities, uncertainty=0.4, n_chains=100, seed=81
        )
        right = core.metropolis_sample(
            utilities, uncertainty=0.4, n_chains=100, seed=81
        )
        np.testing.assert_array_equal(left[0], right[0])
        self.assertEqual(left[1:], right[1:])

    def test_aggregation_conserves_mass(self):
        distribution = core.aggregate_distribution(
            ["a", "a", "b"], [0.2, 0.3, 0.5]
        )
        self.assertAlmostEqual(distribution["a"], 0.5)
        self.assertAlmostEqual(distribution["b"], 0.5)
        self.assertAlmostEqual(sum(distribution.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
