"""
Integrity Tests - Pytest suite for dataset validation.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pytest
from scipy import stats

from conftest import KS_P, MIN_MOTIFS


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_dag_hash(sample: Dict) -> str:
    """Compute structural hash for a DAG."""
    dag = sample.get("dag", {})
    edges = sorted([tuple(e[:2]) for e in dag.get("edges", [])])
    n_nodes = dag.get("nodes", 0)
    canonical = json.dumps({"n_nodes": n_nodes, "edges": edges}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_dag(sample: Dict) -> nx.DiGraph:
    """Build NetworkX DAG from sample."""
    dag_info = sample.get("dag", {})
    G = nx.DiGraph()
    G.add_nodes_from(range(dag_info.get("nodes", 0)))
    for edge in dag_info.get("edges", []):
        G.add_edge(edge[0], edge[1])
    return G


def has_fork(G: nx.DiGraph) -> bool:
    """Check for fork motif (out-degree >= 2)."""
    return any(G.out_degree(n) >= 2 for n in G.nodes())


def has_collider(G: nx.DiGraph) -> bool:
    """Check for collider motif (in-degree >= 2)."""
    return any(G.in_degree(n) >= 2 for n in G.nodes())


def has_chain3(G: nx.DiGraph) -> bool:
    """Check for length-3 chain (4 nodes, 3 edges)."""
    nodes = list(G.nodes())
    for start in nodes:
        for end in nodes:
            if start == end:
                continue
            try:
                for path in nx.all_simple_paths(G, start, end, cutoff=3):
                    if len(path) == 4:
                        return True
            except nx.NetworkXError:
                continue
    return False


# ============================================================================
# DAG INTEGRITY TESTS
# ============================================================================

class TestDagIntegrity:
    """Test DAG structure integrity."""

    def test_no_dag_overlap_train_val(self, train_data, val_data):
        """Verify no DAG overlap between train and val."""
        train_hashes = {get_dag_hash(s) for s in train_data}
        val_hashes = {get_dag_hash(s) for s in val_data}
        overlap = train_hashes & val_hashes
        assert len(overlap) == 0, f"Train-Val overlap: {len(overlap)} DAGs"

    def test_no_dag_overlap_train_test(self, train_data, test_data):
        """Verify no DAG overlap between train and test."""
        train_hashes = {get_dag_hash(s) for s in train_data}
        test_hashes = {get_dag_hash(s) for s in test_data}
        overlap = train_hashes & test_hashes
        assert len(overlap) == 0, f"Train-Test overlap: {len(overlap)} DAGs"

    def test_no_dag_overlap_val_test(self, val_data, test_data):
        """Verify no DAG overlap between val and test."""
        val_hashes = {get_dag_hash(s) for s in val_data}
        test_hashes = {get_dag_hash(s) for s in test_data}
        overlap = val_hashes & test_hashes
        assert len(overlap) == 0, f"Val-Test overlap: {len(overlap)} DAGs"

    def test_all_dags_acyclic(self, all_splits):
        """Verify all DAGs are acyclic."""
        for split_name, samples in all_splits.items():
            for i, s in enumerate(samples[:100]):
                G = build_dag(s)
                assert nx.is_directed_acyclic_graph(G), \
                    f"{split_name}[{i}] is not acyclic"

    def test_motif_coverage(self, train_data):
        """Verify DAGs have required motifs."""
        fork_count = 0
        collider_count = 0
        chain_count = 0

        for s in train_data[:100]:
            G = build_dag(s)
            if has_fork(G):
                fork_count += 1
            if has_collider(G):
                collider_count += 1
            if has_chain3(G):
                chain_count += 1

        assert fork_count >= MIN_MOTIFS, f"Fork motifs: {fork_count} < {MIN_MOTIFS}"
        assert collider_count >= MIN_MOTIFS, f"Collider motifs: {collider_count} < {MIN_MOTIFS}"
        assert chain_count >= MIN_MOTIFS, f"Chain motifs: {chain_count} < {MIN_MOTIFS}"


# ============================================================================
# DISTRIBUTION TESTS
# ============================================================================

class TestDistributions:
    """Test variable distribution integrity."""

    def test_ks_variable_distributions(self, train_data, test_data):
        """Verify KS test compliance for variable distributions."""
        train_vars = defaultdict(list)
        test_vars = defaultdict(list)

        for s in train_data[:1000]:
            x = s.get("X_factual", [])
            for i, v in enumerate(x):
                train_vars[i].append(v)

        for s in test_data[:500]:
            x = s.get("X_factual", [])
            for i, v in enumerate(x):
                test_vars[i].append(v)

        failures = []
        for var_idx in sorted(train_vars.keys()):
            if var_idx not in test_vars or len(test_vars[var_idx]) < 10:
                continue

            stat, p = stats.ks_2samp(train_vars[var_idx], test_vars[var_idx])
            if p < KS_P:
                failures.append((var_idx, p))

        assert len(failures) == 0, \
            f"KS test failed for variables: {[f[0] for f in failures]}"

    def test_rung_distribution(self, test_data):
        """Verify balanced rung distribution in test set."""
        rung_counts = defaultdict(int)

        for s in test_data:
            rung = s.get("query", {}).get("rung", 0)
            rung_counts[rung] += 1

        total = len(test_data)
        for rung in [1, 2, 3]:
            if rung in rung_counts:
                proportion = rung_counts[rung] / total
                assert 0.2 <= proportion <= 0.5, \
                    f"Rung {rung} proportion {proportion:.2%} out of range"


# ============================================================================
# EFFECT SIZE TESTS
# ============================================================================

class TestEffectSize:
    """Test effect size calculations."""

    def test_effect_size_sign(self, test_data):
        """
        Check sign of effect agrees with factual vs counterfactual mean.
        
        For rung 2/3: effect_size should have same sign as 
        (X_counterfactual[outcome] - X_factual[outcome])
        """
        mismatches = 0
        tested = 0

        for s in test_data[:500]:
            query = s.get("query", {})
            rung = query.get("rung", 1)

            if rung < 2:
                continue

            x_factual = s.get("X_factual", [])
            x_cf = s.get("X_counterfactual", [])
            effect_size = s.get("effect_size", 0.0)
            outcome_node = query.get("outcome_node", 0)

            if x_cf is None or len(x_cf) <= outcome_node:
                continue

            # Compute expected sign
            actual_diff = x_cf[outcome_node] - x_factual[outcome_node]

            # Check sign agreement (allowing for small tolerance)
            if abs(effect_size) > 0.01 and abs(actual_diff) > 0.01:
                if (effect_size > 0) != (actual_diff > 0):
                    mismatches += 1

            tested += 1

        if tested > 0:
            error_rate = mismatches / tested
            assert error_rate < 0.05, \
                f"Effect size sign mismatch: {mismatches}/{tested} ({error_rate:.1%})"


# ============================================================================
# FIELD VALIDATION TESTS
# ============================================================================

class TestFieldValidation:
    """Test required fields are present."""

    def test_required_fields_present(self, all_splits):
        """Verify all required fields are present."""
        required = ["dag_id", "dag", "query", "X_factual", "binary_answer"]

        for split_name, samples in all_splits.items():
            for i, s in enumerate(samples[:100]):
                for field in required:
                    assert field in s, f"{split_name}[{i}] missing field: {field}"

    def test_dag_structure_valid(self, all_splits):
        """Verify DAG structures are valid."""
        for split_name, samples in all_splits.items():
            for i, s in enumerate(samples[:100]):
                dag = s.get("dag", {})
                n_nodes = dag.get("nodes", 0)
                edges = dag.get("edges", [])

                assert n_nodes > 0, f"{split_name}[{i}] has 0 nodes"

                for edge in edges:
                    u, v = edge[0], edge[1]
                    assert 0 <= u < n_nodes, f"{split_name}[{i}] invalid edge source: {u}"
                    assert 0 <= v < n_nodes, f"{split_name}[{i}] invalid edge target: {v}"
                    assert u != v, f"{split_name}[{i}] self-loop: {u}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
