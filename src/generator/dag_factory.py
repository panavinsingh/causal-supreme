"""
DAG Factory - Generate random DAGs with guaranteed structural motifs.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel


class DagConfig(BaseModel):
    """Configuration for DAG generation."""
    n_nodes: int
    max_fanin: int
    p_collider: float
    p_fork: float
    p_chain: float
    seed: int


class DagFactory:
    """Factory for generating random DAGs with structural guarantees."""

    def __init__(self, config: DagConfig):
        self.config = config

    def build(self) -> nx.DiGraph:
        """
        Build a random DAG with guaranteed motifs.

        Guarantees:
        - Acyclic graph
        - In-degree <= max_fanin for all nodes
        - At least one fork (A->B, A->C)
        - At least one collider (A->C, B->C)
        - At least one length-3 chain (A->B->C->D)

        Returns:
            NetworkX DiGraph

        Raises:
            ValueError: If motif requirements cannot be satisfied
        """
        # Seed both numpy and random
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        n = self.config.n_nodes
        if n < 4:
            raise ValueError("n_nodes must be >= 4 to guarantee all motifs")

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        # First, guarantee the required motifs
        # Fork: 0 -> 1, 0 -> 2
        G.add_edge(0, 1, weight=np.random.uniform(-2, 2))
        G.add_edge(0, 2, weight=np.random.uniform(-2, 2))

        # Collider: 1 -> 3, 2 -> 3
        G.add_edge(1, 3, weight=np.random.uniform(-2, 2))
        G.add_edge(2, 3, weight=np.random.uniform(-2, 2))

        # Chain of length 3: need A->B->C->D (3 edges)
        # Use nodes 0->1->3->... (need node 4+ for the chain)
        if n >= 5:
            G.add_edge(3, 4, weight=np.random.uniform(-2, 2))
        else:
            # For n=4, use existing edges: 0->1->3 is length 2
            # We need a length-3 chain (4 nodes, 3 edges)
            # With only 4 nodes, 0->1, 1->3 gives length 2
            # Add 0->2->3 as alternative path, but we need 3 edges in sequence
            # Actually for n=4: 0->1->3 and 0->2->3, no length-3 chain possible
            raise ValueError("n_nodes must be >= 5 to guarantee length-3 chain")

        # Add remaining random edges
        for i in range(n):
            for j in range(i + 1, n):
                if G.has_edge(i, j):
                    continue

                # Check in-degree constraint
                if G.in_degree(j) >= self.config.max_fanin:
                    continue

                # Probabilistically add edge based on motif type
                # Determine if this would create a fork, collider, or chain
                out_degree_i = G.out_degree(i)
                in_degree_j = G.in_degree(j)

                p_add = 0.0
                if out_degree_i >= 1:  # Would extend a fork
                    p_add = max(p_add, self.config.p_fork)
                if in_degree_j >= 1:  # Would create/extend a collider
                    p_add = max(p_add, self.config.p_collider)
                if G.in_degree(i) >= 1:  # Would extend a chain
                    p_add = max(p_add, self.config.p_chain)

                # Base probability if none of the above
                if p_add == 0.0:
                    p_add = 0.2

                if np.random.random() < p_add:
                    G.add_edge(i, j, weight=np.random.uniform(-2, 2))

        # Verify motifs
        if not self._has_fork(G):
            raise ValueError("Failed to create fork motif")
        if not self._has_collider(G):
            raise ValueError("Failed to create collider motif")
        if not self._has_chain3(G):
            raise ValueError("Failed to create length-3 chain motif")

        # Verify acyclic
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Generated graph is not acyclic")

        return G

    def _has_fork(self, G: nx.DiGraph) -> bool:
        """Check if DAG has at least one fork (node with out-degree >= 2)."""
        for node in G.nodes():
            if G.out_degree(node) >= 2:
                return True
        return False

    def _has_collider(self, G: nx.DiGraph) -> bool:
        """Check if DAG has at least one collider (node with in-degree >= 2)."""
        for node in G.nodes():
            if G.in_degree(node) >= 2:
                return True
        return False

    def _has_chain3(self, G: nx.DiGraph) -> bool:
        """Check if DAG has at least one chain of length 3 (4 nodes, 3 edges)."""
        nodes = list(G.nodes())
        for start in nodes:
            for end in nodes:
                if start == end:
                    continue
                try:
                    for path in nx.all_simple_paths(G, start, end, cutoff=3):
                        if len(path) == 4:  # 4 nodes = 3 edges
                            return True
                except nx.NetworkXError:
                    continue
        return False


def save(dag: nx.DiGraph, path: Path) -> None:
    """
    Save DAG to JSON with deterministic ordering.

    Format: { "nodes": n, "edges": [[src, dst, weight], ...] }
    """
    edges = []
    for u, v, data in sorted(dag.edges(data=True)):
        weight = data.get("weight", 1.0)
        edges.append([u, v, round(weight, 6)])

    output = {
        "nodes": dag.number_of_nodes(),
        "edges": edges
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    # Demo
    config = DagConfig(
        n_nodes=10,
        max_fanin=3,
        p_collider=0.4,
        p_fork=0.3,
        p_chain=0.3,
        seed=42
    )

    factory = DagFactory(config)
    dag = factory.build()

    print(f"Generated DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(dag)}")

    save(dag, Path("test_dag.json"))
    print("Saved to test_dag.json")
