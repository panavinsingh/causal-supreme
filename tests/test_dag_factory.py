"""
Unit tests for DAG Factory.
"""

import pytest
import networkx as nx
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator.dag_factory import DagConfig, DagFactory, save


class TestDagFactory:
    """Test suite for DAG Factory."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_seeds_produce_acyclic_dags(self, seed: int):
        """Test that 10 random seeds all produce acyclic DAGs."""
        config = DagConfig(
            n_nodes=10,
            max_fanin=3,
            p_collider=0.4,
            p_fork=0.3,
            p_chain=0.3,
            seed=seed
        )

        factory = DagFactory(config)
        dag = factory.build()

        assert nx.is_directed_acyclic_graph(dag), f"Seed {seed} produced cyclic graph"

    @pytest.mark.parametrize("seed", range(10))
    def test_random_seeds_have_fork_motif(self, seed: int):
        """Test that 10 random seeds all have at least one fork."""
        config = DagConfig(
            n_nodes=10,
            max_fanin=3,
            p_collider=0.4,
            p_fork=0.3,
            p_chain=0.3,
            seed=seed
        )

        factory = DagFactory(config)
        dag = factory.build()

        # Check for fork: node with out-degree >= 2
        has_fork = any(dag.out_degree(n) >= 2 for n in dag.nodes())
        assert has_fork, f"Seed {seed} has no fork motif"

    @pytest.mark.parametrize("seed", range(10))
    def test_random_seeds_have_collider_motif(self, seed: int):
        """Test that 10 random seeds all have at least one collider."""
        config = DagConfig(
            n_nodes=10,
            max_fanin=3,
            p_collider=0.4,
            p_fork=0.3,
            p_chain=0.3,
            seed=seed
        )

        factory = DagFactory(config)
        dag = factory.build()

        # Check for collider: node with in-degree >= 2
        has_collider = any(dag.in_degree(n) >= 2 for n in dag.nodes())
        assert has_collider, f"Seed {seed} has no collider motif"

    @pytest.mark.parametrize("seed", range(10))
    def test_random_seeds_have_chain3_motif(self, seed: int):
        """Test that 10 random seeds all have at least one length-3 chain."""
        config = DagConfig(
            n_nodes=10,
            max_fanin=3,
            p_collider=0.4,
            p_fork=0.3,
            p_chain=0.3,
            seed=seed
        )

        factory = DagFactory(config)
        dag = factory.build()

        # Check for length-3 chain: path with 4 nodes (3 edges)
        has_chain3 = False
        nodes = list(dag.nodes())
        for start in nodes:
            for end in nodes:
                if start == end:
                    continue
                try:
                    for path in nx.all_simple_paths(dag, start, end, cutoff=3):
                        if len(path) == 4:
                            has_chain3 = True
                            break
                except nx.NetworkXError:
                    continue
                if has_chain3:
                    break
            if has_chain3:
                break

        assert has_chain3, f"Seed {seed} has no length-3 chain motif"

    @pytest.mark.parametrize("seed", range(10))
    def test_random_seeds_respect_max_fanin(self, seed: int):
        """Test that 10 random seeds all respect max_fanin constraint."""
        max_fanin = 3
        config = DagConfig(
            n_nodes=10,
            max_fanin=max_fanin,
            p_collider=0.4,
            p_fork=0.3,
            p_chain=0.3,
            seed=seed
        )

        factory = DagFactory(config)
        dag = factory.build()

        for node in dag.nodes():
            assert dag.in_degree(node) <= max_fanin, \
                f"Seed {seed}: node {node} has in-degree {dag.in_degree(node)} > {max_fanin}"

    def test_save_produces_valid_json(self, tmp_path: Path):
        """Test that save() produces valid JSON with correct structure."""
        import json

        config = DagConfig(
            n_nodes=5,
            max_fanin=3,
            p_collider=0.4,
            p_fork=0.3,
            p_chain=0.3,
            seed=42
        )

        factory = DagFactory(config)
        dag = factory.build()

        output_path = tmp_path / "test_dag.json"
        save(dag, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert "nodes" in data
        assert "edges" in data
        assert data["nodes"] == dag.number_of_nodes()
        assert len(data["edges"]) == dag.number_of_edges()

        # Check edge format: [src, dst, weight]
        for edge in data["edges"]:
            assert len(edge) == 3
            assert isinstance(edge[0], int)
            assert isinstance(edge[1], int)
            assert isinstance(edge[2], (int, float))

    def test_insufficient_nodes_raises_error(self):
        """Test that n_nodes < 5 raises ValueError."""
        config = DagConfig(
            n_nodes=4,
            max_fanin=3,
            p_collider=0.4,
            p_fork=0.3,
            p_chain=0.3,
            seed=42
        )

        factory = DagFactory(config)

        with pytest.raises(ValueError, match="n_nodes must be >= 5"):
            factory.build()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
