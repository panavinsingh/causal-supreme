"""
SCM Sampler - Linear-Gaussian Structural Causal Models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class LinearSCM:
    """
    Linear-Gaussian Structural Causal Model.

    Structural equation: x_j = Σ_i w_ij x_i + u_j

    where:
    - w_ij are edge weights from the DAG
    - u_j is exogenous noise for node j
    """

    def __init__(self, dag_json: Dict, noise_scale: float = 1.0):
        """
        Initialize LinearSCM from DAG JSON.

        Args:
            dag_json: Dictionary with "nodes" (int) and "edges" ([[src, dst, weight], ...])
            noise_scale: Standard deviation of Gaussian noise
        """
        self.n_nodes = dag_json["nodes"]
        self.noise_scale = noise_scale

        # Build adjacency matrix of weights
        self.W = np.zeros((self.n_nodes, self.n_nodes))
        for edge in dag_json["edges"]:
            src, dst = edge[0], edge[1]
            weight = edge[2] if len(edge) > 2 else np.random.uniform(-2, 2)
            self.W[src, dst] = weight

        # Build NetworkX graph for topological sort
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.n_nodes))
        for edge in dag_json["edges"]:
            self.G.add_edge(edge[0], edge[1])

        # Get topological order
        self.topo_order = list(nx.topological_sort(self.G))

    def sample(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from the SCM.

        Args:
            u: Optional exogenous noise vector. If provided, reuse for counterfactual.
               If None, sample new noise.

        Returns:
            Tuple of (x, u) where:
            - x: Endogenous variables (R^n)
            - u: Exogenous noise used (R^n)
        """
        # Sample or reuse exogenous noise
        if u is None:
            u = np.random.normal(0, self.noise_scale, self.n_nodes)

        # Compute endogenous variables in topological order
        x = np.zeros(self.n_nodes)

        for j in self.topo_order:
            # x_j = Σ_i w_ij x_i + u_j
            parent_contribution = 0.0
            for i in self.G.predecessors(j):
                parent_contribution += self.W[i, j] * x[i]
            x[j] = parent_contribution + u[j]

        return x, u

    def intervene(
        self,
        u: np.ndarray,
        intervention: Dict[int, float]
    ) -> np.ndarray:
        """
        Compute interventional distribution do(X_k = v).

        Args:
            u: Exogenous noise vector
            intervention: Dict mapping node index to intervention value

        Returns:
            x: Endogenous variables under intervention
        """
        x = np.zeros(self.n_nodes)

        for j in self.topo_order:
            if j in intervention:
                # Intervened node: set to intervention value
                x[j] = intervention[j]
            else:
                # x_j = Σ_i w_ij x_i + u_j
                parent_contribution = 0.0
                for i in self.G.predecessors(j):
                    parent_contribution += self.W[i, j] * x[i]
                x[j] = parent_contribution + u[j]

        return x

    def counterfactual(
        self,
        x_factual: np.ndarray,
        intervention: Dict[int, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute counterfactual via abduction-action-prediction.

        Args:
            x_factual: Observed factual values
            intervention: Dict mapping node index to intervention value

        Returns:
            Tuple of (x_cf, u) where:
            - x_cf: Counterfactual values
            - u: Abducted noise
        """
        # Abduction: infer u from x_factual
        u = np.zeros(self.n_nodes)
        for j in self.topo_order:
            parent_contribution = 0.0
            for i in self.G.predecessors(j):
                parent_contribution += self.W[i, j] * x_factual[i]
            u[j] = x_factual[j] - parent_contribution

        # Action + Prediction: compute x_cf with intervention
        x_cf = self.intervene(u, intervention)

        return x_cf, u


def load_dag_json(path: Path) -> Dict:
    """Load DAG from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # Demo
    dag_json = {
        "nodes": 5,
        "edges": [[0, 1, 0.5], [0, 2, -0.3], [1, 3, 0.8], [2, 3, 0.4], [3, 4, 1.0]]
    }

    scm = LinearSCM(dag_json, noise_scale=1.0)

    # Sample factual
    x, u = scm.sample()
    print(f"Factual x: {x.round(3)}")
    print(f"Noise u: {u.round(3)}")

    # Intervention
    x_int = scm.intervene(u, {2: 0.0})
    print(f"Intervention do(X2=0): {x_int.round(3)}")

    # Counterfactual
    x_cf, u_abducted = scm.counterfactual(x, {2: 0.0})
    print(f"Counterfactual X2=0: {x_cf.round(3)}")
