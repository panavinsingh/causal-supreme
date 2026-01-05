"""
Dataset Generator - Generate causal reasoning dataset with CLI.

Usage:
    python generate_dataset.py \
        --conf dag_params.yaml \
        --n-dags 1100 --samples-per-dag 30 \
        --out data/raw/shard_000.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generator.dag_factory import DagConfig, DagFactory, save
from src.generator.scm_sampler import LinearSCM


def load_config(conf_path: Optional[Path]) -> Dict:
    """Load DAG configuration from YAML file."""
    if conf_path is None or not conf_path.exists():
        # Default config
        return {
            "n_nodes": 10,
            "max_fanin": 3,
            "p_collider": 0.4,
            "p_fork": 0.3,
            "p_chain": 0.3,
            "noise_scale": 1.0
        }

    with open(conf_path, "r") as f:
        return yaml.safe_load(f)


def generate_queries(
    dag_json: Dict,
    x: np.ndarray,
    u: np.ndarray,
    scm: LinearSCM,
    rung: int
) -> Dict:
    """
    Generate a query for a given DAG, factual world, and rung.

    Returns:
        Query dict with rung, type, treatment, outcome, etc.
    """
    n_nodes = dag_json["nodes"]

    # Select treatment and outcome nodes
    treatment_node = random.randint(0, n_nodes - 2)
    outcome_node = random.randint(treatment_node + 1, n_nodes - 1)

    # Intervention value
    intervention_value = random.uniform(-2, 2)
    intervention = {treatment_node: intervention_value}

    query = {
        "rung": rung,
        "treatment_node": treatment_node,
        "outcome_node": outcome_node,
        "intervention_value": intervention_value,
    }

    x_counterfactual = None
    effect_size = 0.0

    if rung == 1:
        # Association: just observe
        query["type"] = "association"
        effect_size = x[outcome_node]

    elif rung == 2:
        # Intervention: do(X_t = v)
        query["type"] = "intervention"
        x_int = scm.intervene(u, intervention)
        x_counterfactual = x_int
        effect_size = x_int[outcome_node] - x[outcome_node]

    elif rung == 3:
        # Counterfactual: what if X_t had been v?
        query["type"] = "counterfactual"
        x_cf, _ = scm.counterfactual(x, intervention)
        x_counterfactual = x_cf
        effect_size = x_cf[outcome_node] - x[outcome_node]

    return query, x_counterfactual, effect_size


def generate_nl_prompt(dag_json: Dict, query: Dict, x: np.ndarray) -> str:
    """Generate natural language prompt for query."""
    n_nodes = dag_json["nodes"]
    rung = query["rung"]
    t_node = query["treatment_node"]
    o_node = query["outcome_node"]
    int_val = query["intervention_value"]

    if rung == 1:
        return (
            f"In a system with {n_nodes} causally related variables, "
            f"we observe X{t_node} = {x[t_node]:.2f}. "
            f"What is the expected value of X{o_node}?"
        )
    elif rung == 2:
        return (
            f"In a system with {n_nodes} causally related variables, "
            f"if we intervene to set X{t_node} = {int_val:.2f}, "
            f"what would be the expected value of X{o_node}?"
        )
    else:  # rung == 3
        return (
            f"In a system with {n_nodes} causally related variables, "
            f"we observed X{t_node} = {x[t_node]:.2f} and X{o_node} = {x[o_node]:.2f}. "
            f"Had X{t_node} been {int_val:.2f} instead, "
            f"what would X{o_node} have been?"
        )


def generate_binary_answer(effect_size: float, threshold: float = 0.5) -> str:
    """Generate binary answer based on effect size."""
    return "yes" if abs(effect_size) > threshold else "no"


def generate_nl_answer(effect_size: float, binary: str) -> str:
    """Generate natural language answer."""
    if binary == "yes":
        direction = "increase" if effect_size > 0 else "decrease"
        return f"Yes, the intervention would {direction} the outcome by {abs(effect_size):.2f}."
    else:
        return "No, the intervention would not have a significant effect."


def generate_dataset(
    conf_path: Optional[Path],
    n_dags: int,
    samples_per_dag: int,
    output_path: Path,
    base_seed: int = 42
) -> None:
    """
    Generate dataset and write to JSONL.

    Args:
        conf_path: Path to DAG configuration YAML
        n_dags: Number of unique DAGs to generate
        samples_per_dag: Samples per DAG
        output_path: Output JSONL file path
        base_seed: Base random seed
    """
    config_dict = load_config(conf_path)

    # Set seeds
    np.random.seed(base_seed)
    random.seed(base_seed)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_dags} DAGs with {samples_per_dag} samples each...")
    print(f"Output: {output_path}")

    total_samples = 0

    with open(output_path, "w") as f:
        for dag_idx in range(n_dags):
            # Create DAG
            dag_config = DagConfig(
                n_nodes=config_dict.get("n_nodes", 10),
                max_fanin=config_dict.get("max_fanin", 3),
                p_collider=config_dict.get("p_collider", 0.4),
                p_fork=config_dict.get("p_fork", 0.3),
                p_chain=config_dict.get("p_chain", 0.3),
                seed=base_seed + dag_idx
            )

            try:
                factory = DagFactory(dag_config)
                dag = factory.build()
            except ValueError as e:
                print(f"Warning: DAG {dag_idx} failed: {e}")
                continue

            # Convert to JSON format
            dag_json = {
                "nodes": dag.number_of_nodes(),
                "edges": [
                    [u, v, round(data.get("weight", 1.0), 6)]
                    for u, v, data in sorted(dag.edges(data=True))
                ]
            }

            dag_id = f"dag_{dag_idx:05d}"

            # Create SCM
            scm = LinearSCM(dag_json, noise_scale=config_dict.get("noise_scale", 1.0))

            # Generate samples
            for sample_idx in range(samples_per_dag):
                # Sample factual world
                x, u = scm.sample()

                # Select rung (balanced)
                rung = (sample_idx % 3) + 1

                # Generate query
                query, x_cf, effect_size = generate_queries(dag_json, x, u, scm, rung)

                # Generate NL
                nl_prompt = generate_nl_prompt(dag_json, query, x)
                binary_answer = generate_binary_answer(effect_size)
                nl_answer = generate_nl_answer(effect_size, binary_answer)

                # Build record
                record = {
                    "dag_id": dag_id,
                    "dag": dag_json,
                    "query": query,
                    "X_factual": x.round(6).tolist(),
                    "U_noise": u.round(6).tolist(),
                    "X_counterfactual": x_cf.round(6).tolist() if x_cf is not None else None,
                    "effect_size": round(effect_size, 6),
                    "binary_answer": binary_answer,
                    "nl_prompt": nl_prompt,
                    "nl_answer": nl_answer
                }

                f.write(json.dumps(record) + "\n")
                total_samples += 1

            if (dag_idx + 1) % 100 == 0:
                print(f"  Generated {dag_idx + 1}/{n_dags} DAGs...")

    print(f"Done! Total samples: {total_samples}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate causal reasoning dataset"
    )
    parser.add_argument(
        "--conf",
        type=Path,
        default=None,
        help="Path to DAG configuration YAML"
    )
    parser.add_argument(
        "--n-dags",
        type=int,
        default=1100,
        help="Number of unique DAGs to generate"
    )
    parser.add_argument(
        "--samples-per-dag",
        type=int,
        default=30,
        help="Number of samples per DAG"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/shard_000.jsonl"),
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    generate_dataset(
        conf_path=args.conf,
        n_dags=args.n_dags,
        samples_per_dag=args.samples_per_dag,
        output_path=args.out,
        base_seed=args.seed
    )


if __name__ == "__main__":
    main()
