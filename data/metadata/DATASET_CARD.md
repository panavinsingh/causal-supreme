---
license: cc-by-4.0
task_categories:
  - question-answering
  - text-classification
language:
  - en
tags:
  - causal-reasoning
  - structural-causal-models
  - pearl-ladder
  - counterfactual
  - causal-inference
size_categories:
  - 10K<n<100K
---

# Causal-Supreme Dataset

## Dataset Summary

Causal-Supreme is a synthetic benchmark dataset designed to evaluate causal reasoning capabilities in language models. The dataset is constructed from linear-Gaussian structural causal models (SCMs) with random directed acyclic graph (DAG) topologies. Each sample consists of a causal graph, observed values, and a query spanning one of three levels of Pearl's Ladder of Causation: (1) associational queries about observed correlations, (2) interventional queries requiring do-calculus reasoning, and (3) counterfactual queries asking "what would have happened if" questions. The dataset guarantees zero structural overlap between training, validation, and test splits—no DAG topology appears in multiple splits. All samples include natural language prompts and ground-truth answers derived from exact SCM computations, enabling rigorous evaluation of causal reasoning without confounding from heuristic shortcuts.

## Supported Tasks

- **Causal Inference**: Determine causal effects from interventions
- **Counterfactual Reasoning**: Answer "what if" questions about alternative scenarios
- **Binary Question Answering**: Yes/no answers about causal relationships

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `dag_id` | string | Unique identifier for the DAG structure |
| `dag` | object | DAG with `nodes` (int) and `edges` ([[src, dst, weight], ...]) |
| `query` | object | Query info: `rung`, `type`, `treatment_node`, `outcome_node`, `intervention_value` |
| `X_factual` | array | Observed factual values (float[]) |
| `U_noise` | array | Exogenous noise terms (float[]) |
| `X_counterfactual` | array | Counterfactual values (float[] or null) |
| `effect_size` | float | Computed causal effect |
| `binary_answer` | string | Ground truth: "yes" or "no" |
| `nl_prompt` | string | Natural language question |
| `nl_answer` | string | Natural language explanation |

### Data Splits

| Split | Samples | Unique DAGs | Purpose |
|-------|---------|-------------|---------|
| Train | 30,000  | ~880        | Model training |
| Val   | 3,000   | ~110        | Hyperparameter tuning |
| Test  | 6,000   | ~110        | Final evaluation |

### Integrity Guarantees

- **Zero DAG Overlap**: No structural overlap between splits (hash-based splitting)
- **KS Compliance**: Variable distributions match (p > 0.01)
- **Balanced Rungs**: ~33% samples per rung level

## Dataset Creation

### Source Data

Synthetically generated using:
- Random Erdős-Rényi DAGs (5-20 nodes)
- Linear-Gaussian SCMs with random edge weights
- Exact counterfactual computation via abduction-action-prediction

### Curation Rationale

Existing causal reasoning benchmarks suffer from:
- Limited DAG diversity
- Train/test data leakage
- Incomplete Pearl's Ladder coverage

Causal-Supreme addresses these with guaranteed structural separation and full ladder coverage.

## Licensing

**Data License**: Creative Commons Attribution 4.0 International (CC-BY-4.0)

You are free to:
- **Share**: Copy and redistribute the material in any medium or format
- **Adapt**: Remix, transform, and build upon the material for any purpose

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode

## Citation

```bibtex
@dataset{causal_supreme_2026,
  title={Causal-Supreme: A Synthetic Benchmark for Causal Reasoning},
  year={2026},
  publisher={HuggingFace},
  license={CC-BY-4.0}
}
```
