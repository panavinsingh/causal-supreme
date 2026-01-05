# Causal-Supreme

A research-grade synthetic dataset for evaluating causal reasoning in large language models, covering all three rungs of Pearl's Ladder of Causation.

## Project Overview

Causal-Supreme provides:
- **Structural Causal Models (SCMs)** with random DAG topologies (5-20 nodes)
- **Pearl's Ladder Coverage**: Association, Intervention, and Counterfactual queries
- **Guaranteed Integrity**: Zero DAG overlap between splits, KS-compliant distributions
- **LLM Evaluation Subsets**: Pre-stratified samples for cost-efficient API evaluation

```
causal-supreme/
│
├── data/
│   ├── raw/                    # Generator outputs (.jsonl shards)
│   ├── splits/
│   │   ├── train.jsonl         # ~26,400 samples
│   │   ├── val.jsonl           # ~3,300 samples  
│   │   ├── test.jsonl          # ~3,300 samples
│   │   └── openai_subsets/
│   │       ├── cladder_600.csv # CLadder benchmark (200/rung)
│   │       └── p3sa_500.jsonl  # P3SA synthetic (~167/rung)
│   └── metadata/
│
├── src/
│   ├── generator/              # DAG + SCM generation
│   ├── splitting/              # Hash-based splitting + KS fix
│   └── validation/             # Pytest integrity suite
│
├── tests/                      # Unit tests
├── configs/                    # YAML configurations
└── scripts/                    # Utility scripts
```

## Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n causal-supreme python=3.10 -y
conda activate causal-supreme

# Install dependencies
pip install -r requirements.txt
```

### Regenerate Dataset

```bash
# Generate raw data (1100 DAGs × 30 samples = 33,000)
python src/generator/generate_dataset.py \
    --conf configs/default.yaml \
    --n-dags 1100 \
    --samples-per-dag 30 \
    --out data/raw/shard_000.jsonl

# Create splits with KS correction
python src/splitting/create_splits.py --ks-fix --iterations 5

# Run integrity tests
pytest -q
```

### Using the Dataset

```python
import json

# Load P3SA test subset
with open("data/splits/openai_subsets/p3sa_500.jsonl") as f:
    samples = [json.loads(line) for line in f]

for sample in samples[:3]:
    print(f"Rung: {sample['query']['rung']}")
    print(f"Prompt: {sample['nl_prompt']}")
    print(f"Answer: {sample['binary_answer']}\n")
```

## OpenAI Evaluation Subsets

| Dataset | Samples | Per Rung | Format |
|---------|---------|----------|--------|
| CLadder | 600 | 200 | CSV |
| P3SA Synthetic | 498 | ~166 | JSONL |

**Rough API cost** = `(tokens_in × $rate_in) + (tokens_out × $rate_out)`  
We cap completions at 2,000 tokens, giving approximately **$0.15** for 1,098 samples with GPT-4o-mini.

## Data Structure

Each P3SA sample contains:

```json
{
  "dag_id": "dag_00042",
  "dag": {"nodes": 10, "edges": [[0,1,0.5], ...]},
  "query": {
    "rung": 3,
    "type": "counterfactual",
    "treatment_node": 2,
    "outcome_node": 7,
    "intervention_value": 1.5
  },
  "X_factual": [0.1, -0.5, ...],
  "X_counterfactual": [0.1, 1.5, ...],
  "effect_size": 0.613,
  "binary_answer": "yes",
  "nl_prompt": "In a system with 10 causally related variables...",
  "nl_answer": "Yes, the intervention would increase..."
}
```

## Integrity Guarantees

- ✅ **Zero DAG Overlap**: Hash-based splitting ensures no structural leakage
- ✅ **KS Compliance**: Micro-resampling aligns train/test distributions  
- ✅ **Balanced Rungs**: ~33% samples per Pearl's Ladder level
- ✅ **Motif Coverage**: Every DAG contains fork, collider, and chain structures

## Citation

```bibtex
@dataset{causal_supreme_2026,
  title     = {Causal-Supreme: A Synthetic Benchmark for Causal Reasoning in LLMs},
  author    = {Anonymous},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/anonymous/causal-supreme},
  note      = {Dataset for evaluating Pearl's Ladder reasoning}
}
```

## License

- **Code**: MIT License
- **Data**: CC-BY-4.0

See [LICENSE](LICENSE) for full text.
