# Causal-Supreme

A research-grade synthetic dataset for evaluating causal reasoning in language models.

```
causal-supreme/
│
├── data/
│   ├── raw/                    # Generator outputs (.jsonl shards)
│   ├── splits/
│   │   ├── train.jsonl         # 30,000 samples
│   │   ├── val.jsonl           # 3,000 samples
│   │   ├── test.jsonl          # 6,000 samples
│   │   └── openai_subsets/
│   │       ├── cladder_600.csv
│   │       └── p3sa_500.jsonl
│   └── metadata/
│       ├── README_data.md
│       ├── DATASET_CARD.md
│       └── LICENSE
│
├── src/
│   ├── generator/
│   │   ├── dag_factory.py      # DAG generation with motif guarantees
│   │   ├── scm_sampler.py      # Linear-Gaussian SCM sampling
│   │   └── generate_dataset.py # CLI dataset generator
│   ├── splitting/
│   │   └── create_splits.py    # Hash-based splitting + KS fix
│   └── validation/
│       └── integrity_tests.py  # Pytest integrity suite
│
├── notebooks/
│   ├── 00_dag_design.ipynb
│   ├── 01_generation_demo.ipynb
│   └── 02_statistics_exploration.ipynb
│
├── tests/
│   ├── conftest.py             # Pytest fixtures + constants
│   └── test_dag_factory.py     # DAG factory unit tests
│
├── scripts/
│   ├── cost_estimator.py       # API cost estimation
│   └── regen_hashes.py         # Regenerate checksums
│
├── .github/workflows/ci.yml    # GitHub Actions CI
├── Makefile                    # Quick commands
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick generation (100 examples)
make quick_gen

# Full generation (50k examples, ~2h)
make full_gen

# Run validation tests
make validate
```

## Make Commands

| Command | Description |
|---------|-------------|
| `make quick_gen` | Generate 100 toy examples for testing |
| `make full_gen` | Generate 50k examples (takes ~2 hours) |
| `make validate` | Run pytest integrity tests |
| `make hashes` | Regenerate data checksums |
| `make cost` | Estimate API evaluation costs |

## Features

- **Pearl's Ladder Coverage**: Association, Intervention, Counterfactual queries
- **Guaranteed DAG Properties**: Fork, collider, and chain motifs
- **Zero Overlap Splits**: Hash-based splitting ensures no structural leakage
- **KS-Compliant Distributions**: Micro-resampling fixes distribution drift
- **HuggingFace Compatible**: Standard dataset card format

## License

- **Code**: MIT License
- **Data**: CC-BY-4.0

See [LICENSE](LICENSE) for full text.
