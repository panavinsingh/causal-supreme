# Data Directory

This directory contains the Causal-Supreme synthetic dataset for causal reasoning evaluation. The dataset features structural causal models (SCMs) with guaranteed DAG properties, covering all three rungs of Pearl's Ladder: association, intervention, and counterfactual reasoning.

## Dataset Card

For detailed documentation, see [DATASET_CARD.md](metadata/DATASET_CARD.md).

## Generation Checksums

| File | SHA256 | Samples |
|------|--------|---------|
| `splits/train.jsonl` | *generated* | 30,000 |
| `splits/val.jsonl` | *generated* | 3,000 |
| `splits/test.jsonl` | *generated* | 6,000 |
| `splits/openai_subsets/cladder_600.csv` | *generated* | 600 |
| `splits/openai_subsets/p3sa_500.jsonl` | *generated* | 500 |

Checksums are stored in `metadata/hashes.json` and verified by CI on each push.

## Regenerating Data

```bash
# Generate raw data
python src/generator/generate_dataset.py --n-dags 1100 --samples-per-dag 30 --out data/raw/shard_000.jsonl

# Create splits
python src/splitting/create_splits.py --raw-dir data/raw --output-dir data/splits

# Regenerate checksums
python scripts/regen_hashes.py
```
