.PHONY: quick_gen full_gen validate hashes cost clean

# Quick generation (100 examples for testing)
quick_gen:
	python src/generator/generate_dataset.py \
		--n-dags 10 \
		--samples-per-dag 10 \
		--out data/raw/shard_000.jsonl
	python src/splitting/create_splits.py \
		--raw-dir data/raw \
		--output-dir data/splits

# Full generation (50k examples, ~2 hours)
full_gen:
	python src/generator/generate_dataset.py \
		--n-dags 1100 \
		--samples-per-dag 50 \
		--out data/raw/shard_000.jsonl
	python src/splitting/create_splits.py \
		--raw-dir data/raw \
		--output-dir data/splits \
		--ks-fix

# Run validation tests
validate:
	pytest -v

# Regenerate data checksums
hashes:
	python scripts/regen_hashes.py

# Estimate API costs
cost:
	python scripts/cost_estimator.py

# Clean generated files
clean:
	rm -rf data/raw/*.jsonl
	rm -rf data/splits/*.jsonl
	rm -rf data/splits/openai_subsets/*
