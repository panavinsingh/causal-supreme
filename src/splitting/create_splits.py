"""
Create Splits - Hash-based train/val/test splitting with KS correction.

Features:
- Hash dag_id → bucket 0-99 for deterministic splits
- OpenAI subsets: CLadder 600 + P3SA 500
- --ks-fix flag for micro-resampling

Usage:
    python create_splits.py --raw-dir data/raw --output-dir data/splits
    python create_splits.py --ks-fix --iterations 5
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

SEED = 42


# ============================================================================
# JSONL I/O
# ============================================================================

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples


def save_jsonl(samples: List[Dict], filepath: Path) -> None:
    """Save samples to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def save_ids(ids: List[str], filepath: Path) -> None:
    """Save sample IDs to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({"ids": ids, "seed": SEED}, f, indent=2)


# ============================================================================
# HASH-BASED SPLITTING
# ============================================================================

def dag_id_to_bucket(dag_id: str, n_buckets: int = 100) -> int:
    """Hash dag_id to bucket 0-(n_buckets-1)."""
    h = hashlib.sha256(dag_id.encode()).hexdigest()
    return int(h, 16) % n_buckets


def split_by_hash(
    samples: List[Dict],
    train_buckets: range = range(0, 80),
    val_buckets: range = range(80, 90),
    test_buckets: range = range(90, 100)
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples by hashing dag_id to buckets.

    Default: 80% train, 10% val, 10% test
    """
    train, val, test = [], [], []

    for sample in samples:
        dag_id = sample.get("dag_id", "")
        bucket = dag_id_to_bucket(dag_id)

        if bucket in train_buckets:
            train.append(sample)
        elif bucket in val_buckets:
            val.append(sample)
        elif bucket in test_buckets:
            test.append(sample)

    return train, val, test


# ============================================================================
# INTEGRITY CHECKS
# ============================================================================

def get_dag_hash(sample: Dict) -> str:
    """Compute structural hash for a DAG."""
    dag = sample.get("dag", {})
    edges = sorted([tuple(e[:2]) for e in dag.get("edges", [])])
    n_nodes = dag.get("nodes", 0)
    canonical = json.dumps({"n_nodes": n_nodes, "edges": edges}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def verify_no_dag_overlap(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict]
) -> bool:
    """Verify no DAG structure overlap between splits."""
    train_hashes = {get_dag_hash(s) for s in train}
    val_hashes = {get_dag_hash(s) for s in val}
    test_hashes = {get_dag_hash(s) for s in test}

    overlap = (train_hashes & val_hashes) | (train_hashes & test_hashes) | (val_hashes & test_hashes)

    if overlap:
        print(f"[FAIL] DAG Overlap: {len(overlap)} structures shared")
        return False

    print(f"[PASS] No DAG overlap. Train={len(train_hashes)}, Val={len(val_hashes)}, Test={len(test_hashes)}")
    return True


def run_ks_test(
    train: List[Dict],
    test: List[Dict],
    threshold: float = 0.01
) -> List[Tuple[int, float]]:
    """Run KS test on variable distributions. Returns list of (var_idx, p_value) failures."""
    train_vars = defaultdict(list)
    test_vars = defaultdict(list)

    for s in train[:5000]:
        x = s.get("X_factual", [])
        for i, v in enumerate(x):
            train_vars[i].append(v)

    for s in test:
        x = s.get("X_factual", [])
        for i, v in enumerate(x):
            test_vars[i].append(v)

    failures = []
    for var_idx in sorted(train_vars.keys()):
        if var_idx not in test_vars or len(test_vars[var_idx]) < 10:
            continue

        stat, p = stats.ks_2samp(train_vars[var_idx], test_vars[var_idx])
        if p < threshold:
            failures.append((var_idx, p))

    return failures


# ============================================================================
# KS FIX (MICRO-RESAMPLING)
# ============================================================================

def micro_resample(
    test: List[Dict],
    pool: List[Dict],
    train: List[Dict],
    max_iterations: int = 5,
    threshold: float = 0.01
) -> List[Dict]:
    """
    Iteratively replace out-of-band test samples to fix KS failures.

    Args:
        test: Current test samples
        pool: Unused samples to draw replacements from
        train: Training samples for distribution reference
        max_iterations: Max replacement iterations
        threshold: KS test p-value threshold

    Returns:
        Fixed test samples
    """
    print(f"\n🔄 Running KS Micro-Resampling (max {max_iterations} iterations)...")

    current_test = list(test)
    current_pool = list(pool)

    # Get training distribution stats
    train_vars = defaultdict(list)
    for s in train[:5000]:
        x = s.get("X_factual", [])
        for i, v in enumerate(x):
            train_vars[i].append(v)

    for iteration in range(max_iterations):
        failures = run_ks_test(train, current_test, threshold)

        if not failures:
            print(f"   ✓ All variables pass KS test!")
            break

        print(f"   Iter {iteration + 1}: {len(failures)} KS failures")

        # Compute 2-98 percentile bounds for failing vars
        bounds = {}
        for var_idx, _ in failures:
            if var_idx in train_vars:
                bounds[var_idx] = (
                    np.percentile(train_vars[var_idx], 2),
                    np.percentile(train_vars[var_idx], 98)
                )

        # Identify out-of-band samples
        bad_indices = []
        for i, sample in enumerate(current_test):
            x = sample.get("X_factual", [])
            for var_idx, (lo, hi) in bounds.items():
                if var_idx < len(x):
                    if x[var_idx] < lo or x[var_idx] > hi:
                        bad_indices.append(i)
                        break

        print(f"   Found {len(bad_indices)} out-of-band samples")

        if not bad_indices:
            print("   No out-of-band samples found. Stopping.")
            break

        # Replace bad samples with pool samples
        replacements = 0
        for idx in bad_indices:
            if not current_pool:
                break

            # Find valid replacement
            for pool_idx, candidate in enumerate(current_pool):
                cx = candidate.get("X_factual", [])
                is_valid = True

                for var_idx, (lo, hi) in bounds.items():
                    if var_idx < len(cx):
                        if cx[var_idx] < lo or cx[var_idx] > hi:
                            is_valid = False
                            break

                if is_valid:
                    current_test[idx] = candidate
                    current_pool.pop(pool_idx)
                    replacements += 1
                    break

        print(f"   Replaced {replacements} samples")

        if replacements == 0:
            print("   No valid replacements found. Stopping.")
            break

    return current_test


# ============================================================================
# OPENAI SUBSETS
# ============================================================================

def load_cladder_test() -> pd.DataFrame:
    """Load CLadder test-balanced CSV from HuggingFace."""
    url = "https://huggingface.co/datasets/causal-nlp/CLadder/resolve/main/data/test-balanced-v1.5.csv"
    print(f"📥 Loading CLadder from {url}...")
    df = pd.read_csv(url)
    print(f"   Loaded {len(df)} samples")
    return df


def stratified_sample_cladder(df: pd.DataFrame, per_rung: int = 200) -> pd.DataFrame:
    """Stratified sample 200/200/200 from CLadder by rung."""
    print(f"\n🎯 Stratified sampling CLadder: {per_rung} per rung...")

    sampled = []
    for rung in [1, 2, 3]:
        rung_df = df[df["rung"] == rung]
        n = min(per_rung, len(rung_df))
        sampled.append(rung_df.sample(n=n, random_state=SEED))
        print(f"   Rung {rung}: {n}/{len(rung_df)}")

    return pd.concat(sampled, ignore_index=True)


def stratified_sample_p3sa(
    samples: List[Dict],
    target: int = 500
) -> List[Dict]:
    """Stratified sample P3SA by rung."""
    print(f"\n🎯 Stratified sampling P3SA: {target} total...")

    np.random.seed(SEED)

    rung_samples = defaultdict(list)
    for s in samples:
        rung_samples[s.get("query", {}).get("rung", 1)].append(s)

    total = len(samples)
    sampled = []

    for rung in [1, 2, 3]:
        pool = rung_samples[rung]
        n = int((len(pool) / total) * target)
        n = min(n, len(pool))

        indices = np.random.choice(len(pool), n, replace=False)
        sampled.extend([pool[i] for i in indices])
        print(f"   Rung {rung}: {n}/{len(pool)}")

    return sampled


def create_openai_subsets(
    test: List[Dict],
    output_dir: Path,
    ks_fix: bool = False,
    iterations: int = 5,
    train: Optional[List[Dict]] = None,
    pool: Optional[List[Dict]] = None
) -> None:
    """Create OpenAI evaluation subsets."""
    subsets_dir = output_dir / "openai_subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    # CLadder
    cladder_df = load_cladder_test()
    cladder_sampled = stratified_sample_cladder(cladder_df, per_rung=200)
    cladder_sampled.to_csv(subsets_dir / "cladder_600.csv", index=False)
    print(f"💾 Saved: cladder_600.csv ({len(cladder_sampled)} samples)")

    # P3SA
    p3sa_sampled = stratified_sample_p3sa(test, target=500)

    if ks_fix and train and pool:
        p3sa_sampled = micro_resample(
            p3sa_sampled,
            pool,
            train,
            max_iterations=iterations
        )

    save_jsonl(p3sa_sampled, subsets_dir / "p3sa_500.jsonl")
    print(f"💾 Saved: p3sa_500.jsonl ({len(p3sa_sampled)} samples)")

    # Save sample IDs
    cladder_ids = cladder_sampled["id"].tolist() if "id" in cladder_sampled.columns else list(range(len(cladder_sampled)))
    save_ids(cladder_ids, subsets_dir / "cladder_ids.json")

    p3sa_ids = [s.get("dag_id", i) for i, s in enumerate(p3sa_sampled)]
    save_ids(p3sa_ids, subsets_dir / "p3sa_ids.json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits with integrity checks"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for splits"
    )
    parser.add_argument(
        "--ks-fix",
        action="store_true",
        help="Run micro-resampler to fix KS failures"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Max iterations for KS micro-resampling"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CREATING DATASET SPLITS")
    print("=" * 60)

    # Load all raw shards
    raw_samples = []
    for shard in sorted(args.raw_dir.glob("*.jsonl")):
        print(f"Loading {shard.name}...")
        raw_samples.extend(load_jsonl(shard))

    print(f"\nTotal raw samples: {len(raw_samples)}")

    # Split by hash
    print("\n--- Hash-Based Splitting ---")
    train, val, test = split_by_hash(raw_samples)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Verify integrity
    print("\n--- Integrity Checks ---")
    dag_ok = verify_no_dag_overlap(train, val, test)

    ks_failures = run_ks_test(train, test)
    if ks_failures:
        print(f"[WARN] KS test: {len(ks_failures)} failures")
        for v, p in ks_failures[:5]:
            print(f"   Var {v}: p={p:.2e}")
    else:
        print("[PASS] KS test: all variables pass")

    # Save splits
    save_jsonl(train, args.output_dir / "train.jsonl")
    save_jsonl(val, args.output_dir / "val.jsonl")
    save_jsonl(test, args.output_dir / "test.jsonl")

    print(f"\n💾 Saved splits to {args.output_dir}")

    # Create OpenAI subsets
    print("\n--- Creating OpenAI Subsets ---")

    # Pool = unused samples (not in test)
    test_ids = {s.get("dag_id") for s in test}
    pool = [s for s in raw_samples if s.get("dag_id") not in test_ids]

    create_openai_subsets(
        test,
        args.output_dir,
        ks_fix=args.ks_fix,
        iterations=args.iterations,
        train=train,
        pool=pool
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
