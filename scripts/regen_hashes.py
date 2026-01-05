"""
Regenerate Hashes - Recompute SHA256 checksums for data files.

Usage:
    python scripts/regen_hashes.py           # Generate hashes
    python scripts/regen_hashes.py --verify  # Verify existing hashes
"""

import argparse
import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
HASHES_FILE = DATA_DIR / "metadata" / "hashes.json"

# Files to hash
TARGET_FILES = [
    "splits/train.jsonl",
    "splits/val.jsonl",
    "splits/test.jsonl",
    "splits/openai_subsets/cladder_600.csv",
    "splits/openai_subsets/p3sa_500.jsonl",
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_info(filepath: Path) -> dict:
    """Get file info including hash and size."""
    if not filepath.exists():
        return None

    return {
        "sha256": compute_sha256(filepath),
        "size_bytes": filepath.stat().st_size,
        "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2)
    }


def regenerate_hashes() -> dict:
    """Regenerate hashes for all target files."""
    hashes = {}

    for rel_path in TARGET_FILES:
        filepath = DATA_DIR / rel_path
        info = get_file_info(filepath)

        if info:
            hashes[rel_path] = info
            print(f"✓ {rel_path}: {info['sha256'][:16]}...")
        else:
            print(f"✗ {rel_path}: NOT FOUND")

    return hashes


def save_hashes(hashes: dict) -> None:
    """Save hashes to JSON file."""
    HASHES_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)

    print(f"\n💾 Saved: {HASHES_FILE}")


def load_hashes() -> dict:
    """Load existing hashes from JSON file."""
    if not HASHES_FILE.exists():
        return {}

    with open(HASHES_FILE, "r") as f:
        return json.load(f)


def verify_hashes() -> bool:
    """Verify current files match stored hashes."""
    stored = load_hashes()

    if not stored:
        print("No stored hashes found. Run without --verify first.")
        return False

    all_ok = True

    for rel_path, stored_info in stored.items():
        filepath = DATA_DIR / rel_path
        current_info = get_file_info(filepath)

        if not current_info:
            print(f"✗ {rel_path}: FILE MISSING")
            all_ok = False
            continue

        if current_info["sha256"] == stored_info["sha256"]:
            print(f"✓ {rel_path}: MATCH")
        else:
            print(f"✗ {rel_path}: MISMATCH")
            print(f"    Expected: {stored_info['sha256'][:32]}...")
            print(f"    Got:      {current_info['sha256'][:32]}...")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Regenerate or verify data hashes")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing hashes instead of regenerating"
    )

    args = parser.parse_args()

    print("=" * 60)
    if args.verify:
        print("VERIFYING DATA HASHES")
    else:
        print("REGENERATING DATA HASHES")
    print("=" * 60)
    print()

    if args.verify:
        success = verify_hashes()
        print()
        if success:
            print("✓ All hashes verified successfully")
            exit(0)
        else:
            print("✗ Hash verification failed")
            exit(1)
    else:
        hashes = regenerate_hashes()
        save_hashes(hashes)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
