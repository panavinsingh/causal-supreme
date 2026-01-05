"""
Pytest configuration and fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONSTANTS
# ============================================================================

KS_P = 0.01  # KS test p-value threshold
MIN_MOTIFS = 1  # Minimum required motifs (fork, collider, chain)

# ============================================================================
# FIXTURES
# ============================================================================

def load_jsonl(filepath):
    """Load JSONL file."""
    import json
    samples = []
    path = Path(__file__).parent.parent / filepath
    if not path.exists():
        pytest.skip(f"Data file not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples


@pytest.fixture(scope="session")
def train_data():
    """Load training data."""
    return load_jsonl("data/splits/train.jsonl")


@pytest.fixture(scope="session")
def val_data():
    """Load validation data."""
    return load_jsonl("data/splits/val.jsonl")


@pytest.fixture(scope="session")
def test_data():
    """Load test data."""
    return load_jsonl("data/splits/test.jsonl")


@pytest.fixture(scope="session")
def all_splits(train_data, val_data, test_data):
    """Return all splits as dict."""
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
