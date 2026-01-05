#!/bin/bash
# CI validation script

set -e

echo "=== Running Integrity Tests ==="

cd "$(dirname "$0")/../.."

# Run pytest
python -m pytest src/validation/integrity_tests.py -v --tb=short

echo "=== All tests passed ==="
