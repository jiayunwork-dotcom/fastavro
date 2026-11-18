#!/bin/bash
# Re-run the union branch regression tests on both backends.
#
# Usage:
#   ./run-union-branch-tests.sh [pytest args...]
#
# The Cython run builds the extensions in place. The pure Python run copies the
# package to a throwaway directory and removes the compiled modules so that the
# same test file exercises `_read_py`, `_write_py` and `_validation_py`.

set -e

PYTHON=${PYTHON:-python}
ROOT=$(cd "$(dirname "$0")" && pwd)

cd "$ROOT"

echo "== Cython extension backend =="
CC=${CC:-gcc} "$PYTHON" setup.py build_ext --inplace >/dev/null
"$PYTHON" -m pytest -q "$@" tests/test_union_branch.py

echo "== Pure Python backend =="
PURE_DIR=$(mktemp -d)
trap 'rm -rf "$PURE_DIR"' EXIT
cp -r fastavro "$PURE_DIR/fastavro"
find "$PURE_DIR/fastavro" -name '*.so' -delete
find "$PURE_DIR/fastavro" -name '__pycache__' -type d -exec rm -rf {} +
cp -r tests "$PURE_DIR/tests"
cp pytest.ini "$PURE_DIR/pytest.ini" 2>/dev/null || true
(cd "$PURE_DIR" && PYTHONPATH="$PURE_DIR" "$PYTHON" -m pytest -q "$@" tests/test_union_branch.py)
