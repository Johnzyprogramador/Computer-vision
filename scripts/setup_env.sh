#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-.venv-full}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Could not find $PYTHON_BIN. Install Python 3.12 or run:"
  echo "  PYTHON_BIN=python3.11 bash scripts/setup_env.sh $ENV_DIR"
  exit 1
fi

if [ -e "$ENV_DIR" ]; then
  echo "$ENV_DIR already exists. Refusing to overwrite it."
  echo "Activate it with: source $ENV_DIR/bin/activate"
  exit 2
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/pip" install -e ".[all]"
"$ENV_DIR/bin/python" scripts/doctor.py
"$ENV_DIR/bin/pytest" -q

echo
echo "Full environment is ready."
echo "Activate it with: source $ENV_DIR/bin/activate"

