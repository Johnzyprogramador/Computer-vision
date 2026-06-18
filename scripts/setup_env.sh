#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-.venv-full}"

if [ -n "${PYTHON_BIN:-}" ]; then
  CANDIDATES=("$PYTHON_BIN")
else
  CANDIDATES=(python3.12 python3.11 python3.10)
fi

SELECTED_PYTHON=""
for candidate in "${CANDIDATES[@]}"; do
  if command -v "$candidate" >/dev/null 2>&1; then
    SELECTED_PYTHON="$candidate"
    break
  fi
done

if [ -z "$SELECTED_PYTHON" ]; then
  echo "Python 3.10, 3.11, or 3.12 is required."
  echo "Ubuntu example: sudo apt install python3.12 python3.12-venv"
  exit 1
fi

if [ -e "$ENV_DIR" ]; then
  echo "$ENV_DIR already exists. Refusing to overwrite it."
  echo "Activate it with: source $ENV_DIR/bin/activate"
  exit 2
fi

echo "Using $SELECTED_PYTHON ($("$SELECTED_PYTHON" --version))"
"$SELECTED_PYTHON" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/pip" install -e ".[all]"
"$ENV_DIR/bin/python" scripts/doctor.py
"$ENV_DIR/bin/pytest" -q

echo
echo "Full environment is ready."
echo "Activate it with: source $ENV_DIR/bin/activate"
