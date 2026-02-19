#!/usr/bin/env bash
set -euo pipefail

# Node
if [[ -f pnpm-lock.yaml || -f package.json ]]; then
  corepack enable >/dev/null 2>&1 || true
  if command -v pnpm >/dev/null 2>&1; then
    pnpm install
  else
    npm ci
  fi
  exit 0
fi

# Python (poetry / uv / pip)
python -V
python -m pip install --upgrade pip

if [[ -f poetry.lock || -f pyproject.toml ]]; then
  python -m pip install -U poetry
  poetry install --with test || poetry install
  exit 0
fi

if [[ -f requirements.txt ]]; then
  python -m pip install -r requirements.txt
  exit 0
fi

echo "bootstrap: Fant ingen kjent dependency-fil (package.json/poetry.lock/requirements.txt)."
exit 1
