#!/usr/bin/env bash
# Sequential training phases + evaluation using repo-root PYTHONPATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH}"

PY=${PYTHON:-python3}

"${PY}" -m solar_forecasting.training.phase1
"${PY}" -m solar_forecasting.training.phase2a
"${PY}" -m solar_forecasting.training.phase2b
"${PY}" -m solar_forecasting.training.phase3
"${PY}" -m solar_forecasting.evaluation
