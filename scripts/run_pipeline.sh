#!/usr/bin/env bash
# Sequential training phases + evaluation using repo-root PYTHONPATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PY=${PYTHON:-python3}

"${PY}" src/train/phase1_teacher_forcing.py
"${PY}" src/train/phase2a_semi_autoregressive.py
"${PY}" src/train/phase2b_scheduled_sampling.py
"${PY}" src/train/phase3_autoregressive_finetune.py
"${PY}" src/evaluate.py
