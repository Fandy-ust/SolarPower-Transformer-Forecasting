#!/usr/bin/env python3
"""Evaluate a trained Phase 3 checkpoint.

Compatibility wrapper. Main implementation lives in ``solar_forecasting.evaluation``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from solar_forecasting.evaluation import main


if __name__ == "__main__":
    main()
