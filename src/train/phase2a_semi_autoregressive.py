#!/usr/bin/env python3
"""Phase 2A: semi-autoregressive gap filling.

Compatibility wrapper. Main implementation lives in ``solar_forecasting.training.phase2a``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from solar_forecasting.training.phase2a import main


if __name__ == "__main__":
    main()
