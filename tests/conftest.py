"""Shared test fixtures.

We add ``src`` to ``sys.path`` so tests can ``from prod_gemini_agent import ...``
without an editable install. Saves contributors a step.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
