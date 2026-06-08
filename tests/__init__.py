"""Test package bootstrap.

Adds ``src`` to ``sys.path`` so the suite can ``from prod_gemini_agent import ...``
without an editable install. This mirrors what ``conftest.py`` does for pytest,
but lives here so the suite also runs under the standard library:

    python3 -m unittest discover -s tests

The standard ``unittest`` runner imports this package before collecting any
test module, so the path is set up exactly once for every test.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
