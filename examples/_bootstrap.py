"""Shared helpers for the example scripts.

Importing this puts the repo root on ``sys.path`` so the examples run with a
plain ``python examples/0X_*.py`` on a fresh clone, no install required.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def header(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def section(title: str) -> None:
    print(f"\n--- {title} ---")
