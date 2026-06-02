"""Make the repo root importable so ``import trustworthy_evals`` works under
``pytest`` on a fresh clone, with or without an editable install."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
