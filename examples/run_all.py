"""Run every example script in order. Usage: ``python examples/run_all.py``."""

import glob
import os
import runpy

from _bootstrap import header

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    scripts = sorted(
        p for p in glob.glob(os.path.join(HERE, "[0-9][0-9]_*.py"))
    )
    for path in scripts:
        runpy.run_path(path, run_name="__main__")
    header(f"Done - ran {len(scripts)} example scripts")


if __name__ == "__main__":
    main()
