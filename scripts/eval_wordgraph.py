#!/usr/bin/env python3
"""Thin wrapper around the package evaluation entrypoint."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.learn.wordgraph.interface.inference_eval import main


if __name__ == "__main__":
    main()
