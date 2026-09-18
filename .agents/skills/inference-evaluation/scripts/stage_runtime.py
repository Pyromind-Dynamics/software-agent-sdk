#!/usr/bin/env python3
"""Copy the bundled evaluation runtime into a conversation workspace."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = Path(__file__).with_name("evaluate_inference.py")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    print(output)


if __name__ == "__main__":
    main()
