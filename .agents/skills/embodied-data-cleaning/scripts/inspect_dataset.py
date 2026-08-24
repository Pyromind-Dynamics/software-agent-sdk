#!/usr/bin/env python3
"""Inspect an embodied dataset without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openhands.tools.embodied_data import inspect_dataset


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", type=Path)
parser.add_argument("--sample-limit", type=int, default=3)
args = parser.parse_args()

inspection = inspect_dataset(args.path, sample_limit=args.sample_limit)
print(json.dumps(inspection.model_dump(mode="json"), ensure_ascii=False, indent=2))
