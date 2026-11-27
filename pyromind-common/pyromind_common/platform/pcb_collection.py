"""Read-only selection of PCB triplets from a mounted dataset."""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable


PCB_REPORT_IMAGES = ("defect.jpg", "diff.jpg", "gt.jpg")

SELECT_PCB_TRIPLETS = """
import hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
mount, raw, seed_root, priorities = sys.argv[1:5]
offset, filenames, holdout, fixed_rows = sys.argv[5:9]
reports_only = sys.argv[9:] == ["report-triplets"]
mount = Path(mount).resolve(strict=True)
def scoped(path):
    path = path.resolve(strict=True)
    if mount not in path.parents or (reports_only and
                                     mount / "datasets" not in path.parents):
        raise ValueError("Path is outside authorized Storage")
    return path
def media_path(ref, base):
    path = Path(ref)
    if path.is_absolute():
        return scoped(mount / path.relative_to("/workspace"))
    return scoped(base / path)
def digest(path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("An image exceeds 8 MiB")
    return hashlib.sha256(path.read_bytes()).hexdigest()
root = scoped(mount / raw.lstrip("/"))
if not reports_only and (root / "images").is_dir():
    root = scoped(root / "images")
rank = {v: i for i, v in enumerate(json.loads(priorities))}
if root != mount / seed_root.lstrip("/"):
    rank.clear()
fixed_rows = json.loads(fixed_rows)
offset, roles = int(offset), json.loads(filenames)
with os.scandir(root) as entries:
    names = sorted((e.name for e in entries if not e.name.startswith(".")
                    and e.is_dir(follow_symlinks=False)
                    and (not reports_only or e.name.startswith("report_"))),
                   key=lambda n: (rank.get(n, len(rank)), n))
fixed = scoped(mount / holdout.lstrip("/"))
excluded_ids = set()
heldout_paths = defaultdict(set)
def exclude(row, base):
    excluded_ids.add(row["id"])
    paths = [media_path(ref, base) for ref in row["images"]]
    for p in paths:
        excluded_ids.add(p.parent.name)
        heldout_paths[p.stat().st_size].add(p)
    return paths
with fixed.open() as lines:
    for line in lines:
        if line.strip():
            exclude(json.loads(line), fixed.parent)
for row in fixed_rows:
    paths = exclude(row, mount)
    signature = hashlib.sha256("".join(digest(p) for p in paths).encode()).hexdigest()
    if signature != row["triplet_sha256"]:
        raise ValueError("Fixed training validation images changed")
heldout_hashes = {}
def overlaps(path):
    size = path.stat().st_size
    if size not in heldout_paths:
        return False
    if size not in heldout_hashes:
        heldout_hashes[size] = {digest(p) for p in heldout_paths[size]}
    return digest(path) in heldout_hashes[size]
rows = []
skip_excluded = bool(fixed_rows) or reports_only
for name in (names if skip_excluded else names[offset:offset + 5]):
    if name in excluded_ids:
        if skip_excluded:
            continue
        raise ValueError("Selection overlaps the fixed Eval set")
    if reports_only and not all((root / name / f).is_file() for f in roles):
        continue
    paths = [scoped(root / name / f) for f in roles]
    if any(p.parent != root / name or not p.is_file() for p in paths):
        raise ValueError("A selected sample is missing a PCB triplet")
    if any(overlaps(p) for p in paths):
        if skip_excluded:
            continue
        raise ValueError("Selection overlaps the fixed Eval set")
    if skip_excluded and offset:
        offset -= 1
        continue
    rows.append({"id": name, "images": [
        "/workspace/" + p.relative_to(mount).as_posix() for p in paths],
        "image_sha256": [digest(p) for p in paths]})
    if len(rows) == 5:
        break
if len(rows) != 5:
    raise ValueError("Selected range has fewer than five eligible sample directories")
print(json.dumps(rows))
"""


@runtime_checkable
class PcbCollectionReader(Protocol):
    def select_pcb_groups(
        self, image_directory: str, validation_uri: str, offset: int
    ) -> list[dict[str, Any]]: ...


def pcb_selection_command(
    mount: str, image_directory: str, validation_uri: str, offset: int
) -> list[str]:
    return [
        "python3",
        "-I",
        "-c",
        SELECT_PCB_TRIPLETS,
        mount,
        image_directory,
        image_directory,
        "[]",
        str(offset),
        json.dumps(PCB_REPORT_IMAGES),
        validation_uri,
        "[]",
        "report-triplets",
    ]
