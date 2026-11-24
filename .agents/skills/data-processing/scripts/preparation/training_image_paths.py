#!/usr/bin/env python3
"""Check or materialize Storage paths for the flat Vision training loader."""

import argparse
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path


def training_image_record(
    row: dict,
    field: str,
    *,
    source_root: Path | None = None,
    mount: Path = Path("/workspace"),
) -> dict:
    """Return a record ready for serialization by the Vision training exporter."""
    if source_root is not None and not source_root.is_relative_to(mount):
        raise ValueError("source root must be an absolute Storage mount path")
    value = row.get(field) if isinstance(row, dict) else None
    paths = [value] if isinstance(value, str) else value
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{field} must contain an image path or a list of paths")
    checked = []
    for ref in paths:
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError(f"invalid image reference in {field}")
        path = Path(ref)
        if not path.is_absolute():
            if source_root is None:
                raise ValueError(f"relative image path in training manifest: {ref}")
            path = source_root / path
        if ".." in path.parts and path.is_relative_to(mount):
            # Keep the mount alias after resolving parent segments.
            path = mount / path.resolve().relative_to(mount.resolve())
        if not path.is_relative_to(mount):
            raise ValueError(f"image path must be within {mount}: {path}")
        with path.open("rb") as image:
            if not image.read(1):
                raise ValueError(f"empty image file: {path}")
        checked.append(str(path))
    return {**row, field: checked[0] if isinstance(value, str) else checked}


def prepare_images(
    manifest: Path,
    field: str,
    *,
    source_root: Path | None = None,
    output: Path | None = None,
    mount: Path = Path("/workspace"),
) -> tuple[int, int]:
    if (source_root is None) != (output is None):
        raise ValueError(
            "supply --source-root and --output together to create a derivative"
        )
    if output is not None and output.resolve() == manifest.resolve():
        raise ValueError("output must differ from the input manifest")

    rows = images = 0
    with manifest.open(encoding="utf-8") as source, ExitStack() as stack:
        target = None
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            target = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent, delete=False
            )
            stack.callback(Path(target.name).unlink, missing_ok=True)
            stack.enter_context(target)
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                record = training_image_record(
                    row, field, source_root=source_root, mount=mount
                )
                if target is not None:
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
            except (ValueError, OSError) as exc:
                raise ValueError(f"{manifest}:{line_number}: {exc}") from exc
            rows += 1
            images += 1 if isinstance(record[field], str) else len(record[field])
        if not rows:
            raise ValueError(f"empty manifest: {manifest}")
        if target is not None and output is not None:
            target.close()
            os.replace(target.name, output)
    return rows, images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-field", required=True)
    parser.add_argument(
        "--source-root", type=Path, help="original base of relative image paths"
    )
    parser.add_argument(
        "--output", type=Path, help="write a separate training manifest"
    )
    args = parser.parse_args()
    try:
        rows, images = prepare_images(
            args.manifest,
            args.image_field,
            source_root=args.source_root,
            output=args.output,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(
        json.dumps(
            {
                "manifest": str(args.output or args.manifest),
                "rows": rows,
                "images": images,
            }
        )
    )


if __name__ == "__main__":
    main()
