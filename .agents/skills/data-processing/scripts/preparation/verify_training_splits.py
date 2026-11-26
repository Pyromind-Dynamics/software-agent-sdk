"""Read the Dataset exports and verify their SFT and source identity contracts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path


try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def regular_file(path: Path) -> Path:
    if not path.is_file() or path.resolve(strict=True) != path.absolute():
        raise ValueError("Expected a regular Storage file without symlinks")
    return path


def file_sha(path: Path) -> str:
    path = regular_file(path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def rows(path: Path):
    with regular_file(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name} line {number}: expected object")
            yield row


def media_path(value: str, base: Path, mount: Path | None) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError("Expected a local Storage media path")
    path = Path(value)
    if mount is None:
        mount = next(
            (
                Path(m)
                for m in ("/target-workspace", "/workspace")
                if base.is_relative_to(m)
            ),
            None,
        )
    if mount:
        for alias in ("/target-workspace", "/workspace"):
            if path.is_relative_to(alias):
                path = mount / path.relative_to(alias)
                break
        if not (base / path).resolve(strict=True).is_relative_to(mount):
            raise ValueError("Media resolves outside Storage")
    resolved = regular_file(base / path)
    return resolved


def media_identity(paths: list[Path]) -> str:
    return digest(sorted(file_sha(path) for path in paths))


def strings(row: dict, fields: tuple[str, ...]) -> set[str]:
    return {
        str(value)
        for field in fields
        if isinstance((value := row.get(field)), (str, int)) and str(value).strip()
    }


def jsonl_source_ids(path: Path, mount: Path | None) -> dict[str, str]:
    identities = {}
    seen = set()
    for row in rows(path):
        keys = strings(row, ("id", "task_id", "sample_id", "source_id"))
        texts = strings(row, ("text", "prompt", "user_prompt", "question", "input"))
        if isinstance(row.get("messages"), list):
            texts.update(
                m["content"]
                for m in row["messages"]
                if isinstance(m, dict)
                and m.get("role") == "user"
                and isinstance(m.get("content"), str)
            )
        aliases = {"id:" + item for item in keys} | {"text:" + item for item in texts}
        images = row.get("images")
        if isinstance(images, dict):
            images = list(images.values())
        if isinstance(images, list) and images:
            paths = [media_path(p, path.parent, mount) for p in images]
            aliases.add("media:" + media_identity(paths))
        if not aliases:
            raise ValueError(
                "JSONL sample needs stable identity, source text or images"
            )
        primary = next(
            (key for key in aliases if key.startswith("media:")),
            digest([sorted(keys), sorted(texts)]),
        )
        if primary in seen:
            raise ValueError("Duplicate source sample")
        seen.add(primary)
        for alias in aliases:
            if alias in identities and identities[alias] != primary:
                identities[alias] = ""
            else:
                identities[alias] = primary
    return identities


def source_ids(source: Path, mount: Path | None) -> tuple[str, dict[str, str]]:
    if source.is_dir() and (source / "train.jsonl").is_file():
        return "record_directory", jsonl_source_ids(source / "train.jsonl", mount)
    if source.is_file() and source.suffix.lower() == ".jsonl":
        return "jsonl", jsonl_source_ids(source, mount)
    if not source.is_dir():
        raise ValueError("Expected an image directory or a JSONL source")
    groups: dict[Path, list[Path]] = {}
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            groups.setdefault(
                path.parent if path.parent != source else path, []
            ).append(path)
    ids = {"media:" + media_identity(paths) for paths in groups.values()}
    if len(ids) != len(groups) or not ids:
        raise ValueError("Image source has empty or ambiguous sample identities")
    return "image_directory", {item: item for item in ids}


def decoded_image_sha(path: Path) -> str:
    if Image is None:
        raise ModuleNotFoundError("Pillow is required to decode training images")
    with Image.open(path) as image:
        image.load()
    return file_sha(path)


class DeferredImages:
    def __init__(self, mount: Path, limits: list[int]):
        self.mount = mount
        self.max_count, self.max_file, self.max_total = limits
        self.media: dict[str, dict[str, str]] = {}
        self.total = 0

    def collect(self, path: Path) -> str:
        relative = regular_file(path).relative_to(self.mount)
        with path.open("rb") as stream:
            raw = stream.read(self.max_file + 1)
        if not raw or len(raw) > self.max_file:
            raise ValueError("Replay image exceeds the file limit")
        sha = hashlib.sha256(raw).hexdigest()
        if sha not in self.media:
            self.total += len(raw)
            if len(self.media) >= self.max_count or self.total > self.max_total:
                raise ValueError("Replay images exceed the batch limit")
            self.media[sha] = {
                "path": relative.as_posix(),
                "data": base64.b64encode(raw).decode("ascii"),
            }
        return sha


def read_message_row(
    row: dict,
    base: Path,
    mount: Path | None,
    image_sha: Callable[[Path], str] = decoded_image_sha,
) -> set[str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Missing messages")
    expected_images = []
    placeholders = 0
    user_texts = set()
    roles = set()
    for item in messages:
        if not isinstance(item, dict) or item.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError("Invalid message role")
        role = item["role"]
        content = item.get("content")
        blocks = (
            [{"type": "text", "value": content}]
            if isinstance(content, str)
            else content
        )
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("Invalid message content")
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError("Invalid message block")
            kind = block.get("type")
            if kind in ("image", "image_url"):
                value = block.get("value", block.get("image_url", block.get("image")))
                if isinstance(value, dict):
                    value = value.get("url")
                expected_images.append(media_path(value, base, mount))
            elif kind == "text":
                value = block.get("value", block.get("text"))
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Empty message supervision or prompt")
                placeholders += value.count("<image>")
                roles.add(role)
                if role == "user":
                    user_texts.add(value)
            else:
                raise ValueError("Unsupported SFT content block")
    if not {"user", "assistant"} <= roles:
        raise ValueError("Missing user prompt or assistant supervision")
    images = row.get("images", [])
    if (
        not isinstance(images, list)
        or len(images) != len(expected_images) + placeholders
    ):
        raise ValueError("Message image placeholders do not match top-level images")
    paths = [media_path(item, base, mount) for item in images]
    if expected_images and expected_images != paths:
        raise ValueError("Message images do not match training media order")
    if paths:
        return {"media:" + digest(sorted(image_sha(path) for path in paths))}
    keys = strings(row, ("id", "source_id", "sample_id", "task_id"))
    return {"id:" + key for key in keys} | {"text:" + text for text in user_texts}


def verify(
    source: Path,
    output: Path,
    fixed_validation: Path | None,
    mount: Path | None = None,
    image_sha: Callable[[Path], str] = decoded_image_sha,
) -> dict:
    source = source.resolve(strict=True)
    output = output.resolve(strict=True)
    if output == source or output.is_relative_to(source):
        raise ValueError("Replay output must be separate from its input")
    kind, expected = source_ids(source, mount)
    assigned: dict[str, list[str]] = {"train": [], "val": []}
    fingerprints: dict[str, str] = {}
    seen: set[str] = set()
    for split in assigned:
        path = output / f"{split}_messages.jsonl"
        fingerprints[split] = file_sha(path)
        if split == "val" and fixed_validation:
            if fingerprints[split] != file_sha(fixed_validation):
                raise ValueError("Fixed train-time validation has changed")
        for number, row in enumerate(rows(path), 1):
            try:
                aliases = read_message_row(row, path.parent, mount, image_sha)
                matches = {expected[a] for a in aliases if expected.get(a)}
                if split == "val" and fixed_validation:
                    if matches:
                        raise ValueError("Fixed validation overlaps with this batch")
                    identity = digest(sorted(aliases))
                elif len(matches) == 1:
                    identity = matches.pop()
                else:
                    raise ValueError("Sample does not identify one source record")
                if identity in seen:
                    raise ValueError("Sample duplicated across training or validation")
            except (OSError, ValueError) as exc:
                raise ValueError(f"{path.name} line {number}: {exc}") from exc
            seen.add(identity)
            assigned[split].append(identity)
    required = set(expected.values()) - {""}
    consumed = assigned["train"] + ([] if fixed_validation else assigned["val"])
    if not assigned["train"] or Counter(consumed) != Counter(required):
        raise ValueError("Exported splits do not cover the input batch exactly once")
    return {
        "schema_version": 1,
        "source_kind": kind,
        "source_sha256": digest(sorted(required)),
        "train_identity_sha256": digest(sorted(assigned["train"])),
        "train_sha256": fingerprints["train"],
        "val_sha256": fingerprints["val"],
        "train_rows": len(assigned["train"]),
        "val_rows": len(assigned["val"]),
    }


def approved_members(source: Path, approved: Path) -> bool:
    source = source.resolve(strict=True)
    approved = approved.resolve(strict=True)
    if approved == source:
        return True
    if source.is_file() and approved.is_file():
        if source.suffix != ".jsonl" or approved.suffix != ".jsonl":
            return False
        pending = {digest(row) for row in rows(source)}
        if not pending:
            return False
        for row in rows(approved):
            pending.discard(digest(row))
            if not pending:
                return True
        return False
    if source.is_dir() and approved.is_dir():
        if (approved / "train.jsonl").is_file() and (source / "train.jsonl").is_file():
            return approved_members(source / "train.jsonl", approved / "train.jsonl")
        if (source / "train.jsonl").is_file():
            return False
        images = [p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        return bool(images) and all(
            (candidate := approved / p.relative_to(source)).is_file()
            and file_sha(candidate) == file_sha(p)
            for p in images
        )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-validation", type=Path)
    parser.add_argument("--approved-source", type=Path)
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--mount-root", type=Path)
    parser.add_argument(
        "--defer-image-decode", nargs=3, type=int, metavar=("COUNT", "FILE", "TOTAL")
    )
    args = parser.parse_args()
    deferred = None
    if args.defer_image_decode:
        if (
            not args.read_only
            or not args.mount_root
            or min(args.defer_image_decode) < 1
        ):
            parser.error("Deferred decoding requires read-only mount verification")
        if Image is None:
            deferred = DeferredImages(
                args.mount_root.resolve(strict=True), args.defer_image_decode
            )
    if args.mount_root:
        root = args.mount_root.resolve(strict=True)
        for path in (
            args.source_batch,
            args.output_dir,
            args.fixed_validation,
            args.approved_source,
        ):
            if path and not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("Verification input escaped Storage")
    receipt = verify(
        args.source_batch,
        args.output_dir,
        args.fixed_validation,
        args.mount_root,
        deferred.collect if deferred else decoded_image_sha,
    )
    if args.read_only:
        receipt["approved_source_member"] = approved_members(
            args.source_batch, args.approved_source
        )
        if deferred and deferred.media:
            receipt["deferred_media"] = deferred.media
        print(json.dumps(receipt, sort_keys=True))
        return
    path = args.output_dir / "verification.json"
    if path.exists() and json.loads(regular_file(path).read_bytes()) == receipt:
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True)


if __name__ == "__main__":
    main()
