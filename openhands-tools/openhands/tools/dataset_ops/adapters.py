"""Dataset adapters shared by analysis and synthesis runtimes."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_EVAL_SPLITS = {"validation", "val", "test", "eval"}


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    sample_id: str
    source_path: Path
    value: dict[str, Any]
    media_paths: tuple[Path, ...] = ()
    split: str | None = None

    @property
    def is_evaluation(self) -> bool:
        return bool(self.split and self.split.lower() in _EVAL_SPLITS)


class DatasetAdapter(ABC):
    @abstractmethod
    def iter_records(self, source: Path) -> Iterator[DatasetRecord]: ...

    def training_records(self, source: Path) -> Iterator[DatasetRecord]:
        for record in self.iter_records(source):
            if not record.is_evaluation:
                yield record


def _record_split(value: dict[str, Any]) -> str | None:
    split = value.get("split")
    return split.strip().lower() if isinstance(split, str) and split.strip() else None


class JsonlTextAdapter(DatasetAdapter):
    def iter_records(self, source: Path) -> Iterator[DatasetRecord]:
        if not source.is_file():
            raise ValueError("jsonl_text source must be a JSONL file")
        with source.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {index + 1} must be an object")
                sample_id = str(value.get("id") or value.get("sample_id") or index)
                yield DatasetRecord(
                    sample_id=sample_id,
                    source_path=source,
                    value=value,
                    split=_record_split(value),
                )


class VisionManifestAdapter(JsonlTextAdapter):
    def iter_records(self, source: Path) -> Iterator[DatasetRecord]:
        for record in super().iter_records(source):
            raw_images = record.value.get("images")
            if raw_images is None:
                raw_images = record.value.get("image_path")
            if isinstance(raw_images, str):
                raw_images = [raw_images]
            if not isinstance(raw_images, list) or not raw_images:
                raise ValueError(f"sample {record.sample_id!r} has no images")
            media = tuple((source.parent / str(item)).resolve() for item in raw_images)
            yield DatasetRecord(
                sample_id=record.sample_id,
                source_path=record.source_path,
                value=record.value,
                media_paths=media,
                split=record.split,
            )


class AviPcbAdapter(DatasetAdapter):
    required_files = ("defect.jpg", "diff.jpg", "gt.jpg", "meta.json")

    def iter_records(self, source: Path) -> Iterator[DatasetRecord]:
        if not source.is_dir():
            raise ValueError("avi_pcb source must be a directory")
        candidates = [source] if self._is_sample(source) else sorted(source.iterdir())
        for sample_dir in candidates:
            if not sample_dir.is_dir() or not self._is_sample(sample_dir):
                continue
            meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError(f"{sample_dir.name}/meta.json must be an object")
            split = _record_split(meta)
            yield DatasetRecord(
                sample_id=str(meta.get("id") or sample_dir.name),
                source_path=sample_dir,
                value=meta,
                media_paths=tuple(
                    sample_dir / name for name in self.required_files[:3]
                ),
                split=split,
            )

    def _is_sample(self, path: Path) -> bool:
        return all((path / name).is_file() for name in self.required_files)


def create_adapter(name: str) -> DatasetAdapter:
    adapters: dict[str, type[DatasetAdapter]] = {
        "jsonl_text": JsonlTextAdapter,
        "vision_manifest": VisionManifestAdapter,
        "avi_pcb": AviPcbAdapter,
    }
    try:
        return adapters[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported dataset adapter: {name}") from exc
