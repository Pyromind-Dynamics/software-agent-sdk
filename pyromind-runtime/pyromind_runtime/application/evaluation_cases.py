"""Lazy access to evaluation report cases and media."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pyromind_runtime.ports.storage import EvaluationStorage


MANIFEST_NAME = "evaluation_cases_manifest.json"
INDEX_NAME = "evaluation_cases_index.json"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_INDEX_BYTES = 32 * 1024 * 1024
MAX_CHUNK_BYTES = 64 * 1024 * 1024


class EvaluationCaseError(RuntimeError):
    """An evaluation case artifact or page request is invalid."""


class EvaluationCaseRequestError(EvaluationCaseError):
    """A case-page cursor supplied by the client is invalid."""


class EvaluationMediaNotFoundError(EvaluationCaseError):
    """A media id is not present in the run's generated case index."""


@dataclass(frozen=True, slots=True)
class EvaluationCasePage:
    items: list[dict[str, Any]]
    next_cursor: str | None
    total: int
    filtered_total: int
    revision: int | None = None


@dataclass(frozen=True, slots=True)
class EvaluationMedia:
    payload: bytes
    mime_type: str


def _json_object(payload: bytes, name: str, limit: int) -> dict[str, Any]:
    if len(payload) > limit:
        raise EvaluationCaseError(f"{name} exceeds the size limit")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationCaseError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise EvaluationCaseError(f"{name} is not a JSON object")
    return value


def _fingerprint(selection_key: str) -> str:
    return hashlib.sha256(selection_key.encode("utf-8")).hexdigest()[:16]


def _encode_cursor(offset: int, selection_key: str) -> str:
    payload = json.dumps(
        {"offset": offset, "fingerprint": _fingerprint(selection_key)},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(cursor: str | None, selection_key: str) -> int:
    if cursor is None:
        return 0
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationCaseRequestError("invalid case cursor") from exc
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("offset"), int)
        or value["offset"] < 0
        or value.get("fingerprint") != _fingerprint(selection_key)
    ):
        raise EvaluationCaseRequestError("invalid case cursor")
    return value["offset"]


class EvaluationCaseService:
    def __init__(self, storage: EvaluationStorage):
        self._storage = storage

    def list_page(
        self,
        *,
        output_dir: str,
        cursor: str | None,
        limit: int,
        query: str,
        allowed_case_ids: set[str] | None = None,
        selection_key: str = "",
    ) -> EvaluationCasePage:
        prefix = output_dir.rstrip("/")
        manifest = _json_object(
            self._storage.read(f"{prefix}/{MANIFEST_NAME}"),
            MANIFEST_NAME,
            MAX_MANIFEST_BYTES,
        )
        index = _json_object(
            self._storage.read(f"{prefix}/{INDEX_NAME}"),
            INDEX_NAME,
            MAX_INDEX_BYTES,
        )
        chunks = manifest.get("chunks")
        entries = index.get("cases")
        if not isinstance(chunks, list) or not isinstance(entries, list):
            raise EvaluationCaseError("evaluation case index is malformed")
        normalized_query = query.strip().casefold()
        selected = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("id"), str)
            and (allowed_case_ids is None or entry["id"] in allowed_case_ids)
            and (not normalized_query or normalized_query in entry["id"].casefold())
        ]
        cursor_key = f"{normalized_query}\0{selection_key}"
        offset = _decode_cursor(cursor, cursor_key)
        if offset > len(selected):
            raise EvaluationCaseRequestError("invalid case cursor")
        page_entries = selected[offset : offset + limit]
        loaded_chunks: dict[int, list[dict[str, Any]]] = {}
        items: list[dict[str, Any]] = []
        for entry in page_entries:
            chunk_index = entry.get("chunk")
            item_offset = entry.get("offset")
            if (
                not isinstance(chunk_index, int)
                or not isinstance(item_offset, int)
                or not 0 <= chunk_index < len(chunks)
            ):
                raise EvaluationCaseError("evaluation case index is malformed")
            if chunk_index not in loaded_chunks:
                chunk = chunks[chunk_index]
                if not isinstance(chunk, dict) or not isinstance(
                    chunk.get("name"), str
                ):
                    raise EvaluationCaseError("evaluation case manifest is malformed")
                chunk_name = chunk["name"]
                if not chunk_name.startswith("evaluation_cases/chunk-"):
                    raise EvaluationCaseError("evaluation case manifest is malformed")
                chunk_data = _json_object(
                    self._storage.read(f"{prefix}/{chunk_name}"),
                    chunk_name,
                    MAX_CHUNK_BYTES,
                )
                chunk_items = chunk_data.get("items")
                if not isinstance(chunk_items, list) or not all(
                    isinstance(item, dict) for item in chunk_items
                ):
                    raise EvaluationCaseError("evaluation case chunk is malformed")
                loaded_chunks[chunk_index] = chunk_items
            chunk_items = loaded_chunks[chunk_index]
            if not 0 <= item_offset < len(chunk_items):
                raise EvaluationCaseError("evaluation case index is malformed")
            item = chunk_items[item_offset]
            if item.get("id") != entry["id"]:
                raise EvaluationCaseError("evaluation case index is inconsistent")
            items.append(item)
        next_offset = offset + len(page_entries)
        return EvaluationCasePage(
            items=items,
            next_cursor=(
                _encode_cursor(next_offset, cursor_key)
                if next_offset < len(selected)
                else None
            ),
            total=len(entries),
            filtered_total=len(selected),
        )

    def read_media(self, *, output_dir: str, media_id: str) -> EvaluationMedia:
        prefix = output_dir.rstrip("/")
        index = _json_object(
            self._storage.read(f"{prefix}/{INDEX_NAME}"),
            INDEX_NAME,
            MAX_INDEX_BYTES,
        )
        entries = index.get("cases")
        if not isinstance(entries, list):
            raise EvaluationCaseError("evaluation case index is malformed")
        allowed_media = {
            media["media_id"]: media["mime_type"]
            for entry in entries
            if isinstance(entry, dict)
            for media in entry.get("media", [])
            if isinstance(media, dict)
            and isinstance(media.get("media_id"), str)
            and isinstance(media.get("mime_type"), str)
        }
        mime_type = allowed_media.get(media_id)
        if mime_type is None:
            raise EvaluationMediaNotFoundError("evaluation media not found")
        return EvaluationMedia(
            payload=self._storage.read(f"{prefix}/evaluation_media/{media_id}"),
            mime_type=mime_type,
        )
