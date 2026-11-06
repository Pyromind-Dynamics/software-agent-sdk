"""Resolve media strictly from the accepted sample checkpoint."""

from pathlib import PurePosixPath


def checkpoint_media_path(document: dict, sample_index: int, media_index: int) -> str:
    preview = document.get("preview") or []
    if not preview or not 0 <= sample_index < 3 or not 0 <= media_index < 12:
        raise FileNotFoundError("sample media not found")
    outputs = preview[0].get("output_records", [])
    inputs = preview[0].get("input_records", [])
    records = inputs if isinstance(inputs, list) and inputs else outputs
    if not isinstance(records, list) or sample_index >= len(records):
        raise FileNotFoundError("sample media not found")
    record = records[sample_index]
    if not isinstance(record, dict):
        raise FileNotFoundError("sample media not found")
    sources = record.get("source_images")
    paths = (
        list(sources.values())
        if isinstance(sources, dict)
        else record.get("images", [])
    )
    if not isinstance(paths, list) or media_index >= len(paths):
        raise FileNotFoundError("sample media not found")
    value = paths[media_index]
    if (
        not isinstance(value, str)
        or not value
        or any(c in value for c in ("\\", ":", "?", "#"))
    ):
        raise ValueError("sample media must reference a Storage file")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ValueError("sample media must reference a Storage file")
    if not path.is_absolute():
        path = PurePosixPath(document["source_path"]).parent / path
    return str(path)
