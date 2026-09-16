"""Deterministic converters between PyroMind AVI data and Label Studio tasks.

Both converters read/write plain dicts and lists; neither calls the Label
Studio API. ``AVITrainToLabelStudioConverter`` reads user Storage and produces
Task Manifest batches, while ``LabelStudioToAVITrainConverter`` turns Label
Studio export JSON back into PyroMind sample dicts.

Two dataset adapters are supported for the forward conversion:
- ``avi_train``: sample directories containing ``meta_vlm.json`` with
  ``quality``/``findings`` pre-annotation fields.
- ``aoi_export``: AOI inspection export directories containing ``meta.json``
  with whole-sample ``vlm_verdict``/``label``/``note`` fields and no bboxes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from typing import Any, Protocol

import httpx

from openhands.tools.label_studio.models import ManifestBatch, ManifestData


logger = logging.getLogger(__name__)

BATCH_TASK_LIMIT = 500
BATCH_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB per batch file
_STORAGE_CONNECT_RETRIES = 2
_META_FETCH_CONCURRENCY = 8

_SUPPORTED_ADAPTERS = frozenset({"avi_train", "aoi_export"})

# Task data field name and source image file name for each rendered view.
_MEDIA_FIELDS = (
    ("defect_image", "defect.jpg"),
    ("diff_image", "diff.jpg"),
    ("gt_image", "gt.jpg"),
)

# Maps upstream verdict/label spellings onto the quality_label choices used by
# the generated label configs ("defect"/"ok"). Both adapters share this table,
# but they treat a miss differently because their sources differ in kind:
#   * aoi_export reads an *external* inspection system's vlm_verdict/label, so an
#     unrecognised spelling means we genuinely do not know -- no prediction is
#     emitted and the choice is left to the human annotator.
#   * avi_train reads meta_vlm.json from *our own* VLM pipeline, so an
#     unrecognised spelling is still our own verdict -- it is kept verbatim and
#     recorded in the manifest rather than silently dropped.
# The self-mapping entries ("defect"/"ok") are what keep an already-normalised
# meta value from being counted as a miss.
_QUALITY_CHOICE_SYNONYMS = {
    "defect": "defect",
    "true": "defect",
    "bad": "defect",
    "ng": "defect",
    "ok": "ok",
    "good": "ok",
    "pass": "ok",
    "false_positive": "ok",
    "false": "ok",
}


class ConversionError(ValueError):
    """Raised when dataset conversion fails."""


class MediaSigner(Protocol):
    """Resolves Storage object paths into browser-renderable media URLs."""

    def sign_many(self, paths: list[str]) -> dict[str, str]:
        """Return one media URL per requested object path."""
        ...


# A failed download quotes at most this much of the response body.
_DOWNLOAD_ERROR_BODY_LIMIT = 300
_DOWNLOAD_RETRIES = 2


class ConvertedManifest:
    """Holds converted task batches plus the manifest index."""

    def __init__(
        self,
        *,
        project_ref: str,
        dataset_path: str,
        converter_name: str,
        config_hash: str,
        batches: list[tuple[str, bytes, int]],
        total_tasks: int,
        unmapped_quality: tuple[str, ...] = (),
    ) -> None:
        self.project_ref = project_ref
        self.dataset_path = dataset_path
        self.converter_name = converter_name
        self.config_hash = config_hash
        self.batch_payloads = batches
        self.total_tasks = total_tasks
        # Verdicts kept verbatim because no synonym matched. Surfaced through the
        # manifest so a pre-annotation that will not render is visible, not silent.
        self.unmapped_quality = unmapped_quality

    def to_manifest_data(self) -> ManifestData:
        manifest_batches = []
        for index, (path, payload, task_count) in enumerate(self.batch_payloads):
            manifest_batches.append(
                ManifestBatch(
                    path=path,
                    task_count=task_count,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    index=index,
                )
            )
        return ManifestData(
            project_ref=self.project_ref,
            dataset_path=self.dataset_path,
            converter=self.converter_name,
            config_hash=self.config_hash,
            total_tasks=self.total_tasks,
            batches=manifest_batches,
            unmapped_quality=list(self.unmapped_quality),
        )


class AVITrainToLabelStudioConverter:
    """Convert AVI sample directories into Label Studio Task Manifest.

    Each sample directory contains ``defect.jpg``, ``diff.jpg``, ``gt.jpg``,
    and a meta JSON file. Image content is not copied; tasks carry a media URL
    for rendering plus the underlying object path. Meta fields become
    pre-annotations (predictions). ``adapter`` selects the meta layout:
    ``avi_train`` reads ``meta_vlm.json`` (``quality``/``findings``),
    ``aoi_export`` reads ``meta.json`` (whole-sample ``vlm_verdict``/``note``).
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        storage_base_url: str,
        storage_headers: dict[str, str],
        batch_task_limit: int = BATCH_TASK_LIMIT,
        batch_size_limit: int = BATCH_SIZE_LIMIT,
        config_hash: str = "",
        timeout: float = 30.0,
        adapter: str = "avi_train",
        media_signer: MediaSigner | None = None,
    ) -> None:
        if adapter not in _SUPPORTED_ADAPTERS:
            raise ConversionError(
                f"Unsupported adapter '{adapter}'; "
                f"expected one of {sorted(_SUPPORTED_ADAPTERS)}."
            )
        self._adapter = adapter
        self._meta_filename = (
            "meta.json" if adapter == "aoi_export" else "meta_vlm.json"
        )
        self._dataset_path = dataset_path.rstrip("/")
        self._storage_base_url = storage_base_url.rstrip("/")
        self._storage_headers = dict(storage_headers)
        self._batch_task_limit = batch_task_limit
        self._batch_size_limit = batch_size_limit
        self._config_hash = config_hash
        self._timeout = timeout
        self._media_signer = media_signer
        # Quality spellings that matched no synonym. Only add() ever mutates this
        # set and convert() touches it once per sample from a thread pool; a lone
        # add() on a set is atomic under the GIL, so no lock is needed.
        self._unmapped_quality: set[str] = set()

    @property
    def unmapped_quality_values(self) -> tuple[str, ...]:
        """Raw quality values no synonym matched, sorted for stable output."""
        return tuple(sorted(self._unmapped_quality))

    def convert(self) -> ConvertedManifest:
        sample_dirs = self._list_sample_dirs()
        if not sample_dirs:
            raise ConversionError(
                f"No sample directories found under {self._dataset_path}. "
                f"Expected subdirectories containing {self._meta_filename}."
            )

        media_urls = self._resolve_media_urls(
            [
                f"{sample_dir}/{filename}"
                for sample_dir in sample_dirs
                for filename in ("defect.jpg", "diff.jpg", "gt.jpg")
            ]
        )

        tasks: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=_META_FETCH_CONCURRENCY) as pool:
            for task in pool.map(
                lambda sample_dir: self._build_task(sample_dir, media_urls),
                sample_dirs,
            ):
                if task is not None:
                    tasks.append(task)

        if not tasks:
            raise ConversionError(
                f"All sample directories were skipped (missing or invalid "
                f"{self._meta_filename}). Nothing to import."
            )

        batches = self._split_batches(tasks)
        return ConvertedManifest(
            project_ref="",
            dataset_path=self._dataset_path,
            converter_name=self._adapter,
            config_hash=self._config_hash,
            batches=batches,
            total_tasks=len(tasks),
            unmapped_quality=self.unmapped_quality_values,
        )

    def _list_sample_dirs(self) -> list[str]:
        """List direct subdirectories of the dataset path."""
        entries = self._list_entries(self._dataset_path)
        return [entry["path"] for entry in entries if entry.get("is_dir")]

    def _build_task(
        self,
        sample_dir: str,
        media_urls: dict[str, str],
    ) -> dict[str, Any] | None:
        meta = self._read_json_file(f"{sample_dir}/{self._meta_filename}")
        if meta is None:
            return None
        if not isinstance(meta, dict):
            return None

        data: dict[str, Any] = {
            "sample_id": str(meta.get("sample_id", PurePosixPath(sample_dir).name)),
        }
        for field, filename in _MEDIA_FIELDS:
            object_path = f"{sample_dir}/{filename}"
            url = media_urls.get(object_path)
            if not url:
                raise ConversionError(f"No media URL resolved for {object_path}")
            # The URL is what Label Studio renders; the path is what a later
            # refresh or export round-trip needs, since URLs are short-lived.
            data[field] = url
            data[f"{field}_path"] = object_path

        if self._adapter == "aoi_export":
            predictions = self._build_aoi_predictions(meta)
        else:
            predictions = self._build_predictions(meta)
        task: dict[str, Any] = {"data": data}
        if predictions is not None:
            task["predictions"] = [predictions]
        return task

    def _build_predictions(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        results: list[dict[str, Any]] = []

        quality = meta.get("quality")
        if quality:
            raw = str(quality)
            mapped = _QUALITY_CHOICE_SYNONYMS.get(raw.strip().lower())
            if mapped is None:
                # Keep our own verdict instead of dropping it, but record it: a
                # value outside the config's <Choice> list will not render, and
                # that failure has to be visible somewhere.
                self._unmapped_quality.add(raw)
                logger.warning(
                    "avi_train quality %r matched no synonym; written verbatim to "
                    "quality_label, where it may not render",
                    raw,
                )
            results.append(
                {
                    "from_name": "quality_label",
                    "to_name": "defect_image",
                    "type": "choices",
                    "value": {"choices": [mapped or raw]},
                }
            )

        for i, finding in enumerate(meta.get("findings", [])):
            if not isinstance(finding, dict):
                continue
            region_id = f"finding_{i + 1}"
            bbox = finding.get("bbox", {})
            if isinstance(bbox, dict) and bbox:
                x_min = float(bbox.get("x_min_norm", 0))
                y_min = float(bbox.get("y_min_norm", 0))
                x_max = float(bbox.get("x_max_norm", 0))
                y_max = float(bbox.get("y_max_norm", 0))
                x = round(x_min / 10.0, 2)
                y = round(y_min / 10.0, 2)
                w = round((x_max - x_min) / 10.0, 2)
                h = round((y_max - y_min) / 10.0, 2)
                category = str(finding.get("category", ""))
                if category:
                    results.append(
                        {
                            "id": region_id,
                            "from_name": "finding_category",
                            "to_name": "defect_image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "rectanglelabels": [category],
                            },
                        }
                    )
            observation = finding.get("observation", "")
            if observation:
                results.append(
                    {
                        "id": region_id,
                        "from_name": "finding_observation",
                        "to_name": "defect_image",
                        "type": "textarea",
                        "value": {"text": [str(observation)]},
                    }
                )

        if not results:
            return None
        return {
            "model_version": str(meta.get("model_version", "vlm-v1")),
            "score": float(meta.get("score", 0.92)),
            "result": results,
        }

    def _build_aoi_predictions(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        verdict = str(meta.get("vlm_verdict") or meta.get("label") or "")
        verdict = verdict.strip().lower()
        results: list[dict[str, Any]] = []

        quality = _QUALITY_CHOICE_SYNONYMS.get(verdict)
        if quality:
            results.append(
                {
                    "from_name": "quality_label",
                    "to_name": "defect_image",
                    "type": "choices",
                    "value": {"choices": [quality]},
                }
            )

        note = str(meta.get("note") or "").strip()
        if note:
            results.append(
                {
                    "from_name": "overall_note",
                    "to_name": "defect_image",
                    "type": "textarea",
                    "value": {"text": [note]},
                }
            )

        if not results:
            return None
        return {
            "model_version": str(meta.get("model_version", "aoi-vlm-v1")),
            "score": float(meta.get("vlm_confidence", 0.95)),
            "result": results,
        }

    def _resolve_media_urls(self, paths: list[str]) -> dict[str, str]:
        """Resolve a media URL for every requested object path.

        With a media signer, one batch request covers the whole dataset. Without
        one, each object is signed individually through the Storage get_url API;
        those URLs land in task data, so they must stay least-privilege: one
        object per URL, never bucket-wide credentials.
        """
        if self._media_signer is not None:
            return self._media_signer.sign_many(paths)

        def resolve(path: str) -> tuple[str, str]:
            payload = self._storage_post("get_url", {"path": path})
            data = self._extract_api_data("get_url", payload)
            url = data.get("url") if isinstance(data, dict) else None
            if not isinstance(url, str) or not url.strip():
                raise ConversionError(f"Storage get_url returned no url for {path}")
            return path, url

        urls: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=_META_FETCH_CONCURRENCY) as pool:
            for path, url in pool.map(resolve, paths):
                urls[path] = url
        return urls

    def _split_batches(
        self, tasks: list[dict[str, Any]]
    ) -> list[tuple[str, bytes, int]]:
        batches: list[tuple[str, bytes, int]] = []
        current: list[dict[str, Any]] = []

        def flush() -> None:
            if not current:
                return
            payload = json.dumps(current, ensure_ascii=False).encode("utf-8")
            if len(payload) > self._batch_size_limit and len(current) > 1:
                mid = len(current) // 2
                first, second = current[:mid], current[mid:]
                current.clear()
                current.extend(first)
                flush()
                current.clear()
                current.extend(second)
                flush()
                return
            index = len(batches)
            filename = f"tasks-{index + 1:05d}.json"
            batches.append((filename, payload, len(current)))
            current.clear()

        for task in tasks:
            current.append(task)
            if len(current) >= self._batch_task_limit:
                flush()
        flush()
        return batches

    def _list_entries(self, path: str) -> list[dict[str, Any]]:
        payload = self._storage_post("file_list", {"path": path, "search": ""})
        data = self._extract_api_data("file_list", payload)
        raw = data.get("list")
        if not isinstance(raw, list):
            raise ConversionError(f"Storage file_list response missing list: {path}")
        entries: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            entry_type = str(item.get("type") or "").lower()
            item_path = item.get("path")
            if not item_path:
                continue
            entries.append(
                {
                    "path": str(item_path),
                    "name": str(item.get("name") or ""),
                    "is_dir": entry_type in ("folder", "directory", "dir"),
                }
            )
        return entries

    def _read_json_file(self, storage_path: str) -> Any | None:
        content = self._download_file(storage_path, max_bytes=1024 * 1024)
        if content is None:
            return None
        try:
            return json.loads(content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _download_file(self, storage_path: str, *, max_bytes: int) -> bytes | None:
        """Return file bytes, or None only when the object genuinely is absent.

        A transient or server-side failure raises instead: treating it as "no
        metadata file" silently drops samples from the import.
        """
        payload = self._storage_post("get_url", {"path": storage_path})
        data = self._extract_api_data("get_url", payload)
        url = data.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ConversionError(f"Storage get_url returned no url for {storage_path}")

        last_error: str = ""
        for attempt in range(1, _DOWNLOAD_RETRIES + 1):
            try:
                with httpx.stream(
                    "GET", url, timeout=self._timeout, follow_redirects=True
                ) as download:
                    if download.status_code == 404:
                        return None
                    if download.status_code >= 400:
                        body = download.read().decode("utf-8", errors="replace")
                        if download.status_code < 500:
                            raise ConversionError(
                                f"Storage download of {storage_path} returned HTTP "
                                f"{download.status_code}: "
                                f"{body[:_DOWNLOAD_ERROR_BODY_LIMIT]}"
                            )
                        last_error = (
                            f"Storage download of {storage_path} returned HTTP "
                            f"{download.status_code}"
                        )
                    else:
                        content = bytearray()
                        for chunk in download.iter_bytes():
                            if len(content) + len(chunk) > max_bytes:
                                raise ConversionError(
                                    f"Storage file exceeds {max_bytes} bytes: "
                                    f"{storage_path}"
                                )
                            content.extend(chunk)
                        return bytes(content)
            except httpx.RequestError as exc:
                last_error = (
                    f"Storage download of {storage_path} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if attempt < _DOWNLOAD_RETRIES:
                time.sleep(2 ** (attempt - 1))
        raise ConversionError(f"{last_error} (after {_DOWNLOAD_RETRIES} attempts)")

    def _storage_post(self, route: str, body: dict[str, Any]) -> dict[str, Any] | str:
        attempt = 0
        while True:
            try:
                response = httpx.post(
                    f"{self._storage_base_url}/{route}",
                    headers=self._storage_headers,
                    json=body,
                    timeout=self._timeout,
                )
            except httpx.ConnectError:
                if attempt >= _STORAGE_CONNECT_RETRIES:
                    raise ConversionError(
                        f"Storage {route} API unreachable after retries"
                    ) from None
                attempt += 1
                time.sleep(2 ** (attempt - 1))
                continue
            except httpx.RequestError as exc:
                raise ConversionError(f"Storage {route} API failed: {exc}") from exc
            try:
                payload: Any = response.json()
            except ValueError:
                return response.text[:500]
            return payload

    def _extract_api_data(
        self, route: str, payload: dict[str, Any] | str
    ) -> dict[str, Any]:
        if isinstance(payload, str):
            raise ConversionError(f"Storage {route} API error: {payload}")
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return payload["data"]
        return payload


class LabelStudioToAVITrainConverter:
    """Convert Label Studio export JSON back into PyroMind AVI Train samples."""

    def convert(self, export_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for task in export_data:
            sample = self._task_to_sample(task)
            if sample is not None:
                samples.append(sample)
        return samples

    def _task_to_sample(self, task: dict[str, Any]) -> dict[str, Any] | None:
        data = task.get("data", {})
        annotations = task.get("annotations", [])
        if not annotations:
            return None

        latest = annotations[-1]
        sample: dict[str, Any] = {
            "sample_id": data.get("sample_id", ""),
            # Prefer the stored object path so exports stay usable after the
            # rendered media URLs expire; fall back for tasks imported before
            # paths were recorded.
            "defect_image_path": data.get("defect_image_path")
            or data.get("defect_image", ""),
            "diff_image_path": data.get("diff_image_path")
            or data.get("diff_image", ""),
            "gt_image_path": data.get("gt_image_path") or data.get("gt_image", ""),
            "quality": None,
            "findings": [],
        }

        findings_by_region: dict[str, dict[str, Any]] = defaultdict(dict)
        for result in latest.get("result", []):
            from_name = result.get("from_name", "")
            region_id = result.get("id", "")
            value = result.get("value", {})
            if not isinstance(value, dict):
                continue

            if from_name == "quality_label":
                choices = value.get("choices", [])
                if choices:
                    sample["quality"] = choices[0]
            elif from_name == "finding_category":
                labels = value.get("rectanglelabels", [])
                if labels:
                    findings_by_region[region_id]["category"] = labels[0]
                x = float(value.get("x", 0))
                y = float(value.get("y", 0))
                w = float(value.get("width", 0))
                h = float(value.get("height", 0))
                findings_by_region[region_id]["bbox"] = {
                    "x_min_norm": round(x * 10, 1),
                    "y_min_norm": round(y * 10, 1),
                    "x_max_norm": round((x + w) * 10, 1),
                    "y_max_norm": round((y + h) * 10, 1),
                }
            elif from_name == "finding_observation":
                texts = value.get("text", [])
                if texts:
                    findings_by_region[region_id]["observation"] = texts[0]
            elif from_name == "overall_note":
                # aoi_export has no regions, so its free text is a whole-sample note
                # rather than a finding. Keyed the way meta.json spells it, which is
                # what the forward converter reads back; avi_train configs have no
                # such control, so their samples keep the shape they always had.
                texts = value.get("text", [])
                if texts:
                    sample["note"] = texts[0]

        sample["findings"] = list(findings_by_region.values())
        return sample
