"""Deterministic converters between PyroMind AVI data and Label Studio tasks.

Both converters read/write plain dicts and lists; neither calls the Label
Studio API. ``AVITrainToLabelStudioConverter`` reads user Storage and produces
Task Manifest batches, while ``LabelStudioToAVITrainConverter`` turns Label
Studio export JSON back into PyroMind sample dicts.

Three dataset adapters are supported for the forward conversion:
- ``avi_train``: sample directories containing ``meta_vlm.json`` with
  ``quality``/``findings`` pre-annotation fields.
- ``aoi_export``: AOI inspection export directories containing ``meta.json``
  with whole-sample ``vlm_verdict``/``label``/``note`` fields. Regions are
  pre-annotated too whenever the export carries coordinates.
- ``jsonl``: one JSON Lines object carrying both its own metadata and its image
  object paths, so a pipeline's output file is imported as written instead of
  being materialised into sample directories first.

Region geometry is accepted in every shape our producers emit (see
``_geometry_from``): a mismatch between the converter's expectation and what a
pipeline actually wrote is silent -- Label Studio renders no box and raises no
error -- so the converter reads the shapes it can and logs the ones it cannot.
A ``FieldMap`` binding may declare the coordinate unit; doing so replaces the
magnitude guess with a check, so an out-of-range value fails loudly.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from typing import Any, Protocol

import httpx

from openhands.tools.label_studio.field_map import (
    EXPORT_FIELD_MAP,
    FieldMap,
    ImageBinding,
    RegionBinding,
    SampleFieldBinding,
    default_field_map,
    is_glob,
)
from openhands.tools.label_studio.models import ManifestBatch, ManifestData


logger = logging.getLogger(__name__)

BATCH_TASK_LIMIT = 500
BATCH_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB per batch file
_STORAGE_CONNECT_RETRIES = 2
_META_FETCH_CONCURRENCY = 8

_SUPPORTED_ADAPTERS = frozenset({"avi_train", "aoi_export", "jsonl"})

# Adapters whose ``dataset_path`` names one JSON Lines object instead of a
# directory of sample directories.
FILE_ADAPTERS = frozenset({"jsonl"})

# The metadata file each directory adapter reads inside one sample directory.
_META_FILENAMES = {"avi_train": "meta_vlm.json", "aoi_export": "meta.json"}

# A dataset file is read into memory whole, so it is capped at the size one
# import is expected to handle; a larger export has to be split upstream.
DATASET_FILE_MAX_BYTES = 64 * 1024 * 1024

# Adapters whose coordinates are optional by design. An AOI export is a
# whole-sample verdict and most of them carry no boxes at all, so a region
# binding that matches nothing there is the documented normal case rather than
# a declaration that missed the data.
_REGION_SOURCE_OPTIONAL_ADAPTERS = frozenset({"aoi_export"})

# Where a region's geometry may live inside a finding. The first key that holds
# something readable wins; a finding may also carry x/y/width/height flat.
_REGION_GEOMETRY_KEYS = ("bbox", "boxes", "box", "value")

# Label Studio stores region coordinates as percentages of the image. When a
# binding declares no unit, the scale of an incoming candidate is inferred from
# its magnitude, which is what lets one parser read norm1000 corners, percent
# boxes, and unit-normalised boxes without being told which is which.
_UNIT_MAX = 1.0 + 1e-6
_PERCENT_MAX = 100.0 + 1e-6
_NORM1000_MAX = 1000.0 + 1e-6

_UNIT_FACTOR = {"unit": 100.0, "percent": 1.0, "norm1000": 0.1}
_UNIT_MAXIMUM = {
    "unit": _UNIT_MAX,
    "percent": _PERCENT_MAX,
    "norm1000": _NORM1000_MAX,
}


def _named_numbers(
    source: dict[str, Any], keys: tuple[str, str, str, str]
) -> tuple[float, float, float, float] | None:
    """Return the four named values as floats, or None when any is missing."""
    values: list[float] = []
    for key in keys:
        value = source.get(key)
        if value is None:
            return None
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None
    return values[0], values[1], values[2], values[3]


def _percent_geometry(
    x: float, y: float, width: float, height: float
) -> tuple[float, float, float, float] | None:
    """Clip one rectangle into percent space, or drop it when degenerate.

    An out-of-range box is clipped rather than dropped: a box that is partly off
    the image still tells the annotator where to look, and Label Studio would
    otherwise refuse to render the region at all.
    """
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    x = min(max(x, 0.0), 100.0)
    y = min(max(y, 0.0), 100.0)
    width = min(width, 100.0 - x)
    height = min(height, 100.0 - y)
    if width <= 0 or height <= 0:
        return None
    return round(x, 2), round(y, 2), round(width, 2), round(height, 2)


def _scaled_geometry(
    x: float,
    y: float,
    width: float,
    height: float,
    unit: str = "auto",
) -> tuple[float, float, float, float] | None:
    """Turn a rectangle in any coordinate scale into percent geometry.

    Producers are inconsistent about units -- the same field has arrived as
    norm1000, as percent, and as a 0-1 fraction -- and none of them announce
    which. With ``unit="auto"`` magnitude is the only signal available, and it is
    unambiguous as long as the box stays inside its own coordinate space.

    A binding that declares its unit replaces that guess with a check: a value
    outside the declared scale raises instead of being read at whichever scale
    its magnitude happened to suggest, because a silently misplaced box looks
    exactly like a correct one.
    """
    if unit != "auto":
        largest = max(abs(x), abs(y), abs(x + width), abs(y + height))
        if largest > _UNIT_MAXIMUM[unit]:
            raise ConversionError(
                f"Region coordinate {largest:g} is outside the declared "
                f"{unit!r} scale (max {_UNIT_MAXIMUM[unit]:g}). Fix the unit in "
                f"the field map, or set unit='auto' to infer it from magnitude."
            )
        factor = _UNIT_FACTOR[unit]
        return _percent_geometry(
            x * factor, y * factor, width * factor, height * factor
        )

    largest = max(abs(x), abs(y), abs(x + width), abs(y + height))
    if largest <= _UNIT_MAX:
        factor = 100.0
    elif largest <= _PERCENT_MAX:
        factor = 1.0
    elif largest <= _NORM1000_MAX:
        factor = 0.1
    else:
        return None
    return _percent_geometry(x * factor, y * factor, width * factor, height * factor)


def _geometry_from(
    candidate: Any,
    unit: str = "auto",
) -> tuple[float, float, float, float] | None:
    """Parse one geometry candidate into percent ``(x, y, width, height)``.

    Understood shapes, all of which have been seen in real datasets:

    - ``{"x_min_norm": 400, ...}`` -- the documented contract (norm1000 corners)
    - ``{"x": 40.0, "y": 38.0, "width": 20.0, "height": 24.0}`` -- percent
    - ``{"x": 0.4, "y": 0.38, ...}`` -- unit-normalised
    - ``[400, 380, 600, 620]`` -- norm1000 ``x1y1x2y2`` corners
    - ``{"x1": ..., "y1": ..., "x2": ..., "y2": ...}`` / ``x_min``/``y_min``

    ``unit`` is the scale a binding declared. Only the magnitude guess is
    replaced: the key names keep deciding whether the numbers are corners or a
    width/height pair. The ``*_norm`` keys state their own unit, so a declared
    unit overrides them rather than being overridden by them.
    """
    if isinstance(candidate, (list, tuple)):
        if len(candidate) < 4:
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in candidate[:4])
        except (TypeError, ValueError):
            return None
        return _scaled_geometry(x1, y1, x2 - x1, y2 - y1, unit)

    if not isinstance(candidate, dict) or not candidate:
        return None

    # A "*_norm" key states its own unit, so under "auto" its magnitude must not
    # be consulted to guess: norm1000 corners can be small enough to look like
    # percentages. A declared unit is applied to the numbers as written, so a
    # source that means norm1000 and a map that says norm1000 agree, and a map
    # that says anything else is reported rather than quietly reinterpreted.
    corners = _named_numbers(
        candidate, ("x_min_norm", "y_min_norm", "x_max_norm", "y_max_norm")
    )
    if corners is not None:
        x1, y1, x2, y2 = corners
        if unit == "auto":
            return _percent_geometry(
                x1 / 10.0, y1 / 10.0, (x2 - x1) / 10.0, (y2 - y1) / 10.0
            )
        return _scaled_geometry(x1, y1, x2 - x1, y2 - y1, unit)

    for keys in (
        ("x_min", "y_min", "x_max", "y_max"),
        ("x1", "y1", "x2", "y2"),
    ):
        corners = _named_numbers(candidate, keys)
        if corners is not None:
            x1, y1, x2, y2 = corners
            return _scaled_geometry(x1, y1, x2 - x1, y2 - y1, unit)

    for keys in (("x", "y", "width", "height"), ("x", "y", "w", "h")):
        box = _named_numbers(candidate, keys)
        if box is not None:
            return _scaled_geometry(*box, unit)
    return None


def _geometry_candidates(candidate: Any) -> list[Any]:
    """Split one geometry field into the boxes it holds.

    A field holds several boxes when every element is itself a box
    (``[[x1, y1, x2, y2], ...]`` or ``[{...}, {...}]``) rather than a coordinate
    (``[x1, y1, x2, y2]``). Readings that name one box keep reaching
    ``_geometry_from`` unchanged.
    """
    if (
        isinstance(candidate, (list, tuple))
        and candidate
        and all(isinstance(item, (list, tuple, dict)) for item in candidate)
    ):
        return list(candidate)
    return [candidate]


def _region_geometries(
    finding: dict[str, Any],
    binding: RegionBinding,
) -> list[tuple[float, float, float, float]]:
    """Return every percent geometry a finding carries, in order.

    One finding may name one box or several -- a defect seen in two places is
    still one finding -- and Label Studio can only render one rectangle per
    result, so each box becomes its own rectangle downstream.

    A declared ``geometry`` key is read first rather than instead of the
    documented ones: an adapter that rewraps a row can store the coordinates
    under a name other than the row-level field a binding was written against,
    and a rectangle that silently never renders reads exactly like a region with
    no coordinates.
    """
    keys: list[str] = [binding.geometry] if binding.geometry else []
    keys.extend(key for key in _REGION_GEOMETRY_KEYS if key not in keys)
    for key in keys:
        candidate = finding.get(key)
        if not candidate:
            continue
        geometries = [
            geometry
            for item in _geometry_candidates(candidate)
            if (geometry := _geometry_from(item, binding.unit)) is not None
        ]
        if geometries:
            return geometries
        logger.warning(
            "Finding carries a %r field the converter cannot read: %r",
            key,
            candidate,
        )
    geometry = _geometry_from(finding, binding.unit)
    return [] if geometry is None else [geometry]


class ConversionError(ValueError):
    """Raised when dataset conversion fails."""


def _row_sample_id(row: dict[str, Any], index: int) -> str:
    """Return a row's sample id, falling back to its one-based position."""
    for key in ("sample_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"line-{index + 1}"


def _row_path(row: dict[str, Any], source: str) -> str | None:
    """Return the non-empty string at a dotted path inside a row, or None."""
    current: Any = row
    for part in source.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    if isinstance(current, str) and current.strip():
        return current.strip()
    return None


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
        unmatched_regions: tuple[str, ...] = (),
        field_map_hash: str = "",
        field_map_path: str | None = None,
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
        # Region fields no sample carries: the binding names something the data
        # does not have, which otherwise shows up only as an empty editor.
        self.unmatched_regions = unmatched_regions
        self.field_map_hash = field_map_hash
        self.field_map_path = field_map_path

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
            field_map_hash=self.field_map_hash,
            field_map_path=self.field_map_path,
            total_tasks=self.total_tasks,
            batches=manifest_batches,
            unmapped_quality=list(self.unmapped_quality),
            unmatched_regions=list(self.unmatched_regions),
        )


class AVITrainToLabelStudioConverter:
    """Convert AVI sample directories into Label Studio Task Manifest.

    Each sample directory contains ``defect.jpg``, ``diff.jpg``, ``gt.jpg``,
    and a meta JSON file. Image content is not copied; tasks carry a media URL
    for rendering plus the underlying object path. Meta fields become
    pre-annotations (predictions). ``adapter`` selects the meta layout:
    ``avi_train`` reads ``meta_vlm.json`` (``quality``/``findings``),
    ``aoi_export`` reads ``meta.json`` (whole-sample ``vlm_verdict``/``note``),
    and ``jsonl`` reads one task per line of a JSON Lines file whose rows carry
    their own image paths.

    Where those values go is decided by ``field_map``, which defaults to the
    adapter's built-in bindings: the control names, image slots, and region
    source are declarations, not constants, so a caller can bind a renamed
    control, an optional fourth image, or a region list the adapter never
    looked at.
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
        field_map: FieldMap | None = None,
        field_map_hash: str = "",
        field_map_path: str | None = None,
        dataset_content: bytes | None = None,
    ) -> None:
        if adapter not in _SUPPORTED_ADAPTERS:
            raise ConversionError(
                f"Unsupported adapter '{adapter}'; "
                f"expected one of {sorted(_SUPPORTED_ADAPTERS)}."
            )
        self._adapter = adapter
        self._meta_filename = _META_FILENAMES.get(adapter, "")
        # A file dataset the caller already read, so it is not fetched twice.
        self._dataset_content = dataset_content
        self._field_map = (
            field_map if field_map is not None else default_field_map(adapter)
        )
        self._field_map_hash = field_map_hash
        self._field_map_path = field_map_path
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
        # Same, for the region fields the samples actually carry.
        self._region_source_hits: set[str] = set()

    @property
    def field_map(self) -> FieldMap:
        """Bindings this conversion writes through."""
        return self._field_map

    @property
    def unmapped_quality_values(self) -> tuple[str, ...]:
        """Raw quality values no synonym matched, sorted for stable output."""
        return tuple(sorted(self._unmapped_quality))

    @property
    def unmatched_region_sources(self) -> tuple[str, ...]:
        """Declared region fields that no sample carries, sorted.

        A binding naming a field the dataset never has produces no rectangles
        and no other symptom, so it is reported the same way an unmapped verdict
        is rather than left for the annotator to notice in an empty editor.
        """
        if self._adapter in _REGION_SOURCE_OPTIONAL_ADAPTERS:
            return ()
        declared = {binding.source for binding in self._field_map.regions}
        return tuple(sorted(declared - self._region_source_hits))

    def convert(self) -> ConvertedManifest:
        tasks = (
            self._tasks_from_jsonl()
            if self._adapter in FILE_ADAPTERS
            else self._tasks_from_directories()
        )
        if not tasks:
            raise ConversionError(
                f"Every sample in {self._dataset_path} was skipped: no readable "
                f"sample metadata. Nothing to import."
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
            unmatched_regions=self.unmatched_region_sources,
            field_map_hash=self._field_map_hash,
            field_map_path=self._field_map_path,
        )

    def _tasks_from_directories(self) -> list[dict[str, Any]]:
        """Build one task per sample directory, in parallel."""
        sample_dirs = self._list_sample_dirs()
        if not sample_dirs:
            raise ConversionError(
                f"No sample directories found under {self._dataset_path}. "
                f"Expected subdirectories containing {self._meta_filename}."
            )

        bound_files = self._bind_sample_images(sample_dirs)
        media_urls = self._resolve_media_urls(
            [path for files in bound_files.values() for path in files.values()]
        )

        tasks: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=_META_FETCH_CONCURRENCY) as pool:
            for task in pool.map(
                lambda sample_dir: self._build_directory_task(
                    sample_dir, bound_files[sample_dir], media_urls
                ),
                sample_dirs,
            ):
                if task is not None:
                    tasks.append(task)
        return tasks

    def _build_directory_task(
        self,
        sample_dir: str,
        bound_files: dict[str, str],
        media_urls: dict[str, str],
    ) -> dict[str, Any] | None:
        """Build one sample's task, or None when its meta file is unreadable."""
        meta = self._read_json_file(f"{sample_dir}/{self._meta_filename}")
        if not isinstance(meta, dict):
            return None
        return self._build_task(
            str(meta.get("sample_id", PurePosixPath(sample_dir).name)),
            meta,
            bound_files,
            media_urls,
        )

    def _tasks_from_jsonl(self) -> list[dict[str, Any]]:
        """Build one task per line of a JSON Lines dataset file."""
        rows = self._read_jsonl_rows()
        if not rows:
            raise ConversionError(f"{self._dataset_path} holds no samples.")

        bound_files = self._bind_row_images(rows)
        media_urls = self._resolve_media_urls(
            [path for files in bound_files.values() for path in files.values()]
        )
        return [
            self._build_task(
                _row_sample_id(row, index),
                row,
                bound_files[index],
                media_urls,
            )
            for index, row in enumerate(rows)
        ]

    def _read_jsonl_rows(self) -> list[dict[str, Any]]:
        """Parse the dataset file, one JSON object per line.

        A blank line is skipped rather than rejected -- hand-edited exports grow
        them -- but a line that is not a JSON object fails the whole conversion:
        silently dropping it would import a dataset that is quietly incomplete.
        """
        content = self._dataset_content
        if content is None:
            content = self._download_file(
                self._dataset_path, max_bytes=DATASET_FILE_MAX_BYTES
            )
        if content is None:
            raise ConversionError(
                f"Dataset file not found in Storage: {self._dataset_path}"
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConversionError(
                f"Dataset file is not UTF-8 text: {self._dataset_path}"
            ) from exc

        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversionError(
                    f"{self._dataset_path} line {line_number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ConversionError(
                    f"{self._dataset_path} line {line_number} is not a JSON object."
                )
            rows.append(row)
        return rows

    def _bind_row_images(self, rows: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
        """Resolve every row's bound image object paths.

        A row names its own images instead of having them listed from a sample
        directory: a binding's ``source`` is a dotted path into the row, so both
        a flat ``{"defect_image": "/datasets/0001/defect.jpg"}`` and a nested
        ``{"images": {"defect_image": ...}}`` are expressible without a second
        convention.
        """
        bound: dict[int, dict[str, str]] = {}
        for index, row in enumerate(rows):
            files: dict[str, str] = {}
            for binding in self._field_map.images:
                path = _row_path(row, binding.source)
                if path is None:
                    if binding.required:
                        raise ConversionError(
                            f"{self._dataset_path} line {index + 1} has no image "
                            f"path at {binding.source!r} for image field "
                            f"{binding.field!r}"
                        )
                    continue
                files[binding.field] = path
            bound[index] = files
        return bound

    def _list_sample_dirs(self) -> list[str]:
        """List direct subdirectories of the dataset path."""
        entries = self._list_entries(self._dataset_path)
        return [entry["path"] for entry in entries if entry.get("is_dir")]

    def _bind_sample_images(self, sample_dirs: list[str]) -> dict[str, dict[str, str]]:
        """Resolve every sample's bound image files to Storage object paths.

        A binding that names one required file is taken at its word and the path
        is built directly, which is exactly what the converter did before field
        maps existed. Only a binding naming a pattern, or one marked optional,
        needs the directory listing -- only then is "which file" or "is it even
        there" a question at all -- so the built-in maps keep the old single pass
        and do not pay for a listing per sample.
        """
        bindings = self._field_map.images
        needs_listing = any(
            is_glob(binding.source) or not binding.required for binding in bindings
        )
        bound: dict[str, dict[str, str]] = {}
        for sample_dir in sample_dirs:
            listing = self._sample_files(sample_dir) if needs_listing else {}
            files: dict[str, str] = {}
            for binding in bindings:
                path = self._bind_one_image(sample_dir, binding, listing)
                if path is None:
                    if binding.required:
                        raise ConversionError(
                            f"No file matching {binding.source!r} for image field "
                            f"{binding.field!r} in {sample_dir}"
                        )
                    continue
                files[binding.field] = path
            bound[sample_dir] = files
        return bound

    def _bind_one_image(
        self,
        sample_dir: str,
        binding: ImageBinding,
        listing: dict[str, str],
    ) -> str | None:
        """Return the object path one image binding resolves to, or None."""
        if is_glob(binding.source):
            matches = sorted(
                path
                for name, path in listing.items()
                if fnmatch.fnmatch(name, binding.source)
            )
            return matches[0] if matches else None
        if not binding.required:
            return listing.get(binding.source)
        return f"{sample_dir}/{binding.source}"

    def _sample_files(self, sample_dir: str) -> dict[str, str]:
        """Return ``{file name: object path}`` for a sample directory's files."""
        return {
            entry["name"]: entry["path"]
            for entry in self._list_entries(sample_dir)
            if entry.get("name") and not entry.get("is_dir")
        }

    def _build_task(
        self,
        sample_id: str,
        meta: dict[str, Any],
        bound_files: dict[str, str],
        media_urls: dict[str, str],
    ) -> dict[str, Any]:
        """Build one task from a sample's meta and its bound image paths."""
        data: dict[str, Any] = {"sample_id": sample_id}
        for binding in self._field_map.regions:
            if binding.source in meta:
                self._region_source_hits.add(binding.source)
        for binding in self._field_map.images:
            object_path = bound_files.get(binding.field)
            if object_path is None:
                # Only an optional binding can be absent here.
                continue
            url = media_urls.get(object_path)
            if not url:
                raise ConversionError(f"No media URL resolved for {object_path}")
            # The URL is what Label Studio renders; the path is what a later
            # refresh or export round-trip needs, since URLs are short-lived.
            data[binding.field] = url
            data[f"{binding.field}_path"] = object_path

        if self._adapter == "aoi_export":
            predictions = self._build_aoi_predictions(meta)
        else:
            predictions = self._build_predictions(meta)
        self._flatten_filter_values(data, predictions)
        task: dict[str, Any] = {"data": data}
        if predictions is not None:
            task["predictions"] = [predictions]
        return task

    def _flatten_filter_values(
        self, data: dict[str, Any], predictions: dict[str, Any] | None
    ) -> None:
        """Mirror the map's categorical values into the task's own data.

        Label Studio's Data Manager filters task data by its columns; a value
        that lives only inside a prediction's result JSON is reachable through
        unstructured text search, which cannot tell one defect category from
        another. Copying a categorical control's value into ``data`` under the
        control's own name turns it into a column the annotator can select tasks
        by.

        Only categorical results -- choices and the various label controls -- are
        copied; free text stays in the prediction where it belongs. A binding
        can keep one control out of the columns with ``filterable=False``.
        """
        if predictions is None:
            return
        opted_out = {
            binding.control
            for binding in (*self._field_map.samples, *self._field_map.regions)
            if not binding.filterable
        }
        for result in predictions["result"]:
            control = result["from_name"]
            if control in opted_out:
                continue
            result_type = result["type"]
            if result_type == "choices":
                values = result["value"].get("choices") or []
            elif result_type.endswith("labels"):
                values = result["value"].get(result_type) or []
            else:
                continue
            if not values:
                continue
            # The first value wins: a sample's primary category is its first
            # labelled region, and an image binding that shares the control's
            # name keeps the URL it already put there.
            data.setdefault(control, values[0])

    def _sample_values(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Return the whole-sample values this adapter's bindings read.

        The field names a map declares are shared, but where each adapter finds
        them is not: an aoi export carries its verdict under one of two spellings
        and a note, while an avi sample carries a quality. That extraction stays
        here rather than becoming a fallback chain in the map, so the declaration
        describes bindings instead of restating each source's quirks.
        """
        if self._adapter == "aoi_export":
            return {
                "vlm_verdict": meta.get("vlm_verdict") or meta.get("label"),
                "note": meta.get("note"),
            }
        if self._adapter in FILE_ADAPTERS:
            # A row is already the mapping a binding reads by field name, so the
            # declaration -- not this adapter -- decides where each value lives.
            return meta
        return {"quality": meta.get("quality")}

    def _sample_result(
        self,
        binding: SampleFieldBinding,
        values: dict[str, Any],
        to_name: str,
    ) -> dict[str, Any] | None:
        """Build the prediction one whole-sample binding produces, if any."""
        raw = values.get(binding.field)
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None

        mapped = text
        if binding.synonyms:
            # A binding only normalises when it declares a table; free text such
            # as a note has nothing to map from, so it is written through.
            mapped = binding.synonyms.get(text.lower())
            if mapped is None:
                if binding.on_unmapped == "drop":
                    # The source is a foreign system's verdict, so an unknown
                    # spelling means we do not know -- leave it to the annotator.
                    return None
                # Our own pipeline's verdict: keep it, but record it, because a
                # value outside the config's <Choice> list does not render and
                # that has to be visible somewhere.
                self._unmapped_quality.add(text)
                logger.warning(
                    "%s value %r matched no synonym; written verbatim to control "
                    "%r, where it may not render",
                    binding.field,
                    text,
                    binding.control,
                )
                mapped = text

        value = (
            {"choices": [mapped]} if binding.type == "choices" else {"text": [mapped]}
        )
        return {
            "from_name": binding.control,
            "to_name": to_name,
            "type": binding.type,
            "value": value,
        }

    def _build_predictions(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        results = self._whole_sample_results(meta)
        results.extend(self._region_results(meta))
        if not results:
            return None
        return {
            "model_version": str(meta.get("model_version", "vlm-v1")),
            "score": float(meta.get("score", 0.92)),
            "result": results,
        }

    def _build_aoi_predictions(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        results = self._whole_sample_results(meta)
        results.extend(self._region_results(meta))
        if not results:
            return None
        return {
            "model_version": str(meta.get("model_version", "aoi-vlm-v1")),
            "score": float(meta.get("vlm_confidence", 0.95)),
            "result": results,
        }

    def _whole_sample_results(self, meta: dict[str, Any]) -> list[dict[str, Any]]:
        """Pre-annotate every whole-sample control the map declares."""
        values = self._sample_values(meta)
        to_name = self._field_map.to_name_for()
        results: list[dict[str, Any]] = []
        for binding in self._field_map.samples:
            result = self._sample_result(binding, values, to_name)
            if result is not None:
                results.append(result)
        return results

    def _region_sources(self, meta: dict[str, Any]) -> dict[str, list[Any]]:
        """Return each region list an adapter can offer, keyed by meta field.

        Most AOI exports are a whole-sample verdict with no coordinates at all,
        so nothing is returned and the sample is annotated by hand. When
        coordinates do appear, an explicit ``findings`` list wins over the flat
        ``boxes`` plus the sample's category, which is why only one of the two is
        ever offered.

        A dataset row is read where its declaration says instead of only under
        these two names: a row that keeps its regions under ``regions`` is
        offered as written, so importing a pipeline's output needs a binding and
        not a rewrite. Bare boxes in such a list still get the row's sample-level
        category copied on, because a rectangle with no label does not render.
        """
        findings = meta.get("findings")
        sources: dict[str, list[Any]] = {
            "findings": [item for item in findings if isinstance(item, dict)]
            if isinstance(findings, list)
            else []
        }
        if sources["findings"]:
            return sources

        boxes = meta.get("boxes")
        category = str(meta.get("category") or meta.get("vlm_category") or "").strip()
        if isinstance(boxes, list):
            sources["boxes"] = [
                {"bbox": box, "category": category} for box in boxes if box
            ]
        if self._adapter in FILE_ADAPTERS:
            for binding in self._field_map.regions:
                value = meta.get(binding.source)
                if not isinstance(value, list) or not value:
                    continue
                sources.setdefault(
                    binding.source,
                    [
                        item
                        if isinstance(item, dict)
                        else {"bbox": item, "category": category}
                        for item in value
                        if item
                    ],
                )
        return sources

    def _region_results(self, meta: dict[str, Any]) -> list[dict[str, Any]]:
        """Pre-annotate every region control the map declares.

        A region needs geometry *and* a label. Label Studio draws a
        rectanglelabels prediction only when its label list is non-empty, so a
        region with coordinates but no label is skipped and logged rather than
        written as a box that cannot render.
        """
        sources = self._region_sources(meta)
        to_name = self._field_map.to_name_for()
        results: list[dict[str, Any]] = []
        numbered = 0
        for binding in self._field_map.regions:
            for finding in sources.get(binding.source) or []:
                if not isinstance(finding, dict):
                    continue
                numbered += 1
                region_id = f"finding_{numbered}"
                results.extend(
                    self._region_result(binding, finding, region_id, to_name)
                )
        return results

    def _region_result(
        self,
        binding: RegionBinding,
        finding: dict[str, Any],
        region_id: str,
        to_name: str,
    ) -> list[dict[str, Any]]:
        """Build the controls one region produces: its boxes and its text.

        A region naming several boxes becomes several rectangles, because Label
        Studio renders one rectangle per result. The note describes the whole
        finding, so every rectangle carries it; a region whose label is missing
        keeps its note and drops the rectangle that could not render.
        """
        results: list[dict[str, Any]] = []
        geometries = _region_geometries(finding, binding)
        label = str(finding.get(binding.label) or "").strip()
        observation = finding.get(binding.observation) if binding.observation else None
        if geometries and label:
            if binding.label_synonyms:
                label = binding.label_synonyms.get(label.lower(), label)
            for index, geometry in enumerate(geometries):
                box_id = region_id if index == 0 else f"{region_id}_{index + 1}"
                x, y, width, height = geometry
                results.append(
                    {
                        "id": box_id,
                        "from_name": binding.control,
                        "to_name": to_name,
                        "type": "rectanglelabels",
                        "value": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "rectanglelabels": [label],
                        },
                    }
                )
                if observation and binding.observation_control:
                    results.append(
                        {
                            "id": box_id,
                            "from_name": binding.observation_control,
                            "to_name": to_name,
                            "type": "textarea",
                            "value": {"text": [str(observation)]},
                        }
                    )
            return results

        if geometries:
            logger.warning(
                "Region %s has usable geometry but no %r label; Label Studio "
                "would render an unlabelled rectangle, so it is skipped",
                region_id,
                binding.label,
            )
        if observation and binding.observation_control:
            results.append(
                {
                    "id": region_id,
                    "from_name": binding.observation_control,
                    "to_name": to_name,
                    "type": "textarea",
                    "value": {"text": [str(observation)]},
                }
            )
        return results

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


def _result_value(control_type: str, value: dict[str, Any]) -> str | None:
    """Return the first value an export wrote for one whole-sample control."""
    written = value.get("choices" if control_type == "choices" else "text")
    if isinstance(written, list) and written:
        return str(written[0])
    return None


class LabelStudioToAVITrainConverter:
    """Convert Label Studio export JSON back into PyroMind AVI Train samples.

    The bindings are looked up by control name in the same ``FieldMap`` the
    forward conversion wrote through. Renaming a control in a declaration would
    otherwise import pre-annotations cleanly and then export empty fields, so
    both directions read one map rather than each hard-coding its own names.
    """

    def __init__(self, field_map: FieldMap | None = None) -> None:
        self._field_map = field_map if field_map is not None else EXPORT_FIELD_MAP

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
        }
        # Prefer the stored object path so exports stay usable after the rendered
        # media URLs expire; fall back for tasks imported before paths were
        # recorded. Every bound image gets a key, blank when the sample had none.
        for image in self._field_map.images:
            sample[f"{image.field}_path"] = data.get(f"{image.field}_path") or data.get(
                image.field, ""
            )

        region_bindings = {
            binding.control: binding for binding in self._field_map.regions
        }
        observation_bindings = {
            binding.observation_control: binding
            for binding in self._field_map.regions
            if binding.observation_control
        }
        sample_bindings = {
            binding.control: binding for binding in self._field_map.samples
        }
        # Regions land in the list their binding reads from, so a map pointed at
        # another meta key exports that key instead of a hard-coded "findings".
        grouped: dict[str, dict[str, dict[str, Any]]] = {
            binding.source: {} for binding in self._field_map.regions
        }

        for result in latest.get("result", []):
            from_name = result.get("from_name", "")
            region_id = result.get("id", "")
            value = result.get("value", {})
            if not isinstance(value, dict):
                continue

            binding = sample_bindings.get(from_name)
            if binding is not None:
                text = _result_value(binding.type, value)
                if text is not None:
                    sample[binding.field] = text
                continue

            region_binding = region_bindings.get(from_name)
            if region_binding is not None:
                regions = grouped.setdefault(region_binding.source, {})
                region = regions.setdefault(region_id, {})
                labels = value.get("rectanglelabels", [])
                if labels:
                    region[region_binding.label] = labels[0]
                x = float(value.get("x", 0))
                y = float(value.get("y", 0))
                w = float(value.get("width", 0))
                h = float(value.get("height", 0))
                region["bbox"] = {
                    "x_min_norm": round(x * 10, 1),
                    "y_min_norm": round(y * 10, 1),
                    "x_max_norm": round((x + w) * 10, 1),
                    "y_max_norm": round((y + h) * 10, 1),
                }
                continue

            observation_binding = observation_bindings.get(from_name)
            if observation_binding is not None and observation_binding.observation:
                texts = value.get("text", [])
                if texts:
                    regions = grouped.setdefault(observation_binding.source, {})
                    regions.setdefault(region_id, {})[
                        observation_binding.observation
                    ] = texts[0]

        for source, regions in grouped.items():
            sample[source] = list(regions.values())
        return sample
