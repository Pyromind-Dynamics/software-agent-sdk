"""Declarative bindings between dataset fields and label-config controls.

A ``FieldMap`` is the contract between a dataset and a label config: which image
file fills which ``<Image>`` object, which meta field feeds which control, and
which list of regions becomes rectangles. Both conversion directions read the
same map -- ``AVITrainToLabelStudioConverter`` writes predictions through it and
``LabelStudioToAVITrainConverter`` looks controls back up in it -- because a
binding that only works one way would import cleanly and then export nothing.

The built-in ``DEFAULT_FIELD_MAPS`` reproduce the pre-``FieldMap`` behaviour
exactly, so a caller that declares nothing keeps the old semantics. Declaring a
map is how a caller says "bind it this way instead": a renamed control, an
optional reference image, an explicit coordinate unit.

Only ``choices`` and ``textarea`` whole-sample controls are supported. The other
Label Studio control types are not modelled until a real dataset needs one;
guessing their value payloads would produce pre-annotations that do not render.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


# Maps upstream verdict/label spellings onto the quality choices used by the
# generated label configs ("defect"/"ok"). Both adapters share this table, but
# they treat a miss differently because their sources differ in kind:
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
    "fault": "defect",
    "faulty": "defect",
    "ok": "ok",
    "good": "ok",
    "pass": "ok",
    "false_positive": "ok",
    "false": "ok",
}

# Coordinate scales a binding can declare. "auto" infers the scale from the
# values' magnitude, which is what the converter did before units were
# declarable; the rest pin it, and an out-of-range value is then an error rather
# than something to guess about.
Unit = Literal["auto", "norm1000", "percent", "unit"]

SampleControlType = Literal["choices", "textarea"]

# Prefixes that turn a file name into a glob, matching what pathlib.glob accepts.
_GLOB_CHARS = ("*", "?", "[")


class ImageBinding(BaseModel):
    """One image file bound to one ``<Image>`` object in the label config."""

    field: str = Field(
        description="Task-data field name, which is also the <Image name=...>.",
    )
    source: str = Field(
        description=(
            "File name inside the sample directory, or a glob such as '*_cam.bmp'."
        ),
    )
    required: bool = Field(
        default=True,
        description=(
            "False binds the object only when the file exists; a sample without "
            "it renders an empty view instead of failing the conversion."
        ),
    )


class SampleFieldBinding(BaseModel):
    """One whole-sample meta field bound to one control."""

    field: str = Field(description="Meta key holding the value, e.g. 'quality'.")
    control: str = Field(description="Control name, i.e. its <... name=...>.")
    type: SampleControlType = Field(
        default="choices",
        description="'choices' writes <Choices>, 'textarea' writes <TextArea>.",
    )
    required: bool = Field(
        default=True,
        description=(
            "Whether the label config must declare this control. False is for a "
            "control only some datasets in the batch need."
        ),
    )
    synonyms: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Case-insensitive value normalisation table. Empty writes the value "
            "through untouched."
        ),
    )
    on_unmapped: Literal["keep", "drop"] = Field(
        default="drop",
        description=(
            "'keep' pre-annotates the raw value and reports it as unmapped; "
            "'drop' emits nothing so a human fills the control in."
        ),
    )


class RegionBinding(BaseModel):
    """One list of regions bound to a rectangle control and its text control."""

    source: str = Field(
        description="Meta key holding the region list, e.g. 'findings'.",
    )
    control: str = Field(description="Rectangle control name, a <RectangleLabels>.")
    label: str = Field(
        default="category",
        description="Key inside each region holding its label.",
    )
    required: bool = Field(
        default=True,
        description=(
            "Whether the label config must declare the rectangle control. False "
            "is for a source that only some samples carry coordinates for."
        ),
    )
    geometry: str | None = Field(
        default=None,
        description=(
            "Key inside each region holding its coordinates; None probes the "
            "known shapes."
        ),
    )
    unit: Unit = Field(
        default="auto",
        description=(
            "'auto' infers the coordinate scale from magnitude; the others pin "
            "it and reject out-of-range values."
        ),
    )
    label_synonyms: dict[str, str] = Field(
        default_factory=dict,
        description="Case-insensitive normalisation table for region labels.",
    )
    observation: str | None = Field(
        default=None,
        description="Key inside each region holding per-region text.",
    )
    observation_control: str | None = Field(
        default=None,
        description="TextArea control name for that text.",
    )

    @model_validator(mode="after")
    def _text_needs_both_ends(self) -> RegionBinding:
        """Per-region text needs the meta key *and* the control it lands on.

        Declaring one without the other is a mistake the converter cannot
        recover from -- it would either read a key nobody wrote or write to a
        control nobody declared -- so it is rejected where it is declared rather
        than discarded during conversion.
        """
        if bool(self.observation) != bool(self.observation_control):
            raise ValueError(
                "observation and observation_control must be declared together."
            )
        return self


class FieldMap(BaseModel):
    """Every binding one dataset/adapter pair declares."""

    version: int = Field(default=1, ge=1)
    primary_image: str | None = Field(
        default=None,
        description=(
            "Image object every control points its toName at. Defaults to the "
            "first image binding."
        ),
    )
    images: list[ImageBinding] = Field(default_factory=list)
    samples: list[SampleFieldBinding] = Field(default_factory=list)
    regions: list[RegionBinding] = Field(default_factory=list)

    def to_name_for(self, field: str | None = None) -> str:
        """Return the object a binding's control must point its toName at.

        Region and whole-sample controls all hang off one image object, which is
        what makes a box or a verdict appear beside the picture it describes.
        """
        if field is not None:
            return self.image_for(field).field
        if self.primary_image:
            return self.primary_image
        if self.images:
            return self.images[0].field
        raise ValueError(
            "FieldMap declares no image binding, so no control can name a toName."
        )

    def image_for(self, field: str) -> ImageBinding:
        for binding in self.images:
            if binding.field == field:
                return binding
        raise ValueError(f"Image field {field!r} is not bound in this field map.")

    @model_validator(mode="after")
    def _bindings_need_an_image(self) -> FieldMap:
        """Every control names a toName, so a map with controls needs an image."""
        if (self.samples or self.regions) and not self.images:
            raise ValueError(
                "a field map with sample or region bindings must also declare the "
                "image object their controls point at."
            )
        if self.primary_image and self.primary_image not in {
            binding.field for binding in self.images
        }:
            raise ValueError(
                f"primary_image {self.primary_image!r} is not one of the bound "
                f"image fields."
            )
        return self


# The pre-FieldMap behaviour of both adapters, spelled out as bindings. Passing
# no map reproduces exactly this, which is what keeps the change backward
# compatible: same control names, same toName, same three required images.
DEFAULT_FIELD_MAPS: dict[str, FieldMap] = {
    "avi_train": FieldMap(
        images=[
            ImageBinding(field="defect_image", source="defect.jpg"),
            ImageBinding(field="diff_image", source="diff.jpg"),
            ImageBinding(field="gt_image", source="gt.jpg"),
        ],
        samples=[
            SampleFieldBinding(
                field="quality",
                control="quality_label",
                type="choices",
                synonyms=_QUALITY_CHOICE_SYNONYMS,
                on_unmapped="keep",
            ),
        ],
        regions=[
            RegionBinding(
                source="findings",
                control="finding_category",
                label="category",
                observation="observation",
                observation_control="finding_observation",
            ),
        ],
    ),
    "aoi_export": FieldMap(
        images=[
            ImageBinding(field="defect_image", source="defect.jpg"),
            ImageBinding(field="diff_image", source="diff.jpg"),
            ImageBinding(field="gt_image", source="gt.jpg"),
        ],
        samples=[
            SampleFieldBinding(
                field="vlm_verdict",
                control="quality_label",
                type="choices",
                synonyms=_QUALITY_CHOICE_SYNONYMS,
                on_unmapped="drop",
            ),
            SampleFieldBinding(
                field="note",
                control="overall_note",
                type="textarea",
            ),
        ],
        regions=[
            # An AOI export is usually a whole-sample verdict with no
            # coordinates at all, so its region controls are only needed by the
            # exports that do carry them -- which is why they are not required.
            RegionBinding(
                source="findings",
                control="finding_category",
                label="category",
                required=False,
                observation="observation",
                observation_control="finding_observation",
            ),
            RegionBinding(
                source="boxes",
                control="finding_category",
                label="category",
                required=False,
                unit="norm1000",
            ),
        ],
    ),
}

# Export cannot know which adapter produced a project -- a caller may hand it any
# export JSON -- so the default reverse index spans every built-in control. Both
# adapters bind "quality_label" to the same field, so avi_train's map plus the
# one control only aoi_export has covers them all.
_EXPORT_BASE = DEFAULT_FIELD_MAPS["avi_train"]
EXPORT_FIELD_MAP = _EXPORT_BASE.model_copy(
    deep=True,
    update={
        "samples": [
            *_EXPORT_BASE.samples,
            SampleFieldBinding(field="note", control="overall_note", type="textarea"),
        ]
    },
)


def is_glob(source: str) -> bool:
    """Return whether a binding's source names a pattern rather than one file."""
    return any(char in source for char in _GLOB_CHARS)


def default_field_map(adapter: str) -> FieldMap:
    """Return the built-in map for an adapter, or raise for an unknown one."""
    try:
        return DEFAULT_FIELD_MAPS[adapter].model_copy(deep=True)
    except KeyError:
        raise ValueError(
            f"Unsupported adapter {adapter!r}; "
            f"expected one of {sorted(DEFAULT_FIELD_MAPS)}."
        ) from None


def parse_field_map(raw: Any, *, adapter: str) -> FieldMap:
    """Validate caller-supplied bindings and fill in the adapter's defaults.

    A declaration is *merged* over the adapter's default map rather than
    replacing it: the common case is adjusting one binding, and requiring the
    caller to restate the other five would turn a targeted change into a chance
    to mistype the ones that were already right. Section keys present in the
    declaration replace that whole section; a partial section is not merged
    entry by entry, because "which entry did they mean to replace" has no
    reliable answer when an entry is renamed.

    Args:
        raw: Parsed JSON of the declaration.
        adapter: Adapter whose defaults fill the sections left out.

    Raises:
        ValueError: If the declaration is not an object, names an unknown
            adapter, or its bindings fail validation.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"field map must be a JSON object, got {type(raw).__name__}.")
    unknown = sorted(set(raw) - set(FieldMap.model_fields))
    if unknown:
        raise ValueError(
            f"field map has unknown keys: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(FieldMap.model_fields))}."
        )

    merged = default_field_map(adapter).model_dump()
    merged.update(raw)
    try:
        return FieldMap.model_validate(merged)
    except ValidationError as exc:
        raise ValueError(f"invalid field map: {exc}") from exc


def field_map_digest(field_map: FieldMap) -> str:
    """Return a stable hash of a map's bindings.

    The digest is part of the project seed, so a project created with one map is
    never silently reused for a dataset converted with another -- the reuse check
    only looks at status, and would otherwise serve the old bindings forever.
    Sorted keys keep the digest stable across dict ordering.
    """
    payload = json.dumps(
        field_map.model_dump(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
