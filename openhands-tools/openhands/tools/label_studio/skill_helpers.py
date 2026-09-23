"""Shared helpers for validating Label Studio XML configurations.

The checks here mirror the ones Label Studio runs in
``label_studio/core/label_config.py:validate_label_config`` so that a config
rejected locally is also rejected by the server, and are deliberately stricter in
the two places where the server's check is looser than its own runtime:

- the server greps the raw text for ``name="..."``, so ``name = "x"`` slips past
  it; parsing the document catches that spelling.
- the server only requires ``toName`` to name *something*; a control whose
  ``toName`` points at another control is accepted and then renders with no data,
  so objects are required here.

The adapter contract is checked against the same ``FieldMap`` the converter
writes through. Its bindings decide which controls have to exist and what kind
they have to be, so a renamed control is fine as long as the declaration and the
XML agree -- what is rejected is a binding that cannot land anywhere.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from openhands.tools.label_studio.field_map import (
    DEFAULT_FIELD_MAPS,
    FieldMap,
)
from openhands.tools.label_studio.value_mapping import normalize_value_key


_OBJECT_TAGS = {"Image", "Text", "Audio", "Video", "HyperText", "TimeSeries"}
_CONTROL_TAG_TO_TYPE = {
    "Choices": "choices",
    "RectangleLabels": "rectanglelabels",
    "Labels": "labels",
    "TextArea": "textarea",
    "BrushLabels": "brushlabels",
    "PolygonLabels": "polygonlabels",
    "KeyPointLabels": "keypointlabels",
}
_TYPE_TO_TAG = {value: tag for tag, value in _CONTROL_TAG_TO_TYPE.items()}

# The element a control's own values live in. Every labelled control nests
# ``<Label>`` except ``<Choices>``, whose options are ``<Choice>``.
_CONTROL_CHILD_TAG = {"Choices": "Choice"}

# Objects a region control can legally draw on. Label Studio draws rectangles on
# images and video frames alike.
_MEDIA_TAGS = {"Image", "Video"}

# Label Studio treats a label config as a document rooted at <View>.
_ROOT_TAG = "View"

# The image object the built-in maps bind every control to. Kept as a module
# constant because it names the media object in the default bindings' errors.
REQUIRED_MEDIA_OBJECT = "defect_image"


class LabelConfigValidationError(ValueError):
    """Raised when a label_config.xml has structural problems."""


def required_controls(adapter: str) -> dict[str, str]:
    """Return the ``{control name: type}`` an adapter's predictions rely on.

    Only bindings marked required count: an adapter may declare a control for
    data only some of its samples carry, and demanding it of every config would
    force a control onto projects that never use it.
    """
    field_map = DEFAULT_FIELD_MAPS.get(adapter)
    if field_map is None:
        return {}
    controls = {
        binding.control: binding.type
        for binding in field_map.samples
        if binding.required
    }
    for binding in field_map.regions:
        if not binding.required:
            continue
        controls[binding.control] = "rectanglelabels"
        if binding.observation_control:
            controls[binding.observation_control] = "textarea"
    return controls


def extract_control_names(xml_content: str) -> set[str]:
    """Return all control ``from_name`` values defined in the XML."""
    return {control["from_name"] for control in validate_label_config_xml(xml_content)}


def extract_control_values(xml_content: str) -> dict[str, tuple[str, ...]]:
    """Return ``{control name: values it can render}`` for a label config.

    Label Studio draws a pre-annotation only when its value is one of the
    control's own choices/labels; anything else imports without error and shows
    nothing. The converter uses these to tell a value the config already accepts
    from a foreign spelling that needs normalising, so a declared synonym table
    cannot shadow a value that needs no translation.

    Controls without values (``<TextArea>``) map to an empty tuple.
    """
    root = ET.fromstring(xml_content)
    values: dict[str, tuple[str, ...]] = {}
    for elem in root.iter():
        if elem.tag not in _CONTROL_TAG_TO_TYPE:
            continue
        name = elem.get("name", "")
        if not name:
            continue
        found = [value for child in elem if (value := child.get("value"))]
        values[name] = tuple(found)
    return values


@dataclass(frozen=True)
class WidenedConfig:
    """A label config reconciled with the values the dataset actually carries."""

    xml: str
    added: dict[str, tuple[str, ...]] = field(default_factory=dict)
    missing_controls: tuple[str, ...] = ()


def widen_label_config(
    xml_content: str,
    values: Mapping[str, Iterable[str]],
) -> WidenedConfig:
    """Add data-carried values to the value lists a config declares.

    The config a caller writes is a template: it names the controls and the
    values it starts from. A value the data carries that the template does not
    list imports as a pre-annotation that draws nothing -- Label Studio compares
    the value against the control's own list and silently ignores a miss -- so
    the value list follows the data while the controls, their names, and their
    layout stay exactly as declared.

    Returns the reconciled config plus what changed: ``added`` per control, and
    ``missing_controls`` for values whose control the config does not declare at
    all, which no widening can place. The original text is returned unchanged
    when there is nothing to add, so an already-complete template is untouched.
    """
    candidates = {
        control: list(values_for_control)
        for control, values_for_control in values.items()
        if values_for_control
    }
    if not candidates:
        return WidenedConfig(xml_content)

    # Comments are preserved so the config the project runs on still reads like
    # the template it came from; only the added values differ.
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(xml_content, parser=parser)
    controls = {
        element.get("name"): element
        for element in root.iter()
        if element.tag in _CONTROL_TAG_TO_TYPE and element.get("name")
    }

    added: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    for control, raw_values in candidates.items():
        element = controls.get(control)
        if element is None:
            missing.append(control)
            continue
        declared = {
            normalize_value_key(child.get("value", ""))
            for child in element
            if child.get("value")
        }
        new_values: list[str] = []
        for value in raw_values:
            key = normalize_value_key(value)
            if not key or key in declared:
                continue
            declared.add(key)
            new_values.append(value)
        if not new_values:
            continue
        child_tag = _CONTROL_CHILD_TAG.get(element.tag, "Label")
        tail = element[-1].tail if len(element) else element.text
        for value in new_values:
            child = ET.SubElement(element, child_tag)
            child.set("value", value)
            child.tail = tail
        added[control] = tuple(new_values)

    if not added:
        return WidenedConfig(xml_content, {}, tuple(sorted(missing)))
    return WidenedConfig(
        ET.tostring(root, encoding="unicode"), added, tuple(sorted(missing))
    )


def validate_label_config_xml(
    xml_content: str,
    *,
    adapter: str | None = None,
    field_map: FieldMap | None = None,
) -> list[dict[str, str]]:
    """Validate a Label Studio label config and return control triplets.

    Returns a list of ``{"from_name", "to_name", "type"}`` dicts for every
    control tag found in the XML. ``to_name`` keeps the attribute verbatim, which
    may list several comma-separated objects.

    Pass ``adapter`` to also require the controls that adapter's built-in
    bindings pre-annotate against, or ``field_map`` to check a caller's own
    bindings instead. A declared map wins over the adapter's default.

    Raises:
        LabelConfigValidationError: If the XML is malformed or structurally
            inconsistent.
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        raise LabelConfigValidationError(f"Malformed XML: {exc}") from exc

    if root.tag != _ROOT_TAG:
        raise LabelConfigValidationError(
            f"Root element must be <{_ROOT_TAG}>, found <{root.tag}>."
        )

    # Label Studio requires every name="..." in the document to be unique, across
    # object and control tags alike, so the document is walked twice: once to
    # collect the names an object defines, once to resolve the controls against
    # them (a control may be declared before the object it references).
    objects: dict[str, str] = {}
    seen_names: set[str] = set()
    for elem in root.iter():
        name = elem.get("name", "")
        if not name:
            continue
        if name in seen_names:
            raise LabelConfigValidationError(f"Duplicate name: {name}")
        seen_names.add(name)
        if elem.tag in _OBJECT_TAGS:
            objects[name] = elem.tag

    controls: list[dict[str, str]] = []
    for elem in root.iter():
        tag = elem.tag
        if tag not in _CONTROL_TAG_TO_TYPE:
            continue
        name = elem.get("name", "")
        if not name:
            continue
        to_name = elem.get("toName", "")
        targets = [target.strip() for target in to_name.split(",") if target.strip()]
        if not targets:
            raise LabelConfigValidationError(f"Control '{name}' has no toName.")
        for target in targets:
            if target not in objects:
                raise LabelConfigValidationError(
                    f"Control '{name}' references unknown object '{target}'"
                )
        controls.append(
            {
                "from_name": name,
                "to_name": to_name,
                "type": _CONTROL_TAG_TO_TYPE[tag],
            }
        )

    bindings = field_map
    if bindings is None and adapter is not None:
        # An unknown adapter is left alone: the converter rejects it outright,
        # and duplicating that check would only produce two error messages.
        bindings = DEFAULT_FIELD_MAPS.get(adapter)
    if bindings is not None:
        check_field_map(bindings, objects, controls)
    return controls


def check_field_map(
    field_map: FieldMap,
    objects: dict[str, str],
    controls: list[dict[str, str]],
) -> None:
    """Reject a config these bindings could not write predictions through.

    Every problem names the binding that failed rather than a name that has to
    exist: the caller chose the names when they declared the map, and what is
    checked here is that each one reaches a control of the declared kind.
    """
    problems: list[str] = []
    primary = field_map.to_name_for() if field_map.images else None
    by_name = {control["from_name"]: control for control in controls}

    # Every control points its toName at the primary image, so that object has to
    # exist or nothing renders at all. The other bound images only feed task data:
    # a config that leaves one out shows fewer views rather than losing
    # pre-annotations, so it is checked only when it is declared.
    for image in field_map.images:
        tag = objects.get(image.field)
        if tag is None:
            if image.field == primary:
                problems.append(f'missing <Image name="{image.field}" value="$..."/>')
        elif tag not in _MEDIA_TAGS:
            problems.append(f"'{image.field}' is <{tag}> but must be <Image>")

    def check_control(name: str, expected_type: str, required: bool, bound_by: str):
        expected_tag = _TYPE_TO_TAG[expected_type]
        control = by_name.get(name)
        if control is None:
            if required:
                problems.append(
                    f'missing <{expected_tag} name="{name}" toName="{primary}"> '
                    f"({bound_by} is pre-annotated)"
                )
            return
        if control["type"] != expected_type:
            problems.append(
                f"'{name}' is <{_TYPE_TO_TAG[control['type']]}> "
                f"but must be <{expected_tag}>"
            )
            return
        targets = [
            target.strip() for target in control["to_name"].split(",") if target.strip()
        ]
        if primary is not None and primary not in targets:
            problems.append(
                f'\'{name}\' must have toName="{primary}", found "{control["to_name"]}"'
            )

    for binding in field_map.samples:
        check_control(
            binding.control,
            binding.type,
            binding.required,
            f"meta field '{binding.field}'",
        )

    for binding in field_map.regions:
        check_control(
            binding.control,
            "rectanglelabels",
            binding.required,
            f"regions in '{binding.source}'",
        )
        if binding.observation_control:
            # Per-region text rides on the same regions as the box, so it is
            # only required where the box is.
            check_control(
                binding.observation_control,
                "textarea",
                binding.required,
                f"'{binding.observation}' in '{binding.source}'",
            )

    if problems:
        raise LabelConfigValidationError(
            "label_config.xml does not match the prediction contract declared for "
            "this project: "
            + "; ".join(problems)
            + ". A config that omits or renames one of these imports with no "
            "pre-annotations and exports empty fields."
        )
