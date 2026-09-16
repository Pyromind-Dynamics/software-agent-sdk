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
"""

from __future__ import annotations

import xml.etree.ElementTree as ET


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

# Label Studio treats a label config as a document rooted at <View>.
_ROOT_TAG = "View"

# The dataset converter writes predictions against these exact names and reads
# the same names back on export, so a config that renames any of them imports
# cleanly and then silently drops every pre-annotation. Keep this table in step
# with converter.AVITrainToLabelStudioConverter; the test named
# test_required_controls_matches_converter fails when the two drift apart.
REQUIRED_MEDIA_OBJECT = "defect_image"
_REQUIRED_CONTROLS: dict[str, dict[str, str]] = {
    "avi_train": {
        "quality_label": "choices",
        "finding_category": "rectanglelabels",
        "finding_observation": "textarea",
    },
    "aoi_export": {
        "quality_label": "choices",
        "overall_note": "textarea",
    },
}


class LabelConfigValidationError(ValueError):
    """Raised when a label_config.xml has structural problems."""


def required_controls(adapter: str) -> dict[str, str]:
    """Return the ``{control name: type}`` an adapter's predictions rely on."""
    return dict(_REQUIRED_CONTROLS.get(adapter, {}))


def extract_control_names(xml_content: str) -> set[str]:
    """Return all control ``from_name`` values defined in the XML."""
    return {control["from_name"] for control in validate_label_config_xml(xml_content)}


def validate_label_config_xml(
    xml_content: str,
    *,
    adapter: str | None = None,
) -> list[dict[str, str]]:
    """Validate a Label Studio label config and return control triplets.

    Returns a list of ``{"from_name", "to_name", "type"}`` dicts for every
    control tag found in the XML. ``to_name`` keeps the attribute verbatim, which
    may list several comma-separated objects.

    Pass ``adapter`` to also require the controls that adapter's converter
    pre-annotates against.

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

    if adapter is not None:
        _check_adapter_contract(adapter, objects, controls)
    return controls


def _check_adapter_contract(
    adapter: str,
    objects: dict[str, str],
    controls: list[dict[str, str]],
) -> None:
    """Reject a config the adapter's converter could not write predictions to.

    An unknown adapter is left alone here: the converter rejects it outright, and
    duplicating that check would only produce two different error messages.
    """
    required = _REQUIRED_CONTROLS.get(adapter)
    if not required:
        return

    problems: list[str] = []
    media_tag = objects.get(REQUIRED_MEDIA_OBJECT)
    if media_tag is None:
        problems.append(f'missing <Image name="{REQUIRED_MEDIA_OBJECT}" value="$..."/>')
    elif media_tag != "Image":
        problems.append(
            f"'{REQUIRED_MEDIA_OBJECT}' is <{media_tag}> but must be <Image>"
        )

    by_name = {control["from_name"]: control for control in controls}
    for name, expected_type in required.items():
        expected_tag = _TYPE_TO_TAG[expected_type]
        control = by_name.get(name)
        if control is None:
            problems.append(
                f'missing <{expected_tag} name="{name}" '
                f'toName="{REQUIRED_MEDIA_OBJECT}">'
            )
            continue
        if control["type"] != expected_type:
            problems.append(
                f"'{name}' is <{_TYPE_TO_TAG[control['type']]}> "
                f"but must be <{expected_tag}>"
            )
            continue
        targets = [
            target.strip() for target in control["to_name"].split(",") if target.strip()
        ]
        if REQUIRED_MEDIA_OBJECT not in targets:
            problems.append(
                f"'{name}' must have toName=\"{REQUIRED_MEDIA_OBJECT}\", "
                f'found "{control["to_name"]}"'
            )

    if problems:
        raise LabelConfigValidationError(
            f"label_config.xml does not match the '{adapter}' prediction contract: "
            + "; ".join(problems)
            + ". These control names, types, and toName targets are fixed by the "
            "converter; a config that renames them imports with no pre-annotations "
            "and exports empty fields."
        )
