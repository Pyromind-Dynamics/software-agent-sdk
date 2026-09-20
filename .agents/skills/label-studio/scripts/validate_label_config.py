#!/usr/bin/env python3
"""Validate a Label Studio label_config.xml for structural correctness.

Usage:
    python validate_label_config.py <path-to-label_config.xml> [--adapter NAME]
    python validate_label_config.py <path-to-label_config.xml> --adapter NAME \
        --field-map field_map.json

Checks (mirroring label_studio/core/label_config.py:validate_label_config):
1. XML is well-formed and rooted at <View>.
2. Every name="..." in the document is unique, object and control tags alike.
3. Control tags (Choices, RectangleLabels, TextArea, ...) declare a toName that
   resolves to an object tag; a comma-separated toName is checked per target.
4. With --adapter, the controls that adapter's bindings pre-annotate against
   exist, have the right type, and point at the primary image. With --field-map,
   the declared bindings are checked instead of the adapter's built-in ones.

Exit codes: 0 valid, 1 invalid, 2 usage error.
Prints a JSON summary with the extracted object and control tags.

Keep the rules here in step with
openhands-tools/openhands/tools/label_studio/skill_helpers.py, which runs the
same checks server-side; tests/tools/label_studio/test_validate_config.py pins the
two implementations together.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET


OBJECT_TAGS = {"Image", "Text", "Audio", "Video", "HyperText", "TimeSeries"}
CONTROL_TAG_TO_TYPE = {
    "Choices": "choices",
    "RectangleLabels": "rectanglelabels",
    "Labels": "labels",
    "TextArea": "textarea",
    "BrushLabels": "brushlabels",
    "PolygonLabels": "polygonlabels",
    "KeyPointLabels": "keypointlabels",
}
TYPE_TO_TAG = {value: tag for tag, value in CONTROL_TAG_TO_TYPE.items()}
ROOT_TAG = "View"

# Objects a region control can legally draw on.
MEDIA_TAGS = {"Image", "Video"}

# Mirrors the built-in bindings in
# openhands-tools/openhands/tools/label_studio/field_map.py. Only what the checks
# below read is kept: which controls an adapter binds, and which of them a config
# has to declare. Synonyms and unit handling live in the converter, not here.
DEFAULT_FIELD_MAPS = {
    "avi_train": {
        "images": [
            {"field": "defect_image"},
            {"field": "diff_image"},
            {"field": "gt_image"},
        ],
        "samples": [
            {
                "field": "quality",
                "control": "quality_label",
                "type": "choices",
                "required": True,
            }
        ],
        "regions": [
            {
                "source": "findings",
                "control": "finding_category",
                "label": "category",
                "required": True,
                "observation": "observation",
                "observation_control": "finding_observation",
            }
        ],
    },
    "aoi_export": {
        "images": [
            {"field": "defect_image"},
            {"field": "diff_image"},
            {"field": "gt_image"},
        ],
        "samples": [
            {
                "field": "vlm_verdict",
                "control": "quality_label",
                "type": "choices",
                "required": True,
            },
            {
                "field": "note",
                "control": "overall_note",
                "type": "textarea",
                "required": True,
            },
        ],
        # An AOI export usually carries no coordinates, so its region controls are
        # only needed by the exports that do -- hence required: false.
        "regions": [
            {
                "source": "findings",
                "control": "finding_category",
                "label": "category",
                "required": False,
                "observation": "observation",
                "observation_control": "finding_observation",
            },
            {
                "source": "boxes",
                "control": "finding_category",
                "label": "category",
                "required": False,
            },
        ],
    },
    # One JSON Lines object per sample, so only the first image is required: a
    # processed row usually names one picture rather than three.
    "jsonl": {
        "images": [
            {"field": "defect_image"},
            {"field": "diff_image"},
            {"field": "gt_image"},
        ],
        "samples": [
            {
                "field": "quality",
                "control": "quality_label",
                "type": "choices",
                "required": True,
            },
        ],
        "regions": [
            {
                "source": "findings",
                "control": "finding_category",
                "label": "category",
                "required": False,
                "observation": "observation",
                "observation_control": "finding_observation",
            },
            {
                "source": "boxes",
                "control": "finding_category",
                "label": "category",
                "required": False,
            },
        ],
    },
}


def merge_field_map(adapter: str, declared: object | None) -> dict:
    """Overlay a declared field map on an adapter's built-in bindings.

    Sections named in the declaration replace that section whole, matching what
    the server does; a partial section is not merged entry by entry, because
    "which entry did they mean to replace" has no reliable answer once an entry
    is renamed.
    """
    base = DEFAULT_FIELD_MAPS.get(adapter)
    if base is None:
        raise ValueError(f"unknown adapter '{adapter}'")
    merged = {key: [dict(entry) for entry in value] for key, value in base.items()}

    if declared is None:
        return merged
    if not isinstance(declared, dict):
        raise ValueError("field map must be a JSON object")
    unknown = sorted(set(declared) - set(merged))
    if unknown:
        raise ValueError(
            f"field map has unknown keys: {', '.join(unknown)}; "
            f"known keys: {', '.join(sorted(merged))}"
        )
    for key, value in declared.items():
        merged[key] = value
    return merged


def validate(
    xml_path: str, adapter: str | None = None, field_map: dict | None = None
) -> tuple[dict[str, str], list[dict[str, str]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != ROOT_TAG:
        raise ValueError(f"Root element must be <{ROOT_TAG}>, found <{root.tag}>.")

    objects: dict[str, str] = {}
    seen_names: set[str] = set()
    for elem in root.iter():
        name = elem.get("name", "")
        if not name:
            continue
        if name in seen_names:
            raise ValueError(f"Duplicate name: {name}")
        seen_names.add(name)
        if elem.tag in OBJECT_TAGS:
            objects[name] = elem.tag

    controls: list[dict[str, str]] = []
    for elem in root.iter():
        tag = elem.tag
        if tag not in CONTROL_TAG_TO_TYPE:
            continue
        name = elem.get("name", "")
        if not name:
            continue
        to_name = elem.get("toName", "")
        targets = [target.strip() for target in to_name.split(",") if target.strip()]
        if not targets:
            raise ValueError(f"Control '{name}' has no toName.")
        for target in targets:
            if target not in objects:
                raise ValueError(
                    f"Control '{name}' references unknown object '{target}'"
                )
        controls.append(
            {
                "from_name": name,
                "to_name": to_name,
                "type": CONTROL_TAG_TO_TYPE[tag],
            }
        )

    if field_map is not None:
        check_field_map(field_map, objects, controls)
    elif adapter:
        check_field_map(merge_field_map(adapter, None), objects, controls)
    return objects, controls


def check_field_map(
    field_map: dict,
    objects: dict[str, str],
    controls: list[dict[str, str]],
) -> None:
    """Reject a config these bindings could not write predictions through."""
    problems: list[str] = []
    images = field_map.get("images") or []
    primary = images[0]["field"] if images else None
    by_name = {control["from_name"]: control for control in controls}

    # Every control points its toName at the primary image, so that object has to
    # exist; the other bound images only feed task data.
    for image in images:
        tag = objects.get(image["field"])
        if tag is None:
            if image["field"] == primary:
                problems.append(
                    f'missing <Image name="{image["field"]}" value="$..."/>'
                )
        elif tag not in MEDIA_TAGS:
            problems.append(f"'{image['field']}' is <{tag}> but must be <Image>")

    def check_control(name: str, expected_type: str, required: bool, bound_by: str):
        expected_tag = TYPE_TO_TAG[expected_type]
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
                f"'{name}' is <{TYPE_TO_TAG[control['type']]}> "
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

    for binding in field_map.get("samples") or []:
        check_control(
            binding["control"],
            binding["type"],
            bool(binding.get("required", True)),
            f"meta field '{binding['field']}'",
        )

    for binding in field_map.get("regions") or []:
        check_control(
            binding["control"],
            "rectanglelabels",
            bool(binding.get("required", True)),
            f"regions in '{binding['source']}'",
        )
        observation_control = binding.get("observation_control")
        if observation_control:
            check_control(
                observation_control,
                "textarea",
                bool(binding.get("required", True)),
                f"'{binding.get('observation')}' in '{binding['source']}'",
            )

    if problems:
        raise ValueError(
            "label_config.xml does not match the prediction contract declared for "
            "this project: "
            + "; ".join(problems)
            + ". A config that omits or renames one of these imports with no "
            "pre-annotations and exports empty fields."
        )


def load_field_map(path: str, adapter: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        declared = json.load(handle)
    return merge_field_map(adapter, declared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_path", help="path to label_config.xml")
    parser.add_argument(
        "--adapter",
        help="also require the controls this adapter pre-annotates against",
    )
    parser.add_argument(
        "--field-map",
        dest="field_map_path",
        help="field map JSON to check against instead of the adapter's built-in one",
    )
    args = parser.parse_args()

    if args.adapter and args.adapter not in DEFAULT_FIELD_MAPS:
        parser.error(
            f"unknown adapter '{args.adapter}'; "
            f"expected one of {', '.join(sorted(DEFAULT_FIELD_MAPS))}"
        )
    if args.field_map_path and not args.adapter:
        parser.error("--field-map needs --adapter to name the bindings it extends")

    try:
        field_map = (
            load_field_map(args.field_map_path, args.adapter)
            if args.field_map_path
            else None
        )
        objects, controls = validate(args.xml_path, args.adapter, field_map)
    except (ET.ParseError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)
    print(
        json.dumps(
            {"valid": True, "objects": objects, "controls": controls},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
