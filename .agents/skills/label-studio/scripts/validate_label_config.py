#!/usr/bin/env python3
"""Validate a Label Studio label_config.xml for structural correctness.

Usage: python validate_label_config.py <path-to-label_config.xml> [--adapter NAME]

Checks (mirroring label_studio/core/label_config.py:validate_label_config):
1. XML is well-formed and rooted at <View>.
2. Every name="..." in the document is unique, object and control tags alike.
3. Control tags (Choices, RectangleLabels, TextArea, ...) declare a toName that
   resolves to an object tag; a comma-separated toName is checked per target.
4. With --adapter, the controls that adapter's converter pre-annotates against
   exist, have the right type, and point at the media object.

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

REQUIRED_MEDIA_OBJECT = "defect_image"
REQUIRED_CONTROLS = {
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


def validate(
    xml_path: str, adapter: str | None = None
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

    if adapter:
        check_adapter_contract(adapter, objects, controls)
    return objects, controls


def check_adapter_contract(
    adapter: str,
    objects: dict[str, str],
    controls: list[dict[str, str]],
) -> None:
    required = REQUIRED_CONTROLS.get(adapter)
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
        expected_tag = TYPE_TO_TAG[expected_type]
        control = by_name.get(name)
        if control is None:
            problems.append(
                f'missing <{expected_tag} name="{name}" '
                f'toName="{REQUIRED_MEDIA_OBJECT}">'
            )
            continue
        if control["type"] != expected_type:
            problems.append(
                f"'{name}' is <{TYPE_TO_TAG[control['type']]}> "
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
        raise ValueError(
            f"label_config.xml does not match the '{adapter}' prediction contract: "
            + "; ".join(problems)
            + ". These control names, types, and toName targets are fixed by the "
            "converter; a config that renames them imports with no pre-annotations "
            "and exports empty fields."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_path", help="path to label_config.xml")
    parser.add_argument(
        "--adapter",
        choices=sorted(REQUIRED_CONTROLS),
        help="also require the controls this adapter pre-annotates against",
    )
    args = parser.parse_args()

    try:
        objects, controls = validate(args.xml_path, args.adapter)
    except (ET.ParseError, OSError, ValueError) as exc:
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
