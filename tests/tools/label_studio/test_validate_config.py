"""Tests for label config XML validation helpers."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from openhands.tools.label_studio.converter import AVITrainToLabelStudioConverter
from openhands.tools.label_studio.field_map import DEFAULT_FIELD_MAPS
from openhands.tools.label_studio.skill_helpers import (
    REQUIRED_MEDIA_OBJECT,
    LabelConfigValidationError,
    extract_control_names,
    required_controls,
    validate_label_config_xml,
)


SKILL_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".agents"
    / "skills"
    / "label-studio"
    / "scripts"
    / "validate_label_config.py"
)

VALID_XML = """\
<View>
  <Image name="defect_image" value="$defect_image"/>
  <Choices name="quality_label" toName="defect_image">
    <Choice value="ok"/>
    <Choice value="defect"/>
  </Choices>
  <RectangleLabels name="finding_category" toName="defect_image">
    <Label value="开路"/>
  </RectangleLabels>
  <TextArea name="finding_observation" toName="defect_image"/>
</View>
"""

# The whole-sample layout: no regions, so no RectangleLabels.
AOI_XML = """\
<View>
  <Image name="defect_image" value="$defect_image"/>
  <Choices name="quality_label" toName="defect_image">
    <Choice value="ok"/>
    <Choice value="defect"/>
  </Choices>
  <TextArea name="overall_note" toName="defect_image"/>
</View>
"""

# Every control any built-in map requires, so one document satisfies them all.
ALL_ADAPTERS_XML = """\
<View>
  <Image name="defect_image" value="$defect_image"/>
  <Choices name="quality_label" toName="defect_image">
    <Choice value="ok"/>
    <Choice value="defect"/>
  </Choices>
  <RectangleLabels name="finding_category" toName="defect_image">
    <Label value="开路"/>
  </RectangleLabels>
  <TextArea name="finding_observation" toName="defect_image"/>
  <TextArea name="overall_note" toName="defect_image"/>
</View>
"""


def _run_skill_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SKILL_SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def test_valid_xml_returns_controls():
    controls = validate_label_config_xml(VALID_XML)
    names = {c["from_name"] for c in controls}
    assert names == {"quality_label", "finding_category", "finding_observation"}
    assert extract_control_names(VALID_XML) == names


def test_malformed_xml_raises():
    with pytest.raises(LabelConfigValidationError, match="Malformed XML"):
        validate_label_config_xml("<View><unclosed>")


def test_root_must_be_a_view():
    """Label Studio's schema roots the document at <View>; anything else is invalid."""
    with pytest.raises(LabelConfigValidationError, match="Root element must be"):
        validate_label_config_xml(
            '<NotView><Image name="defect_image" value="$defect_image"/></NotView>'
        )


def test_duplicate_object_name_raises():
    xml = VALID_XML.replace(
        "</View>",
        '<Image name="defect_image" value="$other"/></View>',
    )
    with pytest.raises(LabelConfigValidationError, match="Duplicate name"):
        validate_label_config_xml(xml)


def test_object_and_control_cannot_share_a_name():
    """Label Studio checks uniqueness across every name in the document, not per tag.

    It greps the whole text for ``name="..."``, so an object and a control that
    spell the same name are rejected there too.
    """
    xml = (
        '<View><Image name="shared" value="$shared"/>'
        '<Choices name="shared" toName="shared">'
        '<Choice value="ok"/></Choices></View>'
    )
    with pytest.raises(LabelConfigValidationError, match="Duplicate name: shared"):
        validate_label_config_xml(xml)


def test_unknown_to_name_raises():
    xml = VALID_XML.replace('toName="defect_image"', 'toName="ghost"')
    with pytest.raises(LabelConfigValidationError, match="unknown object"):
        validate_label_config_xml(xml)


def test_to_name_accepts_a_comma_separated_list():
    """Label Studio splits toName on commas, so a multi-object control is legal.

    Treating the whole attribute as one name rejects configs the server accepts.
    """
    xml = (
        '<View><Image name="defect_image" value="$defect_image"/>'
        '<Image name="diff_image" value="$diff_image"/>'
        '<Choices name="quality_label" toName="defect_image, diff_image">'
        '<Choice value="ok"/></Choices></View>'
    )
    controls = validate_label_config_xml(xml)
    assert controls[0]["to_name"] == "defect_image, diff_image"


def test_control_may_be_declared_before_the_object_it_references():
    """Resolution must not depend on document order."""
    xml = (
        '<View><Choices name="quality_label" toName="defect_image">'
        '<Choice value="ok"/></Choices>'
        '<Image name="defect_image" value="$defect_image"/></View>'
    )
    assert extract_control_names(xml) == {"quality_label"}


class TestAdapterContract:
    """The converter writes predictions to fixed names, so create enforces them."""

    def test_avi_train_config_is_accepted(self):
        controls = validate_label_config_xml(VALID_XML, adapter="avi_train")
        assert {c["from_name"] for c in controls} >= set(required_controls("avi_train"))

    def test_aoi_export_config_is_accepted(self):
        controls = validate_label_config_xml(AOI_XML, adapter="aoi_export")
        assert {c["from_name"] for c in controls} >= set(
            required_controls("aoi_export")
        )

    def test_jsonl_config_needs_only_a_verdict_and_one_image(self):
        """A dataset row names its own images, so its built-in map has no fixed
        set to demand: one image and the verdict control are the whole contract.
        """
        xml = (
            '<View><Image name="defect_image" value="$defect_image"/>'
            '<Choices name="quality_label" toName="defect_image">'
            '<Choice value="ok"/><Choice value="defect"/></Choices></View>'
        )
        controls = validate_label_config_xml(xml, adapter="jsonl")
        assert {c["from_name"] for c in controls} == set(required_controls("jsonl"))

        with pytest.raises(LabelConfigValidationError, match="missing <Choices"):
            validate_label_config_xml(
                '<View><Image name="defect_image" value="$defect_image"/></View>',
                adapter="jsonl",
            )

    def test_avi_train_rejects_a_renamed_control(self):
        xml = VALID_XML.replace('name="quality_label"', 'name="verdict"')
        with pytest.raises(LabelConfigValidationError, match="prediction contract"):
            validate_label_config_xml(xml, adapter="avi_train")

    def test_avi_train_rejects_a_missing_region_control(self):
        """aoi_export's layout has no finding_category, so avi_train must reject it."""
        with pytest.raises(
            LabelConfigValidationError, match="missing <RectangleLabels"
        ):
            validate_label_config_xml(AOI_XML, adapter="avi_train")

    def test_aoi_export_rejects_a_missing_note_control(self):
        with pytest.raises(LabelConfigValidationError, match="missing <TextArea"):
            validate_label_config_xml(VALID_XML, adapter="aoi_export")

    def test_wrong_control_type_is_rejected(self):
        """A TextArea where Choices belongs would silently drop the verdict."""
        xml = VALID_XML.replace(
            """  <Choices name="quality_label" toName="defect_image">
    <Choice value="ok"/>
    <Choice value="defect"/>
  </Choices>""",
            '  <TextArea name="quality_label" toName="defect_image"/>',
        )
        assert '<TextArea name="quality_label"' in xml
        with pytest.raises(
            LabelConfigValidationError, match="'quality_label' is <TextArea>"
        ):
            validate_label_config_xml(xml, adapter="avi_train")

    def test_to_name_pointing_elsewhere_is_rejected(self):
        xml = VALID_XML.replace(
            '<TextArea name="finding_observation" toName="defect_image"/>',
            '<TextArea name="finding_observation" toName="other_object"/>'
            '<Text name="other_object" value="$other"/>',
        )
        with pytest.raises(
            LabelConfigValidationError,
            match="'finding_observation' must have toName",
        ):
            validate_label_config_xml(xml, adapter="avi_train")

    def test_a_missing_media_object_is_rejected(self):
        xml = VALID_XML.replace('name="defect_image"', 'name="picture"').replace(
            'toName="defect_image"', 'toName="picture"'
        )
        expected = f'missing <Image name="{REQUIRED_MEDIA_OBJECT}"'
        with pytest.raises(LabelConfigValidationError, match=expected):
            validate_label_config_xml(xml, adapter="avi_train")

    def test_unknown_adapter_is_left_to_the_converter(self):
        """An unknown adapter is the converter's error to raise, not this check's."""
        assert validate_label_config_xml(VALID_XML, adapter="bogus") != []

    def test_required_controls_matches_what_the_converter_writes(self):
        """Pin the contract to the converter, so the two cannot drift apart.

        The contract is what create enforces; the converter is what actually writes
        the predictions. If a name changes in one place only, create would accept a
        config whose control the converter never fills in.
        """
        avi = AVITrainToLabelStudioConverter(
            dataset_path="/datasets/pcb",
            storage_base_url="http://storage.example.com",
            storage_headers={},
            adapter="avi_train",
        )
        predictions = avi._build_predictions(
            {
                "quality": "defect",
                "findings": [
                    {
                        "category": "开路",
                        "observation": "断线",
                        "bbox": {
                            "x_min_norm": 1,
                            "y_min_norm": 2,
                            "x_max_norm": 3,
                            "y_max_norm": 4,
                        },
                    }
                ],
            }
        )
        assert predictions is not None
        written = {
            result["from_name"]: result["type"] for result in predictions["result"]
        }
        assert written == required_controls("avi_train")
        assert {r["to_name"] for r in predictions["result"]} == {REQUIRED_MEDIA_OBJECT}

        aoi = AVITrainToLabelStudioConverter(
            dataset_path="/datasets/aoi",
            storage_base_url="http://storage.example.com",
            storage_headers={},
            adapter="aoi_export",
        )
        aoi_predictions = aoi._build_aoi_predictions(
            {"vlm_verdict": "defect", "note": "划痕"}
        )
        assert aoi_predictions is not None
        assert {
            result["from_name"]: result["type"] for result in aoi_predictions["result"]
        } == required_controls("aoi_export")
        assert {r["to_name"] for r in aoi_predictions["result"]} == {
            REQUIRED_MEDIA_OBJECT
        }


class TestSkillScript:
    """The script runs in the agent sandbox, so it is a second implementation.

    It cannot import the tool package, so these tests are what keeps the two sets
    of rules from drifting apart.
    """

    def test_valid_xml(self, tmp_path: Path):
        xml_file = tmp_path / "label_config.xml"
        xml_file.write_text(VALID_XML, encoding="utf-8")
        result = _run_skill_script(str(xml_file))
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["valid"] is True
        assert len(data["controls"]) == 3

    def test_adapter_flag_enforces_the_contract(self, tmp_path: Path):
        xml_file = tmp_path / "label_config.xml"
        xml_file.write_text(AOI_XML, encoding="utf-8")

        ok = _run_skill_script(str(xml_file), "--adapter", "aoi_export")
        assert ok.returncode == 0

        rejected = _run_skill_script(str(xml_file), "--adapter", "avi_train")
        assert rejected.returncode == 1
        payload = json.loads(rejected.stdout)
        assert payload["valid"] is False
        assert "prediction contract" in payload["error"]

    def test_invalid_xml_exits_one(self, tmp_path: Path):
        xml_file = tmp_path / "label_config.xml"
        xml_file.write_text("<View><unclosed>", encoding="utf-8")
        result = _run_skill_script(str(xml_file))
        assert result.returncode == 1
        assert json.loads(result.stdout)["valid"] is False

    @pytest.mark.parametrize(
        ("label", "xml"),
        [
            ("valid", VALID_XML),
            ("malformed", "<View><unclosed>"),
            (
                "root",
                '<Other><Image name="defect_image" value="$defect_image"/></Other>',
            ),
            (
                "duplicate_across_tags",
                '<View><Image name="shared" value="$shared"/>'
                '<Choices name="shared" toName="shared">'
                '<Choice value="ok"/></Choices></View>',
            ),
            ("comma_to_name", None),  # replaced below; None keeps the table readable
        ],
    )
    def test_script_agrees_with_the_server_side_rules(
        self, tmp_path: Path, label: str, xml: str | None
    ):
        if xml is None:
            xml = (
                '<View><Image name="defect_image" value="$defect_image"/>'
                '<Image name="diff_image" value="$diff_image"/>'
                '<Choices name="quality_label" toName="defect_image, diff_image">'
                '<Choice value="ok"/></Choices></View>'
            )
        xml_file = tmp_path / f"{label}.xml"
        xml_file.write_text(xml, encoding="utf-8")

        try:
            validate_label_config_xml(xml)
            raises = False
        except LabelConfigValidationError:
            raises = True

        result = _run_skill_script(str(xml_file))
        assert (result.returncode != 0) is raises, (
            f"{label}: script exit={result.returncode} stdout={result.stdout}"
        )

    @pytest.mark.parametrize("adapter", sorted(DEFAULT_FIELD_MAPS))
    def test_script_knows_every_adapter_the_tool_defines(
        self, tmp_path: Path, adapter: str
    ):
        """The script carries its own copy of the bindings, so an adapter missing
        there would make ``--adapter`` unusable for a layout create accepts.
        """
        xml_file = tmp_path / "label_config.xml"
        xml_file.write_text(ALL_ADAPTERS_XML, encoding="utf-8")

        result = _run_skill_script(str(xml_file), "--adapter", adapter)
        assert "unknown adapter" not in result.stderr, result.stderr
        assert result.returncode == 0, result.stdout
