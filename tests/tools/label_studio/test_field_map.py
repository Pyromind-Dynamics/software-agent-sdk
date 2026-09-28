"""Tests for declared prediction bindings (field maps).

A field map is what replaced the fixed control names: the same converter writes
through whichever bindings it is given, and export reads them back through the
same ones. These tests pin the three things that makes possible -- an optional
image, a renamed control, a declared coordinate unit -- plus the guarantee that
declaring nothing keeps the old behaviour.
"""

import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from openhands.tools.label_studio.converter import (
    AVITrainToLabelStudioConverter,
    ConversionError,
    LabelStudioToAVITrainConverter,
    _geometry_from,
)
from openhands.tools.label_studio.executor import _project_ref
from openhands.tools.label_studio.field_map import (
    DEFAULT_FIELD_MAPS,
    EXPORT_FIELD_MAP,
    default_field_map,
    field_map_digest,
    parse_field_map,
)
from openhands.tools.label_studio.skill_helpers import (
    LabelConfigValidationError,
    required_controls,
    validate_label_config_xml,
)


EXAMPLES = (
    Path(__file__).resolve().parents[3]
    / ".agents"
    / "skills"
    / "label-studio"
    / "references"
    / "examples"
)

SAMPLE_META = {
    "sample_id": "sample_001",
    "quality": "defect",
    "score": 0.95,
    "findings": [
        {
            "category": "开路",
            "observation": "线路断开",
            "bbox": {
                "x_min_norm": 100,
                "y_min_norm": 200,
                "x_max_norm": 400,
                "y_max_norm": 500,
            },
        }
    ],
}


def _converter(
    monkeypatch,
    sample_dirs,
    meta,
    *,
    files=None,
    resolve_media: bool = True,
    **overrides,
):
    converter = AVITrainToLabelStudioConverter(
        dataset_path="/datasets/pcb-001",
        storage_base_url="http://storage.example.com",
        storage_headers={},
        **overrides,
    )
    monkeypatch.setattr(converter, "_list_sample_dirs", lambda: sample_dirs)
    monkeypatch.setattr(converter, "_read_json_file", lambda path: meta)
    if resolve_media:
        monkeypatch.setattr(
            converter,
            "_resolve_media_urls",
            lambda paths: {path: f"https://media/{path}?token=x" for path in paths},
        )
    if files is not None:
        monkeypatch.setattr(converter, "_list_entries", lambda path: list(files))
    return converter


def _file(path: str, name: str) -> dict:
    return {"path": path, "name": name, "is_dir": False}


class TestDefaultBindings:
    """Declaring nothing has to keep behaving exactly as before field maps."""

    def test_the_default_map_binds_what_the_contract_requires(self):
        for adapter, bindings in DEFAULT_FIELD_MAPS.items():
            declared = {
                binding.control: binding.type
                for binding in bindings.samples
                if binding.required
            }
            for binding in bindings.regions:
                if binding.required:
                    declared[binding.control] = "rectanglelabels"
                    if binding.observation_control:
                        declared[binding.observation_control] = "textarea"
            assert declared == required_controls(adapter)

    def test_an_unknown_adapter_has_no_default(self):
        with pytest.raises(ValueError, match="Unsupported adapter"):
            default_field_map("bogus")


class TestOptionalImage:
    """An image a sample may not have: bind it, but do not require it."""

    def test_a_missing_optional_image_is_left_out_of_the_task(self, monkeypatch):
        """No stand-in is written, so the annotator never sees a defect photo
        sitting in the reference-image slot."""

        field_map = parse_field_map(
            {
                "images": [
                    {"field": "defect_image", "source": "defect.jpg"},
                    {"field": "gt_image", "source": "*_cam.bmp", "required": False},
                ]
            },
            adapter="avi_train",
        )
        converter = _converter(
            monkeypatch,
            ["/datasets/pcb-001/s1"],
            SAMPLE_META,
            files=[_file("/datasets/pcb-001/s1/defect.jpg", "defect.jpg")],
            field_map=field_map,
        )
        task = json.loads(converter.convert().batch_payloads[0][1])[0]

        assert task["data"]["defect_image_path"] == "/datasets/pcb-001/s1/defect.jpg"
        assert "gt_image" not in task["data"]
        assert "gt_image_path" not in task["data"]

    def test_a_glob_binding_finds_the_sample_reference_image(self, monkeypatch):
        field_map = parse_field_map(
            {
                "images": [
                    {"field": "defect_image", "source": "defect.jpg"},
                    {"field": "gt_image", "source": "*_cam.bmp", "required": False},
                ]
            },
            adapter="avi_train",
        )
        converter = _converter(
            monkeypatch,
            ["/datasets/pcb-001/s1"],
            SAMPLE_META,
            files=[
                _file("/datasets/pcb-001/s1/defect.jpg", "defect.jpg"),
                _file("/datasets/pcb-001/s1/10_B1_cam.bmp", "10_B1_cam.bmp"),
            ],
            field_map=field_map,
        )
        task = json.loads(converter.convert().batch_payloads[0][1])[0]

        assert task["data"]["gt_image_path"].endswith("10_B1_cam.bmp")

    def test_a_required_pattern_with_no_match_fails_loudly(self, monkeypatch):
        """A binding that demands a file the sample does not have is an error the
        caller has to see: silently dropping it would import an incomplete view."""

        field_map = parse_field_map(
            {"images": [{"field": "defect_image", "source": "*_defect.jpg"}]},
            adapter="avi_train",
        )
        converter = _converter(
            monkeypatch,
            ["/datasets/pcb-001/s1"],
            SAMPLE_META,
            files=[_file("/datasets/pcb-001/s1/other.jpg", "other.jpg")],
            field_map=field_map,
        )
        with pytest.raises(ConversionError, match="No file matching"):
            converter.convert()


class TestRenamedControls:
    """Names are the caller's to choose, as long as both directions agree."""

    RENAMED: ClassVar[dict] = {
        "samples": [
            {"field": "quality", "control": "my_verdict", "type": "choices"},
        ],
        "regions": [
            {
                "source": "findings",
                "control": "my_box",
                "label": "category",
                "observation": "observation",
                "observation_control": "my_note",
            }
        ],
    }

    def test_predictions_use_the_declared_names(self, monkeypatch):
        converter = _converter(
            monkeypatch,
            ["/datasets/pcb-001/s1"],
            SAMPLE_META,
            field_map=parse_field_map(self.RENAMED, adapter="avi_train"),
        )
        task = json.loads(converter.convert().batch_payloads[0][1])[0]
        written = {result["from_name"] for result in task["predictions"][0]["result"]}
        assert written == {"my_verdict", "my_box", "my_note"}

    def test_export_reads_the_same_declared_names_back(self, monkeypatch):
        """The forward and reverse conversions share one map; a name the forward
        half writes and the reverse half does not know would export as nothing."""

        field_map = parse_field_map(self.RENAMED, adapter="avi_train")
        converter = _converter(
            monkeypatch, ["/datasets/pcb-001/s1"], SAMPLE_META, field_map=field_map
        )
        task = json.loads(converter.convert().batch_payloads[0][1])[0]

        samples = LabelStudioToAVITrainConverter(field_map).convert(
            [
                {
                    "data": task["data"],
                    "annotations": [{"result": task["predictions"][0]["result"]}],
                }
            ]
        )
        assert len(samples) == 1
        assert samples[0]["quality"] == "defect"
        assert samples[0]["findings"][0]["category"] == "开路"
        assert samples[0]["findings"][0]["observation"] == "线路断开"
        assert samples[0]["defect_image_path"] == "/datasets/pcb-001/s1/defect.jpg"

    def test_the_default_reverse_map_still_spans_both_adapters(self):
        """Export cannot know the adapter, so its default binds every built-in
        control -- including the one only aoi_export declares."""
        fields = {
            binding.control: binding.field for binding in EXPORT_FIELD_MAP.samples
        }
        assert fields["quality_label"] == "quality"
        assert fields["overall_note"] == "note"


class TestDeclaredUnit:
    """A declared unit replaces the magnitude guess with a check."""

    def test_a_declared_unit_is_applied_as_written(self):
        assert _geometry_from(
            {"x": 400, "y": 380, "width": 200, "height": 240}, "norm1000"
        ) == (40.0, 38.0, 20.0, 24.0)

    def test_norm_keys_are_read_under_the_declared_unit(self):
        """A "*_norm" key states its own unit; declaring the same one must not
        make it unreadable."""
        assert _geometry_from(
            {
                "x_min_norm": 400,
                "y_min_norm": 380,
                "x_max_norm": 600,
                "y_max_norm": 620,
            },
            "norm1000",
        ) == (40.0, 38.0, 20.0, 24.0)

    def test_a_declared_unit_rejects_what_does_not_fit_it(self):
        """Silent misplacement is the failure this replaces: a box read at the
        wrong scale looks exactly like a correct one."""
        with pytest.raises(
            ConversionError, match="outside the declared 'percent' scale"
        ):
            _geometry_from({"x": 400, "y": 380, "width": 200, "height": 240}, "percent")

    def test_a_small_percentage_box_is_still_guessed_wrong_under_auto(self):
        """The behaviour a declared unit exists to fix, pinned so nobody mistakes
        it for a regression when they see it."""
        assert _geometry_from({"x": 0.5, "y": 0.5, "width": 0.3, "height": 0.3}) == (
            50.0,
            50.0,
            30.0,
            30.0,
        )
        assert _geometry_from(
            {"x": 0.5, "y": 0.5, "width": 0.3, "height": 0.3}, "percent"
        ) == (0.5, 0.5, 0.3, 0.3)


class TestParsingADeclaration:
    def test_an_omitted_section_keeps_its_default(self):
        declared = parse_field_map(
            {
                "regions": [
                    {"source": "findings", "control": "my_box", "label": "category"}
                ]
            },
            adapter="avi_train",
        )
        assert [image.field for image in declared.images] == [
            "defect_image",
            "diff_image",
            "gt_image",
        ]
        assert [binding.control for binding in declared.samples] == ["quality_label"]
        assert [binding.control for binding in declared.regions] == ["my_box"]

    def test_unknown_keys_are_rejected_with_the_known_ones(self):
        with pytest.raises(ValueError, match="unknown keys: image"):
            parse_field_map({"image": []}, adapter="avi_train")

    def test_a_non_object_declaration_is_rejected(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            parse_field_map(["images"], adapter="avi_train")

    def test_half_declared_per_region_text_is_rejected(self):
        with pytest.raises(ValueError, match="declared together"):
            parse_field_map(
                {
                    "regions": [
                        {
                            "source": "findings",
                            "control": "my_box",
                            "label": "category",
                            "observation": "observation",
                        }
                    ]
                },
                adapter="avi_train",
            )

    def test_bindings_without_an_image_are_rejected(self):
        with pytest.raises(ValueError, match="must also declare the image object"):
            parse_field_map(
                {
                    "images": [],
                    "samples": [
                        {"field": "quality", "control": "q", "type": "choices"}
                    ],
                },
                adapter="avi_train",
            )


class TestDigest:
    def test_the_digest_is_stable_and_content_addressed(self):
        one = parse_field_map({}, adapter="avi_train")
        two = parse_field_map({}, adapter="avi_train")
        renamed = parse_field_map(
            {
                "samples": [
                    {"field": "quality", "control": "verdict", "type": "choices"}
                ]
            },
            adapter="avi_train",
        )
        assert field_map_digest(one) == field_map_digest(two)
        assert field_map_digest(one) != field_map_digest(renamed)

    def test_the_digest_separates_projects_that_differ_only_in_bindings(self):
        """Reuse keys off the project ref alone, so a changed map has to move it
        or create would hand back the project built from the old bindings."""

        args = (
            "http://ls.example.com",
            "us-west-2",
            "/datasets/pcb",
            "avi_train",
            "cfg",
        )
        default = _project_ref(*args, None)
        declared = _project_ref(
            *args, None, field_map_digest(parse_field_map({}, adapter="avi_train"))
        )
        assert default != declared


class TestSkillExamples:
    """The examples are what an agent reads instead of probing; they must be true."""

    @pytest.mark.parametrize(
        ("adapter", "meta_name"),
        [("avi_train", "meta_vlm.json"), ("aoi_export", "meta.json")],
    )
    def test_the_expected_predictions_match_the_converter(
        self, adapter: str, meta_name: str
    ):
        converter = AVITrainToLabelStudioConverter(
            dataset_path="/datasets/example",
            storage_base_url="http://storage.example.com",
            storage_headers={},
            adapter=adapter,
        )
        meta = json.loads((EXAMPLES / adapter / meta_name).read_text(encoding="utf-8"))
        build = (
            converter._build_aoi_predictions
            if adapter == "aoi_export"
            else converter._build_predictions
        )
        expected = json.loads(
            (EXAMPLES / adapter / "expected_predictions.json").read_text(
                encoding="utf-8"
            )
        )
        assert build(meta) == expected

    @pytest.mark.parametrize("adapter", ["avi_train", "aoi_export"])
    def test_the_example_config_passes_the_contract(self, adapter: str):
        xml = (EXAMPLES / adapter / "label_config.xml").read_text(encoding="utf-8")
        validate_label_config_xml(xml, adapter=adapter)

    def test_every_example_image_exists(self):
        for adapter in ("avi_train", "aoi_export"):
            for filename in ("defect.jpg", "diff.jpg", "gt.jpg"):
                assert (EXAMPLES / adapter / filename).is_file()

    @pytest.mark.parametrize(
        "name",
        [
            "optional_reference_image",
            "pcb_prelabel",
            "renamed_controls",
            "pinned_unit",
        ],
    )
    def test_every_field_map_example_parses(self, name: str):
        declared = json.loads(
            (EXAMPLES / "field-maps" / f"{name}.json").read_text(encoding="utf-8")
        )
        assert parse_field_map(declared, adapter="avi_train") is not None


class TestFieldMapValidation:
    def test_a_renamed_control_is_accepted_when_the_map_declares_it(self):
        xml = (
            '<View><Image name="defect_image" value="$defect_image"/>'
            '<Choices name="my_verdict" toName="defect_image">'
            '<Choice value="ok"/><Choice value="defect"/></Choices>'
            '<RectangleLabels name="my_box" toName="defect_image">'
            '<Label value="开路"/></RectangleLabels>'
            '<TextArea name="my_note" toName="defect_image" perRegion="true"/>'
            "</View>"
        )
        declared = parse_field_map(
            {
                "samples": [
                    {"field": "quality", "control": "my_verdict", "type": "choices"}
                ],
                "regions": [
                    {
                        "source": "findings",
                        "control": "my_box",
                        "label": "category",
                        "observation": "observation",
                        "observation_control": "my_note",
                    }
                ],
            },
            adapter="avi_train",
        )
        validate_label_config_xml(xml, field_map=declared)

    def test_the_same_config_is_rejected_under_the_default_bindings(self):
        xml = (
            '<View><Image name="defect_image" value="$defect_image"/>'
            '<Choices name="my_verdict" toName="defect_image">'
            '<Choice value="ok"/></Choices></View>'
        )
        with pytest.raises(LabelConfigValidationError, match="prediction contract"):
            validate_label_config_xml(xml, adapter="avi_train")

    def test_a_declared_control_of_the_wrong_kind_is_reported_by_binding(self):
        xml = (
            '<View><Image name="defect_image" value="$defect_image"/>'
            '<TextArea name="my_verdict" toName="defect_image"/>'
            '<RectangleLabels name="my_box" toName="defect_image">'
            '<Label value="开路"/></RectangleLabels>'
            '<TextArea name="my_note" toName="defect_image"/></View>'
        )
        declared = parse_field_map(
            {
                "samples": [
                    {"field": "quality", "control": "my_verdict", "type": "choices"}
                ],
                "regions": [
                    {
                        "source": "findings",
                        "control": "my_box",
                        "label": "category",
                        "observation": "observation",
                        "observation_control": "my_note",
                    }
                ],
            },
            adapter="avi_train",
        )
        with pytest.raises(
            LabelConfigValidationError, match="'my_verdict' is <TextArea>"
        ):
            validate_label_config_xml(xml, field_map=declared)


class TestUnmappedLedger:
    def test_a_binding_without_synonyms_writes_its_value_through(self, monkeypatch):
        """Free text has nothing to normalise, so it must not be read as a miss."""
        converter = _converter(
            monkeypatch,
            ["/datasets/aoi/s1"],
            {"vlm_verdict": "defect", "note": "金手指氧化"},
            adapter="aoi_export",
        )
        manifest = converter.convert()
        task = json.loads(manifest.batch_payloads[0][1])[0]
        note = next(
            result
            for result in task["predictions"][0]["result"]
            if result["from_name"] == "overall_note"
        )
        assert note["value"]["text"] == ["金手指氧化"]
        assert manifest.unmapped_quality == ()

    def test_a_miss_on_a_kept_binding_is_still_recorded(self, monkeypatch):
        converter = _converter(
            monkeypatch,
            ["/datasets/pcb/s1"],
            dict(SAMPLE_META, quality="weird-state"),
        )
        manifest = converter.convert()
        assert manifest.unmapped_quality == ("weird-state",)
        assert manifest.to_manifest_data().field_map_hash == ""


def test_manifest_carries_the_declared_bindings_identity(monkeypatch):
    declared = parse_field_map(
        {"samples": [{"field": "quality", "control": "verdict", "type": "choices"}]},
        adapter="avi_train",
    )
    converter = _converter(
        monkeypatch,
        ["/datasets/pcb/s1"],
        SAMPLE_META,
        field_map=declared,
        field_map_hash=field_map_digest(declared),
        field_map_path="field_map.json",
    )
    data = converter.convert().to_manifest_data()
    assert data.field_map_hash == field_map_digest(declared)
    assert data.field_map_path == "field_map.json"


def test_a_signer_is_asked_for_exactly_the_bound_files(monkeypatch):
    """Optional and patterned bindings must not widen the signing request."""
    signer = MagicMock()
    field_map = parse_field_map(
        {
            "images": [
                {"field": "defect_image", "source": "defect.jpg"},
                {"field": "gt_image", "source": "*_cam.bmp", "required": False},
            ]
        },
        adapter="avi_train",
    )
    converter = _converter(
        monkeypatch,
        ["/datasets/pcb/s1"],
        SAMPLE_META,
        files=[
            _file("/datasets/pcb/s1/defect.jpg", "defect.jpg"),
            _file("/datasets/pcb/s1/10_B1_cam.bmp", "10_B1_cam.bmp"),
            _file("/datasets/pcb/s1/unrelated.txt", "unrelated.txt"),
        ],
        field_map=field_map,
        media_signer=signer,
        resolve_media=False,
    )
    signer.sign_many.return_value = {
        "/datasets/pcb/s1/defect.jpg": "https://media/defect",
        "/datasets/pcb/s1/10_B1_cam.bmp": "https://media/gt",
    }
    converter.convert()
    assert sorted(signer.sign_many.call_args.args[0]) == [
        "/datasets/pcb/s1/10_B1_cam.bmp",
        "/datasets/pcb/s1/defect.jpg",
    ]
