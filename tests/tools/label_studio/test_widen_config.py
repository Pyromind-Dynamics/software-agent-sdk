"""Tests for reconciling a label config template with the data it imports."""

from openhands.tools.label_studio.skill_helpers import (
    extract_control_values,
    widen_label_config,
)


TEMPLATE = (
    "<View>"
    '<Image name="defect_image" value="$defect_image"/>'
    '<Choices name="quality_label" toName="defect_image">'
    '<Choice value="ok"/><Choice value="defect"/>'
    "</Choices>"
    '<RectangleLabels name="finding_category" toName="defect_image">'
    '<Label value="短路"/></RectangleLabels>'
    "</View>"
)


def test_nothing_to_add_leaves_the_template_byte_for_byte():
    result = widen_label_config(TEMPLATE, {})
    assert result.xml == TEMPLATE
    assert result.added == {}
    assert result.missing_controls == ()


def test_a_value_the_config_lacks_is_added_to_its_own_list():
    result = widen_label_config(TEMPLATE, {"finding_category": ("氧化",)})
    assert result.added == {"finding_category": ("氧化",)}
    # The values the template declared are kept beside the new one.
    assert extract_control_values(result.xml)["finding_category"] == (
        "短路",
        "氧化",
    )
    assert result.xml.count("<RectangleLabels") == 1
    assert result.xml.count("</RectangleLabels>") == 1


def test_choices_are_widened_with_choice_children():
    result = widen_label_config(TEMPLATE, {"quality_label": ("NG+",)})
    assert extract_control_values(result.xml)["quality_label"] == (
        "ok",
        "defect",
        "NG+",
    )
    assert '<Label value="NG+"' not in result.xml


def test_a_value_the_config_already_lists_is_not_added_again():
    result = widen_label_config(TEMPLATE, {"quality_label": ("OK", "defect")})
    assert result.xml == TEMPLATE
    assert result.added == {}


def test_a_spelling_that_normalises_onto_a_declared_value_is_not_added():
    result = widen_label_config(TEMPLATE, {"finding_category": ("短 路",)})
    assert result.xml == TEMPLATE
    assert result.added == {}


def test_an_undeclared_control_is_reported_rather_than_added():
    result = widen_label_config(TEMPLATE, {"finding_observation": ("a note",)})
    assert result.missing_controls == ("finding_observation",)
    assert result.added == {}
    assert result.xml == TEMPLATE


def test_an_empty_control_element_is_widened_in_place():
    xml = (
        "<View>"
        '<Image name="defect_image" value="$defect_image"/>'
        '<Choices name="quality_label" toName="defect_image"/>'
        "</View>"
    )
    result = widen_label_config(xml, {"quality_label": ("ok", "defect")})
    assert result.xml.count("<Choices") == 1
    assert result.xml.endswith("</View>")
    assert extract_control_values(result.xml)["quality_label"] == ("ok", "defect")


def test_a_value_needing_escaping_round_trips():
    value = 'a&b<c>"d'
    result = widen_label_config(TEMPLATE, {"finding_category": (value,)})
    assert "&amp;" in result.xml
    assert extract_control_values(result.xml)["finding_category"] == (
        "短路",
        value,
    )
