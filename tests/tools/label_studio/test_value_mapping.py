"""Tests for tolerant value resolution between data and config values."""

from openhands.tools.label_studio.value_mapping import (
    VERDICT_SYNONYMS,
    ControlValueIndex,
    normalize_value_key,
    resolve_control_value,
)


def test_normalize_folds_case_width_and_separators():
    assert normalize_value_key(" Defect ") == "defect"
    assert normalize_value_key("ｄｅｆｅｃｔ") == "defect"
    assert normalize_value_key("short-circuit") == "shortcircuit"
    assert normalize_value_key("短 路") == "短路"
    assert normalize_value_key("氧化（film）") == "氧化film"


def test_index_exact_returns_the_config_own_spelling():
    index = ControlValueIndex(["Defect", "OK"])
    assert index.exact("defect") == "Defect"
    assert index.exact(" o k ") == "OK"
    assert index.exact("unknown") is None


def test_index_closest_matches_a_near_miss_but_refuses_ambiguity():
    # Two equally close candidates: guessing one would be worse than reporting.
    assert ControlValueIndex(["cat", "bat"]).closest("at") is None
    assert ControlValueIndex(["氧化", "短路", "断路"]).closest("氧化膜") == "氧化"
    assert ControlValueIndex(["短路", "断路"]).closest("断路") == "断路"
    assert ControlValueIndex(["短 路"]).closest("短路") == "短 路"
    assert ControlValueIndex(["defect", "ok"]).closest("broken") is None


def test_a_config_value_is_never_shadowed_by_a_synonym_table():
    """The regression the resolver exists for: a table maps foreign spellings."""
    match = resolve_control_value(
        "defect",
        config_values=("ok", "defect"),
        synonyms={"true": "defect", "false": "ok"},
    )
    assert match.matched
    assert match.value == "defect"


def test_a_synonym_is_adopted_with_the_config_own_spelling():
    match = resolve_control_value(
        "NG",
        config_values=("OK", "Defect"),
        synonyms={"ng": "defect"},
    )
    assert match.value == "Defect"


def test_the_builtin_vocabulary_applies_without_a_declared_table():
    match = resolve_control_value(
        "PASS",
        config_values=("ok", "defect"),
        builtin=VERDICT_SYNONYMS,
    )
    assert match.matched
    assert match.value == "ok"


def test_a_mapping_to_a_value_the_config_lacks_is_not_adopted():
    """A synonym that points outside the config must not make things worse."""
    match = resolve_control_value(
        "true",
        config_values=("好", "坏"),
        synonyms={"true": "defect"},
    )
    assert not match.matched
    assert match.value == "true"


def test_an_unmatched_value_is_returned_verbatim_for_the_caller():
    match = resolve_control_value(
        "weird-state",
        config_values=("ok", "defect"),
        builtin=VERDICT_SYNONYMS,
    )
    assert not match.matched
    assert match.value == "weird-state"


def test_no_declared_values_leaves_a_table_as_the_only_intent():
    match = resolve_control_value("NG", synonyms={"ng": "defect"})
    assert match.matched
    assert match.value == "defect"
    assert not resolve_control_value("whatever").matched
