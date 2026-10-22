"""Tests for Label Studio tool definition contracts."""

import pytest
from jsonschema import Draft202012Validator

from openhands.tools.label_studio import LabelStudioProjectTool
from openhands.tools.label_studio.executor import LabelStudioProjectExecutor


def test_tool_name_and_schema_are_registered() -> None:
    tool = LabelStudioProjectTool.create()[0]
    assert tool.name == "label_studio_project"
    assert tool.action_type.to_mcp_schema()["type"] == "object"

    schema = tool.to_mcp_tool()["inputSchema"]
    Draft202012Validator.check_schema(schema)
    for parameters in (
        tool.to_openai_tool()["function"].get("parameters"),
        tool.to_responses_tool()["parameters"],
    ):
        assert isinstance(parameters, dict)
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        for name, field in schema["properties"].items():
            assert properties[name] == field
        assert "oneOf" not in parameters
    assert "oneOf" not in schema
    assert schema["required"] == ["operation"]
    assert schema["properties"]["adapter"]["enum"] == [
        "avi_train",
        "aoi_export",
        "jsonl",
    ]


@pytest.mark.parametrize(
    "operation", ["create", "get", "update_config", "status", "export", "refresh_media"]
)
def test_lifecycle_schema_is_flat_with_operation_inputs_checked_by_executor(
    operation: str,
) -> None:
    schema = LabelStudioProjectTool.create()[0].to_mcp_tool()["inputSchema"]
    validator = Draft202012Validator(schema)
    arguments = {
        "operation": operation,
        "adapter": "jsonl",
        "idempotency_key": "badcase-step320-c9175",
        "output_path": "storage/datasets/badcase_step320_ls.jsonl",
    }
    # Conditional requirements stay in the executor so provider schemas remain
    # flat, and a bad create gets the same actionable error in both harnesses.
    assert validator.is_valid(arguments)
    for field in ("dataset_path", "label_config_path"):
        assert validator.is_valid({**arguments, field: "storage/datasets/input"})
    assert validator.is_valid(
        {
            **arguments,
            "dataset_path": "storage/datasets/badcase.jsonl",
            "label_config_path": "storage/datasets/label_config.xml",
        }
    )


def test_unknown_create_param_rejected() -> None:
    with pytest.raises(ValueError, match="unknown params: unexpected_param"):
        LabelStudioProjectTool.create(unexpected_param="value")


def test_cluster_reaches_the_executor() -> None:
    """Media URLs must be signed against the cluster this conversation runs on."""
    tool = LabelStudioProjectTool.create(
        ls_base_url="http://ls",
        portal_base_url="http://portal",
        storage_base_url="http://storage",
        cluster="us-west-1#pre",
    )[0]

    assert isinstance(tool.executor, LabelStudioProjectExecutor)
    assert tool.executor._cluster == "us-west-1#pre"


def test_integration_host_defaults_to_label_studio(monkeypatch) -> None:
    """The plugin hosts the SSO/media/token routes, so LS is the default host."""
    for name in (
        "LABEL_STUDIO_PORTAL_BASE_URL",
        "LABEL_STUDIO_SSO_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    tool = LabelStudioProjectTool.create(ls_base_url="http://ls")[0]

    assert isinstance(tool.executor, LabelStudioProjectExecutor)
    assert tool.executor._portal_base_url == "http://ls"
