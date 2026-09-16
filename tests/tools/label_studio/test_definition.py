"""Tests for Label Studio tool definition contracts."""

import pytest

from openhands.tools.label_studio import LabelStudioProjectTool
from openhands.tools.label_studio.executor import LabelStudioProjectExecutor


def test_tool_name_and_schema_are_registered() -> None:
    tool = LabelStudioProjectTool.create()[0]
    assert tool.name == "label_studio_project"
    assert tool.action_type.to_mcp_schema()["type"] == "object"


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
