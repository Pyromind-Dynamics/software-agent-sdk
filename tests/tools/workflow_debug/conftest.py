from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.models import TrainingTaskCreateResponse

from pyromind_common.sdk.conversation.base import ToolConversation
from pyromind_common.sdk.workspace.local import LocalWorkspace
from pyromind_common.tools.workflow_debug import execution


@pytest.fixture
def debug_conversation(tmp_path):
    path = tmp_path / "public_data/workflow_canvas/workflow.py"
    path.parent.mkdir(parents=True)
    path.write_text("workflow_dsl = 'test'\n")
    return cast(
        ToolConversation,
        SimpleNamespace(
            id="debug-conversation",
            workspace=LocalWorkspace(working_dir=tmp_path),
            state=SimpleNamespace(
                agent_state={},
                secret_registry=SimpleNamespace(
                    get_secret_value=Mock(return_value="request-token")
                ),
            ),
        ),
    )


@pytest.fixture
def debug_client(monkeypatch):
    client = PyroMindAPIClient(api_key="derived-key", max_retries=0)
    create = Mock(
        return_value=TrainingTaskCreateResponse(
            task_id="debug-task", name="debug-workflow", status="Pending"
        )
    )
    monkeypatch.setattr(client.studio, "create", create)
    monkeypatch.setattr(
        execution, "create_workflow_api_client", Mock(return_value=client)
    )
    monkeypatch.setattr(
        execution,
        "convert_dsl_to_xyflow",
        Mock(
            side_effect=lambda *args, **kwargs: {
                "name": "debug-workflow",
                "nodes": [],
                "edges": [],
            }
        ),
    )
    yield client, create
    client.close()
