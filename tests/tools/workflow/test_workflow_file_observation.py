from openhands.sdk.event.conversation_state import ConversationStateUpdateEvent
from openhands.sdk.llm import TextContent
from openhands.tools.workflow import WorkflowFileObservation, read_workflow_file


def test_read_workflow_file_reads_workflow_py(tmp_path):
    workflow_dir = tmp_path / "public_data" / "workflow_canvas"
    workflow_dir.mkdir(parents=True)
    workflow_path = workflow_dir / "workflow.py"
    workflow_path.write_text(
        "# workflow: Train demo\n"
        "\n"
        'dataset = CloneAndCacheDataset(id="1", dataset="openai/gsm8k")\n',
        encoding="utf-8",
    )

    observation = read_workflow_file(tmp_path, summary="created demo workflow")

    assert observation.exists is True
    assert observation.path == str(workflow_path)
    assert observation.name == "Train demo"
    assert observation.summary == "created demo workflow"
    assert "CloneAndCacheDataset" in observation.workflow
    assert observation.kind == "WorkflowFileObservation"
    content = observation.to_llm_content[0]
    assert isinstance(content, TextContent)
    assert content.text == "Workflow Train demo (3 lines)."


def test_read_workflow_file_missing_file(tmp_path):
    observation = read_workflow_file(tmp_path)

    assert observation.exists is False
    assert observation.workflow == ""
    assert observation.name is None
    assert observation.path == str(
        tmp_path / "public_data" / "workflow_canvas" / "workflow.py"
    )
    content = observation.to_llm_content[0]
    assert isinstance(content, TextContent)
    assert "No workflow.py found" in content.text


def test_workflow_file_observation_round_trips_in_state_event():
    observation = WorkflowFileObservation.from_text(
        text="Workflow Train demo (1 lines).",
        workflow="# workflow: Train demo\n",
        path="/tmp/workflow.py",
        name="Train demo",
        summary=None,
        exists=True,
    )
    event = ConversationStateUpdateEvent(
        key="pyromind_workflow",
        value=observation.model_dump(mode="json"),
    )

    dumped = event.model_dump(mode="json")
    assert dumped["value"]["kind"] == "WorkflowFileObservation"
    assert dumped["value"]["workflow"] == "# workflow: Train demo\n"

    restored = ConversationStateUpdateEvent.model_validate(dumped)
    restored_observation = WorkflowFileObservation.model_validate(restored.value)
    assert restored_observation.name == "Train demo"
