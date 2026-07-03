from openhands.sdk.event.llm_convertible import ObservationEvent
from openhands.tools.workflow import (
    PublishedWorkflowObservation,
    PublishWorkflowAction,
    PublishWorkflowExecutor,
    PublishWorkflowTool,
)


def test_publish_workflow_executor_reads_workflow_py(tmp_path):
    workflow_path = tmp_path / "workflow.py"
    workflow_path.write_text(
        "# workflow: Train demo\n"
        "\n"
        'dataset = CloneAndCacheDataset(id="1", dataset="openai/gsm8k")\n',
        encoding="utf-8",
    )

    observation = PublishWorkflowExecutor(str(tmp_path))(
        PublishWorkflowAction(summary="created demo workflow")
    )

    assert observation.exists is True
    assert observation.path == str(workflow_path)
    assert observation.name == "Train demo"
    assert observation.summary == "created demo workflow"
    assert "CloneAndCacheDataset" in observation.workflow
    assert observation.kind == "PublishedWorkflowObservation"
    assert observation.to_llm_content[0].text == "Published Train demo (3 lines)."


def test_publish_workflow_executor_missing_file(tmp_path):
    observation = PublishWorkflowExecutor(str(tmp_path))(PublishWorkflowAction())

    assert observation.exists is False
    assert observation.workflow == ""
    assert observation.name is None
    assert observation.path == str(tmp_path / "workflow.py")
    assert "No workflow.py found" in observation.to_llm_content[0].text


def test_publish_workflow_tool_name():
    assert PublishWorkflowTool.name == "publish_workflow"


def test_published_workflow_observation_round_trips_in_event():
    event = ObservationEvent(
        tool_name="publish_workflow",
        tool_call_id="call_1",
        action_id="action_1",
        observation=PublishedWorkflowObservation.from_text(
            text="Published Train demo (1 lines).",
            workflow="# workflow: Train demo\n",
            path="/tmp/workflow.py",
            name="Train demo",
            summary=None,
            exists=True,
        ),
    )

    dumped = event.model_dump(mode="json")
    assert dumped["observation"]["kind"] == "PublishedWorkflowObservation"
    assert dumped["observation"]["workflow"] == "# workflow: Train demo\n"

    restored = ObservationEvent.model_validate(dumped)
    assert isinstance(restored.observation, PublishedWorkflowObservation)
    assert restored.observation.name == "Train demo"
