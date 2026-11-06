from uuid import uuid4

from pyromind_agent_server.api.echomind_projects import _build_agent_stage_start_context
from pyromind_agent_server.echomind_paths import sandbox_path_for_storage_uri
from pyromind_agent_server.experiment_plan_projects import (
    EchoMindExperimentPlanProjectBinder,
)
from pyromind_agent_server.profiles import (
    DEFAULT_PRODUCT_PROFILE,
    ECHOMIND_PRODUCT_PROFILE,
    session_policy_for_profile,
)
from pyromind_runtime.domain.experiment_plans import (
    PlanOwner,
    PlanRevision,
    WorkflowPlan,
)
from pyromind_runtime.domain.project_graphs import GraphPatch

from openhands.agent_server.agent_profiles.echomind_dataset import (
    ECHOMIND_DATASET_ROLE,
)
from openhands.agent_server.agent_profiles.echomind_manager import (
    ECHOMIND_MANAGER_DEVELOPMENT_SKILLS,
    ECHOMIND_MANAGER_DEVELOPMENT_TOOLS,
    ECHOMIND_MANAGER_PROFILE,
)
from openhands.agent_server.agent_profiles.echomind_train import ECHOMIND_TRAIN_ROLE
from openhands.agent_server.agent_profiles.pyromind import PYROMIND_PROFILE
from openhands.agent_server.echomind_projects import ProjectStore
from openhands.agent_server.echomind_stage_runs import ProjectRunStore


def _experiment_plan(*, name: str = "AVI experiment") -> WorkflowPlan:
    return WorkflowPlan.model_validate(
        {
            "skill": "experiment-planning",
            "name": name,
            "goal": "Build a Dataset -> Inference -> Evaluator -> Train loop.",
            "resources": [
                {
                    "id": "images",
                    "name": "AVI images",
                    "uri": "datasets/avi/train.jsonl",
                }
            ],
            "stages": [
                {"id": "dataset", "name": "Dataset"},
                {"id": "inference", "name": "Inference"},
                {"id": "evaluator", "name": "Evaluator"},
                {"id": "train", "name": "Train"},
            ],
            "edges": [
                {"source": "dataset", "target": "inference"},
                {"source": "inference", "target": "evaluator"},
                {"source": "evaluator", "target": "train"},
            ],
        }
    )


def test_echomind_manager_planning_and_development_capabilities_are_explicit():
    assert ECHOMIND_PRODUCT_PROFILE.profile_id == "echomind"
    assert ECHOMIND_PRODUCT_PROFILE.agentgenome_enabled is False
    assert DEFAULT_PRODUCT_PROFILE.agentgenome_enabled is True
    assert (
        session_policy_for_profile(ECHOMIND_PRODUCT_PROFILE).agentgenome_enabled
        is False
    )
    assert "Do not query AgentGenome" in ECHOMIND_PRODUCT_PROFILE.instructions
    assert 'role_id="dataset"' in ECHOMIND_PRODUCT_PROFILE.instructions
    assert "Do not reproduce its data processing" in (
        ECHOMIND_PRODUCT_PROFILE.instructions
    )
    assert set(ECHOMIND_PRODUCT_PROFILE.tool_allowlist or ()) == {
        "read",
        "write",
        "edit",
        "terminal",
        "grep",
        "find",
        "ls",
        "ask_user",
        "get_preparation",
        "update_preparation",
        "set_preparation_plan",
        "inspect_resource",
    }
    assert ECHOMIND_PRODUCT_PROFILE.skill_allowlist == ("experiment-planning",)
    assert ECHOMIND_MANAGER_PROFILE.usage_id == "echomind-manager"

    assert set(ECHOMIND_MANAGER_DEVELOPMENT_TOOLS) == {
        *PYROMIND_PROFILE.tool_allowlist,
        "create_experiment_plan",
        "revise_experiment_plan",
        "get_experiment_plan",
        "get_project_graph",
        "apply_graph_patch",
        "open_project_stage",
        "get_project_stage_run",
        "request_project_stage_task",
        "approve_project_stage_plan",
        "confirm_project_stage_run",
        "record_project_stage_result",
        "get_experiment_context",
        "update_experiment_context",
        "request_experiment_task",
        "get_experiment_execution",
        "control_experiment_task",
        "read_experiment_artifact",
    }
    assert set(ECHOMIND_PRODUCT_PROFILE.development_tool_allowlist or ()) == {
        *ECHOMIND_MANAGER_DEVELOPMENT_TOOLS,
        "ask_user",
        "get_preparation",
        "inspect_resource",
        "ls",
        "set_preparation_plan",
        "grep",
        "find",
        "update_preparation",
    }
    assert not {"request_experiment_task", "get_experiment_execution"}.intersection(
        PYROMIND_PROFILE.tool_allowlist
    )
    assert set(ECHOMIND_MANAGER_DEVELOPMENT_SKILLS) == {
        *PYROMIND_PROFILE.skill_allowlist,
        "experiment-planning",
        "experiment-graph-development",
        "avi-baseline-evaluation",
    }
    assert set(ECHOMIND_PRODUCT_PROFILE.development_skill_allowlist or ()) == {
        *ECHOMIND_MANAGER_DEVELOPMENT_SKILLS,
    }
    assert ECHOMIND_PRODUCT_PROFILE.preparation_policy is not None
    assert ECHOMIND_PRODUCT_PROFILE.preparation_policy.checklist_id == (
        "experiment-planning"
    )
    assert {
        node.id for node in ECHOMIND_PRODUCT_PROFILE.preparation_policy.workflow_nodes
    } == {"dataset", "train", "inference", "evaluator"}
    assert (
        not ECHOMIND_PRODUCT_PROFILE.preparation_policy.create_project_starts_entry_task
    )
    assert ECHOMIND_PRODUCT_PROFILE.command_allowlist == (
        "user_message",
        "input_response",
        "cancel",
        "create_project",
        "permission_response",
        "control_experiment_task",
        "resolve_experiment_submission",
    )
    assert ECHOMIND_PRODUCT_PROFILE.command_allowlist is not None
    assert "permission_response" in ECHOMIND_PRODUCT_PROFILE.command_allowlist


def test_echomind_dataset_sample_report_contract_is_explicit():
    prompt = ECHOMIND_DATASET_ROLE.prompt
    assert "A sample validation does not require a platform task" in prompt
    assert "public_data/experiment-runs/<run_id>/<report-file>" in prompt
    assert "do not retry alternative path forms" in prompt
    assert "Do not\nsynthesize invalid or adversarial records" in prompt
    assert "stop once the declared sample and acceptance criteria" in prompt
    manager_prompt = ECHOMIND_PRODUCT_PROFILE.instructions
    assert "do not\nrequest synthetic or adversarial fixtures" in manager_prompt


def test_echomind_train_workflow_is_run_scoped():
    prompt = ECHOMIND_TRAIN_ROLE.prompt
    assert "public_data/experiment-runs/<run_id>/workflow.py" in prompt
    assert "overrides the Skill's default public_data/workflow_canvas/" in prompt


def test_dataset_sandbox_paths_keep_the_full_storage_path(monkeypatch):
    monkeypatch.setenv("PYROMIND_SANDBOX_MOUNT_PATH", "/target-workspace")

    assert (
        sandbox_path_for_storage_uri("/datasets/pcb_avi_eval_3/")
        == "/target-workspace/datasets/pcb_avi_eval_3"
    )
    assert (
        sandbox_path_for_storage_uri("datasets/pcb_avi_eval_3/train.jsonl")
        == "/target-workspace/datasets/pcb_avi_eval_3/train.jsonl"
    )
    assert (
        sandbox_path_for_storage_uri(
            "/target-workspace/datasets/pcb_avi_eval_3/eval.jsonl"
        )
        == "/target-workspace/datasets/pcb_avi_eval_3/eval.jsonl"
    )
    assert sandbox_path_for_storage_uri("public_data/report.json") is None
    assert sandbox_path_for_storage_uri("central://checkpoint-1") is None


def test_agent_stage_start_context_enters_clarification_before_execution(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_experiment_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run, _ = ProjectRunStore(project_store.root).start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )

    context = _build_agent_stage_start_context(project, "dataset", run)

    assert f"project_id: {project.project.project_id}" in context
    assert f"node_id: {project.graph.nodes[0].id}" in context
    assert f"run_id: {run.run_id}" in context
    assert "clarification phase" in context
    assert "get_project_graph" in context
    assert "get_project_stage_run" in context
    assert "approve_project_stage_plan" in context
    assert "request_project_stage_task" in context
    assert 'task.inputs.phase="sample"' in context
    assert "Do not inspect Storage" in context
    assert "Do not execute the Node implementation" in context
    assert "Required development sequence" not in context
    assert "pipeline_cli" not in context
    assert "record_project_stage_result" in context


def test_agent_stage_start_context_skips_sample_cycle_for_inference(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_experiment_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    inference = next(node for node in project.graph.nodes if node.id == "inference")
    run, _ = ProjectRunStore(project_store.root).start_agent_development(
        project=project,
        node=inference,
        request_id="start-inference",
    )

    context = _build_agent_stage_start_context(project, "inference", run)

    assert "execution_role: inference" in context
    assert 'task.inputs.phase="execute"' in context
    assert 'task.inputs.phase="sample"' not in context
    assert "does not require the Dataset sample-confirmation cycle" in context


def test_plan_binding_creates_and_updates_one_project_with_revision_id(tmp_path):
    owner = PlanOwner(
        user_id="42",
        conversation_id="conversation-1",
        profile_id="echomind",
    )
    project_id = uuid4().hex
    store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    first = store.bind_plan(
        PlanRevision(
            project_id=project_id,
            revision=1,
            plan=_experiment_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id=owner.conversation_id,
        owner_user_id=owner.user_id,
    )

    assert first.project.project_id == project_id
    assert first.graph.graph_id == project_id
    assert first.graph.project_id == project_id
    assert first.graph.template == "experiment-planning/v1"
    assert [node.id for node in first.graph.nodes] == [
        "dataset",
        "inference",
        "evaluator",
        "train",
    ]
    assert {node.launch_mode for node in first.graph.nodes} == {"agent"}
    assert {node.id: node.execution_role for node in first.graph.nodes} == {
        "dataset": "dataset",
        "inference": "inference",
        "evaluator": "eval",
        "train": "train",
    }
    assert all("execution_role" not in node.config for node in first.graph.nodes)
    assert first.project.requirements[0].key == "images"

    store.apply_graph_patch(
        project_id,
        owner_user_id=owner.user_id,
        patch=GraphPatch.model_validate(
            {
                "request_id": "rename-dataset",
                "operations": [
                    {
                        "op": "update_node",
                        "node_id": "dataset",
                        "title": "Confirmed Dataset",
                    }
                ],
            }
        ),
    )

    second = store.bind_plan(
        PlanRevision(
            project_id=project_id,
            revision=2,
            plan=_experiment_plan(name="Revised AVI experiment"),
            confirmed_by="42",
        ),
        echomind_conversation_id=owner.conversation_id,
        owner_user_id=owner.user_id,
    )
    assert second.project.created_at == first.project.created_at
    assert second.graph.graph_id == first.graph.graph_id
    assert second.project.name == "Revised AVI experiment"
    assert second.graph.nodes[0].title == "Confirmed Dataset"
    assert len(list((tmp_path / "projects").iterdir())) == 1


def test_project_graph_follows_plan_topology_instead_of_skill_name(tmp_path):
    plan = WorkflowPlan.model_validate(
        {
            "skill": "experiment-planning",
            "name": "Custom topology",
            "goal": "Validate generic graph generation.",
            "stages": [
                {"id": "prepare", "name": "Dataset"},
                {"id": "score", "name": "Evaluator"},
            ],
            "edges": [{"source": "prepare", "target": "score"}],
        }
    )
    project_id = uuid4().hex
    project = ProjectStore.for_conversations_dir(tmp_path / "conversations").bind_plan(
        PlanRevision(
            project_id=project_id,
            revision=1,
            plan=plan,
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )

    assert [node.id for node in project.graph.nodes] == ["prepare", "score"]
    assert project.graph.edges[0].source == "prepare"
    assert project.graph.edges[0].target == "score"
    assert project.graph.nodes[1].config["artifact_format"] == "atif"


def test_binder_uses_plan_owner_for_project_identity(tmp_path):
    owner = PlanOwner(
        user_id="42",
        conversation_id="conversation-1",
        profile_id="echomind",
    )
    revision = PlanRevision(
        project_id=uuid4().hex,
        revision=1,
        plan=_experiment_plan(),
        confirmed_by=owner.user_id,
    )

    EchoMindExperimentPlanProjectBinder(tmp_path / "conversations").bind(
        revision, owner
    )

    project = ProjectStore.for_conversations_dir(tmp_path / "conversations").load(
        revision.project_id, owner_user_id=owner.user_id
    )
    assert project.project.echomind_conversation_id == owner.conversation_id
