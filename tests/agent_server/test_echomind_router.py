import json
from uuid import uuid4

import pytest
from pyromind_agent_server.experiment_plan_projects import (
    EchoMindExperimentPlanProjectBinder,
)
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.domain.experiment_plans import (
    ExperimentContext,
    ExperimentStage,
    PlanOwner,
    PlanRevision,
    WorkflowPlan,
)
from pyromind_runtime.domain.preparation import ProjectView

from openhands.agent_server.agent_profiles.echomind_manager import (
    ECHOMIND_MANAGER_DEVELOPMENT_SKILLS,
    ECHOMIND_MANAGER_DEVELOPMENT_TOOLS,
    ECHOMIND_MANAGER_PLANNING_SKILLS,
    ECHOMIND_MANAGER_PLANNING_TOOLS,
    ECHOMIND_MANAGER_PROFILE,
)
from openhands.agent_server.agent_profiles.pyromind import PYROMIND_PROFILE
from openhands.agent_server.echomind_projects import (
    ProjectClusterMismatchError,
    ProjectNotFoundError,
    ProjectStore,
)


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
    assert ECHOMIND_PRODUCT_PROFILE.tool_allowlist == ECHOMIND_MANAGER_PLANNING_TOOLS
    assert set(ECHOMIND_MANAGER_PLANNING_TOOLS) == {
        "read",
        "terminal",
        "create_experiment_plan",
        "revise_experiment_plan",
        "get_experiment_plan",
    }
    assert ECHOMIND_PRODUCT_PROFILE.skill_allowlist == ECHOMIND_MANAGER_PLANNING_SKILLS
    assert ECHOMIND_MANAGER_PROFILE.usage_id == "echomind-manager"

    assert set(ECHOMIND_MANAGER_DEVELOPMENT_TOOLS) == {
        *PYROMIND_PROFILE.tool_allowlist,
        "create_experiment_plan",
        "revise_experiment_plan",
        "get_experiment_plan",
        "request_experiment_task",
        "get_experiment_execution",
    }
    assert not {"request_experiment_task", "get_experiment_execution"}.intersection(
        PYROMIND_PROFILE.tool_allowlist
    )
    assert set(ECHOMIND_MANAGER_DEVELOPMENT_SKILLS) == {
        *PYROMIND_PROFILE.skill_allowlist,
        "experiment-planning",
    }
    assert ECHOMIND_PRODUCT_PROFILE.development_tool_allowlist == (
        ECHOMIND_MANAGER_DEVELOPMENT_TOOLS
    )
    assert ECHOMIND_PRODUCT_PROFILE.development_skill_allowlist == (
        ECHOMIND_MANAGER_DEVELOPMENT_SKILLS
    )


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
        conversation_id=owner.conversation_id,
        cluster="us-west-1#pre",
        owner_user_id=owner.user_id,
    )

    assert first.project.project_id == project_id
    assert first.graph.project_id == project_id
    assert first.graph.template == "experiment-planning/v1"
    assert [node.id for node in first.graph.nodes] == [
        "dataset",
        "inference",
        "evaluator",
        "train",
    ]
    assert first.graph.nodes[0].launch_mode == "agent"
    assert first.project.requirements[0].key == "images"

    second = store.bind_plan(
        PlanRevision(
            project_id=project_id,
            revision=2,
            plan=_experiment_plan(name="Revised AVI experiment"),
            confirmed_by="42",
        ),
        conversation_id=owner.conversation_id,
        cluster="us-west-1#pre",
        owner_user_id=owner.user_id,
    )
    assert second.project.created_at == first.project.created_at
    assert second.graph.graph_id == first.graph.graph_id
    assert second.project.name == "Revised AVI experiment"
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
        conversation_id="conversation-1",
        cluster="us-west-1#pre",
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
        revision, owner, cluster="us-west-1#pre"
    )

    project = ProjectStore.for_conversations_dir(tmp_path / "conversations").load(
        revision.project_id, owner_user_id=owner.user_id
    )
    assert project.project.conversation_id == owner.conversation_id
    assert project.project.cluster == "us-west-1#pre"


def test_project_list_filters_owner_and_ignores_legacy_clusterless_records(
    tmp_path,
):
    store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    owner = PlanOwner(
        user_id="42",
        conversation_id="conversation-1",
        profile_id="echomind",
    )
    first = store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_experiment_plan(name="First"),
            confirmed_by="42",
        ),
        conversation_id=owner.conversation_id,
        cluster="us-west-1#pre",
        owner_user_id=owner.user_id,
    )
    second = store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_experiment_plan(name="Second"),
            confirmed_by="42",
        ),
        conversation_id="conversation-2",
        cluster="us-east-1#pre",
        owner_user_id=owner.user_id,
    )
    store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_experiment_plan(name="Other user"),
            confirmed_by="7",
        ),
        conversation_id="conversation-3",
        cluster="us-west-1#pre",
        owner_user_id="7",
    )

    legacy_id = str(uuid4())
    legacy_dir = store.root / legacy_id
    legacy_dir.mkdir()
    legacy_project = first.project.model_dump(mode="json")
    legacy_project.update(
        {
            "schema_version": 1,
            "project_id": legacy_id,
            "conversation_id": "legacy-conversation",
        }
    )
    legacy_project.pop("cluster")
    legacy_graph = first.graph.model_copy(update={"project_id": legacy_id})
    (legacy_dir / "project.json").write_text(
        json.dumps(legacy_project), encoding="utf-8"
    )
    (legacy_dir / "graph.json").write_text(
        legacy_graph.model_dump_json(), encoding="utf-8"
    )

    summaries = store.list(owner_user_id="42")

    assert {item.project_id for item in summaries} == {
        first.project.project_id,
        second.project.project_id,
    }
    assert {item.cluster for item in summaries} == {
        "us-west-1#pre",
        "us-east-1#pre",
    }
    assert {item.conversation_id for item in summaries} == {
        "conversation-1",
        "conversation-2",
    }


def test_project_access_requires_exact_owner(tmp_path):
    store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_experiment_plan(),
            confirmed_by="42",
        ),
        conversation_id="conversation-1",
        cluster="us-west-1#pre",
        owner_user_id="42",
    )
    ownerless_id = str(uuid4())
    ownerless_dir = store.root / ownerless_id
    ownerless_dir.mkdir()
    ownerless = project.project.model_copy(
        update={"project_id": ownerless_id, "owner_user_id": None}
    )
    ownerless_graph = project.graph.model_copy(update={"project_id": ownerless_id})
    (ownerless_dir / "project.json").write_text(
        ownerless.model_dump_json(), encoding="utf-8"
    )
    (ownerless_dir / "graph.json").write_text(
        ownerless_graph.model_dump_json(), encoding="utf-8"
    )

    with pytest.raises(ProjectNotFoundError):
        store.load(project.project.project_id, owner_user_id="7")
    with pytest.raises(ProjectNotFoundError):
        store.load(ownerless_id, owner_user_id="42")
    with pytest.raises(PermissionError, match="another owner"):
        store.bind_plan(
            PlanRevision(
                project_id=ownerless_id,
                revision=1,
                plan=_experiment_plan(),
                confirmed_by="42",
            ),
            conversation_id="conversation-1",
            cluster="us-west-1#pre",
            owner_user_id="42",
        )


def test_project_cluster_guard_rejects_missing_or_mismatched_route(tmp_path):
    project = ProjectStore.for_conversations_dir(tmp_path / "conversations").bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_experiment_plan(),
            confirmed_by="42",
        ),
        conversation_id="conversation-1",
        cluster="us-west-1#pre",
        owner_user_id="42",
    )
    store = ProjectStore.for_conversations_dir(tmp_path / "conversations")

    with pytest.raises(ProjectClusterMismatchError, match="required"):
        store.require_cluster(project, None)
    with pytest.raises(ProjectClusterMismatchError, match="belongs"):
        store.require_cluster(project, "us-east-1#pre")
    store.require_cluster(project, "us-west-1#pre")


def test_preparation_binding_uses_the_same_project_identity(tmp_path):
    project_id = uuid4().hex
    view = ProjectView(
        project_id=project_id,
        conversation_id="conversation-1",
        plan_sequence=1,
        plan=ExperimentContext(
            goal="Build AVI dataset",
            stages=(ExperimentStage(id="dataset", name="Dataset", kind="dataset"),),
        ),
        creation_command_id="create",
        initial_run_id="run",
        cluster="us-west-1#pre",
    )
    store = ProjectStore.for_conversations_dir(tmp_path / "conversations")

    project = store.bind_preparation(
        view,
        cluster="us-west-1#pre",
        owner_user_id="42",
    )

    assert project.project.project_id == project_id
    assert project.project.name == "Build AVI dataset"
    assert project.project.conversation_id == "conversation-1"
    assert project.project.cluster == "us-west-1#pre"
    assert [node.id for node in project.graph.nodes] == ["dataset"]
