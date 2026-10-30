from uuid import uuid4

from pyromind_agent_server.experiment_plan_projects import (
    EchoMindExperimentPlanProjectBinder,
)
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.domain.experiment_plans import (
    PlanOwner,
    PlanRevision,
    WorkflowPlan,
)

from openhands.agent_server.agent_profiles.echomind_manager import (
    ECHOMIND_MANAGER_DEVELOPMENT_SKILLS,
    ECHOMIND_MANAGER_DEVELOPMENT_TOOLS,
    ECHOMIND_MANAGER_PLANNING_SKILLS,
    ECHOMIND_MANAGER_PLANNING_TOOLS,
    ECHOMIND_MANAGER_PROFILE,
)
from openhands.agent_server.agent_profiles.pyromind import PYROMIND_PROFILE
from openhands.agent_server.echomind_projects import ProjectStore


def _avi_plan(*, name: str = "AVI experiment") -> WorkflowPlan:
    return WorkflowPlan.model_validate(
        {
            "skill": "avi-single-turn",
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
    assert ECHOMIND_PRODUCT_PROFILE.skill_allowlist == ECHOMIND_MANAGER_PLANNING_SKILLS
    assert ECHOMIND_MANAGER_PROFILE.usage_id == "echomind-manager"

    assert set(ECHOMIND_MANAGER_DEVELOPMENT_TOOLS) == {
        *PYROMIND_PROFILE.tool_allowlist,
        "create_experiment_plan",
        "revise_experiment_plan",
        "get_experiment_plan",
    }
    assert set(ECHOMIND_MANAGER_DEVELOPMENT_SKILLS) == {
        *PYROMIND_PROFILE.skill_allowlist,
        "avi-single-turn",
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
            plan=_avi_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id=owner.conversation_id,
        owner_user_id=owner.user_id,
    )

    assert first.project.project_id == project_id
    assert first.graph.project_id == project_id
    assert first.graph.template == "avi-single-turn/v1"
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
            plan=_avi_plan(name="Revised AVI experiment"),
            confirmed_by="42",
        ),
        echomind_conversation_id=owner.conversation_id,
        owner_user_id=owner.user_id,
    )
    assert second.project.created_at == first.project.created_at
    assert second.graph.graph_id == first.graph.graph_id
    assert second.project.name == "Revised AVI experiment"
    assert len(list((tmp_path / "projects").iterdir())) == 1


def test_binder_uses_plan_owner_for_project_identity(tmp_path):
    owner = PlanOwner(
        user_id="42",
        conversation_id="conversation-1",
        profile_id="echomind",
    )
    revision = PlanRevision(
        project_id=uuid4().hex,
        revision=1,
        plan=_avi_plan(),
        confirmed_by=owner.user_id,
    )

    EchoMindExperimentPlanProjectBinder(tmp_path / "conversations").bind(
        revision, owner
    )

    project = ProjectStore.for_conversations_dir(tmp_path / "conversations").load(
        revision.project_id, owner_user_id=owner.user_id
    )
    assert project.project.echomind_conversation_id == owner.conversation_id
