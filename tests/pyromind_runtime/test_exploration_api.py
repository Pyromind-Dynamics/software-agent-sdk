import hashlib
import json

import httpx
import pytest
from pyromind_agent_server.api.router import create_product_router
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_plans import PlanOwner

from . import test_standalone_experiments as experiment_fixtures
from .fake_adapter import FakeAdapter
from .legacy_experiment_profile import EXPERIMENT_PROFILE_V4 as ECHOMIND_PRODUCT_PROFILE
from .test_product_api import _app


experiment = experiment_fixtures.experiment
claim = experiment_fixtures.claim
request = experiment_fixtures.request


async def test_new_profile_has_no_graph_gate_and_artifacts_remain_isolated(
    tmp_path, experiment
):
    service, _, engine = experiment
    adapter = FakeAdapter("pi")
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        {"pi": adapter},
        default_harness_id="pi",
        experiment_plan_store=service.plans,
        experiment_execution_store=service.store,
        execution_roles=("eval",),
    )
    app = _app(tmp_path, runtime)
    app.include_router(create_product_router(ECHOMIND_PRODUCT_PROFILE))
    base = ECHOMIND_PRODUCT_PROFILE.route_prefix
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"x-pyromind-debug-user-id": "42"},
        ) as client:
            created = await client.post(base, json={"llm": {"model": "test"}})
            assert created.status_code == 201
            identity = created.json()["conversation_id"]
            owner = PlanOwner(
                user_id="42",
                conversation_id=identity,
                profile_id=ECHOMIND_PRODUCT_PROFILE.profile_id,
            )
            context = RequestContext(user_id="42", profile_id=owner.profile_id)
            spec = adapter.created_specs[0]
            assert spec.tool_allowlist is not None
            assert spec.skill_allowlist is not None
            assert spec.delegated_roles is not None
            assert {"read", "write", "edit", "terminal", "df_submit_evaluation"} <= set(
                spec.tool_allowlist
            )
            assert "data-processing" in spec.skill_allowlist
            assert "inference-evaluation" in spec.skill_allowlist
            assert "read_skill" not in spec.tool_allowlist
            assert "subagent" not in spec.tool_allowlist
            assert [role.name for role in spec.delegated_roles] == ["eval"]
            assert "df_submit_evaluation" in spec.delegated_roles[0].tools
            assert "update_experiment_context" not in spec.delegated_roles[0].tools
            run = await runtime.request_experiment_task(identity, request(), context)
            assert run["revision"] is None and service.plans.view(owner).current is None
            snapshot = await client.get(f"{base}/{identity}/snapshot")
            assert (
                snapshot.status_code == 200
                and len(snapshot.json()["experiment_executions"]) == 1
            )
            scope = claim(service, owner, run)
            body = b'{"status":"succeeded","evaluation_status":"validated"}'
            path = f"public_data/experiment-runs/{run['id']}/report.json"
            destination = tmp_path / "conversations" / identity / path
            destination.parent.mkdir(parents=True)
            destination.write_bytes(body)
            service.store.managed_submission(
                owner, run["id"], scope, {"operation": "inference"}
            )
            service.store.managed_receipt(
                owner,
                run["id"],
                scope,
                "inference",
                {"task_id": "evaluated", "output_dir": "/evaluation"},
            )
            service.store.ingest(
                "evaluation-complete", "evaluated", "callback", {"status": "succeeded"}
            )
            service.store.apply_observations()
            service.store.register_artifacts(
                owner,
                run["id"],
                [
                    {
                        "name": name,
                        "uri": path if name == "report.json" else "/evaluation/" + name,
                        "sha256": hashlib.sha256(body).hexdigest(),
                    }
                    for name in ("metrics.json", "report.json", "predictions.jsonl")
                ],
                {},
            )
            service.store.complete_agent_turn(
                owner,
                run["id"],
                scope,
                "Evaluation report ready.",
                report=True,
                completed=True,
            )
            route = (
                f"{base}/{identity}/experiment-executions/"
                f"{run['id']}/artifacts/report.json"
            )
            assert (await client.get(route)).json()["content"] == body.decode()
            assert await runtime.experiment_plan_call(
                identity,
                "read_experiment_artifact",
                {"run_id": run["id"], "name": "report.json"},
                context,
            ) == {"name": "report.json", "content": body.decode()}
            association = {
                "expected_sequence": 0,
                "context": {
                    "goal": "Reference created after execution",
                    "stages": [{"id": "later-stage", "name": "Evaluation"}],
                    "stage_executions": {"later-stage": [run["id"]]},
                },
            }
            linked = await runtime.experiment_plan_call(
                identity, "update_experiment_context", association, context
            )
            assert isinstance(linked["context"], dict)
            assert linked["context"]["stage_executions"] == {"later-stage": [run["id"]]}
            assert service.store.list(owner)[0]["task_spec"]["stage_id"] is None
            foreign_session = await client.post(base, json={"llm": {"model": "test"}})
            foreign = service.request(
                owner.model_copy(
                    update={
                        "conversation_id": foreign_session.json()["conversation_id"],
                    }
                ),
                request(intent="prepare"),
                ("eval",),
            )
            with pytest.raises(ValueError, match="this conversation"):
                await runtime.experiment_plan_call(
                    identity,
                    "update_experiment_context",
                    {
                        **association,
                        "expected_sequence": 1,
                        "context": {
                            **association["context"],
                            "stage_executions": {"later-stage": [foreign["id"]]},
                        },
                    },
                    context,
                )
            executions = await runtime.experiment_plan_call(
                identity, "get_experiment_execution", {"run_id": run["id"]}, context
            )
            assert (
                json.loads(json.dumps(executions))["executions"][0]["run_id"]
                == run["id"]
            )
            assert (
                await runtime.experiment_plan_call(
                    identity, "get_experiment_execution", {"run_id": "unknown"}, context
                )
            )["executions"] == []
            with pytest.raises(FileNotFoundError):
                await runtime.experiment_plan_call(
                    identity,
                    "read_experiment_artifact",
                    {"run_id": "another-task", "name": "report.json"},
                    context,
                )
            with pytest.raises(FileNotFoundError):
                await runtime.experiment_plan_call(
                    identity,
                    "read_experiment_artifact",
                    {"run_id": run["id"], "name": "../report.md"},
                    context,
                )
            assert (
                await client.get(route, headers={"x-pyromind-debug-user-id": "99"})
            ).status_code == 404
            assert (
                await client.get(route.replace(base, "/api/v2/pyromind/conversations"))
            ).status_code == 404
            destination.write_text("changed")
            assert (await client.get(route)).status_code == 409
            service.store.queue.finish(scope["work_id"], scope["generation"], {})
            await runtime.flush_execution_events()
            await runtime.flush_execution_events()
            items = service.store.queue.list(run["owner"])
            assert all(item["kind"] != "manager_turn" for item in items)
            events = runtime._store(identity).replay()
            changes = [
                event.event_id
                for event in events
                if event.type == "experiment_execution.changed"
            ]
            assert changes and len(changes) == len(set(changes))
            with pytest.raises(PermissionError):
                await runtime.experiment_plan_call(
                    identity, "create_experiment_plan", {}, context
                )
    finally:
        await runtime.close()


def test_manager_messages_are_deduplicated_and_scoped_to_the_conversation(experiment):
    service, owner, _ = experiment
    command = UserMessageCommand(
        command_id="continue",
        content=(TextContent(text="Continue with the registered result"),),
    )
    identity = service.store.enqueue_manager_message(owner, command)
    assert service.store.enqueue_manager_message(owner, command) == identity
    foreign = owner.model_copy(update={"conversation_id": "another-conversation"})
    assert service.store.enqueue_manager_message(foreign, command) != identity
    item = service.store.queue.claim(kinds=("manager_turn",))
    assert item is not None
    assert item["payload"] == {
        "owner": owner.model_dump(mode="json"),
        "command": command.model_dump(mode="json"),
    }
    service.store.queue.finish(item["id"], item["generation"], {})
