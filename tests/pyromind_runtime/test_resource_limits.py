from __future__ import annotations

import pytest
from harness_adapter.openhands_adapter.adapter import OPENHANDS_CAPABILITIES
from harness_adapter.pi_adapter.adapter import PI_CAPABILITIES, _session_config
from pydantic import ValidationError
from pyromind_agent_server.bootstrap import resource_limits_from_environment
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import ResourceLimits
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter


def test_resource_limits_rejects_unusable_values() -> None:
    with pytest.raises(ValidationError):
        ResourceLimits(memory_limit_bytes=0)
    with pytest.raises(ValidationError):
        ResourceLimits(nproc_limit=1)


def test_resource_limits_read_single_configuration_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OH_SANDBOX_VMEM_LIMIT", "1G")
    monkeypatch.setenv("OH_SANDBOX_NPROC_LIMIT", "4")

    limits = resource_limits_from_environment()

    assert limits == ResourceLimits(
        memory_limit_bytes=1024**3,
        nproc_limit=4,
    )


@pytest.mark.parametrize(
    ("memory_raw", "nproc_raw"),
    [("bad", "2"), ("500M", "bad"), ("500M", "1")],
)
def test_resource_limits_reject_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
    memory_raw: str,
    nproc_raw: str,
) -> None:
    monkeypatch.setenv("OH_SANDBOX_VMEM_LIMIT", memory_raw)
    monkeypatch.setenv("OH_SANDBOX_NPROC_LIMIT", nproc_raw)

    with pytest.raises(RuntimeError):
        resource_limits_from_environment()


def test_resource_limits_are_a_cross_harness_contract() -> None:
    assert OPENHANDS_CAPABILITIES.enforced_limits == {"memory", "nproc"}
    assert PI_CAPABILITIES.enforced_limits == {"memory", "nproc"}

    spec = SessionSpec(
        conversation_id="conversation-1",
        user_id="42",
        workspace_root="/tmp/conversation-1",
        resource_limits=ResourceLimits(
            memory_limit_bytes=256 * 1024 * 1024,
            nproc_limit=3,
        ),
    )

    assert spec.resource_limits is not None
    assert spec.resource_limits.model_dump(mode="json") == {
        "memory_limit_bytes": 256 * 1024 * 1024,
        "nproc_limit": 3,
    }


def test_pi_session_config_carries_resource_limits() -> None:
    limits = ResourceLimits(memory_limit_bytes=256 * 1024 * 1024, nproc_limit=3)
    config = _session_config(
        SessionSpec(
            conversation_id="conversation-1",
            user_id="42",
            workspace_root="/tmp/conversation-1",
            resource_limits=limits,
        )
    )

    assert config["resource_limits"] == limits.model_dump(mode="json")


async def test_runtime_injects_resource_limits_into_harness_contract(
    tmp_path,
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(
        conversations,
        adapter,
        resource_limits=ResourceLimits(memory_limit_bytes=256 * 1024 * 1024),
    )

    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="conversation-1",
                user_id="42",
                workspace_root=str(conversations),
            ),
            RequestContext(user_id="42"),
        )
    finally:
        await runtime.close()

    assert adapter.created_specs[0].resource_limits == ResourceLimits(
        memory_limit_bytes=256 * 1024 * 1024
    )
