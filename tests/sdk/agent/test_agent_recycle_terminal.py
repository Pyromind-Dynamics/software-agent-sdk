from collections.abc import Sequence
from typing import ClassVar

from openhands.sdk import LLM, Conversation
from openhands.sdk.agent import Agent
from openhands.sdk.llm.message import TextContent
from openhands.sdk.tool import ToolDefinition
from openhands.sdk.tool.tool import Action, Observation, ToolExecutor


class _Action(Action):
    text: str


class _Obs(Observation):
    out: str

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [TextContent(text=self.out)]


class _RecorderExecutor(ToolExecutor[_Action, _Obs]):
    def __init__(self) -> None:
        self.closed = 0

    def __call__(self, action: _Action, conversation=None) -> _Obs:
        return _Obs(out=action.text)

    def close(self) -> None:
        self.closed += 1


class _TerminalTool(ToolDefinition[_Action, _Obs]):
    name: ClassVar[str] = "terminal"

    @classmethod
    def create(cls, conv_state=None, **params):
        return [
            cls(
                description="t",
                action_type=_Action,
                observation_type=_Obs,
                **params,
            )
        ]


class _OtherTool(ToolDefinition[_Action, _Obs]):
    name: ClassVar[str] = "other"

    @classmethod
    def create(cls, conv_state=None, **params):
        return [
            cls(
                description="o",
                action_type=_Action,
                observation_type=_Obs,
                **params,
            )
        ]


def _agent_with_runtime_tools(
    terminal_exec: _RecorderExecutor, other_exec: _RecorderExecutor
) -> Agent:
    llm = LLM(model="test-model", usage_id="test-llm")
    agent = Agent(llm=llm, tools=[], include_default_tools=[])
    conv = Conversation(agent=agent, visualizer=None)
    conv._ensure_agent_ready()
    agent.add_runtime_tools(
        [
            _TerminalTool(
                description="t",
                action_type=_Action,
                observation_type=_Obs,
                executor=terminal_exec,
            ),
            _OtherTool(
                description="o",
                action_type=_Action,
                observation_type=_Obs,
                executor=other_exec,
            ),
        ]
    )
    return agent


def test_recycle_terminal_closes_terminal_executor_only():
    terminal_exec = _RecorderExecutor()
    other_exec = _RecorderExecutor()
    agent = _agent_with_runtime_tools(terminal_exec, other_exec)

    agent.recycle_terminal()

    assert terminal_exec.closed == 1
    assert other_exec.closed == 0


def test_recycle_terminal_is_noop_without_initialization():
    agent = Agent(llm=LLM(model="test", usage_id="test-llm"), tools=[])
    agent.recycle_terminal()  # must not raise before init


def test_recycle_terminal_after_run_env_toggle(monkeypatch):
    from openhands.sdk.conversation.impl.local_conversation import (
        recycle_terminal_after_run_enabled,
    )

    monkeypatch.delenv("OH_TERMINAL_RECYCLE_AFTER_RUN", raising=False)
    assert recycle_terminal_after_run_enabled() is True

    monkeypatch.setenv("OH_TERMINAL_RECYCLE_AFTER_RUN", "0")
    assert recycle_terminal_after_run_enabled() is False
