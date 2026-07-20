"""Tests for TerminalTool subclass."""

import tempfile
from pathlib import Path
from uuid import uuid4

from pydantic import SecretStr

from openhands.sdk.agent import Agent
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.llm import LLM
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.terminal import (
    TerminalAction,
    TerminalObservation,
    TerminalTool,
)
from openhands.tools.terminal.impl import TerminalExecutor


def _create_test_conv_state(temp_dir: str) -> ConversationState:
    """Helper to create a test conversation state."""
    llm = LLM(model="gpt-4o-mini", api_key=SecretStr("test-key"), usage_id="test-llm")
    agent = Agent(llm=llm, tools=[])
    return ConversationState.create(
        id=uuid4(),
        agent=agent,
        workspace=LocalWorkspace(working_dir=temp_dir),
    )


def test_bash_tool_initialization():
    """Test that TerminalTool initializes correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = TerminalTool.create(conv_state)
        tool = tools[0]

        # Check that the tool has the correct name and properties
        assert tool.name == "terminal"
        assert tool.executor is not None
        assert tool.action_type == TerminalAction


def test_bash_tool_with_username():
    """Test that TerminalTool initializes correctly with username."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = TerminalTool.create(conv_state, username="testuser")
        tool = tools[0]

        # Check that the tool has the correct name and properties
        assert tool.name == "terminal"
        assert tool.executor is not None
        assert tool.action_type == TerminalAction


def test_bash_tool_execution():
    """Test that TerminalTool can execute commands."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = TerminalTool.create(conv_state)
        tool = tools[0]

        # Create an action
        action = TerminalAction(command="echo 'Hello, World!'")

        # Execute the action
        result = tool(action)

        # Check the result
        assert result is not None
        assert isinstance(result, TerminalObservation)
        assert "Hello, World!" in result.text


def test_bash_tool_working_directory():
    """Test that TerminalTool respects the working directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = TerminalTool.create(conv_state)
        tool = tools[0]

        # Create an action to check current directory
        action = TerminalAction(command="pwd")

        # Execute the action
        result = tool(action)

        # Check that the working directory is correct
        assert isinstance(result, TerminalObservation)
        assert temp_dir in result.text


def test_terminal_sandbox_override_preserves_persistent_working_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        public_data = Path(temp_dir) / "public_data"
        public_data.mkdir()
        conv_state = _create_test_conv_state(temp_dir)
        tool = TerminalTool.create(
            conv_state,
            terminal_type="subprocess",
            sandbox_mode="off",
        )[0]
        executor = tool.executor
        assert isinstance(executor, TerminalExecutor)

        try:
            first = tool(TerminalAction(command="cd public_data"))
            second = tool(TerminalAction(command="echo hello"))
        finally:
            executor.close()

        assert isinstance(first, TerminalObservation)
        assert isinstance(second, TerminalObservation)
        assert executor._sandbox_mode == "off"
        assert first.command == "cd public_data"
        assert second.command == "echo hello"
        assert second.text.strip() == "hello"
        assert second.metadata is not None
        assert second.metadata.working_dir is not None
        assert Path(second.metadata.working_dir).resolve() == public_data.resolve()
        assert "chdir" not in first.text + second.text
        assert "getcwd" not in first.text + second.text


def test_bash_tool_to_openai_tool():
    """Test that TerminalTool can be converted to OpenAI tool format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = TerminalTool.create(conv_state)
        tool = tools[0]

        # Convert to OpenAI tool format
        openai_tool = tool.to_openai_tool()

        # Check the format
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "terminal"
        assert "description" in openai_tool["function"]
        assert "parameters" in openai_tool["function"]
