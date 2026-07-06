from typing import Any, cast

from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.tools.workflow import (
    DslToXyflowAction,
    DslToXyflowExecutor,
    DslToXyflowObservation,
    DslToXyflowTool,
)


class _FakeDslConverter:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def from_python(self, code: str, *, name: str) -> Any:
        self.calls.append((code, name))
        return self.result


def test_dsl_to_xyflow_converts_with_sdk_converter() -> None:
    xyflow = {
        "name": "Dataset Processing Test",
        "nodes": [{"id": "2", "data": {"nodeType": "CloneAndCacheDataset"}}],
        "edges": [{"source": "2", "target": "3"}],
    }
    converter = _FakeDslConverter(xyflow)
    executor = DslToXyflowExecutor(converter_factory=lambda: converter)

    observation = executor(
        DslToXyflowAction(
            dsl=(
                "# workflow: Dataset Processing Test\n"
                'na1b2c3d = CloneAndCacheDataset(dataset="openai/gsm8k")\n'
            ),
            name="Dataset Processing Test",
        )
    )

    assert isinstance(observation, DslToXyflowObservation)
    assert not observation.is_error
    assert observation.xyflow == xyflow
    assert observation.workflow_name == "Dataset Processing Test"
    assert observation.node_count == 1
    assert observation.edge_count == 1
    assert converter.calls == [
        (
            "# workflow: Dataset Processing Test\n"
            'na1b2c3d = CloneAndCacheDataset(dataset="openai/gsm8k")\n',
            "Dataset Processing Test",
        )
    ]
    assert "Workflow DSL converted to xyflow JSON" in observation.text
    assert '"nodes": [' in observation.text


def test_dsl_to_xyflow_reports_converter_errors() -> None:
    def raise_sdk_error() -> Any:
        raise RuntimeError("DslConverter unavailable")

    observation = DslToXyflowExecutor(converter_factory=raise_sdk_error)(
        DslToXyflowAction(dsl="# workflow: demo\n", name="demo")
    )

    assert observation.is_error
    assert "DslConverter unavailable" in observation.text


def test_dsl_to_xyflow_rejects_non_object_converter_result() -> None:
    observation = DslToXyflowExecutor(
        converter_factory=lambda: _FakeDslConverter(["not", "a", "dict"])
    )(DslToXyflowAction(dsl="# workflow: demo\n", name="demo"))

    assert observation.is_error
    assert "expected dict" in observation.text


def test_dsl_to_xyflow_tool_is_explicitly_available() -> None:
    tool = DslToXyflowTool.create()[0]
    assert tool.name == "dsl_to_xyflow"
    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.openWorldHint is False
    assert {"dsl", "name"} <= set(tool.action_type.model_fields)

    resolved = resolve_tool(
        Tool(name="dsl_to_xyflow"),
        cast(Any, None),
    )
    assert isinstance(resolved[0], DslToXyflowTool)
