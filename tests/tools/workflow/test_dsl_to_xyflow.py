from typing import Any, cast

from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.tools.workflow import (
    DslToXyflowAction,
    DslToXyflowExecutor,
    DslToXyflowObservation,
    DslToXyflowTool,
    convert_dsl_to_xyflow,
    convert_xyflow_to_dsl,
)


class _FakeDslConverter:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []
        self.workflows: list[dict[str, Any]] = []

    def from_python(self, code: str, *, name: str) -> Any:
        self.calls.append((code, name))
        return self.result

    def to_python(self, workflow: dict[str, Any]) -> Any:
        self.workflows.append(workflow)
        self.calls.append(("xyflow", workflow["name"]))
        return self.result


class _FakeBadXyflowConverter:
    def to_python(self, workflow: dict[str, Any]) -> Any:
        return {"not": "dsl"}


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
    assert observation.xyflow == {
        "name": "Dataset Processing Test",
        "nodes": [
            {
                "id": "2",
                "type": "default",
                "data": {"nodeType": "CloneAndCacheDataset"},
            }
        ],
        "edges": [{"source": "2", "target": "3"}],
    }
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


def test_convert_dsl_to_xyflow_rejects_non_object_result() -> None:
    observation = DslToXyflowExecutor(
        converter_factory=lambda: _FakeDslConverter(["not", "a", "dict"])
    )(DslToXyflowAction(dsl="# workflow: demo\n", name="demo"))

    assert observation.is_error
    assert "expected dict" in observation.text


def test_convert_dsl_to_xyflow_normalizes_minimal_sdk_xyflow() -> None:
    xyflow = {
        "name": "demo",
        "nodes": [
            {
                "id": "1",
                "data": {
                    "nodeType": "CloneAndCacheDataset",
                    "config": {
                        "dataset": "pyromind/self-cognition",
                        "target_path": "/workspace/datasets/",
                    },
                },
            },
            {
                "id": "2",
                "data": {"nodeType": "LoadDatasetFile", "config": {}},
            },
        ],
        "edges": [
            {
                "source": "1",
                "target": "2",
                "sourceHandle": "dataset",
                "targetHandle": "dataset",
            }
        ],
    }

    normalized = convert_dsl_to_xyflow(
        "# workflow: demo\n",
        name="demo",
        converter_factory=lambda: _FakeDslConverter(xyflow),
    )

    first_node = normalized["nodes"][0]
    assert first_node["type"] == "default"
    assert first_node["data"]["nodeDefinition"] == {
        "input": {
            "required": {},
            "optional": {
                "dataset": {},
                "target_path": {},
            },
        },
        "output_name": ["dataset"],
    }
    second_node = normalized["nodes"][1]
    assert second_node["type"] == "default"
    assert second_node["data"]["nodeDefinition"]["input"]["optional"] == {"dataset": {}}
    assert "type" not in xyflow["nodes"][0]


def test_convert_xyflow_to_dsl_uses_sdk_converter() -> None:
    converter = _FakeDslConverter("# workflow: demo\n")

    dsl = convert_xyflow_to_dsl(
        {"name": "demo", "nodes": [], "edges": []},
        converter_factory=lambda: converter,
    )

    assert dsl == "# workflow: demo\n"
    assert converter.calls == [("xyflow", "demo")]


def test_convert_xyflow_to_dsl_normalizes_before_sdk_converter() -> None:
    xyflow = {
        "name": "demo",
        "nodes": [
            {
                "id": "1",
                "data": {
                    "nodeType": "CloneAndCacheDataset",
                    "config": {
                        "dataset": "pyromind/self-cognition",
                        "target_path": "/workspace/datasets/",
                    },
                },
            }
        ],
        "edges": [],
    }
    converter = _FakeDslConverter("# workflow: demo\n")

    convert_xyflow_to_dsl(xyflow, converter_factory=lambda: converter)

    normalized = converter.workflows[0]
    assert normalized["nodes"][0]["type"] == "default"
    assert normalized["nodes"][0]["data"]["nodeDefinition"]["input"]["optional"] == {
        "dataset": {},
        "target_path": {},
    }
    assert "type" not in xyflow["nodes"][0]


def test_convert_xyflow_to_dsl_rejects_non_string_result() -> None:
    try:
        convert_xyflow_to_dsl(
            {"name": "demo"},
            converter_factory=lambda: _FakeBadXyflowConverter(),
        )
    except TypeError as exc:
        assert "expected str" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_convert_xyflow_to_dsl_rejects_invalid_python_result() -> None:
    try:
        convert_xyflow_to_dsl(
            {"name": "demo", "nodes": [], "edges": []},
            converter_factory=lambda: _FakeDslConverter("n1 = (id=1)"),
        )
    except SyntaxError:
        pass
    else:
        raise AssertionError("Expected SyntaxError")


def test_dsl_to_xyflow_reports_converter_errors() -> None:
    def raise_sdk_error() -> Any:
        raise RuntimeError("DslConverter unavailable")

    observation = DslToXyflowExecutor(converter_factory=raise_sdk_error)(
        DslToXyflowAction(dsl="# workflow: demo\n", name="demo")
    )

    assert observation.is_error
    assert "DslConverter unavailable" in observation.text


def test_dsl_to_xyflow_rejects_non_object_converter_result() -> None:
    try:
        convert_dsl_to_xyflow(
            "# workflow: demo\n",
            name="demo",
            converter_factory=lambda: _FakeDslConverter(["not", "a", "dict"]),
        )
    except TypeError as exc:
        assert "expected dict" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


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
