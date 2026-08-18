from pyromind_runtime.contracts.tools import ToolSpec


PREVIEW_DATASET_TOOL_SPEC = ToolSpec(
    name="preview_dataset",
    description=(
        "Inspect a Pyromind dataset path and return bounded schema, statistics, "
        "and representative sample records."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "dataset_path": {"type": "string", "minLength": 1},
            "n": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": ["dataset_path"],
        "additionalProperties": False,
    },
    timeout_seconds=30,
    risk_level="low",
)

VALIDATE_WORKFLOW_DSL_TOOL_SPEC = ToolSpec(
    name="validate_workflow_dsl",
    description=(
        "Validate Pyromind workflow Python DSL and return structured platform "
        "validation issues."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "dsl": {"type": "string", "minLength": 1},
            "name": {"type": "string", "minLength": 1},
        },
        "required": ["dsl"],
        "additionalProperties": False,
    },
    timeout_seconds=30,
    risk_level="low",
)


def first_version_tool_specs() -> tuple[ToolSpec, ...]:
    return (PREVIEW_DATASET_TOOL_SPEC, VALIDATE_WORKFLOW_DSL_TOOL_SPEC)
