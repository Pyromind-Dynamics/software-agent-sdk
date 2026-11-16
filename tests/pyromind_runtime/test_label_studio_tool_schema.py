"""The shared Label Studio schema must survive Pi's actual argument pipeline."""

import json
import subprocess
from pathlib import Path

import pytest
from harness_adapter.pi_adapter.business_tool_host import (
    PyromindBusinessToolHost,
    ToolExecutionContext,
)
from pyromind_runtime.domain.context import RequestContext

from openhands.tools.label_studio.definition import LabelStudioProjectAction


@pytest.fixture
def host():
    repository = Path(__file__).resolve().parents[2]
    return PyromindBusinessToolHost(
        [
            repository / ".agents/skills/data-processing",
            repository / ".agents/skills/training-analysis",
        ]
    )


def test_pi_label_studio_operation_schemas_require_the_right_arguments(host):
    repository = Path(__file__).resolve().parents[2]
    specs = {
        item["name"]: item
        for item in host.specs(sandbox=True)
        if item["name"].startswith("label_studio_")
    }
    assert "label_studio_project" not in specs
    assert set(specs) == {
        "label_studio_create_project",
        "label_studio_get_project",
        "label_studio_project_status",
        "label_studio_update_config",
        "label_studio_export_annotations",
        "label_studio_refresh_media",
    }
    runtime = repository / "harness-adapter/pi-runtime"
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            """
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { makeStrictJsonSchema } from "@earendil-works/pi-ai/api/constrained-sampling";
import { validateToolArguments } from "@earendil-works/pi-ai/utils/validation";
const specs = JSON.parse(readFileSync(0, "utf8"));
const valid = {
  label_studio_create_project: {
  dataset_path: "/datasets/pcb/badcase_step320_ls.jsonl",
  label_config_path: "public_data/label_config_badcase.xml",
  field_map_path: "public_data/field_map.json",
  adapter: "jsonl",
  idempotency_key: "badcase-c320-binary",
  },
  label_studio_get_project: { project_ref: "existing-project" },
  label_studio_project_status: { project_ref: "existing-project" },
  label_studio_update_config: {
    project_ref: "existing-project", label_config_path: "config.xml",
    expected_config_version: 2,
  },
  label_studio_export_annotations: { project_ref: "existing-project" },
  label_studio_refresh_media: { project_ref: "existing-project" },
};
const required = {
  label_studio_create_project: ["dataset_path", "label_config_path"],
  label_studio_get_project: ["project_ref"],
  label_studio_project_status: ["project_ref"],
  label_studio_update_config: ["project_ref", "label_config_path"],
  label_studio_export_annotations: ["project_ref"],
  label_studio_refresh_media: ["project_ref"],
};
for (const spec of Object.values(specs)) {
  const tool = { name: spec.name, parameters: spec.input_schema };
  const strict = makeStrictJsonSchema(tool.parameters);
  assert.deepEqual(strict.required, Object.keys(tool.parameters.properties));
  assert.deepEqual(spec.input_schema.required, required[spec.name]);
  const call = args => ({ type: "toolCall", id: "test", name: spec.name,
    arguments: args });
  assert.deepEqual(
    validateToolArguments(tool, call(valid[spec.name])), valid[spec.name]);
  const nullable = Object.fromEntries(Object.keys(strict.properties).map(
    field => [field, valid[spec.name][field] ?? null]));
  assert.deepEqual(
    validateToolArguments(tool, call(nullable)), nullable);
  for (const field of required[spec.name]) {
    const incomplete = { ...valid[spec.name] };
    delete incomplete[field];
    assert.throws(() => validateToolArguments(tool, call(incomplete)));
  }
}
assert.throws(() => validateToolArguments(
  { name: specs.label_studio_create_project.name,
    parameters: specs.label_studio_create_project.input_schema },
  { type: "toolCall", id: "test", name: "label_studio_create_project",
    arguments: { output_path: "label_config.xml" } }));
assert.throws(() => validateToolArguments(
  { name: specs.label_studio_export_annotations.name,
    parameters: specs.label_studio_export_annotations.input_schema },
  { type: "toolCall", id: "test", name: "label_studio_export_annotations",
    arguments: { output_path: "public_data/export" } }));
process.stdout.write(JSON.stringify(valid.label_studio_create_project));
""",
        ],
        input=json.dumps(specs),
        text=True,
        capture_output=True,
        cwd=runtime,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    action = LabelStudioProjectAction.model_validate(
        {"operation": "create", **json.loads(result.stdout)}
    )
    assert action.adapter == "jsonl"
    assert action.dataset_path == "/datasets/pcb/badcase_step320_ls.jsonl"
    assert action.label_config_path == "public_data/label_config_badcase.xml"
    assert action.field_map_path == "public_data/field_map.json"
    assert action.idempotency_key == "badcase-c320-binary"


@pytest.mark.parametrize(
    ("tool_name", "arguments", "missing_fields"),
    [
        (
            "label_studio_create_project",
            {"output_path": "public_data/label_config_badcase.xml"},
            ("dataset_path", "label_config_path", "output_path"),
        ),
        (
            "label_studio_export_annotations",
            {"output_path": "public_data/export"},
            ("project_ref",),
        ),
    ],
)
async def test_pi_rejects_invalid_operation_arguments_before_execution(
    host, tmp_path, tool_name, arguments, missing_fields
):
    result = await host.execute(
        tool_name,
        arguments,
        ToolExecutionContext(
            conversation_id="label-schema-regression",
            workspace_root=tmp_path,
            request_context=RequestContext(user_id="test"),
            model_configuration={},
        ),
        tool_call_id="incomplete-create",
    )
    assert result["is_error"] is True
    assert result["details"]["error_code"] == "invalid_tool_arguments"
    text = result["content"][0]["text"]
    for field in missing_fields:
        assert field in text


@pytest.mark.parametrize(
    "dataset_path", [None, "/datasets/pcb/badcase_step320_ls.jsonl"]
)
async def test_pi_incomplete_create_reports_executor_input_errors(
    host, tmp_path, dataset_path
):
    # Replay both malformed shapes from conversation 2844c937. The shared
    # executor must report the missing inputs, without touching platform APIs.
    result = await host.execute(
        "label_studio_project",
        {
            "operation": "create",
            "adapter": "jsonl",
            "dataset_path": dataset_path,
            "output_path": "public_data/label_config_badcase.xml",
        },
        ToolExecutionContext(
            conversation_id="label-schema-regression",
            workspace_root=tmp_path,
            request_context=RequestContext(user_id="test"),
            model_configuration={},
        ),
        tool_call_id="incomplete-create",
    )
    assert result["is_error"] is True
    text = "\n".join(block["text"] for block in result["content"])
    assert "label_config_path is required" in text
    if dataset_path is None:
        assert "dataset_path is required" in text
    assert "output_path is only used for operation='export'" in text
    assert "oneOf" not in text
