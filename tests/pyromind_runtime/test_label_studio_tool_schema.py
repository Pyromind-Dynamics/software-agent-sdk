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


def test_pi_label_studio_schema_preserves_create_and_lifecycle_arguments(host):
    repository = Path(__file__).resolve().parents[2]
    spec = next(
        item
        for item in host.specs(sandbox=True)
        if item["name"] == "label_studio_project"
    )
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
const spec = JSON.parse(readFileSync(0, "utf8"));
const tool = { name: spec.name, parameters: spec.input_schema };
const strict = makeStrictJsonSchema(tool.parameters);
assert.deepEqual(strict.required, Object.keys(tool.parameters.properties));
const create = {
  operation: "create",
  dataset_path: "/datasets/pcb/badcase_step320_ls.jsonl",
  label_config_path: "public_data/label_config_badcase.xml",
  field_map_path: "public_data/field_map.json",
  adapter: "jsonl",
  idempotency_key: "badcase-c320-binary",
};
const call = arguments_ => ({ type: "toolCall", id: "test", name: spec.name,
  arguments: arguments_ });
const original = structuredClone(create);
assert.deepEqual(validateToolArguments(tool, call(create)), original);
assert.deepEqual(create, original);
// Strict providers emit null for unused lifecycle parameters. Pi normalizes
// them without changing any of the supplied create inputs.
const nullable = Object.fromEntries(Object.keys(strict.properties).map(
  name => [name, create[name] ?? null]));
assert.deepEqual(validateToolArguments(tool, call(nullable)), create);
const incomplete = { operation: "create", adapter: "jsonl",
  idempotency_key: "badcase-c320-binary",
  output_path: "public_data/label_config_badcase.xml" };
assert.deepEqual(validateToolArguments(tool, call(incomplete)), incomplete);
for (const operation of ["get", "status", "export", "refresh_media"]) {
  const args = { operation, project_ref: "existing-project" };
  assert.deepEqual(validateToolArguments(tool, call(args)), args);
}
process.stdout.write(JSON.stringify(create));
""",
        ],
        input=json.dumps(spec),
        text=True,
        capture_output=True,
        cwd=runtime,
        timeout=30,
        check=True,
    )
    action = LabelStudioProjectAction.model_validate_json(result.stdout)
    assert action.adapter == "jsonl"
    assert action.dataset_path == "/datasets/pcb/badcase_step320_ls.jsonl"
    assert action.label_config_path == "public_data/label_config_badcase.xml"
    assert action.field_map_path == "public_data/field_map.json"
    assert action.idempotency_key == "badcase-c320-binary"


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
