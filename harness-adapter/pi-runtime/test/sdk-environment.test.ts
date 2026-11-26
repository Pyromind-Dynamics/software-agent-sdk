import assert from "node:assert/strict";
import test from "node:test";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";
import { platformSdkEnvironment } from "../src/sdk-environment.js";

test("SDK environment is resolved from the private host RPC without caller identity", async () => {
  const calls: unknown[] = [];
  const environment = {
    PYROMIND_API_KEY: "derived-access-key",
    PYROMIND_CLUSTER: "cn-east-1#pre2",
    PYROMIND_BASE_URL: "https://pre2-api-portal.pyromind.ai/api/v1",
    PYROMIND_SDK_PYTHON: "/opt/sdk/bin/python",
    LLM_BASE_URL: "https://model.example",
    LLM_AUTH_TOKEN: "model-access-key",
    LLM_MODEL: "session-model",
  };
  const peer = { request: async (method: string, params: unknown) => {
    calls.push([method, params]);
    return calls.length === 1 ? environment : {};
  } } as unknown as JsonlRpcPeer;
  assert.deepEqual(await platformSdkEnvironment(peer), environment);
  assert.deepEqual(await platformSdkEnvironment(peer), {});
  assert.deepEqual(calls, [
    ["platform.sdk_environment", {}], ["platform.sdk_environment", {}],
  ]);
});

test("SDK context rejects extra, incomplete and credential-bearing endpoint data", async () => {
  const valid = {
    PYROMIND_API_KEY: "derived-access-key",
    PYROMIND_CLUSTER: "us-west-1",
    PYROMIND_BASE_URL: "https://platform.example/api/v1",
    PYROMIND_SDK_PYTHON: "/opt/sdk/bin/python",
  };
  for (const value of [
    null,
    { ...valid, COOKIE: "raw-cookie-secret" },
    { PYROMIND_API_KEY: valid.PYROMIND_API_KEY },
    { PYROMIND_SDK_PYTHON: "python3" },
    { PYROMIND_SDK_PYTHON: "/opt/sdk/bin/python\0other" },
    { PYROMIND_SDK_PYTHON: valid.PYROMIND_SDK_PYTHON, LLM_AUTH_TOKEN: "orphaned-token" },
    { ...valid, PYROMIND_SDK_PYTHON: "python3" },
    { ...valid, PYROMIND_BASE_URL: "invalid derived-access-key" },
    { ...valid, PYROMIND_BASE_URL: "https://user:derived-access-key@platform.example" },
    { ...valid, PYROMIND_SANDBOX_DEFAULT_IMAGE: "" },
    { ...valid, PYROMIND_SANDBOX_DEFAULT_IMAGE: " \t" },
    { ...valid, PYROMIND_SANDBOX_DEFAULT_IMAGE: 123 },
    { ...valid, PYROMIND_SANDBOX_DEFAULT_IMAGE: "image\0name" },
    { ...valid, PYROMIND_TASK_ASSOCIATIONS: "" },
    { ...valid, PYROMIND_TASK_ASSOCIATIONS: " \t" },
    { ...valid, PYROMIND_TASK_ASSOCIATIONS: [] },
    { ...valid, PYROMIND_TASK_ASSOCIATIONS: "[]\0" },
    { ...valid, PYROMIND_TASK_CONTEXTS: "" },
    { ...valid, PYROMIND_TASK_CONTEXTS: {} },
    { ...valid, PYROMIND_PROJECT_EXECUTIONS: [] },
    { ...valid, PYROMIND_PROJECT_EXECUTIONS: "[]\0" },
    { ...valid, PYROMIND_DEBUG_STATE_PATH: "" },
    { ...valid, PYROMIND_DEBUG_STATE_PATH: "public_data/budget.json" },
    { ...valid, LLM_AUTH_TOKEN: "" },
    { ...valid, LLM_AUTH_TOKEN: 123 },
    { ...valid, LLM_BASE_URL: "model\0url" },
  ]) {
    const peer = { request: async () => value } as unknown as JsonlRpcPeer;
    await assert.rejects(platformSdkEnvironment(peer), (error: unknown) => {
      assert(error instanceof Error);
      assert.match(error.message, /platform SDK/);
      assert(!error.message.includes(valid.PYROMIND_API_KEY));
      assert(!error.message.includes("raw-cookie-secret"));
      return true;
    });
  }
});

test("local SDK preparation accepts an interpreter without platform credentials", async () => {
  const environment = { PYROMIND_SDK_PYTHON: "/opt/sdk/bin/python" };
  const peer = { request: async () => environment } as unknown as JsonlRpcPeer;
  assert.deepEqual(await platformSdkEnvironment(peer), environment);
});

test("SDK context preserves host task routing and Project execution references", async () => {
  const environment = {
    PYROMIND_API_KEY: "derived-access-key",
    PYROMIND_CLUSTER: "us-west-1",
    PYROMIND_BASE_URL: "https://platform.example/api/v1",
    PYROMIND_SDK_PYTHON: "/opt/sdk/bin/python",
    PYROMIND_SANDBOX_DEFAULT_IMAGE: "registry.example/sdk:stable",
    PYROMIND_TASK_ASSOCIATIONS: JSON.stringify([{ task_id: "task-1", run_id: "run-1", output_dir: "/runs/1" }]),
    PYROMIND_TASK_CONTEXTS: JSON.stringify({ workflow_run: "server-signed-workflow-context" }),
    PYROMIND_PROJECT_EXECUTIONS: JSON.stringify([{
      project_id: "project-1", task_id: "train", attempt: 2,
      out_ids: { workflow_run: "server-signed-project-context" },
    }]),
    PYROMIND_DEBUG_STATE_PATH: "/workspace/conversations/current/pi/business-state.json",
  };
  const peer = { request: async () => environment } as unknown as JsonlRpcPeer;
  assert.deepEqual(await platformSdkEnvironment(peer), environment);
});
