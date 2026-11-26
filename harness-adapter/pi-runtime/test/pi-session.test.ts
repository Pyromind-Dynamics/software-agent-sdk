import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, writeFile, rm, realpath } from "node:fs/promises";
import { join } from "node:path";
import { createPiSession, modelRetrySettings } from "../src/pi-session.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";
import { fixture, text } from "./fixtures/no-spin.js";

test("the first notification of each phase uses its prompt and skills without touching the sandbox", async (t) => {
  const { config, peer, requests, root } = await fixture(t, (_, response) => text(response, "Ready."));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const name of ["planning", "execution"]) {
    await mkdir(join(root, "skills", name));
    await writeFile(join(root, "skills", name, "SKILL.md"), `---\nname: ${name}\ndescription: A test skill.\n---\n`);
  }
  const rpc = t.mock.method(peer, "request");
  for (const [enabled, disabled] of [["planning", "execution"], ["execution", "planning"]]) {
    const prompt = `Use the ${enabled} contract. Storage is mounted at /workspace.`;
    const { session } = await createPiSession({
      ...config, tool_allowlist: ["read", "terminal", "evidence"],
      skill_allowlist: [enabled], system_prompt: prompt,
    }, peer);
    try {
      await session.sendCustomMessage({
        customType: "pyromind.external_task", content: "Requirement queued.", display: false,
      }, { triggerTurn: true });
      const request = requests.at(-1);
      assert.deepEqual(request.tools.map((tool: { function: { name: string } }) => tool.function.name).sort(),
        ["evidence", "read", "terminal"]);
      const system = request.messages[0].content;
      assert.ok(system.includes(prompt));
      assert.ok(system.includes(`<location>.agents/skills/${enabled}/SKILL.md</location>`));
      assert.ok(!system.includes(disabled));
      assert.doesNotMatch(system, /\/tmp\/pi-no-spin-/);
    } finally {
      session.dispose();
    }
  }
  assert.equal(requests.length, 2);
  assert.ok(!rpc.mock.calls.some(call => call.arguments[0].startsWith("sandbox.")));
});

test("model retries default to a small budget on both layers", () => {
  assert.deepEqual(modelRetrySettings({}), {
    enabled: true,
    maxRetries: 2,
    baseDelayMs: 2_000,
    provider: { maxRetries: 2 },
  });
});

test("model retries come from the environment when configured", () => {
  assert.deepEqual(modelRetrySettings({
    PYROMIND_PI_MODEL_MAX_RETRIES: " 4 ",
    PYROMIND_PI_MODEL_RETRY_BASE_DELAY_MS: "500",
  }), {
    enabled: true,
    maxRetries: 4,
    baseDelayMs: 500,
    provider: { maxRetries: 4 },
  });
  assert.equal(modelRetrySettings({ PYROMIND_PI_MODEL_MAX_RETRIES: "0" }).enabled, false);
  assert.throws(() => modelRetrySettings({ PYROMIND_PI_MODEL_MAX_RETRIES: "many" }),
    /must be a non-negative integer/);
  assert.throws(() => modelRetrySettings({ PYROMIND_PI_MODEL_RETRY_BASE_DELAY_MS: "-1" }),
    /must be a non-negative integer/);
});

test("manager retrieval uses resource mirrors and the sandbox, never the host workspace", async (t) => {
  const root = await realpath(await mkdtemp("/tmp/pi-manager-retrieval-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "pi", "terminal-output"), { recursive: true });
  const skills = join(root, "skills");
  for (const name of ["allowed", "denied"]) {
    await mkdir(join(skills, name), { recursive: true });
    await writeFile(join(skills, name, "SKILL.md"),
      `---\nname: ${name}\ndescription: Test resource\n---\nRESOURCE_EVIDENCE\n`);
  }
  let ensures = 0;
  const peer = {
    emit: () => {},
    request: async (method: string) => {
      assert.equal(method, "sandbox.ensure");
      ensures++;
      throw new Error("REMOTE_STORAGE_REACHED");
    },
  } as unknown as JsonlRpcPeer;
  const { session, subagents } = await createPiSession({
    session_id: "manager-retrieval", workspace_root: root, terminal_backend: "sandbox",
    session_path: join(root, "pi", "session.jsonl"), system_prompt: "Inspect resources",
    skills_directory: skills, skill_roots: [], skill_allowlist: ["allowed"], tools: [],
    tool_allowlist: ["read", "write", "edit", "terminal", "grep", "find", "ls"],
    model: { provider: "openai", id: "custom-chat-model", api_key: "test",
      base_url: "http://127.0.0.1:1/v1" },
  }, peer);
  t.after(async () => {
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose();
  });
  assert.equal(subagents, undefined);
  assert.equal(session.getToolDefinition("subagent"), undefined);
  const context = session.extensionRunner.createContext();
  for (const name of ["grep", "find", "ls"]) {
    const tool = session.getToolDefinition(name)!;
    assert.match(tool.description, /authorized Storage/);
    const result = await tool.execute(name, {
      path: ".agents/skills/allowed", ...(name === "grep" ? { pattern: "RESOURCE_EVIDENCE" } : {}),
    }, undefined, undefined, context);
    assert.match(JSON.stringify(result.content), name === "grep" ? /RESOURCE_EVIDENCE/ : /SKILL.md/);
    await assert.rejects(() => tool.execute("denied", {
      path: ".agents/skills/denied", ...(name === "grep" ? { pattern: "RESOURCE_EVIDENCE" } : {}),
    }, undefined, undefined, context), /not enabled/);
  }
  assert.ok(session.getToolDefinition("terminal"));
  assert.equal(ensures, 0);
  await assert.rejects(() => session.getToolDefinition("ls")!.execute("storage", {
    path: "/target-workspace/datasets",
  }, undefined, undefined, context), /REMOTE_STORAGE_REACHED/);
  assert.equal(ensures, 1);
});
