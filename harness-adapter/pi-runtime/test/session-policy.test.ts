import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, symlink, rm } from "node:fs/promises";
import { realpathSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { capabilityCeiling, restrictSkillReads } from "../src/session-policy.js";

test("conversation capability ceiling hides plugin tools and rejects direct calls", () => {
  const handlers = new Map<string, (...args: any[]) => any>();
  let active: string[] = [];
  const pi = {
    on: (name: string, handler: any) => handlers.set(name, handler),
    getAllTools: () => ["read", "create_experiment_plan", "terminal", "subagent", "genome_run"].map(name => ({ name })),
    setActiveTools: (names: string[]) => { active = names; },
  } as unknown as ExtensionAPI;
  capabilityCeiling(["read", "create_experiment_plan"])(pi);
  handlers.get("session_start")!();
  assert.deepEqual(active, ["read", "create_experiment_plan"]);
  assert.equal(handlers.get("tool_call")!({ toolName: "subagent" }).block, true);
  assert.equal(handlers.get("tool_call")!({ toolName: "read" }), undefined);
  active.push("terminal");
  handlers.get("before_agent_start")!();
  assert.deepEqual(active, ["read", "create_experiment_plan"]);
});

test("skill capability ceiling enforces reads through aliases, absolute paths and symlinks", async t => {
  const root = realpathSync(await mkdtemp("/tmp/pi-skill-policy-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const skills = join(root, "skills"), allowed = join(skills, "selected"), denied = join(skills, "other");
  await mkdir(allowed, { recursive: true }); await mkdir(denied);
  await writeFile(join(allowed, "SKILL.md"), "selected");
  await writeFile(join(denied, "SKILL.md"), "other");
  await symlink(denied, join(allowed, "escape"));
  let calls = 0;
  const tool = { name: "read", execute: async () => { calls++; return { content: [{ type: "text", text: "read" }], details: {} }; } } as unknown as AgentTool;
  const restricted = restrictSkillReads(tool, root, [skills], [allowed], skills);
  await restricted.execute("allowed", { path: ".agents/skills/selected/SKILL.md" });
  for (const path of [".agents/skills/other/SKILL.md", join(denied, "SKILL.md"), ".agents/skills/selected/escape/SKILL.md", ".agents/skills"]) {
    await assert.rejects(() => restricted.execute("denied", { path }), /not enabled/);
  }
  assert.equal(calls, 1);
});
