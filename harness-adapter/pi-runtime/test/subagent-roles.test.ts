import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { parseSubagentRoles, writeSubagentRoles } from "../src/subagent-roles.js";
import { SubagentHost, validateSubagentCall } from "../src/subagent-host.js";

const role = { name: "evaluator", description: "Evaluate frozen inputs", prompt: "Return evidence", tools: ["read"], skills: ["evaluation"] };

test("missing tools identify the role and cannot be supplied by another role", () => {
  assert.throws(() => new SubagentHost("/unused", "test", {} as any,
    [{name: "read"}] as any, {} as any, {} as any, "", [role],
    {evaluator: []}), /evaluator.*read/);
});

test("trusted roles validate names and reject duplicate roles or injected frontmatter", () => {
  assert.deepEqual(parseSubagentRoles([role]), [role]);
  for (const value of [
    [{ ...role, name: "../../escape" }], [role, role],
    [{ ...role, description: "text\ntools: terminal" }],
    [{ ...role, tools: ["read\nterminal"] }], [{ ...role, skills: ["../secret"] }],
  ]) assert.throws(() => parseSubagentRoles(value));
  assert.equal(validateSubagentCall({ agent: "evaluator", task: "bench" }, { evaluator: ["read"] }), undefined);
  assert.ok(validateSubagentCall({ agent: "general_purpose", task: "bench" }, { evaluator: ["read"] }));
});

test("role files disable inherited context and connect only host-mediated tools", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-role-config-"));
  try {
    await mkdir(join(root, "agents"));
    writeSubagentRoles(root, [{ ...role, skillPaths: ["/trusted/skills/evaluation"] }], "/trusted/subagent-child-extension.js");
    const config = await readFile(join(root, "agents/evaluator.md"), "utf8");
    assert.match(config, /tools: read\n/);
    assert.match(config, /inheritSkills: false/);
    assert.match(config, /skills: evaluation/);
    assert.match(config, /skillPath: \/trusted\/skills\/evaluation/);
    assert.match(config, /inheritProjectContext: false/);
    const extension = await readFile(join(root, "role-extensions/evaluator.mjs"), "utf8");
    assert.match(extension, /installChildHost\(pi, "evaluator"\)/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
