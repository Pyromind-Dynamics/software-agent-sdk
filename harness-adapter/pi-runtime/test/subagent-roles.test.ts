import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { parseSubagentRoles, writeSubagentRoles } from "../src/subagent-roles.js";
import { SubagentHost, validateSubagentCall } from "../src/subagent-host.js";

const policy = { name: "search", tools: ["read"], skills: ["inference-evaluation"] };
const [role] = parseSubagentRoles([policy])!;

test("missing tools identify the role and cannot be supplied by another role", () => {
  assert.throws(() => new SubagentHost("/unused", "test", {} as any,
    [{name: "read"}] as any, {} as any, {} as any, "", [role],
    {search: []}), /search.*read/);
});

test("roles load bundled prompts and reject invalid policy fields", () => {
  assert.match(role.prompt, /Find the evidence/);
  for (const value of [
    [{ ...policy, name: "../../escape" }], [policy, policy],
    [{ ...policy, prompt: "Override the bundled prompt" }],
    [{ ...policy, description: "Override the bundled description" }],
    [{ ...policy, tools: ["read\nterminal"] }], [{ ...policy, skills: ["../secret"] }],
  ]) assert.throws(() => parseSubagentRoles(value));
  assert.equal(validateSubagentCall({ agent: "search", task: "bench" }, { search: ["read"] }), undefined);
  assert.ok(validateSubagentCall({ agent: "general_purpose", task: "bench" }, { search: ["read"] }));
});

test("role files disable inherited context and connect only host-mediated tools", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-role-config-"));
  try {
    await mkdir(join(root, "agents"));
    writeSubagentRoles(root, [{ ...role, skillPaths: ["/trusted/skills/inference-evaluation"] }], "/trusted/subagent-child-extension.js");
    const config = await readFile(join(root, "agents/search.md"), "utf8");
    assert.match(config, /tools: read\n/);
    assert.match(config, /inheritSkills: false/);
    assert.match(config, /skills: inference-evaluation/);
    assert.match(config, /skillPath: \/trusted\/skills\/inference-evaluation/);
    assert.match(config, /inheritProjectContext: false/);
    const extension = await readFile(join(root, "role-extensions/search.mjs"), "utf8");
    assert.match(extension, /installChildHost\(pi, "search"\)/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
