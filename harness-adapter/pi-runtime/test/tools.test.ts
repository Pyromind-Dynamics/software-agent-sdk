import assert from "node:assert/strict";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { safePath } from "../src/tools.js";

test("safePath resolves workspace, skill, and knowledge paths by scope", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-tools-"));
  const workspace = join(root, "conversation");
  const skill = join(root, "skill");
  const knowledge = join(root, "knowledge");
  await mkdir(workspace);
  await mkdir(join(skill, "references"), { recursive: true });
  await mkdir(join(knowledge, "nodes", "InputNode"), { recursive: true });
  await writeFile(join(skill, "SKILL.md"), "skill");
  await writeFile(join(skill, "references", "workflow-contracts.md"), "contracts");
  await writeFile(join(knowledge, "nodes", "InputNode", "InputNode.md"), "node");

  assert.equal(
    await safePath("public_data/workflow.py", workspace, skill, knowledge, false),
    join(workspace, "public_data/workflow.py"),
  );
  assert.equal(
    await safePath(join(skill, "SKILL.md"), workspace, skill, knowledge, true),
    join(skill, "SKILL.md"),
  );
  assert.equal(
    await safePath("knowledge/nodes/InputNode/InputNode.md", workspace, skill, knowledge, true),
    join(knowledge, "nodes", "InputNode", "InputNode.md"),
  );
  assert.equal(
    await safePath(join(knowledge, "nodes", "InputNode", "InputNode.md"), workspace, skill, knowledge, true),
    join(knowledge, "nodes", "InputNode", "InputNode.md"),
  );
  await assert.rejects(
    () => safePath(join(workspace, "public_data/workflow.py"), workspace, skill, knowledge, true),
    /use public_data\/workflow\.py/,
  );
  await assert.rejects(
    () => safePath(join(skill, "references", "workflow-contracts.md"), workspace, skill, knowledge, false),
    /read-only/,
  );
  await assert.rejects(
    () => safePath("knowledge/nodes/InputNode/InputNode.md", workspace, skill, knowledge, false),
    /knowledge is read-only/,
  );
  await assert.rejects(
    () => safePath(join(root, "unknown.md"), workspace, skill, knowledge, true),
    /PATH_SCOPE_ERROR/,
  );
  await assert.rejects(
    () => safePath("knowledge/nodes/InputNode/InputNode.md", workspace, skill, undefined, true),
    /knowledge\/ is not configured/,
  );
  await assert.rejects(
    () => safePath("../escape", workspace, skill, knowledge, false),
    /unsafe path/,
  );
});

test("safePath rejects symlink escape", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-tools-"));
  const workspace = join(root, "conversation");
  const skill = join(root, "skill");
  const knowledge = join(root, "knowledge");
  await mkdir(workspace);
  await mkdir(skill);
  await mkdir(knowledge);
  await symlink(skill, join(workspace, "linked"));
  await assert.rejects(
    () => safePath("linked/SKILL.md", workspace, skill, knowledge, true),
    /symlink/,
  );
});
