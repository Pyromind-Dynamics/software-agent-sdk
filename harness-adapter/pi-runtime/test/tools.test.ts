import assert from "node:assert/strict";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { safePath } from "../src/tools.js";

test("safePath confines mutations and allows read-only skill access", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-tools-"));
  const workspace = join(root, "conversation");
  const skill = join(root, "skill");
  await mkdir(workspace);
  await mkdir(skill);
  await writeFile(join(skill, "SKILL.md"), "skill");
  assert.equal(await safePath("public_data/workflow.py", workspace, skill, false), join(workspace, "public_data/workflow.py"));
  assert.equal(await safePath(join(skill, "SKILL.md"), workspace, skill, true), join(skill, "SKILL.md"));
  await assert.rejects(() => safePath("../escape", workspace, skill, false), /unsafe path/);
  await assert.rejects(() => safePath(join(skill, "SKILL.md"), workspace, skill, false), /escapes/);
});

test("safePath rejects symlink escape", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-tools-"));
  const workspace = join(root, "conversation");
  const skill = join(root, "skill");
  await mkdir(workspace);
  await mkdir(skill);
  await symlink(skill, join(workspace, "linked"));
  await assert.rejects(() => safePath("linked/SKILL.md", workspace, skill, true), /symlink/);
});
