import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { normalizeBusinessToolContent, safePath } from "../src/tools.js";
import { WorkspaceAccessPolicy } from "../src/workspace-policy.js";

test("business tool content converts OpenHands inline image URLs for Pi", () => {
  assert.deepEqual(normalizeBusinessToolContent([
    { type: "text", text: "preview" },
    {
      type: "image",
      image_urls: [
        "data:image/png;base64,aGVsbG8=",
        "data:image/jpeg;base64,d29ybGQ=",
      ],
    },
  ]), [
    { type: "text", text: "preview" },
    { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
    { type: "image", data: "d29ybGQ=", mimeType: "image/jpeg" },
  ]);
});

test("business tool content accepts Pi MIME aliases", () => {
  assert.deepEqual(normalizeBusinessToolContent([
    { type: "image", data: "aGVsbG8=", mime_type: "image/png" },
    { type: "image", data: "d29ybGQ=", mimeType: "image/jpeg" },
  ]), [
    { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
    { type: "image", data: "d29ybGQ=", mimeType: "image/jpeg" },
  ]);
});

test("business tool content makes unsupported image URLs visible", () => {
  assert.deepEqual(normalizeBusinessToolContent([
    {
      type: "image",
      image_urls: [
        "https://example.com/image.png",
        "data:image/png;base64,not base64",
      ],
    },
  ]), [
    { type: "text", text: "[Image omitted: Pi only accepts inline base64 image data.]" },
    { type: "text", text: "[Image omitted: Pi only accepts inline base64 image data.]" },
  ]);
});

async function workspaceLayout(prefix = "pi-tools-") {
  const root = await mkdtemp(join(tmpdir(), prefix));
  const workspace = join(root, "conversation");
  const skill = join(root, "skill");
  const knowledge = join(root, "knowledge");
  await mkdir(join(workspace, "public_data"), { recursive: true });
  await mkdir(join(workspace, "pi", "terminal-output"), { recursive: true });
  await mkdir(join(skill, "references"), { recursive: true });
  await mkdir(join(knowledge, "nodes", "InputNode"), { recursive: true });
  await writeFile(join(skill, "SKILL.md"), "skill");
  await writeFile(join(skill, "references", "workflow-contracts.md"), "contracts");
  await writeFile(join(knowledge, "nodes", "InputNode", "InputNode.md"), "node");
  const canonicalRoot = await realpath(root);
  return {
    root: canonicalRoot,
    workspace: join(canonicalRoot, "conversation"),
    skill: join(canonicalRoot, "skill"),
    knowledge: join(canonicalRoot, "knowledge"),
  };
}

test("skills directory grants recursive read access independently of skill discovery", async (context) => {
  const { root, workspace, skill, knowledge } = await workspaceLayout();
  context.after(() => rm(root, { recursive: true, force: true }));
  const skills = join(root, "all-skills");
  await mkdir(skills);
  const policy = await WorkspaceAccessPolicy.create({
    workspaceRoot: workspace,
    readOnlyRoots: [skill],
    skillsDirectory: skills,
    knowledgeRoot: knowledge,
  });
  // New files inherit the directory rule without recreating the policy.
  for (const suffix of [
    "inference-evaluation/SKILL.md",
    "inference-evaluation/references/deep/contracts.md",
    "ordinary/scripts/helper.py",
    ".hidden/config.txt",
  ]) {
    const target = join(skills, suffix);
    await mkdir(join(target, ".."), { recursive: true });
    await writeFile(target, suffix);
    for (const input of [target, `.agents/skills/${suffix}`, `./.agents/skills/${suffix}`]) {
      assert.equal(await readFile(await policy.resolvePath(input, "read"), "utf8"), suffix);
      await assert.rejects(() => policy.resolvePath(input, "write"), /PATH_SCOPE_ERROR/);
    }
  }
  assert.equal(await policy.resolvePath(".agents/skills/", "read"), skills);
  assert.equal(await policy.resolvePath(join(skill, "SKILL.md"), "read"), join(skill, "SKILL.md"));
  assert(policy.terminalReadRoots.includes(skills));
  assert(!policy.terminalWriteRoots.includes(skills));
  await assert.rejects(
    async () => readFile(await policy.resolvePath(".agents/skills/missing/SKILL.md", "read")),
    { code: "ENOENT" },
  );

  const outside = join(root, "all-skills-private");
  await mkdir(outside);
  await writeFile(join(outside, "private.txt"), "private");
  await symlink(outside, join(skills, "escape"));
  await symlink(join(skills, "ordinary"), join(skills, "contained"));
  assert.equal(
    await policy.resolvePath(".agents/skills/contained/scripts/helper.py", "read"),
    join(skills, "ordinary/scripts/helper.py"),
  );
  for (const path of [
    join(outside, "private.txt"),
    ".agents/skills/../all-skills-private/private.txt",
    "./.agents/skills/escape/private.txt",
    ".agents/skills/escape/missing.txt",
    "pi/session.jsonl",
    "../other-conversation/public_data/data.txt",
  ]) {
    await assert.rejects(() => policy.resolvePath(path, "read"), /PATH_SCOPE_ERROR/);
  }
});

test("safePath resolves relative, absolute, normalized, and missing public paths", async () => {
  const { workspace, skill, knowledge } = await workspaceLayout();
  const workflow = join(workspace, "public_data", "workflow.py");

  assert.equal(
    await safePath("public_data/workflow.py", workspace, skill, knowledge, false),
    workflow,
  );
  assert.equal(
    await safePath(workflow, workspace, skill, knowledge, false),
    workflow,
  );
  assert.equal(
    await safePath(
      "public_data/generated/../workflow.py",
      workspace,
      skill,
      knowledge,
      true,
    ),
    workflow,
  );
  assert.equal(
    await safePath(
      "public_data/new/nested/output.jsonl",
      workspace,
      skill,
      knowledge,
      false,
    ),
    join(workspace, "public_data", "new", "nested", "output.jsonl"),
  );
  assert.equal(
    await safePath(join(skill, "SKILL.md"), workspace, skill, knowledge, true),
    join(skill, "SKILL.md"),
  );
  assert.equal(
    await safePath(
      "knowledge/nodes/InputNode/InputNode.md",
      workspace,
      skill,
      knowledge,
      true,
    ),
    join(knowledge, "nodes", "InputNode", "InputNode.md"),
  );
});

test("safePath denies private, read-only, and out-of-scope paths", async () => {
  const { root, workspace, skill, knowledge } = await workspaceLayout();
  await assert.rejects(
    () => safePath("pi/session.jsonl", workspace, skill, knowledge, true),
    /PATH_SCOPE_ERROR/,
  );
  await assert.rejects(
    () => safePath("product/snapshot.json", workspace, skill, knowledge, true),
    /PATH_SCOPE_ERROR/,
  );
  await assert.rejects(
    () => safePath(join(skill, "SKILL.md"), workspace, skill, knowledge, false),
    /public_data/,
  );
  await assert.rejects(
    () => safePath("knowledge/nodes/InputNode/InputNode.md", workspace, skill, knowledge, false),
    /public_data/,
  );
  await assert.rejects(
    () => safePath(join(root, "unknown.md"), workspace, skill, knowledge, true),
    /PATH_SCOPE_ERROR/,
  );
  await assert.rejects(
    () => safePath("../escape", workspace, skill, knowledge, false),
    /PATH_SCOPE_ERROR/,
  );
  await assert.rejects(
    () => safePath("knowledge/file.md", workspace, skill, undefined, true),
    /knowledge\/ is not configured/,
  );
});

test("safePath permits contained symlinks and rejects canonical escapes", async () => {
  const { root, workspace, skill, knowledge } = await workspaceLayout();
  const contained = join(workspace, "public_data", "contained");
  const outside = join(root, "outside");
  await mkdir(contained);
  await mkdir(outside);
  await symlink(contained, join(workspace, "public_data", "inside-link"));
  await symlink(outside, join(workspace, "public_data", "escape-link"));

  assert.equal(
    await safePath(
      "public_data/inside-link/new.txt",
      workspace,
      skill,
      knowledge,
      false,
    ),
    join(contained, "new.txt"),
  );
  await assert.rejects(
    () => safePath(
      "public_data/escape-link/new.txt",
      workspace,
      skill,
      knowledge,
      false,
    ),
    /PATH_SCOPE_ERROR/,
  );
});

test("safePath accepts each named skill root but keeps them read-only", async () => {
  const { root, workspace } = await workspaceLayout("pi-tools-skills-");
  const cleaning = join(root, "data-cleaning");
  const preparation = join(root, "data-preparation");
  await mkdir(cleaning);
  await mkdir(preparation);
  await writeFile(join(cleaning, "SKILL.md"), "cleaning");
  await writeFile(join(preparation, "SKILL.md"), "preparation");
  const skills = [
    { name: "data-cleaning", path: cleaning },
    { name: "data-preparation", path: preparation },
  ];
  assert.equal(
    await safePath(join(preparation, "SKILL.md"), workspace, skills, undefined, true),
    join(preparation, "SKILL.md"),
  );
  await assert.rejects(
    () => safePath(join(cleaning, "SKILL.md"), workspace, skills, undefined, false),
    /public_data/,
  );
});
