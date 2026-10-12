import assert from "node:assert/strict";
import test from "node:test";
import {
  applySandboxPathAliases,
  applySandboxWorkingDirectory,
  insideSandboxPath,
  SandboxReadMirror,
  SandboxPathPolicy,
} from "../src/sandbox-paths.js";

const WORKSPACE = "/target-workspace/.pyromind-agent/conv-1";
const SKILLS = "/opt/pi/skills";
const KNOWLEDGE = "/opt/pi/knowledge";
const STORAGE = "/target-workspace";

function policy(
  overrides: {
    skillsDirectory?: string;
    storageRoot?: string;
  } = {},
) {
  return SandboxPathPolicy.create({
    workspacePath: WORKSPACE,
    readOnlyRoots: [],
    skillsDirectory: "skillsDirectory" in overrides ? overrides.skillsDirectory : SKILLS,
    knowledgeRoot: KNOWLEDGE,
    storageRoot: "storageRoot" in overrides ? overrides.storageRoot : STORAGE,
  });
}

test("sandbox paths resolve relative reads against the conversation root", () => {
  assert.equal(
    policy().resolvePath("public_data/train.csv", "read"),
    `${WORKSPACE}/public_data/train.csv`,
  );
  assert.equal(
    policy().resolvePath(`${WORKSPACE}/public_data/train.csv`, "read"),
    `${WORKSPACE}/public_data/train.csv`,
  );
});

test("sandbox paths map the skills and knowledge aliases to their roots", () => {
  assert.equal(
    policy().resolvePath(".agents/skills/data-cleaning/SKILL.md", "read"),
    `${SKILLS}/data-cleaning/SKILL.md`,
  );
  assert.equal(
    policy().resolvePath("knowledge/handbook.md", "read"),
    `${KNOWLEDGE}/handbook.md`,
  );
});

test("sandbox paths map the storage alias to the mounted Storage root", () => {
  assert.equal(
    policy().resolvePath("storage/datasets/train.jsonl", "read"),
    `${STORAGE}/datasets/train.jsonl`,
  );
  assert.throws(
    () => policy().resolvePath("storage/datasets/out.json", "write"),
    /PATH_SCOPE_ERROR/,
  );
  assert.throws(
    () => policy({ storageRoot: undefined }).resolvePath("storage/datasets/train.jsonl", "read"),
    /PATH_SCOPE_ERROR/,
  );
});

test("sandbox paths reject writes outside public_data", () => {
  for (const target of [
    "product/report.json",
    "public_data/../product/report.json",
    `${WORKSPACE}/pi/session.jsonl`,
    "/etc/passwd",
    "knowledge/handbook.md",
    "storage/datasets/out.json",
  ]) {
    assert.throws(
      () => policy().resolvePath(target, "write"),
      /PATH_SCOPE_ERROR/,
      target,
    );
  }
  assert.equal(
    policy().resolvePath("public_data/out.json", "write"),
    `${WORKSPACE}/public_data/out.json`,
  );
});

test("sandbox paths reject reads outside the allowed roots", () => {
  for (const target of ["pi/session.jsonl", `${WORKSPACE}/../other/secret`, "/etc/passwd"]) {
    assert.throws(
      () => policy().resolvePath(target, "read"),
      /PATH_SCOPE_ERROR/,
      target,
    );
  }
});

test("sandbox paths require the skills alias to be configured", () => {
  assert.throws(
    () => policy({ skillsDirectory: undefined }).resolvePath(".agents/skills/x/SKILL.md", "read"),
    /PATH_SCOPE_ERROR/,
  );
});

test("insideSandboxPath respects segment boundaries", () => {
  assert.equal(insideSandboxPath("/a/b", "/a/b"), true);
  assert.equal(insideSandboxPath("/a/b/c", "/a/b"), true);
  assert.equal(insideSandboxPath("/a/bc", "/a/b"), false);
});

test("system prompt skill locations use runtime aliases instead of host paths", () => {
  const prompt = [
    "<location>/opt/pi/skills/data-processing/SKILL.md</location>",
    "<location>/opt/pi/skills</location>",
    "<location>/opt/pi/knowledge</location>",
  ].join("\n");

  const rewritten = applySandboxPathAliases(prompt, [
    { host: "/opt/pi/skills", alias: ".agents/skills" },
    { host: "/opt/pi/knowledge", alias: "knowledge" },
  ]);

  assert.equal(
    rewritten,
    [
      "<location>.agents/skills/data-processing/SKILL.md</location>",
      "<location>.agents/skills</location>",
      "<location>knowledge</location>",
    ].join("\n"),
  );
});

test("system prompt working directory never exposes the host conversation root", () => {
  const hostCwd = "/Users/dev/conversations/conv-1";
  const prompt = `Guidelines\nCurrent working directory: ${hostCwd}\n`;

  const rewritten = applySandboxWorkingDirectory(prompt, hostCwd);

  assert.equal(rewritten.includes(hostCwd), false);
  assert.equal(rewritten.startsWith("Guidelines\n"), true);
  assert.match(rewritten, /Current working directory: \./);
});

test("system prompt without a working directory line is left untouched", () => {
  const prompt = "Guidelines\nNo directory here\n";

  assert.equal(
    applySandboxWorkingDirectory(prompt, "/Users/dev/conversations/conv-1"),
    prompt,
  );
});

test("sandbox read mirror answers skill and knowledge aliases from host roots", () => {
  const mirror = new SandboxReadMirror([
    { host: "/repo/.agents/skills", alias: ".agents/skills" },
    { host: "/opt/extra/aoi-labeling", alias: ".agents/skills/aoi-labeling" },
    { host: "/repo/knowledge", alias: "knowledge" },
  ]);

  assert.equal(
    mirror.resolveAlias(".agents/skills/data-processing/SKILL.md"),
    "/repo/.agents/skills/data-processing/SKILL.md",
  );
  assert.equal(
    mirror.resolveAlias(".agents/skills/aoi-labeling/notes.md"),
    "/opt/extra/aoi-labeling/notes.md",
  );
  assert.equal(
    mirror.resolveAlias("knowledge/handbook.md"),
    "/repo/knowledge/handbook.md",
  );
});

test("sandbox read mirror leaves non-resource paths to the sandbox", () => {
  const mirror = new SandboxReadMirror([
    { host: "/repo/.agents/skills", alias: ".agents/skills" },
  ]);

  assert.equal(mirror.resolveAlias("public_data/train.csv"), undefined);
  assert.equal(mirror.resolveAlias("/repo/.agents/skills/notes.md"), undefined);
  assert.equal(
    mirror.resolveAlias(".agents/skills/../../../etc/passwd"),
    undefined,
  );
});

test("sandbox read mirror only claims host files inside its roots", () => {
  const mirror = new SandboxReadMirror([
    { host: "/repo/.agents/skills", alias: ".agents/skills" },
  ]);

  assert.equal(
    mirror.resolveHost("/repo/.agents/skills/data-processing/SKILL.md"),
    "/repo/.agents/skills/data-processing/SKILL.md",
  );
  assert.equal(mirror.resolveHost("/repo/.agents/skills-other/notes.md"), undefined);
  assert.equal(mirror.resolveHost("skills/notes.md"), undefined);
});
