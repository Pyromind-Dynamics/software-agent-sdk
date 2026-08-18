import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { extendSystemPromptWithSkills } from "../src/skill-prompt.ts";

test("loads configured skills with progressive disclosure", async () => {
  const root = mkdtempSync(join(tmpdir(), "pi-skills-"));
  const skillDir = join(root, ".pyromind", "skills", "generate-workflow-dsl");
  mkdirSync(skillDir, { recursive: true });
  writeFileSync(
    join(skillDir, "SKILL.md"),
    [
      "---",
      "name: generate-workflow-dsl",
      "description: Generate a workflow DSL.",
      "---",
      "FULL SKILL BODY",
    ].join("\n"),
  );
  const env = new NodeExecutionEnv({ cwd: root });

  try {
    const prompt = await extendSystemPromptWithSkills(
      "Base prompt",
      env,
      [join(root, ".pyromind", "skills")],
    );

    assert.match(prompt, /<available_skills>/);
    assert.match(prompt, /<name>generate-workflow-dsl<\/name>/);
    assert.match(prompt, /Generate a workflow DSL/);
    assert.match(prompt, /\.pyromind\/skills\/generate-workflow-dsl\/SKILL\.md/);
    assert.doesNotMatch(prompt, /FULL SKILL BODY/);
  } finally {
    await env.cleanup();
    rmSync(root, { recursive: true, force: true });
  }
});

test("rejects invalid configured skills instead of silently hiding them", async () => {
  const root = mkdtempSync(join(tmpdir(), "pi-skills-invalid-"));
  const skillDir = join(root, "skills", "invalid");
  mkdirSync(skillDir, { recursive: true });
  writeFileSync(join(skillDir, "SKILL.md"), "missing frontmatter");
  const env = new NodeExecutionEnv({ cwd: root });

  try {
    await assert.rejects(
      extendSystemPromptWithSkills("Base prompt", env, [join(root, "skills")]),
      /Failed to load configured Pi skills/,
    );
  } finally {
    await env.cleanup();
    rmSync(root, { recursive: true, force: true });
  }
});
