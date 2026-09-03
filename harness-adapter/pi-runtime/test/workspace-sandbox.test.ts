import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  SandboxManager,
  type SandboxRuntimeConfig,
} from "@anthropic-ai/sandbox-runtime";
import type { BashOperations } from "@earendil-works/pi-coding-agent";
import {
  createWorkspaceSandboxedBashOperations,
  workspacePolicyDenyPaths,
  workspacePolicyDenyWritePaths,
} from "../src/workspace-sandbox.js";
import { WorkspaceAccessPolicy } from "../src/workspace-policy.js";

const osSandboxAvailable = process.platform === "darwin"
  ? existsSync("/usr/bin/sandbox-exec")
  : SandboxManager.checkDependencies();

async function workspaceTree() {
  const root = await mkdtemp(join(tmpdir(), "pi-workspace-sandbox-"));
  const home = join(root, "home");
  const repository = join(home, "Desktop", "repo");
  const conversations = join(repository, "workspace", "conversations");
  const workspace = join(conversations, "current");
  const publicData = join(workspace, "public_data");
  const terminalTemp = join(workspace, "pi", "terminal-output");
  const repositorySource = join(repository, "src");
  const skill = join(repository, ".agents", "skills", "data-cleaning");
  const knowledge = join(repository, "knowledge");
  const otherConversation = join(conversations, "other");
  await mkdir(publicData, { recursive: true });
  await mkdir(terminalTemp, { recursive: true });
  await mkdir(join(workspace, "product"), { recursive: true });
  await writeFile(join(workspace, "pi", "session.jsonl"), "private");
  await mkdir(repositorySource, { recursive: true });
  await mkdir(skill, { recursive: true });
  await mkdir(knowledge, { recursive: true });
  await mkdir(otherConversation, { recursive: true });
  await mkdir(join(home, ".ssh"), { recursive: true });
  const policy = await WorkspaceAccessPolicy.create({
    workspaceRoot: workspace,
    readOnlyRoots: [skill],
    knowledgeRoot: knowledge,
  });
  return {
    home,
    workspace,
    publicData,
    terminalTemp,
    repositorySource,
    skill,
    knowledge,
    otherConversation,
    policy,
  };
}

test("policy deny rules hide private state, repository source, and sibling conversations", async () => {
  const tree = await workspaceTree();
  const denied = await workspacePolicyDenyPaths(
    tree.policy,
    tree.home,
    [tree.home],
  );

  assert(denied.includes(await realpath(join(tree.home, ".ssh"))));
  assert(denied.includes(await realpath(tree.repositorySource)));
  assert(denied.includes(await realpath(tree.otherConversation)));
  assert(denied.includes(await realpath(join(tree.workspace, "product"))));
  assert(denied.includes(await realpath(join(tree.workspace, "pi", "session.jsonl"))));
  assert(!denied.includes(await realpath(tree.publicData)));
  assert(!denied.includes(await realpath(tree.terminalTemp)));
  assert(!denied.includes(await realpath(tree.skill)));
  assert(!denied.includes(await realpath(tree.knowledge)));
});

test("write deny rules preserve only public_data and terminal temp branches", async () => {
  const tree = await workspaceTree();
  const denied = await workspacePolicyDenyWritePaths(tree.policy, tree.home);

  assert(denied.includes(`${tree.policy.workspaceRoot}/*`));
  assert(denied.includes(`${join(tree.policy.workspaceRoot, "pi")}/*`));
  assert(denied.includes(await realpath(join(tree.workspace, "product"))));
  assert(denied.includes(await realpath(join(tree.workspace, "pi", "session.jsonl"))));
  assert(!denied.includes(await realpath(tree.publicData)));
  assert(!denied.includes(await realpath(tree.terminalTemp)));
});

test("sandboxed commands allow cd within one call and reset cwd for every call", async () => {
  const tree = await workspaceTree();
  let initialized: SandboxRuntimeConfig | undefined;
  let updated: SandboxRuntimeConfig | undefined;
  let checkedRipgrep: { command: string; args?: string[] } | undefined;
  const localCalls: Array<{
    command: string;
    cwd: string;
    env: NodeJS.ProcessEnv | undefined;
  }> = [];
  const controller = {
    checkDependencies: (ripgrepConfig?: { command: string; args?: string[] }) => {
      checkedRipgrep = ripgrepConfig;
      return true;
    },
    initialize: async (config: SandboxRuntimeConfig) => {
      initialized = config;
    },
    isSandboxingEnabled: () => true,
    updateConfig: (config: SandboxRuntimeConfig) => {
      updated = config;
    },
    wrapWithSandbox: async (command: string) => `sandboxed:${command}`,
  };
  const localOperations: BashOperations = {
    exec: async (command, cwd, options) => {
      localCalls.push({ command, cwd, env: options.env });
      return { exitCode: 0 };
    },
  };
  const operations = createWorkspaceSandboxedBashOperations(tree.policy, {
    controller,
    localOperations,
    userHome: tree.home,
    runtimeReadRoots: [],
  });

  await operations.exec("cd public_data && pwd", tree.workspace, {
    onData: () => undefined,
    env: { PATH: "/bin" },
  });
  await operations.exec("pwd", tree.workspace, {
    onData: () => undefined,
    env: { PATH: "/bin" },
  });

  assert.deepEqual(localCalls.map((call) => call.command), [
    "sandboxed:cd public_data && pwd",
    "sandboxed:pwd",
  ]);
  assert(localCalls.every((call) => call.cwd === tree.policy.workspaceRoot));
  assert(localCalls.every((call) => call.env?.TMPDIR === tree.policy.terminalTempRoot));
  assert.deepEqual(initialized?.network.allowedDomains, []);
  assert.deepEqual(
    checkedRipgrep,
    process.platform === "darwin" ? { command: "rg" } : undefined,
  );
  assert.deepEqual(updated?.filesystem.allowWrite, [
    tree.policy.publicDataRoot,
    tree.policy.terminalTempRoot,
  ]);
  assert(updated?.filesystem.denyWrite.includes(`${tree.policy.workspaceRoot}/*`));
  assert(updated?.filesystem.denyRead.includes(await realpath(tree.repositorySource)));
});

test("workspace bash operations fail closed when sandbox is unavailable", async () => {
  const tree = await workspaceTree();
  const operations = createWorkspaceSandboxedBashOperations(tree.policy, {
    controller: {
      checkDependencies: () => false,
      initialize: async () => undefined,
      isSandboxingEnabled: () => false,
      updateConfig: () => undefined,
      wrapWithSandbox: async (command: string) => command,
    },
    localOperations: {
      exec: async () => {
        throw new Error("must not execute");
      },
    },
    userHome: tree.home,
    runtimeReadRoots: [],
  });

  await assert.rejects(
    () => operations.exec("pwd", tree.workspace, { onData: () => undefined }),
    /WORKSPACE_SANDBOX_UNAVAILABLE/,
  );
  await assert.rejects(
    () => operations.exec("pwd", join(tree.workspace, "public_data"), {
      onData: () => undefined,
    }),
    /WORKSPACE_SCOPE_ERROR/,
  );
});

test(
  "OS sandbox reads public_data but denies repository source files",
  { skip: !osSandboxAvailable },
  async () => {
    const tree = await workspaceTree();
    const workspaceFile = join(tree.publicData, "input.txt");
    const repositoryFile = join(tree.repositorySource, "dependency.py");
    const skillFile = join(tree.skill, "SKILL.md");
    const otherConversationFile = join(tree.otherConversation, "private.txt");
    await writeFile(workspaceFile, "workspace-data");
    await writeFile(repositoryFile, "repository-secret");
    await writeFile(skillFile, "skill-reference");
    await writeFile(otherConversationFile, "other-conversation-secret");
    const operations = createWorkspaceSandboxedBashOperations(tree.policy, {
      userHome: tree.home,
      runtimeReadRoots: [],
    });
    let output = "";

    try {
      const allowed = await operations.exec(
        "cd public_data && cat input.txt",
        tree.workspace,
        {
          onData: (data) => {
            output += data.toString();
          },
        },
      );
      assert.equal(allowed.exitCode, 0);
      assert.match(output, /workspace-data/);

      output = "";
      const changedDirectory = await operations.exec(
        "cd public_data && printf generated > generated.txt && pwd",
        tree.workspace,
        {
          onData: (data) => {
            output += data.toString();
          },
        },
      );
      assert.equal(changedDirectory.exitCode, 0);
      assert.equal(output.trim(), tree.policy.publicDataRoot);
      assert.equal(
        await readFile(join(tree.publicData, "generated.txt"), "utf8"),
        "generated",
      );

      output = "";
      const resetDirectory = await operations.exec("pwd", tree.workspace, {
        onData: (data) => {
          output += data.toString();
        },
      });
      assert.equal(resetDirectory.exitCode, 0);
      assert.equal(output.trim(), tree.policy.workspaceRoot);

      output = "";
      const temporaryWrite = await operations.exec(
        'printf temporary > "$TMPDIR/scratch.txt"',
        tree.workspace,
        {
          onData: (data) => {
            output += data.toString();
          },
        },
      );
      assert.equal(temporaryWrite.exitCode, 0, output);
      assert.equal(
        await readFile(join(tree.terminalTemp, "scratch.txt"), "utf8"),
        "temporary",
      );

      output = "";
      const runtime = await operations.exec(
        `${JSON.stringify(process.execPath)} -e "process.stdout.write('runtime-ok')"`,
        tree.workspace,
        {
          onData: (data) => {
            output += data.toString();
          },
        },
      );
      assert.equal(runtime.exitCode, 0);
      assert.equal(output, "runtime-ok");

      const privateSession = join(tree.workspace, "pi", "session.jsonl");
      const deniedWrite = await operations.exec(
        `printf hacked > ${JSON.stringify(privateSession)}`,
        tree.workspace,
        { onData: () => undefined },
      );
      assert.notEqual(deniedWrite.exitCode, 0);
      assert.equal(await readFile(privateSession, "utf8"), "private");

      const deniedSkillWrite = await operations.exec(
        `printf hacked > ${JSON.stringify(skillFile)}`,
        tree.workspace,
        { onData: () => undefined },
      );
      assert.notEqual(deniedSkillWrite.exitCode, 0);
      assert.equal(await readFile(skillFile, "utf8"), "skill-reference");

      output = "";
      const deniedOtherConversation = await operations.exec(
        `cat ${JSON.stringify(otherConversationFile)}`,
        tree.workspace,
        {
          onData: (data) => {
            output += data.toString();
          },
        },
      );
      assert.notEqual(deniedOtherConversation.exitCode, 0);
      assert.doesNotMatch(output, /other-conversation-secret/);

      output = "";
      const denied = await operations.exec(
        `cat ${JSON.stringify(repositoryFile)}`,
        tree.workspace,
        {
          onData: (data) => {
            output += data.toString();
          },
        },
      );
      assert.notEqual(denied.exitCode, 0);
      assert.doesNotMatch(output, /repository-secret/);
    } finally {
      await SandboxManager.reset();
    }
  },
);
