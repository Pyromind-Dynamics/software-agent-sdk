import assert from "node:assert/strict";
import { mkdtemp, mkdir, realpath, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  SandboxManager,
  type SandboxRuntimeConfig,
} from "@anthropic-ai/sandbox-runtime";
import type { BashOperations } from "@earendil-works/pi-coding-agent";
import {
  assertNoDirectoryChange,
  createWorkspaceSandboxedBashOperations,
  workspaceUserDataDenyPaths,
} from "../src/workspace-sandbox.js";

async function workspaceTree(): Promise<{
  home: string;
  workspace: string;
  repositorySource: string;
  otherConversation: string;
}> {
  const root = await mkdtemp(join(tmpdir(), "pi-workspace-sandbox-"));
  const home = join(root, "home");
  const repository = join(home, "Desktop", "repo");
  const conversations = join(repository, "workspace", "conversations");
  const workspace = join(conversations, "current");
  const repositorySource = join(repository, "src");
  const otherConversation = join(conversations, "other");
  await mkdir(workspace, { recursive: true });
  await mkdir(repositorySource, { recursive: true });
  await mkdir(otherConversation, { recursive: true });
  await mkdir(join(home, ".ssh"), { recursive: true });
  return { home, workspace, repositorySource, otherConversation };
}

test("workspace deny rules hide repository sources and sibling conversations", async () => {
  const tree = await workspaceTree();
  const denied = await workspaceUserDataDenyPaths(tree.workspace, tree.home);

  assert(denied.includes(await realpath(join(tree.home, ".ssh"))));
  assert(denied.includes(await realpath(tree.repositorySource)));
  assert(denied.includes(await realpath(tree.otherConversation)));
  assert(!denied.includes(await realpath(tree.workspace)));
});

test("terminal rejects shell directory changes because cwd is already pinned", () => {
  assert.doesNotThrow(() => assertNoDirectoryChange("python pipeline.py input.jsonl"));
  assert.throws(() => assertNoDirectoryChange("cd .. && ls"), /WORKSPACE_SCOPE_ERROR/);
  assert.throws(
    () => assertNoDirectoryChange("echo ok; builtin cd /tmp"),
    /WORKSPACE_SCOPE_ERROR/,
  );
});

test("workspace bash operations wrap commands and pin cwd and temp paths", async () => {
  const tree = await workspaceTree();
  let initialized: SandboxRuntimeConfig | undefined;
  let updated: SandboxRuntimeConfig | undefined;
  let localCall:
    | { command: string; cwd: string; env: NodeJS.ProcessEnv | undefined }
    | undefined;
  const controller = {
    checkDependencies: () => true,
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
      localCall = { command, cwd, env: options.env };
      return { exitCode: 0 };
    },
  };
  const operations = createWorkspaceSandboxedBashOperations(tree.workspace, {
    controller,
    localOperations,
    userHome: tree.home,
  });

  const result = await operations.exec("pwd", tree.workspace, {
    onData: () => undefined,
    env: { PATH: "/bin" },
  });

  assert.equal(result.exitCode, 0);
  assert.equal(localCall?.command, "sandboxed:pwd");
  assert.equal(localCall?.cwd, tree.workspace);
  assert.equal(localCall?.env?.TMPDIR, join(tree.workspace, ".tmp"));
  assert.deepEqual(initialized?.network.allowedDomains, []);
  assert.deepEqual(updated?.filesystem.allowWrite, [await realpath(tree.workspace)]);
  assert.deepEqual(updated?.filesystem.denyWrite, [
    join(await realpath(tree.workspace), "pi", "terminal-output"),
  ]);
  assert(updated?.filesystem.denyRead.includes(await realpath(tree.repositorySource)));
});

test("workspace bash operations fail closed when sandbox is unavailable", async () => {
  const tree = await workspaceTree();
  const operations = createWorkspaceSandboxedBashOperations(tree.workspace, {
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
  });

  await assert.rejects(
    () =>
      operations.exec("pwd", tree.workspace, {
        onData: () => undefined,
      }),
    /WORKSPACE_SANDBOX_UNAVAILABLE/,
  );
  await assert.rejects(
    () =>
      operations.exec("pwd", join(tree.workspace, ".."), {
        onData: () => undefined,
      }),
    /WORKSPACE_SCOPE_ERROR/,
  );
  await assert.rejects(
    () =>
      operations.exec("cd ..", tree.workspace, {
        onData: () => undefined,
      }),
    /WORKSPACE_SCOPE_ERROR/,
  );
});

test(
  "OS sandbox reads workspace files but denies repository source files",
  { skip: !SandboxManager.checkDependencies() },
  async () => {
    const tree = await workspaceTree();
    const workspaceFile = join(tree.workspace, "input.txt");
    const repositoryFile = join(tree.repositorySource, "dependency.py");
    await writeFile(workspaceFile, "workspace-data");
    await writeFile(repositoryFile, "repository-secret");
    const operations = createWorkspaceSandboxedBashOperations(tree.workspace, {
      userHome: tree.home,
    });
    let output = "";

    try {
      const allowed = await operations.exec("cat input.txt", tree.workspace, {
        onData: (data) => {
          output += data.toString();
        },
      });
      assert.equal(allowed.exitCode, 0);
      assert.match(output, /workspace-data/);

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
