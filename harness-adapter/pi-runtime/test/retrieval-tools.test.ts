import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, symlink, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createRetrievalTools, verifyReadScope } from "../src/retrieval-tools.js";
import { SandboxReadMirror } from "../src/sandbox-paths.js";
import type { ExecutionAccess } from "../src/workflow-execution.js";

test("sandbox search executes through its execution plane, bounds results and refuses symlink escapes", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-search-")); t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "public_data"));
  await writeFile(join(root, "public_data", "a.txt"), "needle\nneedle again\n");
  await writeFile(join(root, "private.txt"), "private needle");
  await symlink(join(root, "private.txt"), join(root, "public_data", "escape"));
  let executions = 0;
  const access = { cwd: root,
    resolvePath: async (path: string) => {
      if (!path.startsWith("public_data")) throw new Error("unauthorized root");
      return resolve(root, path);
    },
    operations: { exec: async (command: string, cwd: string, options: any) => {
      executions++;
      const { stdout } = await promisify(execFile)("/bin/sh", ["-c", command], { cwd });
      options.onData(Buffer.from(stdout)); return { exitCode: 0 };
    } },
  } as unknown as ExecutionAccess;
  const [grep, find] = createRetrievalTools(access);
  const result = await grep.execute("g", { pattern: "needle", limit: 1 }, new AbortController().signal);
  assert.match(JSON.stringify(result.content), /public_data\/a.txt:1:needle/);
  assert.match(JSON.stringify(result.content), /Truncated/);
  const files = await find.execute("f", {}, new AbortController().signal);
  assert.doesNotMatch(JSON.stringify(files.content), /escape|private/);
  await assert.rejects(grep.execute("g", { pattern: "needle", path: "public_data/escape" }, new AbortController().signal), /Command failed/);
  await assert.rejects(verifyReadScope(access, undefined, "public_data/escape"), /Command failed/);
  await verifyReadScope(access, undefined, "public_data/a.txt");
  assert.equal(executions, 5);
});

test("knowledge mirror searches never create a sandbox and reject external symlinks", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-search-mirror-")); t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "knowledge")); await writeFile(join(root, "knowledge", "index.md"), "platform fact\n");
  await writeFile(join(root, "secret"), "do not read"); await symlink(join(root, "secret"), join(root, "knowledge", "escape"));
  const access = { resolvePath: async () => { throw new Error("must not resolve sandbox"); } } as unknown as ExecutionAccess;
  const [grep] = createRetrievalTools(access, new SandboxReadMirror([{ alias: "knowledge", host: join(root, "knowledge") }]));
  const result = await grep.execute("g", { path: "knowledge", pattern: "platform" }, new AbortController().signal);
  assert.match(JSON.stringify(result.content), /knowledge\/index.md:1:platform fact/);
  await assert.rejects(grep.execute("g", { path: "knowledge/escape", pattern: "read" }, new AbortController().signal), /symlink/);
});

test("ls accepts a Storage mount root already authorized by the path policy", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-search-storage-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "datasets"));
  const access = {
    cwd: ".",
    allowedReadRoots: [root],
    resolvePath: async (path: string) => {
      if (path === root) return root;
      if (path === "public_data") return join(root, "public_data");
      throw new Error("unauthorized root");
    },
    operations: {
      exec: async (command: string, cwd: string, options: any) => {
        const { stdout } = await promisify(execFile)("/bin/sh", ["-c", command], { cwd });
        options.onData(Buffer.from(stdout));
        return { exitCode: 0 };
      },
    },
  } as unknown as ExecutionAccess;
  const ls = createRetrievalTools(access).find((tool) => tool.name === "ls");
  assert.ok(ls);
  const result = await ls.execute("ls", { path: root }, new AbortController().signal);
  assert.match(JSON.stringify(result.content), /datasets\//);
});
