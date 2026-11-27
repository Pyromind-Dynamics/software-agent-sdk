import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { createServer } from "node:http";
import { mkdtemp, mkdir, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { SandboxManager } from "@anthropic-ai/sandbox-runtime";
import { validateToolArguments } from "@earendil-works/pi-ai";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";
import { restrictSkillReads } from "../src/session-policy.js";
import {
  businessTool,
  createTools,
  normalizeBusinessToolContent,
  safePath,
  type SandboxTools,
} from "../src/tools.js";
import { WorkspaceAccessPolicy } from "../src/workspace-policy.js";
import { LazySandboxTerminalOperations } from "../src/sandbox-terminal.js";

test("business tools carry trusted context separately from model arguments", async () => {
  let request: unknown;
  const peer = { request: async (_method: string, params: unknown) => {
    request = params; return {content: [], details: {}, is_error: false};
  } } as unknown as JsonlRpcPeer;
  const config = { name: "set_preparation_plan", description: "", inputSchema: {} };
  const tool = businessTool(peer, config, undefined, () => ({preparation_revision: 5}));
  const args = {entry_task_id: "first", plan: {}};
  assert.ok(tool.prepareArguments);
  assert.equal(tool.prepareArguments(args), args);
  for (const plan of ["[]", "null", "incomplete {"]) {
    const invalid = { ...args, plan };
    assert.equal(tool.prepareArguments(invalid), invalid);
  }
  const prepared = tool.prepareArguments({ ...args, plan: JSON.stringify(args.plan) });
  await tool.execute("call", prepared, undefined);
  assert.deepEqual(request, {
    preparation_revision: 5, tool_call_id: "call",
    tool_name: "set_preparation_plan", arguments: args,
  });
});

test("encoded preparation plans still require a single object matching the tool schema", () => {
  const tool = businessTool({} as JsonlRpcPeer, {
    name: "set_preparation_plan", description: "",
    inputSchema: { type: "object", properties: {
      plan: { type: "object", properties: { goal: { type: "string" } }, required: ["goal"] },
    }, required: ["plan"] },
  });
  for (const plan of ["{", "[]", "null", JSON.stringify('{"goal":"PCB"}'), '{"wrong":"field"}']) {
    const args = tool.prepareArguments!({ plan });
    assert.throws(() => validateToolArguments(tool, {
      type: "toolCall", id: "plan", name: tool.name, arguments: args,
    }), /Validation failed/);
  }
});

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

test("sandbox read resolves the remote endpoint for Storage paths", async (context) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-tools-sandbox-read-")));
  context.after(() => rm(root, { recursive: true, force: true }));
  const requested: string[] = [];
  const fileApi = createServer((request, response) => {
    const url = new URL(request.url ?? "/", "http://localhost");
    requested.push(url.searchParams.get("path") ?? "");
    response.end("sample");
  });
  await new Promise<void>((resolve) => fileApi.listen(0, "127.0.0.1", resolve));
  context.after(() => fileApi.close());
  const endpointUrl = `http://127.0.0.1:${(fileApi.address() as { port: number }).port}`;
  let ensures = 0;
  const tools = await createTools(
    {} as JsonlRpcPeer,
    root,
    "sandbox",
    [],
    undefined,
    undefined,
    [],
    {
      sandbox: async () => {
        ensures++;
        return {
          baseUrl: endpointUrl,
          wsBaseUrl: endpointUrl,
          sandboxId: "sandbox-1",
          apiKey: "key",
          workspacePath: "/workspace/.pyromind-agent/conv-1",
          storagePath: "/workspace",
        };
      },
    },
  );
  const read = tools.find((tool) => tool.name === "read")!;
  const restricted = restrictSkillReads(
    read,
    root,
    [],
    [],
    undefined,
    { canonicalizeWorkspacePaths: false },
  );

  await restricted.execute("remote-read", {
    path: "/workspace/datasets/demo/avi/train.jsonl",
  });

  assert.equal(ensures, 1);
  assert.deepEqual(requested, ["/workspace/datasets/demo/avi/train.jsonl"]);
});

test("sandbox lanes resolve logical task paths without changing Storage paths", async (context) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-tools-lane-")));
  context.after(() => rm(root, { recursive: true, force: true }));
  const fileApi = createServer((_request, response) => response.end("sample"));
  await new Promise<void>((resolve) => fileApi.listen(0, "127.0.0.1", resolve));
  context.after(() => fileApi.close());
  const endpointUrl = `http://127.0.0.1:${(fileApi.address() as { port: number }).port}`;
  let sandbox!: SandboxTools;
  await createTools(
    {} as JsonlRpcPeer,
    root,
    "sandbox",
    [],
    undefined,
    undefined,
    [],
    {
      sandbox: async () => ({
        baseUrl: endpointUrl,
        wsBaseUrl: endpointUrl,
        sandboxId: "sandbox-1",
        apiKey: "key",
        workspacePath: "/workspace/.pyromind-agent/conv-1",
        storagePath: "/workspace",
      }),
      onSandboxToolsReady: (tools) => { sandbox = tools; },
    },
  );

  for (const path of ["../escape", "/absolute"]) {
    assert.throws(() => sandbox.createLane("child", path), /inside the conversation root/);
  }
  const lane = sandbox.createLane("child", "public_data/experiment-runs/run-1");
  assert.equal(lane.access.cwd, "public_data/experiment-runs/run-1");
  assert.equal(
    await lane.access.resolvePath("public_data/output.txt", "write"),
    "/workspace/.pyromind-agent/conv-1/public_data/experiment-runs/run-1/public_data/output.txt",
  );
  assert.equal(
    await lane.access.resolvePath("/workspace/datasets/input.jsonl", "read"),
    "/workspace/datasets/input.jsonl",
  );
  await assert.rejects(
    () => lane.access.resolvePath(
      "/workspace/.pyromind-agent/conv-1/public_data/experiment-runs/run-2/public_data/output.txt",
      "write",
    ),
    /PATH_SCOPE_ERROR/,
  );
});

test("only the main Sandbox terminal receives dynamic platform SDK credentials", async (context) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-sdk-lanes-")));
  context.after(() => rm(root, { recursive: true, force: true }));
  const original = LazySandboxTerminalOperations.prototype.exec;
  context.after(() => { LazySandboxTerminalOperations.prototype.exec = original; });
  const environments: Array<NodeJS.ProcessEnv | undefined> = [];
  LazySandboxTerminalOperations.prototype.exec = async (_command, _cwd, options) => {
    environments.push(options.env);
    options.onData(Buffer.from("ok"));
    return { exitCode: 0 };
  };
  let sandbox!: SandboxTools;
  let contexts = 0;
  await createTools(
    {} as JsonlRpcPeer,
    root, "sandbox", [], undefined, undefined, [],
    {
      sandbox: async () => { throw new Error("No endpoint resolution expected"); },
      platformSdkEnvironment: async () => {
        contexts++;
        return { PYROMIND_API_KEY: "main-sdk-key" };
      },
      onSandboxToolsReady: (tools) => { sandbox = tools; },
    },
  );
  for (const lane of [sandbox.main, sandbox.workflow, sandbox.createLane("child-worker")]) {
    const terminal = lane.tools.find((tool) => tool.name === "terminal");
    assert(terminal);
    const result = await terminal.execute("call", { command: "echo ok" });
    assert.equal(result.details.resultCategory, "success");
    assert.equal(lane.access.env?.PYROMIND_API_KEY, undefined);
  }
  assert.equal(contexts, 1);
  assert.deepEqual(environments.map((environment) => environment?.PYROMIND_API_KEY),
    ["main-sdk-key", undefined, undefined]);
});

test("OS child terminals do not reuse the main SDK environment provider", {
  skip: process.platform === "darwin"
    ? !existsSync("/usr/bin/sandbox-exec") : !SandboxManager.checkDependencies(),
}, async (context) => {
  const { root, workspace } = await workspaceLayout("pi-sdk-os-");
  context.after(() => rm(root, { recursive: true, force: true }));
  const previous = { TMPDIR: process.env.TMPDIR, TMP: process.env.TMP, TEMP: process.env.TEMP };
  context.after(() => {
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[name]; else process.env[name] = value;
    }
  });
  let child: AgentTool | undefined;
  let contexts = 0;
  const tools = await createTools(
    {} as JsonlRpcPeer,
    workspace, "os-sandbox", [], undefined, undefined, [], {
      platformSdkEnvironment: async () => {
        contexts++;
        return {
          PYROMIND_API_KEY: "main-only-sdk-key",
          PYROMIND_BASE_URL: "https://sdk.example/api/v1",
          PYROMIND_SDK_PYTHON: process.execPath,
        };
      },
      onChildTerminalReady: (tool) => { child = tool; },
    },
  );
  assert(child);
  const childResult = await child.execute("child", { command: "test -z \"$PYROMIND_API_KEY\"" });
  assert.equal(childResult.details.resultCategory, "success");
  assert.equal(contexts, 0);
  const main = tools.find((tool) => tool.name === "terminal");
  assert(main);
  const result = await main.execute("main", { command: "printf '%s' \"$PYROMIND_API_KEY\"" });
  assert.equal(result.details.resultCategory, "success");
  assert.equal(contexts, 1);
  assert(JSON.stringify(result).includes("[REDACTED]"));
  assert(!JSON.stringify(result).includes("main-only-sdk-key"));
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

test("terminal output logs are read-only and scoped to their conversation", async (t) => {
  const { root, workspace } = await workspaceLayout();
  t.after(() => rm(root, { recursive: true, force: true }));
  const policy = await WorkspaceAccessPolicy.create({ workspaceRoot: workspace, readOnlyRoots: [] });
  const log = join(policy.terminalTempRoot, "pi-bash-0123456789abcdef.log");
  await writeFile(log, "output");
  assert.equal(await policy.resolvePath(log, "read"), log);
  await assert.rejects(policy.resolvePath(log, "write"), /PATH_SCOPE_ERROR/);
  const foreign = join(root, "other", "pi", "terminal-output", "pi-bash-fedcba9876543210.log");
  await mkdir(join(foreign, ".."), { recursive: true });
  await writeFile(foreign, "other output");
  const link = join(policy.terminalTempRoot, "pi-bash-aaaaaaaaaaaaaaaa.log");
  await symlink(foreign, link);
  await assert.rejects(policy.resolvePath(link, "read"), /PATH_SCOPE_ERROR/);
  await assert.rejects(policy.resolvePath(foreign, "read"), /PATH_SCOPE_ERROR/);
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
