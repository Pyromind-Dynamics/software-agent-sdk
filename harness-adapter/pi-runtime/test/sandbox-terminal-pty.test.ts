import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import { once } from "node:events";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { createTools, type SandboxTools } from "../src/tools.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import {
  SandboxTerminalOperations, type SandboxTerminalSocket,
} from "../src/sandbox-terminal.js";
import { SandboxTerminalFileClient } from "../src/sandbox-terminal-files.js";

const hasPty = process.platform !== "win32" &&
  spawnSync("python3", ["-c", "import pty"]).status === 0;

test("real PTY stages scripts inline when the file API is unavailable", {
  skip: !hasPty && "requires Python and a POSIX PTY",
  timeout: 20000,
}, async () => {
  const root = await mkdtemp(join(tmpdir(), "terminal-inline-pty-"));
  const bin = join(root, "bin");
  await mkdir(bin);
  // macOS does not ship setsid. Use the same POSIX process-group semantics.
  const python = spawnSync("which", ["python3"], {encoding: "utf8"}).stdout.trim();
  await writeFile(join(bin, "setsid"), `#!${python}\nimport os,sys\nos.setsid()\nos.execvp(sys.argv[1],sys.argv[1:])\n`, {mode: 0o755});
  const child = spawn("python3", ["-u", fileURLToPath(new URL("./fixtures/terminal_pty.py", import.meta.url))], {
    env: { ...process.env, PATH: `${bin}:${process.env.PATH}` },
  });
  const socket: SandboxTerminalSocket = {
    send: (data) => { child.stdin!.write(data); },
    close: () => { child.stdin!.end(); },
    onOpen: (handler) => { queueMicrotask(handler); },
    onMessage: (handler) => { child.stdout!.on("data", (data: Buffer) => handler(Uint8Array.from(data).buffer)); },
    onClose: (handler) => { child.on("close", handler); },
    onError: (handler) => { child.on("error", handler); },
  };
  const operations = new SandboxTerminalOperations({
    baseUrl: "https://unused.example", wsBaseUrl: "https://unused.example",
    sandboxId: "pty", apiKey: "unused", workspacePath: root, storagePath: root,
  }, {
    // macOS caps one canonical line at 1024 bytes, so the local PTY only
    // accepts bursts the Linux sandbox queue would accept with headroom.
    startupTimeoutMs: 3000, batchBudgetBytes: 1024, connect: () => socket,
  });
  try {
    // A long script forces the staging to span many chunked lines.
    const payload = "inline-" + "z".repeat(6000);
    const chunks: Buffer[] = [];
    const result = await operations.exec(`printf '%s\\n' '${payload}'`, root, {
      timeout: 10, onData: (data) => chunks.push(data),
    });
    assert.equal(result.exitCode, 0);
    // No file API, so the batched TTY staging path carries the script.
    assert.equal(result.execution.staging, "batch");
    assert.equal(Buffer.concat(chunks).toString().replaceAll("\r\n", "\n"), `${payload}\n`);
  } finally {
    child.stdin!.end();
    if (child.exitCode === null && child.signalCode === null)
      await new Promise<void>((resolve) => child.once("close", () => resolve()));
    await rm(root, {recursive: true, force: true});
  }
});

test("real PTY carries file bytes over the terminal for write and read", {
  skip: !hasPty && "requires Python and a POSIX PTY",
  timeout: 60000,
}, async () => {
  const root = await mkdtemp(join(tmpdir(), "terminal-file-pty-"));
  const bin = join(root, "bin");
  await mkdir(bin);
  // macOS does not ship setsid. Use the same POSIX process-group semantics.
  const python = spawnSync("which", ["python3"], {encoding: "utf8"}).stdout.trim();
  await writeFile(join(bin, "setsid"), `#!${python}\nimport os,sys\nos.setsid()\nos.execvp(sys.argv[1],sys.argv[1:])\n`, {mode: 0o755});
  await mkdir(join(root, "public_data"), { recursive: true });
  const processes: ReturnType<typeof spawn>[] = [];
  const connect = (): SandboxTerminalSocket => {
    const child = spawn("python3", ["-u", fileURLToPath(new URL("./fixtures/terminal_pty.py", import.meta.url))], {
      env: { ...process.env, PATH: `${bin}:${process.env.PATH}` },
    });
    processes.push(child);
    return {
      send: (data) => { child.stdin!.write(data); },
      close: () => { child.stdin!.end(); },
      onOpen: (handler) => { queueMicrotask(handler); },
      onMessage: (handler) => { child.stdout!.on("data", (data: Buffer) => handler(Uint8Array.from(data).buffer)); },
      onClose: (handler) => { child.on("close", handler); },
      onError: (handler) => { child.on("error", handler); },
    };
  };
  const operations = new SandboxTerminalOperations({
    baseUrl: "https://unused.example", wsBaseUrl: "https://unused.example",
    sandboxId: "pty", apiKey: "unused", workspacePath: root, storagePath: root,
  }, {
    // macOS caps one canonical line at 1024 bytes, so the local PTY only
    // accepts bursts the Linux sandbox queue would accept with headroom.
    startupTimeoutMs: 15000, batchBudgetBytes: 1024, connect,
  });
  const client = new SandboxTerminalFileClient(operations);
  try {
    const content = "#!/usr/bin/env python3\n# 中文注释\n" + "x".repeat(4000) + "\n";
    await client.write("public_data/script.py", content);
    assert.equal(await readFile(join(root, "public_data", "script.py"), "utf8"), content);
    assert.deepEqual(await client.read("public_data/script.py"), Buffer.from(content));
  } finally {
    for (const child of processes) {
      child.stdin!.end();
      if (child.exitCode === null && child.signalCode === null)
        await new Promise<void>((resolve) => child.once("close", () => resolve()));
    }
    await rm(root, {recursive: true, force: true});
  }
});

test("real PTY runs long uploaded scripts and acknowledged large environment without leaking secrets", {
  skip: !hasPty && "requires Python and a POSIX PTY",
  timeout: 60000,
}, async () => {
  const root = await mkdtemp(join(tmpdir(), "terminal-real-pty-"));
  const bin = join(root, "bin");
  await mkdir(bin);
  // macOS does not ship setsid. Use the same POSIX process-group semantics.
  const python = spawnSync("which", ["python3"], {encoding: "utf8"}).stdout.trim();
  await writeFile(join(bin, "setsid"), `#!${python}\nimport os,sys\nos.setsid()\nos.execvp(sys.argv[1],sys.argv[1:])\n`, {mode: 0o755});
  const processes: ReturnType<typeof spawn>[] = [];
  const uploaded: string[] = [];
  let loseEnvironmentAck = false;
  let suppressOutput = false;
  const operations = new SandboxTerminalOperations({
    baseUrl: "https://unused.example", wsBaseUrl: "https://unused.example",
    sandboxId: "pty", apiKey: "unused", workspacePath: root, storagePath: root,
  }, {
    // The first command also stages the session harness, and this test sends an
    // 18KB secret which is staged in bounded bursts.
    startupTimeoutMs: 15000,
    batchBudgetBytes: 1024,
    writeFile: async (path, content) => {
      uploaded.push(path);
      await mkdir(dirname(path), {recursive: true});
      await writeFile(path, content);
    },
    connect: () => {
      suppressOutput = false;
      const child = spawn("python3", ["-u", fileURLToPath(new URL("./fixtures/terminal_pty.py", import.meta.url))], {
        env: { ...process.env, PATH: `${bin}:${process.env.PATH}` },
      });
      processes.push(child);
      const socket: SandboxTerminalSocket = {
        send: (data) => {
          if (loseEnvironmentAck && data.includes("env.b64.b64")) suppressOutput = true;
          if (data.includes(":stop")) suppressOutput = false;
          child.stdin!.write(data);
        },
        close: () => { child.stdin!.end(); },
        onOpen: (handler) => { queueMicrotask(handler); },
        onMessage: (handler) => { child.stdout!.on("data", (data: Buffer) => {
          if (suppressOutput) return;
          handler(Uint8Array.from(data).buffer);
        }); },
        onClose: (handler) => { child.on("close", handler); },
        onError: (handler) => { child.on("error", handler); },
      };
      return socket;
    },
  });
  try {
    const chunks: Buffer[] = [];
    const secret = "private-test-key-" + "abc".repeat(6000);
    const encoded = Buffer.from(JSON.stringify({DF_API_KEY: secret})).toString("base64");
    const output = "历史经验".repeat(3000);
    const result = await operations.exec(`python3 -c 'import os;print(os.environ["DF_API_KEY"]);print("${output}")'`, root, {
      env: { DF_API_KEY: secret }, timeout: 10, onData: (data) => chunks.push(data),
    });
    assert.equal(result.exitCode, 0);
    assert.equal(Buffer.concat(chunks).toString().replaceAll("\r\n", "\n"), `[REDACTED]\n${output}\n`);
    for (const path of uploaded) {
      const text = await readFile(path, "utf8");
      assert.ok(!text.includes(secret));
      assert.ok(!text.includes(encoded));
      if (path.endsWith("cmd.sh")) {
        const log = await readFile(join(dirname(path), "out.log"), "utf8");
        assert.ok(!log.includes(secret));
        assert.ok(!log.includes(encoded));
      }
    }
    const quiet = await operations.exec("true", root, {timeout: 5, onData: () => assert.fail("no stdout expected")});
    assert.equal(quiet.exitCode, 0);

    // Verify that cancellation stops the actual remote process group.
    const controller = new AbortController();
    const running = operations.exec("printf 'ready\\n'; sleep 30", root, {
      timeout: 10, signal: controller.signal,
      onData: () => controller.abort(),
    });
    await assert.rejects(running, (error: unknown) =>
      error instanceof Error && "execution" in error &&
      (error.execution as {stopped: boolean}).stopped === true);
    assert.equal((await operations.exec("true", root, {timeout: 5, onData: () => {}})).exitCode, 0);
    loseEnvironmentAck = true;
    // The command deadline also caps the startup budget, so a one second
    // timeout keeps the lost acknowledgement from stalling the test.
    await assert.rejects(operations.exec("touch should-not-start", root, {
      env: { TEST_VALUE: "a".repeat(4000) }, timeout: 1, onData: () => {},
    }), (error: unknown) => error instanceof Error && "execution" in error &&
      (error.execution as {phase: string; started: boolean; stopped: boolean}).phase === "environment" &&
      (error.execution as {started: boolean; stopped: boolean}).started === false &&
      (error.execution as {stopped: boolean}).stopped === true);
    await assert.rejects(readFile(join(root, "should-not-start")));
    loseEnvironmentAck = false;
    assert.equal((await operations.exec("true", root, {timeout: 5, onData: () => {}})).exitCode, 0);
  } finally {
    for (const child of processes) {
      child.stdin!.end();
      if (child.exitCode === null && child.signalCode === null)
        await new Promise<void>((resolve) => child.once("close", () => resolve()));
    }
    await rm(root, {recursive: true, force: true});
  }
});

test("agent lanes in one sandbox use separate PTYs and isolate concurrent environments and cancellation", {
  skip: !hasPty && "requires Python and a POSIX PTY",
  timeout: 30_000,
}, async (t) => {
  const root = await mkdtemp(join(tmpdir(), "terminal-agent-lanes-"));
  const fastPath = process.env.PYROMIND_PI_TERMINAL_FAST_PATH;
  // A local macOS PTY cannot accept Linux-sized input bursts.
  process.env.PYROMIND_PI_TERMINAL_FAST_PATH = "0";
  t.after(() => {
    if (fastPath === undefined) delete process.env.PYROMIND_PI_TERMINAL_FAST_PATH;
    else process.env.PYROMIND_PI_TERMINAL_FAST_PATH = fastPath;
  });
  const bin = join(root, "bin");
  await mkdir(bin);
  const python = spawnSync("which", ["python3"], { encoding: "utf8" }).stdout.trim();
  await writeFile(join(bin, "setsid"), `#!${python}\nimport os,sys\nos.setsid()\nos.execvp(sys.argv[1],sys.argv[1:])\n`, { mode: 0o755 });
  const processes: ReturnType<typeof spawn>[] = [];
  const connect = (): SandboxTerminalSocket => {
    const child = spawn("python3", ["-u", fileURLToPath(new URL("./fixtures/terminal_pty.py", import.meta.url))], {
      env: { ...process.env, PATH: `${bin}:${process.env.PATH}` },
    });
    processes.push(child);
    return {
      send: (data) => { child.stdin!.write(data); },
      close: () => { child.stdin!.end(); },
      onOpen: (handler) => { queueMicrotask(handler); },
      onMessage: (handler) => { child.stdout!.on("data", (data: Buffer) => handler(Uint8Array.from(data).buffer)); },
      onClose: (handler) => { child.on("close", handler); },
      onError: (handler) => { child.on("error", handler); },
    };
  };
  const fileApi = createServer((_request, response) => { response.writeHead(503).end(); });
  fileApi.listen(0, "127.0.0.1");
  await once(fileApi, "listening");
  t.after(() => { fileApi.closeAllConnections(); fileApi.close(); });
  const address = fileApi.address();
  assert.ok(address && typeof address !== "string");
  let resolves = 0;
  let sandbox!: SandboxTools;
  await createTools({} as JsonlRpcPeer, new NodeExecutionEnv({ cwd: root }), root, "sandbox", [], undefined, undefined, [], {
    terminalConnect: connect,
    scriptChunkChars: 512,
    terminalTimeouts: { startupTimeoutMs: 5000 },
    sandbox: async () => {
      resolves++;
      return {
        baseUrl: `http://127.0.0.1:${address.port}`, wsBaseUrl: "https://unused.example",
        sandboxId: "one-sandbox", apiKey: "unused", workspacePath: root, storagePath: root,
      };
    },
    onSandboxToolsReady: (tools) => { sandbox = tools; },
  });
  const main = sandbox.main.terminal;
  const childA = sandbox.createLane("child-a").terminal;
  const childBLane = sandbox.createLane("child-b");
  const childB = childBLane.terminal;
  assert.equal(processes.length, 0, "idle lanes should not open a socket");
  const controller = new AbortController();
  const results: string[] = ["", "", ""];
  const run = (operations: SandboxTools["main"]["terminal"], index: number, signal?: AbortSignal) =>
    operations.exec('touch "ready-$TEST_OWNER"; while [ ! -f release ]; do sleep 0.1; done; printf "%s\\n" "$TEST_OWNER"; exit "$TEST_STATUS"', root, {
      timeout: 15, signal,
      env: { TEST_OWNER: String(index), TEST_STATUS: String(index) },
      onData: (data) => { results[index] += data.toString(); },
    });
  const calls = [run(main, 0), run(childA, 1, controller.signal), run(childB, 2)];
  const settled = Promise.allSettled(calls);
  try {
    for (let attempt = 0; ; attempt++) {
      try {
        await Promise.all([0, 1, 2].map((index) => readFile(join(root, `ready-${index}`))));
        break;
      } catch (error) {
        if (attempt >= 200) throw error;
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
    }
    assert.equal(processes.length, 3);
    assert.equal(resolves, 1);
    const mainJournal = JSON.parse(await readFile(join(root, "pi/terminal-runs/main.json"), "utf8"));
    assert.equal(mainJournal.sandboxId, "one-sandbox");
    controller.abort();
    await assert.rejects(calls[1]!, (error: any) => error.execution?.stopped === true);
    await writeFile(join(root, "release"), "");
    const mainResult = await calls[0]!;
    const childResult = await calls[2]!;
    assert.equal(mainResult.exitCode, 0);
    assert.equal(childResult.exitCode, 2);
    assert.deepEqual(results.map((value) => value.trim()), ["0", "", "2"]);
    assert.match(mainResult.execution.remote_run_dir, /\/v2\/main\//);
    assert.match(childResult.execution.remote_run_dir, /\/v2\/child-b\//);
    await assert.rejects(readFile(join(root, ".pyromind-agent-runs/v2/child-b/harness/env.b64")), { code: "ENOENT" });
    assert.equal((await childB.exec("true", root, { onData: () => {} })).exitCode, 0);
    assert.equal(processes.length, 3, "the surviving lane must reuse its socket");
    assert.equal((await childBLane.access.readFile!(join(root, "ready-2"), 32)).byteLength, 0);
    assert.equal(processes.length, 3, "HTTP file fallback must reuse the child's PTY");
    const journal = join(root, "pi/terminal-runs/main.json");
    await assert.rejects(readFile(journal), { code: "ENOENT" });
    await main.disconnect();
    // Simulate a restart with unresolved evidence. Recovery can probe an old
    // PID but must neither signal it nor launch another command on that lane.
    const oldPid = join(mainResult.execution.remote_run_dir, "launch_pid");
    const completedPid = await readFile(oldPid, "utf8");
    await writeFile(oldPid, String(process.pid));
    await writeFile(journal, JSON.stringify(mainJournal));
    const recovered = sandbox.createLane("main").terminal;
    await assert.rejects(recovered.exec("touch after-restart", root, { onData: () => {} }),
      (error: any) => error.execution?.error_code === "termination_unconfirmed" && !error.execution.started);
    await assert.rejects(readFile(join(root, "after-restart")), { code: "ENOENT" });
    await writeFile(oldPid, completedPid);
    assert.equal((await recovered.exec("touch after-restart", root, { onData: () => {} })).exitCode, 0);
  } finally {
    await writeFile(join(root, "release"), "");
    controller.abort();
    await settled;
    for (const child of processes) {
      child.stdin!.end();
      if (child.exitCode === null && child.signalCode === null) {
        await new Promise<void>((resolve) => child.once("close", () => resolve()));
      }
    }
    await rm(root, { recursive: true, force: true });
  }
});
