import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import {
  SandboxTerminalOperations, type SandboxTerminalSocket,
} from "../src/sandbox-terminal.js";

const hasPty = process.platform !== "win32" &&
  spawnSync("python3", ["-c", "import pty"]).status === 0;

test("real PTY runs long uploaded scripts and acknowledged large environment without leaking secrets", {
  skip: !hasPty && "requires Python and a POSIX PTY",
  timeout: 20000,
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
    startupTimeoutMs: 3000,
    writeFile: async (path, content) => {
      uploaded.push(path);
      await mkdir(dirname(path), {recursive: true});
      await writeFile(path, content);
    },
    connect: () => {
      const child = spawn("python3", ["-u", fileURLToPath(new URL("./fixtures/terminal_pty.py", import.meta.url))], {
        env: { ...process.env, PATH: `${bin}:${process.env.PATH}` },
      });
      processes.push(child);
      const socket: SandboxTerminalSocket = {
        send: (data) => {
          if (loseEnvironmentAck && data.includes(":env0")) suppressOutput = true;
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
    await assert.rejects(operations.exec("touch should-not-start", root, {
      env: { TEST_VALUE: "a".repeat(4000) }, timeout: 5, onData: () => {},
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
