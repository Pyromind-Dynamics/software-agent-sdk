import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  buildCommandScript,
  buildResumeLine,
  buildStartLine,
  buildWatchScript,
  SandboxTerminalOperations,
  sandboxTerminalUrl,
  TERMINAL_SCRIPT_CHUNK_CHARS,
  TerminalMarkerScanner,
  TERMINAL_BEGIN_MARKER,
  TERMINAL_END_MARKER,
  TERMINAL_HEARTBEAT_MARKER,
  type SandboxTerminalConnector,
  type SandboxTerminalSocket,
} from "../src/sandbox-terminal.js";
import type { SandboxEndpoint } from "../src/sandbox-operations.js";

const ENDPOINT: SandboxEndpoint = {
  baseUrl: "https://portal.example.com",
  wsBaseUrl: "https://cluster.example.com",
  sandboxId: "sbx-1",
  apiKey: "secret",
  workspacePath: "/target-workspace/.pyromind-agent/conv-1",
  storagePath: "/target-workspace",
};

function frames(token: string) {
  return {
    begin: `${TERMINAL_BEGIN_MARKER}${token}\n`,
    heartbeat: `${TERMINAL_HEARTBEAT_MARKER}${token}\n`,
    end: (exitCode: number): string => `${TERMINAL_END_MARKER}${token}:${exitCode}\n`,
  };
}

class FakeSocket implements SandboxTerminalSocket {
  readonly sent: string[] = [];
  onSend: ((line: string) => void) | undefined;
  private openHandler: (() => void) | undefined;
  private messageHandler: ((data: string | ArrayBuffer) => void) | undefined;
  private closeHandler: (() => void) | undefined;
  private errorHandler: ((error: unknown) => void) | undefined;

  onOpen(handler: () => void): void {
    this.openHandler = handler;
  }

  onMessage(handler: (data: string | ArrayBuffer) => void): void {
    this.messageHandler = handler;
  }

  onClose(handler: () => void): void {
    this.closeHandler = handler;
  }

  onError(handler: (error: unknown) => void): void {
    this.errorHandler = handler;
  }

  send(data: string): void {
    this.sent.push(data);
    this.onSend?.(data);
  }

  close(): void {
    this.closeHandler?.();
  }

  open(): void {
    this.openHandler?.();
  }

  emit(data: string): void {
    this.messageHandler?.(data);
  }

  fail(error: unknown): void {
    this.errorHandler?.(error);
  }
}

type Episode = (socket: FakeSocket, line: string, token: string) => void;

function scriptedConnector(episodes: Episode[]): {
  connect: SandboxTerminalConnector;
  sockets: FakeSocket[];
} {
  const sockets: FakeSocket[] = [];
  let index = 0;
  const connect: SandboxTerminalConnector = () => {
    const episode = episodes[Math.min(index, episodes.length - 1)]!;
    index += 1;
    const socket = new FakeSocket();
    sockets.push(socket);
    let started = false;
    socket.onSend = (line) => {
      if (started) return;
      started = true;
      const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
      assert.ok(token, "start and resume lines carry the run token");
      episode(socket, line, token);
    };
    queueMicrotask(() => socket.open());
    return socket;
  };
  return { connect, sockets };
}

function run(
  episodes: Episode[],
  options: { signal?: AbortSignal; timeout?: number } = {},
) {
  const { connect, sockets } = scriptedConnector(episodes);
  const operations = new SandboxTerminalOperations(ENDPOINT, { connect });
  const chunks: Buffer[] = [];
  const result = operations.exec("ls -la", "", {
    onData: (data) => chunks.push(data),
    ...options,
  });
  return { result, sockets, output: () => Buffer.concat(chunks).toString("utf8") };
}

function runCommandScript(command: string): {
  stdout: string;
  stderr: string;
  exitCode: string;
} {
  const root = mkdtempSync(join(tmpdir(), "sandbox-terminal-"));
  const runDir = join(root, "run");
  const workspace = join(root, "workspace");
  mkdirSync(runDir);
  mkdirSync(workspace);
  const scriptPath = join(runDir, "cmd.sh");
  writeFileSync(scriptPath, buildCommandScript(runDir, workspace, command));
  try {
    const result = spawnSync("sh", [scriptPath], { encoding: "utf8" });
    return {
      stdout: result.stdout,
      stderr: result.stderr,
      exitCode: readFileSync(join(runDir, "rc"), "utf8"),
    };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test("terminal scanner drops the prompt, swallows heartbeats, and reads the exit code", () => {
  const scanner = new TerminalMarkerScanner("token");
  const marker = frames("token");
  const first = scanner.push(Buffer.from(`prompt$ ls\r\n${marker.begin}`));
  assert.equal(first.output.toString("utf8"), "");
  const second = scanner.push(Buffer.from(`hello\n${marker.heartbeat}world\n${marker.end(3)}`));
  assert.equal(second.output.toString("utf8"), "hello\nworld\n");
  assert.equal(second.exitCode, 3);
  assert.equal(scanner.done, true);
});

test("terminal scanner holds back markers split across frames", () => {
  const scanner = new TerminalMarkerScanner("token");
  const marker = frames("token");
  scanner.push(Buffer.from(marker.begin));
  const partial = scanner.push(Buffer.from("payload\n__PM_EN"));
  assert.equal(partial.output.toString("utf8"), "payload\n");
  const rest = scanner.push(Buffer.from("D__token:0\n"));
  assert.equal(rest.output.toString("utf8"), "");
  assert.equal(rest.exitCode, 0);
});

test("terminal scanner consumes CRLF control lines from a TTY", () => {
  const scanner = new TerminalMarkerScanner("token");
  const marker = frames("token");
  scanner.push(Buffer.from(`prompt\r\n${marker.begin.replace("\n", "\r\n")}`));
  const output = scanner.push(
    Buffer.from(
      `hello\r\n${marker.heartbeat.replace("\n", "\r\n")}` +
        `world\r\n${marker.end(0).replace("\n", "\r\n")}`,
    ),
  );
  assert.equal(output.output.toString("utf8"), "hello\r\nworld\r\n");
  assert.equal(output.exitCode, 0);
});

test("terminal scanner waits for a split exit code", () => {
  const scanner = new TerminalMarkerScanner("token");
  scanner.push(Buffer.from(frames("token").begin));
  assert.equal(scanner.push(Buffer.from("payload\n")).output.toString("utf8"), "payload\n");
  const partial = scanner.push(Buffer.from(`${TERMINAL_END_MARKER}token:1`));
  assert.equal(partial.output.toString("utf8"), "");
  assert.equal(scanner.done, false);
  const complete = scanner.push(Buffer.from("2\n"));
  assert.equal(complete.output.toString("utf8"), "");
  assert.equal(complete.exitCode, 12);
  assert.equal(scanner.done, true);
});

test("terminal start line launches the command before tailing its output", () => {
  const runDir = "/target-workspace/.pyromind-agent/conv-1/.pyromind-agent-runs/call-1";
  const line = buildStartLine(
    runDir,
    "token",
    buildCommandScript(runDir, "/target-workspace/.pyromind-agent/conv-1", "echo hi"),
    buildWatchScript(60),
  );
  assert.match(
    line,
    /setsid sh '\/target-workspace\/\.pyromind-agent\/conv-1\/\.pyromind-agent-runs\/call-1\/cmd\.sh' > '\/target-workspace\/\.pyromind-agent\/conv-1\/\.pyromind-agent-runs\/call-1\/out\.log' 2>&1 &/,
  );
  assert.ok(line.indexOf("cmd.sh") < line.lastIndexOf("watch.sh"));
});

test("large script uploads survive terminal line limits and preserve exact bytes", () => {
  const root = mkdtempSync(join(tmpdir(), "genome-tty-"));
  try {
    const runDir = join(root, "run");
    const command = "printf '%s' '" + "历史经验".repeat(2000) + "'";
    const script = buildCommandScript(runDir, root, command);
    const watch = buildWatchScript(60).replace("stty echo 2>/dev/null", "true");
    const line = buildStartLine(runDir, "token", script, watch);
    assert.ok(line.split("\n").every((part) => Buffer.byteLength(part) < 1024));
    // Supply terminal/process-group primitives while exercising actual shell parsing.
    execFileSync("sh", ["-c", "stty() { :; }; setsid() { \"$@\"; };\n" + line], { timeout: 10000 });
    assert.equal(readFileSync(join(runDir, "cmd.sh"), "utf8"), script);
    assert.equal(readFileSync(join(runDir, "out.log"), "utf8"), "历史经验".repeat(2000));
    assert.equal(readFileSync(join(runDir, "rc"), "utf8"), "0");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("terminal start line keeps every line below the canonical input cap", () => {
  const runDir = "/target-workspace/.pyromind-agent/conv-1/.pyromind-agent-runs/call-1";
  // Linux caps one canonical TTY line at MAX_CANON (4096 bytes); stay far below.
  const lineBudget = 1024;
  const command = [
    "cd /target-workspace/datasets/test_100/test_100",
    "python3 - <<'EOF'",
    "import json",
    `print(${JSON.stringify("x".repeat(3000))})`,
    // The measured failure packed this much into one 5 KB start line.
    `payload = ${JSON.stringify("y".repeat(2000))}`,
    "EOF",
  ].join("\n");
  const line = buildStartLine(
    runDir,
    "token",
    buildCommandScript(runDir, "/target-workspace/.pyromind-agent/conv-1", command),
    buildWatchScript(60),
  );
  const lines = line.split("\n").filter((entry) => entry.length > 0);
  assert.ok(lines.length > 3, "the start line is split into standalone commands");
  for (const entry of lines) {
    assert.ok(
      Buffer.byteLength(entry, "utf8") < lineBudget,
      `start line fragment is too long for the TTY: ${entry.slice(0, 60)}`,
    );
  }
  assert.ok(TERMINAL_SCRIPT_CHUNK_CHARS < lineBudget);
  assert.ok(line.endsWith("\n"));
});

test("terminal start line setup rebuilds both scripts byte for byte", () => {
  const root = mkdtempSync(join(tmpdir(), "sandbox-start-line-"));
  try {
    const runDir = join(root, "run");
    const workspace = join(root, "workspace");
    mkdirSync(workspace);
    const command = ["cat <<'EOF'", "hello", "EOF", "exit 3"].join("\n");
    const commandScript = buildCommandScript(runDir, workspace, command);
    const watchScript = buildWatchScript(60);
    const lines = buildStartLine(runDir, "token", commandScript, watchScript)
      .split("\n")
      .filter((entry) => entry.length > 0);
    const launchAt = lines.findIndex((entry) => entry.includes("setsid"));
    assert.ok(launchAt > 0, "the launch line follows the staged scripts");
    const setup = lines.slice(0, launchAt);
    const result = spawnSync("sh", ["-c", setup.join("\n")], { encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(readFileSync(join(runDir, "cmd.sh"), "utf8"), commandScript);
    assert.equal(readFileSync(join(runDir, "watch.sh"), "utf8"), watchScript);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("command script supports heredocs and records syntax errors", () => {
  const heredoc = runCommandScript(["cat <<'EOF'", "hello", "EOF"].join("\n"));
  assert.equal(heredoc.stdout, "hello\n");
  assert.equal(heredoc.exitCode, "0");

  const syntaxError = runCommandScript("if true; then");
  assert.match(syntaxError.stderr, /syntax error/i);
  assert.notEqual(syntaxError.exitCode, "0");

  assert.equal(runCommandScript("exit 7").exitCode, "7");
});

test("terminal operations stream output and resolve the exit code", async () => {
  const { result, output } = run([
    (socket, _line, token) => {
      const frame = frames(token);
      socket.emit(frame.begin);
      socket.emit("partial ");
      socket.emit(frame.heartbeat);
      socket.emit("output\n");
      socket.emit(frame.end(0));
    },
  ]);
  assert.deepEqual(await result, { exitCode: 0 });
  assert.equal(output(), "partial output\n");
});

test("terminal operations resume the same run after a dropped connection", async () => {
  const { result, output, sockets } = run([
    (socket, _line, token) => {
      socket.emit(frames(token).begin);
      socket.emit("first half ");
      socket.close();
    },
    (socket, _line, token) => {
      socket.emit(frames(token).begin);
      socket.emit("second half\n");
      socket.emit(frames(token).end(7));
    },
  ]);
  assert.deepEqual(await result, { exitCode: 7 });
  assert.equal(output(), "first half second half\n");
  assert.equal(sockets.length, 2);
  assert.ok(sockets[1]!.sent[0]!.includes("watch.sh"));
});

test("terminal operations abort by cancelling the process group", async () => {
  const controller = new AbortController();
  const { result, sockets } = run(
    [
      (socket, _line, token) => {
        socket.emit(frames(token).begin);
        setTimeout(() => controller.abort(), 10);
      },
    ],
    { signal: controller.signal },
  );
  await assert.rejects(result, /aborted/);
  assert.ok(sockets[0]!.sent.includes("\u0003"));
  assert.ok(sockets[0]!.sent.some((line) => line.includes("kill -TERM")));
});

test("terminal operations surface a lost connection as an error", async () => {
  const { result } = run([
    (socket, _line, token) => {
      socket.emit(frames(token).begin);
      socket.close();
    },
  ]);
  await assert.rejects(result, /connection lost/);
});

test("sandbox terminal url uses the cluster WebSocket scheme and carries the token", () => {
  const url = new URL(sandboxTerminalUrl(ENDPOINT));
  assert.equal(url.protocol, "wss:");
  assert.equal(url.host, "cluster.example.com");
  assert.equal(url.pathname, "/sandboxes/sbx-1/terminal");
  assert.equal(url.searchParams.get("token"), "secret");
});

test("environment reaches remote commands without entering staged scripts or logs", () => {
  const root = mkdtempSync(join(tmpdir(), "terminal-env-"));
  try {
    const runDir = join(root, "run");
    const env = { ORDINARY: "space ' quote $value", PYROMIND_DATAFLOW_PROFILES: JSON.stringify({text:{DF_API_KEY:"very-private-key"}}) };
    const command = "python3 -c 'import os,json; print(os.environ[\"ORDINARY\"]); print(json.loads(os.environ[\"PYROMIND_DATAFLOW_PROFILES\"])[\"text\"][\"DF_API_KEY\"])'";
    const script = buildCommandScript(runDir, root, command, true);
    const line = buildStartLine(runDir, "token", script, buildWatchScript(60).replace("stty echo 2>/dev/null", "true"), env);
    assert.ok(line.split("\n").every((part) => Buffer.byteLength(part) < 1024));
    execFileSync("sh", ["-c", 'stty() { :; }; setsid() { "$@"; };\n' + line], {timeout:10000});
    assert.equal(readFileSync(join(runDir,"rc"),"utf8"), "0");
    assert.equal(readFileSync(join(runDir,"out.log"),"utf8"), "space ' quote $value\n[REDACTED]\n");
    assert.ok(!readFileSync(join(runDir,"cmd.sh"),"utf8").includes("very-private-key"));
    assert.ok(!buildResumeLine(runDir,"token").includes("PYROMIND_DATAFLOW_PROFILES"));
  } finally { rmSync(root,{recursive:true,force:true}); }
});
