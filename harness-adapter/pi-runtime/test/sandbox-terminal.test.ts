import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  buildCommandScript,
  buildLaunchScript,
  buildResumeLine,
  buildCancelLine,
  SandboxTerminalExecutionError,
  buildStartLine,
  buildWatchScript,
  SandboxTerminalOperations,
  LazySandboxTerminalOperations,
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
import { SandboxEndpointSession } from "../src/sandbox-operations.js";

const ENDPOINT: SandboxEndpoint = {
  baseUrl: "https://portal.example.com",
  wsBaseUrl: "https://cluster.example.com",
  sandboxId: "sbx-1",
  apiKey: "secret",
  workspacePath: "/target-workspace/.pyromind-agent/conv-1",
  storagePath: "/target-workspace",
};

test("endpoint resolution shares the bounded startup deadline", async () => {
  const terminal = new LazySandboxTerminalOperations(
    new SandboxEndpointSession(() => new Promise(() => {})),
    { startupTimeoutMs: 20 },
  );
  await assert.rejects(terminal.exec("true", ".", { onData: () => {} }),
    (error: unknown) => error instanceof SandboxTerminalExecutionError &&
      error.execution.phase === "connecting" &&
      error.execution.error_code === "startup_timeout" &&
      error.execution.started === false && error.execution.stopped === true);
});

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

export type Episode = (socket: FakeSocket, line: string, token: string) => void;

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
    socket.onSend = (line) => {
      if (line === "\u0003") return;
      const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)' '([^']+)'/.exec(line);
      if (ack) {
        const value = ack[2] === "stop" ? "yes" : ack[3];
        socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${value}\n`);
        return;
      }
      const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
      assert.ok(token, "watch line carries the run token");
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
  const operations = new SandboxTerminalOperations(ENDPOINT, { connect, writeFile: async () => {} });
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

test("launch uses uploaded scripts and an atomic guard, without staging or secrets", () => {
  const line = buildStartLine("/workspace/run", "token", true);
  assert.match(buildLaunchScript("/workspace/run"), /mkdir '.*launched'/);
  assert.match(buildLaunchScript("/workspace/run"), /setsid sh/);
  assert.match(line, /launch\.sh/);
  assert.ok(!line.includes("base64"));
  assert.ok(!line.includes("watch.sh"));
  assert.ok(Buffer.byteLength(line) < 1024);
  assert.ok(!buildResumeLine("/workspace/run", "token").includes("setsid"));
  assert.ok(!buildCancelLine("/workspace/run", "token").includes("rm -f"));
  assert.ok(TERMINAL_SCRIPT_CHUNK_CHARS < 1024);
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
  const completed = await result;
  assert.equal(completed.exitCode, 0);
  assert.equal(completed.execution.phase, "finished");
  assert.equal(completed.execution.started, true);
  assert.equal(completed.execution.stopped, true);
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
  assert.equal((await result).exitCode, 7);
  assert.equal(output(), "first half second half\n");
  assert.equal(sockets.length, 2);
  assert.ok(sockets[1]!.sent.some((line) => line.includes("watch.sh")));
  assert.equal(sockets.flatMap((socket) => socket.sent).filter((line) => line.includes("launch.sh")).length, 1);
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

test("startup deadline includes a socket which never opens", async () => {
  const operations = new SandboxTerminalOperations(ENDPOINT, {
    writeFile: async () => {},
    connect: () => new FakeSocket(),
    startupTimeoutMs: 20, stopTimeoutMs: 20,
  });
  await assert.rejects(operations.exec("true", "", {onData: () => {}}),
    (error: unknown) => error instanceof SandboxTerminalExecutionError &&
      error.execution.phase === "connecting" &&
      error.execution.error_code === "startup_timeout" &&
      error.execution.stopped === true);
});

test("missing environment acknowledgement never launches and clears the buffer", async () => {
  const socket = new FakeSocket();
  socket.onSend = (line) => {
    const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)'/.exec(line);
    if (!ack || ack[2] === "env0") return;
    socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${ack[2] === "stop" ? "yes" : "ok"}\n`);
  };
  const writes: Buffer[] = [];
  const operations = new SandboxTerminalOperations(ENDPOINT, {
    connect: () => {queueMicrotask(() => socket.open()); return socket;},
    writeFile: async (_path, bytes) => {writes.push(bytes);},
    startupTimeoutMs: 30, stopTimeoutMs: 100,
  });
  await assert.rejects(operations.exec("true", "", {
    env: {API_KEY: "private-" + "x".repeat(2000)}, onData: () => {},
  }), (error: unknown) => error instanceof SandboxTerminalExecutionError &&
    error.execution.phase === "environment" && error.execution.stopped);
  assert.ok(!socket.sent.some((line) => line.includes("setsid")));
  assert.ok(socket.sent.some((line) => line.startsWith("unset PYROMIND_ENV_BUFFER")));
  assert.equal(socket.sent.filter((line) => line.includes("env0")).length, 1);
  assert.ok(!socket.sent.some((line) => line.includes("env512")));
  assert.ok(writes.every((bytes) => !bytes.toString().includes("private-")));
});

test("unconfirmed termination blocks new commands until the same process is confirmed stopped", async () => {
  let stopped = false;
  let launches = 0;
  const controller = new AbortController();
  const operations = new SandboxTerminalOperations(ENDPOINT, {
    writeFile: async () => {},
    stopTimeoutMs: 100,
    connect: () => {
      const socket = new FakeSocket();
      socket.onSend = (line) => {
        const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)'/.exec(line);
        if (ack) {
          if (ack[2] === "launch") launches++;
          socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${ack[2] === "stop" ? (stopped ? "yes" : "unknown") : "ok"}\n`);
          return;
        }
        const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
        if (token) {
          socket.emit(frames(token).begin);
          if (launches === 1) queueMicrotask(() => controller.abort());
          else socket.emit(frames(token).end(0));
        }
      };
      queueMicrotask(() => socket.open());
      return socket;
    },
  });
  await assert.rejects(operations.exec("sleep 30", "", {signal: controller.signal, onData: () => {}}),
    (error: unknown) => error instanceof SandboxTerminalExecutionError && !error.execution.stopped);
  await assert.rejects(operations.exec("touch /must-not-start", "", {onData: () => {}}), /terminal blocked/);
  assert.equal(launches, 1);
  stopped = true;
  assert.equal((await operations.exec("true", "", {onData: () => {}})).exitCode, 0);
  assert.equal(launches, 2);
});

test("lost launch acknowledgement does not dispatch a second launch", async () => {
  const socket = new FakeSocket();
  socket.onSend = (line) => {
    const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)'/.exec(line);
    if (!ack || ack[2] === "launch") return;
    socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${ack[2] === "stop" ? "yes" : "ok"}\n`);
  };
  const operations = new SandboxTerminalOperations(ENDPOINT, {
    writeFile: async () => {},
    connect: () => {queueMicrotask(() => socket.open()); return socket;},
    startupTimeoutMs: 30, stopTimeoutMs: 100,
  });
  await assert.rejects(operations.exec("true", "", {onData: () => {}}),
    (error: unknown) => error instanceof SandboxTerminalExecutionError &&
      error.execution.phase === "starting" && error.execution.stopped);
  assert.equal(socket.sent.filter((line) => line.includes("launch.sh")).length, 1);
});

test("command environment redacts secrets before persistence", () => {
  const root = mkdtempSync(join(tmpdir(), "terminal-env-"));
  try {
    const runDir = join(root, "run");
    mkdirSync(runDir);
    const environment = { DF_API_KEY: "very-private-key", ORDINARY: "space ' quote $value" };
    const script = buildCommandScript(runDir, root, "printf '%s\\n' \"$DF_API_KEY\" \"$ORDINARY\"", true);
    writeFileSync(join(runDir, "cmd.sh"), script);
    const result = spawnSync("sh", [join(runDir, "cmd.sh")], {encoding:"utf8", env: {
      ...process.env, PYROMIND_TERMINAL_ENV: Buffer.from(JSON.stringify(environment)).toString("base64"),
    }});
    assert.equal(result.stdout, "[REDACTED]\nspace ' quote $value\n");
    assert.equal(readFileSync(join(runDir, "rc"), "utf8"), "0");
    assert.ok(!script.includes("very-private-key"));
    assert.notEqual(runCommandScript("cd /definitely/absent").exitCode, "0");
  } finally { rmSync(root, {recursive:true, force:true}); }
});
