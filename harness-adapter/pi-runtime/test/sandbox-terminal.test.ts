import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  buildCommandScript,
  buildLaunchBatchLine,
  buildLaunchScript,
  buildReadyLine,
  buildResumeLine,
  buildCancelLine,
  buildStageBatches,
  ENVIRONMENT_RUNNER,
  SandboxTerminalExecutionError,
  buildStartLine,
  buildWatchScript,
  SandboxTerminalOperations,
  LazySandboxTerminalOperations,
  SANDBOX_ENVIRONMENT_FILE,
  SANDBOX_SCRIPT_CHUNK_CHARS,
  sandboxTerminalFastPath,
  sandboxTerminalScriptChunkChars,
  sandboxTerminalTrace,
  sandboxTerminalUrl,
  sandboxTerminalTimeouts,
  stagedEnvironment,
  TERMINAL_BATCH_BUDGET_BYTES,
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

test("operator timeouts come from the environment when configured", () => {
  assert.deepEqual(sandboxTerminalTimeouts({}), {
    resolveTimeoutMs: undefined, startupTimeoutMs: undefined,
    queueTimeoutMs: undefined,
  });
  assert.deepEqual(sandboxTerminalTimeouts({
    PYROMIND_PI_TERMINAL_RESOLVE_TIMEOUT_MS: "60000",
    PYROMIND_PI_TERMINAL_STARTUP_TIMEOUT_MS: " 90000 ",
  }), { resolveTimeoutMs: 60_000, startupTimeoutMs: 90_000 });
  assert.throws(() => sandboxTerminalTimeouts({ PYROMIND_PI_TERMINAL_STARTUP_TIMEOUT_MS: "soon" }),
    /must be a positive number of milliseconds/);
});

test("the staging chunk comes from the environment when configured", () => {
  assert.equal(sandboxTerminalScriptChunkChars({}), undefined);
  assert.equal(sandboxTerminalScriptChunkChars({
    PYROMIND_PI_TERMINAL_SCRIPT_CHUNK_CHARS: " 1024 ",
  }), 1024);
  assert.throws(() => sandboxTerminalScriptChunkChars({
    PYROMIND_PI_TERMINAL_SCRIPT_CHUNK_CHARS: "many",
  }), /must be a positive number of characters/);
});

test("endpoint resolution fails on its own deadline, not the startup budget", async () => {
  const terminal = new LazySandboxTerminalOperations(
    new SandboxEndpointSession(() => new Promise(() => {})),
    { resolveTimeoutMs: 20 },
  );
  await assert.rejects(terminal.exec("true", ".", { onData: () => {} }),
    (error: unknown) => error instanceof SandboxTerminalExecutionError &&
      error.execution.phase === "connecting" &&
      error.execution.error_code === "startup_timeout" &&
      error.execution.started === false && error.execution.stopped === true);
});

test("a slow endpoint resolution does not consume the startup budget", async () => {
  const endpoints = new SandboxEndpointSession(async () => {
    await new Promise((resolve) => setTimeout(resolve, 200));
    return ENDPOINT;
  });
  const { connect, sockets } = scriptedConnector([
    (socket, _line, token) => {
      socket.emit(frames(token).begin);
      socket.emit(frames(token).end(0));
    },
  ]);
  const operations = new LazySandboxTerminalOperations(endpoints, {
    connect, writeFile: async () => {}, startupTimeoutMs: 100, resolveTimeoutMs: 5_000,
  });
  const completed = await operations.exec("true", ".", { onData: () => {} });
  assert.equal(completed.exitCode, 0);
  assert.equal(sockets.length, 1);
});

test("a rejected endpoint resolution is retried on the same budget", async () => {
  const requests: { refresh: boolean }[] = [];
  let attempts = 0;
  const endpoints = new SandboxEndpointSession(async (request) => {
    requests.push(request);
    attempts += 1;
    if (attempts === 1) throw new Error("request_failed: POST /api/v1/sandboxes -> 503");
    return ENDPOINT;
  });
  const { connect, sockets } = scriptedConnector([
    (socket, _line, token) => {
      socket.emit(frames(token).begin);
      socket.emit(frames(token).end(0));
    },
  ]);
  const terminal = new LazySandboxTerminalOperations(endpoints, {
    connect, writeFile: async () => {}, resolveRetryBaseMs: 1,
  });
  const completed = await terminal.exec("true", ".", { onData: () => {} });
  assert.equal(completed.exitCode, 0);
  assert.equal(attempts, 2);
  assert.deepEqual(requests, [{ refresh: false }, { refresh: true }]);
  assert.equal(sockets.length, 1);
});

test("a persistent endpoint resolution failure reports the control-plane error", async () => {
  const cause = new Error("request_failed: POST /api/v1/sandboxes -> 503 too many 503 error responses");
  const endpoints = new SandboxEndpointSession(async () => {
    throw cause;
  });
  const terminal = new LazySandboxTerminalOperations(endpoints, {
    resolveTimeoutMs: 60, resolveRetryBaseMs: 1,
  });
  await assert.rejects(terminal.exec("true", ".", { onData: () => {} }),
    (error: unknown) => error instanceof SandboxTerminalExecutionError &&
      error.execution.phase === "connecting" &&
      error.execution.error_code === "connection_lost" &&
      error.execution.started === false && error.execution.stopped === true &&
      error.message.includes("too many 503 error responses") &&
      error.message.includes("resolve budget exhausted") &&
      error.cause === cause);
});

function countingEndpoints(): { endpoints: SandboxEndpointSession; resolved: () => number } {
  let resolved = 0;
  const endpoints = new SandboxEndpointSession(async () => {
    resolved += 1;
    return { ...ENDPOINT, sandboxId: `sbx-${resolved}` };
  });
  return { endpoints, resolved: () => resolved };
}

test("a handshake which never completes replays the command on a replacement endpoint", async () => {
  const { endpoints, resolved } = countingEndpoints();
  const sockets: FakeSocket[] = [];
  const connect: SandboxTerminalConnector = () => {
    const generation = resolved();
    const socket = new FakeSocket();
    sockets.push(socket);
    socket.onSend = (line) => {
      const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)' '([^']+)'/.exec(line);
      if (ack) {
        socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${ack[3]}\n`);
        return;
      }
      const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
      assert.ok(token, "watch line carries the run token");
      socket.emit(frames(token).begin);
      socket.emit(frames(token).end(0));
    };
    queueMicrotask(() => {
      socket.open();
      if (generation === 1) socket.close();
    });
    return socket;
  };
  const terminal = new LazySandboxTerminalOperations(endpoints, { connect, writeFile: async () => {} });
  const completed = await terminal.exec("true", ".", { onData: () => {} });
  assert.equal(completed.exitCode, 0);
  assert.equal(resolved(), 2);
  assert.equal(sockets.flatMap(launchInvocations).length, 1);
});

test("a command which started is reported instead of replayed on a replacement endpoint", async () => {
  const { endpoints, resolved } = countingEndpoints();
  const { connect } = scriptedConnector([
    (socket, _line, token) => {
      socket.emit(frames(token).begin);
      socket.close();
    },
  ]);
  const terminal = new LazySandboxTerminalOperations(endpoints, { connect, writeFile: async () => {} });
  await assert.rejects(terminal.exec("true", ".", { onData: () => {} }),
    (error: unknown) => error instanceof SandboxTerminalExecutionError &&
      error.execution.error_code === "connection_lost" && error.execution.started &&
      error.execution.stopped);
  assert.equal(resolved(), 1);
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
  // A sandbox keeps the session harness once it has been staged, so only a
  // handshake which has not seen one reports it missing.
  let harnessOnDisk = false;
  let cachedEnvironment: string | undefined;
  let pendingEnvironment: string | undefined;
  const connect: SandboxTerminalConnector = () => {
    const episode = episodes[Math.min(index, episodes.length - 1)]!;
    index += 1;
    const socket = new FakeSocket();
    sockets.push(socket);
    socket.onSend = (line) => {
      if (line === "\u0003") return;
      const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)' '([^']+)'/.exec(line);
      if (ack) {
        if (ack[2] === "staged" && !line.includes(SANDBOX_ENVIRONMENT_FILE)) harnessOnDisk = true;
        if (ack[2] === "staged" && line.includes(SANDBOX_ENVIRONMENT_FILE)) cachedEnvironment = pendingEnvironment;
        const signature = /" = "([0-9a-f]{16})"/.exec(line)?.[1];
        if (ack[2] === "ready" && signature !== undefined) pendingEnvironment = signature;
        const value = ack[2] === "stop" ? "yes"
          : ack[2] === "ready" && !harnessOnDisk ? "missing"
          : ack[2] === "ready" && signature !== undefined && signature !== cachedEnvironment ? "env"
          : ack[3];
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

/**
 * The harness presence check in the handshake names the same scripts, so a
 * launch count has to look for the invocation rather than the file name.
 */
function launchInvocations(socket: FakeSocket): string[] {
  return socket.sent.filter((line) => line.includes("launch.sh' '"));
}

function run(
  episodes: Episode[],
  options: {
    command?: string;
    signal?: AbortSignal;
    timeout?: number;
    startupTimeoutMs?: number;
  } = {},
) {
  const { connect, sockets } = scriptedConnector(episodes);
  const { command = "ls -la", startupTimeoutMs, ...execOptions } = options;
  const operations = new SandboxTerminalOperations(ENDPOINT, {
    connect, writeFile: async () => {}, startupTimeoutMs,
  });
  const chunks: Buffer[] = [];
  const result = operations.exec(command, "", {
    onData: (data) => chunks.push(data),
    ...execOptions,
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

test("launch uses the session harness, its cached environment, and an atomic guard", () => {
  const line = buildStartLine("/workspace/harness", "/workspace/run", "token", true);
  assert.match(buildLaunchScript(), /mkdir "\$run\/launched"/);
  assert.match(buildLaunchScript(), /setsid sh "\$run\/cmd\.sh"/);
  assert.match(line, /harness\/launch\.sh' '\/workspace\/run'/);
  // The environment travels in the session file, never in the launch line.
  // The file already holds base64, which is the form env_runner.py decodes.
  assert.match(line, /PYROMIND_TERMINAL_ENV="\$\(cat '\/workspace\/harness\/env\.b64'\)"/);
  const burst = buildLaunchBatchLine("/workspace/harness", "/workspace/run", "token",
    {"cmd.sh": "true\n"}, true);
  assert.match(burst, /PYROMIND_TERMINAL_ENV="\$\(cat '\/workspace\/harness\/env\.b64'\)"/);
  assert.ok(!line.includes("private-"));
  assert.ok(!line.includes("watch.sh"));
  assert.ok(Buffer.byteLength(line) < 1024);
  assert.ok(!buildResumeLine("/workspace/run", "token").includes("setsid"));
  assert.ok(!buildCancelLine("/workspace/run", "token").includes("rm -f"));
  assert.ok(TERMINAL_SCRIPT_CHUNK_CHARS < 1024);
  assert.ok(TERMINAL_BATCH_BUDGET_BYTES < 4096);
});

function finishEpisode(): Episode {
  return (socket, _line, token) => {
    socket.emit(frames(token).begin);
    socket.emit(frames(token).end(0));
  };
}

/**
 * A command long enough that its script no longer fits the TTY staging path.
 *
 * The burst budget groups the staged lines, so the payload has to need more
 * bursts than `TERMINAL_STAGING_MAX_BATCHES` before the file API is worth it.
 */
const LARGE_COMMAND = `printf '%s' '${"z".repeat(20_000)}'`;

test("scripts stage over the terminal in bounded bursts when no file API is available", async () => {
  const { connect, sockets } = scriptedConnector([finishEpisode()]);
  const operations = new SandboxTerminalOperations(ENDPOINT, { connect });
  const completed = await operations.exec("printf 'staged\\n'", "", { onData: () => {} });
  assert.equal(completed.exitCode, 0);
  assert.equal(completed.execution.staging, "batch");
  const staging = sockets[0]!.sent.filter((line) => line.includes(".b64"));
  assert.ok(staging.some((line) => line.includes("cmd.sh.b64")));
  assert.ok(staging.some((line) => line.includes("watch.sh.b64")));
  assert.ok(staging.some((line) => line.includes("launch.sh.b64")));
  // A burst is one send, and it has to leave room in the PTY input queue.
  assert.ok(sockets[0]!.sent.every((burst) => Buffer.byteLength(burst) <= TERMINAL_BATCH_BUDGET_BYTES));
  // Every staged line has to stay below the PTY canonical input cap.
  const lines = sockets[0]!.sent.flatMap((payload) => payload.split("\n"));
  assert.ok(lines.every((line) => Buffer.byteLength(line) < 1024));
});

test("a session which already has the harness stages only the command script", async () => {
  const { connect, sockets } = scriptedConnector([finishEpisode(), finishEpisode()]);
  const operations = new SandboxTerminalOperations(ENDPOINT, { connect });
  await operations.exec("true", "", { onData: () => {} });
  const afterFirst = sockets[0]!.sent.length;
  await operations.exec("true", "", { onData: () => {} });
  // The session keeps one socket for every command it runs.
  assert.equal(sockets.length, 1);
  const second = sockets[0]!.sent.slice(afterFirst).filter((line) => line.includes(".b64"));
  assert.ok(second.some((line) => line.includes("cmd.sh.b64")));
  assert.ok(!second.some((line) => line.includes("watch.sh.b64")));
  assert.equal(launchInvocations(sockets[0]!).length, 2);
});

test("a warm socket which stops answering is replaced before the command runs", async () => {
  const sockets: FakeSocket[] = [];
  let silent = false;
  const connect: SandboxTerminalConnector = () => {
    const socket = new FakeSocket();
    sockets.push(socket);
    socket.onSend = (line) => {
      // A bridge can drop a session without a close frame, which leaves the
      // socket looking alive until an exchange goes unanswered.
      if (silent && sockets.length === 1) return;
      const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)' '([^']+)'/.exec(line);
      if (ack) {
        socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${ack[3]}\n`);
        return;
      }
      const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
      assert.ok(token, "watch line carries the run token");
      socket.emit(frames(token).begin);
      socket.emit(frames(token).end(0));
    };
    queueMicrotask(() => socket.open());
    return socket;
  };
  const operations = new SandboxTerminalOperations(ENDPOINT, {
    connect, probeTimeoutMs: 50, startupTimeoutMs: 1_000,
  });
  const first = await operations.exec("true", "", { onData: () => {} });
  assert.equal(first.exitCode, 0);
  silent = true;
  const recovered = await operations.exec("true", "", { onData: () => {} });
  assert.equal(recovered.exitCode, 0);
  // The unanswered handshake is what proves the socket is gone: the command is
  // replayed on a replacement socket instead of failing on the dead one.
  assert.equal(sockets.length, 2);
  assert.ok((recovered.execution.round_trips ?? 0) >= 3);
});

test("a failed script upload falls back to terminal staging", async () => {
  const { connect, sockets } = scriptedConnector([finishEpisode()]);
  let uploads = 0;
  const operations = new SandboxTerminalOperations(ENDPOINT, {
    connect,
    writeFile: async () => {
      uploads += 1;
      throw new Error("sandbox file request failed");
    },
  });
  const completed = await operations.exec(LARGE_COMMAND, "", { onData: () => {} });
  assert.equal(completed.exitCode, 0);
  assert.equal(completed.execution.staging, "batch-fallback");
  assert.equal(uploads, 1);
  assert.ok(sockets[0]!.sent.some((line) => line.includes("base64 -d")));
});

test("a script upload slower than the startup budget falls back to terminal staging", async () => {
  const { connect, sockets } = scriptedConnector([finishEpisode()]);
  let uploadSignal: AbortSignal | undefined;
  const operations = new SandboxTerminalOperations(ENDPOINT, {
    connect,
    startupTimeoutMs: 20,
    writeFile: (_path, _content, signal) => {
      uploadSignal = signal;
      return new Promise(() => {});
    },
  });
  const completed = await operations.exec(LARGE_COMMAND, "", { onData: () => {} });
  assert.equal(completed.exitCode, 0);
  assert.equal(completed.execution.staging, "batch-fallback");
  // The abandoned upload must not keep running and overwrite the staged scripts.
  assert.equal(uploadSignal?.aborted, true);
  assert.ok(sockets[0]!.sent.some((line) => line.includes("base64 -d")));
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
  assert.equal(completed.execution.staging, "batch");
  // The first command pays for the session harness; later commands do not.
  assert.ok((completed.execution.round_trips ?? 0) > 2);
  assert.ok(completed.execution.timings_ms?.total !== undefined);
  assert.equal(output(), "partial output\n");
});

test("a command too large for the terminal stages its script through the file API", async () => {
  const { result, sockets } = run([
    (socket, _line, token) => {
      socket.emit(frames(token).begin);
      socket.emit(frames(token).end(0));
    },
  ], { command: LARGE_COMMAND });
  const completed = await result;
  assert.equal(completed.exitCode, 0);
  assert.equal(completed.execution.staging, "file");
  assert.ok(!sockets[0]!.sent.some((line) => line.includes("cmd.sh.b64")));
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
  assert.ok(sockets[1]!.sent.some((line) => line.includes("watch.sh' '")));
  assert.equal(sockets.flatMap(launchInvocations).length, 1);
});

test("terminal operations replay a handshake which drops before the launch", async () => {
  const sockets: FakeSocket[] = [];
  const connect: SandboxTerminalConnector = () => {
    const socket = new FakeSocket();
    const first = sockets.length === 0;
    sockets.push(socket);
    socket.onSend = (line) => {
      const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)' '([^']+)'/.exec(line);
      if (ack) {
        socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${ack[3]}\n`);
        return;
      }
      const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
      assert.ok(token, "watch line carries the run token");
      socket.emit(frames(token).begin);
      socket.emit(frames(token).end(0));
    };
    queueMicrotask(() => {
      socket.open();
      if (first) socket.close();
    });
    return socket;
  };
  const operations = new SandboxTerminalOperations(ENDPOINT, { connect, writeFile: async () => {} });
  const completed = await operations.exec("true", "", { onData: () => {} });
  assert.equal(completed.exitCode, 0);
  assert.equal(sockets.length, 2);
  assert.equal(sockets.flatMap(launchInvocations).length, 1);
});

test("a launch acknowledgement lost to a dropped socket is observed, not relaunched", async () => {
  const sockets: FakeSocket[] = [];
  const connect: SandboxTerminalConnector = () => {
    const socket = new FakeSocket();
    const first = sockets.length === 0;
    sockets.push(socket);
    socket.onSend = (line) => {
      const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)' '([^']+)'/.exec(line);
      if (ack) {
        if (first && ack[2] === "launch") {
          socket.close();
          return;
        }
        socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${ack[3]}\n`);
        return;
      }
      const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
      assert.ok(token, "watch line carries the run token");
      socket.emit(frames(token).begin);
      socket.emit(frames(token).end(0));
    };
    queueMicrotask(() => socket.open());
    return socket;
  };
  const operations = new SandboxTerminalOperations(ENDPOINT, { connect, writeFile: async () => {} });
  const completed = await operations.exec("true", "", { onData: () => {} });
  assert.equal(completed.exitCode, 0);
  assert.equal(sockets.length, 2);
  assert.equal(sockets.flatMap(launchInvocations).length, 1);
});

test("a dropped connection is still observed after the startup budget has elapsed", async () => {
  const { result, sockets } = run([
    (socket, _line, token) => {
      socket.emit(frames(token).begin);
      setTimeout(() => socket.close(), 120);
    },
    (socket, _line, token) => {
      socket.emit(frames(token).begin);
      socket.emit(frames(token).end(0));
    },
  ], { timeout: 10, startupTimeoutMs: 50 });
  assert.equal((await result).exitCode, 0);
  assert.equal(sockets.length, 2);
});

test("a command timeout does not cap the cold-start preparation", async () => {
  const socket = new FakeSocket();
  socket.onSend = (line) => {
    if (line === "\u0003") return;
    const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)' '([^']+)'/.exec(line);
    if (ack) {
      const value = ack[2] === "stop" ? "yes" : ack[2] === "ready" ? "missing" : ack[3];
      // Staging the harness is what a cold sandbox pays before any command runs.
      const delay = /^staged/.test(ack[2]) ? 80 : 0;
      setTimeout(() => socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${value}\n`), delay);
      return;
    }
    const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
    assert.ok(token, "watch line carries the run token");
    socket.emit(frames(token).begin);
    socket.emit(frames(token).end(0));
  };
  const operations = new SandboxTerminalOperations(ENDPOINT, {
    connect: () => {queueMicrotask(() => socket.open()); return socket;},
    writeFile: async () => {},
    startupTimeoutMs: 5_000,
  });
  const completed = await operations.exec("true", "", { timeout: 0.05, onData: () => {} });
  assert.equal(completed.exitCode, 0);
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

test("an environment burst which is never acknowledged fails before the launch", async () => {
  const socket = new FakeSocket();
  let harnessOnDisk = false;
  socket.onSend = (line) => {
    if (line.includes(`${SANDBOX_ENVIRONMENT_FILE}.b64`)) return;
    const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)'/.exec(line);
    if (!ack) return;
    if (ack[2] === "staged") harnessOnDisk = true;
    const value = ack[2] === "stop" ? "yes"
      : ack[2] === "ready" && !harnessOnDisk ? "missing" : "ok";
    socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${value}\n`);
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
    error.execution.phase === "environment" && error.execution.stopped &&
    error.execution.started === false);
  // The burst carries base64, and nothing launches without an environment.
  assert.ok(!socket.sent.some((line) => line.includes("setsid")));
  assert.ok(!socket.sent.some((line) => line.includes("launch.sh' '")));
  assert.ok(!socket.sent.some((line) => line.includes("private-")));
  assert.equal(socket.sent.filter((line) => line.includes(`${SANDBOX_ENVIRONMENT_FILE}.b64`)).length, 1);
  assert.equal(writes.length, 0);
});

test("the session environment is staged once and reused by later commands", async () => {
  const { connect, sockets } = scriptedConnector([finishEpisode(), finishEpisode()]);
  const operations = new SandboxTerminalOperations(ENDPOINT, { connect });
  const environment = { PYROMIND_DATAFLOW_PROFILES: JSON.stringify({ text: { model: "m" } }) };
  await operations.exec("true", "", { env: environment, onData: () => {} });
  const afterFirst = sockets[0]!.sent.length;
  await operations.exec("true", "", { env: environment, onData: () => {} });
  assert.ok(sockets[0]!.sent.some((line) => line.includes(`${SANDBOX_ENVIRONMENT_FILE}.b64`)));
  assert.ok(!sockets[0]!.sent.slice(afterFirst)
    .some((line) => line.includes(`${SANDBOX_ENVIRONMENT_FILE}.b64`)));
  // The reused session still hands the cached environment to the launch.
  assert.ok(sockets[0]!.sent.slice(afterFirst).some((line) => line.includes(`cat`) &&
    line.includes(`/${SANDBOX_ENVIRONMENT_FILE}'`)));
  // The warm session pays the handshake and the combined burst only.
  const second = await operations.exec("true", "", { env: environment, onData: () => {} });
  assert.equal(second.execution.round_trips, 2);
});

test("a changed environment is restaged before the next launch", async () => {
  const { connect, sockets } = scriptedConnector([finishEpisode(), finishEpisode()]);
  const operations = new SandboxTerminalOperations(ENDPOINT, { connect });
  await operations.exec("true", "", { env: { DF_API_KEY: "first" }, onData: () => {} });
  const afterFirst = sockets[0]!.sent.length;
  await operations.exec("true", "", { env: { DF_API_KEY: "second" }, onData: () => {} });
  const second = sockets[0]!.sent.slice(afterFirst);
  assert.ok(second.some((line) => line.includes(`${SANDBOX_ENVIRONMENT_FILE}.b64`)));
  assert.ok(!second.some((line) => line.includes("first")));
});

test("staging bursts stay inside the PTY budget and check decoded bytes", () => {
  const payload = "x".repeat(4000);
  const batches = buildStageBatches("/workspace/run", "token", { "cmd.sh": payload });
  assert.ok(batches.length > 1);
  assert.ok(batches.every((batch) => Buffer.byteLength(batch.command) <= TERMINAL_BATCH_BUDGET_BYTES));
  assert.ok(batches.every((batch) => batch.command.split("\n")
    .every((line) => Buffer.byteLength(line) < 1024)));
  assert.equal(batches.at(-1)!.stage, "staged");
  assert.ok(batches.at(-1)!.command.includes(`= "${Buffer.byteLength(payload)}"`));
  assert.equal(buildStageBatches("/workspace/run", "token", { "cmd.sh": "true\n" }).length, 1);
});

test("a sandbox-sized chunk stages the same payload in fewer bursts", () => {
  // A write tool call stages the base64 of a script which itself carries the
  // file's base64, so a realistic staged payload is this size.
  const payload = "x".repeat(16_000);
  const files = { "cmd.sh": payload };
  const runDir = "/target-workspace/.pyromind-agent/conv-1/public_data/data-preparation/" +
    ".pyromind-agent-runs/0123456789abcdef0123456789abcdef";
  const small = buildStageBatches(runDir, "token", files);
  const sandbox = buildStageBatches(runDir, "token", files, TERMINAL_BATCH_BUDGET_BYTES, false,
    SANDBOX_SCRIPT_CHUNK_CHARS);
  assert.ok(sandbox.length < small.length, `${sandbox.length} bursts beat ${small.length}`);
  assert.ok(sandbox.every((batch) => Buffer.byteLength(batch.command) <= TERMINAL_BATCH_BUDGET_BYTES));
  assert.ok(sandbox.every((batch) => batch.command.split("\n")
    .every((line) => Buffer.byteLength(line) < 4096)), "the sandbox PTY caps one line at 4096");
  assert.equal(sandbox.at(-1)!.stage, "staged");
  assert.ok(sandbox.at(-1)!.command.includes(`= "${Buffer.byteLength(payload)}"`));
});

test("a refused staging burst is repaired one acknowledged line at a time", async () => {
  const socket = new FakeSocket();
  let refused = false;
  socket.onSend = (line) => {
    const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)' '([^']+)'/.exec(line);
    if (!ack) {
      const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
      assert.ok(token, "watch line carries the run token");
      socket.emit(frames(token).begin);
      socket.emit(frames(token).end(0));
      return;
    }
    if (ack[2] === "launch" && !refused) {
      refused = true;
      socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:missing\n`);
      return;
    }
    socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${ack[2] === "stop" ? "yes" : "ok"}\n`);
  };
  const operations = new SandboxTerminalOperations(ENDPOINT, {
    connect: () => {queueMicrotask(() => socket.open()); return socket;},
    writeFile: async () => {},
  });
  const completed = await operations.exec("true", "", { onData: () => {} });
  assert.equal(completed.exitCode, 0);
  assert.equal(completed.execution.staging, "batch-fallback");
  // The refused burst is followed by the acknowledged staging and one launch.
  assert.equal(socket.sent.filter((line) =>
    line.includes("cmd.sh.b64") && line.includes("launch.sh' '")).length, 1);
  assert.ok(socket.sent.some((line) =>
    line.includes("cmd.sh.b64") && line.trimEnd().split("\n").length === 1));
  assert.equal(launchInvocations(socket).length, 2);
});

test("the batched staging path can be switched off for a rollout", async () => {
  assert.equal(sandboxTerminalFastPath({}), true);
  assert.equal(sandboxTerminalFastPath({ PYROMIND_PI_TERMINAL_FAST_PATH: "0" }), false);
  const { connect, sockets } = scriptedConnector([finishEpisode()]);
  const operations = new SandboxTerminalOperations(ENDPOINT, { connect, fastPath: false });
  const completed = await operations.exec("true", "", { onData: () => {} });
  assert.equal(completed.execution.staging, "terminal");
  // One acknowledgement per staging line, and none of them a burst.
  assert.ok(sockets[0]!.sent.filter((line) => line.includes(".b64")).length > 5);
  assert.ok(sockets[0]!.sent.every((line) => Buffer.byteLength(line) < 1024));
});

test("the latency trace reports every exchange only when switched on", async () => {
  assert.equal(sandboxTerminalTrace({}), false);
  assert.equal(sandboxTerminalTrace({ PYROMIND_PI_TERMINAL_TRACE: "1" }), true);
  const { connect } = scriptedConnector([finishEpisode()]);
  const written: string[] = [];
  const write = process.stderr.write;
  process.stderr.write = ((chunk: string | Uint8Array) => {
    written.push(chunk.toString());
    return true;
  }) as typeof process.stderr.write;
  try {
    const operations = new SandboxTerminalOperations(ENDPOINT, { connect, trace: true });
    await operations.exec("true", "", { onData: () => {} });
  } finally {
    process.stderr.write = write;
  }
  const events = written.flatMap((chunk) => chunk.trim().split("\n"))
    .map((line) => JSON.parse(line) as {
      event: string; stage?: string; round_trip?: number; round_trips?: number;
      timings_ms?: Record<string, number>;
    });
  assert.equal(events.at(-1)!.event, "terminal.execution");
  assert.ok(events.at(-1)!.round_trips! > 1);
  assert.equal(typeof events.at(-1)!.timings_ms!.total, "number");
  const exchanges = events.slice(0, -1);
  assert.ok(exchanges.length > 1);
  assert.ok(exchanges.every((event) =>
    event.event === "terminal.exchange" &&
    typeof event.stage === "string" &&
    typeof event.round_trip === "number"));
  assert.equal(exchanges[0]!.stage, "ready");
  assert.equal(exchanges.at(-1)!.stage, "launch");
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

test("an unconfirmed termination retires the bundle and replays the command", async () => {
  const sockets: FakeSocket[] = [];
  let launches = 0;
  let resolutions = 0;
  const controller = new AbortController();
  const endpoints = new SandboxEndpointSession(async () => {
    resolutions += 1;
    return ENDPOINT;
  });
  const connect: SandboxTerminalConnector = () => {
    const socket = new FakeSocket();
    sockets.push(socket);
    socket.onSend = (line) => {
      if (line === "\u0003") return;
      const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)' '([^']+)'/.exec(line);
      if (ack) {
        // The bridge answers every exchange but never confirms the stop.
        const value = ack[2] === "stop" ? "unknown" : ack[2] === "ready" ? "missing" : ack[3];
        if (ack[2] === "launch") launches += 1;
        socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:${value}\n`);
        return;
      }
      const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
      assert.ok(token, "watch line carries the run token");
      socket.emit(frames(token).begin);
      if (launches === 1) queueMicrotask(() => controller.abort());
      else socket.emit(frames(token).end(0));
    };
    queueMicrotask(() => socket.open());
    return socket;
  };
  const terminal = new LazySandboxTerminalOperations(endpoints, {
    connect, writeFile: async () => {}, stopTimeoutMs: 100,
  });
  await assert.rejects(
    terminal.exec("sleep 30", ".", {signal: controller.signal, onData: () => {}}),
    (error: unknown) => error instanceof SandboxTerminalExecutionError && !error.execution.stopped);
  const completed = await terminal.exec("touch /must-not-start", ".", {onData: () => {}});
  assert.equal(completed.exitCode, 0);
  // A blocked session is not reusable, so the command is replayed on a bundle
  // resolved after the one which could not confirm the stop was retired.
  assert.equal(resolutions, 2);
  assert.equal(sockets.length, 3);
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
  assert.equal(launchInvocations(socket).length, 1);
});

test("command environment redacts secrets before persistence", () => {
  const root = mkdtempSync(join(tmpdir(), "terminal-env-"));
  try {
    const runDir = join(root, "run");
    mkdirSync(runDir);
    const environment = { DF_API_KEY: "very-private-key", ORDINARY: "space ' quote $value" };
    const runner = join(runDir, "env_runner.py");
    writeFileSync(runner, ENVIRONMENT_RUNNER);
    const script = buildCommandScript(
      runDir, root, "printf '%s\\n' \"$DF_API_KEY\" \"$ORDINARY\"", runner);
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
