import assert from "node:assert/strict";
import test from "node:test";
import {
  buildCommandScript,
  buildStartLine,
  buildWatchScript,
  SandboxTerminalOperations,
  sandboxTerminalUrl,
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
