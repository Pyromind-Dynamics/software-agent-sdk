import assert from "node:assert/strict";
import test from "node:test";
import {
  LazySandboxTerminalOperations,
  SandboxTerminalExecutionError,
  type SandboxTerminalConnector,
  type SandboxTerminalSocket,
} from "../src/sandbox-terminal.js";
import { SandboxEndpointSession } from "../src/sandbox-operations.js";
import type { SandboxEndpoint } from "../src/sandbox-operations.js";

const ENDPOINT: SandboxEndpoint = {
  baseUrl: "https://portal.example.com",
  wsBaseUrl: "https://cluster.example.com",
  sandboxId: "sbx-1",
  apiKey: "secret",
  workspacePath: "/target-workspace/.pyromind-agent/conv-1",
  storagePath: "/target-workspace",
};

/** A bridge that accepts the socket and then never answers a handshake. */
class SilentSocket implements SandboxTerminalSocket {
  sent: string[] = [];

  onOpen(handler: () => void): void {
    queueMicrotask(handler);
  }

  onClose(): void {}

  onError(): void {}

  onMessage(): void {}

  send(command: string): void {
    this.sent.push(command);
  }

  close(): void {}
}

function silentConnector(): { connect: SandboxTerminalConnector; sockets: SilentSocket[] } {
  const sockets: SilentSocket[] = [];
  return {
    sockets,
    connect: () => {
      const socket = new SilentSocket();
      sockets.push(socket);
      return socket;
    },
  };
}

test("a queued terminal call gives up instead of waiting behind a stuck command", async () => {
  const { connect } = silentConnector();
  const terminal = new LazySandboxTerminalOperations(
    new SandboxEndpointSession(async () => ENDPOINT),
    { connect, queueTimeoutMs: 40, startupTimeoutMs: 60_000 },
  );
  const controller = new AbortController();
  // The first call owns the channel and never finishes: this is a workflow `run`
  // node or a shell command whose bridge stopped answering.
  const first = terminal.exec("sleep 3600", ".", {
    timeout: 3600,
    signal: controller.signal,
    onData: () => {},
  });
  await new Promise((resolve) => setTimeout(resolve, 10));

  const started = Date.now();
  await assert.rejects(
    terminal.exec("ls", ".", { timeout: 5, onData: () => {} }),
    (error: unknown) =>
      error instanceof SandboxTerminalExecutionError &&
      error.execution.error_code === "queue_timeout" &&
      error.execution.started === false &&
      error.execution.stopped === true,
  );
  // The wait is bounded by the queue budget, not by the command in front.
  assert.ok(Date.now() - started < 1_000, "queue wait must not follow the stuck command");

  controller.abort();
  await assert.rejects(first);
});
