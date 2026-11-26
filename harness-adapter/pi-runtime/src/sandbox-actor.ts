import { randomUUID } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";

import { isRecord } from "./protocol.js";
import type { ActorEndpoint } from "./sandbox-operations.js";
import {
  SandboxTerminalExecutionError,
  type TerminalExecOptions,
  type TerminalExecResult,
  type TerminalExecutionEvidence,
} from "./terminal-execution.js";

const PAGE_BYTES = 64 * 1024;
const MAX_EXEC_REQUEST_BYTES = 64 * 1024;

function actorUrl(endpoint: ActorEndpoint, path: string): URL {
  return new URL(endpoint.baseUrl.replace(/\/+$/, "") + path);
}

export function actorFileUrl(endpoint: ActorEndpoint, path: string): URL {
  const url = actorUrl(endpoint, "/files");
  url.searchParams.set("path", path);
  return url;
}

export class ActorRequestError extends Error {
  constructor(method: string, readonly status: number) {
    super(`Actor ${method} request failed (HTTP ${status})`);
  }
}

export async function actorRequest(endpoint: ActorEndpoint, url: URL, init: RequestInit): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("ate-target-actor", endpoint.actorTarget);
  const timeout = AbortSignal.timeout(120_000);
  const response = await fetch(url, {
    ...init, headers, redirect: "error",
    signal: init.signal ? AbortSignal.any([init.signal, timeout]) : timeout,
  });
  if (response.ok) return response;
  await response.body?.cancel();
  throw new ActorRequestError(init.method ?? "GET", response.status);
}

const statuses = new Set(["running", "succeeded", "failed", "timed_out", "canceled"]);

async function state(response: Response, id: string) {
  const data: unknown = await response.json();
  if (!isRecord(data) || data.run_id !== id || typeof data.status !== "string" ||
      !statuses.has(data.status) || typeof data.stdout !== "string" || typeof data.stderr !== "string" ||
      typeof data.stdout_offset !== "number" || !Number.isSafeInteger(data.stdout_offset) ||
      typeof data.stderr_offset !== "number" || !Number.isSafeInteger(data.stderr_offset)) {
    throw new Error("invalid Actor execution response");
  }
  return {
    status: data.status,
    exitCode: typeof data.exit_code === "number" && Number.isInteger(data.exit_code) ? data.exit_code : null,
    stdout: Buffer.from(data.stdout, "base64"),
    stderr: Buffer.from(data.stderr, "base64"),
    stdoutOffset: data.stdout_offset,
    stderrOffset: data.stderr_offset,
    truncated: data.stdout_truncated === true || data.stderr_truncated === true,
  };
}

function transient(error: unknown): boolean {
  return error instanceof TypeError || error instanceof ActorRequestError && error.status >= 500;
}

async function pollRun(endpoint: ActorEndpoint, url: URL, id: string, signal: AbortSignal) {
  const observe = async () => state(await actorRequest(endpoint, url, { signal }), id);
  try {
    return await observe();
  } catch (error) {
    if (signal.aborted || !transient(error)) throw error;
    await delay(100, undefined, { signal });
    return await observe();
  }
}

async function cancelRun(endpoint: ActorEndpoint, url: URL, id: string): Promise<boolean> {
  const signal = AbortSignal.timeout(5_000);
  try {
    let response = await actorRequest(endpoint, new URL(url.href + "/cancel"), { method: "POST", signal });
    while ((await state(response, id)).status === "running") {
      await delay(100, undefined, { signal });
      response = await actorRequest(endpoint, url, { signal });
    }
    return true;
  } catch {
    return false;
  }
}

export async function runActorCommand(
  endpoint: ActorEndpoint, argv: string[], cwd: string, options: TerminalExecOptions,
): Promise<TerminalExecResult> {
  options.signal?.throwIfAborted();
  const env = [];
  for (const [name, value] of Object.entries(options.env ?? {})) {
    if (value === undefined) continue;
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name) || value.includes("\0")) {
      throw new Error("invalid process environment");
    }
    env.push(`${name}=${value}`);
  }
  const timeout = Math.ceil(options.timeout && options.timeout > 0 ? options.timeout : 300);
  if (!Number.isFinite(timeout) || timeout > 3600) throw new Error("Actor command timeout must be at most 3600 seconds");
  const id = randomUUID();
  const body = JSON.stringify({
    run_id: id, command: env.length ? ["/usr/bin/env", ...env, ...argv] : argv, cwd, timeout_seconds: timeout,
  });
  if (Buffer.byteLength(body) > MAX_EXEC_REQUEST_BYTES) {
    throw new Error("Actor command exceeds the 64 KiB request limit. Write the script to a workspace file and execute it by path.");
  }
  const deadline = AbortSignal.timeout((timeout + 5) * 1000);
  const signal = options.signal ? AbortSignal.any([options.signal, deadline]) : deadline;
  const url = actorUrl(endpoint, "/exec/" + id);
  const execution: TerminalExecutionEvidence = {
    execution_id: id, remote_run_dir: "", phase: "starting", started: false, stopped: false,
  };
  try {
    let view;
    try {
      const response = await actorRequest(endpoint, actorUrl(endpoint, "/exec"), {
        method: "POST", headers: { "Content-Type": "application/json" }, body, signal,
      });
      view = await state(response, id);
    } catch (error) {
      if (signal.aborted || !transient(error)) throw error;
      // A lost launch response is recoverable by run ID without replaying the shell command.
      try { view = await pollRun(endpoint, url, id, signal); }
      catch { throw error; }
    }
    execution.started = true;
    execution.phase = "observing";
    let stdoutOffset = 0;
    let stderrOffset = 0;
    for (;;) {
      execution.stopped = view.status !== "running";
      if (view.truncated || view.stdoutOffset !== stdoutOffset + view.stdout.length ||
          view.stderrOffset !== stderrOffset + view.stderr.length) {
        throw new Error("Actor command output was lost. Redirect large output to a workspace file.");
      }
      if (view.stdout.length) options.onData(view.stdout);
      if (view.stderr.length) options.onData(view.stderr);
      const fullPage = view.stdout.length === PAGE_BYTES || view.stderr.length === PAGE_BYTES;
      if (execution.stopped && !fullPage) {
        execution.phase = "finished";
        if (view.status === "timed_out" || view.status === "canceled") {
          execution.error_code = view.status === "timed_out" ? "execution_timeout" : "aborted";
          throw new Error(view.status === "timed_out" ? `timeout:${timeout}` : "aborted");
        }
        if (view.exitCode === null) throw new Error("Actor command exited without an exit code");
        return { exitCode: view.exitCode, execution };
      }
      stdoutOffset = view.stdoutOffset;
      stderrOffset = view.stderrOffset;
      const poll = new URL(url);
      poll.searchParams.set("stdout_offset", String(stdoutOffset));
      poll.searchParams.set("stderr_offset", String(stderrOffset));
      if (!fullPage) await delay(100, undefined, { signal });
      view = await pollRun(endpoint, poll, id, signal);
    }
  } catch (error) {
    const rejected = !execution.started && error instanceof ActorRequestError && error.status < 500;
    if (!execution.stopped) execution.stopped = rejected || await cancelRun(endpoint, url, id);
    if (!rejected) execution.started = true; // An accepted launch may have lost its response.
    const message = options.signal?.aborted ? "aborted" : deadline.aborted ? `timeout:${timeout}`
      : error instanceof Error ? error.message : "Actor execution failed";
    execution.error_code ??= options.signal?.aborted ? "aborted" : deadline.aborted ? "execution_timeout" : "actor_execution_failed";
    throw new SandboxTerminalExecutionError(message, execution);
  }
}
