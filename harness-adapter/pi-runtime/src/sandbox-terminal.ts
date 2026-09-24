import { randomBytes, randomUUID } from "node:crypto";
import { posix } from "node:path";
import type { BashOperations } from "@earendil-works/pi-coding-agent";
import type { SandboxEndpoint, SandboxEndpointSession } from "./sandbox-operations.js";

/** Raised when the TTY bridge cannot stay connected for a whole command. */
export class SandboxTerminalConnectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SandboxTerminalConnectionError";
  }
}

export const SANDBOX_RUNS_DIRNAME = ".pyromind-agent-runs";
export const TERMINAL_BEGIN_MARKER = "__PM_BEGIN__";
export const TERMINAL_HEARTBEAT_MARKER = "__PM_HB__";
export const TERMINAL_END_MARKER = "__PM_END__";

const TERMINAL_COLS = 160;
const TERMINAL_ROWS = 48;
const DEFAULT_HEARTBEAT_SECONDS = 60;
const DEFAULT_MAX_RECONNECTS = 2;
const CANCEL_GRACE_MS = 1_000;

export interface SandboxTerminalSocket {
  send(data: string): void;
  close(): void;
  onOpen(handler: () => void): void;
  onMessage(handler: (data: string | ArrayBuffer) => void): void;
  onClose(handler: () => void): void;
  onError(handler: (error: unknown) => void): void;
}

export type SandboxTerminalConnector = (url: string) => SandboxTerminalSocket;

export interface SandboxTerminalOptions {
  /** Directory holding per-call `out.log` / `pid` / `rc`, inside the workspace. */
  runsRoot?: string;
  heartbeatSeconds?: number;
  maxReconnects?: number;
  connect?: SandboxTerminalConnector;
}

type AttemptOutcome =
  | { kind: "exit"; exitCode: number | null }
  | { kind: "aborted" }
  | { kind: "timeout" }
  | { kind: "closed" };

/**
 * Splits the terminal byte stream into model-visible output and control lines.
 *
 * Everything before the begin marker (shell prompt, echoed command) is dropped;
 * heartbeat lines are swallowed so they only reset the platform idle timer; the
 * end marker carries the command's exit code. Markers can be split across
 * WebSocket frames, so a partial marker suffix is held back until resolved.
 */
export class TerminalMarkerScanner {
  private readonly prefixes: string[];
  private buffer: Buffer = Buffer.alloc(0);
  private begun = false;
  private finished = false;

  constructor(private readonly token: string) {
    this.prefixes = [
      `${TERMINAL_BEGIN_MARKER}${token}`,
      `${TERMINAL_HEARTBEAT_MARKER}${token}`,
      `${TERMINAL_END_MARKER}${token}:`,
    ];
  }

  get done(): boolean {
    return this.finished;
  }

  push(data: Buffer): { output: Buffer; exitCode?: number } {
    if (this.finished) return { output: Buffer.alloc(0) };
    this.buffer = Buffer.concat([this.buffer, data]);
    const output: Buffer[] = [];
    let exitCode: number | undefined;
    while (!this.finished) {
      const marker = this.nextMarker();
      if (marker === undefined) break;
      const before = this.buffer.subarray(0, marker.index);
      if (this.begun && before.byteLength > 0) output.push(before);
      this.buffer = this.buffer.subarray(marker.index + marker.length);
      if (marker.kind === "begin") {
        this.begun = true;
      } else if (marker.kind === "end") {
        exitCode = marker.exitCode;
        this.finished = true;
      }
    }
    if (this.begun && !this.finished) {
      const hold = partialMarkerSuffix(this.buffer, this.prefixes);
      const flushable = this.buffer.subarray(0, this.buffer.byteLength - hold);
      if (flushable.byteLength > 0) output.push(flushable);
      this.buffer = this.buffer.subarray(this.buffer.byteLength - hold);
    }
    return { output: Buffer.concat(output), exitCode };
  }

  private nextMarker():
    | { kind: "begin" | "heartbeat"; index: number; length: number }
    | { kind: "end"; index: number; length: number; exitCode: number }
    | undefined {
    let best: TerminalMarker | undefined;
    const kinds = ["begin", "heartbeat", "end"] as const;
    for (let index = 0; index < kinds.length; index += 1) {
      const kind = kinds[index]!;
      const prefix = this.prefixes[index]!;
      const at = this.buffer.indexOf(prefix);
      if (at < 0) continue;
      if (best !== undefined && at >= best.index) continue;
      if (kind === "end") {
        const digits = this.buffer
          .subarray(at + prefix.length)
          .toString("ascii")
          .match(/^-?\d+/);
        if (digits === null) continue;
        const markerEnd = at + prefix.length + digits[0].length;
        const lineEnd = markerLineEndingLength(this.buffer, markerEnd);
        if (lineEnd === undefined) continue;
        best = {
          kind,
          index: at,
          length: prefix.length + digits[0].length + lineEnd,
          exitCode: Number(digits[0]),
        };
        continue;
      }
      const lineEnd = markerLineEndingLength(this.buffer, at + prefix.length);
      if (lineEnd === undefined) continue;
      best = { kind, index: at, length: prefix.length + lineEnd };
    }
    return best;
  }
}

type TerminalMarker =
  | { kind: "begin" | "heartbeat"; index: number; length: number }
  | { kind: "end"; index: number; length: number; exitCode: number };

function partialMarkerSuffix(buffer: Buffer, prefixes: string[]): number {
  const max = Math.min(
    buffer.byteLength,
    prefixes.reduce((longest, prefix) => Math.max(longest, prefix.length), 0) + 16,
  );
  for (let length = max; length > 0; length -= 1) {
    const suffix = buffer.subarray(buffer.byteLength - length).toString("utf8");
    for (const prefix of prefixes) {
      if (prefix.startsWith(suffix)) return length;
      const markerStart = suffix.indexOf(prefix);
      if (markerStart < 0) continue;
      const tail = suffix.slice(markerStart + prefix.length);
      if (prefix.startsWith(TERMINAL_END_MARKER)) {
        if (/^-?\d*\r?$/.test(tail)) return length;
      } else if (tail === "" || tail === "\r") {
        return length;
      }
    }
  }
  return 0;
}

function markerLineEndingLength(buffer: Buffer, start: number): number | undefined {
  if (start >= buffer.byteLength) return undefined;
  if (buffer[start] === 0x0a) return 1;
  if (buffer[start] !== 0x0d) return undefined;
  if (start + 1 >= buffer.byteLength) return undefined;
  return buffer[start + 1] === 0x0a ? 2 : 1;
}

export function buildCommandScript(
  runDir: string,
  workspacePath: string,
  command: string,
): string {
  return [
    "#!/bin/sh",
    `echo $$ > ${shellQuote(`${runDir}/pid`)}`,
    `cd ${shellQuote(workspacePath)} || exit 127`,
    "( " + command + " )",
    "rc=$?",
    `printf '%s' "$rc" > ${shellQuote(`${runDir}/rc`)}`,
    "",
  ].join("\n");
}

export function buildWatchScript(heartbeatTicks: number): string {
  return [
    "#!/bin/sh",
    "run=$1",
    "token=$2",
    'out="$run/out.log"',
    'rc_file="$run/rc"',
    'off_file="$run/offset"',
    `[ -f "$off_file" ] || printf '%s' 1 > "$off_file"`,
    "ticks=0",
    `printf '%s%s\\n' "${TERMINAL_BEGIN_MARKER}" "$token"`,
    "while :; do",
    '  off=$(cat "$off_file" 2>/dev/null)',
    '  case "$off" in "" | *[!0-9]*) off=1 ;; esac',
    '  if [ -f "$out" ]; then',
    '    size=$(wc -c < "$out" 2>/dev/null | tr -d " ")',
    '    case "$size" in "" | *[!0-9]*) size=0 ;; esac',
    '    if [ "$size" -ge "$off" ]; then',
    '      tail -c +"$off" "$out" 2>/dev/null | head -c "$((size - off + 1))"',
    '      printf "%s" "$((size + 1))" > "$off_file"',
    "    fi",
    "  fi",
    '  if [ -f "$rc_file" ]; then',
    `    printf '%s%s:%s\\n' "${TERMINAL_END_MARKER}" "$token" "$(cat "$rc_file" 2>/dev/null)"`,
    "    break",
    "  fi",
    `  if [ "$ticks" -ge ${heartbeatTicks} ]; then`,
    `    printf '%s%s\\n' "${TERMINAL_HEARTBEAT_MARKER}" "$token"`,
    "    ticks=0",
    "  fi",
    "  ticks=$((ticks + 1))",
    "  sleep 1",
    "done",
    "stty echo 2>/dev/null",
    "",
  ].join("\n");
}

export function buildStartLine(
  runDir: string,
  token: string,
  commandScript: string,
  watchScript: string,
): string {
  const write = (name: string, content: string): string =>
    `printf '%s' ${shellQuote(Buffer.from(content, "utf8").toString("base64"))} ` +
    `| base64 -d > ${shellQuote(`${runDir}/${name}`)}`;
  const setup = [
    "stty -echo 2>/dev/null",
    `mkdir -p ${shellQuote(runDir)}`,
    write("cmd.sh", commandScript),
    write("watch.sh", watchScript),
  ].join(" && ");
  const launch =
    `{ setsid sh ${shellQuote(`${runDir}/cmd.sh`)} ` +
    `> ${shellQuote(`${runDir}/out.log`)} 2>&1 & }`;
  return (
    `${setup} && ${launch} && ` +
    `sh ${shellQuote(`${runDir}/watch.sh`)} ${shellQuote(runDir)} ${token}\n`
  );
}

export function buildResumeLine(runDir: string, token: string): string {
  return (
    `stty -echo 2>/dev/null; sh ${shellQuote(`${runDir}/watch.sh`)} ` +
    `${shellQuote(runDir)} ${token}\n`
  );
}

export function buildCancelLine(runDir: string): string {
  const pid = `"$(cat ${shellQuote(`${runDir}/pid`)} 2>/dev/null)"`;
  return (
    `kill -TERM -${pid} 2>/dev/null; sleep 1; kill -KILL -${pid} 2>/dev/null; ` +
    `rm -f ${shellQuote(`${runDir}/rc`)}\n`
  );
}

export function sandboxTerminalUrl(endpoint: SandboxEndpoint): string {
  const base = endpoint.wsBaseUrl
    .replace(/^https:\/\//, "wss://")
    .replace(/^http:\/\//, "ws://")
    .replace(/\/+$/, "");
  const query = new URLSearchParams({
    token: endpoint.apiKey,
    cols: String(TERMINAL_COLS),
    rows: String(TERMINAL_ROWS),
  });
  return `${base}/sandboxes/${endpoint.sandboxId}/terminal?${query}`;
}

export class SandboxTerminalOperations implements BashOperations {
  private readonly runsRoot: string;
  private readonly heartbeatTicks: number;
  private readonly maxReconnects: number;
  private readonly connect: SandboxTerminalConnector;
  private tail: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly endpoint: SandboxEndpoint,
    options: SandboxTerminalOptions = {},
  ) {
    this.runsRoot = options.runsRoot ?? `${endpoint.workspacePath}/${SANDBOX_RUNS_DIRNAME}`;
    this.heartbeatTicks = options.heartbeatSeconds ?? DEFAULT_HEARTBEAT_SECONDS;
    this.maxReconnects = options.maxReconnects ?? DEFAULT_MAX_RECONNECTS;
    this.connect = options.connect ?? connectWebSocket;
  }

  exec(
    command: string,
    cwd: string,
    options: {
      onData: (data: Buffer) => void;
      signal?: AbortSignal;
      timeout?: number;
      env?: NodeJS.ProcessEnv;
    },
  ): Promise<{ exitCode: number | null }> {
    const run = (): Promise<{ exitCode: number | null }> =>
      this.execute(command, cwd, options);
    const chained = this.tail.then(run, run);
    this.tail = chained.then(
      () => undefined,
      () => undefined,
    );
    return chained;
  }

  private async execute(
    command: string,
    cwd: string,
    options: {
      onData: (data: Buffer) => void;
      signal?: AbortSignal;
      timeout?: number;
    },
  ): Promise<{ exitCode: number | null }> {
    options.signal?.throwIfAborted();
    const callId = randomUUID().replace(/-/g, "").slice(0, 16);
    const token = randomBytes(9).toString("hex");
    const runDir = `${this.runsRoot}/${callId}`;
    const workspace = resolveCwd(this.endpoint.workspacePath, cwd);
    const startLine = buildStartLine(
      runDir,
      token,
      buildCommandScript(runDir, workspace, command),
      buildWatchScript(this.heartbeatTicks),
    );
    const resumeLine = buildResumeLine(runDir, token);
    const deadline =
      options.timeout !== undefined && options.timeout > 0
        ? Date.now() + options.timeout * 1000
        : undefined;

    for (let attempt = 0; attempt <= this.maxReconnects; attempt += 1) {
      const outcome = await this.attempt(
        attempt === 0 ? startLine : resumeLine,
        token,
        runDir,
        options,
        deadline,
      );
      if (outcome.kind === "exit") return { exitCode: outcome.exitCode };
      if (outcome.kind === "aborted") throw new Error("aborted");
      if (outcome.kind === "timeout") {
        throw new Error(`timeout:${options.timeout}`);
      }
    }
    throw new SandboxTerminalConnectionError(
      "sandbox terminal connection lost before the command finished",
    );
  }

  private attempt(
    line: string,
    token: string,
    runDir: string,
    options: {
      onData: (data: Buffer) => void;
      signal?: AbortSignal;
    },
    deadline: number | undefined,
  ): Promise<AttemptOutcome> {
    return new Promise<AttemptOutcome>((resolve) => {
      const scanner = new TerminalMarkerScanner(token);
      let socket: SandboxTerminalSocket | undefined;
      let settled = false;
      let timer: NodeJS.Timeout | undefined;
      const finish = (outcome: AttemptOutcome): void => {
        if (settled) return;
        settled = true;
        if (timer !== undefined) clearTimeout(timer);
        options.signal?.removeEventListener("abort", onAbort);
        try {
          socket?.close();
        } catch {
          // Closing a dead socket is best effort.
        }
        resolve(outcome);
      };
      const cancel = (outcome: AttemptOutcome): void => {
        if (settled) return;
        try {
          socket?.send("\u0003");
          setTimeout(() => {
            try {
              socket?.send(buildCancelLine(runDir));
            } catch {
              // The socket may already be closed; the group kill is best effort.
            }
            finish(outcome);
          }, CANCEL_GRACE_MS);
        } catch {
          finish(outcome);
        }
      };
      const onAbort = (): void => cancel({ kind: "aborted" });

      try {
        socket = this.connect(sandboxTerminalUrl(this.endpoint));
      } catch {
        finish({ kind: "closed" });
        return;
      }
      socket.onOpen(() => {
        socket?.send(line);
        if (deadline !== undefined) {
          timer = setTimeout(
            () => cancel({ kind: "timeout" }),
            Math.max(0, deadline - Date.now()),
          );
        }
        if (options.signal?.aborted) onAbort();
        else options.signal?.addEventListener("abort", onAbort, { once: true });
      });
      socket.onMessage((data) => {
        const chunk = typeof data === "string" ? Buffer.from(data, "utf8") : Buffer.from(data);
        const { output, exitCode } = scanner.push(chunk);
        if (output.byteLength > 0) options.onData(output);
        if (scanner.done) finish({ kind: "exit", exitCode: exitCode ?? null });
      });
      socket.onError(() => finish({ kind: "closed" }));
      socket.onClose(() => finish({ kind: "closed" }));
    });
  }
}

/**
 * Resolves the sandbox endpoint on first use and serializes every call through
 * one terminal session, so concurrent tool calls never interleave TTY output.
 *
 * A terminal that cannot stay connected is the signature of an endpoint bundle
 * the control plane has already replaced, so the bundle is dropped and the next
 * command re-requests it. The failed command is never retried on its own: shell
 * commands are not idempotent.
 */
export class LazySandboxTerminalOperations implements BashOperations {
  private operations: SandboxTerminalOperations | undefined;
  private tail: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly endpoints: SandboxEndpointSession,
    private readonly options: SandboxTerminalOptions = {},
  ) {}

  exec(
    command: string,
    cwd: string,
    options: {
      onData: (data: Buffer) => void;
      signal?: AbortSignal;
      timeout?: number;
      env?: NodeJS.ProcessEnv;
    },
  ): Promise<{ exitCode: number | null }> {
    const run = async (): Promise<{ exitCode: number | null }> => {
      try {
        return await (await this.operationsFor()).exec(command, cwd, options);
      } catch (error) {
        if (error instanceof SandboxTerminalConnectionError) {
          this.operations = undefined;
          this.endpoints.invalidate();
        }
        throw error;
      }
    };
    const chained = this.tail.then(run, run);
    this.tail = chained.then(
      () => undefined,
      () => undefined,
    );
    return chained;
  }

  private async operationsFor(): Promise<SandboxTerminalOperations> {
    if (this.operations === undefined) {
      const endpoint = await this.endpoints.resolve();
      this.operations = new SandboxTerminalOperations(endpoint, this.options);
    }
    return this.operations;
  }
}

function connectWebSocket(url: string): SandboxTerminalSocket {
  const socket = new WebSocket(url);
  socket.binaryType = "arraybuffer";
  return {
    send: (data) => socket.send(data),
    close: () => socket.close(),
    onOpen: (handler) => socket.addEventListener("open", () => handler()),
    onMessage: (handler) =>
      socket.addEventListener("message", (event) =>
        handler(event.data as string | ArrayBuffer),
      ),
    onClose: (handler) => socket.addEventListener("close", () => handler()),
    onError: (handler) => socket.addEventListener("error", (event) => handler(event)),
  };
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

function resolveCwd(workspacePath: string, cwd: string): string {
  if (!cwd || cwd === ".") return workspacePath;
  return posix.resolve(workspacePath, cwd);
}
