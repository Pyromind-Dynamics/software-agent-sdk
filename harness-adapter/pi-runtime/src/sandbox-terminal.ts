import { createHash, randomBytes, randomUUID } from "node:crypto";
import { posix } from "node:path";
import { bounded, TerminalChannel } from "./terminal-channel.js";
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
export const SANDBOX_HARNESS_DIRNAME = "harness";
export const TERMINAL_BEGIN_MARKER = "__PM_BEGIN__";
export const TERMINAL_HEARTBEAT_MARKER = "__PM_HB__";
export const TERMINAL_END_MARKER = "__PM_END__";
/**
 * Bytes of base64 one acknowledged staging line carries.
 *
 * A canonical PTY truncates a line longer than its input cap silently, which
 * leaves the shell waiting for a quote which never closes. The cap is 1024
 * bytes on macOS and 4096 on Linux, so a line - chunk plus quoting plus the
 * acknowledgement - stays under the smaller one.
 */
export const TERMINAL_SCRIPT_CHUNK_CHARS = 512;
/**
 * Staging chunk for a transport whose PTY is the Linux sandbox's own queue.
 *
 * The sandbox PTY caps one line at 4096 bytes, and a chunk this size still
 * leaves a line - chunk plus quoting - inside
 * {@link TERMINAL_BATCH_BUDGET_BYTES}, so the same payload travels in far
 * fewer bursts than the macOS-safe default allows. Every burst pays one round
 * trip, which is what makes the burst count the cost of a staging transfer.
 */
export const SANDBOX_SCRIPT_CHUNK_CHARS = 3072;
/**
 * Bytes of one staging burst written to the TTY before it is acknowledged.
 *
 * A burst is written while the shell is idle at a prompt, and the canonical
 * input queue of the Linux sandbox holds 4096 bytes, so 3584 stays inside it
 * while one acknowledgement covers the whole burst. Individual lines stay
 * under {@link TERMINAL_SCRIPT_CHUNK_CHARS}, so a burst is bounded twice over.
 * A transport whose queue is smaller (a macOS PTY caps one line at 1024) passes
 * its own `batchBudgetBytes`.
 */
export const TERMINAL_BATCH_BUDGET_BYTES = 3584;
/** Session file holding the base64 of the terminal environment. */
export const SANDBOX_ENVIRONMENT_FILE = "env.b64";
/** Session file holding the signature the environment file was staged for. */
export const SANDBOX_ENVIRONMENT_SIGNATURE_FILE = "env.sig";

const TERMINAL_COLS = 160;
const TERMINAL_ROWS = 48;
const DEFAULT_HEARTBEAT_SECONDS = 60;
const DEFAULT_MAX_RECONNECTS = 2;
/**
 * Budget for the handshake of an already-warm socket.
 *
 * A bridge which drops an idle session without a close frame leaves the socket
 * looking alive; the next handshake is what notices, and it has to fail fast
 * enough to replay the preparation on a replacement socket instead of spending
 * a whole startup budget on a dead connection.
 */
const DEFAULT_SESSION_PROBE_MS = 30_000;
/** How many replacement endpoints one command may be replayed on. */
const MAX_ENDPOINT_REFRESHES = 1;
/**
 * Resolving the endpoint waits on the control plane's own budgets (600s create
 * plus 120s workspace preparation), so this deadline is a safety net well above
 * them rather than the first thing to fail.
 */
const DEFAULT_RESOLVE_TIMEOUT_MS = 900_000;
const DEFAULT_STARTUP_TIMEOUT_MS = 180_000;
const DEFAULT_COMMAND_TIMEOUT_SECONDS = 300;
const STOP_TIMEOUT_MS = 10_000;
const ACK_MARKER = "__PM_ACK__";
/**
 * Staging bursts above which a payload is worth the file API's own round trips.
 *
 * The terminal socket is already open when a command runs, so staging over it
 * costs one acknowledged burst per {@link TERMINAL_BATCH_BUDGET_BYTES} and no
 * HTTP request. The file API costs three requests per file through the portal
 * domain, which only pays off once the payload would need many bursts.
 */
const TERMINAL_STAGING_MAX_BATCHES = 4;

export interface TerminalExecutionEvidence {
  execution_id: string;
  remote_run_dir: string;
  phase: "preparing" | "connecting" | "environment" | "starting" | "observing" | "finished";
  started: boolean;
  stopped: boolean;
  /** How the per-command script reached the sandbox, decided before the launch. */
  staging?: "file" | "terminal" | "batch" | "batch-fallback";
  /** Acknowledged exchanges spent on this command, for latency diagnosis. */
  round_trips?: number;
  /** Milliseconds per preparation phase, for latency diagnosis. */
  timings_ms?: Record<string, number>;
  error_code?: string;
}

export class SandboxTerminalExecutionError extends Error {
  constructor(message: string, readonly execution: TerminalExecutionEvidence) {
    super(`${message} (phase=${execution.phase}, started=${execution.started}, stopped=${execution.stopped})`);
    this.name = "SandboxTerminalExecutionError";
  }
}

export interface SandboxTerminalSocket {
  send(data: string): void;
  close(): void;
  onOpen(handler: () => void): void;
  onMessage(handler: (data: string | ArrayBuffer) => void): void;
  onClose(handler: () => void): void;
  onError(handler: (error: unknown) => void): void;
}

export type SandboxTerminalConnector = (url: string) => SandboxTerminalSocket;

export interface SandboxTerminalTimeouts {
  /**
   * Budget for resolving the endpoint; the control plane creates, resumes, and
   * prepares the sandbox inside this call.
   */
  resolveTimeoutMs?: number;
  /**
   * Budget for one preparation attempt: staging the run scripts, the TTY
   * handshake, the launch, and the start acknowledgement.
   */
  startupTimeoutMs?: number;
  /** Budget for confirming that a cancelled process group stopped. */
  stopTimeoutMs?: number;
}

export interface SandboxTerminalOptions extends SandboxTerminalTimeouts {
  /** Directory holding per-call `out.log` / `pid` / `rc`, inside the workspace. */
  runsRoot?: string;
  heartbeatSeconds?: number;
  maxReconnects?: number;
  /**
   * Whether the command script may travel in bounded bursts instead of one
   * acknowledged line at a time. See {@link sandboxTerminalFastPath}.
   */
  fastPath?: boolean;
  /**
   * Bytes of one acknowledged staging burst. Defaults to
   * {@link TERMINAL_BATCH_BUDGET_BYTES}; a transport whose PTY input queue is
   * smaller than the sandbox's Linux queue can lower it.
   */
  batchBudgetBytes?: number;
  /**
   * Base64 characters one staging line carries. Defaults to
   * {@link TERMINAL_SCRIPT_CHUNK_CHARS}; a transport which knows its PTY is
   * the sandbox's Linux queue raises it to
   * {@link SANDBOX_SCRIPT_CHUNK_CHARS} and stages the same payload in fewer
   * acknowledged bursts.
   */
  scriptChunkChars?: number;
  /**
   * Budget for the handshake of a socket already used by this session.
   * Defaults to {@link DEFAULT_SESSION_PROBE_MS}.
   */
  probeTimeoutMs?: number;
  /**
   * Whether each exchange and each command report their own latency on stderr.
   * See {@link sandboxTerminalTrace}.
   */
  trace?: boolean;
  connect?: SandboxTerminalConnector;
  /**
   * Optional file API used to upload a script too large to stage over the TTY.
   *
   * The runner leaves it unset: the file API is reached through a portal domain
   * which is not cluster-direct, so every script travels the socket instead.
   */
  writeFile?: (path: string, content: Buffer, signal?: AbortSignal) => Promise<void>;
}

/**
 * Operator overrides for the bridge budgets, in milliseconds.
 *
 * The runner process only inherits the variables the adapter forwards, so these
 * names are mirrored in `harness_adapter.pi_adapter.runner._runner_environment`.
 */
export function sandboxTerminalTimeouts(env: NodeJS.ProcessEnv = process.env): SandboxTerminalTimeouts {
  return {
    resolveTimeoutMs: positiveNumber(env, "PYROMIND_PI_TERMINAL_RESOLVE_TIMEOUT_MS", "milliseconds"),
    startupTimeoutMs: positiveNumber(env, "PYROMIND_PI_TERMINAL_STARTUP_TIMEOUT_MS", "milliseconds"),
  };
}

/**
 * Operator override for the staging chunk, in base64 characters.
 *
 * The runner process only inherits the variables the adapter forwards, so this
 * name is mirrored in `harness_adapter.pi_adapter.runner._runner_environment`.
 * Lowering it is the escape hatch for a transport whose PTY turns out to
 * truncate lines earlier than its documented cap.
 */
export function sandboxTerminalScriptChunkChars(
  env: NodeJS.ProcessEnv = process.env,
): number | undefined {
  return positiveNumber(env, "PYROMIND_PI_TERMINAL_SCRIPT_CHUNK_CHARS", "characters");
}

/**
 * Operator switch for the batched staging path.
 *
 * `0` restores one acknowledged line per staging line, which is the slow path
 * but pays no burst-size assumptions; anything else keeps the batched path.
 */
export function sandboxTerminalFastPath(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.PYROMIND_PI_TERMINAL_FAST_PATH !== "0";
}

/**
 * Operator switch for per-command latency diagnostics.
 *
 * `1` writes one JSON line per acknowledged exchange and one per finished
 * command. The lines go to stderr, which sits outside the JSONL protocol, so a
 * traced session behaves exactly like an untraced one.
 */
export function sandboxTerminalTrace(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.PYROMIND_PI_TERMINAL_TRACE === "1";
}

function positiveNumber(
  env: NodeJS.ProcessEnv,
  name: string,
  unit: string,
): number | undefined {
  const raw = env[name];
  if (raw === undefined || raw.trim() === "") return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive number of ${unit}`);
  }
  return value;
}

interface TerminalExecOptions {
  onData: (data: Buffer) => void;
  signal?: AbortSignal;
  timeout?: number;
  env?: NodeJS.ProcessEnv;
}

interface TerminalExecResult {
  exitCode: number | null;
  execution: TerminalExecutionEvidence;
}

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

// Credentials arrive only in launch-process memory, never in cmd.sh or run specs.
export const ENVIRONMENT_RUNNER = `import os,sys,json,base64,subprocess
extra=json.loads(base64.b64decode(os.environ.pop("PYROMIND_TERMINAL_ENV")))
secrets=[]
def collect(values):
 for key,value in values.items():
  if isinstance(value,dict): collect(value)
  elif isinstance(value,str) and any(s in key.upper() for s in ("KEY","TOKEN","SECRET","AUTHORIZATION")) and value: secrets.append(value)
collect(extra)
if "PYROMIND_DATAFLOW_PROFILES" in extra:
 collect(json.loads(extra["PYROMIND_DATAFLOW_PROFILES"]))
secrets.sort(key=len,reverse=True)
process=subprocess.Popen(["sh","-c",sys.argv[1]],env={**os.environ,**extra},stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
# Redact before bytes enter the remote out.log, including split writes.
pending=b""
keys=[s.encode() for s in secrets]
keep=max([len(s) for s in keys]+[1])-1
def output(data,final=False):
 while data and (final or len(data)>keep):
  match=next((key for key in keys if data.startswith(key)),None)
  if match: sys.stdout.buffer.write(b"[REDACTED]"); data=data[len(match):]
  else: sys.stdout.buffer.write(data[:1]); data=data[1:]
 sys.stdout.buffer.flush()
 return data
while True:
 chunk=os.read(process.stdout.fileno(),65536)
 if not chunk: break
 pending=output(pending+chunk)
output(pending,True)
sys.exit(process.wait())
`;

export function buildCommandScript(
  runDir: string,
  workspacePath: string,
  command: string,
  environmentRunner?: string,
): string {
  return [
    "#!/bin/sh",
    `echo $$ > ${shellQuote(`${runDir}/pid`)}`,
    `trap ${shellQuote(`rc=$?; printf "%s" "$rc" > ${shellQuote(`${runDir}/rc`)}`)} EXIT`,
    `cd ${shellQuote(workspacePath)} || exit 127`,
    environmentRunner === undefined
      ? `sh -c ${shellQuote(command)}`
      : `python3 -u ${shellQuote(environmentRunner)} ${shellQuote(command)}`,
    'exit "$?"',
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

function acknowledgement(token: string, stage: string, value = "ok"): string {
  return `printf '%s%s:%s\\n' '${ACK_MARKER}' '${token}:${stage}' ${shellQuote(value)}`;
}

export function buildLaunchScript(): string {
  // Atomic guard belongs to the remote execution, not the WebSocket connection.
  // launch_pid is written by the controlling shell before it starts observing.
  return `#!/bin/sh\nrun=$1\nif mkdir "$run/launched" 2>/dev/null; then { ` +
    `setsid sh "$run/cmd.sh" > "$run/out.log" 2>&1 & }; ` +
    `printf '%s' "$!" > "$run/launch_pid"; fi\n`;
}

/** The one script a single run needs on the sandbox side, keyed by file name. */
export type RunScripts = Record<"cmd.sh", string>;

/**
 * Files the sandbox keeps for the whole terminal session, keyed by file name.
 *
 * The harness carries no per-command state, so it is staged once and reused;
 * only `cmd.sh` is rebuilt for every command.
 */
export type HarnessFiles = Record<"watch.sh" | "launch.sh" | "env_runner.py", string>;

export interface TerminalStageLine {
  /** Name this line acknowledges itself under. */
  stage: string;
  /** Shell line which ends by acknowledging its own execution. */
  command: string;
}

/**
 * Lines which carry the staged bytes, without any acknowledgement.
 *
 * The payload travels as base64 chunks over the shell's own input, so it needs
 * no file API; each line stays far below the canonical input cap (see
 * {@link TERMINAL_SCRIPT_CHUNK_CHARS}), and callers group the lines into
 * bursts no larger than {@link TERMINAL_BATCH_BUDGET_BYTES}.
 *
 * Private staging wraps every file-touching line in its own `umask 077`
 * subshell, so the transient `.b64` files are unreadable while they exist and
 * the decoded files are created unreadable too. The wrap is per line because a
 * batch can split the group anywhere: a subshell left open across bursts would
 * trap a burst's acknowledgement behind the still-unparsed closing parenthesis.
 */
function stagePayloadLines(
  dir: string,
  files: Record<string, string>,
  privateStaging = false,
  chunkChars = TERMINAL_SCRIPT_CHUNK_CHARS,
): string[] {
  const lines: string[] = [];
  const names = Object.keys(files);
  const privateLine = (command: string): string =>
    privateStaging ? `(umask 077; ${command})` : command;
  // The directory is named once: repeating an absolute path in every chunk,
  // decode, and check would push lines past the smaller canonical input cap.
  lines.push(`pm_dir=${shellQuote(dir)}; ${privateLine(`mkdir -p "$pm_dir"`)}`);
  // One truncate for every file keeps the staging files empty for a rerun.
  lines.push(names.map((name) => privateLine(`: > "$pm_dir/${name}.b64"`)).join("; "));
  for (const name of names) {
    const encoded = Buffer.from(files[name]!, "utf8").toString("base64");
    for (let at = 0; at < encoded.length; at += chunkChars) {
      lines.push(privateLine(
        `printf '%s' ${shellQuote(encoded.slice(at, at + chunkChars))} >> "$pm_dir/${name}.b64"`));
    }
  }
  // The staging files are removed only when every decode succeeded.
  lines.push(privateLine(names
    .map((name) => `base64 -d < "$pm_dir/${name}.b64" > "$pm_dir/${name}"`)
    .join(" && ")) + ` && rm -f ${names.map((name) => `"$pm_dir/${name}.b64"`).join(" ")}`);
  return lines;
}

/**
 * Condition which is true only when every file decoded to its exact bytes.
 *
 * The byte check is what makes one acknowledgement safe: a burst which lost
 * bytes decodes to a different size, so the batch reports `missing` instead of
 * launching a corrupt script. The condition reads `$pm_dir`, which the caller
 * sets to the staging directory on the same line: naming that directory once
 * keeps the line short enough for the smaller canonical input cap.
 */
function stagedCondition(files: Record<string, string>): string {
  return Object.entries(files)
    .map(([name, content]) => `[ -s "$pm_dir/${name}" ] && ` +
      `[ ! -f "$pm_dir/${name}.b64" ] && ` +
      `[ "$(wc -c < "$pm_dir/${name}" 2>/dev/null | tr -d ' ')" = "${Buffer.byteLength(content)}" ]`)
    .join(" && ");
}

export function buildStageLines(
  dir: string,
  token: string,
  files: Record<string, string>,
  privateStaging = false,
  chunkChars = TERMINAL_SCRIPT_CHUNK_CHARS,
): TerminalStageLine[] {
  return stagePayloadLines(dir, files, privateStaging, chunkChars).map((command, index) => {
    const stage = `stage${index}`;
    return { stage, command: `${command}; ${acknowledgement(token, stage)}\n` };
  });
}

/** Final staging check: reports `ok` only when every script decoded. */
export function buildStageCheckLine(
  dir: string,
  token: string,
  files: Record<string, string>,
): TerminalStageLine {
  return {
    stage: "staged",
    command: `pm_dir=${shellQuote(dir)}; if ${stagedCondition(files)}; then ${acknowledgement(token, "staged")}; ` +
      `else ${acknowledgement(token, "staged", "missing")}; fi\n`,
  };
}

/** One acknowledged burst of staging lines. */
export interface TerminalStageBatch {
  stage: string;
  command: string;
}

/**
 * Groups the staging lines into bursts which each end in one acknowledgement.
 *
 * The safe path acknowledges every line; a batch pays one exchange for up to
 * {@link TERMINAL_BATCH_BUDGET_BYTES} of them, and the final batch verifies
 * every decoded file byte for byte so a dropped burst is detected instead of
 * staged as a corrupt script.
 */
export function buildStageBatches(
  dir: string,
  token: string,
  files: Record<string, string>,
  budgetBytes = TERMINAL_BATCH_BUDGET_BYTES,
  privateStaging = false,
  chunkChars = TERMINAL_SCRIPT_CHUNK_CHARS,
): TerminalStageBatch[] {
  const check = buildStageCheckLine(dir, token, files).command;
  // The burst ends with the check which reads the decoded bytes back, so the
  // check's own length is part of the budget the payload is grouped against.
  const checkBytes = Buffer.byteLength(check);
  const payloadBudget = budgetBytes - checkBytes - 1;
  const groups: string[][] = [];
  let current: string[] = [];
  let currentBytes = 0;
  for (const line of stagePayloadLines(dir, files, privateStaging, chunkChars)) {
    const bytes = Buffer.byteLength(line) + 1;
    if (current.length > 0 && currentBytes + bytes > payloadBudget) {
      groups.push(current);
      current = [];
      currentBytes = 0;
    }
    current.push(line);
    currentBytes += bytes;
  }
  if (current.length > 0) groups.push(current);
  return groups.map((group, index) => {
    const final = index === groups.length - 1;
    const stage = final ? "staged" : `staged${index}`;
    return {
      stage,
      command: final
        ? `${group.join("\n")}\n${check}`
        : `${group.join("\n")}\n${acknowledgement(token, stage)}\n`,
    };
  });
}

/**
 * One burst which stages the command script, launches it, and acknowledges.
 *
 * The launch is guarded by the staged byte check and by `launch.sh`'s own
 * atomic guard, so a lost acknowledgement never dispatches a second run.
 */
export function buildLaunchBatchLine(
  harnessDir: string,
  runDir: string,
  token: string,
  files: Record<string, string>,
  withEnvironment = false,
  chunkChars = TERMINAL_SCRIPT_CHUNK_CHARS,
): string {
  const environment = withEnvironment
    ? `PYROMIND_TERMINAL_ENV="$(cat ${shellQuote(`${harnessDir}/${SANDBOX_ENVIRONMENT_FILE}`)})" `
    : "";
  return `${stagePayloadLines(runDir, files, false, chunkChars).join("\n")}\n` +
    `pm_dir=${shellQuote(runDir)}; if ${stagedCondition(files)}; then ` +
    `${environment}sh ${shellQuote(`${harnessDir}/launch.sh`)} ${shellQuote(runDir)}; ` +
    `${acknowledgement(token, "launch")}; ` +
    `else ${acknowledgement(token, "launch", "missing")}; fi\n`;
}

/** What the session handshake found on disk. */
export type TerminalReadyState = "ok" | "missing" | "env";

/**
 * Session handshake: disables echo and history, then reports which session
 * files are already on disk.
 *
 * No payload is sent before this acknowledgement, so a shell which cannot
 * disable echo fails closed instead of echoing the environment. `missing` means
 * the harness has to be staged, `env` means only the environment has to be, and
 * `ok` means the command can go straight to its launch line.
 */
export function buildReadyLine(
  harnessDir: string,
  token: string,
  harnessFiles: Record<string, string>,
  environmentSignature?: string,
): string {
  // The directory is named once: the harness and environment checks repeat it,
  // and a long workspace path would otherwise push the line past the PTY cap.
  const harnessChecks = Object.keys(harnessFiles)
    .map((name) => `[ -s "$pm_dir/${name}" ]`).join(" && ");
  const environmentChecks = environmentSignature === undefined ? undefined :
    `[ -s "$pm_dir/${SANDBOX_ENVIRONMENT_FILE}" ] && ` +
    `[ "$(cat "$pm_dir/${SANDBOX_ENVIRONMENT_SIGNATURE_FILE}" 2>/dev/null)" ` +
    `= "${environmentSignature}" ]`;
  const ready = environmentChecks === undefined
    ? acknowledgement(token, "ready")
    : `if ${environmentChecks}; then ${acknowledgement(token, "ready")}; ` +
      `else ${acknowledgement(token, "ready", "env")}; fi`;
  return `pm_dir=${shellQuote(harnessDir)}; stty -echo && unset HISTFILE && set +x && HISTSIZE=0 && ` +
    `{ if [ -n "\${BASH_VERSION-}" ]; then set +H; fi; ` +
    `if [ -n "\${ZSH_VERSION-}" ]; then unsetopt BANG_HIST; fi; } && ` +
    `if ${harnessChecks}; then ${ready}; ` +
    `else ${acknowledgement(token, "ready", "missing")}; fi\n`;
}

export function buildStartLine(
  harnessDir: string,
  runDir: string,
  token: string,
  withEnvironment = false,
): string {
  const environment = withEnvironment
    ? `PYROMIND_TERMINAL_ENV="$(cat ${shellQuote(`${harnessDir}/${SANDBOX_ENVIRONMENT_FILE}`)})" `
    : "";
  return `${environment}sh ${shellQuote(`${harnessDir}/launch.sh`)} ${shellQuote(runDir)}; ` +
    `${acknowledgement(token, "launch")}\n`;
}

/** The base64 environment plus the files and signature the session caches it under. */
export interface StagedEnvironment {
  encoded: string;
  signature: string;
  files: Record<string, string>;
}

/**
 * Encodes one command's environment for the session cache.
 *
 * The signature is a short digest of the encoded bytes, used as a cache key so
 * a changed environment is restaged instead of reused; it is not a security
 * boundary, and the bytes it names never leave the user's own sandbox.
 */
export function stagedEnvironment(values: Record<string, string>): StagedEnvironment {
  const encoded = Buffer.from(JSON.stringify(values)).toString("base64");
  const signature = createHash("sha256").update(encoded).digest("hex").slice(0, 16);
  return {
    encoded,
    signature,
    files: {
      [SANDBOX_ENVIRONMENT_FILE]: encoded,
      [SANDBOX_ENVIRONMENT_SIGNATURE_FILE]: signature,
    },
  };
}

export function buildResumeLine(runDir: string, token: string): string {
  // Watcher BEGIN only indicates observation. First confirm the workload's pid
  // or completed rc (also covers commands which finish before we reconnect).
  return `if [ -s ${shellQuote(`${runDir}/pid`)} ] || [ -s ${shellQuote(`${runDir}/rc`)} ]; then ` +
    `${acknowledgement(token, "started")}; else ${acknowledgement(token, "started", "pending")}; fi\n`;
}

export function buildCancelLine(runDir: string, token: string): string {
  return `unset PYROMIND_ENV_BUFFER; ` +
    `p=$(cat ${shellQuote(`${runDir}/launch_pid`)} 2>/dev/null); ` +
    `case "$p" in ''|*[!0-9]*) ` +
    `if mkdir ${shellQuote(`${runDir}/launched`)} 2>/dev/null; then ${acknowledgement(token, "stop", "yes")}; ` +
    `else ${acknowledgement(token, "stop", "unknown")}; fi ;; *) ` +
    `kill -TERM -"$p" 2>/dev/null; sleep 1; kill -KILL -"$p" 2>/dev/null; ` +
    `i=0; while kill -0 -"$p" 2>/dev/null && [ "$i" -lt 20 ]; do sleep 0.1; i=$((i + 1)); done; ` +
    `if kill -0 -"$p" 2>/dev/null; then ${acknowledgement(token, "stop", "unknown")}; ` +
    `else ${acknowledgement(token, "stop", "yes")}; fi ;; esac\n`;
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
  private readonly fastPath: boolean;
  private readonly batchBudgetBytes: number;
  private readonly scriptChunkChars: number;
  private readonly probeTimeoutMs: number;
  private readonly trace: boolean;
  private readonly connect: SandboxTerminalConnector;
  private tail: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly endpoint: SandboxEndpoint,
    options: SandboxTerminalOptions = {},
  ) {
    this.settings = options;
    this.runsRoot = options.runsRoot ?? `${endpoint.workspacePath}/${SANDBOX_RUNS_DIRNAME}`;
    this.heartbeatTicks = options.heartbeatSeconds ?? DEFAULT_HEARTBEAT_SECONDS;
    this.maxReconnects = options.maxReconnects ?? DEFAULT_MAX_RECONNECTS;
    this.fastPath = options.fastPath ?? sandboxTerminalFastPath();
    this.batchBudgetBytes = options.batchBudgetBytes ?? TERMINAL_BATCH_BUDGET_BYTES;
    this.scriptChunkChars = options.scriptChunkChars ?? TERMINAL_SCRIPT_CHUNK_CHARS;
    this.probeTimeoutMs = options.probeTimeoutMs ?? DEFAULT_SESSION_PROBE_MS;
    this.trace = options.trace ?? sandboxTerminalTrace();
    this.connect = options.connect ?? connectWebSocket;
  }

  exec(
    command: string,
    cwd: string,
    options: TerminalExecOptions,
  ): Promise<TerminalExecResult> {
    const run = (): Promise<TerminalExecResult> =>
      this.execute(command, cwd, options);
    const chained = this.tail.then(run, run);
    this.tail = chained.then(
      () => undefined,
      () => undefined,
    );
    return chained;
  }

  private quarantined: { runDir: string; token: string; execution: TerminalExecutionEvidence } | undefined;
  private readonly settings: SandboxTerminalOptions;
  /** The socket this session keeps for every command; never one per command. */
  private session: TerminalChannel | undefined;
  /** Successful exchanges on the current socket; zero means it is new. */
  private sessionExchanges = 0;

  private channel(): TerminalChannel {
    if (this.session !== undefined && !this.session.dead) return this.session;
    const channel = new TerminalChannel(this.connect(sandboxTerminalUrl(this.endpoint)));
    this.session = channel;
    this.sessionExchanges = 0;
    return channel;
  }

  /**
   * Drops one socket instead of holding it for the next command.
   *
   * A session socket is only dropped when it died or when an exchange failed:
   * the platform bridge closes idle sessions, and every command's handshake is
   * what proves a socket still reaches a shell.
   */
  private retire(channel: TerminalChannel): void {
    if (this.session === channel) {
      this.session = undefined;
      this.sessionExchanges = 0;
    }
    channel.close();
  }

  /** Closes the session socket; the next command resolves a fresh one. */
  dispose(): void {
    const channel = this.session;
    this.session = undefined;
    this.sessionExchanges = 0;
    channel?.close();
  }

  private async stop(runDir: string, token: string, channel?: TerminalChannel): Promise<boolean> {
    const deadline = Date.now() + (this.settings.stopTimeoutMs ?? STOP_TIMEOUT_MS);
    // The signal disturbs the foreground job, the input queue, and the shell's
    // own line state, so a socket which carried an interrupt is dropped
    // instead of being handed to the next command.
    const interrupted = channel !== undefined;
    let current = channel;
    try {
      if (current) {
        current.interrupt();
        // Ctrl-C must finish the foreground watcher before the shell reads stop.
        await bounded(new Promise((resolve) => setTimeout(resolve, 50)), deadline);
      } else current = this.channel();
      const stopped = await current.acknowledge(buildCancelLine(runDir, token),
        `${ACK_MARKER}${token}:stop:`, deadline) === "yes";
      if (interrupted) this.retire(current);
      return stopped;
    } catch {
      if (current !== undefined) this.retire(current);
      if (!channel || Date.now() >= deadline) return false;
      // A dead observation socket does not imply a stopped process.
      try {
        current = this.channel();
        const stopped = await current.acknowledge(buildCancelLine(runDir, token),
          `${ACK_MARKER}${token}:stop:`, deadline) === "yes";
        this.retire(current);
        return stopped;
      } catch { return false; }
    }
  }

  /**
   * Preparation gets a fresh startup budget per attempt; the command deadline
   * still caps it so a short command timeout stays short.
   */
  private prepareDeadline(deadline: number): number {
    return Math.min(deadline, Date.now() + (this.settings.startupTimeoutMs ?? DEFAULT_STARTUP_TIMEOUT_MS));
  }

  /**
   * One acknowledged exchange, counted for latency diagnosis.
   */
  private async acknowledge(
    channel: TerminalChannel,
    evidence: TerminalExecutionEvidence,
    command: string,
    marker: string,
    deadline: number,
    signal?: AbortSignal,
  ): Promise<string> {
    evidence.round_trips = (evidence.round_trips ?? 0) + 1;
    const started = Date.now();
    try {
      const value = await channel.acknowledge(command, marker, deadline, signal);
      if (this.session === channel) this.sessionExchanges += 1;
      this.traceExchange(evidence, marker, started, value, Buffer.byteLength(command));
      return value;
    } catch (error) {
      this.traceExchange(evidence, marker, started, "error", Buffer.byteLength(command));
      throw error;
    }
  }

  /**
   * The first exchange of a command also waits for the socket to open, so its
   * duration carries the connection cost the later exchanges do not.
   */
  private traceExchange(
    evidence: TerminalExecutionEvidence,
    marker: string,
    started: number,
    outcome: string,
    bytes: number,
  ): void {
    if (!this.trace) return;
    // Markers are `<ack><token>:<stage>:`, so the second field names the stage.
    const stage = marker.split(":")[1] ?? marker;
    process.stderr.write(`${JSON.stringify({
      event: "terminal.exchange", execution_id: evidence.execution_id,
      round_trip: evidence.round_trips, stage, outcome, bytes, ms: Date.now() - started,
    })}\n`);
  }

  private traceExecution(evidence: TerminalExecutionEvidence): void {
    if (!this.trace) return;
    process.stderr.write(`${JSON.stringify({ event: "terminal.execution", ...evidence })}\n`);
  }

  /**
   * Picks the transport for the per-command script.
   *
   * Transport is a cost decision, not a capability one: the TTY is already open
   * and carries a small payload in a single burst which also launches it, while
   * the file API spends three requests per file. Only a payload which would need
   * many TTY bursts is worth that, and a file API which is missing or failing is
   * never fatal - the TTY is the one channel every sandbox has.
   */
  private scriptTransport(
    harnessDir: string,
    runDir: string,
    token: string,
    scripts: RunScripts,
    withEnvironment: boolean,
  ): "single" | "file" | "batches" | "lines" {
    if (!this.fastPath) return "lines";
    if (Buffer.byteLength(buildLaunchBatchLine(harnessDir, runDir, token, scripts, withEnvironment,
      this.scriptChunkChars)) <=
        this.batchBudgetBytes) return "single";
    if (this.settings.writeFile !== undefined &&
        buildStageBatches(runDir, token, scripts, this.batchBudgetBytes, false, this.scriptChunkChars)
          .length > TERMINAL_STAGING_MAX_BATCHES) return "file";
    return "batches";
  }

  /**
   * Uploads the per-command script through the file API.
   *
   * The upload gets its own startup budget so a slow file API cannot spend the
   * budget the launch needs, and it is aborted when it loses that budget: a
   * timed-out upload keeps running in the background, and its late chunks would
   * overwrite the script the TTY staging path writes in the meantime.
   */
  private async uploadScripts(
    runDir: string,
    scripts: RunScripts,
    deadline: number,
    options: TerminalExecOptions,
  ): Promise<void> {
    const writeFile = this.settings.writeFile;
    if (writeFile === undefined) throw new Error("sandbox file API unavailable");
    const uploadDeadline = this.prepareDeadline(deadline);
    const upload = new AbortController();
    const signal = options.signal
      ? AbortSignal.any([upload.signal, options.signal])
      : upload.signal;
    try {
      for (const [name, content] of Object.entries(scripts)) {
        await bounded(writeFile(`${runDir}/${name}`, Buffer.from(content), signal), uploadDeadline, signal);
      }
    } catch (error) {
      upload.abort();
      if (options.signal?.aborted || (error instanceof Error && error.message === "aborted")) throw error;
      throw new Error("sandbox terminal script upload failed");
    }
  }

  /**
   * Stages one file set in bounded bursts.
   *
   * Every burst but the last acknowledges unconditionally; the last one reports
   * whether every file decoded to its exact bytes, so a burst which lost bytes
   * is refused instead of staged as a corrupt file.
   */
  private async stageBursts(
    channel: TerminalChannel,
    evidence: TerminalExecutionEvidence,
    dir: string,
    token: string,
    files: Record<string, string>,
    deadline: number,
    options: TerminalExecOptions,
    privateStaging = false,
  ): Promise<boolean> {
    const batches = buildStageBatches(dir, token, files, this.batchBudgetBytes, privateStaging,
      this.scriptChunkChars);
    let staged = false;
    for (let index = 0; index < batches.length; index += 1) {
      const batch = batches[index]!;
      const value = await this.acknowledge(channel, evidence, batch.command,
        `${ACK_MARKER}${token}:${batch.stage}:`, deadline, options.signal);
      staged = index === batches.length - 1 && value === "ok";
    }
    return staged;
  }

  /**
   * Stages the run scripts over the TTY, one acknowledged line at a time.
   *
   * This is the repair path for a burst which lost bytes, and the staging path
   * when the batched fast path is switched off: the shell executes each line
   * before the next one arrives, so nothing depends on the PTY input queue
   * holding more than one line.
   */
  private async stageOverTerminal(
    channel: TerminalChannel,
    evidence: TerminalExecutionEvidence,
    dir: string,
    token: string,
    files: Record<string, string>,
    deadline: number,
    options: TerminalExecOptions,
    privateStaging = false,
  ): Promise<void> {
    for (const line of buildStageLines(dir, token, files, privateStaging, this.scriptChunkChars)) {
      await this.acknowledge(channel, evidence, line.command,
        `${ACK_MARKER}${token}:${line.stage}:`, deadline, options.signal);
    }
    const check = buildStageCheckLine(dir, token, files);
    const staged = await this.acknowledge(channel, evidence, check.command,
      `${ACK_MARKER}${token}:${check.stage}:`, deadline, options.signal);
    if (staged !== "ok") throw new Error("sandbox terminal script staging failed");
  }

  private async execute(
    command: string,
    cwd: string,
    options: TerminalExecOptions,
  ): Promise<TerminalExecResult> {
    options.signal?.throwIfAborted();
    if (this.quarantined) {
      const previous = this.quarantined;
      if (!await this.stop(previous.runDir, previous.token)) {
        throw new SandboxTerminalExecutionError("terminal blocked: previous command termination is unconfirmed",
          { ...previous.execution, error_code: "termination_unconfirmed" });
      }
      this.quarantined = undefined;
    }
    const callId = randomUUID().replace(/-/g, "");
    // Every preparation attempt gets its own token: a socket reused across
    // commands can still deliver a late acknowledgement, and a token only this
    // attempt prints keeps that output from being read as this attempt's own.
    let token = randomBytes(9).toString("hex");
    const runDir = `${this.runsRoot}/${callId}`;
    const timeout = options.timeout && options.timeout > 0
      ? options.timeout : DEFAULT_COMMAND_TIMEOUT_SECONDS;
    const deadline = Date.now() + timeout * 1000;
    const evidence: TerminalExecutionEvidence = { execution_id: callId, remote_run_dir: runDir,
      phase: "preparing", started: false, stopped: false };
    const openedAt = Date.now();
    const timings: Record<string, number> = {};
    const mark = (name: string, from: number): void => {
      timings[name] = (timings[name] ?? 0) + Date.now() - from;
    };
    let channel: TerminalChannel | undefined;
    let dispatched = false;
    try {
      const values: Record<string, string> = {};
      for (const [name, value] of Object.entries(options.env ?? {})) {
        if (value !== undefined) values[name] = value;
      }
      if (Object.keys(values).some((name) => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) ||
          Object.values(values).some((value) => value.includes("\0"))) {
        throw new Error("invalid process environment");
      }
      const withEnvironment = Object.keys(values).length > 0;
      const environment = withEnvironment ? stagedEnvironment(values) : undefined;
      const harnessDir = `${this.runsRoot}/${SANDBOX_HARNESS_DIRNAME}`;
      const harness: HarnessFiles = {
        "watch.sh": buildWatchScript(this.heartbeatTicks),
        "launch.sh": buildLaunchScript(),
        "env_runner.py": ENVIRONMENT_RUNNER,
      };
      const scripts: RunScripts = {
        "cmd.sh": buildCommandScript(
          runDir, resolveCwd(this.endpoint.workspacePath, cwd), command,
          withEnvironment ? `${harnessDir}/env_runner.py` : undefined),
      };
      const transport = this.scriptTransport(harnessDir, runDir, token, scripts, withEnvironment);
      evidence.staging = transport === "file" ? "file" : transport === "lines" ? "terminal" : "batch";
      for (let attempt = 0; ; attempt += 1) {
        evidence.phase = "connecting";
        // Tokens are always 18 hex characters, so the transport decision taken
        // above still describes the burst this attempt builds.
        token = randomBytes(9).toString("hex");
        const startupDeadline = this.prepareDeadline(deadline);
        channel = this.channel();
        // A socket this session already used carries no open cost, but it can
        // have been dropped without a close frame while it sat idle. Its
        // handshake therefore gets a probe budget: failing fast is what lets
        // the preparation replay on a replacement socket.
        const warm = this.sessionExchanges > 0;
        const readyDeadline = warm
          ? Math.min(startupDeadline, Date.now() + this.probeTimeoutMs)
          : startupDeadline;
        let handshakeDone = false;
        try {
          // Fail closed if echo/history disabling cannot be confirmed. No secrets
          // are sent before this acknowledgement, and no raw setup output is kept.
          // The same acknowledgement reports which session files are already on
          // disk, so a healthy session pays nothing to restage them.
          let phase = Date.now();
          const ready = await this.acknowledge(channel, evidence,
            buildReadyLine(harnessDir, token, harness, environment?.signature),
            `${ACK_MARKER}${token}:ready:`, readyDeadline, options.signal);
          handshakeDone = true;
          mark("ready", phase);
          if (ready !== "ok") {
            if (ready === "missing") {
              phase = Date.now();
              if (!this.fastPath ||
                  !await this.stageBursts(channel, evidence, harnessDir, token, harness, startupDeadline, options)) {
                await this.stageOverTerminal(channel, evidence, harnessDir, token, harness, startupDeadline, options);
              }
              mark("stage_harness", phase);
            }
            if (environment !== undefined) {
              evidence.phase = "environment";
              phase = Date.now();
              if (!this.fastPath || !await this.stageBursts(channel, evidence, harnessDir, token, environment.files,
                  startupDeadline, options, true)) {
                await this.stageOverTerminal(channel, evidence, harnessDir, token, environment.files,
                  startupDeadline, options, true);
              }
              mark("stage_env", phase);
            }
          }
          evidence.phase = "starting";
          phase = Date.now();
          if (transport === "single") {
            // The burst stages the script and launches it, so from here on a
            // lost acknowledgement is observed rather than replayed.
            dispatched = true;
            const launched = await this.acknowledge(channel, evidence,
              buildLaunchBatchLine(harnessDir, runDir, token, scripts, withEnvironment,
                this.scriptChunkChars),
              `${ACK_MARKER}${token}:launch:`, startupDeadline, options.signal);
            mark("launch", phase);
            if (launched === "ok") evidence.started = true;
            else {
              // The byte check refused the burst and the launch guard proves
              // nothing started, so the acknowledged path can repair the script.
              evidence.staging = "batch-fallback";
              phase = Date.now();
              await this.stageOverTerminal(channel, evidence, runDir, token, scripts, startupDeadline, options);
              mark("stage_script", phase);
            }
          } else {
            phase = Date.now();
            if (transport === "file") {
              try {
                await this.uploadScripts(runDir, scripts, deadline, options);
              } catch (error) {
                if (options.signal?.aborted ||
                    (error instanceof Error && error.message === "aborted")) throw error;
                evidence.staging = "batch-fallback";
                await this.stageOverTerminal(channel, evidence, runDir, token, scripts,
                  this.prepareDeadline(deadline), options);
              }
            } else if (transport === "lines") {
              await this.stageOverTerminal(channel, evidence, runDir, token, scripts, startupDeadline, options);
            } else if (!await this.stageBursts(channel, evidence, runDir, token, scripts, startupDeadline, options)) {
              evidence.staging = "batch-fallback";
              await this.stageOverTerminal(channel, evidence, runDir, token, scripts, startupDeadline, options);
            }
            mark(transport === "file" ? "upload_script" : "stage_script", phase);
          }
          if (!evidence.started) {
            dispatched = true; // Once sent (or possibly sent), never launch it again.
            phase = Date.now();
            await this.acknowledge(channel, evidence, buildStartLine(harnessDir, runDir, token, withEnvironment),
              `${ACK_MARKER}${token}:launch:`, startupDeadline, options.signal);
            mark("launch", phase);
            // The acknowledgement proves the launch line ran, so the happy path
            // goes straight to the watcher; only a replacement socket confirms
            // the run before observing it.
            evidence.started = true;
          }
          break;
        } catch (error) {
          // A warm socket which stops answering is the signature of a bridge
          // which dropped the connection without a close frame: retire it and
          // replay the preparation on a replacement socket. Only the handshake
          // is allowed to read a timeout that way: once the shell answered, a
          // later timeout is a slow or lost exchange, not a dead socket.
          const recoverable = isConnectionLost(error) ||
            (warm && isTimeout(error) && !handshakeDone && !dispatched);
          if (!recoverable) throw error;
          this.retire(channel);
          // Once the launch line is sent the run may exist, so the replacement
          // socket only observes it: the resume line reports whether it started,
          // and the launch is never replayed.
          if (dispatched) {
            channel = this.channel();
            break;
          }
          // Nothing has run yet, and the staged scripts are already on disk, so
          // the handshake is replayed instead of failing the command.
          if (attempt === this.maxReconnects) throw error;
        }
      }
      for (let attempt = 0; attempt <= this.maxReconnects; attempt += 1) {
        try {
          while (!evidence.started) {
            // Only a socket which replaced a dispatched run reaches this poll:
            // the launch acknowledgement was lost with the old socket, so the
            // replacement confirms the pid or the exit code before observing.
            // Every poll gets its own budget, so a reconnect is never starved by
            // a startup budget which already elapsed on the previous socket; the
            // command deadline stays the outer bound.
            const pollDeadline = this.prepareDeadline(deadline);
            const phase = Date.now();
            const state = await this.acknowledge(channel, evidence, buildResumeLine(runDir, token),
              `${ACK_MARKER}${token}:started:`, pollDeadline, options.signal);
            mark("started", phase);
            if (state === "ok") evidence.started = true;
            else await bounded(new Promise((resolve) => setTimeout(resolve, 50)), pollDeadline, options.signal);
          }
          evidence.phase = "observing";
          const scanner = new TerminalMarkerScanner(token);
          const phase = Date.now();
          const exitCode = await channel.receive<number | null>(
            `sh ${shellQuote(`${harnessDir}/watch.sh`)} ${shellQuote(runDir)} ${token}\n`, deadline, options.signal,
            (chunk, resolve) => {
              const result = scanner.push(chunk);
              if (result.output.length) options.onData(result.output);
              if (scanner.done) resolve(result.exitCode ?? null);
            });
          mark("watch", phase);
          // END proves an exit code, but cancellation additionally checks the
          // process group. Normal completion preserves rc as durable evidence.
          evidence.phase = "finished";
          evidence.stopped = true;
          evidence.timings_ms = { ...timings, total: Date.now() - openedAt };
          return { exitCode, execution: { ...evidence } };
        } catch (error) {
          if (!isConnectionLost(error) || attempt === this.maxReconnects) throw error;
          channel.close();
          channel = this.channel(); // Only observe, never resend the launch.
        }
      }
      throw new Error("closed");
    } catch (error) {
      const reason = error instanceof Error ? error.message : "execution failed";
      evidence.timings_ms = { ...timings, total: Date.now() - openedAt };
      // Only a dispatched run owns a process group; failing before the launch
      // line leaves nothing to confirm, so the error surfaces without paying
      // for a replacement socket to cancel it. The session files staged so far
      // stay on disk: the next command reuses them.
      if (dispatched) {
        evidence.stopped = await this.stop(runDir, token, channel);
      } else {
        evidence.stopped = true;
      }
      evidence.error_code = reason === "timeout"
        ? (evidence.phase === "observing" ? "execution_timeout" : "startup_timeout")
        : reason === "aborted" ? "aborted" : reason === "closed" ? "connection_lost" : "preparation_failed";
      if (!evidence.stopped) this.quarantined = { runDir, token, execution: evidence };
      const message = reason === "timeout" ? `timeout:${timeout}`
        : reason === "closed" ? "sandbox terminal connection lost before the command finished"
        : reason === "aborted" ? "aborted" : "sandbox terminal preparation failed";
      throw new SandboxTerminalExecutionError(message, evidence);
    } finally {
      this.traceExecution(evidence);
      // A socket which carried the command stays open for the next one. A
      // socket left behind by a failure is dropped instead: a burst which timed
      // out may have been half consumed, so its input state is no longer known.
      if (channel !== undefined && (channel.dead || evidence.phase !== "finished")) {
        this.retire(channel);
      }
    }
  }

}

/**
 * Resolves the sandbox endpoint on first use and serializes every call through
 * one terminal session, so concurrent tool calls never interleave TTY output.
 *
 * A terminal that cannot stay connected is the signature of an endpoint bundle
 * the control plane has already replaced, so the bundle is dropped and the
 * command is replayed on a replacement endpoint while nothing has run yet. A
 * command whose run already started is reported instead: shell commands are not
 * idempotent.
 *
 * Resolution runs on its own deadline: the control plane may have to create the
 * sandbox, and the startup budget only starts once the endpoint is known.
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
    options: TerminalExecOptions,
  ): Promise<TerminalExecResult> {
    const run = async (): Promise<TerminalExecResult> => {
      for (let attempt = 0; ; attempt += 1) {
        const operations = await this.resolveOperations(options);
        try {
          return await operations.exec(command, cwd, options);
        } catch (error) {
          if (!isStaleTerminal(error)) throw error;
          this.operations?.dispose();
          this.operations = undefined;
          this.endpoints.invalidate();
          if (attempt >= MAX_ENDPOINT_REFRESHES || !isPreLaunchDrop(error)) throw error;
        }
      }
    };
    const chained = this.tail.then(run, run);
    this.tail = chained.then(
      () => undefined,
      () => undefined,
    );
    return chained;
  }

  private async resolveOperations(options: TerminalExecOptions): Promise<SandboxTerminalOperations> {
    try {
      const resolveDeadline =
        Date.now() + (this.options.resolveTimeoutMs ?? DEFAULT_RESOLVE_TIMEOUT_MS);
      return await bounded(this.operationsFor(), resolveDeadline, options.signal);
    } catch (error) {
      this.operations?.dispose();
      this.operations = undefined;
      this.endpoints.invalidate();
      throw new SandboxTerminalExecutionError("sandbox terminal endpoint resolution failed", {
        execution_id: randomUUID().replace(/-/g, ""), remote_run_dir: "",
        phase: "connecting", started: false, stopped: true,
        error_code: error instanceof Error && error.message === "timeout" ? "startup_timeout" :
          error instanceof Error && error.message === "aborted" ? "aborted" : "connection_lost",
      });
    }
  }

  private async operationsFor(): Promise<SandboxTerminalOperations> {
    if (this.operations === undefined) {
      const endpoint = await this.endpoints.resolve();
      this.operations = new SandboxTerminalOperations(endpoint, this.options);
    }
    return this.operations;
  }

  /**
   * The bundle the operations run against.
   *
   * File operations resolved through this client need the same conversation
   * root the terminal resolves relative paths against, and resolving through
   * the session keeps both on one bundle.
   */
  async endpoint(): Promise<SandboxEndpoint> {
    await this.resolveOperations({ onData: () => undefined });
    return (await this.endpoints.resolve());
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

/** The TTY bridge reports a dropped socket as a bare `closed` rejection. */
function isConnectionLost(error: unknown): boolean {
  return error instanceof Error && error.message === "closed";
}

/** A bounded exchange which ran out of budget without a socket error. */
function isTimeout(error: unknown): boolean {
  return error instanceof Error && error.message === "timeout";
}

/** A connection which dropped in a way the control plane can repair. */
function isStaleTerminal(error: unknown): boolean {
  return error instanceof SandboxTerminalConnectionError ||
    (error instanceof SandboxTerminalExecutionError && error.execution.stopped &&
     error.execution.error_code === "connection_lost");
}

/** A drop before the launch line left no process group behind to repeat. */
function isPreLaunchDrop(error: unknown): boolean {
  return error instanceof SandboxTerminalExecutionError &&
    !error.execution.started && error.execution.stopped;
}

function resolveCwd(workspacePath: string, cwd: string): string {
  if (!cwd || cwd === ".") return workspacePath;
  return posix.resolve(workspacePath, cwd);
}
