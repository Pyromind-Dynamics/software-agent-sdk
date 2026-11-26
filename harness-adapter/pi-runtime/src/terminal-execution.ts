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
  constructor(
    message: string,
    readonly execution: TerminalExecutionEvidence,
    options?: ErrorOptions,
  ) {
    super(
      `${message} (phase=${execution.phase}, started=${execution.started}, stopped=${execution.stopped})`,
      options,
    );
    this.name = "SandboxTerminalExecutionError";
  }
}

export interface TerminalExecOptions {
  onData: (data: Buffer) => void;
  signal?: AbortSignal;
  timeout?: number;
  env?: NodeJS.ProcessEnv;
}

export interface TerminalExecResult {
  exitCode: number | null;
  execution: TerminalExecutionEvidence;
}
