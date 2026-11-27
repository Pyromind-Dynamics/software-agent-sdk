import type { AgentTool } from "@earendil-works/pi-agent-core";
import { StringDecoder } from "node:string_decoder";
import {
  createBashTool,
  type BashOperations,
  type BashToolDetails,
  type BashToolOptions,
} from "@earendil-works/pi-coding-agent";
import { SandboxTerminalExecutionError, type TerminalExecutionEvidence } from "./sandbox-terminal.js";
import { redact } from "./observability/recorder.js";
import { isRecord } from "./protocol.js";

export type TerminalEnvironmentProvider = (signal?: AbortSignal) => Promise<Record<string, string>>;

interface TerminalExecutionResult {
  exitCode: number | null;
  execution?: TerminalExecutionEvidence;
}

interface TerminalOperations {
  exec(...args: Parameters<BashOperations["exec"]>): Promise<TerminalExecutionResult>;
}

export interface TerminalToolDetails extends BashToolDetails {
  cwd: string;
  exitCode?: number | null;
  execution?: TerminalExecutionEvidence;
  error_code?: string;
  timedOut?: boolean;
  cancelled?: boolean;
  resultCategory?: "success" | "failure";
}

// Follow Oh My Pi's result-first shell completion while retaining Pi's native
// tool: https://github.com/can1357/oh-my-pi/blob/2220e58146ebe31508e7a216242840d373e082d0/packages/coding-agent/src/tools/bash.ts
export function createTerminalTool(
  cwd: string,
  options: BashToolOptions & {
    operations: TerminalOperations;
    environmentProvider?: TerminalEnvironmentProvider;
  },
): AgentTool<ReturnType<typeof createBashTool>["parameters"], TerminalToolDetails> {
  const native = createBashTool(cwd, options);
  const timeout = {
    ...native.parameters.properties.timeout,
    description: "Execution timeout in seconds. The backend default applies when omitted.",
  };
  const knownSecrets = new Set<string>();
  return {
    ...native,
    name: "terminal",
    label: "terminal",
    description: "Run a Bash command or script. Each call starts at the workspace root. " +
      "The result contains stdout and stderr, with the tail shown for long output. " +
      "Redirect large results to a workspace file when you need to inspect them later.",
    parameters: {
      ...native.parameters,
      properties: { ...native.parameters.properties, timeout },
    },
    async execute(callId, params, signal, onUpdate) {
      let commandCwd = cwd;
      let completed: TerminalExecutionResult | undefined;
      let executionError: unknown;
      let outputDetails: BashToolDetails | undefined;
      const tool = createBashTool(cwd, {
        ...options,
        operations: {
          async exec(command, workingDir, execOptions) {
            commandCwd = workingDir;
            try {
              if (options.environmentProvider) {
                onUpdate?.({
                  content: [{ type: "text", text: "正在检查平台 SDK 环境…" }],
                  details: { cwd: commandCwd },
                });
              }
              const sdk = await options.environmentProvider?.(execOptions.signal);
              if (sdk && !sdk.PYROMIND_API_KEY) {
                onUpdate?.({
                  content: [{ type: "text", text: sdk.PYROMIND_SDK_PYTHON
                    ? "平台 AK 暂不可用。SDK 解释器可用于本地准备，提交平台任务需要当前账号的 AK。"
                    : "当前终端没有平台 SDK 授权。请检查当前会话的登录状态。" }],
                  details: { cwd: commandCwd },
                });
              }
              const environment = { ...execOptions.env, ...sdk };
              for (const name of ["PYROMIND_API_KEY", "LLM_AUTH_TOKEN", "DF_API_KEY",
                "LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]) {
                if (environment[name]) knownSecrets.add(environment[name]);
              }
              if (environment.PYROMIND_DATAFLOW_PROFILES) {
                try {
                  const profiles: unknown = JSON.parse(environment.PYROMIND_DATAFLOW_PROFILES);
                  if (!isRecord(profiles)) throw new Error();
                  for (const profile of Object.values(profiles)) {
                    if (!isRecord(profile)) throw new Error();
                    if (typeof profile.DF_API_KEY === "string" && profile.DF_API_KEY) {
                      knownSecrets.add(profile.DF_API_KEY);
                    }
                  }
                } catch { throw new Error("Invalid terminal model environment"); }
              }
              const output = protectedOutput(execOptions.onData, [...knownSecrets]);
              try {
                completed = await options.operations.exec(command, workingDir, {
                  ...execOptions,
                  env: environment,
                  onData: output.write,
                });
                return completed;
              } finally { output.finish(); }
            } catch (error) {
              executionError = error;
              throw error;
            }
          },
        },
      });
      const details = (): TerminalToolDetails => {
        const execution = executionError instanceof SandboxTerminalExecutionError
          ? executionError.execution : completed?.execution;
        const timedOut = execution?.error_code === "execution_timeout" ||
          execution?.error_code === "startup_timeout" ||
          (executionError instanceof Error && executionError.message.startsWith("timeout:"));
        return {
          ...outputDetails,
          cwd: commandCwd,
          ...(completed ? { exitCode: completed.exitCode } : {}),
          ...(execution ? { execution } : {}),
          ...(execution?.error_code ? { error_code: execution.error_code } : {}),
          ...(timedOut ? { timedOut: true } : {}),
          ...(signal?.aborted ? { cancelled: true } : {}),
        };
      };
      try {
        const result = await tool.execute(callId, params, signal, (partial) => {
          outputDetails = partial.details;
          onUpdate?.({ ...partial, details: { ...partial.details, cwd: commandCwd } });
        });
        return {
          ...result,
          details: { ...details(), ...result.details, resultCategory: result.isError ? "failure" : "success" },
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: redact(error instanceof Error ? error.message : String(error), [...knownSecrets]) }],
          details: { ...details(), resultCategory: "failure" },
        };
      }
    },
  };
}

function protectedOutput(onData: (data: Buffer) => void, secrets: readonly string[]): {
  write(data: Buffer): void;
  finish(): void;
} {
  if (secrets.length === 0) return { write: onData, finish: () => {} };
  const decoder = new StringDecoder("utf8");
  const window = Math.max(...secrets.map((secret) => secret.length)) - 1;
  let pending = "";
  const emit = (final: boolean) => {
    let end = final ? pending.length : Math.max(0, pending.length - window);
    let previous: number;
    do {
      previous = end;
      for (const secret of secrets) {
        const index = pending.indexOf(secret, Math.max(0, end - secret.length + 1));
        if (index >= 0 && index < end && index + secret.length > end) end = index;
      }
    } while (end !== previous);
    if (end > 0) onData(Buffer.from(redact(pending.slice(0, end), secrets)));
    pending = pending.slice(end);
  };
  return {
    write(data) { pending += decoder.write(data); emit(false); },
    finish() { pending += decoder.end(); emit(true); },
  };
}
