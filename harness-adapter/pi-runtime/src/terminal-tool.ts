import type { AgentTool } from "@earendil-works/pi-agent-core";
import {
  createBashTool,
  type BashOperations,
  type BashToolDetails,
  type BashToolOptions,
} from "@earendil-works/pi-coding-agent";
import { SandboxTerminalExecutionError, type TerminalExecutionEvidence } from "./sandbox-terminal.js";

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
  options: BashToolOptions & { operations: TerminalOperations },
): AgentTool<ReturnType<typeof createBashTool>["parameters"], TerminalToolDetails> {
  const native = createBashTool(cwd, options);
  return {
    ...native,
    name: "terminal",
    label: "terminal",
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
              completed = await options.operations.exec(command, workingDir, execOptions);
              return completed;
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
        return { ...result, details: { ...details(), ...result.details, resultCategory: "success" } };
      } catch (error) {
        return {
          content: [{ type: "text", text: error instanceof Error ? error.message : String(error) }],
          details: { ...details(), resultCategory: "failure" },
        };
      }
    },
  };
}
