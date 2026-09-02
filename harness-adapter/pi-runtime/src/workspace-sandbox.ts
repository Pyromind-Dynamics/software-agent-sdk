import { mkdir, readdir, realpath } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import {
  SandboxManager,
  type SandboxRuntimeConfig,
} from "@anthropic-ai/sandbox-runtime";
import {
  createLocalBashOperations,
  type BashOperations,
} from "@earendil-works/pi-coding-agent";

interface SandboxController {
  checkDependencies(): boolean;
  initialize(config: SandboxRuntimeConfig): Promise<void>;
  isSandboxingEnabled(): boolean;
  updateConfig(config: SandboxRuntimeConfig): void;
  wrapWithSandbox(
    command: string,
    binShell?: string,
    customConfig?: Partial<SandboxRuntimeConfig>,
    abortSignal?: AbortSignal,
  ): Promise<string>;
}

interface WorkspaceSandboxDependencies {
  controller?: SandboxController;
  localOperations?: BashOperations;
  userHome?: string;
}

/**
 * Run Pi terminal commands with OS-enforced workspace write and user-data read
 * isolation. System executables and libraries remain readable so workspace
 * scripts can use the installed Python/Node runtimes.
 */
export function createWorkspaceSandboxedBashOperations(
  workspaceRoot: string,
  dependencies: WorkspaceSandboxDependencies = {},
): BashOperations {
  const workspace = resolve(workspaceRoot);
  const controller = dependencies.controller ?? SandboxManager;
  const localOperations =
    dependencies.localOperations ?? createLocalBashOperations();
  const userHome = resolve(dependencies.userHome ?? homedir());
  let initialization: Promise<void> | undefined;
  let sandboxBroken = false;

  async function configuration(): Promise<SandboxRuntimeConfig> {
    const canonicalWorkspace = await realpath(workspace);
    const temporary = join(canonicalWorkspace, ".tmp");
    await mkdir(temporary, { recursive: true });
    return {
      network: {
        allowedDomains: [],
        deniedDomains: [],
        allowUnixSockets: [],
        allowLocalBinding: false,
      },
      filesystem: {
        denyRead: await workspaceUserDataDenyPaths(
          canonicalWorkspace,
          userHome,
        ),
        allowWrite: [canonicalWorkspace],
        denyWrite: [join(canonicalWorkspace, "pi", "terminal-output")],
      },
    };
  }

  async function prepare(): Promise<SandboxRuntimeConfig> {
    const config = await configuration();
    if (!initialization) {
      initialization = (async () => {
        if (!controller.checkDependencies()) {
          throw new Error("sandbox runtime dependencies are unavailable");
        }
        await controller.initialize(config);
        if (!controller.isSandboxingEnabled()) {
          throw new Error("sandbox runtime did not enable isolation");
        }
      })();
    }
    await initialization;
    controller.updateConfig(config);
    return config;
  }

  return {
    async exec(command, cwd, options) {
      const resolvedCwd = resolve(cwd);
      if (!inside(resolvedCwd, workspace)) {
        throw new Error(
          "WORKSPACE_SCOPE_ERROR: terminal cwd must remain inside the conversation workspace",
        );
      }
      assertNoDirectoryChange(command);
      const temporary = join(workspace, ".tmp");
      await mkdir(temporary, { recursive: true });
      const localEnv = {
        ...options.env,
        TMPDIR: temporary,
        TMP: temporary,
        TEMP: temporary,
      };
      if (sandboxBroken) {
        return localOperations.exec(command, workspace, { ...options, env: localEnv });
      }
      try {
        const config = await prepare();
        const wrapped = await controller.wrapWithSandbox(
          command,
          undefined,
          config,
          options.signal,
        );
        return await localOperations.exec(wrapped, workspace, {
          ...options,
          env: localEnv,
        });
      } catch (error) {
        if (
          error instanceof Error &&
          (error.message.startsWith("WORKSPACE_") ||
            error.message.startsWith("sandboxed command"))
        ) {
          throw error;
        }
        // Sandbox setup can be permanently unavailable on a host (missing
        // OS support). Commands must keep running unsandboxed instead of
        // failing every call; the cwd/temp constraints above still apply.
        sandboxBroken = true;
        console.error(
          `[workspace-sandbox] ${error instanceof Error ? error.message : "unknown error"}; falling back to unsandboxed execution`,
        );
        return localOperations.exec(command, workspace, { ...options, env: localEnv });
      }
    },
  };
}

export function assertNoDirectoryChange(command: string): void {
  const directoryChange =
    /(^|[\s;&|()])(?:(?:builtin|command)\s+)?(?:cd|pushd|popd)(?=$|[\s;&|()])/;
  if (directoryChange.test(command)) {
    throw new Error(
      "WORKSPACE_SCOPE_ERROR: terminal directory changes are disabled; cwd is already the conversation workspace",
    );
  }
}

/**
 * The sandbox runtime uses deny-only read rules. Build those rules by denying
 * every sibling along the path from the user's home to this conversation,
 * leaving only the conversation branch readable.
 */
export async function workspaceUserDataDenyPaths(
  workspaceRoot: string,
  userHome: string = homedir(),
): Promise<string[]> {
  const workspace = await realpath(resolve(workspaceRoot));
  const home = await realpath(resolve(userHome));
  const boundary = inside(workspace, home) ? home : resolve(workspace, "..");
  const branch = relative(boundary, workspace).split(sep).filter(Boolean);
  const denied: string[] = [];
  let current = boundary;

  for (const selected of branch) {
    for (const entry of await readdir(current, { withFileTypes: true })) {
      if (entry.name !== selected) denied.push(resolve(current, entry.name));
    }
    current = resolve(current, selected);
  }
  return denied;
}

function inside(target: string, root: string): boolean {
  const rel = relative(root, target);
  return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel));
}
