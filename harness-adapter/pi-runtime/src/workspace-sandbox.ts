import { readdir, realpath, stat } from "node:fs/promises";
import { homedir } from "node:os";
import {
  delimiter,
  basename,
  dirname,
  parse,
  resolve,
  sep,
} from "node:path";
import { fileURLToPath } from "node:url";
import {
  SandboxManager,
  type SandboxRuntimeConfig,
} from "@anthropic-ai/sandbox-runtime";
import {
  createLocalBashOperations,
  type BashOperations,
} from "@earendil-works/pi-coding-agent";
import { inside, WorkspaceAccessPolicy } from "./workspace-policy.js";

interface SandboxController {
  checkDependencies(ripgrepConfig?: { command: string; args?: string[] }): boolean;
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
  runtimeReadRoots?: string[];
}

export type PiTerminalBackend = "os-sandbox";

interface WorkspaceSandboxHandle {
  operations: BashOperations;
  initialize(): Promise<void>;
}

export async function createWorkspaceBashOperations(
  backend: PiTerminalBackend,
  policy: WorkspaceAccessPolicy,
  dependencies: WorkspaceSandboxDependencies = {},
): Promise<BashOperations> {
  if (backend !== "os-sandbox") {
    throw new Error("WORKSPACE_SANDBOX_UNAVAILABLE: os-sandbox is required");
  }
  const sandbox = createWorkspaceSandbox(policy, dependencies);
  await sandbox.initialize();
  return sandbox.operations;
}

export function createWorkspaceSandboxedBashOperations(
  policy: WorkspaceAccessPolicy,
  dependencies: WorkspaceSandboxDependencies = {},
): BashOperations {
  return createWorkspaceSandbox(policy, dependencies).operations;
}

function createWorkspaceSandbox(
  policy: WorkspaceAccessPolicy,
  dependencies: WorkspaceSandboxDependencies,
): WorkspaceSandboxHandle {
  const controller = dependencies.controller ?? SandboxManager;
  const localOperations = dependencies.localOperations ?? createLocalBashOperations();
  const userHome = resolve(dependencies.userHome ?? homedir());
  let initialization: Promise<void> | undefined;

  async function configuration(): Promise<SandboxRuntimeConfig> {
    const runtimeReadRoots = dependencies.runtimeReadRoots
      ?? await configuredRuntimeReadRoots();
    const denyRead = await workspacePolicyDenyPaths(
      policy,
      userHome,
      runtimeReadRoots,
    );
    return {
      network: {
        allowedDomains: [],
        deniedDomains: [],
        allowUnixSockets: [],
        allowLocalBinding: false,
      },
      filesystem: {
        denyRead,
        allowWrite: [...policy.terminalWriteRoots],
        denyWrite: await workspacePolicyDenyWritePaths(
          policy,
          userHome,
        ),
      },
      // sandbox-runtime 0.0.26 incorrectly requires rg on macOS even though
      // only its Linux filesystem implementation invokes it. Naming the
      // default command as an explicit override skips that false-negative;
      // if a future macOS implementation starts using rg, execution still
      // fails closed when the command is unavailable.
      ...(process.platform === "darwin"
        ? { ripgrep: { command: "rg" } }
        : {}),
    };
  }

  async function prepare(): Promise<SandboxRuntimeConfig> {
    const config = await configuration();
    if (!initialization) {
      initialization = (async () => {
        if (!controller.checkDependencies(config.ripgrep)) {
          const required = process.platform === "linux"
            ? "rg, bwrap, and socat"
            : process.platform === "darwin"
              ? "/usr/bin/sandbox-exec"
              : "Linux or macOS";
          throw new Error(
            `sandbox runtime dependencies are unavailable; required: ${required}`,
          );
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
    async initialize() {
      try {
        await prepare();
      } catch (error) {
        throw sandboxUnavailable(error);
      }
    },
    operations: {
      async exec(command, cwd, options) {
        await assertWorkspaceCwd(cwd, policy.workspaceRoot);
        try {
          const config = await prepare();
          const wrapped = await wrapWithPrivateTemp(
            controller,
            command,
            config,
            policy.terminalTempRoot,
            options.signal,
          );
          return await localOperations.exec(wrapped, policy.workspaceRoot, {
            ...options,
            env: {
              ...options.env,
              TMPDIR: policy.terminalTempRoot,
              TMP: policy.terminalTempRoot,
              TEMP: policy.terminalTempRoot,
            },
          });
        } catch (error) {
          if (
            error instanceof Error
            && (error.message.startsWith("WORKSPACE_")
              || error.message.startsWith("sandboxed command"))
          ) {
            throw error;
          }
          throw sandboxUnavailable(error);
        }
      },
    },
  };
}

async function wrapWithPrivateTemp(
  controller: SandboxController,
  command: string,
  config: SandboxRuntimeConfig,
  temporary: string,
  signal?: AbortSignal,
): Promise<string> {
  const previous = process.env.CLAUDE_TMPDIR;
  process.env.CLAUDE_TMPDIR = temporary;
  try {
    return await controller.wrapWithSandbox(
      command,
      undefined,
      config,
      signal,
    );
  } finally {
    if (previous === undefined) {
      delete process.env.CLAUDE_TMPDIR;
    } else {
      process.env.CLAUDE_TMPDIR = previous;
    }
  }
}

async function assertWorkspaceCwd(cwd: string, workspace: string): Promise<void> {
  let canonicalCwd: string;
  try {
    canonicalCwd = await realpath(resolve(cwd));
  } catch {
    throw new Error(
      "WORKSPACE_SCOPE_ERROR: terminal cwd must be an existing conversation root",
    );
  }
  if (canonicalCwd !== workspace) {
    throw new Error(
      "WORKSPACE_SCOPE_ERROR: every terminal call must start at the conversation root",
    );
  }
}

function sandboxUnavailable(error: unknown): Error {
  if (
    error instanceof Error
    && error.message.startsWith("WORKSPACE_SANDBOX_UNAVAILABLE:")
  ) {
    return error;
  }
  const reason = error instanceof Error ? error.message : "unknown error";
  return new Error(`WORKSPACE_SANDBOX_UNAVAILABLE: ${reason}`);
}

/**
 * sandbox-runtime exposes deny-only read rules. Compile the policy's allow
 * roots into deny entries at the three protected boundaries: the user's home,
 * the conversations directory, and the repository that owns skills/knowledge.
 */
export async function workspacePolicyDenyPaths(
  policy: WorkspaceAccessPolicy,
  userHome: string = homedir(),
  runtimeReadRoots: string[] = [],
): Promise<string[]> {
  const canonicalHome = await realpath(resolve(userHome)).catch(() => undefined);
  const allowed = (await existingCanonicalPaths([
    ...policy.terminalReadRoots,
    ...runtimeReadRoots,
  ])).filter((root) => root !== canonicalHome);
  const boundaries = await protectedBoundaries(policy, userHome);
  const denied = new Set<string>();
  for (const boundary of boundaries) {
    for (const path of await denyUnapprovedEntries(boundary, allowed)) {
      denied.add(path);
    }
  }
  return [...denied].sort();
}

/**
 * macOS sandbox-runtime grants its system TMPDIR parent write access. Explicit
 * deny rules keep protected workspace and repository trees closed even when a
 * conversation is created below that directory. Immediate-child globs prevent
 * creating new siblings beside an allowed branch; existing unapproved entries
 * are denied recursively.
 */
export async function workspacePolicyDenyWritePaths(
  policy: WorkspaceAccessPolicy,
  userHome: string = homedir(),
): Promise<string[]> {
  const allowed = policy.terminalWriteRoots;
  const boundaries = await protectedBoundaries(policy, userHome);
  const denied = new Set<string>();
  for (const boundary of boundaries) {
    for (const path of await denyWritesOutsideAllowedRoots(boundary, allowed)) {
      denied.add(path);
    }
  }
  return [...denied].sort();
}

async function protectedBoundaries(
  policy: WorkspaceAccessPolicy,
  userHome: string,
): Promise<string[]> {
  const candidates = [
    resolve(userHome),
    dirname(policy.workspaceRoot),
    resourceBoundary([...policy.readOnlyRoots]),
  ].filter((value): value is string => Boolean(value));
  const existing = await existingCanonicalPaths(candidates);
  const filesystemRoot = parse(policy.workspaceRoot).root;
  return existing
    .filter((boundary) => boundary !== filesystemRoot)
    .filter((boundary, index, values) =>
      !values.some((other, otherIndex) =>
        otherIndex !== index && inside(boundary, other) && boundary !== other
      )
    );
}

async function denyUnapprovedEntries(
  boundary: string,
  allowedRoots: string[],
): Promise<string[]> {
  const denied: string[] = [];

  async function visit(directory: string): Promise<void> {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = resolve(directory, entry.name);
      if (allowedRoots.some((allowed) => inside(path, allowed))) continue;
      const leadsToAllowed = allowedRoots.some((allowed) => inside(allowed, path));
      if (leadsToAllowed && entry.isDirectory()) {
        await visit(path);
      } else {
        denied.push(path);
      }
    }
  }

  await visit(boundary);
  return denied;
}

async function denyWritesOutsideAllowedRoots(
  boundary: string,
  allowedRoots: readonly string[],
): Promise<string[]> {
  const denied: string[] = [];

  async function visit(directory: string): Promise<void> {
    denied.push(`${directory}${sep}*`);
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = resolve(directory, entry.name);
      if (allowedRoots.some((allowed) => inside(path, allowed))) continue;
      const leadsToAllowed = allowedRoots.some((allowed) => inside(allowed, path));
      if (leadsToAllowed && entry.isDirectory()) {
        await visit(path);
      } else {
        denied.push(path);
      }
    }
  }

  await visit(boundary);
  return denied;
}

function resourceBoundary(readOnlyRoots: string[]): string | undefined {
  if (readOnlyRoots.length === 0) return undefined;
  const candidates = readOnlyRoots.map((root) => {
    const marker = `${sep}.agents${sep}skills${sep}`;
    const index = root.indexOf(marker);
    if (index >= 0) return root.slice(0, index);
    return dirname(root);
  });
  return candidates.reduce((left, right) => commonAncestor(left, right));
}

function commonAncestor(left: string, right: string): string {
  let candidate = resolve(left);
  const target = resolve(right);
  while (!inside(target, candidate)) {
    const parent = dirname(candidate);
    if (parent === candidate) return parent;
    candidate = parent;
  }
  return candidate;
}

export async function configuredRuntimeReadRoots(): Promise<string[]> {
  const pathRoots = (process.env.PATH ?? "")
    .split(delimiter)
    .filter(Boolean)
    .map(runtimeRootForBinDirectory);
  return existingCanonicalPaths([
    ...pathRoots,
    runtimeRootForBinDirectory(dirname(process.execPath)),
    ...["SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS"]
      .map((name) => process.env[name])
      .filter((value): value is string => Boolean(value)),
    ...piRuntimeReadRoots(),
  ]);
}

/**
 * The pi-runtime installation holds sandbox-runtime's apply-seccomp binary,
 * which the sandboxed command itself must exec inside the bwrap namespace.
 * Compiled deny rules mount tmpfs over unapproved read paths, so without this
 * root the namespace hides the very binary the wrapper then fails to exec
 * (ENOENT, exit 127) — every terminal command would fail.
 */
export function piRuntimeReadRoots(): string[] {
  const configured = process.env.PYROMIND_PI_RUNTIME?.trim();
  if (configured) {
    return [resolve(dirname(dirname(configured)))];
  }
  // Local development layout: <pi-runtime>/src|dist/workspace-sandbox.js
  return [resolve(dirname(dirname(fileURLToPath(import.meta.url))))];
}

function runtimeRootForBinDirectory(path: string): string {
  const directory = resolve(path);
  const parent = dirname(directory);
  const parentName = basename(parent).toLowerCase();
  if (
    basename(directory) === "bin"
    && (parentName === ".venv"
      || parentName === "venv"
      || parentName === "env"
      || parentName.endsWith("-venv")
      || parent.includes(`${sep}.nvm${sep}versions${sep}node${sep}`))
  ) {
    return parent;
  }
  return directory;
}

async function existingCanonicalPaths(paths: string[]): Promise<string[]> {
  const canonical: string[] = [];
  for (const path of paths) {
    try {
      await stat(path);
      const resolved = await realpath(path);
      if (resolved !== parse(resolved).root) canonical.push(resolved);
    } catch {
      // Optional runtime roots may not exist in every image.
    }
  }
  return [...new Set(canonical)];
}
