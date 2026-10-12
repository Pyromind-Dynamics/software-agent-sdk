import { isAbsolute, posix, resolve } from "node:path";

import { inside } from "./workspace-policy.js";

export type SandboxPathOperation = "read" | "write";

export interface SandboxPathAlias {
  /** Resource root as it exists on the host, e.g. `/repo/.agents/skills`. */
  host: string;
  /** Runtime alias the execution plane resolves, e.g. `.agents/skills`. */
  alias: string;
}

/**
 * Agent-host copies of the read-only resource roots a sandbox mirrors.
 *
 * Skill and knowledge files are uploaded into the sandbox from these very
 * directories, so reading one from the host returns the same bytes without a
 * sandbox round trip, and it still resolves when the sandbox copy is missing
 * because the container predates the last upload. Reads only: writes to these
 * roots stay with `SandboxPathPolicy`, which rejects them.
 */
export class SandboxReadMirror {
  private readonly roots: readonly SandboxPathAlias[];

  constructor(roots: readonly SandboxPathAlias[]) {
    // A nested root has to win over the directory that contains it.
    this.roots = [...roots].sort(
      (left, right) => right.alias.split("/").length - left.alias.split("/").length,
    );
  }

  /** Host file for a model-supplied alias, e.g. `.agents/skills/x/notes.md`. */
  resolveAlias(input: string): string | undefined {
    if (isAbsolute(input)) return undefined;
    const parts = input.split("/").filter((part) => part !== "" && part !== ".");
    for (const { alias, host } of this.roots) {
      const aliasParts = alias.split("/");
      if (parts.length < aliasParts.length) continue;
      if (!aliasParts.every((part, index) => parts[index] === part)) continue;
      const target = resolve(host, ...parts.slice(aliasParts.length));
      return inside(target, host) ? target : undefined;
    }
    return undefined;
  }

  /** The same host file, for a path a caller already resolved into a root. */
  resolveHost(path: string): string | undefined {
    if (!isAbsolute(path)) return undefined;
    return this.roots.some(({ host }) => inside(path, host)) ? path : undefined;
  }
}

/**
 * Rewrite host resource roots in the system prompt to their sandbox-side
 * aliases so the model never sees paths that only exist on the host.
 *
 * Aliases must be applied longest-host-first: a nested root would otherwise be
 * partially rewritten by its parent prefix.
 */
export function applySandboxPathAliases(
  systemPrompt: string,
  aliases: readonly SandboxPathAlias[],
): string {
  let rewritten = systemPrompt;
  for (const { host, alias } of aliases) {
    rewritten = rewritten.split(host).join(alias);
  }
  return rewritten;
}

/** Replaces the host working directory pi appends to the system prompt. */
const SANDBOX_WORKING_DIRECTORY =
  "Current working directory: . (the conversation root inside the sandbox " +
  "execution environment; resolve every path relative to it, using relative " +
  "paths such as storage/..., public_data/... and .agents/skills/...)";

/**
 * Point the model at the sandbox-side conversation root instead of the host
 * directory pi appended.
 *
 * The execution plane resolves relative paths against the conversation root,
 * so a host path in the prompt sends terminal commands to a directory that
 * does not exist inside the sandbox.
 */
export function applySandboxWorkingDirectory(
  systemPrompt: string,
  workspaceRoot: string,
): string {
  return systemPrompt.replace(
    `Current working directory: ${workspaceRoot.replace(/\\/g, "/")}`,
    SANDBOX_WORKING_DIRECTORY,
  );
}

export interface SandboxPathPolicyInput {
  /** Sandbox-side conversation root, e.g. `/target-workspace/.pyromind-agent/<id>`. */
  workspacePath: string;
  readOnlyRoots: string[];
  skillsDirectory?: string;
  knowledgeRoot?: string;
  /** Sandbox-side Storage mount root; the model addresses it as `storage/`. */
  storageRoot?: string;
}

/**
 * Lexical POSIX path policy for the sandbox execution plane.
 *
 * The local `WorkspaceAccessPolicy` canonicalizes through the host filesystem,
 * which cannot see the sandbox. This mirrors its model-visible vocabulary
 * (relative paths start at the conversation root, `public_data/` is writable,
 * `.agents/skills/`, `knowledge/`, and `storage/` are read-only aliases) using
 * pure string arithmetic so it works without any filesystem access.
 */
export class SandboxPathPolicy {
  private constructor(
    readonly workspacePath: string,
    readonly publicDataPath: string,
    readonly readOnlyRoots: readonly string[],
    readonly skillsDirectory: string | undefined,
    readonly knowledgeRoot: string | undefined,
    readonly storageRoot: string | undefined,
  ) {}

  static create(input: SandboxPathPolicyInput): SandboxPathPolicy {
    const workspacePath = posix.resolve(input.workspacePath);
    const readOnlyRoots = input.readOnlyRoots.map((root) => posix.resolve(root));
    const skillsDirectory = input.skillsDirectory
      ? posix.resolve(input.skillsDirectory)
      : undefined;
    const knowledgeRoot = input.knowledgeRoot
      ? posix.resolve(input.knowledgeRoot)
      : undefined;
    const storageRoot = input.storageRoot
      ? posix.resolve(input.storageRoot)
      : undefined;
    return new SandboxPathPolicy(
      workspacePath,
      posix.join(workspacePath, "public_data"),
      [
        ...new Set([
          ...readOnlyRoots,
          ...(skillsDirectory ? [skillsDirectory] : []),
          ...(knowledgeRoot ? [knowledgeRoot] : []),
        ]),
      ],
      skillsDirectory,
      knowledgeRoot,
      storageRoot,
    );
  }

  resolvePath(input: string, operation: SandboxPathOperation): string {
    if (!input) throw pathScopeError(operation);
    const resolved = this.resolveInput(input);
    const target = posix.resolve(resolved.path);
    const allowedRoots =
      operation === "write"
        ? [this.publicDataPath]
        : [this.publicDataPath, ...this.readOnlyRoots];
    const storageRead =
      operation === "read" &&
      resolved.storageAlias &&
      this.storageRoot !== undefined &&
      insideSandboxPath(target, this.storageRoot);
    if (
      !storageRead &&
      !allowedRoots.some((root) => insideSandboxPath(target, root))
    ) {
      throw pathScopeError(operation);
    }
    return target;
  }

  private resolveInput(input: string): {
    path: string;
    storageAlias: boolean;
  } {
    const parts = input.split("/").filter((part) => part !== "" && part !== ".");
    if (!input.startsWith("/") && parts[0] === ".agents" && parts[1] === "skills") {
      if (!this.skillsDirectory) {
        throw new Error(
          "PATH_SCOPE_ERROR: .agents/skills/ is not configured for this Pi session",
        );
      }
      return {
        path: posix.join(this.skillsDirectory, ...parts.slice(2)),
        storageAlias: false,
      };
    }
    if (!input.startsWith("/") && parts[0] === "knowledge") {
      if (!this.knowledgeRoot) {
        throw new Error(
          "PATH_SCOPE_ERROR: knowledge/ is not configured for this Pi session",
        );
      }
      return {
        path: posix.join(this.knowledgeRoot, ...parts.slice(1)),
        storageAlias: false,
      };
    }
    if (!input.startsWith("/") && parts[0] === "storage") {
      if (!this.storageRoot) {
        throw new Error(
          "PATH_SCOPE_ERROR: storage/ requires Storage to be mounted for this Pi session",
        );
      }
      return {
        path: posix.join(this.storageRoot, ...parts.slice(1)),
        storageAlias: true,
      };
    }
    return {
      path: input.startsWith("/") ? input : posix.join(this.workspacePath, input),
      storageAlias: false,
    };
  }
}

export function insideSandboxPath(target: string, root: string): boolean {
  if (target === root) return true;
  return target.startsWith(root.endsWith("/") ? root : `${root}/`);
}

function pathScopeError(operation: SandboxPathOperation): Error {
  const scope =
    operation === "write"
      ? "write and edit paths must stay within public_data/"
      : "read paths must stay within public_data/, storage/, configured skill directories, or knowledge/";
  return new Error(`PATH_SCOPE_ERROR: ${scope}`);
}
