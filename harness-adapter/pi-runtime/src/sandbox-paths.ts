import { posix } from "node:path";

export type SandboxPathOperation = "read" | "write";

export interface SandboxPathAlias {
  /** Resource root as it exists on the host, e.g. `/repo/.agents/skills`. */
  host: string;
  /** Runtime alias the execution plane resolves, e.g. `.agents/skills`. */
  alias: string;
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
