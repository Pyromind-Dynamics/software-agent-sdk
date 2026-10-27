import { isAbsolute, posix, relative, resolve, sep } from "node:path";
import { realpath } from "node:fs/promises";

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

  async containsCanonical(path: string): Promise<boolean> {
    const target = await realpath(path);
    for (const { host } of this.roots) {
      if (inside(target, await realpath(host))) return true;
    }
    return false;
  }

  /**
   * The model-visible alias for a host path, so a read which finds nothing on
   * the host is reported in the vocabulary the model used instead of a host
   * path the sandbox cannot resolve.
   */
  resolveHostAlias(path: string): string | undefined {
    if (!isAbsolute(path)) return undefined;
    let match: SandboxPathAlias | undefined;
    for (const root of this.roots) {
      if (!inside(path, root.host)) continue;
      if (match === undefined || root.host.length > match.host.length) match = root;
    }
    if (match === undefined) return undefined;
    const suffix = relative(match.host, path).split(sep).filter(Boolean).join("/");
    return suffix ? `${match.alias}/${suffix}` : match.alias;
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

export interface SandboxPathPolicyInput {
  /** Sandbox-side conversation root, e.g. `/target-workspace/.pyromind-agent/<id>`. */
  workspacePath: string;
  readOnlyRoots: string[];
  skillsDirectory?: string;
  knowledgeRoot?: string;
  /** Sandbox-side Storage mount root; the model addresses it absolutely. */
  storageRoot?: string;
}

/**
 * Lexical POSIX path policy for the sandbox execution plane.
 *
 * The local `WorkspaceAccessPolicy` canonicalizes through the host filesystem,
 * which cannot see the sandbox. This mirrors its model-visible vocabulary
 * (relative paths start at the conversation root, `public_data/` is writable,
 * `.agents/skills/` and `knowledge/` are read-only aliases) using pure string
 * arithmetic so it works without any filesystem access.
 *
 * The user Storage mount is addressed by its absolute path. It used to be
 * reachable through a `storage/` symlink in the workspace; that alias is gone,
 * because one file then had two spellings and every path handed to a platform
 * tool had to drop the prefix by hand.
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
    if (!input) throw pathScopeError(operation, this.storageRoot);
    const target = posix.resolve(this.resolveInput(input));
    if (this.insideAgentArea(target)) {
      const refusal = this.agentAreaRefusal(target);
      if (refusal !== undefined) {
        throw new Error(`PATH_SCOPE_ERROR: ${target} is ${refusal}`);
      }
      if (operation === "write" && !insideSandboxPath(target, this.publicDataPath)) {
        // The conversation's own directories are readable, but the agent only
        // writes public_data/; tool staging is written by the tools themselves.
        throw pathScopeError(operation, this.storageRoot);
      }
    }
    const storageRoots = this.storageRoot ? [this.storageRoot] : [];
    const allowedRoots =
      operation === "write"
        ? [...storageRoots, this.publicDataPath]
        : [...storageRoots, this.publicDataPath, ...this.readOnlyRoots];
    if (!allowedRoots.some((root) => insideSandboxPath(target, root))) {
      throw pathScopeError(operation, this.storageRoot);
    }
    return target;
  }

  /** True for the mount's agent directory: this workspace and its siblings. */
  private insideAgentArea(target: string): boolean {
    return insideSandboxPath(target, posix.dirname(this.workspacePath));
  }

  /**
   * Why a path inside the agent directory stays closed, or undefined when it is
   * reachable.
   *
   * This conversation's own directory is fair game: `public_data/`, the tool
   * staging under `data_preparation/`, and the product state all live there.
   * The runner's `pi/` bookkeeping and the loose files at the workspace root
   * are not, and a sibling conversation is never reachable at all.
   */
  private agentAreaRefusal(target: string): string | undefined {
    if (!insideSandboxPath(target, this.workspacePath)) {
      return "inside another conversation's workspace";
    }
    if (insideSandboxPath(target, posix.join(this.workspacePath, RUNNER_STATE_DIRNAME))) {
      return `the runner's session state (${RUNNER_STATE_DIRNAME}/)`;
    }
    const relative = posix.relative(this.workspacePath, target);
    if (!relative.includes("/")) return "conversation metadata";
    return undefined;
  }

  private resolveInput(input: string): string {
    const parts = input.split("/").filter((part) => part !== "" && part !== ".");
    if (!input.startsWith("/") && parts[0] === ".agents" && parts[1] === "skills") {
      if (!this.skillsDirectory) {
        throw new Error(
          "PATH_SCOPE_ERROR: .agents/skills/ is not configured for this Pi session",
        );
      }
      return posix.join(this.skillsDirectory, ...parts.slice(2));
    }
    if (!input.startsWith("/") && parts[0] === "knowledge") {
      if (!this.knowledgeRoot) {
        throw new Error(
          "PATH_SCOPE_ERROR: knowledge/ is not configured for this Pi session",
        );
      }
      return posix.join(this.knowledgeRoot, ...parts.slice(1));
    }
    if (!input.startsWith("/") && parts[0] === LEGACY_STORAGE_ALIAS) {
      throw new Error(legacyStorageAliasError(this.storageRoot));
    }
    return input.startsWith("/") ? input : posix.join(this.workspacePath, input);
  }
}

export function insideSandboxPath(target: string, root: string): boolean {
  if (target === root) return true;
  return target.startsWith(root.endsWith("/") ? root : `${root}/`);
}

/** The removed workspace symlink; kept only so a stale habit gets a clear error. */
const LEGACY_STORAGE_ALIAS = "storage";

/** The runner's own bookkeeping inside the conversation workspace. */
const RUNNER_STATE_DIRNAME = "pi";

function legacyStorageAliasError(storageRoot: string | undefined): string {
  const target = storageRoot
    ? `user Storage is addressed by its absolute path (${storageRoot}/<path>)`
    : "this session has no Storage mount";
  return `PATH_SCOPE_ERROR: the storage/ alias was removed; ${target}`;
}

function pathScopeError(
  operation: SandboxPathOperation,
  storageRoot: string | undefined,
): Error {
  const mount = storageRoot
    ? `the user Storage mount (${storageRoot})`
    : "the user Storage mount";
  const scope =
    operation === "write"
      ? `write and edit paths must stay within public_data/ or ${mount}`
      : `read paths must stay within public_data/, ${mount}, configured skill directories, or knowledge/`;
  return new Error(`PATH_SCOPE_ERROR: ${scope}`);
}
