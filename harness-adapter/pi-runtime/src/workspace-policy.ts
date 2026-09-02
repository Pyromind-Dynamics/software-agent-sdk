import { lstat, realpath } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";

export type WorkspacePathOperation = "read" | "write";

export interface WorkspacePolicyInput {
  workspaceRoot: string;
  readOnlyRoots: string[];
  knowledgeRoot?: string;
}

/**
 * One canonical path policy shared by Pi's native file tools and terminal.
 * Relative paths always start at the conversation root. Only public_data is
 * writable; configured skills and knowledge are additional read-only roots.
 */
export class WorkspaceAccessPolicy {
  private constructor(
    readonly workspaceRoot: string,
    readonly publicDataRoot: string,
    readonly terminalTempRoot: string,
    readonly readOnlyRoots: readonly string[],
    readonly knowledgeRoot: string | undefined,
  ) {}

  static async create(input: WorkspacePolicyInput): Promise<WorkspaceAccessPolicy> {
    const workspace = await canonicalDirectory(
      input.workspaceRoot,
      "conversation root",
      true,
    );
    const publicData = await canonicalDirectory(
      resolve(workspace, "public_data"),
      "public_data",
      true,
    );
    const terminalTemp = await canonicalDirectory(
      resolve(workspace, "pi", "terminal-output"),
      "terminal-output",
      true,
    );
    if (!inside(publicData, workspace) || !inside(terminalTemp, workspace)) {
      throw new Error(
        "PI_WORKSPACE_INVALID: public_data and terminal-output must stay inside the conversation root",
      );
    }

    const readOnlyRoots = await Promise.all(
      input.readOnlyRoots.map((root) => canonicalDirectory(root, "read-only root")),
    );
    const knowledge = input.knowledgeRoot
      ? await canonicalDirectory(input.knowledgeRoot, "knowledge root")
      : undefined;
    const allReadOnlyRoots = [...new Set([
      ...readOnlyRoots,
      ...(knowledge ? [knowledge] : []),
    ])];
    return new WorkspaceAccessPolicy(
      workspace,
      publicData,
      terminalTemp,
      allReadOnlyRoots,
      knowledge,
    );
  }

  get terminalReadRoots(): readonly string[] {
    return [this.publicDataRoot, this.terminalTempRoot, ...this.readOnlyRoots];
  }

  get terminalWriteRoots(): readonly string[] {
    return [this.publicDataRoot, this.terminalTempRoot];
  }

  async resolvePath(input: string, operation: WorkspacePathOperation): Promise<string> {
    if (!input) throw pathScopeError(operation);
    const candidate = this.resolveInput(input);
    const target = await canonicalCandidate(candidate);
    const allowedRoots = operation === "write"
      ? [this.publicDataRoot]
      : [this.publicDataRoot, ...this.readOnlyRoots];
    if (!allowedRoots.some((root) => inside(target, root))) {
      throw pathScopeError(operation);
    }
    return target;
  }

  private resolveInput(input: string): string {
    if (!isAbsolute(input) && input.split(/[\\/]/)[0] === "knowledge") {
      if (!this.knowledgeRoot) {
        throw new Error(
          "PATH_SCOPE_ERROR: knowledge/ is not configured for this Pi session",
        );
      }
      const suffix = input.split(/[\\/]/).filter(Boolean).slice(1);
      return resolve(this.knowledgeRoot, ...suffix);
    }
    return isAbsolute(input)
      ? resolve(input)
      : resolve(this.workspaceRoot, input);
  }
}

export function inside(target: string, root: string): boolean {
  const rel = relative(root, target);
  return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel));
}

async function canonicalDirectory(
  input: string,
  label: string,
  rejectSymbolicLink = false,
): Promise<string> {
  const lexical = resolve(input);
  const info = await lstat(lexical).catch((error: unknown) => {
    throw new Error(
      `PI_WORKSPACE_INVALID: ${label} does not exist: ${lexical}`,
      { cause: error },
    );
  });
  if (!info.isDirectory() || (rejectSymbolicLink && info.isSymbolicLink())) {
    throw new Error(`PI_WORKSPACE_INVALID: ${label} must be a real directory: ${lexical}`);
  }
  return realpath(lexical);
}

async function canonicalCandidate(input: string): Promise<string> {
  let current = resolve(input);
  const missing: string[] = [];
  while (true) {
    try {
      await lstat(current);
      const canonicalParent = await realpath(current).catch((error: unknown) => {
        throw new Error(
          `PATH_SCOPE_ERROR: path contains an unresolved symbolic link: ${input}`,
          { cause: error },
        );
      });
      return resolve(canonicalParent, ...missing.reverse());
    } catch (error) {
      if (!isMissing(error)) throw error;
      const parent = dirname(current);
      if (parent === current) throw pathScopeError("read");
      missing.push(current.slice(parent.length + (parent.endsWith(sep) ? 0 : 1)));
      current = parent;
    }
  }
}

function isMissing(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error
    && (error as { code?: unknown }).code === "ENOENT";
}

function pathScopeError(operation: WorkspacePathOperation): Error {
  const scope = operation === "write"
    ? "write and edit paths must stay within public_data/"
    : "read paths must stay within public_data/, an advertised skill, or knowledge/";
  return new Error(`PATH_SCOPE_ERROR: ${scope}`);
}
