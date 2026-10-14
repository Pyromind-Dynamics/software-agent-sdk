import { realpath } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { TSchema } from "@earendil-works/pi-ai";
import type { ExecutionAccess } from "./workflow-execution.js";
import type { SandboxReadMirror } from "./sandbox-paths.js";
import { isRecord } from "./protocol.js";
import { safeShellEnvironment } from "./tools.js";

const execFileAsync = promisify(execFile);
const quote = (value: string): string => "'" + value.replaceAll("'", "'\"'\"'") + "'";

// Fixed read-only program. All model input is JSON data, never shell source.
// Refuse symlinks escaping the selected root, including symlinked directories.
export const RETRIEVAL_SCRIPT = String.raw`
import os, sys, json, re, fnmatch
q=json.loads(sys.argv[1]); root=os.path.realpath(q['root']); out=[]; count=0; truncated=False
if not any(os.path.commonpath([os.path.realpath(r),root]) == os.path.realpath(r) for r in q['allowedRoots']): raise PermissionError('search path is outside authorized roots')
def allowed(p):
    return os.path.commonpath([root,os.path.realpath(p)]) == root
def emit(s):
    global truncated
    if len(out) >= q['limit'] or sum(map(len,out))+len(s)>24000:
        truncated=True; return False
    out.append(s); return True
regex=re.compile(re.escape(q.get('pattern','')) if q.get('literal') else q.get('pattern',''), re.I if q.get('ignoreCase') else 0)
if not os.path.exists(root): raise FileNotFoundError(q['label'])
if q['kind']=='verify':
    print(json.dumps({'lines':[],'truncated':False})); sys.exit(0)
if q['kind']=='ls':
    if not os.path.isdir(root): raise NotADirectoryError(q['label'])
    for name in sorted(os.listdir(root)):
        p=os.path.join(root,name)
        if allowed(p) and not emit(name+('/' if os.path.isdir(p) else '')): break
else:
    if os.path.isfile(root): files=[root]
    else:
        files=[]
        for base,dirs,names in os.walk(root,followlinks=False):
            dirs[:]=sorted(d for d in dirs if not d.startswith('.') and allowed(os.path.join(base,d)) and not os.path.islink(os.path.join(base,d)))
            if os.path.relpath(base,root).count(os.sep)>=20: dirs[:]=[]
            for name in sorted(names):
                p=os.path.join(base,name)
                if allowed(p): files.append(p)
                if len(files)>=5000: truncated=True; dirs[:]=[]; break
            if len(files)>=5000: break
    for p in files:
        rel=os.path.relpath(p,root) if os.path.isdir(root) else os.path.basename(p)
        if not fnmatch.fnmatch(rel,q.get('glob','*')) and not fnmatch.fnmatch(os.path.basename(p),q.get('glob','*')): continue
        label=q['label'].rstrip('/')+'/'+rel if os.path.isdir(root) else q['label']
        if q['kind']=='find':
            if not emit(label): break
        else:
            try:
                if os.path.getsize(p)>1048576: truncated=True; continue
                with open(p,encoding='utf-8') as f:
                    for n,line in enumerate(f,1):
                        if regex.search(line) and not emit(label+':'+str(n)+':'+line.rstrip()[:2000]): break
            except (UnicodeError,PermissionError,OSError): continue
            if len(out)>=q['limit']: break
print(json.dumps({'lines':out,'truncated':truncated}))
`;

export function createRetrievalTools(access: ExecutionAccess, mirror?: SandboxReadMirror): AgentTool[] {
  return ["grep", "find", "ls"].map((kind): AgentTool => ({
    name: kind, label: kind,
    description: `${kind === "grep" ? "Search text with line numbers" : kind === "find" ? "Find file paths" : "List directory entries"} in an authorized knowledge/ or public_data/ path. Results are bounded; narrow the path when truncated.`,
    parameters: { type: "object", properties: {
      path: { type: "string", description: "Logical resource or workspace path; defaults to public_data/." },
      ...(kind === "grep" ? { pattern: { type: "string" }, literal: { type: "boolean" }, ignoreCase: { type: "boolean" } } : {}),
      ...(kind !== "ls" ? { glob: { type: "string" } } : {}),
      limit: { type: "integer", minimum: 1, maximum: 200 },
    }, required: kind === "grep" ? ["pattern"] : [], additionalProperties: false } as TSchema,
    async execute(_id, args, signal) {
      if (!isRecord(args)) throw new Error("invalid search arguments");
      const result = await runRetrieval(access, mirror, args, kind, signal);
      return { content: [{ type: "text", text: result.lines.join("\n") + (result.truncated ? "\n[Truncated: narrow the path or query.]" : "") }], details: { truncated: result.truncated } };
    },
  }));
}

async function runRetrieval(access: ExecutionAccess, mirror: SandboxReadMirror | undefined, args: Record<string, unknown>, kind: string, signal?: AbortSignal): Promise<{ lines: string[]; truncated: boolean }> {
  const label = typeof args.path === "string" ? args.path : "public_data";
  const host = mirror?.resolveAlias(label);
  const root = host ? await realpath(host) : await access.resolvePath(label, "read");
  if (host && !await mirror!.containsCanonical(root)) throw new Error("search resource symlink escapes its configured root");
  const allowedRoots = host ? [root] : (await Promise.all(["public_data", "knowledge", ".agents/skills", "storage"].map(async (path) => {
    try { return await access.resolvePath(path, "read"); } catch { return undefined; }
  }))).filter((path): path is string => path !== undefined);
  const query = JSON.stringify({ ...args, kind, label, root, allowedRoots, limit: Math.min(200, Math.max(1, Number(args.limit ?? 100))) });
  let stdout: string;
  if (host) {
    ({ stdout } = await execFileAsync("python3", ["-c", RETRIEVAL_SCRIPT, query], {
      signal, timeout: 30_000, maxBuffer: 256 * 1024, env: safeShellEnvironment(),
    }));
  } else {
    stdout = "";
    const result = await access.operations.exec(`python3 -c ${quote(RETRIEVAL_SCRIPT)} ${quote(query)}`, access.cwd, {
      signal, timeout: 30, env: access.env,
      onData: (data) => { stdout += Buffer.from(data).toString("utf8"); if (stdout.length > 256 * 1024) throw new Error("search output exceeds limit"); },
    });
    if (result.exitCode !== 0) throw new Error(`search failed: ${stdout.slice(-2000)}`);
  }
  return JSON.parse(stdout.trim());

}


/** Canonical checks happen in the execution plane before a child reads bytes. */
export async function verifyReadScope(access: ExecutionAccess, mirror: SandboxReadMirror | undefined, path: string, signal?: AbortSignal): Promise<void> {
  const host = mirror?.resolveAlias(path);
  if (host) {
    if (!await mirror!.containsCanonical(await realpath(host))) throw new Error("PATH_SCOPE_ERROR: resource symlink escapes its configured root");
    return;
  }
  await runRetrieval(access, mirror, { path }, "verify", signal);
}
