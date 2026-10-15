import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import type { AgentSession, ExtensionAPI, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { registerSubagentCapabilityCeiling } from "pi-subagents/capability-ceiling";
import { registerRequiredChildExtensions } from "pi-subagents/required-child-extensions";
import { serveSubagentIpc } from "./subagent-ipc.js";
import { SubagentReports } from "./subagent-reports.js";
import { isRecord, type JsonObject, type RunnerEvent, PROTOCOL_VERSION } from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";

const ROLE_TOOLS = { search: ["read", "grep", "find", "ls"], general_purpose: ["read", "grep", "find", "ls", "write", "edit", "terminal"] };
export function subagentModule(name: string): string {
  const js = fileURLToPath(new URL(`./${name}.js`, import.meta.url));
  return existsSync(js) ? js : js.replace(/\.js$/, ".ts");
}

export function prepareSubagentEnvironment(workspace: string) {
  const agentDir = join(workspace, "pi", "agent");
  const stateRoot = join(workspace, "pi", "subagents");
  for (const path of [agentDir, stateRoot, join(agentDir, "agents"), join(agentDir, "extensions", "subagent")]) mkdirSync(path, { recursive: true, mode: 0o700 });
  process.env.PI_CODING_AGENT_DIR = agentDir;
  process.env.PI_SUBAGENTS_TEMP_ROOT = stateRoot;
  process.env.PYROMIND_SUBAGENT_SOCKET = `/tmp/pyromind-pi-${createHash("sha256").update(workspace).digest("hex").slice(0, 24)}.sock`;
  const roleRoot = fileURLToPath(new URL(existsSync(fileURLToPath(new URL("./agents/search.md", import.meta.url))) ? "./agents/" : "../agents/", import.meta.url));
  for (const role of Object.keys(ROLE_TOOLS)) {
    const extension = subagentModule(role === "search" ? "subagent-search-extension" : "subagent-general-extension");
    writeFileSync(join(agentDir, "agents", `${role}.md`), readFileSync(join(roleRoot, `${role}.md`), "utf8").replace("{{extension}}", extension), { mode: 0o600 });
  }
  writeFileSync(join(agentDir, "settings.json"), JSON.stringify({ subagents: {
    disableBuiltins: true, agentExcludeDirs: [join(workspace, ".agents"), join(workspace, ".pi")],
    modelScope: { allow: ["inherit"] },
  } }), { mode: 0o600 });
  writeFileSync(join(agentDir, "subagent-tool-description.md"), `Delegate a bounded task to one of these conversation agents:
- search: read-only research in the workspace and platform knowledge base; returns evidence, paths and line numbers.
- general_purpose: analysis, scripts, file changes and verification within the conversation's sandbox and approval rules.

Launch: {agent:"search"|"general_purpose",task:"...",async:true}. Background execution is the default; async:false waits for the result.
Include the goal, necessary background, relevant paths and expected result. Children have independent context. Handle simple operations directly and continue from the child's final handoff.
Inspect: {action:"status",id:"<run-id>"}. Continue: {action:"resume",id:"<run-id>",message:"..."}. Stop: {action:"stop",id:"<run-id>"}. Discover roles: {action:"list"}.
Resume restores the child session and returns a new run ID; use the latest returned ID. Native completion notifications deliver background results.
When a result points to Full output in public_data/subagents/, use read with that exact path and paginate as needed. Do not rerun the child to retrieve a truncated report.
Only these two exact role names and these calls are supported. Do not pass options, model, tools, cwd, workflow, tasks or chain overrides.\n`, { mode: 0o600 });
  writeFileSync(join(agentDir, "extensions", "subagent", "config.json"), JSON.stringify({
    toolActivation: "eager", toolDescriptionMode: "custom", asyncByDefault: true,
    defaultSubagentContext: "fresh", intercomBridge: { mode: "off" }, maxActiveAsyncRunsPerSession: 4, globalConcurrencyLimit: 4,
    maxSubagentDepth: 1, scheduledRuns: { enabled: false }, missions: { enabled: false },
    defaultSessionDir: join(stateRoot, "sessions"),
    disabledFeatures: ["agent-management", "watchdog", "panes", "missions", "lane-management", "spawn-budget-grants", "preflight", "lane-metadata", "gates", "extension-bindings", "external-machines", "workflow-scripts"],
  }), { mode: 0o600 });
  return { agentDir, stateRoot, socket: process.env.PYROMIND_SUBAGENT_SOCKET };
}

/** Bridges capabilities and product observations; all execution lifecycle belongs to pi-subagents. */
export class SubagentHost {
  private session?: AgentSession;
  private pi?: ExtensionAPI;
  private closeIpc?: () => Promise<void>;
  private registrations: Array<{ dispose(): void }> = [];
  private readonly states = new Map<string, string>();
  private readonly launches = new Map<string, string | undefined>();
  private readonly backgroundIds = new Set<string>();
  private lastUsage = "";
  private usagePending = false;
  private readonly mutatedSessions = new Set<string>();
  private disposed = false;
  private monitor?: NodeJS.Timeout;
  private refreshPending?: Promise<void>;
  private readonly reports: SubagentReports;
  readonly paths: ReturnType<typeof prepareSubagentEnvironment>;

  constructor(workspace: string, private sessionId: string, private peer: JsonlRpcPeer,
    private tools: AgentTool[], private modelRuntime: ModelRuntime, private model: Model<Api>, private instructions: string) {
    this.paths = prepareSubagentEnvironment(workspace);
    const writer = tools.find(tool => tool.name === "write");
    if (!writer) throw new Error("Subagent report delivery requires the conversation write tool");
    this.reports = new SubagentReports(workspace, () => this.session!.sessionId, writer);
  }

  async start() {
    this.closeIpc = await serveSubagentIpc(this.paths.socket, async (request, signal, update) => {
      if (!isRecord(request) || !(request.role === "search" || request.role === "general_purpose")) throw new Error("Unknown child role");
      const allowed = ROLE_TOOLS[request.role];
      if (request.method === "config") return {
        model: publicModel(this.session?.model ?? this.model), instructions: this.instructions,
        tools: this.tools.filter((tool) => allowed.includes(tool.name)).map(({ name, label, description, parameters }) => ({ name, label, description: `[Conversation host] ${description}`, parameters })),
      };
      if (request.method === "tool") {
        if (typeof request.name !== "string" || !allowed.includes(request.name)) throw new Error("Child tool is not allowed");
        const tool = this.tools.find((tool) => tool.name === request.name);
        if (!tool) throw new Error("Child tool is unavailable");
        const id = `subagent:${request.session}:${request.id}`;
        try { return await tool.execute(id, request.args, signal, update); }
        finally { if (["write", "edit", "terminal"].includes(request.name)) {
          this.mutatedSessions.add(String(request.session));
          this.emit(`mutation:${randomUUID()}`, "subagent.workflow_modified", {});
        } }
      }
      if (request.method === "model") {
        const options = isRecord(request.options) ? request.options : {};
        const stream = this.modelRuntime.streamSimple(this.session?.model ?? this.model, request.context as any, { ...options, signal });
        for await (const event of stream) update(event);
        return null;
      }
      throw new Error("Unknown subagent host request");
    });
  }

  extension = (pi: ExtensionAPI) => {
    this.pi = pi;
    pi.on("turn_start", () => this.reports.beginTurn());
    pi.on("tool_result", async (event, ctx) => {
      if (event.toolName !== "subagent") return;
      return this.reports.transform(event.content, event.details, ctx.signal);
    });
    pi.on("context", async (event, ctx) => {
      const messages: AgentMessage[] = [];
      for (const message of event.messages) {
        if (message.role === "toolResult" && message.toolName === "subagent") {
          messages.push({ ...message, ...await this.reports.transform(message.content, message.details, ctx.signal) });
        } else if (message.role === "custom" && message.customType === "subagent-notify") {
          messages.push({ ...message, ...await this.reports.transform(message.content, message.details, ctx.signal) });
        } else messages.push(message);
      }
      return { messages };
    });
    pi.on("session_start", (_event, ctx) => {
      for (const registration of this.registrations) registration.dispose();
      this.registrations = [
        registerSubagentCapabilityCeiling({ sessionId: ctx.sessionManager.getSessionId(), source: "conversation", ceiling: { allowedAgents: Object.keys(ROLE_TOOLS), allowedTools: ROLE_TOOLS.general_purpose } }),
        registerRequiredChildExtensions({ sessionId: ctx.sessionManager.getSessionId(), extensions: [{ id: "conversation-host", path: subagentModule("subagent-required-extension") }] }),
      ];
    });
    pi.on("tool_call", (event) => {
      if (event.toolName !== "subagent") return;
      const error = validateSubagentCall(event.input);
      if (error) return { block: true, reason: error };
    });
    for (const event of ["subagent:async-started", "subagent:async-complete", "subagent:foreground-complete", "subagent:process-terminal", "subagent:child-status"]) {
      pi.events.on(event, () => { void this.refresh().catch((error) => process.stderr.write(`Subagent status: ${error.message}\n`)); });
    }
  };

  async attach(session: AgentSession) {
    this.session = session;
    if (!session.getToolDefinition("subagent")) throw new Error("pi-subagents did not register its native tool");
    await session.bindExtensions({ mode: "print", onError: (error) => process.stderr.write(`Pi extension ${error.extensionPath}: ${error.error}\n`) });
    await this.refresh();
    this.monitor = setInterval(() => void this.refresh(false).catch((error) => process.stderr.write(`Subagent reconciliation: ${error.message}\n`)), 1000);
    this.monitor.unref();
  }

  async rpc(method: string, params: Record<string, unknown> = {}): Promise<any> {
    if (!this.pi) throw new Error("Subagent plugin not loaded");
    const requestId = randomUUID();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { off(); reject(new Error(`Subagent ${method} timed out`)); }, 15_000);
      const off = this.pi!.events.on(`subagents:rpc:v1:reply:${requestId}`, (reply: any) => {
        clearTimeout(timer); off();
        if (reply.success) resolve(reply.data); else reject(new Error(reply.error?.message ?? "Subagent RPC failed"));
      });
      this.pi!.events.emit("subagents:rpc:v1:request", { version: 1, requestId, method, params });
    });
  }

  observeCall(id: string, input: unknown) {
    if (isRecord(input) && (!input.action || input.action === "resume")) this.launches.set(id, undefined);
  }

  observeProgress(result: unknown, callId: string) {
    if (!this.launches.has(callId)) return;
    if (isRecord(result) && isRecord(result.details) && typeof result.details.runId === "string") {
      this.launches.set(callId, result.details.runId);
      this.observe(result.details.runId, "running", "Subagent");
    }
  }

  observeResult(result: unknown, callId: string, isError = false) {
    if (!this.launches.has(callId)) return;
    const retainedId = this.launches.get(callId);
    this.launches.delete(callId);
    const details = isRecord(result) && isRecord(result.details) ? result.details : {};
    const id = typeof details.runId === "string" ? details.runId : retainedId;
    if (!id) return;
    const background = typeof details.asyncId === "string";
    if (background) this.backgroundIds.add(id);
    const failed = isError || (isRecord(result) && result.isError === true);
    const label = String((details.results as any[])?.[0]?.agent ?? "Subagent");
    const output = JSON.stringify(result) ?? "";
    const handoff = output.length > 16000 ? output.slice(0, 16000) + `\n[Truncated; query subagent status with id ${id}.]` : output;
    this.observe(id, background ? "running" : failed ? "failed" : "complete", label, { result: handoff });
    void this.refresh().catch(() => {});
  }

  refresh(includeUsage = true): Promise<void> {
    this.usagePending ||= includeUsage;
    if (this.disposed || !this.session) return Promise.resolve();
    if (this.refreshPending) return this.refreshPending;
    this.refreshPending = (async () => {
      const data = await this.rpc("status");
      for (const run of data.asyncSnapshot?.runs ?? []) {
        this.backgroundIds.add(run.id);
        this.observe(run.id, run.state, run.label);
      }
      // Documented plugin lifecycle artifacts are authoritative after reconnect and
      // after terminal jobs disappear from the active status projection.
      const runs = join(this.paths.stateRoot, "async-subagent-runs");
      for (const name of existsSync(runs) ? readdirSync(runs) : []) {
        if (!/^[a-zA-Z0-9-]+$/.test(name)) continue;
        try {
          const status = JSON.parse(readFileSync(join(runs, name, "status.json"), "utf8"));
          if (status.lifecycleArtifactVersion !== 3 || status.runId !== name || ![this.session!.sessionManager.getSessionFile(), this.session!.sessionId].includes(status.sessionId)) continue;
          this.backgroundIds.add(name);
          this.observe(name, status.state, status.steps?.[0]?.agent ?? "Subagent");
        } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT" && !(error instanceof SyntaxError)) throw error; }
      }
      if (!this.usagePending) return;
      this.usagePending = false;
      const report = await this.rpc("cost");
      if (report.total && JSON.stringify(report.total) !== this.lastUsage) {
        this.lastUsage = JSON.stringify(report.total);
        this.emit(`usage:${randomUUID()}`, "usage.updated", { input_tokens: report.total.input, output_tokens: report.total.output, cached_tokens: report.total.cacheRead, cost_usd: report.total.cost });
      }
    })().finally(() => {
      this.refreshPending = undefined;
      if (this.usagePending && !this.disposed) void this.refresh().catch(() => {});
    });
    return this.refreshPending;
  }

  private observe(id: string, state: string, label: string, extra: JsonObject = {}) {
    if (this.states.get(id) === state) return;
    if (this.states.has(id) && !["running", "queued"].includes(this.states.get(id)!) && ["running", "queued"].includes(state)) return;
    if (!this.states.has(id)) this.emit(id, "subagent.started", { label });
    this.states.set(id, state);
    if (!["running", "queued"].includes(state)) this.emit(id, "subagent.finished", { status: state, label, mutated: label === "general_purpose" || this.mutatedSessions.size > 0, ...extra });
  }

  private emit(id: string, kind: RunnerEvent["kind"], payload: JsonObject) {
    this.peer.emit({ protocolVersion: PROTOCOL_VERSION, type: "pi.event", eventId: `${id}:${kind}:${payload.status ?? ""}`, sessionId: this.sessionId, runId: id, occurredAt: new Date().toISOString(), kind, payload });
  }

  async stopAll() {
    await this.refresh();
    for (const id of this.backgroundIds) {
      if (["running", "queued"].includes(this.states.get(id) ?? "")) {
        try { await this.rpc("stop", { id }); }
        catch (error) {
          await this.refresh(false);
          if (["running", "queued"].includes(this.states.get(id) ?? "")) throw error;
        }
      }
    }
    // Native stop acknowledges intent. Keep operations alive until native status is terminal.
    const deadline = Date.now() + 20_000;
    while ([...this.backgroundIds].some((id) => ["running", "queued"].includes(this.states.get(id) ?? ""))) {
      if (Date.now() > deadline) throw new Error("Subagent stop is still pending");
      await new Promise((resolve) => setTimeout(resolve, 100));
      await this.refresh();
    }
  }

  async close() {
    this.disposed = true;
    clearInterval(this.monitor);
    for (const registration of this.registrations) registration.dispose();
    await this.closeIpc?.();
  }
}

export function validateSubagentCall(input: Record<string, unknown>): string | undefined {
  const fields = input.action ? ["action", "id", "message", "options"] : ["agent", "task", "async", "context"];
  if (Object.keys(input).some((key) => !fields.includes(key))) return 'Supported launch: {agent:"search"|"general_purpose",task:"...",async:true|false}. Management: {action:"status"|"resume"|"stop"|"list",id?,message?}. Other fields, including options, are host-controlled.';
  if (input.action && !["status", "resume", "stop", "list"].includes(String(input.action))) return "Unsupported subagent action; supported actions: status, resume, stop, list. Available agents: search, general_purpose.";
  if (input.agent && !Object.hasOwn(ROLE_TOOLS, String(input.agent))) return "Unknown subagent role; available agents: search, general_purpose (use the exact name).";
  if (input.context !== undefined && input.context !== "fresh") return "Subagents use an independent fresh context";
  if (input.options !== undefined) return "Subagent options are host-controlled";
  return undefined;
}

function publicModel(model: Model<Api>) {
  return {
    id: model.id, name: model.name, provider: model.provider, api: model.api,
    baseUrl: "http://conversation-host.invalid", reasoning: model.reasoning,
    input: model.input, cost: model.cost, contextWindow: model.contextWindow,
    maxTokens: model.maxTokens, compat: model.compat,
  };
}
