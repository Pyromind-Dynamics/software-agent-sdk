import { readBundledSubagentRole, writeSubagentRoles, type SubagentRoleDefinition } from "./subagent-roles.js";
import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, writeFileSync } from "node:fs";
import { join, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import { SessionManager, type AgentSession, type ExtensionAPI, type ModelRuntime } from "@earendil-works/pi-coding-agent";
import { registerSubagentCapabilityCeiling } from "pi-subagents/capability-ceiling";
import { registerRequiredChildExtensions } from "pi-subagents/required-child-extensions";
import { Type } from "typebox";
import { serveSubagentIpc } from "./subagent-ipc.js";
import { SubagentDisplay, displayText, redactSignedUrls, transcriptWindow } from "./subagent-display.js";
import { SubagentActivity } from "./subagent-activity.js";
import { SubagentRecipients } from "./subagent-recipients.js";
import { SubagentTerminals, usesAgentTerminal, type SubagentTerminalFactory } from "./subagent-terminals.js";
import { SubagentReports } from "./subagent-reports.js";
import { isRecord, type JsonObject, type RunnerEvent, PROTOCOL_VERSION } from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";

const ROLE_TOOLS = { search: ["read", "grep", "find", "ls"], general_purpose: ["read", "grep", "find", "ls", "write", "edit", "terminal"] };
const SUBAGENT_ACTIONS = ["status", "steer", "resume", "stop", "list"];
type CompletionMessage = Parameters<ExtensionAPI["sendMessage"]>[0];
export function subagentModule(name: string): string {
  const js = fileURLToPath(new URL(`./${name}.js`, import.meta.url));
  return existsSync(js) ? js : js.replace(/\.js$/, ".ts");
}

export function prepareSubagentEnvironment(workspace: string, definitions?: SubagentRoleDefinition[]) {
  const agentDir = join(workspace, "pi", "agent");
  const stateRoot = join(workspace, "pi", "subagents");
  for (const path of [agentDir, stateRoot, join(agentDir, "agents"), join(agentDir, "extensions", "subagent")]) mkdirSync(path, { recursive: true, mode: 0o700 });
  process.env.PI_CODING_AGENT_DIR = agentDir;
  process.env.PI_SUBAGENTS_TEMP_ROOT = stateRoot;
  process.env.PYROMIND_SUBAGENT_SOCKET = `/tmp/pyromind-pi-${createHash("sha256").update(workspace).digest("hex").slice(0, 24)}.sock`;
  for (const role of Object.keys(ROLE_TOOLS)) {
    const extension = subagentModule(role === "search" ? "subagent-search-extension" : "subagent-general-extension");
    writeFileSync(join(agentDir, "agents", `${role}.md`), readBundledSubagentRole(role).replace("{{extension}}", extension), { mode: 0o600 });
  }
  if (definitions) writeSubagentRoles(agentDir, definitions, subagentModule("subagent-child-extension"));
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
  if (definitions) writeFileSync(join(agentDir, "subagent-tool-description.md"),
    `Delegate a complete task to ${definitions.map(role => role.name).join(", ")} with {agent,task,async:true}. Include goals, actual input paths, constraints and expected outputs in task text. Use {action:"status",id}, {action:"steer",id,message}, {action:"resume",id,message}, {action:"stop",id}. IDs are native run IDs; resume returns a new ID. Read returned report paths with read. Native completion is not business success; native stop does not cancel platform tasks. Role tools and context are host-controlled.`, { mode: 0o600 });
  writeFileSync(join(agentDir, "extensions", "subagent", "config.json"), JSON.stringify({
    toolActivation: "eager", toolDescriptionMode: "custom", pinToolDeclarations: false, asyncByDefault: true,
    defaultSubagentContext: "fresh", intercomBridge: { mode: "off" }, maxActiveAsyncRunsPerSession: 4, globalConcurrencyLimit: 4,
    maxSubagentDepth: 1, scheduledRuns: { enabled: false }, missions: { enabled: false },
    defaultSessionDir: join(stateRoot, "sessions"),
    disabledFeatures: ["agent-management", "watchdog", "panes", "missions", "lane-management", "spawn-budget-grants", "preflight", "lane-metadata", "gates", "extension-bindings", "external-machines", "workflow-scripts"],
  }), { mode: 0o600 });
  return { agentDir, stateRoot, socket: process.env.PYROMIND_SUBAGENT_SOCKET };
}

/** Bridges capabilities and product observations; all execution lifecycle belongs to pi-subagents. */
export class SubagentHost {
  private readonly roleTools: Record<string, string[]>;
  private session?: AgentSession;
  private pi?: ExtensionAPI;
  private closeIpc?: () => Promise<void>;
  private registrations: Array<{ dispose(): void }> = [];
  private readonly observedLive = new Set<string>();
  private readonly states = new Map<string, string>();
  private readonly finalizing = new Map<string, Promise<void>>();
  private readonly handoffs = new Map<string, string>();
  private readonly launchInputs = new Map<string, Record<string, unknown>>();
  private readonly display: SubagentDisplay;
  private readonly launches = new Map<string, string | undefined>();
  private readonly backgroundIds = new Set<string>();
  private lastUsage = "";
  private usagePending = false;
  private readonly mutatedSessions = new Set<string>();
  private disposed = false;
  private monitor?: NodeJS.Timeout;
  private refreshPending?: Promise<void>;
  private readonly reports: SubagentReports;
  private recipients?: SubagentRecipients;
  private terminals?: SubagentTerminals;
  readonly paths: ReturnType<typeof prepareSubagentEnvironment>;

  constructor(private workspace: string, private sessionId: string, private peer: JsonlRpcPeer,
    private tools: AgentTool[], private modelRuntime: ModelRuntime, private model: Model<Api>, private instructions: string, private definitions?: SubagentRoleDefinition[], private roleToolSets?: Record<string, AgentTool[]>, private skillPathAliases: Array<{ host: string; alias: string }> = [], terminalFactory?: SubagentTerminalFactory) {
    this.roleTools = definitions ? Object.fromEntries(definitions.map(role => [role.name, role.tools])) : ROLE_TOOLS;
    for (const role of definitions ?? []) {
      const available = new Set((roleToolSets?.[role.name] ?? tools).map(tool => tool.name));
      const missing = role.tools.filter(name => !available.has(name));
      if (missing.length) throw new Error(`Delegated role ${role.name} requests unavailable host tools: ${missing.join(", ")}`);
    }
    this.paths = prepareSubagentEnvironment(workspace, definitions);
    if (terminalFactory) this.terminals = new SubagentTerminals(this.paths.stateRoot, terminalFactory);
    const activity = new SubagentActivity(this.paths.stateRoot, join(workspace, "pi", "subagent-artifacts"),
      () => this.session ? [this.session.sessionId, this.session.sessionManager.getSessionFile() ?? ""] : []);
    this.display = new SubagentDisplay((method, params) => this.rpc(method, params), id => {
      this.emit(id, "subagent.progress", { details: this.display.details(id) });
    }, join(workspace, "pi", "subagent-display.json"), id => activity.read(id));
    const writer = tools.find(tool => tool.name === "write");
    if (!writer) throw new Error("Subagent report delivery requires the conversation write tool");
    this.reports = new SubagentReports(workspace, () => this.session!.sessionId, writer, Object.keys(this.roleTools));
  }

  pluginApi(pi: ExtensionAPI): ExtensionAPI {
    const lifecycleOnly = "Lifecycle-only. Do not combine with agent, task, async, or context.";
    const launchOnly = "Launch-only. Omit action, id, and message.";
    const parameters = Type.Object({
      agent: Type.Optional(Type.String({ enum: Object.keys(this.roleTools), description: launchOnly })),
      task: Type.Optional(Type.String({ description: `Required for launch. ${launchOnly}` })),
      async: Type.Optional(Type.Boolean({ description: `Defaults to true. Set false to wait. ${launchOnly}` })),
      context: Type.Optional(Type.String({ enum: ["fresh"], description: launchOnly })),
      action: Type.Optional(Type.String({ enum: SUBAGENT_ACTIONS, description: `${lifecycleOnly} For example: {action:"list"} or {action:"resume",id:"<run-id>",message:"..."}.` })),
      id: Type.Optional(Type.String({ description: `${lifecycleOnly} Native run ID.` })),
      message: Type.Optional(Type.String({ description: `${lifecycleOnly} Guidance for steer or resume.` })),
    }, { additionalProperties: false });
    return {
      ...pi,
      registerTool(tool) {
        // The restricted parameters are a subset of what the native executor accepts.
        pi.registerTool(tool.name === "subagent"
          ? { ...tool, parameters: parameters as unknown as typeof tool.parameters }
          : tool);
      },
    };
  }

  async start() {
    this.closeIpc = await serveSubagentIpc(this.paths.socket, async (request, signal, update) => {
      if (isRecord(request) && request.method === "probe") return { ready: true };
      if (!isRecord(request) || typeof request.role !== "string" || !Object.hasOwn(this.roleTools, request.role)) throw new Error("Unknown child role");
      const allowed = this.roleTools[request.role];
      const roleTools = this.roleToolSets?.[request.role] ?? this.tools;
      if (request.method === "config") return {
        model: publicModel(this.session?.model ?? this.model), instructions: this.instructions,
        skillPathAliases: this.skillPathAliases,
        workspacePathAlias: this.skillPathAliases.length ? {host: this.workspace, alias: "."} : undefined,
        terminalTools: this.terminals ? allowed.filter(usesAgentTerminal) : [],
        tools: roleTools.filter((tool) => allowed.includes(tool.name)).map(({ name, label, description, parameters }) => ({ name, label, description: `[Conversation host] ${description}`, parameters })),
      };
      if (request.method === "terminal.attach" && this.terminals && allowed.some(usesAgentTerminal)) {
        if (typeof request.session !== "string" || typeof request.session_file !== "string") throw new Error("Missing child session identity");
        await this.terminals.attach(request.role, request.session, request.session_file, ".", signal, (id) => update({ id }));
        return null;
      }
      if (request.method === "terminal.detach" && this.terminals) {
        await this.terminals.detach(request);
        return null;
      }
      if (request.method === "tool") {
        if (typeof request.name !== "string" || !allowed.includes(request.name)) throw new Error("Child tool is not allowed");
        const tool = roleTools.find((tool) => tool.name === request.name);
        if (!tool) throw new Error("Child tool is unavailable");
        const id = `subagent:${request.session}:${request.id}`;
        const execute = async (selected: AgentTool, toolSignal: AbortSignal) => {
          try {
            if ("executeChild" in selected && typeof selected.executeChild === "function") {
              const origin = this.recipients!.origin(request.session, request.session_file);
              return await selected.executeChild(id, request.args, toolSignal, origin);
            }
            return await selected.execute(id, request.args, toolSignal, update);
          } finally {
            if (["write", "edit", "terminal"].includes(selected.name)) {
              this.mutatedSessions.add(String(request.session));
              this.emit(`mutation:${randomUUID()}`, "subagent.workflow_modified", {});
            }
          }
        };
        if (this.terminals && usesAgentTerminal(request.name)) return this.terminals.run(request, signal, execute);
        return execute(tool, signal);
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
          const handoff = await this.reports.transform(message.content, message.details, ctx.signal);
          messages.push({ ...message, ...handoff });
        } else messages.push(message);
      }
      return { messages };
    });
    pi.on("session_start", (_event, ctx) => {
      for (const registration of this.registrations) registration.dispose();
      this.registrations = [
        registerSubagentCapabilityCeiling({ sessionId: ctx.sessionManager.getSessionId(), source: "conversation", ceiling: { allowedAgents: Object.keys(this.roleTools), allowedTools: [...new Set(Object.values(this.roleTools).flat())] } }),
        registerRequiredChildExtensions({ sessionId: ctx.sessionManager.getSessionId(), extensions: [{ id: "conversation-host", path: subagentModule("subagent-required-extension") }] }),
      ];
    });
    pi.on("tool_call", (event) => {
      if (event.toolName !== "subagent") return;
      const { input, toolCallId } = event;
      if (input.action === undefined && typeof input.agent === "string" && input.task === undefined &&
          typeof input.message === "string" && input.message.trim()) {
        input.task = input.message;
        delete input.message;
        if (this.launchInputs.has(toolCallId)) this.launchInputs.set(toolCallId, input);
      }
      const error = validateSubagentCall(input, this.roleTools);
      if (error) return { block: true, reason: error };
    });
    for (const event of ["subagent:async-started", "subagent:async-complete", "subagent:foreground-complete", "subagent:process-terminal", "subagent:child-status"]) {
      pi.events.on(event, (data) => {
        void (async () => {
          if (event === "subagent:async-complete") await this.observeCompletion(data);
          await this.refresh();
        })().catch((error) => process.stderr.write(`Subagent status: ${error.message}\n`));
      });
    }
  };

  async attach(session: AgentSession) {
    this.session = session;
    this.recipients = new SubagentRecipients(this.paths.stateRoot, session, (method, params, id) => this.rpc(method, params, id), this.sessionId);
    if (!session.getToolDefinition("subagent")) throw new Error("pi-subagents did not register its native tool");
    await session.bindExtensions({ mode: "print", onError: (error) => process.stderr.write(`Pi extension ${error.extensionPath}: ${error.error}\n`) });
    await this.refresh();
    this.monitor = setInterval(() => void this.refresh(false).catch((error) => process.stderr.write(`Subagent reconciliation: ${error.message}\n`)), 1000);
    this.monitor.unref();
  }

  async notifyRecipient(params: JsonObject): Promise<JsonObject> {
    if (!this.recipients) return { accepted: false, reason: "subagents_not_ready" };
    return this.recipients.notify(params);
  }

  filterCompletion = (message: CompletionMessage): CompletionMessage | undefined => {
    if (!isRecord(message.details) || !Array.isArray(message.details.runs)) return message;
    const runs = message.details.runs;
    const delivered = runs.filter(run => !isRecord(run) || run.status !== "completed" ||
      typeof run.runId !== "string" || !this.awaitingWorkflowResult(run.runId));
    if (delivered.length === runs.length) return message;
    if (!delivered.length) return;
    const content = delivered.map(run => {
      if (!isRecord(run) || typeof run.runId !== "string") return "A subagent completed.";
      const result = this.completionMessages(run.runId)?.findLast(entry =>
        entry.type === "message" && entry.message.role === "assistant");
      const output = result?.type === "message" ? displayText(result.message) : "";
      const preview = output.length > 2000 ? `${output.slice(0, 2000)}\n[preview truncated]` : output;
      return `${run.agent} ${run.status}. Run ID: ${run.runId}\n${preview}\nUse subagent status for the full result.`;
    }).join("\n\n");
    return { ...message, details: { ...message.details, runs: delivered }, content };
  };

  private completionMessages(id: string) {
    if (!this.session || !/^[a-zA-Z0-9-]+$/.test(id)) return;
    try {
      const status = JSON.parse(readFileSync(join(this.paths.stateRoot, "async-subagent-runs", id, "status.json"), "utf8"));
      if (status.lifecycleArtifactVersion !== 3 || status.runId !== id || status.steps?.length !== 1 ||
          ![this.session.sessionId, this.session.sessionManager.getSessionFile()].includes(status.sessionId)) return;
      const step = status.steps[0];
      const start = step.startedAt ?? status.startedAt;
      if (typeof start !== "number") return;
      const path = realpathSync(step.sessionFile ?? status.sessionFile);
      if (!path.startsWith(realpathSync(join(this.paths.stateRoot, "sessions")) + sep)) return;
      return SessionManager.open(path).getBranch().filter(entry =>
        entry.type === "message" && entry.message.timestamp >= start);
    } catch { /* A missing artifact must not hide a completion. */ }
  }

  private awaitingWorkflowResult(id: string): boolean {
    const entry = this.completionMessages(id)?.findLast(item => item.type === "message" &&
      (item.message.role === "user" || item.message.role === "toolResult" && item.message.toolName === "submit_workflow"));
    return entry?.type === "message" && entry.message.role === "toolResult" && !entry.message.isError &&
      isRecord(entry.message.details) && ["submitting", "submitted"].includes(String(entry.message.details.status));
  }

  async rpc(method: string, params: Record<string, unknown> = {}, requestId: string = randomUUID()): Promise<any> {
    if (!this.pi) throw new Error("Subagent plugin not loaded");
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
    if (isRecord(input) && (!input.action || input.action === "resume")) {
      this.launches.set(id, undefined);
      this.launchInputs.set(id, input);
    }
  }

  observeProgress(result: unknown, callId: string) {
    if (!this.launches.has(callId)) return;
    if (isRecord(result) && isRecord(result.details) && typeof result.details.runId === "string") {
      this.launches.set(callId, result.details.runId);
      this.linkRun(result.details.runId, callId);
      const input = this.launchInputs.get(callId);
      if (input?.async === false && !input.action) {
        this.observe(result.details.runId, "running", String(input.agent ?? "Subagent"));
        void this.display.poll(result.details.runId);
      }
    }
  }

  observeResult(result: unknown, callId: string, isError = false) {
    if (!this.launches.has(callId)) return;
    const retainedId = this.launches.get(callId);
    this.launches.delete(callId);
    const details = isRecord(result) && isRecord(result.details) ? result.details : {};
    const id = typeof details.runId === "string" ? details.runId : retainedId;
    if (!id) { this.launchInputs.delete(callId); return; }
    const failed = isError || (isRecord(result) && result.isError === true);
    if (failed && !this.states.has(id)) {
      this.launchInputs.delete(callId);
      return;
    }
    this.linkRun(id, callId);
    this.launchInputs.delete(callId);
    const background = typeof details.asyncId === "string";
    if (background) this.backgroundIds.add(id);
    const label = String((details.results as any[])?.[0]?.agent ?? "Subagent");
    const output = displayText(result);
    const handoff = output.length > 16000 ? output.slice(0, 16000) + `\n[Truncated; query subagent status with id ${id}.]` : output;
    if (!background) this.observe(id, failed ? "failed" : "complete", label, { result: handoff });
    void this.refresh().catch(() => {});
  }

  refresh(includeUsage = true): Promise<void> {
    this.usagePending ||= includeUsage;
    if (this.disposed || !this.session) return Promise.resolve();
    if (this.refreshPending) return this.refreshPending;
    this.refreshPending = (async () => {
      const data = await this.rpc("status");
      const observations = new Map<string, { state: string; label: string; details: JsonObject }>();
      for (const run of data.asyncSnapshot?.runs ?? []) {
        this.backgroundIds.add(run.id);
        const details = {
          ...(typeof run.startedAt === "number" ? { started_at: new Date(run.startedAt).toISOString() } : {}),
          ...(typeof run.endedAt === "number" ? { ended_at: new Date(run.endedAt).toISOString() } : {}),
        };
        observations.set(run.id, { state: run.state, label: run.label, details });
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
          const steps = Array.isArray(status.steps) ? status.steps : [];
          if (steps.length === 1) this.rememberContext(name, steps[0].sessionFile ?? status.sessionFile);
          if (typeof status.toolCallId === "string" && this.launchInputs.has(status.toolCallId)) {
            this.linkRun(name, status.toolCallId);
          }
          const current = observations.get(name);
          const details = {
            ...current?.details,
            ...(typeof status.startedAt === "number" ? { started_at: new Date(status.startedAt).toISOString() } : {}),
            ...(typeof status.endedAt === "number" ? { ended_at: new Date(status.endedAt).toISOString() } : {}),
            ...(typeof status.toolCallId === "string" ? { tool_call_id: status.toolCallId } : {}),
            ...(typeof steps[0]?.task === "string" && !this.display.value(name, "task") ? { task: steps[0].task } : {}),
            ...(typeof status.steps?.[0]?.agent === "string" ? { role: status.steps[0].agent } : {}),
          };
          observations.set(name, { state: status.state, label: steps[0]?.agent ?? current?.label ?? "Subagent", details });
        } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT" && !(error instanceof SyntaxError)) throw error; }
      }
      for (const [id, { state, label, details }] of observations) {
        this.observe(id, state, label, {}, this.display.update(id, details));
      }
      const foreground = join(this.paths.stateRoot, "async-subagent-results", "foreground-history.json");
      if (existsSync(foreground)) {
        try {
          const history = JSON.parse(readFileSync(foreground, "utf8"));
          if (history.version === 1 && Array.isArray(history.runs)) for (const run of history.runs) {
            if (!isRecord(run) || typeof run.runId !== "string" || !/^[a-zA-Z0-9-]+$/.test(run.runId) || this.backgroundIds.has(run.runId)
              || ![this.session!.sessionManager.getSessionFile(), this.session!.sessionId].includes(String(run.sessionId)) || !Array.isArray(run.children)) continue;
            const children = run.children.filter(isRecord);
            if (!children.length || children.some(child => !["completed", "failed", "stopped"].includes(String(child.status)))) continue;
            if (children.length === 1) this.rememberContext(run.runId, children[0].sessionFile);
            const state = children.some(child => child.status === "failed") ? "failed" : children.some(child => child.status === "stopped") ? "stopped" : "complete";
            this.observe(run.runId, state, String(children[0].agent ?? "Subagent"));
          }
        } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT" && !(error instanceof SyntaxError)) throw error; }
      }
      for (const [id, state] of this.states) {
        if (["running", "queued"].includes(state)) void this.display.poll(id);
        else if (this.display.captureActivity(id)) this.emit(id, "subagent.progress", { details: this.display.details(id) });
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

  private async observeCompletion(data: unknown) {
    if (!isRecord(data) || typeof data.runId !== "string" || typeof data.summary !== "string") return;
    const id = data.runId;
    if (data.success === true && this.awaitingWorkflowResult(id)) return;
    // The native completion summary is the final handoff, not the activity window.
    const transformed = await this.reports.transform([{ type: "text", text: data.summary }], data);
    const handoff = redactSignedUrls(displayText(transformed));
    if (this.disposed) return;
    const report = this.reports.publishedPath(id);
    const reportChanged = report ? this.display.update(id, { report_uri: report }) : false;
    if (!reportChanged && this.handoffs.get(id) === handoff) return;
    this.handoffs.set(id, handoff);
    if (this.states.has(id) && !this.finalizing.has(id)) {
      this.emit(id, "subagent.progress", { result: handoff,
        ...(report || this.observedLive.has(id) || this.display.value(id, "activity") ? { details: this.display.details(id) } : {}) });
    }
  }

  private linkRun(id: string, callId: string) {
    const input = this.launchInputs.get(callId) ?? {};
    const changed = this.display.remember(id, {
      tool_call_id: callId,
      ...(typeof input.agent === "string" ? { role: input.agent } : {}),
      ...(typeof input.task === "string" ? { task: input.task } : {}),
      ...(input.action === "resume" && typeof input.id === "string" ? { previous_run_id: input.id } : {}),
    });
    if (changed && this.states.has(id)) this.emit(id, "subagent.progress", { details: this.display.details(id) });
  }

  private rememberContext(id: string, sessionFile: unknown) {
    if (typeof sessionFile !== "string" || !this.recipients || this.display.value(id, "session_key")) return;
    try {
      const path = realpathSync(sessionFile);
      if (!path.startsWith(realpathSync(join(this.paths.stateRoot, "sessions")) + sep)) return;
      const header = JSON.parse(readFileSync(path, "utf8").split("\n", 1)[0]);
      const origin = this.recipients.origin(header.id, path);
      const changed = this.display.remember(id, { session_key: origin.subagent_id });
      if (changed && this.states.has(id)) this.emit(id, "subagent.progress", { details: this.display.details(id) });
    } catch { /* A missing session header must not hide lifecycle events. */ }
  }

  private observe(id: string, state: string, label: string, extra: JsonObject = {}, detailsChanged = false) {
    if (state === "pending") state = "queued";
    if (["running", "queued"].includes(state)) this.observedLive.add(id);
    const outputChanged = typeof extra.result === "string" && this.handoffs.get(id) !== extra.result;
    if (typeof extra.result === "string") this.handoffs.set(id, extra.result);
    const previous = this.states.get(id);
    if (previous && !["running", "queued"].includes(previous) && ["running", "queued"].includes(state)) return;
    const role = Object.hasOwn(this.roleTools, label) ? label : this.display.value(id, "role");
    const presentationChanged = this.display.update(id, { status: state, ...(typeof role === "string" ? { role } : {}) });
    if (!previous) {
      if (["running", "queued"].includes(state) && !this.display.value(id, "started_at")) this.display.update(id, { started_at: new Date().toISOString() });
      this.emit(id, "subagent.started", { label: role ?? label, details: this.display.details(id) });
    }
    if (previous === state) {
      if ((outputChanged || detailsChanged || presentationChanged) && !this.finalizing.has(id)) this.emit(id, "subagent.progress", {
        details: this.display.details(id), ...(outputChanged ? { result: this.handoffs.get(id)! } : {}),
      });
      return;
    }
    this.states.set(id, state);
    if (["running", "queued"].includes(state)) {
      this.emit(id, "subagent.progress", { label, details: this.display.details(id) });
    } else {
      if (!this.display.value(id, "ended_at")) this.display.update(id, { ended_at: new Date().toISOString() });
      const finish = this.finishDisplay(id, state, label, extra).finally(() => this.finalizing.delete(id));
      this.finalizing.set(id, finish);
    }
  }

  private async finishDisplay(id: string, state: string, label: string, extra: JsonObject) {
    await this.display.settled(id);
    this.display.captureActivity(id);
    const awaitingResult = state === "complete" && this.awaitingWorkflowResult(id);
    let result = this.handoffs.get(id) ?? extra.result;
    if (!result && !awaitingResult) {
      try {
        const native = await this.rpc("status", { id, view: "transcript", lines: 200 });
        const window = transcriptWindow(displayText(native));
        const text = window.text + (window.truncated ? "\n[preview truncated]" : "");
        const transformed = await this.reports.transform([{ type: "text", text }], { runs: [{ runId: id }] });
        result = displayText(transformed);
      } catch { /* A terminal operation must still be delivered if artifacts are unavailable. */ }
    }
    await Promise.resolve();
    result = this.handoffs.get(id) ?? result;
    if (this.disposed) return;
    const report = this.reports.publishedPath(id);
    if (report) this.display.update(id, { report_uri: report });
    this.emit(id, "subagent.finished", { status: state, label,
      mutated: label === "general_purpose" || this.mutatedSessions.size > 0,
      ...extra, result: typeof result === "string" ? redactSignedUrls(result).slice(0, 32768) : "",
      ...(report || this.observedLive.has(id) || this.display.value(id, "session_key") ? { details: this.display.details(id) } : {}) });
  }

  private emit(id: string, kind: RunnerEvent["kind"], payload: JsonObject) {
    const details = payload.details;
    const child = isRecord(details) && isRecord(details.subagent) ? details.subagent : undefined;
    const activity = child && isRecord(child.activity) ? child.activity : undefined;
    const safePayload = activity ? {...payload, activity, details: {...details as JsonObject, subagent: Object.fromEntries(Object.entries(child ?? {}).filter(([key]) => key !== "activity"))}} : payload;
    this.peer.emit({ protocolVersion: PROTOCOL_VERSION, type: "pi.event", eventId: kind === "subagent.progress" ? `${id}:${kind}:${randomUUID()}` : `${id}:${kind}:${payload.status ?? ""}`, sessionId: this.sessionId, runId: id, occurredAt: new Date().toISOString(), kind, payload: safePayload });
  }

  async stopAll(action: "stop" | "interrupt" = "stop") {
    await this.refresh();
    for (const id of this.backgroundIds) {
      if (["running", "queued"].includes(this.states.get(id) ?? "")) {
        try { await this.rpc(action, { id }); }
        catch (error) {
          await this.refresh(false);
          if (["running", "queued"].includes(this.states.get(id) ?? "")) throw error;
        }
      }
    }
    await Promise.all(this.finalizing.values());
    // Native stop acknowledges intent. Keep operations alive until native status is terminal.
    const deadline = Date.now() + 20_000;
    while ([...this.backgroundIds].some((id) => ["running", "queued"].includes(this.states.get(id) ?? ""))) {
      if (Date.now() > deadline) throw new Error("Subagent stop is still pending");
      await new Promise((resolve) => setTimeout(resolve, 100));
      await this.refresh();
    }
    await Promise.all(this.finalizing.values());
  }

  async close() {
    this.disposed = true;
    this.display.close();
    clearInterval(this.monitor);
    for (const registration of this.registrations) registration.dispose();
    await this.closeIpc?.();
    await this.terminals?.close();
  }
}

export function validateSubagentCall(input: Record<string, unknown>, roles: Record<string, string[]> = ROLE_TOOLS): string | undefined {
  const launchFields = ["agent", "task", "async", "context"];
  const lifecycleFields = ["id", "message"];
  const hasLaunchFields = launchFields.some((field) => Object.hasOwn(input, field));
  const hasLifecycleFields = lifecycleFields.some((field) => Object.hasOwn(input, field));
  if (typeof input.action === "string" && hasLaunchFields) {
    return `Launch and lifecycle calls are mutually exclusive. To launch: {agent:"${Object.keys(roles)[0]}",task:"...",async:true} (omit action/id/message). To inspect/control: {action:"status"|"steer"|"resume"|"stop"|"list",id?,message?} (omit agent/task/async/context).`;
  }
  if (input.action === undefined && hasLifecycleFields) {
    return `Lifecycle fields require action. Use {action:"status"|"steer"|"resume"|"stop"|"list",id?,message?}, or omit id/message and launch with {agent,task,async:true}.`;
  }
  if (input.action !== undefined && (typeof input.action !== "string" || !input.action.trim())) {
    return `action must be one of ${SUBAGENT_ACTIONS.join(", ")}. Launch calls omit action and use {agent,task,async:true}.`;
  }
  const fields = input.action !== undefined ? ["action", ...lifecycleFields, "options"] : launchFields;
  if (Object.keys(input).some((key) => !fields.includes(key))) {
    return `Unsupported subagent parameters. Launch: {agent,task,async:true|false}. Lifecycle: {action:"status"|"steer"|"resume"|"stop"|"list",id?,message?}. Available roles: ${Object.keys(roles).join(", ")}. Keep launch and lifecycle calls separate; other fields are host-controlled.`;
  }
  if (input.action && !SUBAGENT_ACTIONS.includes(String(input.action))) return `Supported actions: ${SUBAGENT_ACTIONS.join(", ")}. Available agents: ${Object.keys(roles).join(", ")}.`;
  if (input.agent && !Object.hasOwn(roles, String(input.agent))) return `Unknown subagent role; available agents: ${Object.keys(roles).join(", ")} (use the exact name).`;
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
