import { randomUUID } from "node:crypto";
import { setImmediate } from "node:timers";
import { Agent, formatSkillsForSystemPrompt, loadSkills, type AgentMessage } from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { type Api, InMemoryCredentialStore, type Model, type Models } from "@earendil-works/pi-ai";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";
import { PiEventNormalizer } from "./pi-events.js";
import { PROTOCOL_VERSION, isRecord, type JsonObject, type JsonValue, type RunnerEvent } from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";
import { createTools, type BusinessToolConfig } from "./tools.js";

interface SessionConfig {
  sessionId: string;
  provider: string;
  modelId: string;
  apiKey: string;
  baseUrl?: string;
  systemPrompt: string;
  thinkingLevel: "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
  transcript: AgentMessage[];
  workspaceRoot: string;
  skillRoot: string;
  tools: BusinessToolConfig[];
}

export class PiAgentRuntime {
  private agent: Agent | undefined;
  private sessionId: string | undefined;
  private normalizer: PiEventNormalizer | undefined;

  constructor(private readonly peer: JsonlRpcPeer) {}

  async handle(method: string, params: JsonObject): Promise<JsonValue> {
    if (method === "start") return this.start(params);
    if (method === "prompt") return this.prompt(params, false);
    if (method === "steer") return this.prompt(params, true);
    if (method === "cancel") return this.cancel();
    if (method === "close") return this.close();
    throw new Error(`unknown runner method: ${method}`);
  }

  private async start(params: JsonObject): Promise<JsonValue> {
    if (this.agent) throw new Error("Pi session already started");
    const config = parseConfig(params);
    const credentials = new InMemoryCredentialStore();
    await credentials.modify(config.provider, async () => ({ type: "api_key", key: config.apiKey }));
    const models = builtinModels({ credentials });
    const model = resolveModel(models, config);
    const shellEnv = Object.fromEntries(
      ["PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS"]
        .flatMap((name) => process.env[name] ? [[name, process.env[name]!]] : []),
    );
    const env = new NodeExecutionEnv({ cwd: config.workspaceRoot, shellEnv });
    const { skills, diagnostics } = await loadSkills(env, config.skillRoot);
    for (const diagnostic of diagnostics) process.stderr.write(`skill warning: ${diagnostic.message}\n`);
    const skillPrompt = formatSkillsForSystemPrompt(skills);
    const systemPrompt = skillPrompt ? `${config.systemPrompt}\n\n${skillPrompt}` : config.systemPrompt;
    const tools = createTools(this.peer, env, config.workspaceRoot, config.skillRoot, config.tools);
    const agent = new Agent({
      initialState: { systemPrompt, model, thinkingLevel: config.thinkingLevel, tools, messages: config.transcript },
      sessionId: config.sessionId,
      streamFn: models.streamSimple.bind(models),
      toolExecution: "sequential",
      beforeToolCall: async ({ toolCall, args }, signal) => {
        if (toolCall.name !== "terminal") return undefined;
        signal?.throwIfAborted();
        const decision = await this.peer.request("permission.check", {
          tool_call_id: toolCall.id,
          tool_name: "terminal",
          arguments: isRecord(args) ? JSON.parse(JSON.stringify(args)) as JsonObject : {},
        }, signal);
        if (!isRecord(decision) || typeof decision.allow !== "boolean") throw new Error("invalid permission response");
        return decision.allow ? undefined : { block: true, reason: typeof decision.reason === "string" ? decision.reason : "User denied terminal command" };
      },
    });
    agent.subscribe((event) => {
      if (!this.normalizer) return;
      for (const translated of this.normalizer.translate(event, agent.state.messages)) this.peer.emit(translated);
    });
    this.agent = agent;
    this.sessionId = config.sessionId;
    return { ready: true };
  }

  private prompt(params: JsonObject, forceSteer: boolean): JsonValue {
    const agent = this.requireAgent();
    const runId = requiredString(params, "run_id");
    const message: AgentMessage = { role: "user", content: parseContent(params.content), timestamp: Date.now() };
    if (forceSteer || agent.state.isStreaming) {
      agent.steer(message);
      return { accepted: true, steered: true };
    }
    if (this.normalizer) throw new Error("Pi agent is already running");
    this.normalizer = new PiEventNormalizer(this.sessionId!, runId);
    setImmediate(() => void agent.prompt(message).catch((error) => this.unhandled(runId, error)).finally(() => { this.normalizer = undefined; }));
    return { accepted: true, steered: false };
  }

  private cancel(): JsonValue {
    this.requireAgent().abort();
    return { cancelled: true };
  }

  private async close(): Promise<JsonValue> {
    const agent = this.requireAgent();
    agent.abort();
    await agent.waitForIdle();
    setImmediate(() => process.exit(0));
    return { closed: true };
  }

  private requireAgent(): Agent {
    if (!this.agent) throw new Error("Pi session is not started");
    return this.agent;
  }

  private unhandled(runId: string, error: unknown): void {
    if (!this.sessionId) return;
    const event: RunnerEvent = { protocolVersion: PROTOCOL_VERSION, type: "pi.event", eventId: randomUUID(),
      sessionId: this.sessionId, runId, occurredAt: new Date().toISOString(), kind: "agent.failed",
      payload: { error_code: "runner_error", message: error instanceof Error ? error.message : "Pi runner failed" } };
    this.peer.emit(event);
  }
}

export function resolveModel(models: Models, config: Pick<SessionConfig, "provider" | "modelId" | "baseUrl">): Model<Api> {
  const catalog = models.getModel(config.provider, config.modelId);
  if (catalog) return config.baseUrl ? { ...catalog, baseUrl: config.baseUrl } : catalog;
  const template = models.getModels(config.provider)[0];
  if (!template || !config.baseUrl) throw new Error(`unknown Pi model: ${config.provider}/${config.modelId}`);
  return { ...template, id: config.modelId, name: config.modelId, baseUrl: config.baseUrl };
}

function parseConfig(value: JsonObject): SessionConfig {
  const model = record(value, "model");
  const thinking = typeof model.thinking_level === "string" ? model.thinking_level : "off";
  if (!["off", "minimal", "low", "medium", "high", "xhigh", "max"].includes(thinking)) throw new Error("invalid thinking level");
  return {
    sessionId: requiredString(value, "session_id"), provider: requiredString(model, "provider"), modelId: requiredString(model, "id"),
    apiKey: requiredString(model, "api_key"), baseUrl: typeof model.base_url === "string" ? model.base_url : undefined,
    systemPrompt: typeof value.system_prompt === "string" ? value.system_prompt : "You are a coding agent.",
    thinkingLevel: thinking as SessionConfig["thinkingLevel"],
    transcript: Array.isArray(value.transcript) ? value.transcript as unknown as AgentMessage[] : [],
    workspaceRoot: requiredString(value, "workspace_root"),
    skillRoot: requiredString(value, "skill_root"),
    tools: parseTools(value.tools),
  };
}

function parseTools(value: JsonValue | undefined): BusinessToolConfig[] {
  if (!Array.isArray(value)) throw new Error("tools must be an array");
  return value.map((item) => {
    if (!isRecord(item)) throw new Error("invalid tool configuration");
    return {
      name: requiredString(item, "name"),
      description: requiredString(item, "description"),
      inputSchema: record(item, "input_schema") as JsonObject,
    };
  });
}

type PromptBlock =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string };

function parseContent(value: JsonValue | undefined): PromptBlock[] {
  if (!Array.isArray(value)) throw new Error("content must be an array");
  const result: PromptBlock[] = [];
  for (const block of value) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      result.push({ type: "text", text: block.text });
    } else if (block.type === "image" && typeof block.data === "string" && typeof block.mime_type === "string") {
      result.push({ type: "image", data: block.data, mimeType: block.mime_type });
    }
  }
  return result;
}

function record(value: Record<string, unknown>, name: string): Record<string, unknown> {
  const item = value[name]; if (!isRecord(item)) throw new Error(`${name} must be an object`); return item;
}
function requiredString(value: Record<string, unknown>, name: string): string {
  const item = value[name]; if (typeof item !== "string" || !item) throw new Error(`${name} must be a string`); return item;
}
