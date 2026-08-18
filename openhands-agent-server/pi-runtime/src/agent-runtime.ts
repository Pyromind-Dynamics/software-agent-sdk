import { randomUUID } from "node:crypto";
import { setImmediate } from "node:timers";
import {
  Agent,
  type AgentHarnessTool,
  type AgentTool,
  createBashTool,
  createEditTool,
  createReadTool,
  createWriteTool,
  type ExecutionEnv,
  type ExecutionToolContext,
} from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import {
  type Api,
  type ImageContent,
  InMemoryCredentialStore,
  type Model,
  type Models,
  type TextContent,
  type TSchema,
} from "@earendil-works/pi-ai";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";
import { PiEventNormalizer } from "./pi-events.ts";
import { PROTOCOL_VERSION, type JsonObject, type JsonValue, type PiRunnerEvent } from "./protocol.ts";
import { RemoteExecutionEnv } from "./remote-execution-env.ts";
import type { JsonlRpcPeer } from "./rpc-peer.ts";
import { extendSystemPromptWithSkills } from "./skill-prompt.ts";

interface SessionConfig {
  sessionId: string;
  workspaceRoot: string;
  executionEnv: "node" | "remote";
  provider: string;
  modelId: string;
  apiKey: string;
  baseUrl: string | undefined;
  systemPrompt: string;
  thinkingLevel: "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
  nativeTools: ReadonlySet<string>;
  skillDirs: string[];
  tools: ToolConfig[];
}

interface ToolConfig {
  name: string;
  description: string;
  inputSchema: JsonObject;
}

export class PiAgentRuntime {
  private readonly peer: JsonlRpcPeer;
  private agent: Agent | undefined;
  private env: ExecutionEnv | undefined;
  private sessionId: string | undefined;

  constructor(peer: JsonlRpcPeer) {
    this.peer = peer;
  }

  async handle(method: string, params: JsonObject): Promise<JsonValue> {
    if (method === "session.start") return this.start(params);
    if (method === "run.prompt") return this.prompt(params);
    if (method === "run.cancel") return this.cancel();
    if (method === "session.close") return this.close();
    throw new Error(`Unknown runner method: ${method}`);
  }

  private async start(params: JsonObject): Promise<JsonValue> {
    if (this.agent) throw new Error("Pi session is already initialized");
    const config = parseSessionConfig(params);
    const credentials = new InMemoryCredentialStore();
    await credentials.modify(config.provider, async () => ({ type: "api_key", key: config.apiKey }));
    const models = builtinModels({ credentials });
    const model = resolveConfiguredModel(models, config);

    this.sessionId = config.sessionId;
    this.env =
      config.executionEnv === "node"
        ? new NodeExecutionEnv({ cwd: config.workspaceRoot })
        : new RemoteExecutionEnv(this.peer, config.workspaceRoot);
    const systemPrompt = await extendSystemPromptWithSkills(
      config.systemPrompt,
      this.env,
      config.skillDirs,
    );
    const tools = [
      ...nativeTools(this.env, config.nativeTools),
      ...config.tools.map((tool) => businessTool(this.peer, tool)),
    ];
    const agent = new Agent({
      initialState: {
        systemPrompt,
        model,
        thinkingLevel: config.thinkingLevel,
        tools,
      },
      sessionId: config.sessionId,
      streamFn: models.streamSimple.bind(models),
      toolExecution: "parallel",
    });
    agent.subscribe((event) => {
      const normalizer = this.activeNormalizer;
      if (!normalizer) return;
      for (const translated of normalizer.translate(event)) this.peer.emit(translated);
    });
    this.agent = agent;
    return { ready: true };
  }

  private activeNormalizer: PiEventNormalizer | undefined;

  private prompt(params: JsonObject): JsonValue {
    const agent = this.requireAgent();
    if (agent.state.isStreaming || this.activeNormalizer) throw new Error("Pi agent is already running");
    const runId = requiredString(params, "command_id");
    const content = parseContent(params.content);
    const sessionId = this.sessionId;
    if (!sessionId) throw new Error("Pi session is not initialized");
    const workspaceRoot = this.env?.cwd ?? "/workspace";
    this.activeNormalizer = new PiEventNormalizer(sessionId, runId, workspaceRoot);
    setImmediate(() => {
      void agent
        .prompt({ role: "user", content, timestamp: Date.now() })
        .catch((error: unknown) => this.emitUnhandledFailure(runId, error))
        .finally(() => {
          this.activeNormalizer = undefined;
        });
    });
    return { accepted: true };
  }

  private cancel(): JsonValue {
    const agent = this.requireAgent();
    agent.abort();
    return { cancelled: true };
  }

  private async close(): Promise<JsonValue> {
    const agent = this.requireAgent();
    agent.abort();
    await agent.waitForIdle();
    await this.env?.cleanup();
    setImmediate(() => process.exit(0));
    return { closed: true };
  }

  private requireAgent(): Agent {
    if (!this.agent) throw new Error("Pi session is not initialized");
    return this.agent;
  }

  private emitUnhandledFailure(runId: string, error: unknown): void {
    const sessionId = this.sessionId;
    if (!sessionId) return;
    const event: PiRunnerEvent = {
      protocolVersion: PROTOCOL_VERSION,
      type: "pi.event",
      eventId: randomUUID(),
      sessionId,
      runId,
      occurredAt: new Date().toISOString(),
      kind: "agent.failed",
      payload: {
        error_code: "runner_error",
        message: error instanceof Error ? error.message : "Pi runner failed",
      },
    };
    this.peer.emit(event);
  }
}

export function resolveConfiguredModel(
  models: Models,
  config: Pick<SessionConfig, "provider" | "modelId" | "baseUrl">,
): Model<Api> {
  const catalogModel = models.getModel(config.provider, config.modelId);
  if (catalogModel) {
    return config.baseUrl ? { ...catalogModel, baseUrl: config.baseUrl } : catalogModel;
  }

  if (!config.baseUrl) {
    throw new Error(`Unknown Pi model: ${config.provider}/${config.modelId}`);
  }
  const template = models.getModels(config.provider)[0];
  if (!template) {
    throw new Error(`Unknown Pi provider: ${config.provider}`);
  }
  return {
    ...template,
    id: config.modelId,
    name: config.modelId,
    baseUrl: config.baseUrl,
  };
}

function nativeTools(env: ExecutionEnv, enabled: ReadonlySet<string>): AgentTool[] {
  const tools: AgentTool[] = [];
  const read = createReadTool();
  const write = createWriteTool();
  const edit = createEditTool();
  const bash = createBashTool();
  if (enabled.has(read.name)) tools.push(bindNativeTool(read, env));
  if (enabled.has(write.name)) tools.push(bindNativeTool(write, env));
  if (enabled.has(edit.name)) tools.push(bindNativeTool(edit, env));
  if (enabled.has(bash.name)) tools.push(bindNativeTool(bash, env));
  return tools;
}

function bindNativeTool<TParameters extends TSchema, TDetails>(
  tool: AgentHarnessTool<ExecutionToolContext, TParameters, TDetails>,
  env: ExecutionEnv,
): AgentTool<TParameters, TDetails> {
  return {
    ...tool,
    execute: (toolCallId, params, signal, onUpdate) =>
      tool.execute(toolCallId, params, signal, onUpdate, { env }),
  };
}

function businessTool(peer: JsonlRpcPeer, config: ToolConfig): AgentTool<TSchema, unknown> {
  return {
    name: config.name,
    label: config.name,
    description: config.description,
    parameters: config.inputSchema,
    async execute(_toolCallId, params, signal) {
      if (!isRecord(params)) throw new Error("Business tool arguments must be an object");
      const value = await peer.request(
        "tool.execute",
        { tool_name: config.name, arguments: toJsonObject(params) },
        signal,
      );
      if (!isRecord(value) || typeof value.is_error !== "boolean" || !Array.isArray(value.content)) {
        throw new Error("Invalid Python tool result");
      }
      const content: Array<TextContent | ImageContent> = [];
      for (const block of value.content) {
        if (!isRecord(block)) continue;
        if (block.type === "text" && typeof block.text === "string") {
          content.push({ type: "text", text: block.text });
          continue;
        }
        if (
          block.type === "image" &&
          typeof block.data === "string" &&
          typeof block.mime_type === "string"
        ) {
          content.push({ type: "image", data: block.data, mimeType: block.mime_type });
        }
      }
      if (value.is_error) {
        const text = content.find((block) => block.type === "text")?.text ?? "Business tool failed";
        throw new Error(text);
      }
      const result = {
        content,
        ...(isRecord(value.details) ? { details: toJsonObject(value.details) } : { details: undefined }),
      };
      return result;
    },
  };
}

function parseSessionConfig(params: JsonObject): SessionConfig {
  const runtime = requiredRecord(params, "runtime_config");
  const model = requiredRecord(runtime, "model");
  const workspace = requiredRecord(params, "workspace");
  const sandbox = requiredRecord(params, "sandbox");
  const workspaceRoot =
    typeof runtime.workspace_root === "string"
      ? runtime.workspace_root
      : requiredString(workspace, "root");
  const thinking = typeof model.thinking_level === "string" ? model.thinking_level : "off";
  if (!isThinkingLevel(thinking)) throw new Error("Invalid Pi thinking level");
  const native = Array.isArray(runtime.native_tools)
    ? runtime.native_tools.filter((name): name is string => typeof name === "string")
    : ["read", "write", "edit", "bash"];
  const executionEnv =
    runtime.execution_env === undefined ? "node" : runtime.execution_env;
  if (executionEnv !== "node" && executionEnv !== "remote") {
    throw new Error("execution_env must be 'node' or 'remote'");
  }
  return {
    sessionId: requiredString(params, "session_id"),
    workspaceRoot,
    executionEnv,
    provider: requiredString(model, "provider"),
    modelId: requiredString(model, "id"),
    apiKey: requiredString(model, "api_key"),
    baseUrl: typeof model.base_url === "string" ? model.base_url : undefined,
    systemPrompt: typeof runtime.system_prompt === "string" ? runtime.system_prompt : "You are a coding agent.",
    thinkingLevel: thinking,
    nativeTools: new Set(native),
    skillDirs: optionalStringArray(runtime, "skill_dirs"),
    tools: parseTools(params.tools),
  };
}

function optionalStringArray(value: Record<string, unknown>, name: string): string[] {
  const field = value[name];
  if (field === undefined) return [];
  if (!Array.isArray(field) || !field.every((item) => typeof item === "string" && item.length > 0)) {
    throw new Error(`${name} must be an array of non-empty strings`);
  }
  return field;
}

function parseTools(value: JsonValue | undefined): ToolConfig[] {
  if (!Array.isArray(value)) throw new Error("tools must be an array");
  return value.map((item) => {
    if (!isRecord(item)) throw new Error("Invalid tool spec");
    return {
      name: requiredString(item, "name"),
      description: requiredString(item, "description"),
      inputSchema: toJsonObject(requiredRecord(item, "input_schema")),
    };
  });
}

function parseContent(value: JsonValue | undefined): Array<
  { type: "text"; text: string } | { type: "image"; data: string; mimeType: string }
> {
  if (!Array.isArray(value)) throw new Error("content must be an array");
  const content: Array<
    { type: "text"; text: string } | { type: "image"; data: string; mimeType: string }
  > = [];
  for (const block of value) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      content.push({ type: "text", text: block.text });
      continue;
    }
    if (
      block.type === "image" &&
      typeof block.data === "string" &&
      typeof block.mime_type === "string"
    ) {
      content.push({ type: "image", data: block.data, mimeType: block.mime_type });
    }
  }
  return content;
}

function requiredRecord(value: Record<string, unknown>, name: string): Record<string, unknown> {
  const field = value[name];
  if (!isRecord(field)) throw new Error(`${name} must be an object`);
  return field;
}

function requiredString(value: Record<string, unknown>, name: string): string {
  const field = value[name];
  if (typeof field !== "string" || field.length === 0) throw new Error(`${name} must be a string`);
  return field;
}

function isThinkingLevel(value: string): value is SessionConfig["thinkingLevel"] {
  return ["off", "minimal", "low", "medium", "high", "xhigh", "max"].includes(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function toJsonObject(value: Record<string, unknown>): JsonObject {
  return JSON.parse(JSON.stringify(value)) as JsonObject;
}
