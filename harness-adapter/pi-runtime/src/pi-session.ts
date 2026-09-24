import { realpathSync } from "node:fs";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import {
  type AgentSession,
  createAgentSession,
  DefaultResourceLoader,
  SessionManager,
  SettingsManager,
  type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { createPiModelRuntime, type PiModelConfig } from "./pi-model.js";
import { createNoProgressGuardExtension } from "./no-progress-guard.js";
import { isRecord, type JsonObject, type JsonValue } from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";
import type { SandboxEndpoint, SandboxEndpointProvider } from "./sandbox-operations.js";
import { applySandboxPathAliases } from "./sandbox-paths.js";
import { inside } from "./workspace-policy.js";
import {
  createTerminalPermissionExtension,
  createTools,
  type BusinessToolConfig,
  type CreateToolsOptions,
  type SkillRootConfig,
} from "./tools.js";
import type {
  PiTerminalBackend,
  ResourceLimitsConfig,
} from "./workspace-sandbox.js";

interface PiSessionConfig extends PiModelConfig {
  sessionId: string;
  systemPrompt: string;
  thinkingLevel: "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
  workspaceRoot: string;
  terminalBackend: PiTerminalBackend;
  sessionPath: string;
  skillRoots: SkillRootConfig[];
  skillsDirectory?: string;
  knowledgeRoot?: string;
  resourceLimits?: ResourceLimitsConfig;
  tools: BusinessToolConfig[];
}

export interface ParsedPrompt {
  text: string;
  images: Array<{ type: "image"; data: string; mimeType: string }>;
}

export async function createPiSession(params: JsonObject, peer: JsonlRpcPeer): Promise<{
  session: AgentSession;
  sessionId: string;
}> {
  const config = parseConfig(params);
  const { modelRuntime, model } = await createPiModelRuntime(config);
  const env = new NodeExecutionEnv({ cwd: config.workspaceRoot, shellEnv: safeShellEnvironment() });
  const options: CreateToolsOptions = {
    skillsDirectory: config.skillsDirectory,
    ...(config.terminalBackend === "sandbox"
      ? { sandbox: sandboxEndpointProvider(peer) }
      : {}),
  };
  const tools = await createTools(
    peer,
    env,
    config.workspaceRoot,
    config.terminalBackend,
    config.skillRoots,
    config.knowledgeRoot,
    config.resourceLimits,
    config.tools,
    options,
  );
  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: true, reserveTokens: 0, keepRecentTokens: 20_000 },
    retry: { enabled: false, maxRetries: 0 },
    defaultThinkingLevel: config.thinkingLevel,
  });
  const agentDir = join(config.workspaceRoot, "pi", "agent");
  const skillPaths = [
    ...config.skillRoots.map((root) => root.path),
    ...(config.skillsDirectory ? [config.skillsDirectory] : []),
  ].map((path) => realpathSync(path));
  // Sandbox resources are uploaded under the runtime aliases, so the skill
  // locations baked into the system prompt must not expose host paths.
  const sandboxPathAliases =
    config.terminalBackend === "sandbox"
      ? [
          ...config.skillRoots.map((root) => ({
            host: realpathSync(root.path),
            alias: `.agents/skills/${root.name}`,
          })),
          ...(config.skillsDirectory
            ? [{ host: realpathSync(config.skillsDirectory), alias: ".agents/skills" }]
            : []),
          ...(config.knowledgeRoot
            ? [{ host: realpathSync(config.knowledgeRoot), alias: "knowledge" }]
            : []),
        ].sort((left, right) => right.host.length - left.host.length)
      : [];
  const resourceLoader = new DefaultResourceLoader({
    cwd: config.workspaceRoot,
    agentDir,
    settingsManager,
    systemPrompt: config.systemPrompt,
    extensionFactories: [
      createTerminalPermissionExtension(peer),
      createNoProgressGuardExtension(),
      (pi) => {
        pi.on("before_agent_start", (event) => {
          let systemPrompt = event.systemPrompt.replace(
            "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
            "Paths starting with knowledge/ or .agents/skills/ are runtime resource aliases: pass them unchanged to file tools, without prepending the skill directory. For other relative paths referenced by a skill file, resolve them against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
          );
          return {
            systemPrompt: applySandboxPathAliases(systemPrompt, sandboxPathAliases),
          };
        });
      },
    ],
    additionalSkillPaths: skillPaths,
    skillsOverride: ({ skills, diagnostics }) => ({
      skills: skills.filter((skill) => {
        try {
          const path = realpathSync(skill.filePath);
          return skillPaths.some((root) => inside(path, root));
        } catch {
          return false;
        }
      }),
      diagnostics,
    }),
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    noContextFiles: true,
  });
  await resourceLoader.reload();
  const sessionManager = SessionManager.open(
    config.sessionPath,
    join(config.workspaceRoot, "pi"),
    config.workspaceRoot,
  );
  const { session } = await createAgentSession({
    cwd: config.workspaceRoot,
    agentDir,
    modelRuntime,
    model,
    thinkingLevel: config.thinkingLevel,
    noTools: "builtin",
    customTools: tools.map(toolDefinition),
    resourceLoader,
    sessionManager,
    settingsManager,
  });
  return { session, sessionId: config.sessionId };
}

export function parsePromptContent(value: JsonValue | undefined): ParsedPrompt {
  if (!Array.isArray(value)) throw new Error("content must be an array");
  const text: string[] = [];
  const images: ParsedPrompt["images"] = [];
  for (const block of value) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      text.push(block.text);
    } else if (block.type === "image" && typeof block.data === "string" && typeof block.mime_type === "string") {
      images.push({ type: "image", data: block.data, mimeType: block.mime_type });
    }
  }
  return { text: text.join("\n"), images };
}

function parseConfig(value: JsonObject): PiSessionConfig {
  const model = record(value, "model");
  const thinking = typeof model.thinking_level === "string" ? model.thinking_level : "off";
  if (!["off", "minimal", "low", "medium", "high", "xhigh", "max"].includes(thinking)) {
    throw new Error("invalid thinking level");
  }
  const api = model.api;
  if (api !== undefined && api !== "openai-completions" && api !== "openai-responses") {
    throw new Error("invalid model API");
  }
  return {
    sessionId: requiredString(value, "session_id"),
    provider: requiredString(model, "provider"),
    modelId: requiredString(model, "id"),
    apiKey: requiredString(model, "api_key"),
    baseUrl: typeof model.base_url === "string" ? model.base_url : undefined,
    api,
    contextWindow: positiveInteger(model.context_window, "context_window"),
    systemPrompt: typeof value.system_prompt === "string" ? value.system_prompt : "You are a coding agent.",
    thinkingLevel: thinking as PiSessionConfig["thinkingLevel"],
    workspaceRoot: requiredString(value, "workspace_root"),
    terminalBackend: terminalBackend(value),
    sessionPath: requiredString(value, "session_path"),
    skillRoots: parseSkillRoots(value),
    skillsDirectory: optionalString(value, "skills_directory"),
    knowledgeRoot: optionalString(value, "knowledge_root"),
    resourceLimits: parseResourceLimits(value.resource_limits),
    tools: parseTools(value.tools),
  };
}

function parseResourceLimits(
  value: JsonValue | undefined,
): ResourceLimitsConfig | undefined {
  if (value === undefined) return undefined;
  if (!isRecord(value)) throw new Error("resource_limits must be an object");
  const resourceLimits = {
    memoryLimitBytes: requiredInteger(
      value.memory_limit_bytes,
      "resource_limits.memory_limit_bytes",
    ),
    nprocLimit: requiredInteger(value.nproc_limit, "resource_limits.nproc_limit"),
  };
  if (resourceLimits.nprocLimit < 2) {
    throw new Error("resource_limits.nproc_limit must be at least 2");
  }
  return resourceLimits;
}

function requiredInteger(value: unknown, name: string): number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value as number;
}

function terminalBackend(value: JsonObject): PiTerminalBackend {
  const backend = requiredString(value, "terminal_backend");
  if (backend !== "os-sandbox" && backend !== "sandbox") {
    throw new Error("invalid terminal_backend; expected one of: os-sandbox, sandbox");
  }
  return backend;
}

/**
 * The control plane owns sandbox lifecycle and credentials. The runner asks for
 * an endpoint only when it is about to touch the execution workspace, so the
 * access token never travels in the start frame and idle chats create nothing.
 */
function sandboxEndpointProvider(peer: JsonlRpcPeer): SandboxEndpointProvider {
  return async ({ refresh }) => {
    const result = await peer.request("sandbox.ensure", { refresh });
    if (!isRecord(result)) throw new Error("invalid sandbox.ensure response");
    const endpoint: SandboxEndpoint = {
      baseUrl: requiredString(result, "base_url"),
      wsBaseUrl: requiredString(result, "ws_base_url"),
      sandboxId: requiredString(result, "sandbox_id"),
      apiKey: requiredString(result, "api_key"),
      workspacePath: requiredString(result, "workspace_path"),
      storagePath: requiredString(result, "storage_path"),
    };
    const cluster = result.cluster;
    if (typeof cluster === "string" && cluster) endpoint.cluster = cluster;
    return endpoint;
  };
}

function parseSkillRoots(value: JsonObject): SkillRootConfig[] {
  if (Array.isArray(value.skill_roots)) {
    if (value.skill_roots.length === 0 && !optionalString(value, "skills_directory")) {
      throw new Error("skill_roots must not be empty without skills_directory");
    }
    const names = new Set<string>();
    return value.skill_roots.map((item) => {
      if (!isRecord(item)) throw new Error("invalid skill root configuration");
      const name = requiredString(item, "name");
      if (names.has(name)) throw new Error(`duplicate skill root name: ${name}`);
      names.add(name);
      return { name, path: requiredString(item, "path") };
    });
  }
  if (value.skill_root === undefined && optionalString(value, "skills_directory")) return [];
  return [{ name: "skill", path: requiredString(value, "skill_root") }];
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

function record(value: Record<string, unknown>, name: string): Record<string, unknown> {
  const item = value[name];
  if (!isRecord(item)) throw new Error(`${name} must be an object`);
  return item;
}

function requiredString(value: Record<string, unknown>, name: string): string {
  const item = value[name];
  if (typeof item !== "string" || !item) throw new Error(`${name} must be a string`);
  return item;
}

function optionalString(value: Record<string, unknown>, name: string): string | undefined {
  const item = value[name];
  if (item === undefined) return undefined;
  if (typeof item !== "string" || !item) throw new Error(`${name} must be a string`);
  return item;
}

function positiveInteger(value: unknown, name: string): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value as number;
}

function toolDefinition(tool: AgentTool): ToolDefinition {
  return {
    ...tool,
    async execute(callId, params, signal, onUpdate) {
      return tool.execute(callId, params, signal, onUpdate);
    },
  } as ToolDefinition;
}

function safeShellEnvironment(): Record<string, string> {
  const names = ["PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS"];
  return Object.fromEntries(names.flatMap((name) => process.env[name] ? [[name, process.env[name]!]] : []));
}
