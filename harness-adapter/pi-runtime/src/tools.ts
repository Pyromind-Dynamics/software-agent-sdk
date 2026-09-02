import {
  createEditTool,
  createReadTool,
  createWriteTool,
  type AgentHarnessTool,
  type AgentTool,
  type ExecutionEnv,
  type ExecutionToolContext,
} from "@earendil-works/pi-agent-core";
import { type ImageContent, type TextContent, type TSchema } from "@earendil-works/pi-ai";
import {
  createBashTool,
  type InlineExtension,
} from "@earendil-works/pi-coding-agent";
import { isRecord, type JsonObject } from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";
import {
  createWorkspaceBashOperations,
  type PiTerminalBackend,
} from "./workspace-sandbox.js";
import {
  WorkspaceAccessPolicy,
  type WorkspacePathOperation,
} from "./workspace-policy.js";

const OPENHANDS_ERROR_HEADER = "[An error occurred during execution.]";

export interface BusinessToolConfig {
  name: string;
  description: string;
  inputSchema: JsonObject;
}

export interface SkillRootConfig {
  name: string;
  path: string;
}

export async function createTools(
  peer: JsonlRpcPeer,
  env: ExecutionEnv,
  workspaceRoot: string,
  terminalBackend: PiTerminalBackend,
  skillRoots: SkillRootConfig[],
  knowledgeRoot: string | undefined,
  businessTools: BusinessToolConfig[],
): Promise<AgentTool[]> {
  const policy = await WorkspaceAccessPolicy.create({
    workspaceRoot,
    readOnlyRoots: skillRoots.map((root) => root.path),
    knowledgeRoot,
  });
  const terminalOutputTemp = policy.terminalTempRoot;
  const read = createReadTool();
  const write = createWriteTool();
  const edit = createEditTool();
  const terminalOperations = await createWorkspaceBashOperations(
    terminalBackend,
    policy,
  );
  // sandbox-runtime creates Linux bridge sockets under os.tmpdir(). Initialize
  // it before pointing process temp variables at the conversation's much longer
  // terminal-output path, which can exceed sockaddr_un.sun_path's 108-byte limit.
  process.env.TMPDIR = terminalOutputTemp;
  process.env.TMP = terminalOutputTemp;
  process.env.TEMP = terminalOutputTemp;
  const bash = createBashTool(workspaceRoot, {
    operations: terminalOperations,
    exposeSessionEnvironment: false,
    spawnHook: ({ command }) => ({
      command,
      cwd: workspaceRoot,
      env: safeShellEnvironment(),
    }),
  });
  return [
    bindPathTool(read, env, policy, "read"),
    bindPathTool(write, env, policy, "write"),
    bindPathTool(edit, env, policy, "write"),
    {
      ...bash,
      name: "terminal",
      label: "terminal",
      async execute(callId, params: any, signal, onUpdate) {
        return bash.execute(callId, params, signal, onUpdate);
      },
    },
    ...businessTools.map((config) => businessTool(peer, config)),
  ];
}

export function createTerminalPermissionExtension(peer: JsonlRpcPeer): InlineExtension {
  return (pi) => {
    pi.on("tool_call", async (event) => {
      if (event.toolName !== "terminal") return undefined;
      const decision = await peer.request("permission.check", {
        tool_call_id: event.toolCallId,
        tool_name: "terminal",
        arguments: JSON.parse(JSON.stringify(event.input)) as JsonObject,
      });
      if (!isRecord(decision) || typeof decision.allow !== "boolean") {
        throw new Error("invalid permission response");
      }
      return decision.allow ? undefined : {
        block: true,
        reason: typeof decision.reason === "string" ? decision.reason : "User denied terminal command",
      };
    });
  };
}

function bindNative(
  tool: AgentHarnessTool<ExecutionToolContext, any, any>,
  env: ExecutionEnv,
): AgentTool<any, any> {
  return {
    ...tool,
    async execute(callId, params, signal, onUpdate) {
      return tool.execute(callId, params, signal, onUpdate, { env });
    },
  };
}

function bindPathTool(
  tool: AgentHarnessTool<ExecutionToolContext, any, any>,
  env: ExecutionEnv,
  policy: WorkspaceAccessPolicy,
  operation: WorkspacePathOperation,
): AgentTool<any, any> {
  const bound = bindNative(tool, env);
  const pathScope = operation === "read"
    ? "Conversation files are read from public_data/. Advertised skill paths and knowledge/ are read-only. Relative paths start at the conversation root; authorized absolute paths are also accepted."
    : "Write and edit paths must stay within public_data/. Relative paths start at the conversation root; authorized absolute paths are also accepted.";
  const parameters = structuredClone(bound.parameters);
  if (isRecord(parameters.properties) && isRecord(parameters.properties.path)) {
    parameters.properties.path.description = pathScope;
  }
  return {
    ...bound,
    description: `${bound.description}\n\n${pathScope}`,
    parameters,
    async execute(callId, params: any, signal, onUpdate) {
      if (!isRecord(params) || typeof params.path !== "string") throw new Error("path must be a string");
      const safe = {
        ...params,
        path: await policy.resolvePath(params.path, operation),
      };
      return tool.execute(callId, safe, signal, onUpdate, { env });
    },
  };
}

function businessTool(peer: JsonlRpcPeer, config: BusinessToolConfig): AgentTool<TSchema, unknown> {
  return {
    name: config.name,
    label: config.name,
    description: config.description,
    parameters: config.inputSchema as TSchema,
    async execute(callId, params, signal) {
      if (!isRecord(params)) throw new Error("tool arguments must be an object");
      const response = await peer.request("tool.execute", {
        tool_call_id: callId,
        tool_name: config.name,
        arguments: JSON.parse(JSON.stringify(params)) as JsonObject,
      }, signal);
      if (!isRecord(response) || typeof response.is_error !== "boolean" || !Array.isArray(response.content)) {
        throw new Error("invalid Python tool response");
      }
      const content: Array<TextContent | ImageContent> = [];
      for (const block of response.content) {
        if (!isRecord(block)) continue;
        if (block.type === "text" && typeof block.text === "string") content.push({ type: "text", text: block.text });
        else if (block.type === "image" && typeof block.data === "string" && typeof block.mime_type === "string") {
          content.push({ type: "image", data: block.data, mimeType: block.mime_type });
        }
      }
      if (response.is_error) {
        const message = content
          .filter((block): block is TextContent => block.type === "text")
          .map((block) => block.text.trim())
          .filter((text) => text && text !== OPENHANDS_ERROR_HEADER)
          .join("\n");
        throw new Error(message || "tool failed");
      }
      return { content, details: isRecord(response.details) ? response.details : undefined };
    },
  };
}

export async function safePath(
  input: string,
  workspaceRoot: string,
  skillRoots: SkillRootConfig[] | string,
  knowledgeRoot: string | undefined,
  allowReadOnlyResources: boolean,
): Promise<string> {
  const skills = (typeof skillRoots === "string"
    ? [{ name: "skill", path: skillRoots }]
    : skillRoots);
  const policy = await WorkspaceAccessPolicy.create({
    workspaceRoot,
    readOnlyRoots: skills.map((root) => root.path),
    knowledgeRoot,
  });
  return policy.resolvePath(input, allowReadOnlyResources ? "read" : "write");
}

function safeShellEnvironment(): Record<string, string> {
  const names = ["PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS"];
  return Object.fromEntries(names.flatMap((name) => process.env[name] ? [[name, process.env[name]!]] : []));
}
