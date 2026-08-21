import { lstat } from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";
import {
  createBashTool,
  createEditTool,
  createReadTool,
  createWriteTool,
  type AgentHarnessTool,
  type AgentTool,
  type ExecutionEnv,
  type ExecutionToolContext,
} from "@earendil-works/pi-agent-core";
import { type ImageContent, type TextContent, type TSchema } from "@earendil-works/pi-ai";
import type { InlineExtension } from "@earendil-works/pi-coding-agent";
import { isRecord, type JsonObject } from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";

export interface BusinessToolConfig {
  name: string;
  description: string;
  inputSchema: JsonObject;
}

export function createTools(
  peer: JsonlRpcPeer,
  env: ExecutionEnv,
  workspaceRoot: string,
  skillRoot: string,
  knowledgeRoot: string | undefined,
  businessTools: BusinessToolConfig[],
): AgentTool[] {
  const read = createReadTool();
  const write = createWriteTool();
  const edit = createEditTool();
  const bash = createBashTool({
    prepare(execution) {
      execution.cwd = workspaceRoot;
      execution.inheritEnv = false;
      execution.env = safeShellEnvironment();
    },
  });
  return [
    bindPathTool(read, env, workspaceRoot, skillRoot, knowledgeRoot, true),
    bindPathTool(write, env, workspaceRoot, skillRoot, knowledgeRoot, false),
    bindPathTool(edit, env, workspaceRoot, skillRoot, knowledgeRoot, false),
    { ...bindNative(bash, env), name: "terminal", label: "terminal" },
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
  workspaceRoot: string,
  skillRoot: string,
  knowledgeRoot: string | undefined,
  allowReadOnlyResources: boolean,
): AgentTool<any, any> {
  const bound = bindNative(tool, env);
  return {
    ...bound,
    async execute(callId, params: any, signal, onUpdate) {
      if (!isRecord(params) || typeof params.path !== "string") throw new Error("path must be a string");
      const safe = {
        ...params,
        path: await safePath(
          params.path,
          workspaceRoot,
          skillRoot,
          knowledgeRoot,
          allowReadOnlyResources,
        ),
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
      if (response.is_error) throw new Error(content.find((block) => block.type === "text")?.text ?? "tool failed");
      return { content, details: isRecord(response.details) ? response.details : undefined };
    },
  };
}

export async function safePath(
  input: string,
  workspaceRoot: string,
  skillRoot: string,
  knowledgeRoot: string | undefined,
  allowReadOnlyResources: boolean,
): Promise<string> {
  if (!input || input.split(/[\\/]/).includes("..")) throw new Error(`unsafe path: ${input}`);
  const workspace = resolve(workspaceRoot);
  const skill = resolve(skillRoot);
  const knowledge = knowledgeRoot ? resolve(knowledgeRoot) : undefined;

  if (!isAbsolute(input) && input.split(/[\\/]/)[0] === "knowledge") {
    if (!allowReadOnlyResources) {
      throw new Error(
        "PATH_SCOPE_ERROR: knowledge is read-only; write and edit only workspace-relative paths",
      );
    }
    if (!knowledge) {
      throw new Error("PATH_SCOPE_ERROR: knowledge/ is not configured for this Pi session");
    }
    const parts = input.split(/[\\/]/).filter(Boolean).slice(1);
    const target = resolve(knowledge, ...parts);
    if (!inside(target, knowledge)) throw new Error(`unsafe path: ${input}`);
    await rejectSymlinks(knowledge, target);
    return target;
  }

  if (isAbsolute(input)) {
    const target = resolve(input);
    if (!allowReadOnlyResources) {
      throw new Error(
        "PATH_SCOPE_ERROR: write and edit paths must be workspace-relative; skill and knowledge files are read-only",
      );
    }
    if (inside(target, skill)) {
      await rejectSymlinks(skill, target);
      return target;
    }
    if (knowledge && inside(target, knowledge)) {
      await rejectSymlinks(knowledge, target);
      return target;
    }
    if (inside(target, workspace)) {
      const suggested = relative(workspace, target) || ".";
      throw new Error(
        `PATH_SCOPE_ERROR: workspace files must use workspace-relative paths; use ${suggested}`,
      );
    }
    throw new Error(
      "PATH_SCOPE_ERROR: read paths must be workspace-relative, knowledge/..., or an absolute skill location advertised by Pi",
    );
  }

  const target = resolve(workspace, input);
  if (!inside(target, workspace)) throw new Error(`unsafe path: ${input}`);
  await rejectSymlinks(workspace, target);
  return target;
}

function inside(target: string, root: string): boolean {
  const rel = relative(root, target);
  return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel));
}

async function rejectSymlinks(root: string, target: string): Promise<void> {
  if ((await lstat(root)).isSymbolicLink()) throw new Error(`allowed root must not be a symlink: ${root}`);
  const parts = relative(root, target).split(sep).filter(Boolean);
  let current = root;
  for (const part of parts) {
    current = resolve(current, part);
    try {
      if ((await lstat(current)).isSymbolicLink()) throw new Error(`symlink paths are not allowed: ${current}`);
    } catch (error) {
      if (isRecord(error) && error.code === "ENOENT") return;
      throw error;
    }
  }
}

function safeShellEnvironment(): Record<string, string> {
  const names = ["PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS"];
  return Object.fromEntries(names.flatMap((name) => process.env[name] ? [[name, process.env[name]!]] : []));
}
