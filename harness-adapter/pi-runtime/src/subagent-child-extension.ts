import { createAssistantMessageEventStream, type AssistantMessage } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { callSubagentHost } from "./subagent-ipc.js";

/** Loaded by pi-subagents in both its foreground and detached child sessions. */
export async function installChildHost(pi: ExtensionAPI, role: "search" | "general_purpose") {
  const socket = process.env.PYROMIND_SUBAGENT_SOCKET;
  if (!socket) throw new Error("Missing conversation subagent host");
  const config = await callSubagentHost(socket, { method: "config", role });
  const model = config.model;
  pi.registerProvider(model.provider, {
    baseUrl: model.baseUrl, api: model.api, apiKey: "host-mediated",
    models: [model],
    streamSimple: (_model, context, options) => {
      const stream = createAssistantMessageEventStream();
      void callSubagentHost(socket, {
        method: "model", role, context,
        options: { reasoning: options?.reasoning, maxTokens: options?.maxTokens },
      }, options?.signal, (event) => stream.push(event)).then(() => stream.end()).catch((error) => {
        const message: AssistantMessage = {
          role: "assistant", content: [], api: model.api, provider: model.provider, model: model.id,
          usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
          stopReason: options?.signal?.aborted ? "aborted" : "error", errorMessage: error.message, timestamp: Date.now(),
        };
        stream.push({ type: "error", reason: message.stopReason as "error" | "aborted", error: message });
        stream.end();
      });
      return stream;
    },
  });
  for (const tool of config.tools) {
    pi.registerTool({ ...tool, execute: (id: string, args: unknown, signal: AbortSignal | undefined, update: any, ctx: any) =>
      callSubagentHost(socket, { method: "tool", role, name: tool.name, id: `${ctx.sessionManager.getLeafId()}:${id}`, args, session: ctx.sessionManager.getSessionId() }, signal, update) });
  }
  pi.on("session_start", async (_event, ctx) => {
    await callSubagentHost(socket, { method: "config", role });
    const allowed = new Set(config.tools.map((tool: any) => tool.name));
    if (pi.getActiveTools().some((name) => !allowed.has(name))) throw new Error("Unexpected tool in subagent session");
  });
  pi.on("before_agent_start", (event) => ({ systemPrompt: event.systemPrompt + "\n" + config.instructions }));
}
