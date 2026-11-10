import { createAssistantMessageEventStream, type AssistantMessage } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { callSubagentHost } from "./subagent-ipc.js";

/** Loaded by pi-subagents in both its foreground and detached child sessions. */
export async function installChildHost(pi: ExtensionAPI, role: string) {
  const socket = process.env.PYROMIND_SUBAGENT_SOCKET;
  if (!socket) throw new Error("Missing conversation subagent host");
  const config = await callSubagentHost(socket, { method: "config", role });
  const terminalTools = new Set<string>(config.terminalTools);
  let attachment: {
    controller: AbortController;
    ready: Promise<string>;
    identity: { session: string; session_file: string | undefined };
  } | undefined;
  const terminalLease = (ctx: ExtensionContext) => {
    if (attachment) return attachment.ready;
    const controller = new AbortController();
    const { promise: ready, resolve, reject } = Promise.withResolvers<string>();
    const identity = { session: ctx.sessionManager.getSessionId(), session_file: ctx.sessionManager.getSessionFile() };
    attachment = { controller, ready, identity };
    void callSubagentHost(socket, {
      method: "terminal.attach", role, ...identity,
    }, controller.signal, (value) => resolve(value.id), true).catch(reject).finally(() => {
      if (attachment?.controller === controller) attachment = undefined;
    });
    return ready;
  };
  pi.on("session_shutdown", () => { attachment?.controller.abort(); });
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
    pi.registerTool({ ...tool, execute: async (id: string, args: unknown, signal: AbortSignal | undefined, update: any, ctx: ExtensionContext) => {
      const terminal_lease = terminalTools.has(tool.name) ? await terminalLease(ctx) : undefined;
      return callSubagentHost(socket, { method: "tool", role, terminal_lease, name: tool.name, id: `${ctx.sessionManager.getLeafId()}:${id}`, args, session: ctx.sessionManager.getSessionId(), session_file: ctx.sessionManager.getSessionFile() }, signal, update);
    } });
  }
  pi.on("session_start", async (_event, ctx) => {
    await callSubagentHost(socket, { method: "config", role });
    const allowed = new Set(config.tools.map((tool: any) => tool.name));
    if (pi.getActiveTools().some((name) => !allowed.has(name))) throw new Error("Unexpected tool in subagent session");
  });
  pi.on("before_agent_start", async (event, ctx) => {
    // Native resume reuses the child context with a fresh terminal attachment.
    const previous = attachment;
    if (previous) {
      attachment = undefined;
      try {
        await callSubagentHost(socket, {
          method: "terminal.detach", role, ...previous.identity, terminal_lease: await previous.ready,
        }, previous.controller.signal);
      } finally { previous.controller.abort(); }
    }
    await callSubagentHost(socket, {method: "config", role});
    let prompt = event.systemPrompt;
    if (config.workspacePathAlias) {
      event.systemPromptOptions.cwd = config.workspacePathAlias.alias;
      prompt = prompt.replaceAll(config.workspacePathAlias.host, config.workspacePathAlias.alias);
    }
    for (const mapping of config.skillPathAliases ?? []) {
      prompt = prompt.replaceAll(mapping.host, mapping.alias);
      const options = event.systemPromptOptions;
      options.skills = options.skills.map(skill => ({ ...skill,
        filePath: skill.filePath.replace(mapping.host, mapping.alias),
        baseDir: skill.baseDir.replace(mapping.host, mapping.alias),
      }));
    }
    const workspace = `public_data/subagents/${ctx.sessionManager.getSessionId()}/`;
    return { systemPrompt: prompt + "\n" + config.instructions +
      `\nTask workspace: ${workspace}. Keep task files and reports there. ` +
      "When cancelling remote work, call df_stop_task and verify its result. Native stop only stops this Agent." };
  });
}
