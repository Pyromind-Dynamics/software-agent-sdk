import { createJiti } from "jiti";
import type {
  ExtensionAPI,
  ExtensionContext,
  ExtensionFactory,
  ExtensionHandler,
  MessageUpdateEvent,
} from "@earendil-works/pi-coding-agent";

export const REPETITION_ERROR = "model_repetition_detected";
export const REPETITION_MESSAGE = "模型持续重复生成，已停止。请发送新消息后继续。";

let pluginFactory: Promise<ExtensionFactory> | undefined;

function loadPlugin(): Promise<ExtensionFactory> {
  // The upstream package ships TS in node_modules. Native Node imports cannot
  // strip those types; use the same loader technology as Pi's extensions.
  return pluginFactory ??= createJiti(import.meta.url).import<ExtensionFactory>(
    import.meta.resolve("pi-no-spin/extensions/index.ts"), { default: true },
  );
}

/** Keep upstream detection/abort, with a host-owned manual recovery policy. */
export function createNoSpinExtension(onDetected?: () => void): ExtensionFactory {
  return async (pi) => {
    let detected = false;
    let repeatedMessage = false;

    pi.on("session_start", (_event, ctx) => {
      const saved = ctx.sessionManager.getEntries().filter((entry) =>
        entry.type === "custom" && entry.customType === "pinospin-config").at(-1);
      const data = saved?.type === "custom" && saved.data && typeof saved.data === "object"
        ? saved.data as Record<string, unknown> : {};
      // Upstream cooldown avoids repeated automatic reminders. We send none;
      // every manual retry must be protected immediately, even within 10s.
      if (data.cooldownMs !== 0) pi.appendEntry("pinospin-config", { ...data, cooldownMs: 0 });
    });

    function wrapContext(ctx: ExtensionContext): ExtensionContext {
      return new Proxy(ctx, {
        get(target, key) {
          if (key === "abort") return () => {
            // An already cancelled request belongs to the user's cancel path.
            try {
              if (!ctx.signal?.aborted && !detected) {
                detected = true;
                repeatedMessage = true;
                onDetected?.();
              }
            } finally { ctx.abort(); }
          };
          // The plugin's notification claims that a reminder was sent. The
          // product reports its own failure; no TUI or console output is needed.
          if (key === "ui") return new Proxy(target.ui, {
            get(ui, property) { return property === "notify" ? () => {} : Reflect.get(ui, property); },
          });
          return Reflect.get(target, key);
        },
      });
    }

    // Pi overloads on() by event name; the proxy forwards the original event
    // unchanged and only substitutes the context for this plugin's handlers.
    const on = ((name: "message_update", handler: ExtensionHandler<MessageUpdateEvent, void>) =>
      pi.on(name, (event, ctx) => handler(event, wrapContext(ctx)))) as ExtensionAPI["on"];
    const pluginApi = new Proxy(pi, {
      get(target, key) {
        if (key === "on") return on;
        if (key === "sendUserMessage") return () => {}; // suppress upstream automatic follow-up
        return Reflect.get(target, key);
      },
    });
    await (await loadPlugin())(pluginApi);

    pi.on("agent_start", () => { detected = false; repeatedMessage = false; });
    pi.on("message_start", (event) => {
      if (event.message.role === "assistant") repeatedMessage = false;
    });
    pi.on("message_end", (event) => {
      if (!repeatedMessage || event.message.role !== "assistant") return;
      // Also makes detached pi-subagents report an error instead of cancellation.
      return { message: {
        ...event.message, stopReason: "error",
        errorMessage: `${REPETITION_ERROR}: ${REPETITION_MESSAGE}`,
      } };
    });
    pi.on("turn_end", (event) => {
      if (!repeatedMessage || event.message.role !== "assistant") return;
      // Native boundary drafts update both the append-only session and the live
      // projection. Raw output stays available for diagnostics; tool pairs stay.
      return { continue: false, entries: [...event.entries, {
        type: "context_edit", targetId: event.messageEntryId,
        replacement: { content: [
          { type: "text", text: `[${REPETITION_ERROR}] ${REPETITION_MESSAGE}` },
          ...event.message.content.filter((block) => block.type === "toolCall"),
        ] },
      }] };
    });
  };
}
