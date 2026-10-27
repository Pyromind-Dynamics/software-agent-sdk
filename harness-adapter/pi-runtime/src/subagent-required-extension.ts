import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { callSubagentHost } from "./subagent-ipc.js";
import { createToolResultExtension } from "./tool-results.js";
import { createNoSpinExtension } from "./no-spin-extension.js";

/** Required by the host, so startup errors fail the plugin launch closed. */
export default async function (pi: ExtensionAPI) {
  createToolResultExtension()(pi);
  await createNoSpinExtension()(pi);
  pi.on("session_start", async () => {
    const socket = process.env.PYROMIND_SUBAGENT_SOCKET;
    if (!socket) throw new Error("Missing subagent execution host");
    await callSubagentHost(socket, { method: "config", role: "search" });
    const active = new Set(pi.getActiveTools());
    for (const tool of pi.getAllTools()) {
      if (active.has(tool.name) && !tool.description.startsWith("[Conversation host]")) {
        throw new Error(`Subagent tool '${tool.name}' is not connected to its conversation host`);
      }
    }
  });
}
