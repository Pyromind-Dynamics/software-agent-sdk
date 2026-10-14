import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { JsonlRpcPeer } from "./rpc-peer.js";
import { isRecord, type JsonObject } from "./protocol.js";

/** One execution lane per conversation, including permission admission. */
export class ToolCoordinator {
  private tail: Promise<unknown> = Promise.resolve();

  async run<T>(signal: AbortSignal | undefined, work: () => Promise<T>): Promise<T> {
    signal?.throwIfAborted();
    let started = false;
    const scheduled = this.tail.then(async () => {
      signal?.throwIfAborted();
      started = true;
      return work();
    });
    this.tail = scheduled.catch(() => {});
    let onAbort: (() => void) | undefined;
    try {
      if (!signal) return await scheduled;
      return await Promise.race([scheduled, new Promise<never>((_resolve, reject) => {
        onAbort = () => { if (!started) reject(signal.reason ?? new Error("tool cancelled")); };
        signal.addEventListener("abort", onAbort, { once: true });
      })]);
    } finally { if (onAbort) signal?.removeEventListener("abort", onAbort); }
  }

  wrap(tool: AgentTool, peer: JsonlRpcPeer): AgentTool {
    if (!["terminal", "write", "edit"].includes(tool.name)) return tool;
    return { ...tool, execute: (id, args, signal, update) => this.run(signal, async () => {
      if (tool.name === "terminal") {
        const cancelPermission = () => { void peer.request("permission.cancel", { tool_call_id: id }).catch(() => {}); };
        signal?.addEventListener("abort", cancelPermission, { once: true });
        let decision;
        try {
          decision = await peer.request("permission.check", {
            tool_call_id: id, tool_name: "terminal", arguments: JSON.parse(JSON.stringify(args)) as JsonObject,
          });
        } finally { signal?.removeEventListener("abort", cancelPermission); }
        signal?.throwIfAborted();
        if (!isRecord(decision) || typeof decision.allow !== "boolean") throw new Error("invalid permission response");
        if (!decision.allow) throw new Error(typeof decision.reason === "string" ? decision.reason : "User denied terminal command");
      }
      return tool.execute(id, args, signal, update);
    }) };
  }
}
