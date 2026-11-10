import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { JsonlRpcPeer } from "./rpc-peer.js";
import { isRecord, type JsonObject } from "./protocol.js";

/** Serializes work on a tool lane, including cancellation before admission. */
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

  wrap(tool: AgentTool, peer: JsonlRpcPeer, role?: string, recheckAfterApproval = false): AgentTool {
    if (!["terminal", "write", "edit"].includes(tool.name)) return tool;
    const execute = (id: string, args: any, signal?: AbortSignal, update?: any) => this.run(signal, async () => {
      const authorize = async () => {
        signal?.throwIfAborted();
        if (!role) return;
        await peer.request("delegation.authorize_tool", { role, tool_name: tool.name });
        signal?.throwIfAborted();
      };
      const approve = async () => {
        const cancelPermission = () => { void peer.request("permission.cancel", { tool_call_id: id }).catch(() => {}); };
        signal?.addEventListener("abort", cancelPermission, { once: true });
        let decision;
        try {
          decision = await peer.request("permission.check", {
            tool_call_id: id, tool_name: "terminal", arguments: JSON.parse(JSON.stringify(args)) as JsonObject,
            ...(role ? {delegated_role: role} : {}),
          });
        } finally { signal?.removeEventListener("abort", cancelPermission); }
        signal?.throwIfAborted();
        if (!isRecord(decision) || typeof decision.allow !== "boolean") throw new Error("invalid permission response");
        if (!decision.allow) throw new Error(typeof decision.reason === "string" ? decision.reason : "User denied terminal command");
      };
      if (tool.name === "terminal" && recheckAfterApproval) {
        await approve();
        await authorize();
      } else {
        await authorize();
        if (tool.name === "terminal") await approve();
      }
      return tool.execute(id, args, signal, update);
    });
    return { ...tool, execute: (id, args, signal, update) => execute(id, args, signal, update) };
  }
}
