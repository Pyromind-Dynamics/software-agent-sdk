import assert from "node:assert/strict";
import { test } from "node:test";
import { ToolCoordinator } from "../src/tool-coordinator.js";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";

test("permission admission and terminal execution share one lane across agents", async () => {
  const order: string[] = [];
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const peer = { request: async (_method: string, params: any) => { order.push(`permission:${params.tool_call_id}`); if (params.tool_call_id === "first") await gate; return { allow: true }; } } as unknown as JsonlRpcPeer;
  const terminal = new ToolCoordinator().wrap({ name: "terminal", execute: async (id: string) => { order.push(`execute:${id}`); return { content: [] }; } } as unknown as AgentTool, peer);
  const a = terminal.execute("first", {}, new AbortController().signal);
  const b = terminal.execute("second", {}, new AbortController().signal);
  await new Promise((resolve) => setImmediate(resolve)); assert.deepEqual(order, ["permission:first"]);
  release(); await Promise.all([a, b]);
  assert.deepEqual(order, ["permission:first", "execute:first", "permission:second", "execute:second"]);
});

test("a cancelled waiter returns promptly without overtaking the active job", async () => {
  const coordinator = new ToolCoordinator(); let release!: () => void;
  const first = coordinator.run(undefined, () => new Promise<void>((resolve) => { release = resolve; }));
  const controller = new AbortController();
  const cancelled = coordinator.run(controller.signal, async () => { throw new Error("must not execute"); });
  controller.abort(); await assert.rejects(cancelled, /abort/i);
  let ran = false; const third = coordinator.run(undefined, async () => { ran = true; });
  assert.equal(ran, false); release(); await first; await third; assert.equal(ran, true);
});

test("stopping a child cancels its pending permission before the next terminal enters", async () => {
  let permissionReply!: (value: unknown) => void;
  const order: string[] = [];
  const peer = { request: async (method: string, params: any) => {
    order.push(`${method}:${params.tool_call_id}`);
    if (method === "permission.cancel") { permissionReply({ allow: false }); return { cancelled: true }; }
    if (params.tool_call_id === "child") return new Promise((resolve) => { permissionReply = resolve; });
    return { allow: true };
  } } as unknown as JsonlRpcPeer;
  const tool = new ToolCoordinator().wrap({ name: "terminal", execute: async (id: string) => { order.push(`execute:${id}`); return { content: [] }; } } as unknown as AgentTool, peer);
  const controller = new AbortController();
  const child = tool.execute("child", {}, controller.signal);
  const next = tool.execute("parent", {}, new AbortController().signal);
  await new Promise((resolve) => setImmediate(resolve)); controller.abort();
  await assert.rejects(child, /abort/i); await next;
  assert.deepEqual(order, ["permission.check:child", "permission.cancel:child", "permission.check:parent", "execute:parent"]);
});
