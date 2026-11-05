import { createHash, randomUUID } from "node:crypto";
import { once } from "node:events";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { SandboxToolLane } from "./tools.js";
import { verifyChildSession } from "./subagent-recipients.js";

export function usesAgentTerminal(name: string): boolean {
  return ["terminal", "read", "write", "edit", "grep", "find", "ls"].includes(name);
}

export interface SubagentTerminalFactory {
  createLane(owner: string): SandboxToolLane;
  tools(lane: SandboxToolLane, role: string): AgentTool[];
  close(): Promise<void>;
}

interface TerminalLease {
  id: string;
  role: string;
  session: string;
  sessionFile: string;
  controller: AbortController;
  signal: AbortSignal;
  tools: AgentTool[];
  calls: Set<Promise<unknown>>;
  drained: Promise<void>;
}

/** The attachment socket owns the lease. A detached child's process exit closes it too. */
export class SubagentTerminals {
  private readonly leases = new Map<string, TerminalLease>();
  private closed = false;

  constructor(private readonly stateRoot: string, private readonly factory: SubagentTerminalFactory) {}

  async attach(role: string, session: string, sessionFile: string, signal: AbortSignal, ready: (id: string) => void): Promise<void> {
    const path = verifyChildSession(this.stateRoot, session, sessionFile);
    const owner = "child-" + createHash("sha256").update(session + "\0" + path).digest("hex").slice(0, 24);
    let prior = this.leases.get(owner);
    while (prior) {
      if (!prior.signal.aborted) throw new Error("Child terminal already attached");
      await prior.drained;
      signal.throwIfAborted();
      prior = this.leases.get(owner);
    }
    signal.throwIfAborted();
    if (this.closed) throw new Error("Child terminal host closed");
    const lane = this.factory.createLane(owner);
    const { promise: drained, resolve } = Promise.withResolvers<void>();
    const controller = new AbortController();
    const lease: TerminalLease = {
      id: randomUUID(), role, session, sessionFile, controller,
      signal: AbortSignal.any([signal, controller.signal]), tools: this.factory.tools(lane, role),
      calls: new Set(), drained,
    };
    this.leases.set(owner, lease);
    try {
      const disconnected = once(lease.signal, "abort");
      ready(lease.id);
      await disconnected;
    } finally {
      try {
        await Promise.allSettled(lease.calls);
        await lane.terminal.disconnect();
      } finally {
        this.leases.delete(owner);
        resolve();
      }
    }
  }

  async detach(request: Record<string, unknown>): Promise<void> {
    const lease = this.binding(request);
    if (!lease) return;
    lease.controller.abort();
    await lease.drained;
  }

  private binding(request: Record<string, unknown>): TerminalLease | undefined {
    return [...this.leases.values()].find((lease) => lease.id === request.terminal_lease &&
      lease.role === request.role && lease.session === request.session && lease.sessionFile === request.session_file);
  }

  async run<T>(request: Record<string, unknown>, signal: AbortSignal, work: (tool: AgentTool, signal: AbortSignal) => Promise<T>): Promise<T> {
    const lease = this.binding(request);
    if (this.closed || !lease || lease.signal.aborted) {
      throw new Error("Invalid or expired child terminal binding");
    }
    const tool = lease.tools.find((candidate) => candidate.name === request.name);
    if (!tool) throw new Error("Child terminal tool is unavailable");
    const combined = AbortSignal.any([signal, lease.signal]);
    combined.throwIfAborted();
    const pending = work(tool, combined);
    lease.calls.add(pending);
    try { return await pending; }
    finally { lease.calls.delete(pending); }
  }

  async close(): Promise<void> {
    this.closed = true;
    await Promise.all([...this.leases.values()].map((lease) => lease.drained));
    await this.factory.close();
  }
}
