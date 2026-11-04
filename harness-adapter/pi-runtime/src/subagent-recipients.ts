import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, realpathSync } from "node:fs";
import { join, relative, sep } from "node:path";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { isRecord, type JsonObject } from "./protocol.js";

type Rpc = (method: string, params: Record<string, unknown>, requestId?: string) => Promise<any>;
type Recipient = { id: string; sessionFile: string; sessionId: string };
type NativeTarget = { runId: string; index: number; state: string; updatedAt: number; step: any };

/** Only this adapter maps a product recipient to plugin-owned execution records. */
export class SubagentRecipients {
  constructor(private stateRoot: string, private session: AgentSession, private rpc: Rpc, private conversationId = session.sessionId) {}

  origin(sessionId: unknown, sessionFile: unknown): JsonObject {
    if (typeof sessionId !== "string" || typeof sessionFile !== "string") throw new Error("Missing child session identity");
    const path = realpathSync(sessionFile);
    const root = realpathSync(join(this.stateRoot, "sessions"));
    const local = relative(root, path);
    if (!local || local === ".." || local.startsWith(".." + sep) || local.startsWith(sep)) throw new Error("Child session belongs to another conversation");
    const header = JSON.parse(readFileSync(path, "utf8").split("\n", 1)[0]);
    if (header.type !== "session" || header.id !== sessionId) throw new Error("Child session identity mismatch");
    const id = createHash("sha256").update(this.conversationId + ":" + sessionId).digest("hex");
    const existing = this.savedRecipient(id);
    if (!existing) this.session.sessionManager.appendCustomEntry("pyromind.subagent.recipient", { id, sessionId, sessionFile: path });
    else if (existing.sessionFile !== path) throw new Error("Child recipient identity changed");
    return { type: "subagent", subagent_id: id };
  }

  private savedRecipient(id: string): Recipient | undefined {
    for (const entry of this.session.sessionManager.getEntries()) {
      if (entry.type !== "custom" || entry.customType !== "pyromind.subagent.recipient" || !isRecord(entry.data)) continue;
      const value = entry.data;
      if (value.id === id && typeof value.sessionFile === "string" && typeof value.sessionId === "string") return value as Recipient;
    }
  }

  private recipient(id: string): Recipient | undefined {
    const saved = this.savedRecipient(id);
    if (saved) return saved;
    // Plugin session artifacts also recover recipients when the parent has not
    // yet emitted its first assistant message and its log is still deferred.
    const find = (directory: string): Recipient | undefined => {
      for (const entry of readdirSync(directory, {withFileTypes: true})) {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) { const result = find(path); if (result) return result; }
        else if (entry.isFile() && entry.name.endsWith(".jsonl")) {
          const header = JSON.parse(readFileSync(path, "utf8").split("\n", 1)[0]);
          if (header.type !== "session" || typeof header.id !== "string") continue;
          const candidate = createHash("sha256").update(this.conversationId + ":" + header.id).digest("hex");
          if (candidate === id) return {id, sessionId: header.id, sessionFile: path};
        }
      }
    };
    const root = join(this.stateRoot, "sessions");
    return existsSync(root) ? find(root) : undefined;
  }

  private targets(recipient: Recipient): NativeTarget[] {
    const targets: NativeTarget[] = [];
    const owner = this.session.sessionManager.getSessionFile();
    const add = (run: any, steps: any[], state: string, updatedAt: number) => {
      if (![owner, this.session.sessionId].includes(run.sessionId)) return;
      for (const [position, step] of steps.entries()) {
        if (step.sessionFile !== recipient.sessionFile && !(steps.length === 1 && run.sessionFile === recipient.sessionFile)) continue;
        targets.push({ runId: run.runId, index: step.index ?? position, state: step.stopped || step.stopRequested || run.stopped ? "stopped" : step.status ?? state, updatedAt, step });
      }
    };
    const asyncRoot = join(this.stateRoot, "async-subagent-runs");
    for (const name of existsSync(asyncRoot) ? readdirSync(asyncRoot) : []) {
      if (!/^[a-zA-Z0-9-]+$/.test(name)) continue;
      const file = join(asyncRoot, name, "status.json");
      if (!existsSync(file)) continue;
      const run = JSON.parse(readFileSync(file, "utf8"));
      if (run.lifecycleArtifactVersion === 3 && run.runId === name) add(run, run.steps ?? [], run.state, run.startedAt ?? 0);
    }
    const foreground = join(this.stateRoot, "async-subagent-results", "foreground-history.json");
    if (existsSync(foreground)) {
      const history = JSON.parse(readFileSync(foreground, "utf8"));
      if (history.version === 1) for (const run of history.runs ?? []) add(run, run.children ?? [], "completed", run.updatedAt ?? 0);
    }
    return targets.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  async notify(params: JsonObject): Promise<JsonObject> {
    const correlation = isRecord(params.details) && isRecord(params.details.correlation) ? params.details.correlation : undefined;
    if (!correlation || correlation.type !== "subagent" || typeof correlation.subagent_id !== "string") throw new Error("Invalid subagent return address");
    const recipient = this.recipient(correlation.subagent_id);
    if (!recipient) return { accepted: false, reason: "unknown_subagent" };
    const identity = String(params.run_id);
    const marker = `[external-task-receipt:${createHash("sha256").update(identity).digest("hex")}]`;
    const delivered = this.session.sessionManager.getEntries().some(entry => entry.type === "custom" && entry.customType === "pyromind.subagent.delivery" && isRecord(entry.data) && entry.data.id === identity);
    if (delivered || existsSync(recipient.sessionFile) && readFileSync(recipient.sessionFile, "utf8").includes(marker)) return { accepted: true, duplicate: true };
    const target = this.targets(recipient)[0];
    if (!target || ["stopped", "paused", "rejected", "cancelled"].includes(target.state)) return { accepted: false, reason: target ? "subagent_stopped" : "subagent_unavailable" };
    if (params.trigger_turn === false) return { accepted: false, reason: "subagent_context_deferred" };
    const message = `${marker}\n${String(params.content)}`;
    const running = ["running", "queued", "pending"].includes(target.state);
    const result = await this.rpc(running ? "steer" : "resume", {
      id: target.runId, index: target.index, message,
      ...(running ? { mode: "follow_up" } : {}),
    }, identity);
    if (result.isError || ["failed", "missed", "partial"].includes(result.details?.steering?.state ?? result.details?.state)) return { accepted: false, reason: "subagent_delivery_failed" };
    this.session.sessionManager.appendCustomEntry("pyromind.subagent.delivery", { id: identity, subagent_id: recipient.id });
    return { accepted: true, queued: running };
  }
}
