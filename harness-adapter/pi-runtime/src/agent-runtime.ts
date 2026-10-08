import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { setImmediate } from "node:timers";
import type { AgentSession, SessionEntry } from "@earendil-works/pi-coding-agent";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { PiEventNormalizer } from "./pi-events.js";
import { PiOutcomeNormalizer } from "./pi-outcome.js";
import { createPiSession, parsePromptContent, type ParsedPrompt } from "./pi-session.js";
import {
  PROTOCOL_VERSION,
  type JsonObject,
  type JsonValue,
  type RunOutcome,
  type RunnerEvent,
} from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";

import type { WorkflowExecution } from "./workflow-execution.js";

export class PiAgentRuntime {
  private execution: WorkflowExecution | undefined;
  private session: AgentSession | undefined;
  private sessionId: string | undefined;
  private workspaceRoot: string | undefined;
  private terminalBackend: string | undefined;
  private normalizer: PiEventNormalizer | undefined;
  private readonly outcome = new PiOutcomeNormalizer();
  private readonly finishedRuns = new Set<string>();
  private readonly pendingCommandIds: string[] = [];
  private readonly notificationIds = new Set<string>();
  private readonly pendingNotifications: Array<{
    runId: string;
    message: { customType: string; content: string; display: boolean; details: JsonObject };
  }> = [];

  private readonly stages = new Map<string, {
    executionId: string; request: JsonObject; state: "queued" | "running" | "finished";
    promise: Promise<JsonValue>; resolve: (result: JsonValue) => void;
  }>();

  constructor(private readonly peer: JsonlRpcPeer) {}

  async handle(method: string, params: JsonObject): Promise<JsonValue> {
    if (method.startsWith("execution.")) {
      if (!this.execution) throw new Error("execution host not ready");
      return this.execution.handle(method, params);
    }
    if (method === "stage.execute") return this.executeStage(params);
    if (method === "stage.cancel") return this.cancelStage(requiredString(params, "request_id"));
    if (method === "start") return this.start(params);
    if (method === "prompt") return this.prompt(params, false);
    if (method === "steer") return this.prompt(params, true);
    if (method === "cancel") return this.cancel();
    if (method === "notify") return this.notify(params);
    if (method === "context.append") return this.appendContext(params);
    if (method === "fork") return this.fork(params);
    if (method === "close") return this.close();
    throw new Error(`unknown runner method: ${method}`);
  }

  private async start(params: JsonObject): Promise<JsonValue> {
    if (this.session) throw new Error("Pi session already started");
    const { session, sessionId, execution } = await createPiSession(params, this.peer);
    this.execution = execution;
    session.subscribe((event) => {
      this.outcome.observe(event);
      if (!this.normalizer) return;
      for (const translated of this.normalizer.translate(event)) this.peer.emit(translated);
    });
    this.session = session;
    this.sessionId = sessionId;
    this.workspaceRoot = requiredString(params, "workspace_root");
    this.terminalBackend = requiredString(params, "terminal_backend");
    for (const entry of session.sessionManager.getBranch()) {
      if (entry.type !== "custom" || entry.customType !== "pyromind.notification" || !isObject(entry.data)) continue;
      const { runId, message } = entry.data;
      if (typeof runId !== "string" || !isObject(message) || typeof message.content !== "string") continue;
      if (this.notificationIds.has(runId)) continue;
      this.notificationIds.add(runId);
      const delivered = notificationWasDelivered(session.sessionManager.getBranch(), runId);
      if (!delivered) this.pendingNotifications.push({ runId, message: {
        customType: "pyromind.external_task", content: message.content, display: false,
        details: isObject(message.details) ? message.details : {},
      } });
    }
    setImmediate(() => this.drainNotifications());
    return { ready: true };
  }

  private async prompt(params: JsonObject, forceSteer: boolean): Promise<JsonValue> {
    const session = this.requireSession();
    const runId = requiredString(params, "run_id");
    const prompt = parsePromptContent(params.content);
    if (forceSteer || session.isStreaming) {
      this.pendingCommandIds.push(runId);
      try { await session.steer(prompt.text, prompt.images); }
      catch (error) { this.discardCommand(runId); throw error; }
      return { accepted: true, steered: true };
    }
    if (this.normalizer) throw new Error("Pi agent is already running");
    this.normalizer = new PiEventNormalizer(this.sessionId!, runId, this.pendingCommandIds);
    this.outcome.reset();
    this.pendingCommandIds.push(runId);
    setImmediate(() => void this.runPrompt(runId, prompt));
    return { accepted: true, steered: false };
  }

  private discardCommand(commandId: string): void {
    const index = this.pendingCommandIds.indexOf(commandId);
    if (index >= 0) this.pendingCommandIds.splice(index, 1);
  }

  private async runPrompt(runId: string, prompt: ParsedPrompt): Promise<void> {
    try {
      const session = this.requireSession();
      await session.prompt(prompt.text, {
        images: prompt.images,
        expandPromptTemplates: false,
      });
      this.finishRun(runId, this.outcome.normalize(session.messages));
    } catch (error) {
      this.finishRun(runId, {
        status: "failed",
        error_code: "runner_error",
        message: error instanceof Error ? error.message : "Pi runner failed",
      });
    } finally {
      this.discardCommand(runId);
      this.normalizer = undefined;
      this.drainNotifications();
    }
  }

  private finishRun(runId: string, outcome: RunOutcome): void {
    if (!this.sessionId || this.finishedRuns.has(runId)) return;
    this.finishedRuns.add(runId);
    const event: RunnerEvent = {
      protocolVersion: PROTOCOL_VERSION,
      type: "pi.event",
      eventId: randomUUID(),
      sessionId: this.sessionId,
      runId,
      occurredAt: new Date().toISOString(),
      kind: "run.finished",
      payload: {
        outcome: JSON.parse(JSON.stringify(outcome)) as JsonObject,
        checkpoint_entry_id: this.session?.sessionManager?.getLeafId() ?? null,
        ...this.captureWorkflow(),
      },
    };
    this.peer.emit(event);
  }

  private captureWorkflow(): JsonObject {
    // The sandbox execution workspace is the authority for workflow files, so
    // the control plane reads it through the workspace port instead.
    if (this.terminalBackend === "sandbox") return {};
    try {
      return { workflow_dsl: readFileSync(join(this.workspaceRoot!, "public_data/workflow_canvas/workflow.py"), "utf8") };
    } catch (error) {
      return { workflow_dsl: null, workflow_snapshot_error: (error as NodeJS.ErrnoException).code !== "ENOENT" };
    }
  }

  private async cancel(): Promise<JsonValue> {
    await this.requireSession().abort();
    return { cancelled: true };
  }

  private async notify(params: JsonObject): Promise<JsonValue> {
    const session = this.requireSession();
    const runId = requiredString(params, "run_id");
    const content = requiredString(params, "content");
    if (this.notificationIds.has(runId)) return { accepted: true, duplicate: true };
    const details = { ...(isObject(params.details) ? params.details : {}), notification_run_id: runId };
    const message = {
      customType: "pyromind.external_task",
      content,
      display: false,
      details,
    };
    session.sessionManager.appendCustomEntry("pyromind.notification", { runId, message });
    this.notificationIds.add(runId);
    if (session.isStreaming || this.normalizer) {
      this.pendingNotifications.push({ runId, message });
      return { accepted: true, queued: true };
    }
    this.startNotification(runId, message);
    return { accepted: true, queued: false };
  }

  private async appendContext(params: JsonObject): Promise<JsonValue> {
    const session = this.requireSession();
    const content = requiredString(params, "content");
    const details = isObject(params.details) ? params.details : {};
    const triggerTurn = params.trigger_turn === true;
    await session.sendCustomMessage(
      {
        customType: "pyromind.context",
        content,
        display: false,
        details,
      },
      { triggerTurn },
    );
    return {
      accepted: true,
      trigger_turn: triggerTurn,
      checkpoint_entry_id: session.sessionManager.getLeafId(),
    };
  }

  private fork(params: JsonObject): JsonValue {
    const sourceSession = this.requireSession().sessionManager.getSessionFile();
    if (!sourceSession) throw new Error("Pi source session is not persisted");
    const leafId = requiredString(params, "leaf_id");
    const targetSessionDir = requiredString(params, "target_session_dir");
    const targetCwd = requiredString(params, "target_cwd");
    const manager = SessionManager.open(sourceSession, targetSessionDir, targetCwd);
    const sessionPath = manager.createBranchedSession(leafId);
    if (!sessionPath) throw new Error("Pi branched session was not persisted");
    return { session_path: sessionPath };
  }

  private startNotification(
    runId: string,
    message: { customType: string; content: string; display: boolean; details: JsonObject },
  ): void {
    const executionRunId = `${runId}:${randomUUID()}`;
    this.normalizer = new PiEventNormalizer(this.sessionId!, executionRunId, this.pendingCommandIds);
    this.outcome.reset();
    setImmediate(() => void this.runNotification(executionRunId, message, runId));
  }

  private async runNotification(
    runId: string,
    message: { customType: string; content: string; display: boolean; details: JsonObject },
    notificationId: string,
  ): Promise<void> {
    try {
      const session = this.requireSession();
      await session.sendCustomMessage(message, {
        triggerTurn: true,
      });
      this.finishRun(runId, this.outcome.normalize(session.messages));
    } catch (error) {
      this.finishRun(runId, {
        status: "failed",
        error_code: "runner_error",
        message: error instanceof Error ? error.message : "Pi notification failed",
      });
    } finally {
      this.requireSession().sessionManager.appendCustomEntry("pyromind.notification.finished", { runId: notificationId, execution_run_id: runId });
      this.normalizer = undefined;
      this.drainNotifications();
    }
  }

  private executeStage(request: JsonObject): Promise<JsonValue> {
    this.requireSession();
    const id = requiredString(request, "request_id");
    const previous = this.stages.get(id);
    if (previous) {
      if (JSON.stringify(previous.request) !== JSON.stringify(request)) throw new Error("conflicting stage request");
      return previous.promise;
    }
    let resolve!: (result: JsonValue) => void;
    const promise = new Promise<JsonValue>((done) => { resolve = done; });
    this.stages.set(id, { executionId: `stage:${id}:${randomUUID()}`, request,
      state: "queued", promise, resolve });
    this.drainNotifications();
    return promise;
  }

  private async cancelStage(id: string): Promise<JsonValue> {
    const stage = this.stages.get(id);
    if (!stage || stage.state === "finished") return { cancelled: false };
    if (stage.state === "queued") {
      stage.state = "finished";
      stage.resolve({ status: "cancelled", execution_id: stage.executionId });
    } else {
      await this.requireSession().abort();
      await stage.promise;
    }
    return { cancelled: true };
  }

  private async runStage(id: string): Promise<void> {
    const stage = this.stages.get(id)!;
    let result: RunOutcome;
    try {
      await this.requireSession().sendCustomMessage({ customType: "agentgenome.stage",
        content: JSON.stringify(stage.request), display: false,
        details: { request_id: id, execution_id: stage.executionId },
      }, { triggerTurn: true });
      result = this.outcome.normalize(this.requireSession().messages);
    } catch (error) {
      result = { status: "failed", message: error instanceof Error ? error.message : "stage failed" };
    }
    this.finishRun(stage.executionId, result);
    stage.state = "finished";
    this.normalizer = undefined;
    stage.resolve({ status: result.status, execution_id: stage.executionId });
    this.drainNotifications();
  }

  private drainNotifications(): void {
    if (this.normalizer || this.requireSession().isStreaming) return;
    for (const [id, stage] of this.stages) {
      if (stage.state !== "queued") continue;
      stage.state = "running";
      this.normalizer = new PiEventNormalizer(this.sessionId!, stage.executionId, this.pendingCommandIds);
      this.outcome.reset();
      setImmediate(() => void this.runStage(id));
      return;
    }
    const next = this.pendingNotifications.shift();
    if (next) this.startNotification(next.runId, next.message);
  }

  private async close(): Promise<JsonValue> {
    const session = this.requireSession();
    await session.abort();
    await session.waitForIdle();
    session.dispose();
    setImmediate(() => process.exit(0));
    return { closed: true };
  }

  private requireSession(): AgentSession {
    if (!this.session) throw new Error("Pi session is not started");
    return this.session;
  }
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requiredString(value: Record<string, unknown>, name: string): string {
  const item = value[name];
  if (typeof item !== "string" || !item) throw new Error(`${name} must be a string`);
  return item;
}


export function notificationWasDelivered(entries: SessionEntry[], runId: string): boolean {
  return entries.some((item) =>
    (item.type === "custom_message" && item.customType === "pyromind.external_task" &&
      isObject(item.details) && (item.details.notification_run_id === runId ||
        `callback:${item.details.task_id}:${item.details.status}` === runId)) ||
    (item.type === "custom" && item.customType === "pyromind.notification.finished" &&
      isObject(item.data) && item.data.runId === runId));
}
