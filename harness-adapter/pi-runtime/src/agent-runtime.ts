import { randomUUID } from "node:crypto";
import { setImmediate } from "node:timers";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
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

export class PiAgentRuntime {
  private session: AgentSession | undefined;
  private sessionId: string | undefined;
  private normalizer: PiEventNormalizer | undefined;
  private readonly outcome = new PiOutcomeNormalizer();
  private readonly finishedRuns = new Set<string>();

  constructor(private readonly peer: JsonlRpcPeer) {}

  async handle(method: string, params: JsonObject): Promise<JsonValue> {
    if (method === "start") return this.start(params);
    if (method === "prompt") return this.prompt(params, false);
    if (method === "steer") return this.prompt(params, true);
    if (method === "cancel") return this.cancel();
    if (method === "close") return this.close();
    throw new Error(`unknown runner method: ${method}`);
  }

  private async start(params: JsonObject): Promise<JsonValue> {
    if (this.session) throw new Error("Pi session already started");
    const { session, sessionId } = await createPiSession(params, this.peer);
    session.subscribe((event) => {
      this.outcome.observe(event);
      if (!this.normalizer) return;
      for (const translated of this.normalizer.translate(event)) this.peer.emit(translated);
    });
    this.session = session;
    this.sessionId = sessionId;
    return { ready: true };
  }

  private async prompt(params: JsonObject, forceSteer: boolean): Promise<JsonValue> {
    const session = this.requireSession();
    const runId = requiredString(params, "run_id");
    const prompt = parsePromptContent(params.content);
    if (forceSteer || session.isStreaming) {
      await session.steer(prompt.text, prompt.images);
      return { accepted: true, steered: true };
    }
    if (this.normalizer) throw new Error("Pi agent is already running");
    this.normalizer = new PiEventNormalizer(this.sessionId!, runId);
    this.outcome.reset();
    setImmediate(() => void this.runPrompt(runId, prompt));
    return { accepted: true, steered: false };
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
      this.normalizer = undefined;
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
      payload: { outcome: JSON.parse(JSON.stringify(outcome)) as JsonObject },
    };
    this.peer.emit(event);
  }

  private async cancel(): Promise<JsonValue> {
    await this.requireSession().abort();
    return { cancelled: true };
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

function requiredString(value: Record<string, unknown>, name: string): string {
  const item = value[name];
  if (typeof item !== "string" || !item) throw new Error(`${name} must be a string`);
  return item;
}
