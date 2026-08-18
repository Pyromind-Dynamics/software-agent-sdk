import { randomUUID } from "node:crypto";
import { posix } from "node:path";
import type { AgentEvent, AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Usage, UserMessage } from "@earendil-works/pi-ai";
import {
  PROTOCOL_VERSION,
  type JsonObject,
  type JsonValue,
  type PiRunnerEvent,
  type PiRunnerEventKind,
} from "./protocol.ts";

type IdFactory = () => string;

const SENSITIVE_KEY = /(^|[_-])(api[_-]?key|authorization|cookie|password|secret|token)($|[_-])/i;
const MAX_JSON_DEPTH = 8;
const MAX_ARRAY_ITEMS = 100;
const MAX_OBJECT_KEYS = 100;
const MAX_STRING_LENGTH = 64 * 1024;

export class PiEventNormalizer {
  private readonly sessionId: string;
  private readonly runId: string;
  private readonly idFactory: IdFactory;
  private activeMessageId: string | undefined;
  private activeMessageRole: "user" | "assistant" | undefined;
  private terminalOutcome: "completed" | "failed" | "cancelled" | undefined;
  private readonly workspaceRoot: string;
  private readonly toolArguments = new Map<
    string,
    { name: string; path?: string; command?: string }
  >();

  constructor(
    sessionId: string,
    runId: string,
    workspaceRoot = "/workspace",
    idFactory: IdFactory = randomUUID,
  ) {
    this.sessionId = sessionId;
    this.runId = runId;
    this.workspaceRoot = posix.normalize(workspaceRoot);
    this.idFactory = idFactory;
  }

  translate(event: AgentEvent): PiRunnerEvent[] {
    switch (event.type) {
      case "agent_start":
        return [this.event("agent.started", {})];
      case "agent_end":
        if (this.terminalOutcome !== undefined) return [];
        this.terminalOutcome = "completed";
        return [this.event("agent.completed", {})];
      case "message_start":
        return this.startMessage(event.message);
      case "message_update":
        return this.updateMessage(event.assistantMessageEvent);
      case "message_end":
        return this.endMessage(event.message);
      case "tool_execution_start":
        this.toolArguments.set(event.toolCallId, {
          name: event.toolName,
          ...(isRecord(event.args) && typeof event.args.path === "string"
            ? { path: event.args.path }
            : {}),
          ...(isRecord(event.args) && typeof event.args.command === "string"
            ? { command: event.args.command }
            : {}),
        });
        return [
          this.event("tool.started", {
            tool_call_id: event.toolCallId,
            tool_name: event.toolName,
            arguments: sanitizeJson(event.args),
          }),
        ];
      case "tool_execution_update":
        return [
          this.event("tool.progress", {
            tool_call_id: event.toolCallId,
            tool_name: event.toolName,
            ...toolResultPayload(event.partialResult),
          }),
        ];
      case "tool_execution_end":
        return [
          this.event(event.isError ? "tool.failed" : "tool.completed", {
            tool_call_id: event.toolCallId,
            tool_name: event.toolName,
            ...toolResultPayload(event.result),
            ...(event.isError ? { error_code: "tool_execution_failed" } : {}),
          }),
          ...this.resourceEvents(event.toolCallId, event.isError),
        ];
      case "turn_start":
      case "turn_end":
        return [];
    }
  }

  private resourceEvents(toolCallId: string, isError: boolean): PiRunnerEvent[] {
    const tool = this.toolArguments.get(toolCallId);
    this.toolArguments.delete(toolCallId);
    if (isError || !tool) return [];
    const workflowPath = posix.join(
      this.workspaceRoot,
      "public_data/workflow_canvas/workflow.py",
    );
    const relativeWorkflowPath = "public_data/workflow_canvas/workflow.py";
    const touchesWorkflow =
      (tool.name === "bash" && tool.command?.includes(relativeWorkflowPath) === true) ||
      ((tool.name === "write" || tool.name === "edit") &&
        tool.path !== undefined &&
        posix.resolve(this.workspaceRoot, tool.path) === workflowPath);
    if (!touchesWorkflow) return [];
    const eventId = this.idFactory();
    return [
      {
        protocolVersion: PROTOCOL_VERSION,
        type: "pi.event",
        eventId,
        sessionId: this.sessionId,
        runId: this.runId,
        occurredAt: new Date().toISOString(),
        kind: "resource.updated",
        payload: {
          resource_type: "workflow",
          resource_id: "workflow",
          version: eventId,
        },
      },
    ];
  }

  private startMessage(message: AgentMessage): PiRunnerEvent[] {
    if (message.role !== "user" && message.role !== "assistant") return [];
    const messageId = this.idFactory();
    this.activeMessageId = messageId;
    this.activeMessageRole = message.role;
    return [
      this.event("message.started", {
        message_id: messageId,
        role: message.role,
        content: messageContent(message),
      }),
    ];
  }

  private updateMessage(event: { type: string; delta?: string }): PiRunnerEvent[] {
    if (event.type !== "text_delta" || typeof event.delta !== "string") return [];
    const messageId = this.activeMessageId;
    if (messageId === undefined || this.activeMessageRole !== "assistant") return [];
    return [this.event("message.delta", { message_id: messageId, text: event.delta })];
  }

  private endMessage(message: AgentMessage): PiRunnerEvent[] {
    if (message.role !== "user" && message.role !== "assistant") return [];
    const messageId =
      this.activeMessageRole === message.role && this.activeMessageId !== undefined
        ? this.activeMessageId
        : this.idFactory();
    this.activeMessageId = undefined;
    this.activeMessageRole = undefined;
    const output = [
      this.event("message.completed", {
        message_id: messageId,
        role: message.role,
        content: messageContent(message),
      }),
    ];
    if (message.role !== "assistant") return output;

    output.push(this.event("usage.updated", usagePayload(message.usage)));
    if (message.stopReason === "error") {
      this.terminalOutcome = "failed";
      output.push(
        this.event("agent.failed", {
          error_code: "model_error",
          message: message.errorMessage ?? "Pi model request failed",
        }),
      );
    } else if (message.stopReason === "aborted") {
      this.terminalOutcome = "cancelled";
      output.push(this.event("agent.cancelled", { outcome: "cancelled" }));
    }
    return output;
  }

  private event(kind: PiRunnerEventKind, payload: JsonObject): PiRunnerEvent {
    return {
      protocolVersion: PROTOCOL_VERSION,
      type: "pi.event",
      eventId: this.idFactory(),
      sessionId: this.sessionId,
      runId: this.runId,
      occurredAt: new Date().toISOString(),
      kind,
      payload,
    };
  }
}

function messageContent(message: UserMessage | AssistantMessage): JsonValue[] {
  if (typeof message.content === "string") return [{ type: "text", text: message.content }];
  return message.content.flatMap((block): JsonValue[] => {
    if (block.type === "text") return [{ type: "text", text: block.text }];
    if (block.type === "image") {
      return [{ type: "image", mime_type: block.mimeType, data: block.data }];
    }
    return [];
  });
}

function usagePayload(usage: Usage): JsonObject {
  return {
    input_tokens: usage.input,
    output_tokens: usage.output,
    cached_tokens: usage.cacheRead,
    cost_usd: usage.cost.total,
  };
}

function toolResultPayload(result: unknown): JsonObject {
  if (!isRecord(result)) return { content: [] };
  const content = Array.isArray(result.content) ? sanitizeJson(result.content) : [];
  const payload: JsonObject = { content };
  if (result.details === null || isRecord(result.details)) {
    payload.details = sanitizeJson(result.details);
  }
  return payload;
}

export function sanitizeJson(value: unknown, depth = 0, key = ""): JsonValue {
  if (SENSITIVE_KEY.test(key)) return "[REDACTED]";
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "string") return value.slice(0, MAX_STRING_LENGTH);
  if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
  if (depth >= MAX_JSON_DEPTH) return "[TRUNCATED]";
  if (Array.isArray(value)) {
    return value.slice(0, MAX_ARRAY_ITEMS).map((item) => sanitizeJson(item, depth + 1));
  }
  if (!isRecord(value)) return String(value);
  return Object.fromEntries(
    Object.entries(value)
      .slice(0, MAX_OBJECT_KEYS)
      .map(([childKey, child]) => [childKey, sanitizeJson(child, depth + 1, childKey)]),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
