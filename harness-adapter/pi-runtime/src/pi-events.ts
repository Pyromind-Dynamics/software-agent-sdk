import { randomUUID } from "node:crypto";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Usage, UserMessage } from "@earendil-works/pi-ai";
import type { AgentSessionEvent } from "@earendil-works/pi-coding-agent";
import { PROTOCOL_VERSION, isRecord, type JsonObject, type JsonValue, type RunnerEvent, type RunnerEventKind } from "./protocol.js";

const SECRET = /(^|[_-])(api[_-]?key|authorization|cookie|password|secret|token)($|[_-])/i;
const MAX_STRING = 64 * 1024;

export class PiEventNormalizer {
  private messageId: string | undefined;
  private messageRole: "user" | "assistant" | undefined;
  private readonly toolArguments = new Map<string, JsonValue>();

  constructor(private readonly sessionId: string, private readonly runId: string) {}

  translate(event: AgentSessionEvent): RunnerEvent[] {
    switch (event.type) {
      case "agent_start": return [this.make("agent.started", {})];
      case "agent_end": return [];
      case "turn_end": return [];
      case "turn_start": return [];
      case "message_start": return this.startMessage(event.message);
      case "message_update":
        if (event.assistantMessageEvent.type !== "text_delta" || !this.messageId) return [];
        return [this.make("message.delta", { message_id: this.messageId, text: event.assistantMessageEvent.delta })];
      case "message_end": return this.endMessage(event.message);
      case "tool_execution_start":
        this.toolArguments.set(event.toolCallId, sanitizeJson(event.args));
        return [this.make("tool.started", {
          tool_call_id: event.toolCallId, tool_name: visibleToolName(event.toolName), arguments: sanitizeJson(event.args),
        })];
      case "tool_execution_update": return [this.make("tool.progress", {
        tool_call_id: event.toolCallId, tool_name: visibleToolName(event.toolName), ...resultPayload(event.partialResult),
      })];
      case "tool_execution_end": {
        const arguments_ = this.toolArguments.get(event.toolCallId);
        this.toolArguments.delete(event.toolCallId);
        return [this.make(event.isError ? "tool.failed" : "tool.completed", {
          tool_call_id: event.toolCallId, tool_name: visibleToolName(event.toolName), ...resultPayload(event.result),
          ...(arguments_ !== undefined ? { arguments: arguments_ } : {}),
          ...(event.isError ? { error_code: "tool_execution_failed" } : {}),
        })];
      }
      default: return [];
    }
  }

  private startMessage(message: AgentMessage): RunnerEvent[] {
    if (message.role !== "user" && message.role !== "assistant") return [];
    this.messageId = randomUUID();
    this.messageRole = message.role;
    return [this.make("message.started", { message_id: this.messageId, role: message.role, content: messageContent(message) })];
  }

  private endMessage(message: AgentMessage): RunnerEvent[] {
    if (message.role !== "user" && message.role !== "assistant") return [];
    const id = this.messageRole === message.role && this.messageId ? this.messageId : randomUUID();
    this.messageId = undefined;
    this.messageRole = undefined;
    const events = [this.make("message.completed", { message_id: id, role: message.role, content: messageContent(message) })];
    if (message.role === "assistant") {
      events.push(this.make("usage.updated", usagePayload(message.usage)));
    }
    return events;
  }

  private make(kind: RunnerEventKind, payload: JsonObject): RunnerEvent {
    return { protocolVersion: PROTOCOL_VERSION, type: "pi.event", eventId: randomUUID(), sessionId: this.sessionId,
      runId: this.runId, occurredAt: new Date().toISOString(), kind, payload };
  }
}

function visibleToolName(name: string): string { return name === "bash" ? "terminal" : name; }

function messageContent(message: UserMessage | AssistantMessage): JsonValue[] {
  if (typeof message.content === "string") return [{ type: "text", text: message.content }];
  return message.content.flatMap((block): JsonValue[] => block.type === "text"
    ? [{ type: "text", text: block.text }]
    : block.type === "image" ? [{ type: "image", mime_type: block.mimeType, data: block.data }] : []);
}

function usagePayload(usage: Usage): JsonObject {
  return { input_tokens: usage.input, output_tokens: usage.output, cached_tokens: usage.cacheRead, cost_usd: usage.cost.total };
}

function resultPayload(result: unknown): JsonObject {
  if (!isRecord(result)) return { content: [], details: null };
  const details = result.details === null || isRecord(result.details) ? sanitizeJson(result.details) : null;
  return { content: Array.isArray(result.content) ? sanitizeJson(result.content) : [], details };
}

export function sanitizeJson(value: unknown, depth = 0, key = ""): JsonValue {
  if (SECRET.test(key)) return "[REDACTED]";
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "string") return value.slice(0, MAX_STRING);
  if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
  if (depth >= 8) return "[TRUNCATED]";
  if (Array.isArray(value)) return value.slice(0, 100).map((item) => sanitizeJson(item, depth + 1));
  if (!isRecord(value)) return String(value);
  return Object.fromEntries(Object.entries(value).slice(0, 100).map(([name, child]) => [name, sanitizeJson(child, depth + 1, name)]));
}
