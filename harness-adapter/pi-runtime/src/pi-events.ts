import { randomUUID } from "node:crypto";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Usage, UserMessage } from "@earendil-works/pi-ai";
import type { AgentSessionEvent } from "@earendil-works/pi-coding-agent";
import { PROTOCOL_VERSION, isRecord, type JsonObject, type JsonValue, type RunnerEvent, type RunnerEventKind } from "./protocol.js";

const SECRET = /(^|[_-])(api[_-]?key|authorization|cookie|password|secret|token)($|[_-])/i;
const MAX_STRING = 64 * 1024;
const REASONING_TAG = /<\/?(?:think|thinking|analysis|reasoning)(?=[\s/>])/i;

export class PiEventNormalizer {
  private messageId: string | undefined;
  private commandId: string | undefined;
  turnId: string | undefined;
  private messageRole: "user" | "assistant" | undefined;
  private readonly toolArguments = new Map<string, { args: JsonValue; turnId?: string }>();

  constructor(private readonly sessionId: string, readonly runId: string, private readonly pendingCommandIds: string[] = [], private readonly storageUrls = new Map<string, string>(), readonly taskId?: string) {}

  private publicHeader: string | null = "";
  private pendingTag = "";
  private streamedProjectReply = false;
  private readonly publishedArguments = new Map<string, string>();

  translate(event: AgentSessionEvent, publicUpdates = false, preparation = false, projectDialogue = false): RunnerEvent[] {
    const directReply = preparation && !!this.turnId && !this.taskId;
    switch (event.type) {
      case "agent_start": return [this.make("agent.started", {})];
      case "agent_end": return [];
      case "turn_end": return [];
      case "turn_start": return [];
      case "compaction_start":
        return [this.make("context_compaction.updated", { status: "started" })];
      case "compaction_end":
        return [this.make("context_compaction.updated", {
          status: event.errorMessage ? "failed" : event.result ? "completed" : "skipped",
        })];
      case "message_start": return this.startMessage(event.message);
      case "message_update": {
        const update = event.assistantMessageEvent;
        if (publicUpdates && update.type === "toolcall_delta") {
          const block = update.partial.content[update.contentIndex];
          if (block?.type !== "toolCall" || !block.id || block.name !== "publish_echomind_update") return [];
          const message = block.arguments.message;
          const previous = this.publishedArguments.get(block.id) ?? "";
          if (typeof message !== "string" || !message.startsWith(previous)) return [];
          // A JSON escape or a surrogate pair may still be incomplete at the chunk boundary.
          const decoded = message.replace(/[\\\uD800-\uDBFF]+$/u, "");
          const stable = decoded.slice(0, publicTextLength(decoded));
          const text = stable.slice(previous.length);
          if (!text) return [];
          this.publishedArguments.set(block.id, stable);
          return [this.make("message.delta", {
            message_id: `echomind-update:${this.runId}:${block.id}`, text, public_delta: text,
          })];
        }
        if (update.type !== "text_delta" || !this.messageId || this.messageRole !== "assistant") return [];
        const text = update.delta;
        const visible = publicUpdates || directReply || projectDialogue ? this.publicReplyDelta(text) : "";
        const payload: JsonObject = { message_id: this.messageId, text };
        if (visible) {
          if (directReply) payload.preparation_reply_delta = visible;
          else if (projectDialogue) {
            payload.project_reply_delta = visible;
            this.streamedProjectReply = true;
          } else payload.public_delta = visible;
        }
        return [this.make("message.delta", payload)];
      }
      case "message_end": return this.endMessage(event.message, directReply, projectDialogue);
      case "tool_execution_start":
        this.toolArguments.set(event.toolCallId, { args: sanitizeJson(event.args), turnId: this.turnId });
        return [this.make("tool.started", {
          tool_call_id: event.toolCallId, tool_name: visibleToolName(event.toolName), arguments: sanitizeJson(event.args),
        })];
      case "tool_execution_update": return [this.make("tool.progress", {
        tool_call_id: event.toolCallId, tool_name: visibleToolName(event.toolName), ...resultPayload(event.partialResult),
      }, this.toolArguments.get(event.toolCallId)?.turnId ?? null)];
      case "tool_execution_end": {
        if (event.toolName === "get_storage_url" && !event.isError && isRecord(event.result) && Array.isArray(event.result.content)) {
          for (const block of event.result.content) {
            if (!isRecord(block) || block.type !== "text" || typeof block.text !== "string") continue;
            for (const url of block.text.match(URL_PATTERN) ?? []) {
              const key = signedUrlIdentity(url);
              if (key) this.storageUrls.set(key, url);
            }
          }
        }
        const arguments_ = this.toolArguments.get(event.toolCallId);
        this.toolArguments.delete(event.toolCallId);
        const result = resultPayload(event.result);
        const details = isRecord(event.result) ? event.result.details : undefined;
        const errorCode = isRecord(details) &&
          typeof details.error_code === "string" && details.error_code.trim()
          ? details.error_code.slice(0, MAX_STRING) : "tool_execution_failed";
        return [this.make(event.isError ? "tool.failed" : "tool.completed", {
          tool_call_id: event.toolCallId, tool_name: visibleToolName(event.toolName), ...result,
          ...(arguments_ !== undefined ? { arguments: arguments_.args } : {}),
          ...(event.isError ? { error_code: errorCode } : {}),
        }, arguments_?.turnId ?? null)];
      }
      default: return [];
    }
  }

  private publicReplyDelta(text: string): string {
    const header = "[[echomind:reply]]";
    if (this.publicHeader === null) return "";
    if (this.publicHeader !== header) {
      const prefix = (this.publicHeader + text).trimStart();
      if (!prefix.startsWith(header)) {
        this.publicHeader = header.startsWith(prefix) ? prefix : null;
        return "";
      }
      this.publicHeader = header;
      text = prefix.slice(header.length);
    }
    const chunk = this.pendingTag + text;
    const length = publicTextLength(chunk);
    this.pendingTag = chunk.slice(length);
    if (REASONING_TAG.test(this.pendingTag)) {
      this.publicHeader = null;
      this.pendingTag = "";
    }
    return chunk.slice(0, length);
  }

  private startMessage(message: AgentMessage): RunnerEvent[] {
    if (message.role !== "user" && message.role !== "assistant") return [];
    this.publicHeader = "";
    this.pendingTag = "";
    this.streamedProjectReply = false;
    this.publishedArguments.clear();
    this.messageId = randomUUID();
    this.messageRole = message.role;
    this.commandId = message.role === "user" ? this.pendingCommandIds.shift() : undefined;
    if (this.commandId) this.turnId = this.commandId;
    return [this.make("message.started", { message_id: this.messageId, role: message.role, ...(this.commandId ? { command_id: this.commandId } : {}), content: messageContent(message) })];
  }

  private endMessage(message: AgentMessage, directReply: boolean, projectDialogue: boolean): RunnerEvent[] {
    if (message.role !== "user" && message.role !== "assistant") return [];
    const id = this.messageRole === message.role && this.messageId ? this.messageId : randomUUID();
    this.messageId = undefined;
    this.messageRole = undefined;
    const content = messageContent(message, message.role === "assistant"
      ? (text) => text.replace(MARKDOWN_LINK_PATTERN, (_link, label: string, url: string) =>
          `${label}${this.storageUrls.get(signedUrlIdentity(url) ?? "") ?? url})`)
      : undefined);
    // A final answer contains no tool calls. Intermediate tool commentary stays internal.
    let reply: string | undefined;
    if ((directReply || projectDialogue) && message.role === "assistant" && message.stopReason === "stop"
      && !message.content.some(block => block.type === "toolCall")) {
      const text = content.flatMap(block => isRecord(block) && typeof block.text === "string" ? [block.text] : []).join("\n");
      if (!REASONING_TAG.test(text)) reply = text.trimStart().replace(/^\[\[echomind:reply\]\]/, "").trim();
    }
    const events = [this.make("message.completed", {
      message_id: id, role: message.role, ...(this.commandId ? { command_id: this.commandId } : {}), content,
      ...(directReply && reply !== undefined ? { preparation_reply: reply } : {}),
      ...(projectDialogue && message.role === "assistant" && (reply || this.streamedProjectReply)
        ? { project_reply: reply || null } : {}),
    })];
    this.commandId = undefined;
    if (message.role === "assistant") {
      events.push(this.make("usage.updated", usagePayload(message.usage)));
    }
    return events;
  }

  private make(kind: RunnerEventKind, payload: JsonObject, turnId: string | null | undefined = this.turnId): RunnerEvent {
    return { protocolVersion: PROTOCOL_VERSION, type: "pi.event", eventId: randomUUID(), sessionId: this.sessionId,
      runId: this.runId, turnId: turnId ?? undefined, taskId: this.taskId, occurredAt: new Date().toISOString(), kind, payload };
  }
}

// Hold a possible reasoning delimiter across network chunks. Other HTML and
// ordinary less-than signs can be streamed as soon as they are unambiguous.
function publicTextLength(text: string): number {
  const complete = REASONING_TAG.exec(text);
  if (complete) return complete.index;
  const last = text.lastIndexOf("<");
  if (last < 0) return text.length;
  const suffix = text.slice(last).toLowerCase();
  return ["think", "thinking", "analysis", "reasoning"].some(name =>
    (`<${name}`).startsWith(suffix) || (`</${name}`).startsWith(suffix)) ? last : text.length;
}

function visibleToolName(name: string): string { return name === "bash" ? "terminal" : name; }

const URL_PATTERN = /https?:\/\/[^\s<>"'`\\)\]]+/g;
const MARKDOWN_LINK_PATTERN = /(\[[^\]\n]*\]\()(https?:\/\/[^\s<>"'`\\)\]]+)\)/g;

function signedUrlIdentity(value: string): string | undefined {
  try {
    const url = new URL(value);
    if (!url.searchParams.has("X-Amz-Signature")) return undefined;
    url.searchParams.delete("X-Amz-Signature");
    url.searchParams.sort();
    return url.href;
  } catch {
    return undefined;
  }
}

function messageContent(message: UserMessage | AssistantMessage, restore: (text: string) => string = (text) => text): JsonValue[] {
  if (typeof message.content === "string") return [{ type: "text", text: restore(message.content) }];
  return message.content.flatMap((block): JsonValue[] => block.type === "text"
    ? [{ type: "text", text: restore(block.text) }]
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
