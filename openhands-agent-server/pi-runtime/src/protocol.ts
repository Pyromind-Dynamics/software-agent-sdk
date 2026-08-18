export const PROTOCOL_VERSION = 1 as const;

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export type PiRunnerEventKind =
  | "agent.started"
  | "agent.completed"
  | "agent.failed"
  | "agent.cancelled"
  | "message.started"
  | "message.delta"
  | "message.completed"
  | "tool.started"
  | "tool.progress"
  | "tool.completed"
  | "tool.failed"
  | "usage.updated"
  | "resource.updated";

export interface PiRunnerEvent {
  protocolVersion: typeof PROTOCOL_VERSION;
  type: "pi.event";
  eventId: string;
  sessionId: string;
  runId: string;
  occurredAt: string;
  kind: PiRunnerEventKind;
  payload: JsonObject;
}

export interface PiRunnerRequest {
  protocolVersion: typeof PROTOCOL_VERSION;
  type: "request";
  requestId: string;
  method: string;
  params: JsonObject;
}

export interface PiRunnerResponse {
  protocolVersion: typeof PROTOCOL_VERSION;
  type: "response";
  requestId: string;
  result?: JsonValue;
  error?: {
    code: string;
    message: string;
  };
}

export type PiRunnerMessage = PiRunnerEvent | PiRunnerRequest | PiRunnerResponse;

export function encodeMessage(message: PiRunnerMessage): string {
  return JSON.stringify(message);
}

export function decodeMessage(line: string): PiRunnerMessage {
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch {
    throw new Error("invalid JSONL message");
  }
  if (!isRecord(parsed) || parsed.protocolVersion !== PROTOCOL_VERSION) {
    throw new Error(`unsupported protocol version; expected ${PROTOCOL_VERSION}`);
  }
  if (parsed.type === "request") return decodeRequest(parsed);
  if (parsed.type === "response") return decodeResponse(parsed);
  if (parsed.type === "pi.event") return decodeEvent(parsed);
  throw new Error("unknown JSONL message type");
}

function decodeRequest(value: Record<string, unknown>): PiRunnerRequest {
  if (!isNonEmptyString(value.requestId) || !isNonEmptyString(value.method) || !isRecord(value.params)) {
    throw new Error("invalid request message");
  }
  return {
    protocolVersion: PROTOCOL_VERSION,
    type: "request",
    requestId: value.requestId,
    method: value.method,
    params: value.params as JsonObject,
  };
}

function decodeResponse(value: Record<string, unknown>): PiRunnerResponse {
  if (!isNonEmptyString(value.requestId)) throw new Error("invalid response message");
  const hasResult = Object.hasOwn(value, "result");
  const hasError = Object.hasOwn(value, "error");
  if (hasResult === hasError) throw new Error("response must contain exactly one of result or error");
  if (hasError) {
    if (!isRecord(value.error) || !isNonEmptyString(value.error.code) || typeof value.error.message !== "string") {
      throw new Error("invalid response error");
    }
  }
  if (hasError) {
    const error = value.error as { code: string; message: string };
    return {
      protocolVersion: PROTOCOL_VERSION,
      type: "response",
      requestId: value.requestId,
      error,
    };
  }
  return {
    protocolVersion: PROTOCOL_VERSION,
    type: "response",
    requestId: value.requestId,
    result: value.result as JsonValue,
  };
}

function decodeEvent(value: Record<string, unknown>): PiRunnerEvent {
  if (
    !isNonEmptyString(value.eventId) ||
    !isNonEmptyString(value.sessionId) ||
    !isNonEmptyString(value.runId) ||
    !isNonEmptyString(value.occurredAt) ||
    !isEventKind(value.kind) ||
    !isRecord(value.payload)
  ) {
    throw new Error("invalid Pi event message");
  }
  return {
    protocolVersion: PROTOCOL_VERSION,
    type: "pi.event",
    eventId: value.eventId,
    sessionId: value.sessionId,
    runId: value.runId,
    occurredAt: value.occurredAt,
    kind: value.kind,
    payload: value.payload as JsonObject,
  };
}

function isEventKind(value: unknown): value is PiRunnerEventKind {
  return (
    typeof value === "string" &&
    [
      "agent.started",
      "agent.completed",
      "agent.failed",
      "agent.cancelled",
      "message.started",
      "message.delta",
      "message.completed",
      "tool.started",
      "tool.progress",
      "tool.completed",
      "tool.failed",
      "usage.updated",
      "resource.updated",
    ].includes(value)
  );
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}
