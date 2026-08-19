export const PROTOCOL_VERSION = 1 as const;
export const MAX_FRAME_BYTES = 1024 * 1024;

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export type RunnerEventKind =
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
  | "turn.completed";

export interface RunnerEvent {
  protocolVersion: typeof PROTOCOL_VERSION;
  type: "pi.event";
  eventId: string;
  sessionId: string;
  runId: string;
  occurredAt: string;
  kind: RunnerEventKind;
  payload: JsonObject;
}

export interface RequestFrame {
  protocolVersion: typeof PROTOCOL_VERSION;
  type: "request";
  requestId: string;
  method: string;
  params: JsonObject;
}

export interface ResponseFrame {
  protocolVersion: typeof PROTOCOL_VERSION;
  type: "response";
  requestId: string;
  result?: JsonValue;
  error?: { code: string; message: string };
}

export type Frame = RunnerEvent | RequestFrame | ResponseFrame;

export function encodeFrame(frame: Frame): string {
  const line = JSON.stringify(frame);
  if (Buffer.byteLength(line) > MAX_FRAME_BYTES) throw new Error("JSONL frame exceeds size limit");
  return line;
}

export function decodeFrame(line: string): Frame {
  if (Buffer.byteLength(line) > MAX_FRAME_BYTES) throw new Error("JSONL frame exceeds size limit");
  let value: unknown;
  try {
    value = JSON.parse(line);
  } catch {
    throw new Error("invalid JSONL frame");
  }
  if (!isRecord(value) || value.protocolVersion !== PROTOCOL_VERSION) {
    throw new Error(`unsupported protocol version; expected ${PROTOCOL_VERSION}`);
  }
  if (value.type === "request") {
    if (!text(value.requestId) || !text(value.method) || !isRecord(value.params)) {
      throw new Error("invalid request frame");
    }
    return value as unknown as RequestFrame;
  }
  if (value.type === "response") {
    if (!text(value.requestId) || Object.hasOwn(value, "result") === Object.hasOwn(value, "error")) {
      throw new Error("invalid response frame");
    }
    if (Object.hasOwn(value, "error")) {
      if (!isRecord(value.error) || !text(value.error.code) || typeof value.error.message !== "string") {
        throw new Error("invalid response error");
      }
    }
    return value as unknown as ResponseFrame;
  }
  if (value.type === "pi.event") {
    if (!text(value.eventId) || !text(value.sessionId) || !text(value.runId) ||
        !text(value.occurredAt) || !text(value.kind) || !isRecord(value.payload)) {
      throw new Error("invalid Pi event frame");
    }
    return value as unknown as RunnerEvent;
  }
  throw new Error("unknown JSONL frame type");
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function text(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}
