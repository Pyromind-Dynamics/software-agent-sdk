import { randomUUID } from "node:crypto";
import { createInterface } from "node:readline";
import type { Readable, Writable } from "node:stream";
import {
  decodeFrame,
  encodeFrame,
  PROTOCOL_VERSION,
  type Frame,
  type JsonObject,
  type JsonValue,
  type RequestFrame,
  type ResponseFrame,
  type RunnerEvent,
} from "./protocol.js";

export type RequestHandler = (method: string, params: JsonObject) => Promise<JsonValue>;

export class JsonlRpcPeer {
  private readonly pending = new Map<string, {
    resolve: (value: JsonValue) => void;
    reject: (error: Error) => void;
  }>();
  private handler: RequestHandler | undefined;

  constructor(private readonly input: Readable, private readonly output: Writable) {}

  async listen(handler: RequestHandler): Promise<void> {
    this.handler = handler;
    const lines = createInterface({ input: this.input, crlfDelay: Infinity });
    for await (const line of lines) {
      if (line.trim()) this.handle(decodeFrame(line));
    }
    const error = new Error("PiAdapter closed the JSONL stream");
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
  }

  request(method: string, params: JsonObject, signal?: AbortSignal): Promise<JsonValue> {
    signal?.throwIfAborted();
    const requestId = randomUUID();
    return new Promise((resolve, reject) => {
      const onAbort = (): void => {
        if (!this.pending.delete(requestId)) return;
        reject(signal?.reason instanceof Error ? signal.reason : new Error("request aborted"));
      };
      this.pending.set(requestId, {
        resolve: (value) => { signal?.removeEventListener("abort", onAbort); resolve(value); },
        reject: (error) => { signal?.removeEventListener("abort", onAbort); reject(error); },
      });
      signal?.addEventListener("abort", onAbort, { once: true });
      this.write({ protocolVersion: PROTOCOL_VERSION, type: "request", requestId, method, params });
    });
  }

  emit(event: RunnerEvent): void { this.write(event); }

  private handle(frame: Frame): void {
    if (frame.type === "response") {
      const pending = this.pending.get(frame.requestId);
      if (!pending) return;
      this.pending.delete(frame.requestId);
      if (frame.error) pending.reject(new Error(`${frame.error.code}: ${frame.error.message}`));
      else pending.resolve(frame.result ?? null);
      return;
    }
    if (frame.type === "request") {
      void this.handleRequest(frame);
      return;
    }
    throw new Error("PiAdapter must not send event frames");
  }

  private async handleRequest(request: RequestFrame): Promise<void> {
    let response: ResponseFrame;
    try {
      if (!this.handler) throw new Error("runner request handler is not ready");
      response = { protocolVersion: PROTOCOL_VERSION, type: "response", requestId: request.requestId,
        result: await this.handler(request.method, request.params) };
    } catch (error) {
      response = { protocolVersion: PROTOCOL_VERSION, type: "response", requestId: request.requestId,
        error: { code: "request_failed", message: error instanceof Error ? error.message : "request failed" } };
    }
    this.write(response);
  }

  private write(frame: Frame): void { this.output.write(`${encodeFrame(frame)}\n`); }
}
