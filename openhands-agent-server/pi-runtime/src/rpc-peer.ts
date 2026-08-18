import { randomUUID } from "node:crypto";
import { createInterface } from "node:readline";
import type { Readable, Writable } from "node:stream";
import {
  decodeMessage,
  encodeMessage,
  PROTOCOL_VERSION,
  type JsonObject,
  type JsonValue,
  type PiRunnerEvent,
  type PiRunnerRequest,
  type PiRunnerResponse,
} from "./protocol.ts";

export type RpcRequestHandler = (method: string, params: JsonObject) => Promise<JsonValue>;

interface PendingRequest {
  resolve(value: JsonValue): void;
  reject(error: Error): void;
}

export interface RunnerRpcClient {
  request(method: string, params: JsonObject, signal?: AbortSignal): Promise<JsonValue>;
}

export class JsonlRpcPeer implements RunnerRpcClient {
  private readonly input: Readable;
  private readonly output: Writable;
  private readonly pending = new Map<string, PendingRequest>();
  private handler: RpcRequestHandler | undefined;

  constructor(input: Readable, output: Writable) {
    this.input = input;
    this.output = output;
  }

  async listen(handler: RpcRequestHandler): Promise<void> {
    this.handler = handler;
    const lines = createInterface({ input: this.input, crlfDelay: Infinity });
    for await (const line of lines) {
      if (line.trim().length === 0) continue;
      this.handleMessage(decodeMessage(line));
    }
    const error = new Error("Python PiAdapter closed the JSONL stream");
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  request(method: string, params: JsonObject, signal?: AbortSignal): Promise<JsonValue> {
    signal?.throwIfAborted();
    const requestId = randomUUID();
    const request: PiRunnerRequest = {
      protocolVersion: PROTOCOL_VERSION,
      type: "request",
      requestId,
      method,
      params,
    };
    return new Promise<JsonValue>((resolve, reject) => {
      const onAbort = (): void => {
        if (!this.pending.delete(requestId)) return;
        reject(signal?.reason instanceof Error ? signal.reason : new Error("Request aborted"));
        this.write({
          protocolVersion: PROTOCOL_VERSION,
          type: "request",
          requestId: randomUUID(),
          method: "rpc.cancel",
          params: { request_id: requestId },
        });
      };
      this.pending.set(requestId, {
        resolve: (value) => {
          signal?.removeEventListener("abort", onAbort);
          resolve(value);
        },
        reject: (error) => {
          signal?.removeEventListener("abort", onAbort);
          reject(error);
        },
      });
      signal?.addEventListener("abort", onAbort, { once: true });
      this.write(request);
    });
  }

  emit(event: PiRunnerEvent): void {
    this.write(event);
  }

  private handleMessage(message: PiRunnerEvent | PiRunnerRequest | PiRunnerResponse): void {
    if (message.type === "response") {
      const pending = this.pending.get(message.requestId);
      if (!pending) return;
      this.pending.delete(message.requestId);
      if (message.error) {
        pending.reject(new Error(`${message.error.code}: ${message.error.message}`));
      } else {
        pending.resolve(message.result ?? null);
      }
      return;
    }
    if (message.type === "request") {
      void this.handleRequest(message);
      return;
    }
    throw new Error("Python PiAdapter must not send pi.event messages to the runner");
  }

  private async handleRequest(request: PiRunnerRequest): Promise<void> {
    let response: PiRunnerResponse;
    try {
      if (!this.handler) throw new Error("JSONL request handler is not ready");
      const result = await this.handler(request.method, request.params);
      response = {
        protocolVersion: PROTOCOL_VERSION,
        type: "response",
        requestId: request.requestId,
        result,
      };
    } catch (error) {
      response = {
        protocolVersion: PROTOCOL_VERSION,
        type: "response",
        requestId: request.requestId,
        error: {
          code: "request_failed",
          message: error instanceof Error ? error.message : "Runner request failed",
        },
      };
    }
    this.write(response);
  }

  private write(message: PiRunnerEvent | PiRunnerRequest | PiRunnerResponse): void {
    this.output.write(`${encodeMessage(message)}\n`);
  }
}
