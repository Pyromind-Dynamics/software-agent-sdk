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
  timer: NodeJS.Timeout;
}

export interface RunnerRpcClient {
  request(method: string, params: JsonObject, signal?: AbortSignal): Promise<JsonValue>;
}

export interface JsonlRpcPeerOptions {
  requestTimeoutMs?: number;
}

export const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;

export class JsonlRpcPeer implements RunnerRpcClient {
  private readonly input: Readable;
  private readonly output: Writable;
  private readonly requestTimeoutMs: number;
  private readonly pending = new Map<string, PendingRequest>();
  private handler: RpcRequestHandler | undefined;

  constructor(input: Readable, output: Writable, options: JsonlRpcPeerOptions = {}) {
    this.input = input;
    this.output = output;
    this.requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  }

  async listen(handler: RpcRequestHandler): Promise<void> {
    this.handler = handler;
    const lines = createInterface({ input: this.input, crlfDelay: Infinity });
    for await (const line of lines) {
      if (line.trim().length === 0) continue;
      this.handleMessage(decodeMessage(line));
    }
    const error = new Error("Python PiAdapter closed the JSONL stream");
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
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
      const fail = (reason: unknown): void => {
        if (!this.pending.delete(requestId)) return;
        const error = reason instanceof Error ? reason : new Error(String(reason));
        reject(error);
        this.write({
          protocolVersion: PROTOCOL_VERSION,
          type: "request",
          requestId: randomUUID(),
          method: "rpc.cancel",
          params: { request_id: requestId },
        });
      };
      const onAbort = (): void => {
        fail(signal?.reason instanceof Error ? signal.reason : new Error("Request aborted"));
      };
      const timer = setTimeout(
        () => fail(new Error(`JSONL RPC request timed out after ${this.requestTimeoutMs}ms: ${method}`)),
        this.requestTimeoutMs,
      );
      this.pending.set(requestId, {
        resolve: (value) => {
          clearTimeout(timer);
          signal?.removeEventListener("abort", onAbort);
          resolve(value);
        },
        reject: (error) => {
          clearTimeout(timer);
          signal?.removeEventListener("abort", onAbort);
          reject(error);
        },
        timer,
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
