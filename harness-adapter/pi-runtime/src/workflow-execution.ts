import type { BashOperations } from "@earendil-works/pi-coding-agent";
import { StringDecoder } from "node:string_decoder";
import type { JsonObject, JsonValue } from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";

export interface ExecutionAccess {
  operations: BashOperations;
  writeFile(path: string, content: Buffer): Promise<void>;
  resolvePath(path: string, operation: "read" | "write"): Promise<string>;
  cwd: string;
  env?: NodeJS.ProcessEnv;
  confirmsAbort?: boolean;
}

export class WorkflowExecution {
  private readonly running = new Map<string, AbortController>();
  constructor(private readonly access: ExecutionAccess, private readonly peer: JsonlRpcPeer) {}

  async handle(method: string, params: JsonObject): Promise<JsonValue> {
    if (method === "execution.resolve") {
      if (typeof params.path !== "string") throw new Error("path is required");
      return { path: await this.access.resolvePath(params.path, params.write === true ? "write" : "read") };
    }
    if (method === "execution.write") {
      if (typeof params.path !== "string" || typeof params.content !== "string") {
        throw new Error("file path and base64 content are required");
      }
      const path = await this.access.resolvePath(params.path, "write");
      await this.access.writeFile(path, Buffer.from(params.content, "base64"));
      return { path };
    }
    const id = params.id;
    if (typeof id !== "string") throw new Error("execution id is required");
    if (method === "execution.cancel") {
      this.running.get(id)?.abort();
      return { requested: true };
    }
    if (method !== "execution.run" || typeof params.command !== "string") throw new Error("invalid execution request");
    if (this.running.has(id)) throw new Error("execution already active");
    const controller = new AbortController();
    this.running.set(id, controller);
    const decoder = new StringDecoder("utf8");
    let output = "";
    let truncated = false;
    const updates = new Set<Promise<unknown>>();
    let pendingBytes = 0;
    let transportError: Error | undefined;
    const consume = (text: string): void => {
      output += text;
      if (output.length > 65536) { output = output.slice(-65536); truncated = true; }
      if (params.stream === true) {
        for (let start = 0; start < text.length; start += 16384) {
          const chunk = text.slice(start, start + 16384);
          pendingBytes += Buffer.byteLength(chunk);
          if (pendingBytes > 1024 * 1024) {
            transportError = new Error("workflow log receiver is too slow; execution interrupted");
            controller.abort();
            return;
          }
          const pending = this.peer.request("execution.output", { id, text: chunk })
            .catch((error: Error) => { transportError = error; controller.abort(); })
            .finally(() => { updates.delete(pending); pendingBytes -= Buffer.byteLength(chunk); });
          updates.add(pending);
        }
      }
    };
    try {
      const cwd = typeof params.cwd === "string"
        ? await this.access.resolvePath(params.cwd, "write") : this.access.cwd;
      const command = typeof params.cwd === "string"
        ? `cd ${"'" + cwd.replaceAll("'", "'\"'\"'") + "'"} && ${params.command}` : params.command;
      const result = await this.access.operations.exec(command, this.access.cwd, {
        env: this.access.env,
        signal: controller.signal,
        timeout: typeof params.timeout === "number" ? params.timeout : 300,
        onData: (data) => consume(decoder.write(data)),
      });
      consume(decoder.end());
      await Promise.all(updates);
      if (transportError) throw transportError;
      return { rc: result.exitCode, stdout: output, truncated };
    } catch (error) {
      if (transportError) throw transportError;
      if (controller.signal.aborted && this.access.confirmsAbort && error instanceof Error
          && /(^|: )aborted$/.test(error.message)) {
        return { rc: null, stdout: output, stopped: true, truncated };
      }
      throw error;
    } finally {
      this.running.delete(id);
      await Promise.allSettled(updates);
    }
  }
}
