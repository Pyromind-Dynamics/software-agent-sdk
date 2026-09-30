import type { BashOperations } from "@earendil-works/pi-coding-agent";
import { serveJudge, type ModelJudge } from "./workflow-judge.js";
import { join, resolve, relative } from "node:path";
import { StringDecoder } from "node:string_decoder";
import type { JsonObject, JsonValue } from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";

export interface ExecutionAccess {
  operations: BashOperations;
  readFile?(path: string, limit: number): Promise<Buffer>;
  protectedRoot?: string;
  writeFile(path: string, content: Buffer): Promise<void>;
  resolvePath(path: string, operation: "read" | "write"): Promise<string>;
  cwd: string;
  env?: NodeJS.ProcessEnv;
  confirmsAbort?: boolean;
}

export class WorkflowExecution {
  private readonly running = new Map<string, AbortController>();
  constructor(private readonly access: ExecutionAccess, private readonly peer: JsonlRpcPeer, private readonly modelJudge?: ModelJudge) {}

  async handle(method: string, params: JsonObject): Promise<JsonValue> {
    if (method === "execution.capabilities") return {
      agent_task: !!this.access.protectedRoot,
      protected_verification: !!this.access.protectedRoot,
      file_read: !!this.access.readFile,
      revision: !!this.access.readFile,
      model_judge: !!this.access.protectedRoot && !!this.modelJudge,
    };
    if (method === "execution.protect") {
      const { run_id, files } = params;
      if (!this.access.protectedRoot || typeof run_id !== "string" || !/^[a-f0-9]{32}$/.test(run_id)
          || !Array.isArray(files)) throw new Error("protected verification is unavailable");
      const root = join(this.access.protectedRoot, run_id);
      for (const item of files) {
        if (!item || typeof item !== "object" || Array.isArray(item)
            || typeof item.path !== "string" || typeof item.content !== "string") throw new Error("invalid protected file");
        const target = resolve(root, item.path);
        if (!relative(root, target) || relative(root, target).startsWith("..")) throw new Error("invalid protected path");
        await this.access.writeFile(target, Buffer.from(item.content, "base64"));
      }
      return { path: root };
    }
    if (method === "execution.read") {
      if (!this.access.readFile || typeof params.path !== "string") throw new Error("file reading is unavailable");
      const limit = params.limit;
      if (typeof limit !== "number" || !Number.isInteger(limit) || limit < 1 || limit > 262144) throw new Error("invalid file size limit");
      const path = await this.access.resolvePath(params.path, "read");
      const content = await this.access.readFile(path, limit);
      if (content.length > limit) throw new Error("file exceeds size limit");
      return { content: content.toString("base64") };
    }
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
    const judgeController = new AbortController();
    let judgeTask: Promise<void> | undefined;
    let judgeError: unknown;
    let judgeDirectory: string | undefined;
    try {
    if (typeof params.judge_directory === "string") {
      if (!this.modelJudge) throw new Error("model evaluation unavailable");
      judgeDirectory = await this.access.resolvePath(params.judge_directory, "write");
      judgeTask = serveJudge(this.access, judgeDirectory, this.modelJudge,
        AbortSignal.any([controller.signal, judgeController.signal]))
        .catch((error: unknown) => { judgeError = error; controller.abort(); });
    }
      const cwd = typeof params.cwd === "string"
        ? await this.access.resolvePath(params.cwd, "read") : this.access.cwd;
      const command = typeof params.cwd === "string"
        ? `cd ${"'" + cwd.replaceAll("'", "'\"'\"'") + "'"} && ${params.command}` : params.command;
      const result = await this.access.operations.exec(command, this.access.cwd, {
        env: { ...this.access.env, ...(judgeDirectory ? { AGENTGENOME_JUDGE_DIR: judgeDirectory } : {}) },
        signal: controller.signal,
        timeout: typeof params.timeout === "number" ? params.timeout : 300,
        onData: (data) => consume(decoder.write(data)),
      });
      consume(decoder.end());
      await Promise.all(updates);
      if (judgeError) throw judgeError;
      if (transportError) throw transportError;
      return { rc: result.exitCode, stdout: output, truncated };
    } catch (error) {
      if (judgeError) throw judgeError;
      if (transportError) throw transportError;
      if (controller.signal.aborted && this.access.confirmsAbort && error instanceof Error
          && /(^|: )aborted$/.test(error.message)) {
        return { rc: null, stdout: output, stopped: true, truncated };
      }
      throw error;
    } finally {
      judgeController.abort();
      await judgeTask;
      this.running.delete(id);
      await Promise.allSettled(updates);
    }
  }
}
