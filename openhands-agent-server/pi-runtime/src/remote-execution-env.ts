import {
  err,
  ExecutionError,
  type ExecutionErrorCode,
  type ExecutionEnv,
  FileError,
  type FileErrorCode,
  type FileInfo,
  ok,
  type Result,
  type ShellExecOptions,
} from "@earendil-works/pi-agent-core";
import type { JsonObject, JsonValue } from "./protocol.ts";
import type { RunnerRpcClient } from "./rpc-peer.ts";

interface RpcResult {
  ok: boolean;
  value?: JsonValue;
  error?: {
    code?: string;
    message?: string;
    path?: string;
  };
}

export class RemoteExecutionEnv implements ExecutionEnv {
  public cwd: string;
  private readonly rpc: RunnerRpcClient;

  constructor(rpc: RunnerRpcClient, cwd: string) {
    this.rpc = rpc;
    this.cwd = cwd;
  }

  absolutePath(path: string, abortSignal?: AbortSignal): Promise<Result<string, FileError>> {
    return this.fileCall("absolutePath", { path }, isString, abortSignal);
  }

  joinPath(parts: string[], abortSignal?: AbortSignal): Promise<Result<string, FileError>> {
    return this.fileCall("joinPath", { parts }, isString, abortSignal);
  }

  readTextFile(path: string, abortSignal?: AbortSignal): Promise<Result<string, FileError>> {
    return this.fileCall("readTextFile", { path }, isString, abortSignal);
  }

  readTextLines(
    path: string,
    options?: { maxLines?: number; abortSignal?: AbortSignal },
  ): Promise<Result<string[], FileError>> {
    return this.fileCall(
      "readTextLines",
      { path, ...(options?.maxLines === undefined ? {} : { maxLines: options.maxLines }) },
      isStringArray,
      options?.abortSignal,
    );
  }

  async readBinaryFile(path: string, abortSignal?: AbortSignal): Promise<Result<Uint8Array, FileError>> {
    const result = await this.fileCall("readBinaryFile", { path }, isString, abortSignal);
    if (!result.ok) return result;
    return ok(Buffer.from(result.value, "base64"));
  }

  writeFile(
    path: string,
    content: string | Uint8Array,
    abortSignal?: AbortSignal,
  ): Promise<Result<void, FileError>> {
    return this.fileVoidCall("writeFile", contentParams(path, content), abortSignal);
  }

  appendFile(
    path: string,
    content: string | Uint8Array,
    abortSignal?: AbortSignal,
  ): Promise<Result<void, FileError>> {
    return this.fileVoidCall("appendFile", contentParams(path, content), abortSignal);
  }

  renameFile(
    sourcePath: string,
    destinationPath: string,
    abortSignal?: AbortSignal,
  ): Promise<Result<void, FileError>> {
    return this.fileVoidCall("renameFile", { sourcePath, destinationPath }, abortSignal);
  }

  fileInfo(path: string, abortSignal?: AbortSignal): Promise<Result<FileInfo, FileError>> {
    return this.fileCall("fileInfo", { path }, isFileInfo, abortSignal);
  }

  listDir(path: string, abortSignal?: AbortSignal): Promise<Result<FileInfo[], FileError>> {
    return this.fileCall("listDir", { path }, isFileInfoArray, abortSignal);
  }

  canonicalPath(path: string, abortSignal?: AbortSignal): Promise<Result<string, FileError>> {
    return this.fileCall("canonicalPath", { path }, isString, abortSignal);
  }

  exists(path: string, abortSignal?: AbortSignal): Promise<Result<boolean, FileError>> {
    return this.fileCall("exists", { path }, isBoolean, abortSignal);
  }

  createDir(
    path: string,
    options?: { recursive?: boolean; abortSignal?: AbortSignal },
  ): Promise<Result<void, FileError>> {
    return this.fileVoidCall(
      "createDir",
      { path, recursive: options?.recursive ?? true },
      options?.abortSignal,
    );
  }

  remove(
    path: string,
    options?: { recursive?: boolean; force?: boolean; abortSignal?: AbortSignal },
  ): Promise<Result<void, FileError>> {
    return this.fileVoidCall(
      "remove",
      { path, recursive: options?.recursive ?? false, force: options?.force ?? false },
      options?.abortSignal,
    );
  }

  createTempDir(prefix?: string, abortSignal?: AbortSignal): Promise<Result<string, FileError>> {
    return this.fileCall("createTempDir", prefix === undefined ? {} : { prefix }, isString, abortSignal);
  }

  createTempFile(options?: {
    prefix?: string;
    suffix?: string;
    abortSignal?: AbortSignal;
  }): Promise<Result<string, FileError>> {
    return this.fileCall(
      "createTempFile",
      {
        ...(options?.prefix === undefined ? {} : { prefix: options.prefix }),
        ...(options?.suffix === undefined ? {} : { suffix: options.suffix }),
      },
      isString,
      options?.abortSignal,
    );
  }

  async exec(
    command: string,
    options?: ShellExecOptions,
  ): Promise<Result<{ stdout: string; stderr: string; exitCode: number }, ExecutionError>> {
    try {
      const response = parseRpcResult(
        await this.rpc.request(
          "env.exec",
          {
            command,
            ...(options?.cwd === undefined ? {} : { cwd: options.cwd }),
            env: options?.env ?? {},
            inheritEnv: options?.inheritEnv ?? true,
            ...(options?.timeout === undefined ? {} : { timeout: options.timeout }),
          },
          options?.abortSignal,
        ),
      );
      if (!response.ok) return err(executionError(response.error, options?.abortSignal));
      if (!isExecResult(response.value)) return err(new ExecutionError("unknown", "Invalid sandbox response"));
      options?.onStdout?.(response.value.stdout);
      options?.onStderr?.(response.value.stderr);
      return ok(response.value);
    } catch (error) {
      return err(exceptionExecutionError(error, options?.abortSignal));
    }
  }

  async cleanup(): Promise<void> {
    try {
      await this.rpc.request("env.cleanup", {});
    } catch {}
  }

  private async fileCall<T>(
    method: string,
    params: JsonObject,
    validate: (value: unknown) => value is T,
    signal?: AbortSignal,
  ): Promise<Result<T, FileError>> {
    try {
      const response = parseRpcResult(await this.rpc.request(`env.${method}`, params, signal));
      if (!response.ok) return err(fileError(response.error, signal));
      if (!validate(response.value)) return err(new FileError("unknown", "Invalid sandbox response"));
      return ok(response.value);
    } catch (error) {
      return err(exceptionFileError(error, signal));
    }
  }

  private async fileVoidCall(
    method: string,
    params: JsonObject,
    signal?: AbortSignal,
  ): Promise<Result<void, FileError>> {
    const result = await this.fileCall(method, params, (value): value is null => value === null, signal);
    return result.ok ? ok(undefined) : result;
  }
}

function contentParams(path: string, content: string | Uint8Array): JsonObject {
  return typeof content === "string"
    ? { path, content, encoding: "utf8" }
    : { path, content: Buffer.from(content).toString("base64"), encoding: "base64" };
}

function parseRpcResult(value: JsonValue): RpcResult {
  if (!isRecord(value) || typeof value.ok !== "boolean") throw new Error("Invalid sandbox RPC result");
  const error = isRecord(value.error)
    ? {
        ...(typeof value.error.code === "string" ? { code: value.error.code } : {}),
        ...(typeof value.error.message === "string" ? { message: value.error.message } : {}),
        ...(typeof value.error.path === "string" ? { path: value.error.path } : {}),
      }
    : undefined;
  return {
    ok: value.ok,
    ...(Object.hasOwn(value, "value") ? { value: value.value as JsonValue } : {}),
    ...(error ? { error } : {}),
  };
}

function fileError(error: RpcResult["error"], signal?: AbortSignal): FileError {
  const code = signal?.aborted ? "aborted" : fileErrorCode(error?.code);
  return new FileError(code, error?.message ?? "Sandbox file operation failed", error?.path);
}

function executionError(error: RpcResult["error"], signal?: AbortSignal): ExecutionError {
  const code = signal?.aborted ? "aborted" : executionErrorCode(error?.code);
  return new ExecutionError(code, error?.message ?? "Sandbox command failed");
}

function exceptionFileError(error: unknown, signal?: AbortSignal): FileError {
  return new FileError(
    signal?.aborted ? "aborted" : "unknown",
    error instanceof Error ? error.message : "Sandbox file operation failed",
  );
}

function exceptionExecutionError(error: unknown, signal?: AbortSignal): ExecutionError {
  return new ExecutionError(
    signal?.aborted ? "aborted" : "unknown",
    error instanceof Error ? error.message : "Sandbox command failed",
  );
}

function fileErrorCode(value: string | undefined): FileErrorCode {
  switch (value) {
    case "aborted":
    case "not_found":
    case "permission_denied":
    case "not_directory":
    case "is_directory":
    case "invalid":
    case "not_supported":
      return value;
    default:
      return "unknown";
  }
}

function executionErrorCode(value: string | undefined): ExecutionErrorCode {
  switch (value) {
    case "aborted":
    case "timeout":
    case "shell_unavailable":
    case "spawn_error":
    case "callback_error":
      return value;
    default:
      return "unknown";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isBoolean(value: unknown): value is boolean {
  return typeof value === "boolean";
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isFileInfo(value: unknown): value is FileInfo {
  return (
    isRecord(value) &&
    typeof value.name === "string" &&
    typeof value.path === "string" &&
    (value.kind === "file" || value.kind === "directory" || value.kind === "symlink") &&
    typeof value.size === "number" &&
    typeof value.mtimeMs === "number"
  );
}

function isFileInfoArray(value: unknown): value is FileInfo[] {
  return Array.isArray(value) && value.every(isFileInfo);
}

function isExecResult(
  value: unknown,
): value is { stdout: string; stderr: string; exitCode: number } {
  return (
    isRecord(value) &&
    typeof value.stdout === "string" &&
    typeof value.stderr === "string" &&
    typeof value.exitCode === "number"
  );
}
