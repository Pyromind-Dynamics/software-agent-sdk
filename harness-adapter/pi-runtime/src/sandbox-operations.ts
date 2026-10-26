import { readFile } from "node:fs/promises";

import type {
  EditOperations,
  ReadOperations,
  WriteOperations,
} from "@earendil-works/pi-coding-agent";

import type { SandboxReadMirror } from "./sandbox-paths.js";

/** Endpoint bundle handed to the runner by the control plane on demand. */
export interface SandboxEndpoint {
  /** REST base URL (portal domain is fine for the file API). */
  baseUrl: string;
  /** Cluster-direct base URL; the portal domain does not proxy WebSocket. */
  wsBaseUrl: string;
  sandboxId: string;
  apiKey: string;
  cluster?: string;
  /** Sandbox-side conversation root. */
  workspacePath: string;
  /**
   * Sandbox-side Storage mount root, addressed by its absolute path.
   * The workspace holds a symlink with that name, so shell and script
   * relative paths resolve to the same place as the file tools.
   */
  storagePath: string;
}

export interface SandboxEndpointRequest {
  /** Re-read the sandbox instead of returning the control plane's cached one. */
  refresh: boolean;
}

export type SandboxEndpointProvider = (
  request: SandboxEndpointRequest,
) => Promise<SandboxEndpoint>;

/**
 * Session-scoped endpoint bundle.
 *
 * The control plane owns sandbox lifecycle, so a bundle this runner holds goes
 * stale while the runner keeps serving turns: the container can be rebuilt or
 * re-keyed between two calls. Marking the bundle invalid makes the next resolve
 * ask for the sandbox again instead of replaying a dead endpoint forever.
 */
export class SandboxEndpointSession {
  private cached: Promise<SandboxEndpoint> | undefined;
  private stale = false;

  constructor(private readonly provider: SandboxEndpointProvider) {}

  resolve(): Promise<SandboxEndpoint> {
    if (this.cached === undefined) {
      const requested = this.provider({ refresh: this.stale });
      this.stale = false;
      // Drop a rejected request so the next call asks again instead of
      // replaying the same failure.
      requested.catch(() => {
        if (this.cached === requested) this.cached = undefined;
      });
      this.cached = requested;
    }
    return this.cached;
  }

  invalidate(): void {
    this.cached = undefined;
    this.stale = true;
  }
}

export const SANDBOX_FILE_MAX_BYTES = 1024 * 1024 * 1024;
const UPLOAD_CHUNK_BYTES = 2 * 1024 * 1024;
const REQUEST_TIMEOUT_MS = 120_000;
/**
 * Budget for the file API's first request in a session.
 *
 * The portal domain is not cluster-direct, so a runner which cannot reach it
 * has to learn that from one short request instead of spending the full
 * request budget on every file call. Once a request has answered, later
 * requests use the full budget: a large file read may legitimately outlive
 * the probe.
 *
 * Kept short because the client retries this budget while the portal is
 * unproven, and every retry is time the terminal transport could already be
 * serving the call.
 */
export const SANDBOX_FILE_PROBE_TIMEOUT_MS = 5_000;

export class SandboxFileError extends Error {
  constructor(
    message: string,
    readonly status: number | undefined = undefined,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SandboxFileError";
  }
}

export interface SandboxFileClientOptions {
  /** See {@link SANDBOX_FILE_PROBE_TIMEOUT_MS}. */
  probeTimeoutMs?: number;
}

/** Thin HTTP client for the sandbox file API (read / chunked write / delete). */
export class SandboxFileClient {
  private portalProven = false;

  constructor(
    private readonly endpoints: SandboxEndpointSession,
    private readonly options: SandboxFileClientOptions = {},
  ) {}

  async read(path: string, limit = SANDBOX_FILE_MAX_BYTES): Promise<Buffer> {
    return this.withEndpoint(async (endpoint) => {
      const url = this.url(endpoint, `/sandboxes/${endpoint.sandboxId}/files/read`);
      url.searchParams.set("path", path);
      return this.bytes("GET", url, await this.fetch(endpoint, url), limit);
    });
  }

  async access(path: string): Promise<void> {
    await this.read(path);
  }

  async write(path: string, content: string | Buffer, signal?: AbortSignal): Promise<void> {
    const body = typeof content === "string" ? Buffer.from(content, "utf8") : content;
    await this.withEndpoint((endpoint) => this.upload(endpoint, path, body, signal), signal);
  }

  private async upload(
    endpoint: SandboxEndpoint,
    path: string,
    body: Buffer,
    signal?: AbortSignal,
  ): Promise<void> {
    const base = this.url(endpoint, `/sandboxes/${endpoint.sandboxId}/files/chunks`);
    const totalChunks = Math.max(1, Math.ceil(body.byteLength / UPLOAD_CHUNK_BYTES));
    const init = new URL(`${base}/init`);
    init.searchParams.set("total_size", String(body.byteLength));
    init.searchParams.set("total_chunks", String(totalChunks));
    const initialized = await this.json(
      await this.fetch(endpoint, init, { method: "POST", signal }),
    );
    const uploadId = readString(initialized, "upload_id");
    try {
      for (let index = 0; index < totalChunks; index += 1) {
        const chunk = body.subarray(
          index * UPLOAD_CHUNK_BYTES,
          (index + 1) * UPLOAD_CHUNK_BYTES,
        );
        const part = new URL(`${base}/${encodeURIComponent(uploadId)}/part`);
        part.searchParams.set("chunk_index", String(index));
        await this.fetch(endpoint, part, {
          method: "PUT",
          body: new Uint8Array(chunk),
          signal,
          // The body length is left to fetch: the runner imports the Pi
          // packages, whose undici installs a dispatcher that rejects an
          // explicitly set Content-Length on a byte-array body.
          headers: { "Content-Type": "application/octet-stream" },
        });
      }
      const complete = new URL(`${base}/${encodeURIComponent(uploadId)}/complete`);
      complete.searchParams.set("path", path);
      complete.searchParams.set("total_chunks", String(totalChunks));
      await this.fetch(endpoint, complete, { method: "POST", signal });
    } catch (error) {
      await this.fetch(endpoint, new URL(`${base}/${encodeURIComponent(uploadId)}`), {
        method: "DELETE",
        signal,
      }).catch(() => undefined);
      throw error;
    }
  }

  /**
   * Run one call against the current endpoint bundle, escalating one step per
   * failure.
   *
   * A transport failure usually means the network blipped while the bundle was
   * still good, so the same endpoint gets one retry. Only a failure which
   * repeats is treated as a sandbox the control plane has already rebuilt or
   * re-keyed, which is what asking for a fresh bundle repairs. An error the
   * sandbox itself returned is the caller's to handle and never retried.
   */
  private async withEndpoint<T>(
    operation: (endpoint: SandboxEndpoint) => Promise<T>,
    signal?: AbortSignal,
  ): Promise<T> {
    const endpoint = await this.endpoints.resolve();
    try {
      return await operation(endpoint);
    } catch (error) {
      if (signal?.aborted || !isTransportFailure(error)) throw error;
    }
    try {
      return await operation(endpoint);
    } catch (error) {
      if (signal?.aborted || !isTransportFailure(error)) throw error;
    }
    this.endpoints.invalidate();
    try {
      return await operation(await this.endpoints.resolve());
    } catch (error) {
      throw new SandboxFileError(
        `${describeError(error)} (retried with a refreshed sandbox endpoint)`,
        error instanceof SandboxFileError ? error.status : undefined,
        { cause: error },
      );
    }
  }

  private async bytes(method: string, url: URL, response: Response, limit: number): Promise<Buffer> {
    try {
      const reader = response.body?.getReader();
      if (!reader) return Buffer.alloc(0);
      const chunks: Buffer[] = [];
      let total = 0;
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          total += value.byteLength;
          if (total > limit) throw new Error("file exceeds size limit");
          chunks.push(Buffer.from(value));
        }
        return Buffer.concat(chunks);
      } finally { await reader.cancel(); }
    } catch (error) {
      // A connection that dies mid-body is a transport failure too.
      throw transportError(method, url, error);
    }
  }

  private url(endpoint: SandboxEndpoint, path: string): URL {
    return new URL(`${endpoint.baseUrl.replace(/\/+$/, "")}${path}`);
  }

  private async fetch(
    endpoint: SandboxEndpoint,
    url: URL,
    init: RequestInit = {},
  ): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set("Authorization", `Bearer ${endpoint.apiKey}`);
    headers.set("Accept", "application/json");
    if (endpoint.cluster) headers.set("X-Cluster", endpoint.cluster);
    let response: Response;
    try {
      const timeoutMs = this.portalProven
        ? REQUEST_TIMEOUT_MS
        : (this.options.probeTimeoutMs ?? REQUEST_TIMEOUT_MS);
      const requestSignal = init.signal
        ? AbortSignal.any([AbortSignal.timeout(timeoutMs), init.signal])
        : AbortSignal.timeout(timeoutMs);
      response = await fetch(url, {
        ...init,
        headers,
        signal: requestSignal,
      });
    } catch (error) {
      throw transportError(init.method ?? "GET", url, error);
    }
    // Any HTTP answer proves the portal is reachable, even an error status.
    this.portalProven = true;
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new SandboxFileError(
        `sandbox file request failed: ${init.method ?? "GET"} ${url.pathname} ` +
          `-> ${response.status}${detail ? ` ${detail.slice(0, 300)}` : ""}`,
        response.status,
      );
    }
    return response;
  }

  private async json(response: Response): Promise<Record<string, unknown>> {
    const payload: unknown = await response.json();
    if (!isRecord(payload)) throw new SandboxFileError("sandbox file response is not an object");
    const data = isRecord(payload.data) ? payload.data : payload;
    return data;
  }
}

/**
 * How long one file API failure keeps the client on the terminal transport.
 *
 * A portal the runner cannot reach must not cost every call its own timeout,
 * but one blip must not pin the whole session to the slow transport either:
 * the terminal pays an acknowledged round trip per staging burst where the
 * file API pays one request for the same bytes. After this cooldown the client
 * probes the file API again and drops the fallback as soon as it answers.
 */
const FILE_API_RETRY_MS = 30_000;

export interface FallbackSandboxFileClientOptions {
  /** See {@link FILE_API_RETRY_MS}. */
  retryFileApiMs?: number;
}

/**
 * File API first, terminal transport as the fallback.
 *
 * The file API is the cheap path when the runner can reach the portal: a read
 * is one HTTP request and a write is three, while the terminal transport pays
 * a command round trip per staging burst for the same bytes. A transport
 * failure switches the session to the terminal transport, and the client
 * re-probes the file API after {@link FILE_API_RETRY_MS}.
 */
export class FallbackSandboxFileClient implements SandboxFileClientLike {
  private readonly retryFileApiMs: number;
  private terminalOnlyUntil = 0;

  constructor(
    private readonly fileApi: SandboxFileClientLike,
    private readonly terminal: SandboxFileClientLike,
    options: FallbackSandboxFileClientOptions = {},
  ) {
    this.retryFileApiMs = options.retryFileApiMs ?? FILE_API_RETRY_MS;
  }

  async read(path: string, limit?: number): Promise<Buffer> {
    if (this.terminalOnly) return this.terminal.read(path, limit);
    try {
      const content = await this.fileApi.read(path, limit);
      this.recovered();
      return content;
    } catch (error) {
      return this.fallback(error, () => this.terminal.read(path, limit));
    }
  }

  async access(path: string): Promise<void> {
    if (this.terminalOnly) return this.terminal.access(path);
    try {
      await this.fileApi.access(path);
      this.recovered();
    } catch (error) {
      return this.fallback(error, () => this.terminal.access(path));
    }
  }

  async write(path: string, content: string | Buffer, signal?: AbortSignal): Promise<void> {
    if (this.terminalOnly) return this.terminal.write(path, content, signal);
    try {
      await this.fileApi.write(path, content, signal);
      this.recovered();
    } catch (error) {
      // A caller which aborted is not a portal failure: the file API reports
      // the abort as a transport error, and the terminal must not run instead.
      if (signal?.aborted) throw error;
      return this.fallback(error, () => this.terminal.write(path, content, signal));
    }
  }

  /** Whether the file API is inside the cooldown one failure bought it. */
  private get terminalOnly(): boolean {
    return this.terminalOnlyUntil > Date.now();
  }

  private recovered(): void {
    this.terminalOnlyUntil = 0;
  }

  private async fallback<T>(error: unknown, run: () => Promise<T>): Promise<T> {
    if (!isFileApiUnavailable(error)) throw error;
    this.terminalOnlyUntil = Date.now() + this.retryFileApiMs;
    return run();
  }
}

/** A file API failure which the terminal transport can answer instead. */
function isFileApiUnavailable(error: unknown): boolean {
  return error instanceof SandboxFileError
    && (error.status === undefined || error.status >= 500);
}

/**
 * undici reports every transport failure as a bare "fetch failed" and keeps the
 * actionable detail (TLS, DNS, reset socket) in `cause`.
 */
function transportError(method: string, url: URL, error: unknown): SandboxFileError {
  return new SandboxFileError(
    `sandbox file request failed: ${method} ${url.pathname} -> ${describeErrorChain(error)}`,
    undefined,
    { cause: error },
  );
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** A failure which never reached the sandbox, so the bundle may still be good. */
function isTransportFailure(error: unknown): boolean {
  return error instanceof SandboxFileError && error.status === undefined;
}

export function describeErrorChain(error: unknown): string {
  const parts: string[] = [];
  let current: unknown = error;
  while (current !== null && current !== undefined && parts.length < 5) {
    if (!(current instanceof Error)) {
      parts.push(String(current));
      break;
    }
    const code = "code" in current && typeof current.code === "string" ? current.code : "";
    parts.push(code ? `${current.message} [${code}]` : current.message);
    current = current.cause;
  }
  return parts.join(" <- ") || String(error);
}

function readString(value: Record<string, unknown>, key: string): string {
  const item = value[key];
  if (typeof item !== "string" || !item) {
    throw new SandboxFileError(`sandbox file response is missing ${key}`);
  }
  return item;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

const IMAGE_MAGIC: Array<{ mimeType: string; matches: (buffer: Buffer) => boolean }> = [
  { mimeType: "image/png", matches: (b) => b.subarray(0, 8).equals(PNG_MAGIC) },
  { mimeType: "image/jpeg", matches: (b) => b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff },
  { mimeType: "image/gif", matches: (b) => b.subarray(0, 6).toString("ascii").match(/^GIF8[79]a$/) !== null },
  { mimeType: "image/webp", matches: (b) => b.subarray(0, 4).toString("ascii") === "RIFF" && b.subarray(8, 12).toString("ascii") === "WEBP" },
  { mimeType: "image/bmp", matches: (b) => b[0] === 0x42 && b[1] === 0x4d },
];
const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

export function detectImageMimeType(buffer: Buffer): string | undefined {
  return IMAGE_MAGIC.find((candidate) => candidate.matches(buffer))?.mimeType;
}

export interface SandboxFileOperations {
  read: ReadOperations;
  write: WriteOperations;
  edit: EditOperations;
}

/**
 * The file surface both transports implement.
 *
 * `SandboxFileClient` reaches the file API over HTTP; the terminal transport
 * reaches the same sandbox over the cluster-direct WebSocket. Pi's file tools
 * only need these three calls.
 */
export interface SandboxFileClientLike {
  read(path: string, limit?: number): Promise<Buffer>;
  access(path: string): Promise<void>;
  write(path: string, content: string | Buffer, signal?: AbortSignal): Promise<void>;
}

/**
 * Pi file operations backed by a sandbox file transport.
 *
 * The read tool asks for the MIME type before reading the bytes, so the probe
 * keeps the last fetched buffer and lets `readFile` reuse it: one transport
 * round trip per read instead of two.
 */
export function createSandboxFileOperations(
  client: SandboxFileClientLike,
  mirror?: SandboxReadMirror,
): SandboxFileOperations {
  let cached: { path: string; buffer: Buffer } | undefined;
  const readBytes = async (path: string): Promise<Buffer> => {
    const host = mirror?.resolveHost(path);
    if (host !== undefined) {
      const buffer = await readMirroredFile(host).catch((error: unknown) => {
        if (isMissingFile(error)) return undefined;
        throw error;
      });
      if (buffer !== undefined) return buffer;
      // The sandbox copies are uploaded from these roots, so a host miss means
      // the resource is absent. Handing the host path to the file API would
      // only ask the sandbox to read a path that exists on neither side.
      throw missingResource(mirror?.resolveHostAlias(host) ?? host);
    }
    return client.read(path);
  };
  const read = async (path: string): Promise<Buffer> => {
    if (cached?.path === path) {
      const buffer = cached.buffer;
      cached = undefined;
      return buffer;
    }
    return readBytes(path);
  };
  // The read and edit tools both probe before they read, so priming the cache
  // lets the read that follows reuse the probe's bytes instead of paying a
  // second transport round trip for the same file.
  const primeRead = async (path: string): Promise<void> => {
    cached = { path, buffer: await readBytes(path) };
  };
  return {
    read: {
      readFile: read,
      access: primeRead,
      detectImageMimeType: async (path) => {
        const buffer = cached?.path === path ? cached.buffer : await readBytes(path);
        cached = { path, buffer };
        return detectImageMimeType(buffer);
      },
    },
    write: {
      writeFile: (path, content) => client.write(path, content),
      // Both transports create parent directories for a write, so the explicit
      // mkdir the write tool issues before writing is already covered.
      mkdir: async () => undefined,
    },
    edit: {
      readFile: read,
      writeFile: (path, content) => client.write(path, content),
      access: primeRead,
    },
  };
}

/** Read a file the runner already holds, under the file API's size limit. */
async function readMirroredFile(path: string): Promise<Buffer> {
  const buffer = await readFile(path);
  if (buffer.byteLength > SANDBOX_FILE_MAX_BYTES) {
    throw new Error("file exceeds size limit");
  }
  return buffer;
}

function isMissingFile(error: unknown): boolean {
  const code =
    error instanceof Error && "code" in error
      ? (error as { code?: unknown }).code
      : undefined;
  return code === "ENOENT" || code === "ENOTDIR";
}

function missingResource(alias: string): Error {
  return new Error(`File not found: ${alias}`);
}
