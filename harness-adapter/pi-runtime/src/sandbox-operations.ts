import type {
  EditOperations,
  ReadOperations,
  WriteOperations,
} from "@earendil-works/pi-coding-agent";

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
   * Sandbox-side Storage mount root, addressed by the model as `storage/`.
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

/** Thin HTTP client for the sandbox file API (read / chunked write / delete). */
export class SandboxFileClient {
  constructor(private readonly endpoints: SandboxEndpointSession) {}

  async read(path: string): Promise<Buffer> {
    return this.withEndpoint(async (endpoint) => {
      const url = this.url(endpoint, `/sandboxes/${endpoint.sandboxId}/files/read`);
      url.searchParams.set("path", path);
      return this.bytes("GET", url, await this.fetch(endpoint, url));
    });
  }

  async access(path: string): Promise<void> {
    await this.read(path);
  }

  async write(path: string, content: string | Buffer): Promise<void> {
    const body = typeof content === "string" ? Buffer.from(content, "utf8") : content;
    await this.withEndpoint((endpoint) => this.upload(endpoint, path, body));
  }

  private async upload(
    endpoint: SandboxEndpoint,
    path: string,
    body: Buffer,
  ): Promise<void> {
    const base = this.url(endpoint, `/sandboxes/${endpoint.sandboxId}/files/chunks`);
    const totalChunks = Math.max(1, Math.ceil(body.byteLength / UPLOAD_CHUNK_BYTES));
    const init = new URL(`${base}/init`);
    init.searchParams.set("total_size", String(body.byteLength));
    init.searchParams.set("total_chunks", String(totalChunks));
    const initialized = await this.json(
      await this.fetch(endpoint, init, { method: "POST" }),
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
          // The body length is left to fetch: the runner imports the Pi
          // packages, whose undici installs a dispatcher that rejects an
          // explicitly set Content-Length on a byte-array body.
          headers: { "Content-Type": "application/octet-stream" },
        });
      }
      const complete = new URL(`${base}/${encodeURIComponent(uploadId)}/complete`);
      complete.searchParams.set("path", path);
      complete.searchParams.set("total_chunks", String(totalChunks));
      await this.fetch(endpoint, complete, { method: "POST" });
    } catch (error) {
      await this.fetch(endpoint, new URL(`${base}/${encodeURIComponent(uploadId)}`), {
        method: "DELETE",
      }).catch(() => undefined);
      throw error;
    }
  }

  /**
   * Run one call against the current endpoint bundle.
   *
   * A transport failure is the one error the control plane can repair: the
   * sandbox may have been rebuilt or re-keyed since the bundle was fetched, so
   * the call asks for the bundle again and retries once before reporting.
   */
  private async withEndpoint<T>(
    operation: (endpoint: SandboxEndpoint) => Promise<T>,
  ): Promise<T> {
    const endpoint = await this.endpoints.resolve();
    try {
      return await operation(endpoint);
    } catch (error) {
      if (!(error instanceof SandboxFileError) || error.status !== undefined) throw error;
      this.endpoints.invalidate();
      try {
        return await operation(await this.endpoints.resolve());
      } catch (retryError) {
        throw new SandboxFileError(
          `${describeError(retryError)} (retried with a refreshed sandbox endpoint)`,
          retryError instanceof SandboxFileError ? retryError.status : undefined,
          { cause: retryError },
        );
      }
    }
  }

  private async bytes(method: string, url: URL, response: Response): Promise<Buffer> {
    try {
      return Buffer.from(await response.arrayBuffer());
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
      response = await fetch(url, {
        ...init,
        headers,
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
    } catch (error) {
      throw transportError(init.method ?? "GET", url, error);
    }
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

function describeErrorChain(error: unknown): string {
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
 * Pi file operations backed by the sandbox file API.
 *
 * The read tool asks for the MIME type before reading the bytes, so the probe
 * keeps the last fetched buffer and lets `readFile` reuse it: one HTTP round
 * trip per read instead of two.
 */
export function createSandboxFileOperations(client: SandboxFileClient): SandboxFileOperations {
  let cached: { path: string; buffer: Buffer } | undefined;
  const read = async (path: string): Promise<Buffer> => {
    if (cached?.path === path) {
      const buffer = cached.buffer;
      cached = undefined;
      return buffer;
    }
    return client.read(path);
  };
  return {
    read: {
      readFile: read,
      access: async (path) => {
        cached = { path, buffer: await client.read(path) };
      },
      detectImageMimeType: async (path) => {
        const buffer = cached?.path === path ? cached.buffer : await client.read(path);
        cached = { path, buffer };
        return detectImageMimeType(buffer);
      },
    },
    write: {
      writeFile: (path, content) => client.write(path, content),
      // The file API creates parent directories for every upload, so the
      // explicit mkdir the write tool issues before writing is already covered.
      mkdir: async () => undefined,
    },
    edit: {
      readFile: read,
      writeFile: (path, content) => client.write(path, content),
      access: (path) => client.access(path),
    },
  };
}
