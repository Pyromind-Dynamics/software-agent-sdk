import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import {
  createSandboxFileOperations,
  FallbackSandboxFileClient,
  SandboxEndpointSession,
  SandboxFileClient,
  SandboxFileError,
  type SandboxEndpoint,
  type SandboxEndpointRequest,
  type SandboxFileClientLike,
} from "../src/sandbox-operations.js";
import { SandboxReadMirror } from "../src/sandbox-paths.js";

interface MockApi {
  baseUrl: string;
  requests: string[];
  close: () => Promise<void>;
}

async function listen(
  handler: (request: IncomingMessage, response: ServerResponse) => void,
): Promise<MockApi> {
  const requests: string[] = [];
  const server = createServer((request, response) => {
    requests.push(`${request.method} ${request.url}`);
    handler(request, response);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    requests,
    close: () =>
      new Promise<void>((resolve, reject) => {
        // A request which the client abandoned mid-flight leaves its socket
        // open, and `close` alone waits for it.
        server.closeAllConnections();
        server.close((error) => (error ? reject(error) : resolve()));
      }),
  };
}

/** A port nothing listens on, so every request fails at the transport layer. */
async function deadBaseUrl(): Promise<string> {
  const server = createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  await new Promise<void>((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
  return `http://127.0.0.1:${port}`;
}

function endpointFor(baseUrl: string): SandboxEndpoint {
  return {
    baseUrl,
    wsBaseUrl: baseUrl,
    sandboxId: "sbx-1",
    apiKey: "secret",
    workspacePath: "/target-workspace/.pyromind-agent/conv-1",
    storagePath: "/target-workspace",
  };
}

function chunkedUpload(_request: IncomingMessage, response: ServerResponse): void {
  response.writeHead(200, { "Content-Type": "application/json" });
  response.end(JSON.stringify({ data: { upload_id: "upload-1" } }));
}

async function writeFailure(client: SandboxFileClient): Promise<unknown> {
  return client.write("public_data/out.txt", "hello").then(
    () => undefined,
    (reason: unknown) => reason,
  );
}

test("sandbox endpoint session reuses one bundle until it is invalidated", async () => {
  const requests: SandboxEndpointRequest[] = [];
  const session = new SandboxEndpointSession(async (request) => {
    requests.push(request);
    return endpointFor("http://127.0.0.1:1");
  });

  await session.resolve();
  await session.resolve();
  assert.deepEqual(requests, [{ refresh: false }]);

  session.invalidate();
  await session.resolve();
  assert.deepEqual(requests, [{ refresh: false }, { refresh: true }]);
});

test("sandbox file write surfaces the transport cause of a failed refresh", async () => {
  const session = new SandboxEndpointSession(async () =>
    endpointFor(await deadBaseUrl()),
  );

  const error = await writeFailure(new SandboxFileClient(session));

  assert.ok(error instanceof SandboxFileError);
  assert.equal(error.status, undefined);
  assert.match(error.message, /POST \/sandboxes\/sbx-1\/files\/chunks\/init/);
  assert.match(error.message, /ECONNREFUSED/);
  assert.match(error.message, /retried with a refreshed sandbox endpoint/);
});

test("sandbox file write retries once against the refreshed endpoint", async (t) => {
  const api = await listen(chunkedUpload);
  t.after(api.close);
  const dead = await deadBaseUrl();
  const requests: SandboxEndpointRequest[] = [];
  const session = new SandboxEndpointSession(async (request) => {
    requests.push(request);
    return endpointFor(requests.length === 1 ? dead : api.baseUrl);
  });

  await new SandboxFileClient(session).write("public_data/out.txt", "hello");

  assert.deepEqual(requests, [{ refresh: false }, { refresh: true }]);
  assert.deepEqual(api.requests, [
    "POST /sandboxes/sbx-1/files/chunks/init?total_size=5&total_chunks=1",
    "PUT /sandboxes/sbx-1/files/chunks/upload-1/part?chunk_index=0",
    "POST /sandboxes/sbx-1/files/chunks/upload-1/complete?path=public_data%2Fout.txt&total_chunks=1",
  ]);
});

test("sandbox file write reports an API error without refreshing the endpoint", async (t) => {
  const api = await listen((_request, response) => {
    response.writeHead(401, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ detail: "Invalid API key" }));
  });
  t.after(api.close);
  const requests: SandboxEndpointRequest[] = [];
  const session = new SandboxEndpointSession(async (request) => {
    requests.push(request);
    return endpointFor(api.baseUrl);
  });

  const error = await writeFailure(new SandboxFileClient(session));

  assert.ok(error instanceof SandboxFileError);
  assert.equal(error.status, 401);
  assert.match(error.message, /401 \{"detail":"Invalid API key"\}/);
  assert.deepEqual(requests, [{ refresh: false }]);
  assert.equal(api.requests.length, 1);
});

test("sandbox file write stops on abort without refreshing the endpoint", async (t) => {
  let initStarted = (): void => {};
  const started = new Promise<void>((resolve) => {
    initStarted = resolve;
  });
  const api = await listen(() => initStarted());
  t.after(api.close);
  const requests: SandboxEndpointRequest[] = [];
  const session = new SandboxEndpointSession(async (request) => {
    requests.push(request);
    return endpointFor(api.baseUrl);
  });
  const controller = new AbortController();
  const pending = new SandboxFileClient(session).write(
    "public_data/out.txt",
    "hello",
    controller.signal,
  );

  await started;
  controller.abort();

  await assert.rejects(pending, /aborted/i);
  assert.deepEqual(requests, [{ refresh: false }]);
  assert.deepEqual(api.requests, [
    "POST /sandboxes/sbx-1/files/chunks/init?total_size=5&total_chunks=1",
  ]);
});

test("sandbox file reads answer mirrored resources without touching the sandbox", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-mirror-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const skill = join(root, "data-processing", "SKILL.md");
  await mkdir(dirname(skill), { recursive: true });
  await writeFile(skill, "mirrored\n");
  const requests: SandboxEndpointRequest[] = [];
  const session = new SandboxEndpointSession(async (request) => {
    requests.push(request);
    throw new Error("the sandbox must not be contacted for a mirrored read");
  });
  const mirror = new SandboxReadMirror([{ host: root, alias: ".agents/skills" }]);
  const operations = createSandboxFileOperations(
    new SandboxFileClient(session),
    mirror,
  );

  const host = mirror.resolveAlias(".agents/skills/data-processing/SKILL.md");
  assert.ok(host !== undefined);
  assert.equal(host, skill);
  // The read tool probes before it reads, so every stage must stay local.
  await operations.read.access(host);
  assert.equal(await operations.read.detectImageMimeType?.(host), undefined);
  const buffer = await operations.read.readFile(host);

  assert.equal(buffer.toString("utf8"), "mirrored\n");
  assert.deepEqual(requests, []);
});

test("sandbox file reads report a mirrored miss by alias without touching the sandbox", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-mirror-miss-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const requests: SandboxEndpointRequest[] = [];
  const session = new SandboxEndpointSession(async (request) => {
    requests.push(request);
    throw new Error("the sandbox must not be contacted for a mirrored miss");
  });
  const mirror = new SandboxReadMirror([{ host: root, alias: ".agents/skills" }]);
  const operations = createSandboxFileOperations(
    new SandboxFileClient(session),
    mirror,
  );

  await assert.rejects(
    operations.read.readFile(join(root, "missing.md")),
    /File not found: \.agents\/skills\/missing\.md/,
  );
  assert.deepEqual(requests, []);
});

/** Terminal transport stub recording the calls the fallback client makes. */
class RecordingFileClient implements SandboxFileClientLike {
  readonly reads: string[] = [];
  readonly writes: string[] = [];
  readonly accesses: string[] = [];
  constructor(private readonly content = "from terminal") {}

  async read(path: string): Promise<Buffer> {
    this.reads.push(path);
    return Buffer.from(this.content);
  }

  async access(path: string): Promise<void> {
    this.accesses.push(path);
  }

  async write(path: string, content: string | Buffer): Promise<void> {
    this.writes.push(`${path}:${content.toString()}`);
  }
}

test("file tools use the file API when the portal answers", async (t) => {
  const api = await listen((request, response) => {
    if (request.url?.includes("/files/read")) {
      response.writeHead(200, { "Content-Type": "application/octet-stream" });
      response.end("from api");
      return;
    }
    chunkedUpload(request, response);
  });
  t.after(api.close);
  const session = new SandboxEndpointSession(async () => endpointFor(api.baseUrl));
  const terminal = new RecordingFileClient();
  const client = new FallbackSandboxFileClient(new SandboxFileClient(session), terminal);

  assert.equal((await client.read("public_data/a.txt")).toString("utf8"), "from api");
  await client.access("public_data/a.txt");
  await client.write("public_data/a.txt", "hello");

  assert.deepEqual(terminal.reads, []);
  assert.deepEqual(terminal.accesses, []);
  assert.deepEqual(terminal.writes, []);
  assert.equal(api.requests.length, 5, "two reads and one three-request upload");
});

test("an unreachable file API switches the session to the terminal transport", async () => {
  const resolves: SandboxEndpointRequest[] = [];
  const session = new SandboxEndpointSession(async (request) => {
    resolves.push(request);
    return endpointFor(await deadBaseUrl());
  });
  const terminal = new RecordingFileClient();
  const client = new FallbackSandboxFileClient(new SandboxFileClient(session), terminal);

  assert.equal((await client.read("public_data/a.txt")).toString("utf8"), "from terminal");
  assert.equal((await client.read("public_data/b.txt")).toString("utf8"), "from terminal");

  assert.deepEqual(terminal.reads, ["public_data/a.txt", "public_data/b.txt"]);
  // The portal is only probed once; every later call goes straight to the
  // terminal transport instead of paying the file API timeout again.
  assert.deepEqual(resolves, [{ refresh: false }, { refresh: true }]);
});

test("an unresponsive file API is bounded by the probe budget", async (t) => {
  // The portal accepts the connection but never answers; the first request is
  // what has to learn that, so it runs on the short probe budget instead of
  // the full request budget.
  const api = await listen(() => undefined);
  t.after(api.close);
  const session = new SandboxEndpointSession(async () => endpointFor(api.baseUrl));
  const terminal = new RecordingFileClient();
  const client = new FallbackSandboxFileClient(
    new SandboxFileClient(session, { probeTimeoutMs: 50 }),
    terminal,
  );

  assert.equal((await client.read("public_data/a.txt")).toString("utf8"), "from terminal");
  assert.deepEqual(terminal.reads, ["public_data/a.txt"]);
});

test("a file API error the sandbox returned is not answered over the terminal", async (t) => {
  const api = await listen((_request, response) => {
    response.writeHead(404, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ detail: "missing" }));
  });
  t.after(api.close);
  const session = new SandboxEndpointSession(async () => endpointFor(api.baseUrl));
  const terminal = new RecordingFileClient();
  const client = new FallbackSandboxFileClient(new SandboxFileClient(session), terminal);

  await assert.rejects(client.read("public_data/missing.txt"), /404/);
  assert.deepEqual(terminal.reads, []);
});
