import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import test from "node:test";
import {
  SandboxEndpointSession,
  SandboxFileClient,
  SandboxFileError,
  type SandboxEndpoint,
  type SandboxEndpointRequest,
} from "../src/sandbox-operations.js";

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
      new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      ),
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
