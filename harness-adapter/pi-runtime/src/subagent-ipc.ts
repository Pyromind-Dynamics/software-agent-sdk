import { createConnection, createServer, type Socket } from "node:net";
import { createInterface } from "node:readline";
import { chmod, unlink } from "node:fs/promises";

export type IpcMessage = { type: "update" | "result" | "error"; value?: any; error?: string };
const MAX_BYTES = 16 * 1024 * 1024;

/** Conversation-private transport. No credentials or launch state are stored here. */
export async function serveSubagentIpc(path: string, handle: (request: any, signal: AbortSignal, update: (value: any) => void) => Promise<any>) {
  const sockets = new Set<Socket>();
  const server = createServer((socket) => {
    sockets.add(socket);
    const abort = new AbortController();
    socket.on("close", () => { sockets.delete(socket); abort.abort(); });
    socket.on("error", () => abort.abort());
    const send = (message: IpcMessage) => {
      const line = JSON.stringify(message);
      if (Buffer.byteLength(line) > MAX_BYTES) throw new Error("Subagent IPC result exceeds limit");
      if (!socket.destroyed) socket.write(line + "\n");
    };
    let bytes = 0;
    socket.on("data", (chunk) => { bytes += chunk.length; if (bytes > MAX_BYTES) socket.destroy(); });
    const lines = createInterface({ input: socket });
    lines.on("error", () => socket.destroy());
    lines.once("line", (line) => {
      void (async () => {
        try { send({ type: "result", value: await handle(JSON.parse(line), abort.signal, (value) => send({ type: "update", value })) }); }
        catch (error) { send({ type: "error", error: error instanceof Error ? error.message : "Subagent host failed" }); }
        finally { socket.end(); lines.close(); }
      })().catch(() => socket.destroy());
    });
  });
  // Refuse a live owner; only remove a socket left by a dead runner.
  await new Promise<void>((resolve, reject) => {
    const probe = createConnection(path);
    probe.once("connect", () => { probe.destroy(); reject(new Error("Subagent execution host is already active")); });
    probe.once("error", (error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT" || error.code === "ECONNREFUSED") resolve(); else reject(error);
    });
  });
  await unlink(path).catch((error: NodeJS.ErrnoException) => { if (error.code !== "ENOENT") throw error; });
  await new Promise<void>((resolve, reject) => { server.once("error", reject); server.listen(path, resolve); });
  await chmod(path, 0o600);
  server.unref();
  return async () => {
    for (const socket of sockets) socket.destroy();
    await new Promise<void>((resolve) => server.close(() => resolve()));
    await unlink(path).catch(() => {});
  };
}

export function callSubagentHost(path: string, request: any, signal?: AbortSignal, update?: (value: any) => void): Promise<any> {
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    const socket = createConnection(path);
    let settled = false;
    const finish = (error?: Error, result?: any) => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener("abort", onAbort);
      socket.destroy();
      if (error) reject(error); else resolve(result);
    };
    const onAbort = () => finish(new Error("Subagent request cancelled"));
    signal?.addEventListener("abort", onAbort, { once: true });
    socket.once("connect", () => {
      const line = JSON.stringify(request);
      if (Buffer.byteLength(line) > MAX_BYTES) return finish(new Error("Subagent IPC request exceeds limit"));
      socket.write(line + "\n");
    });
    socket.once("error", (error) => finish(new Error(`Subagent host unavailable: ${error.message}`)));
    socket.once("close", () => finish(new Error("Subagent host disconnected")));
    let pendingBytes = 0;
    socket.on("data", (data) => { pendingBytes += data.length; if (pendingBytes > MAX_BYTES) finish(new Error("Subagent IPC frame exceeds limit")); });
    const lines = createInterface({ input: socket });
    lines.on("error", (error) => finish(error));
    lines.on("line", (line) => {
      pendingBytes = 0;
      try {
        const message = JSON.parse(line) as IpcMessage;
        if (message.type === "error") finish(new Error(message.error));
        else if (message.type === "result") finish(undefined, message.value);
        else if (message.type === "update") update?.(message.value);
        else finish(new Error("Invalid subagent IPC response"));
      } catch (error) { finish(error instanceof Error ? error : new Error("Invalid subagent IPC response")); }
    });
  });
}
