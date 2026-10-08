import type { SandboxTerminalSocket } from "./sandbox-terminal.js";

/** One bounded request at a time; unrecognised TTY output is never logged. */
export class TerminalChannel {
  private pending: {
    consume: (chunk: Buffer) => void;
    reject: (error: Error) => void;
  } | undefined;
  private closed = false;
  private readonly opened: Promise<void>;

  constructor(private readonly socket: SandboxTerminalSocket) {
    this.opened = new Promise<void>((resolve, reject) => {
      socket.onOpen(resolve);
      socket.onClose(() => {
        this.closed = true;
        reject(new Error("closed"));
        this.pending?.reject(new Error("closed"));
      });
      socket.onError(() => {
        this.closed = true;
        reject(new Error("closed"));
        this.pending?.reject(new Error("closed"));
      });
      socket.onMessage((data) => this.pending?.consume(
        typeof data === "string" ? Buffer.from(data) : Buffer.from(data),
      ));
    });
    // A socket can fail while scripts are still uploading.
    void this.opened.catch(() => undefined);
  }

  async ready(deadline: number, signal?: AbortSignal): Promise<void> {
    await bounded(this.opened, deadline, signal);
    if (this.closed) throw new Error("closed");
  }

  async acknowledge(
    command: string, marker: string, deadline: number, signal?: AbortSignal,
  ): Promise<string> {
    let buffer = "";
    return this.receive(command, deadline, signal, (chunk, resolve) => {
      buffer = (buffer + chunk.toString("utf8")).slice(-65536);
      // The shell command prints prefix and token separately, so echoed input
      // cannot be mistaken for an acknowledgement.
      const at = buffer.indexOf(marker);
      if (at < 0) return;
      const end = buffer.indexOf("\n", at);
      if (end >= 0) resolve(buffer.slice(at + marker.length, end).replace(/\r$/, ""));
    });
  }

  async receive<T>(
    command: string,
    deadline: number,
    signal: AbortSignal | undefined,
    consume: (chunk: Buffer, resolve: (value: T) => void) => void,
  ): Promise<T> {
    await this.ready(deadline, signal);
    if (this.pending) throw new Error("terminal request already active");
    try {
      return await new Promise<T>((resolve, reject) => {
        let settled = false;
        const finish = (error?: Error, value?: T): void => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          signal?.removeEventListener("abort", abort);
          this.pending = undefined;
          if (error) reject(error);
          else resolve(value as T);
        };
        const abort = (): void => finish(new Error("aborted"));
        const timer = setTimeout(() => finish(new Error("timeout")), Math.max(0, deadline - Date.now()));
        this.pending = { consume: (chunk) => consume(chunk, (value) => finish(undefined, value)), reject: (error) => finish(error) };
        signal?.addEventListener("abort", abort, { once: true });
        if (signal?.aborted) { abort(); return; }
        try { this.socket.send(command); }
        catch { finish(new Error("closed")); }
      });
    } finally {
      this.pending = undefined;
    }
  }

  interrupt(): void {
    if (!this.closed) this.socket.send("\u0003");
  }

  close(): void {
    try { this.socket.close(); }
    catch { /* Closing a disconnected TTY is best effort. */ }
  }
}

export function bounded<T>(promise: Promise<T>, deadline: number, signal?: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const finish = (error?: unknown, value?: T): void => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      if (error) reject(error);
      else resolve(value as T);
    };
    const abort = (): void => finish(new Error("aborted"));
    const timer = setTimeout(() => finish(new Error("timeout")), Math.max(0, deadline - Date.now()));
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    promise.then((value) => finish(undefined, value), (error) => finish(error));
  });
}
