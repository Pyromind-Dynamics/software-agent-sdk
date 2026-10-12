import { randomBytes } from "node:crypto";
import { posix } from "node:path";

import type { BashOperations } from "@earendil-works/pi-coding-agent";

import type { SandboxFileClientLike } from "./sandbox-operations.js";

/** Directory holding the transient base64 staging file of a terminal write. */
const STAGING_DIRNAME = ".pyromind-agent-staging";
/** Base64 characters one staging line carries, below the smaller PTY line cap. */
const WRITE_CHUNK_CHARS = 512;

/**
 * Sandbox file transport over the terminal WebSocket.
 *
 * The file API is reached through a portal domain which is neither
 * cluster-direct nor reliable from every runner, while the terminal socket is
 * already open and cluster-direct. File bytes therefore travel the same channel
 * as commands: reads stream `base64` output back, writes stage base64 lines
 * into a transient file and decode it.
 *
 * A write is one command whose script carries the whole payload, so the
 * terminal's own staging decides how the bytes cross the socket. Staging the
 * command is the same acknowledged-burst transport every command uses, which
 * keeps one connection and one watched run per file instead of one per chunk.
 */
export class SandboxTerminalFileClient implements SandboxFileClientLike {
  constructor(private readonly terminal: BashOperations) {}

  async read(path: string, limit = Number.MAX_SAFE_INTEGER): Promise<Buffer> {
    const chunks: Buffer[] = [];
    let total = 0;
    const result = await this.terminal.exec(
      // BSD and GNU base64 both read stdin; only GNU accepts a path argument.
      `base64 < ${shellQuote(path)} | tr -d '\\n'`,
      ".",
      {
        onData: (data) => {
          total += data.byteLength;
          if (total <= limit) chunks.push(data);
        },
      },
    );
    if (result.exitCode !== 0) {
      throw new Error(`sandbox terminal read failed (${result.exitCode ?? "signal"}): ${path}`);
    }
    if (total > limit) throw new Error("file exceeds size limit");
    return Buffer.from(Buffer.concat(chunks).toString("ascii").replace(/[\r\n]/g, ""), "base64");
  }

  async access(path: string): Promise<void> {
    const result = await this.terminal.exec(
      `test -r ${shellQuote(path)}`,
      ".",
      { onData: () => undefined },
    );
    if (result.exitCode !== 0) {
      throw new Error(`sandbox terminal access failed (${result.exitCode ?? "signal"}): ${path}`);
    }
  }

  async write(path: string, content: string | Buffer, signal?: AbortSignal): Promise<void> {
    const body = typeof content === "string" ? Buffer.from(content, "utf8") : content;
    const encoded = body.toString("base64");
    const staging = `${STAGING_DIRNAME}/${randomStagingName()}`;
    const quotedStaging = shellQuote(staging);
    const quotedPath = shellQuote(path);
    const lines = [
      // The file API created parent directories for every upload; the
      // terminal transport has to do it itself.
      `mkdir -p ${shellQuote(STAGING_DIRNAME)}; ` +
      `mkdir -p ${shellQuote(posix.dirname(path))}; : > ${quotedStaging}`,
      ...stagingLines(encoded, staging),
      // `-A` accepts one continuous line; BSD base64 needs `-D`, so openssl is
      // the portable decoder for both sandbox families.
      `openssl base64 -d -A -in ${quotedStaging} -out ${quotedPath} && ` +
      `[ "$(wc -c < ${quotedPath} | tr -d ' ')" = "${body.byteLength}" ] && ` +
      `rm -f ${quotedStaging}`,
    ];
    try {
      const result = await this.terminal.exec(lines.join("\n"), ".", {
        onData: () => undefined,
        signal,
      });
      if (result.exitCode !== 0) {
        throw new Error(
          `sandbox terminal write failed (${result.exitCode ?? "signal"}): ${path}`,
        );
      }
    } catch (error) {
      await this.terminal
        .exec(`rm -f ${quotedStaging}`, ".", { onData: () => undefined })
        .catch(() => undefined);
      throw error;
    }
  }
}

/**
 * Splits the encoded payload into staging lines.
 *
 * A line stays below the PTY line cap; the terminal's staging groups the lines
 * into acknowledged bursts, so the payload never has to be split into several
 * commands.
 */
function stagingLines(encoded: string, staging: string): string[] {
  const quotedStaging = shellQuote(staging);
  const lines: string[] = [];
  for (let at = 0; at < encoded.length; at += WRITE_CHUNK_CHARS) {
    lines.push(
      `printf '%s' ${shellQuote(encoded.slice(at, at + WRITE_CHUNK_CHARS))} >> ${quotedStaging}`,
    );
  }
  return lines;
}

function randomStagingName(): string {
  return `payload-${randomBytes(6).toString("hex")}.b64`;
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}
