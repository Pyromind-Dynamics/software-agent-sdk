import assert from "node:assert/strict";
import test from "node:test";
import type { BashOperations } from "@earendil-works/pi-coding-agent";
import { SandboxTerminalFileClient } from "../src/sandbox-terminal-files.js";

interface FakeTerminal {
  operations: BashOperations;
  commands: string[];
}

/**
 * A terminal stub which behaves like a shell for the two commands the file
 * client issues: the `base64 < <path>` read, and the staged `printf` /
 * `openssl base64 -d` write sequence.
 */
function fakeTerminal(files: Map<string, Buffer> = new Map()): FakeTerminal {
  const commands: string[] = [];
  const staging = new Map<string, string>();
  const operations: BashOperations = {
    exec: async (command, _cwd, options) => {
      commands.push(command);
      if (command.startsWith("base64 < ")) {
        const path = unquote(command.slice("base64 < ".length).split(" | ")[0]!);
        const content = files.get(path);
        if (content === undefined) return { exitCode: 1 };
        const encoded = content.toString("base64");
        for (let at = 0; at < encoded.length; at += 512) {
          options.onData(Buffer.from(`${encoded.slice(at, at + 512)}\r\n`));
        }
        return { exitCode: 0 };
      }
      if (command.startsWith("test -r ")) {
        return { exitCode: files.has(unquote(command.slice("test -r ".length))) ? 0 : 1 };
      }
      if (command.startsWith("rm -f ")) {
        staging.delete(unquote(command.slice("rm -f ".length)));
        return { exitCode: 0 };
      }
      // The first batch truncates the staging file; later batches only append.
      const staged = /: > '([^']+)'/.exec(command)?.[1] ??
        /printf '%s' '[^']*' >> '([^']+)'/.exec(command)?.[1];
      if (staged === undefined) return { exitCode: 1 };
      let encoded = staging.get(staged) ?? "";
      for (const part of command.split("\n")) {
        const chunk = /printf '%s' '([^']*)' >>/.exec(part)?.[1];
        if (chunk !== undefined) encoded += chunk;
      }
      staging.set(staged, encoded);
      const target = /openssl base64 -d -A -in '[^']+' -out '([^']+)'/.exec(command)?.[1];
      if (target === undefined) return { exitCode: 0 };
      const content = Buffer.from(encoded, "base64");
      if (!command.includes(`"${content.byteLength}"`)) return { exitCode: 1 };
      files.set(target, content);
      staging.delete(staged);
      return { exitCode: 0 };
    },
  };
  return { operations, commands };
}

test("reads file bytes through the terminal as base64", async () => {
  const content = Buffer.from("print('hello')\n" + "x".repeat(700));
  const terminal = fakeTerminal(new Map([["public_data/script.py", content]]));
  const client = new SandboxTerminalFileClient(terminal.operations);
  assert.deepEqual(await client.read("public_data/script.py"), content);
  assert.deepEqual(terminal.commands, ["base64 < 'public_data/script.py' | tr -d '\\n'"]);
});

test("writes file bytes through one bounded terminal command", async () => {
  const terminal = fakeTerminal();
  const client = new SandboxTerminalFileClient(terminal.operations);
  const content = "#!/usr/bin/env python3\n" + "y".repeat(9000);
  await client.write("public_data/large.py", content);
  assert.equal(terminal.commands.length, 1, "one file write is one terminal command");
  assert.match(terminal.commands[0]!, /mkdir -p 'public_data'/,
    "the write creates the target directory like the file API did");
  for (const command of terminal.commands) {
    assert.ok(command.split("\n").every((line) => Buffer.byteLength(line) <= 1024),
      "every line stays below the smaller PTY line cap");
  }
  assert.deepEqual(await client.read("public_data/large.py"), Buffer.from(content));
});

test("access reports a missing file as a failure", async () => {
  const terminal = fakeTerminal();
  const client = new SandboxTerminalFileClient(terminal.operations);
  await assert.rejects(client.access("public_data/missing.py"), /sandbox terminal access failed/);
});

function unquote(value: string): string {
  return value.replace(/^'|'$/g, "").replace(/'\\''/g, "'");
}
