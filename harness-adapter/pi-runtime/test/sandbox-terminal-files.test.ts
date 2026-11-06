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
      if (command.startsWith("test -f ") && command.includes("| tr -d")) {
        const path = readPath(command);
        const content = path === undefined ? undefined : files.get(path);
        if (content === undefined) return { exitCode: 1 };
        let selected = content;
        const first = /-v first=(\d+)/.exec(command)?.[1];
        const count = /-v count=(\d+)/.exec(command)?.[1];
        if (first !== undefined && count !== undefined) {
          selected = Buffer.from(
            content.toString("utf8").split("\n")
              .slice(Number(first) - 1, Number(first) - 1 + Number(count))
              .join("\n"),
          );
        }
        const head = /head -c (\d+)/.exec(command)?.[1];
        if (head !== undefined) selected = selected.subarray(0, Number(head));
        const encoded = selected.toString("base64");
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
  assert.deepEqual(terminal.commands, [
    "test -f 'public_data/script.py' && base64 < 'public_data/script.py' | tr -d '\\n'",
  ]);
});

test("bounded terminal reads stop the byte stream before base64", async () => {
  const content = Buffer.from("abcdef".repeat(1000));
  const terminal = fakeTerminal(new Map([["public_data/large.jsonl", content]]));
  const client = new SandboxTerminalFileClient(terminal.operations);

  assert.equal(
    (await client.readHead("public_data/large.jsonl", 12)).toString(),
    "abcdefabcdef",
  );
  assert.match(terminal.commands[0]!, /head -c 12 'public_data\/large\.jsonl'/);
  assert.doesNotMatch(terminal.commands[0]!, /base64 < 'public_data/);
});

test("line-window reads select lines in the sandbox and cap returned bytes", async () => {
  const content = Buffer.from(["first", "x".repeat(80), "third", "fourth"].join("\n"));
  const terminal = fakeTerminal(new Map([["public_data/large.jsonl", content]]));
  const client = new SandboxTerminalFileClient(terminal.operations);

  const window = await client.readWindow("public_data/large.jsonl", 2, 1, 24);

  assert.equal(window.truncated, true);
  assert.equal(window.content.toString(), "x".repeat(24));
  assert.match(terminal.commands[0]!, /-v first=2 -v count=1/);
  assert.match(terminal.commands[0]!, /'public_data\/large\.jsonl' \| head -c 25 \|/);
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

test("a larger chunk carries the same bytes in fewer script lines", async () => {
  const content = "#!/usr/bin/env python3\n" + "y".repeat(9000);
  const small = fakeTerminal();
  await new SandboxTerminalFileClient(small.operations).write("public_data/large.py", content);
  const large = fakeTerminal();
  await new SandboxTerminalFileClient(large.operations, { chunkChars: 3072 })
    .write("public_data/large.py", content);
  const scriptLines = (commands: string[]): number => commands[0]!.split("\n").length;
  assert.ok(scriptLines(large.commands) < scriptLines(small.commands));
  assert.ok(large.commands.every((command) => command.split("\n")
    .every((line) => Buffer.byteLength(line) < 4096)));
  assert.deepEqual(await new SandboxTerminalFileClient(large.operations).read("public_data/large.py"),
    Buffer.from(content));
});

function unquote(value: string): string {
  return value.replace(/^'|'$/g, "").replace(/'\\''/g, "'");
}

function readPath(command: string): string | undefined {
  return /base64 < '([^']+)'/.exec(command)?.[1]
    ?? /head -c \d+ '([^']+)'/.exec(command)?.[1]
    ?? /exit }' '([^']+)'/.exec(command)?.[1];
}
