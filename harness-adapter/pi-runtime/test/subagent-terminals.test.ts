import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { SubagentTerminals } from "../src/subagent-terminals.js";
import type { SandboxToolLane } from "../src/tools.js";

test("child terminal leases are conversation-scoped and a resumed session waits for cleanup", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "agent-terminal-lease-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "sessions"));
  const file = join(root, "sessions", "a.jsonl");
  const otherFile = join(root, "other.jsonl");
  const header = JSON.stringify({ type: "session", id: "child-a" }) + "\n";
  await Promise.all([writeFile(file, header), writeFile(otherFile, header)]);
  const release = Promise.withResolvers<void>();
  const draining = Promise.withResolvers<void>();
  const tool = { name: "terminal" } as AgentTool;
  const owners: string[] = [];
  const terminals = new SubagentTerminals(root, {
    createLane(owner) {
      owners.push(owner);
      return { terminal: {
        async disconnect() { draining.resolve(); await release.promise; },
      } } as SandboxToolLane;
    },
    tools: () => [tool],
    close: async () => {},
  });
  const first = new AbortController();
  const resumed = new AbortController();
  const firstReady = Promise.withResolvers<string>();
  const resumedReady = Promise.withResolvers<string>();
  await assert.rejects(terminals.attach("general_purpose", "child-a", otherFile, first.signal, firstReady.resolve), /another conversation/);
  await assert.rejects(terminals.attach("general_purpose", "bad-id", file, first.signal, firstReady.resolve), /identity mismatch/);
  const firstLease = terminals.attach("general_purpose", "child-a", file, first.signal, firstReady.resolve);
  const binding = { role: "general_purpose", session: "child-a", session_file: file, name: "terminal", terminal_lease: await firstReady.promise };
  const started = Promise.withResolvers<void>();
  const cancelled = Promise.withResolvers<void>();
  const call = terminals.run(binding, new AbortController().signal, async (selected, signal) => {
    assert.equal(selected, tool);
    started.resolve();
    signal.addEventListener("abort", () => cancelled.resolve(), { once: true });
    await cancelled.promise;
  });
  await started.promise;
  first.abort();
  await call;
  await draining.promise;
  let opened = false;
  const nextLease = terminals.attach("general_purpose", "child-a", file, resumed.signal, (id) => {
    opened = true;
    resumedReady.resolve(id);
  });
  assert.equal(opened, false);
  await assert.rejects(terminals.run(binding, resumed.signal, async () => assert.fail("stale lease")), /expired/);
  release.resolve();
  await firstLease;
  const resumedId = await resumedReady.promise;
  assert.notEqual(resumedId, binding.terminal_lease);
  assert.equal(owners[0], owners[1], "native resume must retain terminal ownership");
  const resumedBinding = { ...binding, terminal_lease: resumedId };
  await terminals.detach(resumedBinding);
  await assert.rejects(terminals.run(resumedBinding, resumed.signal, async () => assert.fail("detached lease")), /expired/);
  resumed.abort();
  await nextLease;
  await terminals.close();
});
