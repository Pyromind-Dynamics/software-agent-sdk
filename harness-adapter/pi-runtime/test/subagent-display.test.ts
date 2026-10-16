import assert from "node:assert/strict";
import test from "node:test";
import { SubagentDisplay, transcriptWindow, displayText } from "../src/subagent-display.js";

test("native transcript windows remove metadata and bound UTF-8 without losing truncation notice", () => {
  const window = transcriptWindow('Run: abc\nSession: /private/session\nArtifacts:\n  Events: /private/events\nTranscript tail from /private/log (tail truncated):\nTool: read\n证据');
  assert.doesNotMatch(window.text, /private/);
  assert.match(window.text, /Tool: read/);
  assert.equal(window.truncated, true);
  const long = transcriptWindow('Live transcript tail:\n' + '证据'.repeat(20000));
  assert.ok(Buffer.byteLength(long.text) <= 32768);
  assert.ok(!long.text.includes('�'));
  assert.equal(long.truncated, true);
  assert.equal(transcriptWindow('Result transcript tail:\nfinal').outputOnly, true);
  assert.equal(transcriptWindow('Unknown format /private/session').availability, 'unavailable');
  assert.equal(displayText({ content: [{ type: 'text', text: 'handoff' }], details: { secret: 'x' } }), 'handoff');
});

test("polling is single flight, rate limited, deduplicated and preserves last success on failure/completion", async t => {
  let now = 10000;
  t.mock.method(Date, 'now', () => now);
  let resolve!: (value: unknown) => void;
  let calls = 0;
  const progress: string[] = [];
  const display = new SubagentDisplay(async (method, params) => {
    assert.equal(method, 'status'); assert.deepEqual(params, { id: 'run', view: 'transcript', lines: 200 });
    calls++; return new Promise(done => { resolve = done; });
  }, id => progress.push(id));
  const first = display.poll('run');
  await display.poll('run'); assert.equal(calls, 1);
  resolve({ text: 'Live transcript tail:\nTool: read' }); await first;
  await display.poll('run'); assert.equal(calls, 1);
  now += 2000;
  const second = display.poll('run'); resolve({ text: 'Live transcript tail:\nTool: read' }); await second;
  assert.equal(progress.length, 1);
  now += 2000;
  const third = display.poll('run'); resolve({ text: 'Result transcript tail:\nfinal output' }); await third;
  assert.equal(progress.length, 1);
  now += 2000;
  const fourth = display.poll('run');
  display.update('run', { ended_at: 'now' }); resolve({ text: 'Live transcript tail:\nlate' }); await fourth;
  assert.match(JSON.stringify(display.details('run')), /Tool: read/);
  display.close(); now += 2000; await display.poll('run'); assert.equal(calls, 4);
});

test("query errors preserve content and retry next period", async t => {
  let now = 10000; t.mock.method(Date, 'now', () => now);
  let calls = 0;
  const display = new SubagentDisplay(async () => { if (++calls === 2) throw Error('offline'); return { text: 'Live transcript tail:\nevidence' }; }, () => {});
  await display.poll('run'); const before = display.details('run');
  now += 2000; await display.poll('run'); assert.deepEqual(display.details('run'), before);
  now += 2000; await display.poll('run'); assert.equal(calls, 3);
});
