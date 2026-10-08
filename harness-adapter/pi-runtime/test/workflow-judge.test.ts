import assert from "node:assert/strict";
import test from "node:test";
import { serveJudge } from "../src/workflow-judge.js";
import type { ExecutionAccess } from "../src/workflow-execution.js";

for (const [answer, valid] of [[JSON.stringify({ passed: true, reason: "ok" }), true], ['{"passed":"true","reason":"bad"}', false], ['not json', false]] as const) {
  test(`judge uses file channel, strict verdict: ${answer}`, async () => {
    const files = new Map<string, Buffer>();
    files.set('/judge/request.json', Buffer.from(JSON.stringify({request_id:'r',criterion:'check',evidence:'report'})));
    let calls = 0;
    const access: ExecutionAccess = {
      cwd: '/session', resolvePath: async (path) => path,
      operations: {exec: async () => { assert.fail('judge must not use terminal'); }},
      readFile: async (path) => files.get(path)!,
      writeFile: async (path, content) => { files.set(path, content); },
    };
    await serveJudge(access, '/judge', async () => { calls++; return answer; }, new AbortController().signal);
    assert.equal(calls, 1);
    assert.ok(files.has('/judge/response.ready'));
    const response = JSON.parse(files.get('/judge/response.json')!.toString());
    assert.equal(response.request_id, 'r');
    assert.equal(Boolean(response.verdict), valid);
    assert.equal(Boolean(response.error), !valid);
  });
}
