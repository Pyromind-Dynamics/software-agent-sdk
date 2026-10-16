import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtemp, mkdir, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import test from "node:test";
import { SubagentReports } from "../src/subagent-reports.js";
import { createTools } from "../src/tools.js";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { ToolCoordinator } from "../src/tool-coordinator.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";

async function fixture(t: test.TestContext) {
  const root = await realpath(await mkdtemp('/tmp/pi-reports-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'pi', 'subagent-artifacts'), { recursive: true });
  await mkdir(join(root, 'pi', 'terminal-output'), { recursive: true });
  await mkdir(join(root, 'public_data'));
  const peer = { request: async () => ({allow:true}) } as unknown as JsonlRpcPeer;
  const tools = await createTools(peer, new NodeExecutionEnv({cwd:root}), root, 'os-sandbox', [], undefined, undefined, []);
  const nativeWriter = new ToolCoordinator().wrap(tools.find(tool => tool.name === 'write')!, peer);
  let writes = 0; let fail = false;
  const writer = {...nativeWriter, execute: async (...args: Parameters<typeof nativeWriter.execute>) => {
    writes++; if (fail) throw new Error('sandbox unavailable');
    return nativeWriter.execute(...args);
  }};
  const parent = randomUUID();
  const reports = new SubagentReports(root, () => parent, writer);
  const record = async (id = randomUUID(), owner = parent) => {
    const source = join(root, 'pi', 'subagent-artifacts', `${id}_search_0_output.md`);
    await writeFile(source, '证据\n'.repeat(1200) + '\nFINAL_MARKER');
    await writeFile(source.replace('_output.md', '_meta.json'), JSON.stringify({runId:id,agent:'search',parentSessionId:owner}));
    const content = [{type:'text',text:`Summary\n…\nFull output: ${source} (large). Read it if needed.`}];
    return {id,source,content,details:{runId:id,results:[{artifactPaths:{outputPath:source}}]}};
  };
  return {root, reports, record, writer, parent, read:tools.find(tool=>tool.name==='read')!, writes:()=>writes, fail:(value:boolean)=>{fail=value;}};
}

test('reports use the existing coordinated write and read, deduplicate, paginate and recover', async t => {
  const f = await fixture(t); const r = await f.record();
  const [a,b] = await Promise.all([f.reports.transform(r.content,r.details),f.reports.transform(r.content,r.details)]);
  const target = `public_data/subagents/${r.id}/report.md`;
  assert.equal(f.writes(),1); assert.deepEqual(a,b);
  assert.match(a.content[0].text,new RegExp(target)); assert.equal(a.details.results[0].artifactPaths.outputPath,target);
  assert.equal(r.details.results[0].artifactPaths.outputPath,r.source);
  const read = await f.read.execute('read-report',{path:target,offset:1201,limit:3});
  assert.match(JSON.stringify(read),/FINAL_MARKER/);
  await rm(join(f.root,target));
  const restored = new SubagentReports(f.root,()=>f.parent,f.writer);
  await restored.transform(a.content,a.details);
  assert.match(await readFile(join(f.root,target),'utf8'),/FINAL_MARKER/);
  const resumed = await f.record(); await restored.transform(resumed.content,resumed.details);
  assert.match(await readFile(join(f.root,`public_data/subagents/${resumed.id}/report.md`),'utf8'),/FINAL_MARKER/);
  assert.equal(f.writes(),3);
});

test('short inline results do not create report copies', async t => {
  const f=await fixture(t);const r=await f.record();
  await f.reports.transform([{type:'text',text:'short handoff'}],r.details);
  assert.equal(f.writes(),0);
});

test('failures retain native references, avoid same-turn retries, and retry next turn', async t => {
  const f=await fixture(t);const r=await f.record();f.fail(true);
  const failed=await f.reports.transform(r.content,r.details);
  assert.match(failed.content[0].text,/完整报告同步失败/);assert.ok(!failed.content[0].text.includes(r.source));
  assert.deepEqual(failed.details,r.details);
  await f.reports.transform(r.content,r.details);assert.equal(f.writes(),1);
  f.fail(false);f.reports.beginTurn();
  const recovered=await f.reports.transform(failed.content,failed.details);assert.match(recovered.content[0].text,/public_data\/subagents/);
  assert.equal(f.writes(),2);
});

test('report records enforce conversation identity, source containment, and source/destination symlinks', async t => {
  const f=await fixture(t);const foreign=await f.record(randomUUID(),randomUUID());
  assert.match(JSON.stringify(await f.reports.transform(foreign.content,foreign.details)),/同步失败/);
  const link=await f.record();const secret=join(f.root,'secret');await writeFile(secret,'PRIVATE_SECRET');
  await rm(link.source);await symlink(secret,link.source);
  const blocked=await f.reports.transform(link.content,link.details);assert.match(JSON.stringify(blocked),/同步失败/);
  assert.ok(!JSON.stringify(blocked).includes('PRIVATE_SECRET'));assert.equal(f.writes(),0);
  const missing=await f.record();await rm(missing.source);
  assert.match(JSON.stringify(await f.reports.transform(missing.content,missing.details)),/已清理或不可用/);
  const ordinary=await f.record();await symlink(join(f.root,'pi'),join(f.root,'public_data','subagents'));
  assert.match(JSON.stringify(await f.reports.transform(ordinary.content,ordinary.details)),/同步失败/);
  const outside='/another/conversation/pi/subagent-artifacts/'+basename(ordinary.source);
  const content=[{type:'text',text:'Full output: '+outside}];
  assert.deepEqual((await f.reports.transform(content,{})).content,content);
});
