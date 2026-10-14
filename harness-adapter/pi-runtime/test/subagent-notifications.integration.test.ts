import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, realpath, readFile } from "node:fs/promises";
import { join } from "node:path";
import { createServer } from "node:http";
import { PiAgentRuntime } from "../src/agent-runtime.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";

test("two native child completions and a durable platform callback share one parent loop", {timeout:45000}, async (t) => {
  const root=await realpath(await mkdtemp('/tmp/pi-notifications-'));
  await mkdir(join(root,'pi','terminal-output'),{recursive:true});
  await mkdir(join(root,'public_data')); await mkdir(join(root,'skills'));
  const requests:any[]=[]; const events:any[]=[]; let children=0;
  let release!:()=>void; const gate=new Promise<void>(resolve=>{release=resolve;});
  const server=createServer(async(req,res)=>{
    let input='';for await(const chunk of req)input+=chunk;
    const body=JSON.parse(input);requests.push(body);
    const parent=body.tools?.some((tool:any)=>tool.function.name==='subagent');
    let delta:any;let reason='stop';
    if(!parent){children++;await gate;delta={content:'CHILD_DONE_'+children};}
    else if(!body.messages.some((m:any)=>m.role==='tool') && !JSON.stringify(body.messages).includes('background') && !JSON.stringify(body.messages).includes('PLATFORM_CALLBACK')){
      delta={tool_calls:['one','two'].map((name,index)=>({index,id:'delegate-'+name,type:'function',function:{name:'subagent',arguments:JSON.stringify({agent:'search',task:'Answer task '+name,async:true})}}))};reason='tool_calls';
    }else delta={content:'PARENT_HANDOFF'};
    res.writeHead(200,{'content-type':'text/event-stream'});
    for(const [part,finish] of [[{role:'assistant'},null],[delta,null],[{},reason]])res.write(`data: ${JSON.stringify({id:'test',object:'chat.completion.chunk',created:1,model:body.model,choices:[{index:0,delta:part,finish_reason:finish}],...(finish?{usage:{prompt_tokens:2,completion_tokens:1,total_tokens:3}}:{})})}\n\n`);
    res.end('data: [DONE]\n\n');
  });
  await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
  const peer={emit:(event:any)=>events.push(event),request:async()=>({allow:true})} as unknown as JsonlRpcPeer;
  const runtime=new PiAgentRuntime(peer);
  t.after(async()=>{release();await runtime.handle('close',{});server.close();});
  await runtime.handle('start',{session_id:'mixed-notifications',workspace_root:root,terminal_backend:'os-sandbox',session_path:join(root,'pi','session.jsonl'),skills_directory:join(root,'skills'),skill_roots:[],system_prompt:'PRIVATE_PARENT_CONTEXT',model:{provider:'openai',id:'custom-chat-model',api_key:'test-key',base_url:`http://127.0.0.1:${(server.address() as any).port}/v1`},tools:[]});
  await runtime.handle('prompt',{run_id:'main',content:[{type:'text',text:'Delegate two searches'}]});
  await eventually(()=>assert.equal(children,2));
  const callback={run_id:'platform-callback',content:'PLATFORM_CALLBACK',details:{task_id:'platform-task'}};
  assert.deepEqual(await runtime.handle('notify',callback),{accepted:true,queued:true});
  assert.deepEqual(await runtime.handle('notify',callback),{accepted:true,duplicate:true});
  release();
  await eventually(()=>assert.ok(events.some(event=>event.kind==='run.finished'&&event.runId.startsWith('platform-callback:'))));
  const active=new Set<string>();
  for(const event of events){if(event.kind==='agent.started')active.add(event.runId);if(event.kind==='run.finished')active.delete(event.runId);assert.ok(active.size<=1,JSON.stringify(events));}
  const starts=events.filter(event=>event.kind==='subagent.started');
  assert.equal(new Set(starts.map(event=>event.runId)).size,2);
  await eventually(()=>assert.equal(events.filter(event=>event.kind==='subagent.finished').length,2));
  const childRequests=requests.filter(body=>!body.tools?.some((tool:any)=>tool.function.name==='subagent'));
  assert.equal(childRequests.length,2);
  assert.equal(childRequests.some(body=>JSON.stringify(body).includes('PRIVATE_PARENT_CONTEXT')),false);
  assert.equal(events.some(event=>event.kind==='message.completed'&&event.payload.role==='assistant'&&JSON.stringify(event.payload.content).includes('CHILD_DONE')),false);
  const history=(await readFile(join(root,'pi','session.jsonl'),'utf8')).split('\n').filter(Boolean).map(line=>JSON.parse(line));
  assert.equal(history.filter(entry=>entry.type==='custom'&&entry.customType==='pyromind.notification').length,1);
  assert.ok(history.some(entry=>JSON.stringify(entry).includes('subagent-notify')));
  await eventually(()=>assert.ok(events.filter(event=>event.kind==='usage.updated').at(-1)?.payload.input_tokens>=4));
});
async function eventually(check:()=>void){const deadline=Date.now()+20000;for(;;){try{return check();}catch(error){if(Date.now()>deadline)throw error;await new Promise(resolve=>setTimeout(resolve,50));}}}
