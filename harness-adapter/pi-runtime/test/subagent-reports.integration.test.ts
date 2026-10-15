import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, mkdir, realpath, readFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import { createPiSession } from "../src/pi-session.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";

for (const backend of ['os-sandbox','sandbox'] as const) test(`long native reports are read once through ${backend} without a second delegation`, {timeout:45000}, async t => {
  const root=await realpath(await mkdtemp('/tmp/pi-report-integration-'));
  await mkdir(join(root,'pi','terminal-output'),{recursive:true});await mkdir(join(root,'public_data'));await mkdir(join(root,'skills'));
  const report='Evidence with Chinese 中文.\n'.repeat(240)+'FINAL_REPORT_MARKER';
  const files=new Map<string,Buffer>(); const uploads=new Map<string,Buffer[]>();
  let writes=0;let children=0;let delegates=0;let reportPath='';let readFinished=false;
  const fileApi=createServer(async(req,res)=>{
    const url=new URL(req.url!,'http://local'); const path=url.searchParams.get('path')!;
    if(url.pathname.endsWith('/init')){const id='upload-'+uploads.size;uploads.set(id,[]);res.end(JSON.stringify({upload_id:id}));return;}
    const id=url.pathname.split('/').at(-2)!;
    if(url.pathname.endsWith('/part')){const chunks=[];for await(const chunk of req)chunks.push(chunk);uploads.get(id)!.push(Buffer.concat(chunks));res.end('{}');return;}
    if(url.pathname.endsWith('/complete')){files.set(path,Buffer.concat(uploads.get(id)!));writes++;res.end('{}');return;}
    if(url.pathname.endsWith('/read')&&files.has(path)){res.end(files.get(path));return;}
    res.writeHead(404);res.end('{}');
  });
  await new Promise<void>(resolve=>fileApi.listen(0,'127.0.0.1',resolve));
  const fileUrl=`http://127.0.0.1:${(fileApi.address() as any).port}`;
  const model=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;
    const body=JSON.parse(raw);const parent=body.tools?.some((tool:any)=>tool.function.name==='subagent');
    let delta:any;let reason='stop';
    if(!parent){children++;delta={content:report};}
    else {
      const results=body.messages.filter((m:any)=>m.role==='tool');
      if(!results.length){delegates++;delta=call('delegate','subagent',{agent:'search',task:'Return the evidence report',async:false});reason='tool_calls';}
      else if(results.length===1){
        const text=JSON.stringify(results[0]);
        reportPath=text.match(/public_data\/subagents\/[a-f0-9-]+\/report\.md/)?.[0]??'';
        assert.ok(reportPath,text);assert.ok(!text.includes('/pi/subagent-artifacts/'),text);
        if(backend==='sandbox')assert.equal(writes,1,'copy completes before model sees reference');
        delta=call('read-report','read',{path:reportPath,offset:241,limit:3});reason='tool_calls';
      } else {readFinished=JSON.stringify(results.at(-1)).includes('FINAL_REPORT_MARKER');delta={content:'Parent finished using full report'};}
    }
    res.writeHead(200,{'content-type':'text/event-stream'});
    for(const [part,finish]of[[{role:'assistant'},null],[delta,null],[{},reason]])res.write(`data: ${JSON.stringify({id:'test',object:'chat.completion.chunk',created:1,model:body.model,choices:[{index:0,delta:part,finish_reason:finish}]})}\n\n`);
    res.end('data: [DONE]\n\n');
  });
  await new Promise<void>(resolve=>model.listen(0,'127.0.0.1',resolve));
  t.after(()=>{model.closeAllConnections();model.close();fileApi.closeAllConnections();fileApi.close();});
  const peer={emit:()=>{},request:async(method:string)=>{
    assert.equal(method,'sandbox.ensure');
    return {base_url:fileUrl,ws_base_url:fileUrl,sandbox_id:'test',api_key:'test',workspace_path:'/sandbox/conversation',storage_path:'/storage'};
  }} as unknown as JsonlRpcPeer;
  const config={session_id:'report-test',workspace_root:root,terminal_backend:backend,session_path:join(root,'pi','session.jsonl'),skills_directory:join(root,'skills'),skill_roots:[],system_prompt:'Use the search agent and read its full report.',model:{provider:'openai',id:'custom-chat-model',api_key:'test-key',base_url:`http://127.0.0.1:${(model.address() as any).port}/v1`},tools:[]};
  const {session,subagents}=await createPiSession(config,peer);
  t.after(async()=>{await session.abort();await session.waitForIdle();await subagents.stopAll();await session.extensionRunner.emit({type:'session_shutdown',reason:'quit'});session.dispose();await subagents.close();});
  await subagents.attach(session);await session.prompt('Research with search and read the complete result.');
  assert.equal(children,1);assert.equal(delegates,1);assert.equal(readFinished,true);
  if(backend==='sandbox')assert.equal(files.get('/sandbox/conversation/'+reportPath)?.toString(),report);
  else assert.equal(await readFile(join(root,reportPath),'utf8'),report);
  assert.equal(session.messages.some((m:any)=>m.role==='toolResult'&&m.toolName==='write'),false);
});
function call(id:string,name:string,args:unknown){return {tool_calls:[{index:0,id,type:'function',function:{name,arguments:JSON.stringify(args)}}]};}
