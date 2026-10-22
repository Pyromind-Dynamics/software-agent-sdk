import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, readFile, readdir, realpath } from "node:fs/promises";
import { join } from "node:path";
import { createServer } from "node:http";
import test from "node:test";
import { callSubagentHost } from "../src/subagent-ipc.js";
import { createPiSession } from "../src/pi-session.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";

test("real plugin bridges tools and model, resumes persisted history, and owns background runs", { timeout: 60000 }, async (t) => {
  const root = await realpath(await mkdtemp("/tmp/pi-plugin-integration-"));
  await mkdir(join(root, "public_data"));
  await mkdir(join(root, "pi", "terminal-output"), {recursive:true});
  await mkdir(join(root, "skills"));
  await writeFile(join(root, "public_data", "evidence.txt"), "CHILD_EVIDENCE_937\n");
  const requests: any[] = [];
  let stalled = 0;
  const server = createServer(async (req, res) => {
    let body = ""; for await (const chunk of req) body += chunk;
    const request = JSON.parse(body); requests.push(request);
    const isParent = request.tools?.some((tool: any) => tool.function.name === "subagent");
    if (!isParent && JSON.stringify(request.messages).includes("WAIT_FOR_STOP")) { stalled++; return; }
    const hasRead = isParent || request.messages.some((message: any) => message.role === "tool" && JSON.stringify(message).includes("CHILD_EVIDENCE_937"));
    const isWriter = !isParent && request.tools?.some((tool: any) => tool.function.name === "write");
    const isTerminalCheck = !isParent && JSON.stringify(request.messages).includes("CHECK_FAILED_TERMINAL");
    const wrote = request.messages.some((m: any) => m.role === "tool");
    const delta = isTerminalCheck && !wrote ? {tool_calls:[{index:0,id:"terminal-1",type:"function",function:{name:"terminal",arguments:JSON.stringify({command:"exit 7"})}}]} : isWriter && !wrote ? {tool_calls:[{index:0,id:"write-1",type:"function",function:{name:"write",arguments:JSON.stringify({path:"public_data/child-output.txt",content:"written by child"})}}]} : (hasRead || wrote) ? { content: `CHILD_EVIDENCE_937; users=${request.messages.filter((m: any) => m.role === "user").length}` } : { tool_calls: [{ index: 0, id: "read-1", type: "function", function: { name: "read", arguments: JSON.stringify({ path: "public_data/evidence.txt" }) } }] };
    res.writeHead(200, { "content-type": "text/event-stream" });
    for (const [d, reason] of [[{role:"assistant"}, null], [delta,null], [{},(hasRead || wrote) ? "stop" : "tool_calls"]]) res.write(`data: ${JSON.stringify({id:"test",object:"chat.completion.chunk",created:1,model:request.model,choices:[{index:0,delta:d,finish_reason:reason}]})}\n\n`);
    res.end("data: [DONE]\n\n");
  });
  await new Promise<void>((resolve) => server.listen(0,"127.0.0.1",resolve));
  t.after(() => server.close());
  const port = (server.address() as any).port;
  const events: any[] = [];
  const peer = { emit: (event: any) => events.push(event), request: async () => ({allow:true}) } as unknown as JsonlRpcPeer;
  const config = {session_id:"plugin-integration",workspace_root:root,terminal_backend:"os-sandbox",session_path:join(root,"pi","session.jsonl"),system_prompt:"Parent only",skills_directory:join(root,"skills"),skill_roots:[],model:{provider:"openai",id:"custom-chat-model",api_key:"PRIVATE_TEST_CREDENTIAL_824",base_url:`http://127.0.0.1:${port}/v1`},tools:[]};
  let {session, subagents} = await createPiSession(config,peer);
  t.after(async () => { await subagents.stopAll(); await session.abort(); await session.waitForIdle(); await session.extensionRunner.emit({type:"session_shutdown",reason:"quit"});session.dispose();await subagents.close(); });
  await subagents.attach(session);
  let tool = session.getToolDefinition("subagent")!;
  assert.ok(tool);
  const execute = (args: any) => tool.execute(crypto.randomUUID(), args, undefined, undefined, session.extensionRunner.createContext());
  const first: any = await execute({agent:"search",task:"Find the evidence",async:false});
  assert.ok(!first.isError,JSON.stringify(first));
  assert.match(JSON.stringify(first),/CHILD_EVIDENCE_937/);
  assert.deepEqual(requests[0].tools.map((item: any) => item.function.name).sort(),["find","grep","ls","read"]);
  await session.prompt("Which delegated roles are available?");
  const parentRequest = requests.find((request) => request.tools?.some((tool: any) => tool.function.name === "subagent"));
  assert.ok(parentRequest);
  const parentContext = JSON.stringify(parentRequest.messages);
  assert.match(parentContext, /<advertised_subagents>/);
  assert.match(parentContext, /<name>search<\/name>/);
  assert.match(parentContext, /<name>general_purpose<\/name>/);
  const description = parentRequest.tools.find((tool: any) => tool.function.name === "subagent").function.description;
  assert.match(description, /search: read-only research/);
  assert.match(description, /general_purpose: analysis/);
  assert.match(description, /action:"resume"/);
  assert.doesNotMatch(description, /guide tool-reference|guide workflows|fields not in this schema go in options/);
  const roles: any = await execute({action:"list"});
  assert.ok(!roles.isError, JSON.stringify(roles));
  assert.match(JSON.stringify(roles), /general_purpose/);
  assert.match(JSON.stringify(roles), /search/);
  const resume: any = await execute({action:"resume",id:first.details.runId,message:"Continue using the saved evidence"});
  assert.ok(resume.details.asyncId,JSON.stringify(resume));
  const resultPath = join(subagents.paths.stateRoot,"async-subagent-runs",resume.details.asyncId,"status.json");
  await eventually(async () => {const value=JSON.parse(await readFile(resultPath,"utf8"));assert.equal(value.state,"complete",JSON.stringify(value));assert.match(await readFile(value.sessionFile,"utf8"),/users=2/);});
  const background: any = await execute({agent:"search",task:"Find evidence in background",async:true});
  assert.ok(background.details.asyncId,JSON.stringify(background));
  await eventually(async () => {const value=JSON.parse(await readFile(join(subagents.paths.stateRoot,"async-subagent-runs",background.details.asyncId,"status.json"),"utf8"));assert.equal(value.state,"complete",JSON.stringify(value));assert.match(await readFile(value.sessionFile,"utf8"),/CHILD_EVIDENCE_937/);});
  for (const name of ["write", "terminal"]) await assert.rejects(callSubagentHost(subagents.paths.socket,{method:"tool",role:"search",name,args:{path:"public_data/forbidden"}}),/not allowed/);
  await assert.rejects(callSubagentHost(subagents.paths.socket,{method:"tool",role:"search",name:"read",args:{path:"/etc/passwd"}}),/PATH_SCOPE|outside|denied|allowed/i);
  const writer: any = await execute({agent:"general_purpose",task:"Write the assigned output",async:false});
  assert.ok(!writer.isError,JSON.stringify(writer));
  assert.equal(await readFile(join(root,"public_data/child-output.txt"),"utf8"),"written by child");
  const terminalCheck: any = await execute({agent:"general_purpose",task:"CHECK_FAILED_TERMINAL",async:false});
  assert.ok(!terminalCheck.isError, JSON.stringify(terminalCheck));
  assert.ok(events.some(event=>event.kind==="subagent.workflow_modified"));
  assert.equal(session.messages.some((m: any)=>m.role==="toolResult"&&m.toolName==="read"),false);
  await subagents.stopAll(); await session.abort(); await session.waitForIdle();
  await session.extensionRunner.emit({type:"session_shutdown",reason:"quit"}); session.dispose(); await subagents.close();
  ({session,subagents}=await createPiSession(config,peer));
  await subagents.attach(session);
  tool=session.getToolDefinition("subagent")!;
  const afterRestart: any = await execute({action:"resume",id:resume.details.asyncId,message:"Continue after host restart"});
  assert.ok(afterRestart.details.asyncId,JSON.stringify(afterRestart));
  await eventually(async()=>{const value=JSON.parse(await readFile(join(subagents.paths.stateRoot,"async-subagent-runs",afterRestart.details.asyncId,"status.json"),"utf8"));assert.equal(value.state,"complete",JSON.stringify(value));assert.match(await readFile(value.sessionFile,"utf8"),/users=3/);});
  const stopped: any = await execute({agent:"search",task:"WAIT_FOR_STOP",async:true});
  await eventually(async()=>assert.equal(stalled,1));
  await subagents.stopAll();
  const stoppedState=JSON.parse(await readFile(join(subagents.paths.stateRoot,"async-subagent-runs",stopped.details.asyncId,"status.json"),"utf8"));
  assert.ok(["stopped","failed"].includes(stoppedState.state),JSON.stringify(stoppedState));
  const terminalResults: any[] = [];
  const scan = async (dir: string): Promise<void> => {
    for (const entry of await readdir(dir,{withFileTypes:true})) {
      const path=join(dir,entry.name);
      if(entry.isDirectory()) await scan(path);
      else if(entry.isFile()) {
        const content = await readFile(path);
        assert.equal(content.includes(Buffer.from("PRIVATE_TEST_CREDENTIAL_824")),false,path);
        if (entry.name === "session.jsonl") for (const line of content.toString().trim().split("\n").filter(Boolean)) {
          const record = JSON.parse(line);
          if (record.message?.role === "toolResult" && record.message.toolName === "terminal") terminalResults.push(record.message);
        }
      }
    }
  };
  await scan(join(root,"pi"));
  assert.ok(terminalResults.length > 0);
  for (const result of terminalResults) {
    assert.equal(result.isError, true);
    assert.equal(result.details.exitCode, 7);
    assert.equal(result.details.resultCategory, "failure");
  }
});
async function eventually(check:()=>Promise<void>) {const deadline=Date.now()+15000;for(;;){try{return await check();}catch(error){if(Date.now()>deadline)throw error;await new Promise(resolve=>setTimeout(resolve,100));}}}
