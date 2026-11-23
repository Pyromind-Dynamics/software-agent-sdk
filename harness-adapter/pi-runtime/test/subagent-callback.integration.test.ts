import { writeFileSync, mkdirSync } from "node:fs";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, realpath, readFile } from "node:fs/promises";
import { join } from "node:path";
import { createServer } from "node:http";
import test from "node:test";
import { createPiSession } from "../src/pi-session.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";

test("real children submit with distinct stable origins and resume only on their own callback", {timeout: 60000}, async t => {
  const root = await realpath(await mkdtemp("/tmp/pi-child-callback-"));
  await mkdir(join(root, "public_data"));
  await mkdir(join(root, "pi", "terminal-output"), {recursive:true});
  await mkdir(join(root, "skills"));
  const requests: any[] = [], submissions: any[] = [];
  const server = createServer(async (req, res) => {
    let raw = ""; for await (const chunk of req) raw += chunk;
    const body = JSON.parse(raw); requests.push(body);
    const parent = body.tools?.some((tool: any) => tool.function.name === "subagent");
    const prompt = JSON.stringify(body.messages);
    const submitted = body.messages.some((message: any) => message.role === "tool");
    const pipeline = prompt.includes("DATASET_PENDING");
    const terminal = prompt.includes("CALLBACK_FOR_FIRST_ONLY") || prompt.includes("CALLBACK_FOR_DATASET");
    const report = pipeline ? "FINAL_DATASET_REPORT" : "FINAL_WORKFLOW_REPORT";
    const staged = pipeline ? "STAGED_DATASET_REPORT" : "STAGED_WORKFLOW_REPORT";
    const delta = parent || submitted ? {content: parent ? "Parent result" : terminal ? report : staged} : {tool_calls:[{index:0,id:"create",type:"function",function:{name:pipeline ? "df_submit_pipeline" : "submit_workflow",arguments:"{}"}}]};
    res.writeHead(200, {"content-type":"text/event-stream"});
    for (const [value, reason] of [[{role:"assistant"},null],[delta,null],[{},parent || submitted ? "stop":"tool_calls"]]) res.write(`data: ${JSON.stringify({id:"test",object:"chat.completion.chunk",created:1,model:body.model,choices:[{index:0,delta:value,finish_reason:reason}]})}\n\n`);
    res.end("data: [DONE]\n\n");
  });
  await new Promise<void>(resolve => server.listen(0,"127.0.0.1",resolve));
  t.after(() => server.close());
  const peer = {emit: () => {}, request: async (method: string, params: any) => {
    if (method === "role.tool.execute") {
      submissions.push(params);
      return params.tool_name === "df_submit_pipeline"
        ? {content:[{type:"text",text:"Pipeline submitted"}],details:{status:"Pending",task_id:"pipeline-1",run_id:"pipeline-run-1"},is_error:false}
        : {content:[{type:"text",text:"Workflow submitted"}],details:{status:"submitted",request_id:`request-${submissions.length}`,task_id:`task-${submissions.length}`},is_error:false};
    }
    return {allow:true};
  }} as unknown as JsonlRpcPeer;
  const role = {name:"train",tools:["submit_workflow","df_submit_pipeline"],skills:[]};
  // Existing conversations retain their materialized role contracts across upgrades.
  const retained = join(root, "pi", "agent", "agents");
  mkdirSync(retained, {recursive:true});
  for (const name of ["init", "dataset", "train", "eval"]) {
    writeFileSync(join(retained, name + ".md"), `---\nname: ${name}\ndescription: Retained role\n---\nLegacy ${name} instructions.\n`);
  }
  const config = {session_id:"callback-conversation", workspace_root:root, terminal_backend:"os-sandbox", session_path:join(root,"pi","session.jsonl"), system_prompt:"Delegate when requested", skills_directory:join(root,"skills"), skill_roots:[], tools:[], delegated_roles:[role], delegated_tool_specs:{train:[{name:"submit_workflow",description:"Submit the workflow",input_schema:{type:"object",properties:{}}},{name:"df_submit_pipeline",description:"Submit the pipeline",input_schema:{type:"object",properties:{}}}]}, model:{provider:"openai",id:"custom-chat-model",api_key:"test",base_url:`http://127.0.0.1:${(server.address() as any).port}/v1`}};
  const {session, subagents} = await createPiSession(config, peer);
  t.after(async () => {await subagents.stopAll(); await session.abort(); await session.waitForIdle(); await session.extensionRunner.emit({type:"session_shutdown",reason:"quit"}); session.dispose(); await subagents.close();});
  await subagents.attach(session);
  await session.prompt("Initialize parent context");
  const tool = session.getToolDefinition("subagent")!;
  const execute = (task: string) => tool.execute(crypto.randomUUID(),{agent:"train",task,async:true},undefined,undefined,session.extensionRunner.createContext());
  const first: any = await execute("lane-one"), second: any = await execute("lane-two");
  const status = (id: string) => readFile(join(subagents.paths.stateRoot,"async-subagent-runs",id,"status.json"),"utf8").then(JSON.parse);
  const waitForCompletionReplay = (id: string) => eventually(async () => {
    await readFile(join(subagents.paths.stateRoot, "async-subagent-results", "completion-replay", `${id}.json`));
  });
  await eventually(async () => {assert.equal(submissions.length,2); assert.equal((await status(first.details.asyncId)).state,"complete"); assert.equal((await status(second.details.asyncId)).state,"complete");});
  await Promise.all([first.details.asyncId, second.details.asyncId].map(waitForCompletionReplay));
  const parentRequests = () => requests.filter(body => body.tools?.some((tool: any) => tool.function.name === "subagent"));
  assert.equal(parentRequests().length, 1, "pending workflow must not wake the parent");
  assert.doesNotMatch(await readFile(config.session_path, "utf8"), /STAGED_WORKFLOW_REPORT/);
  const one = submissions[0].origin, two = submissions[1].origin;
  assert.equal(one.type,"subagent"); assert.notEqual(one.subagent_id,two.subagent_id);
  const result = await subagents.notifyRecipient({run_id:"callback:task-one:succeeded",content:"CALLBACK_FOR_FIRST_ONLY",trigger_turn:true,details:{correlation:{version:1,source:"agent",conversation_id:"callback-conversation",task_kind:"workflow_run",...one}}});
  assert.equal(result.accepted,true);
  await eventually(async () => assert.ok(requests.some(body => !body.tools?.some((tool:any)=>tool.function.name==="subagent") && JSON.stringify(body.messages).includes("CALLBACK_FOR_FIRST_ONLY"))));
  await eventually(async () => assert.ok(parentRequests().some(body => JSON.stringify(body.messages).includes("FINAL_WORKFLOW_REPORT"))));
  const recipients = session.sessionManager.getEntries().filter((entry:any)=>entry.type==="custom"&&entry.customType==="pyromind.subagent.recipient") as any[];
  const secondFile = recipients.find(entry=>entry.data.id===two.subagent_id).data.sessionFile;
  assert.equal((await readFile(secondFile,"utf8")).includes("CALLBACK_FOR_FIRST_ONLY"),false);
  const beforePipeline = parentRequests().length;
  const pipeline: any = await execute("DATASET_PENDING");
  await eventually(async () => {assert.equal(submissions.length,3); assert.equal((await status(pipeline.details.asyncId)).state,"complete");});
  await waitForCompletionReplay(pipeline.details.asyncId);
  assert.equal(parentRequests().length,beforePipeline,"pending pipeline must not wake the parent");
  assert.doesNotMatch(await readFile(config.session_path,"utf8"),/STAGED_DATASET_REPORT/);
  const pipelineOrigin = submissions[2].origin;
  const pipelineCallback = await subagents.notifyRecipient({run_id:"callback:pipeline:succeeded",content:"CALLBACK_FOR_DATASET",trigger_turn:true,details:{correlation:{version:1,source:"agent",conversation_id:"callback-conversation",task_kind:"data_preparation",...pipelineOrigin}}});
  assert.equal(pipelineCallback.accepted,true);
  await eventually(async () => assert.ok(parentRequests().some(body => JSON.stringify(body.messages).includes("FINAL_DATASET_REPORT"))));
});

async function eventually(check:()=>Promise<void>) {const end=Date.now()+20000;for(;;){try{return await check();}catch(error){if(Date.now()>end)throw error;await new Promise(resolve=>setTimeout(resolve,100));}}}
