import assert from "node:assert/strict";
import { mkdtemp, mkdir, realpath } from "node:fs/promises";
import { join } from "node:path";
import { createServer } from "node:http";
import test from "node:test";
import { createPiSession } from "../src/pi-session.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";

test("native business roles need no product task lease and callbacks resume their saved sessions", { timeout: 60000 }, async (t) => {
  const root = await realpath(await mkdtemp("/tmp/pi-native-roles-"));
  await mkdir(join(root, "public_data"), { recursive: true });
  await mkdir(join(root, "pi", "terminal-output"), { recursive: true });
  await mkdir(join(root, "skills"));
  const roles = [
    ["init", "df_submit_evaluation"], ["dataset", "df_submit_pipeline"],
    ["train", "submit_workflow"], ["eval", "df_submit_evaluation"],
  ];
  const calls: any[] = [];
  const events: any[] = [];
  const resumed: string[] = [];
  const prompts: string[] = [];
  const server = createServer(async (req, res) => {
    let body = ""; for await (const chunk of req) body += chunk;
    const request = JSON.parse(body);
    const tools = request.tools?.map((tool: any) => tool.function.name) ?? [];
    const business = roles.find(([, name]) => tools.includes(name))?.[1];
    const complete = request.messages.some((message: any) => message.role === "tool");
    const text = JSON.stringify(request.messages);
    if (business) prompts.push(text);
    if (text.includes("PLATFORM_FINISHED_NATIVE")) resumed.push(text);
    const delta = business && !complete
      ? { tool_calls: [{ index: 0, id: "business-call", type: "function",
          function: { name: business, arguments: "{}" } }] }
      : { content: "Evidence is available; Manager decides business completion." };
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.write(`data: ${JSON.stringify({id:"native",object:"chat.completion.chunk",created:1,model:request.model,choices:[{index:0,delta:{role:"assistant",...delta},finish_reason:business && !complete ? "tool_calls" : "stop"}]})}\n\n`);
    res.end("data: [DONE]\n\n");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());
  const peer = {
    emit(event: any) { events.push(event); },
    async request(method: string, params: any) {
      assert.equal(method, "role.tool.execute");
      assert.equal(params.scope, undefined);
      assert.equal(params.origin.type, "subagent");
      assert.ok(roles.some(([role, tool]) => role === params.role && tool === params.tool_name));
      calls.push(params);
      return { is_error: false, content: [{ type: "text", text: "PLATFORM_RECEIPT_ONLY" }], details: {} };
    },
  } as unknown as JsonlRpcPeer;
  const config = {
    session_id: "native-roles", workspace_root: root, terminal_backend: "os-sandbox",
    session_path: join(root, "pi", "session.jsonl"), system_prompt: "Manager decides completion.",
    skills_directory: join(root, "skills"), skill_roots: [],
    tool_allowlist: ["read", "write", "edit", "terminal", "subagent"],
    skill_allowlist: [], tools: [],
    delegated_roles: roles.map(([name, tool]) => ({ name, tools: [tool], skills: [] })),
    delegated_tool_specs: Object.fromEntries(roles.map(([name, tool]) => [name, [{
      name: tool, description: "Existing platform tool", input_schema: { type: "object", properties: {} },
    }]])),
    model: { provider: "openai", id: "custom-chat-model", api_key: "test",
      base_url: `http://127.0.0.1:${(server.address() as any).port}/v1` },
  };
  let { session, subagents } = await createPiSession(config, peer);
  const close = async () => {
    await subagents.stopAll();
    await session.abort();
    await session.waitForIdle();
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose();
    await subagents.close();
  };
  t.after(close);
  await subagents.attach(session);
  await session.prompt("Keep this experiment's results.");
  const tool = session.getToolDefinition("subagent")!;
  for (const [role] of roles) {
    const result: any = await tool.execute(crypto.randomUUID(),
      { agent: role, task: "Use the platform tool and report evidence.", async: false },
      undefined, undefined, session.extensionRunner.createContext());
    assert.ok(!result.isError, JSON.stringify(result));
    assert.match(JSON.stringify(result), /Manager decides business completion/);
  }
  assert.equal(calls.length, 4);
  for (const role of ["Init", "Dataset", "Train", "Eval"]) {
    assert.ok(prompts.some(prompt => prompt.includes(`You are the EchoMind ${role} agent.`)));
  }
  await close();
  ({ session, subagents } = await createPiSession(config, peer));
  await subagents.attach(session);
  const notification = {
    run_id: "callback:platform-1:succeeded", content: "PLATFORM_FINISHED_NATIVE",
    trigger_turn: true, details: { correlation: calls[0].origin },
  };
  const delivery = await subagents.notifyRecipient(notification);
  assert.equal(delivery.accepted, true, JSON.stringify(delivery));
  await eventually(async () => assert.ok(resumed.length > 0));
  assert.equal((await subagents.notifyRecipient(notification)).duplicate, true);
  assert.equal(calls.length, 4, "callback analysis must not repeat platform submission");
  await eventually(async () => {
    await subagents.refresh();
    const linked = events.filter(event =>
      event.payload?.details?.subagent?.session_key === calls[0].origin.subagent_id);
    assert.ok(new Set(linked.map(event => event.runId)).size >= 2,
      "the original run and its resumed run retain the same context identity");
  });
});

async function eventually(check:()=>Promise<void>) {const deadline=Date.now()+15000;for(;;){try{return await check();}catch(error){if(Date.now()>deadline)throw error;await new Promise(resolve=>setTimeout(resolve,100));}}}
