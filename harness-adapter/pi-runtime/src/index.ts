import { PiAgentRuntime } from "./agent-runtime.js";
import { JsonlRpcPeer } from "./rpc-peer.js";

const peer = new JsonlRpcPeer(process.stdin, process.stdout);
const runtime = new PiAgentRuntime(peer);
peer.listen(runtime.handle.bind(runtime)).catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : "Pi runner failed"}\n`);
  process.exitCode = 1;
});
